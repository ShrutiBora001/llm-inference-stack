"""Zigzag ring attention for prefill context parallelism.

The complete algorithm, framework-agnostic. KV shards rotate around the CP
group; each rank folds every (query chunk, key chunk) pair that is not entirely
in the future, merging partials through log-sum-exp.

Two properties this inherits from the upstream work and must not lose:

  balance   the zigzag layout equalizes work per rank AND per ring step, so no
            rank stalls the group. Contiguous assignment idles roughly half the
            devices under causal masking.
  overlap   the exchange for step s+1 is posted before the compute for step s,
            so the transfer hides under the GEMMs.

Uses `lis.kernels.flash_attention`, so the same code runs on the unfused CPU
oracle and on a fused GPU kernel. That is what lets this be correctness-tested
on a laptop and performance-tested on rented hardware without diverging.
"""

from __future__ import annotations

import math

import torch

from dattn.comms import DistEnv, ring_exchange

from ..kernels import flash_attention
from ..lse import PartialAttention, merge
from .adapter import CPContext, KVProvider
from .paged import PagedKVProvider


def _fold_pair(
    state: PartialAttention | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_off: int,
    k_off: int,
    causal: bool,
    scale: float,
    backend: str | None,
) -> PartialAttention | None:
    """Attend one chunk pair and merge it in. None means nothing computed yet."""
    partial = flash_attention(
        q, k, v, causal=causal, scale=scale,
        q_offset=q_off, k_offset=k_off, backend=backend,
    )
    return partial if state is None else merge(state, partial)


def ring_attention_cp(
    q_local: torch.Tensor,
    kv: KVProvider,
    ctx: CPContext,
    env: DistEnv,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Prefill attention over a sequence-sharded input.

    q_local: [B, H, S_local, Dh] in local-buffer order (zigzag: two chunks)
    returns: [B, H, S_local, Dh], same order -- shard in, shard out

    The KV that arrives at each step belongs to a *different* rank, so its
    chunks carry different global positions. Those positions, not local indices,
    decide the causal mask.
    """
    k_local, v_local = kv.local_kv()

    # Mismatched batch does not raise: matmul broadcasts, so every batch element
    # silently attends batch 0's keys. Found this way in testing, and it would
    # be invisible in production. Check rather than trust.
    if k_local.shape[0] != q_local.shape[0] or v_local.shape[0] != q_local.shape[0]:
        raise ValueError(
            f"batch mismatch: q has {q_local.shape[0]}, k has {k_local.shape[0]}, "
            f"v has {v_local.shape[0]}. Broadcasting would silently attend the "
            "wrong keys rather than error."
        )
    if k_local.shape[2] != ctx.local_len:
        raise ValueError(
            f"KV provider returned {k_local.shape[2]} tokens, layout expects "
            f"{ctx.local_len}"
        )

    scale = scale if scale is not None else 1.0 / math.sqrt(q_local.shape[-1])

    q_chunks = ctx.local_chunks()
    states: list[PartialAttention | None] = [None] * len(q_chunks)

    # One fused buffer for K and V: a single 2x transfer beats two smaller ones,
    # since per-message latency dominates at these sizes.
    cur = torch.stack([k_local, v_local], dim=0).contiguous()
    nxt = torch.empty_like(cur)

    for step in range(ctx.world_size):
        reqs = []
        if step + 1 < ctx.world_size:
            # Posted BEFORE the compute below, which is where the overlap comes
            # from. Note it is outside the causal skip: a rank with nothing to
            # compute must still forward the block or the ring deadlocks.
            reqs = ring_exchange(cur, nxt, env)

        src = (ctx.rank - step) % ctx.world_size

        for qi, qc in enumerate(q_chunks):
            q_slice = q_local[:, :, qc.local_start:qc.local_stop]
            for kc in ctx.chunks_of(src):
                if ctx.causal and kc.global_start > qc.global_stop - 1:
                    continue  # entirely in the future
                states[qi] = _fold_pair(
                    states[qi], q_slice,
                    cur[0][:, :, kc.local_start:kc.local_stop],
                    cur[1][:, :, kc.local_start:kc.local_stop],
                    q_off=qc.global_start, k_off=kc.global_start,
                    causal=ctx.causal, scale=scale, backend=ctx.backend,
                )

        for r in reqs:
            r.wait()
        if step + 1 < ctx.world_size:
            cur, nxt = nxt, cur

    missing = [i for i, s in enumerate(states) if s is None]
    if missing:
        raise RuntimeError(
            f"query chunks {missing} attended nothing. Under causal masking every "
            "query sees at least itself, so this means the layout or the offsets "
            "are wrong."
        )

    return torch.cat([s.out for s in states], dim=2).to(q_local.dtype)


def ring_attention_cp_paged(
    q_local: torch.Tensor,
    provider: PagedKVProvider,
    ctx: CPContext,
    env: DistEnv,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Same ring, but KV is read from a framework's paged cache.

    The only difference is where the local K/V come from. Gathering them into a
    contiguous buffer once per forward is what a first integration should do:
    correct, obvious, and the copy is small next to the attention itself. A
    later optimization passes the block table into the kernel instead.
    """
    return ring_attention_cp(q_local, provider, ctx, env, scale=scale)
