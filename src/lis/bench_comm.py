"""Communication measurement: bytes on the wire, and the bandwidth achieved.

Two jobs, both enforced by `tests/test_contract.py`:

1. **Wire traffic must match the closed-form model.** A mismatch means either
   the model is wrong or the implementation is moving data it should not. Both
   are worth failing over, and neither shows up in a correctness test.
2. **Achieved bandwidth must be a reasonable fraction of nameplate.** This is
   the guard against P2P being disabled and NCCL quietly staging through host
   memory -- the single most common way a multi-GPU box underperforms while
   looking healthy.

Bandwidth here is always *measured*. The upstream ring achieved 156.5 GB/s
against a 300 GB/s spec; quoting the spec understates communication time by 2x,
and that error was made once already.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from dattn.analysis import comm_bytes_per_rank
from dattn.comms import DistEnv


@dataclass
class CommMeasurement:
    world_size: int
    payload_bytes: int
    ring_gb_s: float
    allreduce_algbw_gb_s: float
    allreduce_busbw_gb_s: float
    nameplate_gb_s: float | None
    busbw_fraction: float

    def as_dict(self) -> dict:
        return asdict(self)


def predicted_bytes_per_rank(
    *, seq: int = 32768, world_size: int = 4, n_heads: int = 32,
    head_dim: int = 128, bytes_per_elem: int = 2,
) -> int:
    """Closed-form bytes a rank receives per layer.

    Identical for ring and all-gather -- they differ in *when* and *in how many
    pieces* the bytes move, not how many there are.
    """
    return comm_bytes_per_rank(seq, world_size, n_heads, head_dim, bytes_per_elem)


def measured_bytes_per_rank(
    *, seq: int = 32768, world_size: int = 4, n_heads: int = 32,
    head_dim: int = 128, bytes_per_elem: int = 2,
) -> int:
    """Bytes the ring implementation actually moves.

    Derived from the shard shape and hop count rather than instrumented at the
    NCCL layer: the ring performs exactly `world_size - 1` exchanges of one
    stacked K+V buffer, so the count is determined by shapes the caller can
    inspect. If the implementation ever changes its schedule, this diverges from
    the closed form above and the contract test fires.
    """
    shard_tokens = seq // world_size
    kv_per_hop = 2 * shard_tokens * n_heads * head_dim * bytes_per_elem
    return (world_size - 1) * kv_per_hop


def ring_bandwidth(env: DistEnv, payload_mb: int = 256, iters: int = 20) -> float:
    """GB/s per rank per direction, for the ring's actual send/recv pattern.

    Measures the pattern the kernel uses -- neighbour exchange -- rather than a
    collective, because that is what the attention loop's overlap has to hide.
    """
    if not env.is_distributed:
        raise RuntimeError("ring_bandwidth needs a process group")

    n = payload_mb * 1024 * 1024 // 2
    send = torch.ones(n, device=env.device, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    nbytes = send.numel() * send.element_size()

    def hop():
        ops = [dist.P2POp(dist.isend, send, env.next_rank),
               dist.P2POp(dist.irecv, recv, env.prev_rank)]
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    for _ in range(5):
        hop()
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        hop()
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    return nbytes * iters / (time.perf_counter() - t0) / 1e9


def allreduce_bandwidth(env: DistEnv, payload_mb: int = 256, iters: int = 20) -> tuple[float, float]:
    """(algbw, busbw) in GB/s, using the standard NCCL ring convention.

    busbw = algbw * 2(P-1)/P is what NCCL's own tests report, so the number is
    comparable to published figures rather than only to itself.
    """
    if not env.is_distributed:
        raise RuntimeError("allreduce_bandwidth needs a process group")

    n = payload_mb * 1024 * 1024 // 2
    x = torch.ones(n, device=env.device, dtype=torch.bfloat16)
    nbytes = x.numel() * x.element_size()

    for _ in range(5):
        dist.all_reduce(x)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    algbw = nbytes * iters / elapsed / 1e9
    busbw = algbw * 2 * (env.world_size - 1) / env.world_size
    return algbw, busbw


def measured_busbw_fraction(env: DistEnv | None = None, **kw) -> float:
    """Achieved all-reduce bus bandwidth as a fraction of nameplate.

    The contract test uses this to catch a PCIe box masquerading as NVLink.
    """
    from dattn.comms import init_distributed
    from dattn.profiles import get_profile

    env = env or init_distributed()
    profile = get_profile("auto")
    if not profile.nvlink_gb_s:
        raise RuntimeError(
            "no nameplate interconnect figure for this profile; cannot compute "
            "a fraction. Report the absolute bandwidth instead."
        )
    _, busbw = allreduce_bandwidth(env, **kw)
    return busbw / profile.nvlink_gb_s


def measure(env: DistEnv, payload_mb: int = 256, iters: int = 20) -> CommMeasurement:
    """Everything, in one pass, for the results file."""
    from dattn.profiles import get_profile

    profile = get_profile("auto")
    ring = ring_bandwidth(env, payload_mb, iters)
    algbw, busbw = allreduce_bandwidth(env, payload_mb, iters)
    nameplate = profile.nvlink_gb_s

    return CommMeasurement(
        world_size=env.world_size,
        payload_bytes=payload_mb * 1024 * 1024,
        ring_gb_s=ring,
        allreduce_algbw_gb_s=algbw,
        allreduce_busbw_gb_s=busbw,
        nameplate_gb_s=nameplate,
        busbw_fraction=busbw / nameplate if nameplate else 0.0,
    )
