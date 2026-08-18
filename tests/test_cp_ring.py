"""Prefill CP ring attention, end to end over real processes.

Spawns gloo ranks on CPU and diffs against a single-process reference, which is
the same discipline that let the upstream project settle correctness for free
and then transfer to NCCL with only device-placement fixes.

Covers both KV providers, so the paged path -- the one a framework binding will
use -- is verified to produce identical numbers to the contiguous one. If those
ever diverge, the block-table mapping is wrong.
"""

from __future__ import annotations

import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dattn.comms import init_distributed, shutdown
from dattn.reference import attention_reference
from dattn.utils import max_abs_diff, set_seed
from lis.cp import CPContext, InMemoryKVProvider, ring_attention_cp
from lis.cp.paged import PagedKVProvider

B, H, S, DH, PAGE = 2, 4, 64, 16, 8
SCALE = 1.0 / math.sqrt(DH)
SEED = 99


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def worlds(candidates=(2, 4, 8)):
    limit = torch.cuda.device_count() if torch.cuda.is_available() else max(candidates)
    return [w for w in candidates if w <= limit]


WORLDS = worlds()


def make_qkv():
    set_seed(SEED)
    return (torch.randn(B, H, S, DH), torch.randn(B, H, S, DH), torch.randn(B, H, S, DH))


def _init(rank, world, port):
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    return init_distributed()


def _gather_and_check(out_local, ctx, world, want):
    bufs = [torch.empty_like(out_local) for _ in range(world)]
    dist.all_gather(bufs, out_local.contiguous())
    if ctx.rank == 0:
        # Un-stripe: under zigzag a rank's output is two disjoint regions, so a
        # plain concat would compare scrambled tokens.
        full = ctx.layout.unshard(bufs, dim=2)
        diff = max_abs_diff(full, want)
        assert diff < 1e-4, f"P={world} layout={ctx.layout.name}: diff {diff:.2e}"


def _worker_contiguous_kv(rank, world, port, layout_name, causal):
    env = _init(rank, world, port)
    try:
        ctx = CPContext.create(rank, world, S, layout=layout_name,
                               causal=causal, backend="unfused")
        q, k, v = make_qkv()
        want = attention_reference(q, k, v, causal=causal, scale=SCALE)

        out = ring_attention_cp(
            ctx.layout.shard(q, rank, dim=2),
            InMemoryKVProvider(ctx.layout.shard(k, rank, dim=2),
                               ctx.layout.shard(v, rank, dim=2)),
            ctx, env, scale=SCALE,
        )
        _gather_and_check(out, ctx, world, want)
    finally:
        shutdown()


def _worker_paged_kv(rank, world, port, layout_name):
    """The path a framework binding actually takes."""
    env = _init(rank, world, port)
    try:
        ctx = CPContext.create(rank, world, S, layout=layout_name, backend="unfused")
        q, k, v = make_qkv()
        want = attention_reference(q, k, v, causal=True, scale=SCALE)

        # Shuffled, non-contiguous block table, as a real allocator produces.
        n_pages = S // PAGE
        set_seed(3)
        table = [100 + i for i in torch.randperm(n_pages).tolist()]

        def to_pages(t):
            cache = torch.zeros(max(table) + 1, PAGE, H, DH)
            for page, phys in enumerate(table):
                cache[phys] = t[0, :, page * PAGE:(page + 1) * PAGE, :].permute(1, 0, 2)
            return cache

        # gather_spans returns batch 1, so the query must match or matmul
        # broadcasting silently attends batch 0's KV for every batch element.
        provider = PagedKVProvider(to_pages(k), to_pages(v), table, PAGE, ctx)
        out = ring_attention_cp(ctx.layout.shard(q, rank, dim=2)[:1], provider,
                                ctx, env, scale=SCALE)
        _gather_and_check(out, ctx, world, want[:1])
    finally:
        shutdown()


def _worker_providers_agree(rank, world, port):
    env = _init(rank, world, port)
    try:
        ctx = CPContext.create(rank, world, S, backend="unfused")
        q, k, v = make_qkv()

        n_pages = S // PAGE
        set_seed(3)
        table = [100 + i for i in torch.randperm(n_pages).tolist()]

        def to_pages(t):
            cache = torch.zeros(max(table) + 1, PAGE, H, DH)
            for page, phys in enumerate(table):
                cache[phys] = t[0, :, page * PAGE:(page + 1) * PAGE, :].permute(1, 0, 2)
            return cache

        q1 = ctx.layout.shard(q, rank, dim=2)[:1]
        direct = ring_attention_cp(
            q1,
            InMemoryKVProvider(ctx.layout.shard(k, rank, dim=2)[:1],
                               ctx.layout.shard(v, rank, dim=2)[:1]),
            ctx, env, scale=SCALE,
        )
        paged = ring_attention_cp(
            q1, PagedKVProvider(to_pages(k), to_pages(v), table, PAGE, ctx),
            ctx, env, scale=SCALE,
        )
        assert max_abs_diff(direct, paged) < 1e-6, (
            f"rank {rank}: paged and contiguous KV disagree — the block-table "
            "mapping is wrong"
        )
    finally:
        shutdown()


def run(worker, world, *args):
    mp.spawn(worker, args=(world, free_port(), *args), nprocs=world, join=True)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("layout", ["contiguous", "striped"])
@pytest.mark.parametrize("causal", [False, True])
def test_cp_ring_matches_single_process(world, layout, causal):
    run(_worker_contiguous_kv, world, layout, causal)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("layout", ["contiguous", "striped"])
def test_paged_kv_matches_single_process(world, layout):
    run(_worker_paged_kv, world, layout)


@pytest.mark.parametrize("world", WORLDS)
def test_paged_and_contiguous_providers_agree(world):
    run(_worker_providers_agree, world)


def test_batch_mismatch_is_rejected_not_broadcast():
    """The trap that produced silently wrong numbers during development.

    torch.matmul broadcasts [2,H,S,D] against [1,H,D,S] without complaint, so
    every batch element attends batch 0's keys. No exception, no shape error,
    wrong answer.
    """
    import torch.distributed as _d  # noqa: F401

    from dattn.comms import DistEnv

    ctx = CPContext.create(rank=0, world_size=1, seq_len=S, backend="unfused")
    env = DistEnv(rank=0, world_size=1, local_rank=0,
                  device=torch.device("cpu"), backend="gloo")
    q = torch.randn(2, H, S, DH)
    kv = InMemoryKVProvider(torch.randn(1, H, S, DH), torch.randn(1, H, S, DH))
    with pytest.raises(ValueError, match="batch mismatch"):
        ring_attention_cp(q, kv, ctx, env, scale=SCALE)


def test_wrong_local_length_is_rejected():
    from dattn.comms import DistEnv

    ctx = CPContext.create(rank=0, world_size=1, seq_len=S, backend="unfused")
    env = DistEnv(rank=0, world_size=1, local_rank=0,
                  device=torch.device("cpu"), backend="gloo")
    q = torch.randn(1, H, S, DH)
    kv = InMemoryKVProvider(torch.randn(1, H, S // 2, DH), torch.randn(1, H, S // 2, DH))
    with pytest.raises(ValueError, match="layout expects"):
        ring_attention_cp(q, kv, ctx, env, scale=SCALE)
