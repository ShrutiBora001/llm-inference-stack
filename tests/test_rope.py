"""RoPE under sequence sharding. The highest-value CPU-testable check in Phase 1.

Applying rotary embeddings with local buffer indices instead of global sequence
positions does not crash, does not produce NaN, and does not change any shape. It
silently attends with wrong relative distances. Under a contiguous layout the
error is a constant offset; under a striped layout a device owns two disjoint
position ranges and there is no single offset that is correct.

Several tests here deliberately exercise the *wrong* implementation and assert it
fails. A positional test that passes for both the correct and the broken version
is not testing anything.
"""

from __future__ import annotations

import math

import pytest
import torch

from dattn.layout import ContiguousLayout, StripedLayout, make_layout
from dattn.reference import attention_reference
from dattn.utils import max_abs_diff, set_seed
from lis.rope import (
    apply_rope,
    apply_rope_local_positions_WRONG,
    apply_rope_sharded,
    positions_for_layout,
    rotate_half,
)

B, H, S, DH = 2, 4, 64, 32
WORLDS = [2, 4, 8]


def make_x(seq=S):
    set_seed(0)
    return torch.randn(B, H, seq, DH)


# --------------------------------------------------------------- positions


def test_contiguous_positions_are_a_single_run():
    layout = ContiguousLayout(S, 4)
    for rank in range(4):
        pos = positions_for_layout(layout, rank)
        assert torch.equal(pos, torch.arange(rank * 16, (rank + 1) * 16))


def test_striped_positions_are_two_disjoint_runs():
    """Device r owns chunks r and 2P-1-r. This is the fact RoPE must respect."""
    layout = StripedLayout(S, 4)  # 8 chunks of 8 tokens
    expected = {
        0: [0, 1, 2, 3, 4, 5, 6, 7] + [56, 57, 58, 59, 60, 61, 62, 63],
        1: [8, 9, 10, 11, 12, 13, 14, 15] + [48, 49, 50, 51, 52, 53, 54, 55],
        2: [16, 17, 18, 19, 20, 21, 22, 23] + [40, 41, 42, 43, 44, 45, 46, 47],
        3: [24, 25, 26, 27, 28, 29, 30, 31] + [32, 33, 34, 35, 36, 37, 38, 39],
    }
    for rank, want in expected.items():
        assert positions_for_layout(layout, rank).tolist() == want


@pytest.mark.parametrize("name", ["contiguous", "striped"])
@pytest.mark.parametrize("world", WORLDS)
def test_positions_partition_the_sequence(name, world):
    """Every position owned exactly once across all devices."""
    layout = make_layout(name, S, world)
    seen = torch.cat([positions_for_layout(layout, r) for r in range(world)])
    assert sorted(seen.tolist()) == list(range(S))


# --------------------------------------------------------------- the invariant


@pytest.mark.parametrize("name", ["contiguous", "striped"])
@pytest.mark.parametrize("world", WORLDS)
def test_rope_then_shard_equals_shard_then_rope(name, world):
    """The invariant the whole design rests on.

    Applying RoPE to the full sequence and then sharding must equal sharding and
    then applying RoPE with global positions. If this holds, every downstream
    attention result follows from the already-proven ring correctness — RoPE
    becomes a pointwise transform that commutes with sharding.
    """
    layout = make_layout(name, S, world)
    x = make_x()

    roped_first = apply_rope(x, torch.arange(S))

    for rank in range(world):
        want = layout.shard(roped_first, rank, dim=2)
        got = apply_rope_sharded(layout.shard(x, rank, dim=2), layout, rank)
        assert max_abs_diff(got, want) < 1e-5, f"{name} P={world} rank {rank}"


@pytest.mark.parametrize("world", WORLDS)
def test_local_positions_break_striped_layout(world):
    """Proof the test above has teeth.

    The naive implementation — RoPE with local buffer indices — must *fail* the
    invariant under striping. If this test ever passes, the check above has
    stopped discriminating and is worthless.
    """
    layout = StripedLayout(S, world)
    x = make_x()
    roped_first = apply_rope(x, torch.arange(S))

    worst = 0.0
    for rank in range(world):
        want = layout.shard(roped_first, rank, dim=2)
        wrong = apply_rope_local_positions_WRONG(layout.shard(x, rank, dim=2), layout, rank)
        worst = max(worst, max_abs_diff(wrong, want))

    assert worst > 1e-2, (
        f"the wrong implementation passed at P={world} — the positional test is "
        "not discriminating and must be strengthened"
    )


def test_local_positions_happen_to_work_for_rank_zero_contiguous():
    """Why this bug survives casual testing.

    Under the contiguous layout, rank 0's global positions *are* its local
    indices, so the broken implementation looks correct there. Anyone who tests
    on one device, or checks only rank 0, sees nothing wrong.
    """
    layout = ContiguousLayout(S, 4)
    x_local = layout.shard(make_x(), 0, dim=2)
    assert max_abs_diff(
        apply_rope_local_positions_WRONG(x_local, layout, 0),
        apply_rope_sharded(x_local, layout, 0),
    ) < 1e-6

    # ...and is wrong on every other device.
    x1 = layout.shard(make_x(), 1, dim=2)
    assert max_abs_diff(
        apply_rope_local_positions_WRONG(x1, layout, 1),
        apply_rope_sharded(x1, layout, 1),
    ) > 1e-2


# --------------------------------------------------------------- attention


@pytest.mark.parametrize("name", ["contiguous", "striped"])
@pytest.mark.parametrize("world", WORLDS)
def test_attention_with_rope_matches_single_process(name, world):
    """End to end: sharded RoPE + gathered attention == single-process reference.

    Single process, so it isolates the positional question from the distributed
    machinery, which dattn already covers.
    """
    layout = make_layout(name, S, world)
    set_seed(1)
    q, k, v = (torch.randn(B, H, S, DH) for _ in range(3))
    scale = 1.0 / math.sqrt(DH)

    ref = attention_reference(
        apply_rope(q, torch.arange(S)), apply_rope(k, torch.arange(S)), v,
        causal=True, scale=scale,
    )

    # Rebuild q and k from per-device shards, each roped with its own positions.
    q_parts = [apply_rope_sharded(layout.shard(q, r, dim=2), layout, r) for r in range(world)]
    k_parts = [apply_rope_sharded(layout.shard(k, r, dim=2), layout, r) for r in range(world)]
    got = attention_reference(
        layout.unshard(q_parts, dim=2), layout.unshard(k_parts, dim=2), v,
        causal=True, scale=scale,
    )
    assert max_abs_diff(got, ref) < 1e-5


def test_rope_preserves_norms():
    """RoPE is a rotation: per-pair magnitude is invariant."""
    x = make_x()
    y = apply_rope(x, torch.arange(S))
    half = DH // 2
    for t in (x, y):
        t._norm = (t[..., :half] ** 2 + t[..., half:] ** 2).sum(-1)
    assert max_abs_diff(x._norm, y._norm) < 1e-4


def test_rope_is_relative():
    """The defining property: attention scores depend only on position *difference*.

    Two query/key pairs the same distance apart must produce the same score
    regardless of absolute position. This is why global positions matter — under
    striping, wrong absolute positions produce wrong *distances*.
    """
    set_seed(2)
    q = torch.randn(1, 1, 1, DH)
    k = torch.randn(1, 1, 1, DH)

    def score(qpos, kpos):
        qr = apply_rope(q, torch.tensor([qpos]))
        kr = apply_rope(k, torch.tensor([kpos]))
        return (qr * kr).sum().item()

    assert abs(score(10, 4) - score(30, 24)) < 1e-4   # both distance 6
    assert abs(score(10, 4) - score(10, 5)) > 1e-4    # distance 6 vs 5


def test_rotate_half_is_a_quarter_turn():
    """rotate_half applied four times is the identity."""
    x = make_x()
    assert max_abs_diff(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x) < 1e-6


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError, match="even"):
        apply_rope(torch.randn(1, 1, 4, 7), torch.arange(4))


def test_position_length_mismatch_rejected():
    with pytest.raises(ValueError, match="one-to-one"):
        apply_rope(make_x(), torch.arange(S - 1))
