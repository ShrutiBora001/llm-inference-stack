"""Paged KV cache <-> Chunk mapping. The core of the serving integration.

Entirely CPU-testable, which is the point: this is where the integration bugs
live, and none of them need a GPU to find. A wrong global_start produces no
crash and no shape error -- just a wrong causal mask and wrong RoPE angles, so
these tests check positions explicitly rather than only shapes.

Block tables here are deliberately shuffled and non-contiguous, because that is
what a real allocator produces and a test with a sorted block table would pass
even if the mapping ignored the table entirely.
"""

from __future__ import annotations

import pytest
import torch

from dattn.layout import ContiguousLayout, StripedLayout, make_layout
from dattn.utils import max_abs_diff, set_seed
from lis.rope import apply_rope
from lis.serving.paging import (
    PagedSpan,
    chunk_to_spans,
    gather_spans,
    global_positions,
    layout_to_spans,
    spans_cover_chunks,
)

S, PAGE, H, DH = 64, 8, 4, 16
WORLDS = [2, 4, 8]


def block_table(n_pages: int, shuffle: bool = True) -> list[int]:
    """A realistic block table: physical blocks are neither contiguous nor
    ordered. A sorted table would let a broken mapping pass."""
    table = list(range(100, 100 + n_pages))
    if shuffle:
        table = [table[i] for i in _permutation(n_pages)]
    return table


def _permutation(n: int) -> list[int]:
    idx = list(range(n))
    set_seed(7)
    perm = torch.randperm(n).tolist()
    return [idx[i] for i in perm]


# ------------------------------------------------------------------ coverage


@pytest.mark.parametrize("name", ["contiguous", "striped"])
@pytest.mark.parametrize("world", WORLDS)
def test_spans_reconstruct_the_chunks_exactly(name, world):
    """Every token the layout assigns, in local-buffer order, and no others."""
    layout = make_layout(name, S, world)
    table = block_table(S // PAGE)
    for rank in range(world):
        spans = layout_to_spans(layout, rank, table, PAGE)
        assert spans_cover_chunks(spans, layout.chunks(rank))


@pytest.mark.parametrize("world", WORLDS)
def test_every_token_owned_exactly_once_across_ranks(world):
    layout = StripedLayout(S, world)
    table = block_table(S // PAGE)
    seen = [
        p
        for rank in range(world)
        for s in layout_to_spans(layout, rank, table, PAGE)
        for p in range(s.global_start, s.global_stop)
    ]
    assert sorted(seen) == list(range(S))


def test_span_splits_at_page_boundaries():
    """A chunk longer than a page must split, once per page crossed."""
    from dattn.layout import Chunk

    spans = chunk_to_spans(Chunk(0, 20, 6), block_table(8, shuffle=False), PAGE)
    # positions 6..25 with page_size 8 -> [6,7] [8..15] [16..23] [24,25]
    assert [(s.global_start, s.length) for s in spans] == [(6, 2), (8, 8), (16, 8), (24, 2)]
    assert sum(s.length for s in spans) == 20


def test_span_uses_the_block_table_not_the_page_index():
    """Guards the bug where the mapping ignores the table and uses page index.

    With a shuffled table these differ; with a sorted one they would not, which
    is why the tables here are shuffled.
    """
    from dattn.layout import Chunk

    table = block_table(8)
    spans = chunk_to_spans(Chunk(0, 16, 0), table, PAGE)
    assert [s.physical_block for s in spans] == [table[0], table[1]]
    assert [s.physical_block for s in spans] != [0, 1]


def test_out_of_range_position_is_rejected():
    from dattn.layout import Chunk

    with pytest.raises(IndexError, match="block table"):
        chunk_to_spans(Chunk(0, 8, 200), block_table(4), PAGE)


# -------------------------------------------------------------------- gather


@pytest.mark.parametrize("name", ["contiguous", "striped"])
@pytest.mark.parametrize("world", WORLDS)
def test_gather_matches_direct_sharding(name, world):
    """Reading through the paged cache must equal sharding the full tensor.

    This is the end-to-end statement: paging is a storage detail and must not
    change a single value.
    """
    layout = make_layout(name, S, world)
    table = block_table(S // PAGE)

    set_seed(3)
    full = torch.randn(1, H, S, DH)  # [B, H, S, Dh]

    # Scatter into a paged cache the way a framework would.
    cache = torch.zeros(max(table) + 1, PAGE, H, DH)
    for page, phys in enumerate(table):
        tokens = full[0, :, page * PAGE:(page + 1) * PAGE, :]  # [H, PAGE, Dh]
        cache[phys] = tokens.permute(1, 0, 2)

    for rank in range(world):
        spans = layout_to_spans(layout, rank, table, PAGE)
        got = gather_spans(cache, spans)
        want = layout.shard(full, rank, dim=2)
        assert max_abs_diff(got, want) < 1e-6, f"{name} P={world} rank {rank}"


def test_gather_rejects_empty():
    with pytest.raises(ValueError, match="no spans"):
        gather_spans(torch.zeros(1, PAGE, H, DH), [])


# ----------------------------------------------------------------- positions


@pytest.mark.parametrize("world", WORLDS)
def test_positions_from_spans_are_global(world):
    """The trap. Positions must be global sequence positions, never local
    indices, or RoPE rotates by the wrong angle and the model emits fluent
    nonsense with nothing to catch it."""
    layout = StripedLayout(S, world)
    table = block_table(S // PAGE)
    for rank in range(world):
        spans = layout_to_spans(layout, rank, table, PAGE)
        pos = global_positions(spans)
        expected = torch.cat([
            torch.arange(c.global_start, c.global_stop)
            for c in sorted(layout.chunks(rank), key=lambda c: c.local_start)
        ])
        assert torch.equal(pos, expected)
        assert not torch.equal(pos, torch.arange(len(pos))), (
            "positions are local indices — this is the silent RoPE bug"
        )


@pytest.mark.parametrize("world", WORLDS)
def test_rope_through_paging_matches_rope_on_full_sequence(world):
    """Full path: page the cache, gather a rank's spans, apply RoPE with the
    positions those spans carry. Must equal RoPE on the whole sequence, then
    sharded."""
    layout = StripedLayout(S, world)
    table = block_table(S // PAGE)

    set_seed(11)
    full = torch.randn(1, H, S, DH)
    roped_full = apply_rope(full, torch.arange(S))

    cache = torch.zeros(max(table) + 1, PAGE, H, DH)
    for page, phys in enumerate(table):
        cache[phys] = full[0, :, page * PAGE:(page + 1) * PAGE, :].permute(1, 0, 2)

    for rank in range(world):
        spans = layout_to_spans(layout, rank, table, PAGE)
        got = apply_rope(gather_spans(cache, spans), global_positions(spans))
        want = layout.shard(roped_full, rank, dim=2)
        assert max_abs_diff(got, want) < 1e-5, f"P={world} rank {rank}"


def test_span_arithmetic():
    s = PagedSpan(physical_block=42, block_offset=3, length=5, global_start=100)
    assert s.block_stop == 8
    assert s.global_stop == 105
