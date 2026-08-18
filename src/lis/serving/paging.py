"""Mapping between a framework's paged KV cache and this project's Chunks.

This is the actual difficulty of the serving integration, and it is entirely
CPU-testable, so it belongs here rather than in a rented GPU session.

The mismatch:

  This project      a device owns whole Chunks of the sequence, contiguous in
                    its local buffer, each carrying a `global_start`
  vLLM / SGLang     the KV cache is paged. A request's tokens live in fixed-size
                    physical blocks that are *not* contiguous and not in order;
                    a block table maps logical position -> physical block

So a single Chunk generally spans several physical blocks, and consecutive
physical blocks are generally unrelated. Bridging the two is index arithmetic,
but index arithmetic is exactly the bug class this project keeps finding: a
wrong `global_start` produces no crash, no NaN, and no shape error — just a
wrong causal mask and wrong RoPE angles.

Hence `global_start` is carried explicitly through every span rather than
recomputed, and the tests assert it survives round-tripping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dattn.layout import Chunk, ShardLayout


@dataclass(frozen=True)
class PagedSpan:
    """A contiguous run of tokens inside one physical KV block.

    physical_block:  index into the framework's KV cache
    block_offset:    first token's offset within that block
    length:          number of tokens
    global_start:    that first token's position in the FULL sequence

    `global_start` is the reason this type exists. Physical layout tells you
    where the bytes are; only the global position tells you what the token
    means for causal masking and RoPE.
    """

    physical_block: int
    block_offset: int
    length: int
    global_start: int

    @property
    def block_stop(self) -> int:
        return self.block_offset + self.length

    @property
    def global_stop(self) -> int:
        return self.global_start + self.length


def chunk_to_spans(chunk: Chunk, block_table: list[int], page_size: int) -> list[PagedSpan]:
    """Split one Chunk across the physical blocks that hold it.

    block_table[i] is the physical block holding logical positions
    [i*page_size, (i+1)*page_size).

    A chunk aligned to page boundaries yields ceil(length/page_size) spans; an
    unaligned one yields a partial span at each end.
    """
    spans: list[PagedSpan] = []
    pos = chunk.global_start
    remaining = chunk.length

    while remaining > 0:
        page = pos // page_size
        offset = pos % page_size
        take = min(page_size - offset, remaining)

        if page >= len(block_table):
            raise IndexError(
                f"logical position {pos} needs page {page} but the block table "
                f"has only {len(block_table)} entries"
            )

        spans.append(PagedSpan(
            physical_block=block_table[page],
            block_offset=offset,
            length=take,
            global_start=pos,
        ))
        pos += take
        remaining -= take

    return spans


def layout_to_spans(
    layout: ShardLayout, rank: int, block_table: list[int], page_size: int
) -> list[PagedSpan]:
    """Every paged span a device must read, in local-buffer order.

    Local-buffer order matters: it is the order the attention kernel expects its
    query rows in, and under a striped layout it is *not* global sequence order.
    """
    spans: list[PagedSpan] = []
    for chunk in sorted(layout.chunks(rank), key=lambda c: c.local_start):
        spans.extend(chunk_to_spans(chunk, block_table, page_size))
    return spans


def gather_spans(cache: torch.Tensor, spans: list[PagedSpan]) -> torch.Tensor:
    """Materialize a contiguous tensor from paged storage.

    cache: [num_blocks, page_size, n_heads, head_dim] — the framework's layout
    returns: [1, n_heads, total_tokens, head_dim], matching what the attention
    kernels in `lis.kernels` expect.

    A real backend would pass the block table to a kernel that reads pages
    directly rather than copying. This exists so the integration can be
    validated on CPU: if the spans are right here, they are right there.
    """
    if not spans:
        raise ValueError("no spans to gather")

    parts = [
        cache[s.physical_block, s.block_offset:s.block_stop]  # [len, H, Dh]
        for s in spans
    ]
    out = torch.cat(parts, dim=0)              # [total, H, Dh]
    return out.permute(1, 0, 2).unsqueeze(0)   # [1, H, total, Dh]


def global_positions(spans: list[PagedSpan], device=None) -> torch.Tensor:
    """Global sequence position of every token the spans cover, in order.

    Hand this to `lis.rope.apply_rope`. Deriving positions from a local index
    instead is the silent failure mode — see `lis/rope.py`.
    """
    return torch.cat([
        torch.arange(s.global_start, s.global_stop, device=device, dtype=torch.long)
        for s in spans
    ])


def spans_cover_chunks(spans: list[PagedSpan], chunks: list[Chunk]) -> bool:
    """Whether the spans reconstruct exactly the chunks' global positions.

    Used by tests and as a cheap runtime assertion when wiring up a new
    framework, where an off-by-one in the block table is otherwise invisible.
    """
    from_spans = [p for s in spans for p in range(s.global_start, s.global_stop)]
    from_chunks = [
        p
        for c in sorted(chunks, key=lambda c: c.local_start)
        for p in range(c.global_start, c.global_stop)
    ]
    return from_spans == from_chunks
