"""Rotary position embeddings for sequence-sharded attention.

RoPE rotates each (query, key) pair by an angle proportional to its **absolute
position in the full sequence**. That makes it the single most dangerous piece of
a striped layout.

Under the contiguous layout a device owns positions `[rL, (r+1)L)`, so local
index `i` maps to global `rL + i` — a constant offset, easy to get right by
accident. Under the striped layout a device owns chunks `r` and `2P-1-r`, which
are *disjoint and non-adjacent*. Local index `i` maps to two different global
ranges depending on which half of the buffer it falls in.

If you apply RoPE with local indices, nothing crashes. Shapes match, no NaN
appears, tests that only check for finiteness pass. The model simply attends with
the wrong relative distances and emits fluent, confident nonsense. That failure
mode is far worse than a crash, and it is invisible without a positional test.

`dattn.layout.Chunk.global_start` already carries the information needed. This
module makes using it the only convenient option.
"""

from __future__ import annotations

import torch

from dattn.layout import ShardLayout


def inv_freq(head_dim: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """Inverse frequencies theta_i = base^(-2i/d) for i in [0, d/2)."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    i = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (base ** (i / head_dim))


def positions_for_layout(layout: ShardLayout, rank: int, device=None) -> torch.Tensor:
    """Global sequence positions for a device's local buffer, in buffer order.

    Contiguous: a single contiguous run.
    Striped:    two disjoint runs, one from the front of the sequence and one
                from the back.

    Returns [S_local] int64. This is the tensor that must be handed to
    :func:`apply_rope` — never `arange(S_local)`.
    """
    chunks = sorted(layout.chunks(rank), key=lambda c: c.local_start)
    return torch.cat([
        torch.arange(c.global_start, c.global_stop, device=device, dtype=torch.long)
        for c in chunks
    ])


def rope_tables(
    positions: torch.Tensor, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin lookup for the given absolute positions.

    positions: [S] absolute positions — arbitrary, not necessarily contiguous.
    Returns two [S, head_dim] tensors.
    """
    freqs = positions.float().unsqueeze(-1) * inv_freq(head_dim, base, positions.device)
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[..., :d/2], [..., d/2:] -> [-second half, first half]."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """Apply RoPE to [B, H, S, Dh] at the given absolute positions.

    ``positions`` is [S] and must be *global*. Passing local indices is the bug
    this module exists to prevent, and it is silent — see the module docstring.
    """
    if positions.shape[0] != x.shape[2]:
        raise ValueError(
            f"positions has {positions.shape[0]} entries but x has sequence length "
            f"{x.shape[2]}; these must match one-to-one"
        )
    cos, sin = rope_tables(positions, x.shape[-1], base)
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, S, Dh]
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def apply_rope_sharded(
    x_local: torch.Tensor, layout: ShardLayout, rank: int, base: float = 10000.0
) -> torch.Tensor:
    """Apply RoPE to a device's shard using the layout's global positions.

    The safe entry point. Prefer this over calling :func:`apply_rope` with
    hand-computed offsets — under a striped layout there is no single offset that
    is correct.
    """
    positions = positions_for_layout(layout, rank, device=x_local.device)
    return apply_rope(x_local, positions, base)


def apply_rope_local_positions_WRONG(
    x_local: torch.Tensor, layout: ShardLayout, rank: int, base: float = 10000.0
) -> torch.Tensor:
    """The bug, written down deliberately.

    Applies RoPE with local buffer indices instead of global positions. Kept so
    that `tests/test_rope.py` can assert the test suite actually *detects* it —
    a positional test that passes for both the correct and incorrect version is
    not testing anything.

    Never call this outside tests.
    """
    positions = torch.arange(x_local.shape[2], device=x_local.device, dtype=torch.long)
    return apply_rope(x_local, positions, base)
