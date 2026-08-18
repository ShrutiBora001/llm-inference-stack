"""Unfused reference backend. The oracle every fused kernel is diffed against.

Runs anywhere, including a laptop with no CUDA, which is what lets the whole
serving integration in later phases be tested in CI. Slow by construction: it
materializes the score tile, which is exactly what FlashAttention exists to
avoid. See the upstream report section 6.7 -- this path measures 2.5-3.4% of
A100 peak.

Kept permanently, not as a stopgap. A fast kernel with nothing to be wrong
against is a fast kernel you cannot trust.
"""

from __future__ import annotations

import torch

from ..lse import PartialAttention, attention_with_lse


def flash_attention_unfused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    q_offset: int = 0,
    k_offset: int = 0,
) -> PartialAttention:
    """Attention over one (query chunk, key chunk) pair.

    ``q_offset``/``k_offset`` are GLOBAL sequence positions. Under a striped
    layout a device's chunks are not contiguous, so causal masking cannot be
    derived from local indices -- the same invariant as everywhere else in this
    project.
    """
    mask = None
    if causal:
        qp = torch.arange(q.shape[2], device=q.device) + q_offset
        kp = torch.arange(k.shape[2], device=k.device) + k_offset
        mask = qp[:, None] >= kp[None, :]

    return attention_with_lse(q, k, v, scale=scale, mask=mask)
