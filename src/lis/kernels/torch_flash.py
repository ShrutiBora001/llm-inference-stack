"""Fused backend built on PyTorch's own FlashAttention kernel.

The public `F.scaled_dot_product_attention` returns only the output, and ring
attention needs each block's log-sum-exp to merge partials across chunks. That
is why the upstream project ran unfused for its whole first GPU session.

`torch.ops.aten._scaled_dot_product_flash_attention` *does* return the LSE. It
is a private-ish op, so it is probed at import and the backend simply reports
itself unavailable if the signature has changed -- the dispatcher then falls
back rather than breaking.

Strategically this matters more than the Triton kernel at first: it delivers
fused performance immediately, against a well-tested kernel, so Phase 2 serving
work is never blocked waiting on hand-written code. Triton then has to beat a
real baseline rather than a 3%-MFU strawman.
"""

from __future__ import annotations

import math

import torch

from ..lse import PartialAttention


def available() -> bool:
    """Whether the private flash op exists and CUDA is present.

    Probed rather than assumed: this op is not part of the public API and its
    signature has changed across releases.
    """
    if not torch.cuda.is_available():
        return False
    return hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention")


def flash_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    q_offset: int = 0,
    k_offset: int = 0,
) -> PartialAttention:
    """Fused attention returning (out, lse), matching the unfused contract.

    Restriction worth understanding: the kernel's own `is_causal` assumes the
    query and key blocks are aligned at the same origin. That holds only when
    `q_offset == k_offset` -- the diagonal block of the ring. Off-diagonal
    blocks are either entirely visible (past) or entirely skipped (future), so
    they need no mask at all.

    A partially-overlapping causal block with mismatched offsets cannot be
    expressed through this op, so we refuse rather than silently compute the
    wrong mask. Both layouts in this project produce aligned equal chunks, so
    the case does not arise; the check exists so a future uneven layout fails
    loudly instead of quietly.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    sq, sk = q.shape[2], k.shape[2]

    if causal:
        q_end, k_end = q_offset + sq, k_offset + sk
        if k_end <= q_offset:
            need_mask = False        # entirely in the past: every key visible
        elif k_offset >= q_end:
            raise ValueError(
                "block is entirely in the future; the caller should have skipped it"
            )
        elif q_offset == k_offset and sq == sk:
            need_mask = True         # the diagonal block
        else:
            raise ValueError(
                f"partially-overlapping causal block (q_offset={q_offset}, "
                f"k_offset={k_offset}) cannot use the fused causal path; "
                "use the unfused backend for this layout"
            )
    else:
        need_mask = False

    out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        dropout_p=0.0, is_causal=need_mask, scale=scale,
    )
    # The op returns lse in natural log with shape [B, H, Sq], which is exactly
    # what lse.merge expects. No conversion needed -- but do not assume it, the
    # dispatcher's correctness test pins it against the unfused oracle.
    return PartialAttention(out=out, lse=lse)
