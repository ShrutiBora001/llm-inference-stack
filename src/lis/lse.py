"""Log-sum-exp merging of partial attention results.

This module is what makes the Triton phase tractable. The upstream ring
implementation carries `(acc, m, l)` state through its own tiling loop, which
means a fused kernel would have to be ring-aware — it would need to accept and
return that state. Writing such a kernel from scratch is the expensive path.

The cheaper and standard path: every FlashAttention-style kernel already returns
`(out, lse)` for the block it processed, where `lse = m + log(l)` is the
log-sum-exp of that block's scores. Two such partial results merge exactly:

    lse_ab = logaddexp(lse_a, lse_b)
    out_ab = out_a * exp(lse_a - lse_ab) + out_b * exp(lse_b - lse_ab)

So a stock attention kernel can be called once per (query chunk, key chunk) pair
and the results merged here. The ring never enters the kernel. That reduces the
Phase 1 kernel from "write ring attention in Triton" to "call a flash kernel and
merge", which is a far smaller and better-tested surface.

The merge is associative and commutative, which is the same order-invariance the
ring already depends on — see the upstream report, section 2.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# lse = -inf means "this partial result saw no unmasked key". Representing it
# with an actual -inf makes the merge algebra work without branching, but the
# final normalization has to handle a row that never saw anything.
NEG_INF = float("-inf")


@dataclass
class PartialAttention:
    """One chunk's attention output plus the log-sum-exp needed to merge it.

    out: [B, H, Sq, Dh]  already normalized by that block's own denominator
    lse: [B, H, Sq]      log(sum(exp(scores))) over the keys this block saw
    """

    out: torch.Tensor
    lse: torch.Tensor

    def __post_init__(self) -> None:
        if self.out.shape[:-1] != self.lse.shape:
            raise ValueError(
                f"out {tuple(self.out.shape)} and lse {tuple(self.lse.shape)} disagree; "
                "lse must match out without the head-dim axis"
            )

    @classmethod
    def empty(
        cls, batch: int, heads: int, q_len: int, head_dim: int, *, device, dtype
    ) -> "PartialAttention":
        """Identity element for :func:`merge`. Merging with it is a no-op."""
        return cls(
            out=torch.zeros(batch, heads, q_len, head_dim, device=device, dtype=torch.float32),
            lse=torch.full((batch, heads, q_len), NEG_INF, device=device, dtype=torch.float32),
        )


def merge(a: PartialAttention, b: PartialAttention) -> PartialAttention:
    """Combine two partial attention results into one.

    Exact, not approximate: the result equals attention computed over the union
    of the two key sets in a single pass.

    Associative and commutative, so partials may be combined in any order — the
    property the ring relies on, since blocks arrive in device order rather than
    sequence order.
    """
    lse_a, lse_b = a.lse.float(), b.lse.float()

    # torch.logaddexp handles the -inf cases correctly: logaddexp(-inf, x) = x,
    # and logaddexp(-inf, -inf) = -inf. Doing this by hand with exp() would
    # produce nan for the both-empty case.
    lse_ab = torch.logaddexp(lse_a, lse_b)

    # Where both partials are empty the weights below would be exp(-inf - -inf) =
    # exp(nan). Pin those rows so the weights come out as 0 and the merged output
    # stays zero, matching the upstream -inf guard in block_update.
    both_empty = torch.isneginf(lse_ab)
    safe_lse_ab = torch.where(both_empty, torch.zeros_like(lse_ab), lse_ab)

    w_a = torch.exp(lse_a - safe_lse_ab).unsqueeze(-1)
    w_b = torch.exp(lse_b - safe_lse_ab).unsqueeze(-1)

    return PartialAttention(
        out=a.out.float() * w_a + b.out.float() * w_b,
        lse=lse_ab,
    )


def merge_all(partials: list[PartialAttention]) -> PartialAttention:
    """Fold a list of partials. Order does not affect the result."""
    if not partials:
        raise ValueError("nothing to merge")
    acc = partials[0]
    for p in partials[1:]:
        acc = merge(acc, p)
    return acc


def attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> PartialAttention:
    """Reference partial attention: the contract a fused kernel must satisfy.

    This is the CPU oracle. The Triton kernel in ``lis.kernels.triton_flash``
    must return exactly this, to tolerance, for every shape — which is what the
    kernel's correctness test asserts. Writing the contract down as runnable code
    means the kernel has something precise to be wrong against.

    mask: [Sq, Sk] boolean, True = keep.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

    if mask is not None:
        scores = scores.masked_fill(~mask, NEG_INF)

    lse = torch.logsumexp(scores, dim=-1)  # [B, H, Sq]

    # Rows with no visible key: logsumexp gives -inf, and softmax would be nan.
    # Emit zeros and let lse = -inf mark the row as empty, so merge() treats this
    # partial as contributing nothing.
    safe = torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse)
    weights = torch.exp(scores - safe.unsqueeze(-1))

    return PartialAttention(out=torch.matmul(weights, v.float()), lse=lse)


def normalize(p: PartialAttention) -> torch.Tensor:
    """Final output. Rows that never saw a key come out as zeros, not nan."""
    return p.out
