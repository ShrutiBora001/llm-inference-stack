"""LSE merging of partial attention results.

This is the contract a fused kernel must satisfy. Once it holds, a stock
FlashAttention kernel can be called once per (query chunk, key chunk) pair and
the results combined here — so the Triton work never has to be ring-aware.

If these tests are right and the kernel matches `attention_with_lse` to
tolerance, ring correctness follows without re-deriving it.
"""

from __future__ import annotations

import math

import pytest
import torch

from dattn.reference import attention_reference
from dattn.utils import max_abs_diff, set_seed
from lis.lse import (
    PartialAttention,
    attention_with_lse,
    merge,
    merge_all,
    normalize,
)

B, H, SQ, SK, DH = 2, 4, 32, 64, 16
SCALE = 1.0 / math.sqrt(DH)


def make_qkv(sq=SQ, sk=SK, seed=0):
    set_seed(seed)
    return (
        torch.randn(B, H, sq, DH),
        torch.randn(B, H, sk, DH),
        torch.randn(B, H, sk, DH),
    )


def test_partial_matches_reference_when_unsplit():
    """One partial over all keys is just attention."""
    q, k, v = make_qkv()
    got = normalize(attention_with_lse(q, k, v, scale=SCALE))
    assert max_abs_diff(got, attention_reference(q, k, v, scale=SCALE)) < 1e-5


def test_lse_is_logsumexp_of_scores():
    """The lse must be the real thing, not a rescaled proxy."""
    q, k, v = make_qkv()
    p = attention_with_lse(q, k, v, scale=SCALE)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * SCALE
    assert max_abs_diff(p.lse, torch.logsumexp(scores, dim=-1)) < 1e-5


@pytest.mark.parametrize("split", [1, 8, 16, 32, 63])
def test_merge_equals_single_pass(split):
    """Splitting the keys anywhere and merging must be exact."""
    q, k, v = make_qkv()
    want = attention_reference(q, k, v, scale=SCALE)
    got = normalize(merge(
        attention_with_lse(q, k[:, :, :split], v[:, :, :split], scale=SCALE),
        attention_with_lse(q, k[:, :, split:], v[:, :, split:], scale=SCALE),
    ))
    assert max_abs_diff(got, want) < 1e-5


def test_merge_is_order_invariant():
    """Same property the ring depends on: blocks may arrive in any order."""
    q, k, v = make_qkv()
    chunks = [
        attention_with_lse(q, k[:, :, i:i + 16], v[:, :, i:i + 16], scale=SCALE)
        for i in range(0, SK, 16)
    ]
    forward = normalize(merge_all(chunks))
    shuffled = normalize(merge_all([chunks[i] for i in (2, 0, 3, 1)]))
    assert max_abs_diff(forward, shuffled) < 1e-6


def test_merge_is_associative():
    q, k, v = make_qkv()
    a, b, c = (
        attention_with_lse(q, k[:, :, i:i + 16], v[:, :, i:i + 16], scale=SCALE)
        for i in (0, 16, 32)
    )
    assert max_abs_diff(normalize(merge(merge(a, b), c)),
                        normalize(merge(a, merge(b, c)))) < 1e-6


def test_empty_partial_is_the_identity():
    """Merging with an empty partial changes nothing — needed so a device can
    skip a causally-masked block without special-casing the merge."""
    q, k, v = make_qkv()
    real = attention_with_lse(q, k, v, scale=SCALE)
    empty = PartialAttention.empty(B, H, SQ, DH, device=q.device, dtype=q.dtype)
    assert max_abs_diff(normalize(merge(real, empty)), normalize(real)) < 1e-6
    assert max_abs_diff(normalize(merge(empty, real)), normalize(real)) < 1e-6


def test_two_empty_partials_do_not_produce_nan():
    """logaddexp(-inf, -inf) = -inf, and the weights must not become nan.

    Same edge case as the upstream -inf guard: a query row that has seen no key
    yet must stay at zero rather than poisoning the merge.
    """
    empty = PartialAttention.empty(B, H, SQ, DH, device="cpu", dtype=torch.float32)
    merged = merge(empty, empty)
    assert torch.isfinite(merged.out).all()
    assert torch.isneginf(merged.lse).all()
    assert normalize(merged).abs().max().item() == 0.0


def test_fully_masked_partial_is_empty():
    """A block entirely in the future must merge as a no-op."""
    q, k, v = make_qkv()
    masked = attention_with_lse(
        q, k, v, scale=SCALE, mask=torch.zeros(SQ, SK, dtype=torch.bool)
    )
    assert torch.isneginf(masked.lse).all()
    real = attention_with_lse(q, k, v, scale=SCALE)
    assert max_abs_diff(normalize(merge(real, masked)), normalize(real)) < 1e-6


def test_causal_chunked_merge_matches_reference():
    """The real use case: causal attention assembled from chunk pairs.

    This is exactly what the ring does — per (q chunk, k chunk) pair, skip if
    entirely future, otherwise compute a partial and merge.
    """
    set_seed(3)
    S = 64
    q, k, v = (torch.randn(B, H, S, DH) for _ in range(3))
    want = attention_reference(q, k, v, causal=True, scale=SCALE)

    chunk = 16
    outs = []
    for qs in range(0, S, chunk):
        parts = []
        q_pos = torch.arange(qs, qs + chunk)
        for ks in range(0, S, chunk):
            k_pos = torch.arange(ks, ks + chunk)
            mask = q_pos[:, None] >= k_pos[None, :]
            if not mask.any():
                continue  # entirely in the future
            parts.append(attention_with_lse(
                q[:, :, qs:qs + chunk], k[:, :, ks:ks + chunk], v[:, :, ks:ks + chunk],
                scale=SCALE, mask=mask,
            ))
        outs.append(normalize(merge_all(parts)))

    assert max_abs_diff(torch.cat(outs, dim=2), want) < 1e-5


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="disagree"):
        PartialAttention(out=torch.zeros(1, 1, 4, 8), lse=torch.zeros(1, 1, 5))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_partials_merge(dtype):
    """Kernels return reduced precision; the merge stays fp32 internally."""
    q, k, v = (t.to(dtype) for t in make_qkv())
    want = attention_reference(q, k, v, scale=SCALE)
    got = normalize(merge(
        attention_with_lse(q, k[:, :, :32], v[:, :, :32], scale=SCALE),
        attention_with_lse(q, k[:, :, 32:], v[:, :, 32:], scale=SCALE),
    ))
    assert max_abs_diff(got, want) < 2e-2
