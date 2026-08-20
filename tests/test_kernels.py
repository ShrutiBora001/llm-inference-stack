"""Kernel dispatch and the cross-backend equivalence contract.

The point of these tests is that *any* backend must be substitutable for any
other. Once that holds, Phase 2 and Phase 3 can be written and CI-tested on a
laptop against `unfused`, then run unchanged on a fused kernel on a GPU.

Backends that need CUDA skip visibly. The equivalence tests are written so they
automatically cover every backend present, so they will start exercising
torch_flash and triton on a GPU with no edit.
"""

from __future__ import annotations

import math

import pytest
import torch

from dattn.reference import attention_reference
from dattn.utils import max_abs_diff, set_seed
from lis.kernels import (
    PREFERENCE,
    autotune_report,
    available_backends,
    flash_attention,
    last_backend_used,
    triton_available,
)
from lis.lse import merge, normalize

B, H, S, DH = 2, 4, 64, 32
SCALE = 1.0 / math.sqrt(DH)
BACKENDS = available_backends()
FUSED = [b for b in BACKENDS if b != "unfused"]
CUDA = torch.cuda.is_available()

# Tolerances by dtype. fp32 against an fp32 reference is an near-identity, so it
# is held tight; bf16 carries ~8 mantissa bits and cannot be.
TOL = {torch.float32: 1e-4, torch.bfloat16: 2e-2}
LSE_TOL = {torch.float32: 1e-3, torch.bfloat16: 2e-2}


def placement(backend: str | None = None):
    """Where a backend's inputs must live, and in what dtype.

    Not cosmetic. `torch_flash` calls
    `aten._scaled_dot_product_flash_attention`, which accepts only fp16/bf16 and
    only on CUDA; the Triton kernel needs real device pointers. Handing either
    of them a CPU fp32 tensor fails with a dispatcher error or "Pointer argument
    cannot be accessed from Triton (cpu tensor?)".

    That failure cannot be caught on a laptop, because only `unfused` exists
    there and it happily accepts anything. So these tests passed locally for
    every backend that was never actually running -- the same trap as building
    CPU tensors in an NCCL worker. Device and dtype are therefore derived from
    the backend under test rather than assumed.
    """
    fused = bool(FUSED) if backend is None else backend != "unfused"
    if fused and CUDA:
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cuda" if CUDA else "cpu"), torch.float32


def make_qkv(sq=S, sk=S, seed=0, backend: str | None = None):
    """q, k, v placed where `backend` can actually read them."""
    device, dtype = placement(backend)
    set_seed(seed)
    return tuple(
        torch.randn(B, H, n, DH, device=device, dtype=dtype) for n in (sq, sk, sk)
    )


# ------------------------------------------------------------------ dispatch


def test_unfused_is_always_available():
    """The oracle must never be absent — it is what everything else is checked
    against, and what lets serving work be tested without a GPU."""
    assert "unfused" in available_backends()


def test_dispatch_reports_which_backend_ran():
    q, k, v = make_qkv(backend=None)
    flash_attention(q, k, v, causal=True)
    assert last_backend_used() in BACKENDS


def test_dispatch_prefers_fused_when_present():
    """On a GPU the default must not quietly pick the slow path."""
    q, k, v = make_qkv(backend=None)
    flash_attention(q, k, v, causal=True)
    assert last_backend_used() == BACKENDS[0]


def test_forcing_a_backend_is_honoured():
    q, k, v = make_qkv(backend="unfused")
    flash_attention(q, k, v, causal=True, backend="unfused")
    assert last_backend_used() == "unfused"


def test_unknown_backend_rejected():
    q, k, v = make_qkv(backend="unfused")
    with pytest.raises(ValueError, match="unknown backend"):
        flash_attention(q, k, v, backend="cutlass")


def test_unavailable_backend_rejected_rather_than_silently_downgraded():
    """Asking for a backend you do not have must fail loudly. Silently running
    something else is how a project ends up claiming a kernel it never used."""
    missing = [b for b in PREFERENCE if b not in BACKENDS]
    if not missing:
        pytest.skip("every backend is available here")
    q, k, v = make_qkv(backend="unfused")
    with pytest.raises(RuntimeError, match="not available"):
        flash_attention(q, k, v, backend=missing[0])


def test_triton_availability_is_honest():
    assert isinstance(triton_available(), bool)
    if not torch.cuda.is_available():
        assert not triton_available()
    assert isinstance(autotune_report(), dict)


# --------------------------------------------------------------- correctness


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("causal", [False, True])
def test_backend_matches_reference(backend, causal):
    """Every backend against the single-process reference.

    The reference is itself pinned to PyTorch's SDPA upstream, so the trust
    root is PyTorch rather than this codebase.
    """
    q, k, v = make_qkv(backend=backend)
    got = flash_attention(q, k, v, causal=causal, scale=SCALE, backend=backend)
    want = attention_reference(q, k, v, causal=causal, scale=SCALE)
    tol = TOL[q.dtype]
    assert max_abs_diff(normalize(got), want) < tol, f"{backend} causal={causal}"


@pytest.mark.skipif(len(FUSED) == 0, reason="no fused backend here — GPU-gated")
@pytest.mark.parametrize("backend", FUSED)
@pytest.mark.parametrize("causal", [False, True])
def test_fused_agrees_with_unfused(backend, causal):
    """Substitutability, stated directly: fused and unfused must agree.

    This is the test that makes 'Phase 2 works with either kernel' true rather
    than hoped.
    """
    q, k, v = make_qkv(backend=backend)
    a = flash_attention(q, k, v, causal=causal, scale=SCALE, backend=backend)
    b = flash_attention(q, k, v, causal=causal, scale=SCALE, backend="unfused")
    assert max_abs_diff(normalize(a), normalize(b)) < TOL[q.dtype]
    assert max_abs_diff(a.lse, b.lse) < LSE_TOL[q.dtype], (
        "lse disagrees — merging would be wrong"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_lse_enables_correct_chunk_merging(backend):
    """The property the ring depends on, per backend.

    Splitting the keys and merging through lse must equal a single pass. If a
    backend's lse convention differed (log2 instead of ln, say) this fails
    while the outputs alone would still look right.
    """
    q, k, v = make_qkv(backend=backend)
    want = attention_reference(q, k, v, scale=SCALE)
    got = merge(
        flash_attention(q, k[:, :, :32], v[:, :, :32], scale=SCALE, backend=backend),
        flash_attention(q, k[:, :, 32:], v[:, :, 32:], scale=SCALE, backend=backend),
    )
    assert max_abs_diff(normalize(got), want) < TOL[q.dtype]


@pytest.mark.parametrize("backend", BACKENDS)
def test_past_block_needs_no_mask(backend):
    """An off-diagonal block entirely in the past is fully visible.

    This is the common case in the ring: most (q chunk, k chunk) pairs are
    either wholly past or wholly future, and only the diagonal is triangular.
    """
    q, k, v = make_qkv(sq=32, sk=32, backend=backend)
    got = flash_attention(q, k, v, causal=True, scale=SCALE,
                          q_offset=64, k_offset=0, backend=backend)
    want = attention_reference(q, k, v, causal=False, scale=SCALE)
    assert max_abs_diff(normalize(got), want) < TOL[q.dtype]


def test_offset_causal_block_matches_reference():
    """Query chunk at a global offset against keys from position 0.

    Mirrors what a striped device does with a remote KV chunk, and is where a
    wrong offset would silently produce a wrong mask.
    """
    device, dtype = placement("unfused")
    set_seed(5)
    q_full, k_full, v_full = (
        torch.randn(B, H, 128, DH, device=device, dtype=dtype) for _ in range(3)
    )
    want = attention_reference(q_full, k_full, v_full, causal=True, scale=SCALE)

    got = flash_attention(
        q_full[:, :, 64:], k_full[:, :, :64], v_full[:, :, :64],
        causal=True, scale=SCALE, q_offset=64, k_offset=0, backend="unfused",
    )
    # Queries 64..127 against keys 0..63 — all in the past, so this partial is
    # the first half of their attention; merge with the diagonal to get the rest.
    diag = flash_attention(
        q_full[:, :, 64:], k_full[:, :, 64:], v_full[:, :, 64:],
        causal=True, scale=SCALE, q_offset=64, k_offset=64, backend="unfused",
    )
    assert max_abs_diff(normalize(merge(got, diag)), want[:, :, 64:]) < 1e-4


def test_future_block_is_refused_or_empty():
    """A block wholly in the future should never be computed.

    The ring skips these; a backend reached with one anyway must not silently
    return a plausible-looking answer.
    """
    q, k, v = make_qkv(sq=32, sk=32, backend="unfused")
    got = flash_attention(q, k, v, causal=True, scale=SCALE,
                          q_offset=0, k_offset=64, backend="unfused")
    assert torch.isneginf(got.lse).all()
    assert normalize(got).abs().max().item() == 0.0
