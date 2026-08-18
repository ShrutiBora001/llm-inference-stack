"""Kernel dispatch. One interface, several backends, identical contract.

Every backend takes the same arguments and returns a `PartialAttention`
(out, lse), so they are substitutable at any call site. That is what lets
Phase 2 and Phase 3 be developed against *either* kernel:

  - the serving integration, the persistent CUDA ring and the NVSHMEM decode
    path can all be written and CI-tested on a laptop against `unfused`
  - the same code runs on `torch_flash` or `triton` on a GPU with no change
  - benchmarking both isolates the kernel's contribution from the serving
    stack's, which a single-kernel design cannot do

Backends, in dispatch preference:

  triton       hand-written, autotuned. NEVER EXECUTED - see triton_flash.py
  torch_flash  PyTorch's fused kernel via the private aten op that returns LSE
  unfused      reference. Runs anywhere. The oracle, kept permanently

Preference is deliberate: `torch_flash` outranks `triton` until the Triton
kernel is measured, because an untested hand-written kernel should not silently
become the default. Flip `PREFERENCE` once it has earned it.
"""

from __future__ import annotations

import os

import torch

from ..lse import PartialAttention
from .torch_flash import available as _torch_flash_available
from .torch_flash import flash_attention_torch
from .unfused import flash_attention_unfused

PREFERENCE = ("torch_flash", "triton", "unfused")

_LAST_BACKEND: str | None = None


def _triton_available() -> bool:
    try:
        from .triton_flash import available
    except ImportError:  # pragma: no cover
        return False
    return available()


def triton_available() -> bool:
    """Whether a real Triton path exists here.

    Reported rather than assumed. Silently falling back to eager while still
    claiming a Triton kernel is the failure mode the contract suite guards.
    """
    return _triton_available()


def available_backends() -> list[str]:
    """Backends usable on this machine, in dispatch order."""
    checks = {
        "triton": _triton_available,
        "torch_flash": _torch_flash_available,
        "unfused": lambda: True,
    }
    return [name for name in PREFERENCE if checks[name]()]


def last_backend_used() -> str | None:
    """Which backend the most recent flash_attention call actually took."""
    return _LAST_BACKEND


def autotune_report() -> dict:
    """Triton autotuner selections, keyed by problem shape. Empty off-GPU."""
    try:
        from .triton_flash import LAST_CONFIG
    except ImportError:  # pragma: no cover
        return {}
    return dict(LAST_CONFIG)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    q_offset: int = 0,
    k_offset: int = 0,
    backend: str | None = None,
) -> PartialAttention:
    """Attention over one (query chunk, key chunk) pair.

    ``backend`` forces a specific implementation; the default picks the best
    available. Forcing is how the benchmark A/B's kernels against each other on
    identical inputs, and how tests pin the oracle.

    ``q_offset``/``k_offset`` are GLOBAL sequence positions -- the invariant
    every layer of this project shares.
    """
    global _LAST_BACKEND

    if backend is None:
        order = available_backends()
    else:
        if backend not in PREFERENCE:
            raise ValueError(f"unknown backend {backend!r}; choose from {PREFERENCE}")
        if backend != "unfused" and backend not in available_backends():
            raise RuntimeError(f"backend {backend!r} is not available on this machine")
        order = [backend]

    kw = dict(causal=causal, scale=scale, q_offset=q_offset, k_offset=k_offset)
    last_error: Exception | None = None

    for name in order:
        try:
            if name == "triton":
                from .triton_flash import flash_attention_triton
                result = flash_attention_triton(q, k, v, **kw)
            elif name == "torch_flash":
                result = flash_attention_torch(q, k, v, **kw)
            else:
                result = flash_attention_unfused(q, k, v, **kw)
        except ValueError as e:
            # A backend that cannot express this case (torch_flash refuses
            # partially-overlapping causal blocks) hands off to the next one.
            # A wrong answer would be far worse than a slower one.
            last_error = e
            if backend is not None:
                raise
            continue

        _LAST_BACKEND = name
        return result

    raise RuntimeError(f"no backend could handle this case: {last_error}")


__all__ = [
    "PREFERENCE",
    "autotune_report",
    "available_backends",
    "flash_attention",
    "last_backend_used",
    "triton_available",
]
