"""Kernel implementations. Triton is optional and guarded.

Triton ships with PyTorch on Linux/CUDA and is absent on macOS, so every import
here is guarded. The CPU test suite must run without it — the contract suite
skips the GPU-gated checks visibly rather than passing them vacuously.
"""

from __future__ import annotations

_LAST_BACKEND: str | None = None
_AUTOTUNE: dict = {}


def triton_available() -> bool:
    """Whether a real Triton path exists on this machine.

    Reported rather than assumed: the contract suite asserts the project knows
    the answer, because silently falling back to eager PyTorch while still
    claiming a Triton kernel is the failure mode this guards.
    """
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def last_backend_used() -> str | None:
    """Which path the most recent flash_attention call actually took."""
    return _LAST_BACKEND


def autotune_report() -> dict:
    """Autotuner selections, keyed by problem shape. Populated on GPU."""
    return dict(_AUTOTUNE)


__all__ = ["triton_available", "last_backend_used", "autotune_report"]
