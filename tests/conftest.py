"""Shared test configuration.

One job: make fp32 mean fp32.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def _disable_tf32():
    """TF32 off for the whole suite.

    Some CUDA builds enable TF32 for fp32 matmul by default. It silently drops
    the mantissa from 24 bits to about 10, which is invisible until a tolerance
    that holds everywhere else starts failing on GPU only -- and it reads like a
    correctness bug in the code under test rather than a precision setting.

    This cost real debugging time upstream (see the experimental-setup notes in
    the causal-attention-load-imbalance repo) and cost it again here: the
    kernel-equivalence tests began running fp32 on CUDA for the first time and
    two of them failed on precision alone.

    Benchmarks are unaffected -- they run bf16 and set their own policy. This is
    about correctness tests only, where the whole point is an exact identity.
    """
    if not torch.cuda.is_available():
        yield
        return

    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
