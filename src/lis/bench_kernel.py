"""Kernel throughput measurement: TFLOP/s, MFU, and roofline position.

Importable rather than a script, because `tests/test_contract.py` calls
`measure_mfu` to enforce the gate. The CLI in `bench/kernel.py` is a thin
wrapper that sweeps and writes results.

The numerator matters more than it looks. FLOPs are counted over the elements
the kernel was *actually asked to compute*, including entries it masks away --
not the causally-useful count. A causal kernel skips whole key blocks past the
diagonal but computes the diagonal block in full, so the honest count depends on
the tile size. Using the useful count instead would inflate MFU by roughly the
masked fraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import torch

from dattn.analysis import attention_flops
from dattn.profiles import get_profile

from .kernels import available_backends, flash_attention, last_backend_used
from .metrics import achieved_tflops, arithmetic_intensity, mfu

# HBM traffic a *fused* kernel cannot avoid: read Q, K, V once, write O.
# The LSE vector is O(S) and negligible beside the O(S*D) tensors.
_TENSORS_TOUCHED = 4


def elements_computed(sq: int, sk: int, causal: bool, block_m: int = 128) -> int:
    """Score entries the kernel asks the tensor cores for.

    Non-causal: the full rectangle. Causal: each query tile of `block_m` rows
    processes keys up to the end of its own tile, so the count is the triangle
    plus a half-tile of overshoot per row-block. That overshoot is real work and
    belongs in the denominator of MFU.
    """
    if not causal:
        return sq * sk
    total = 0
    for start in range(0, sq, block_m):
        rows = min(block_m, sq - start)
        keys = min(sk, start + block_m)  # kernel stops at the tile's diagonal
        total += rows * keys
    return total


def hbm_bytes_fused(seq: int, heads: int, head_dim: int, bytes_per_elem: int) -> int:
    """Minimum HBM traffic for a fused kernel: Q, K, V in, O out.

    The score tile never leaves SRAM -- that is the whole point of fusing, and
    the reason the roofline position moves.
    """
    return _TENSORS_TOUCHED * seq * heads * head_dim * bytes_per_elem


def hbm_bytes_unfused(
    seq: int, heads: int, head_dim: int, bytes_per_elem: int, block_n: int, passes: int = 3
) -> int:
    """HBM traffic when the score tile is materialized.

    The tile is `seq x block_n` in fp32 and is written, re-read for the masked
    softmax, and read again by the second GEMM. `passes` is that traffic
    multiplier -- an estimate, and labelled as one. Upstream measured ~90 GiB at
    S=32768 against a fused ideal of ~1 GiB, which is the gap this quantifies.
    """
    tiles = max(1, seq // block_n)
    score_traffic = passes * seq * heads * block_n * 4 * tiles
    return hbm_bytes_fused(seq, heads, head_dim, bytes_per_elem) + score_traffic


@dataclass
class KernelMeasurement:
    backend: str
    seq: int
    heads: int
    head_dim: int
    causal: bool
    dtype: str
    device: str
    median_ms: float
    min_ms: float
    elements: int
    flops: float
    tflops: float
    mfu: float
    arithmetic_intensity: float
    peak_tflops: float | None
    profile: str

    def as_dict(self) -> dict:
        return asdict(self)


def _time_call(fn, iters: int, warmup: int, cuda: bool) -> list[float]:
    """Median-of-N timing. CUDA events on GPU, since launches are async and a
    wall clock would measure launch overhead rather than execution."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if cuda:
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            fn()
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return sorted(samples)


def benchmark(
    *,
    seq: int = 8192,
    heads: int = 32,
    head_dim: int = 128,
    causal: bool = True,
    backend: str | None = None,
    dtype: torch.dtype | None = None,
    iters: int = 10,
    warmup: int = 3,
    block_m: int = 128,
    profile_name: str = "auto",
) -> KernelMeasurement:
    """Measure one backend at one shape."""
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if cuda else torch.float32

    q, k, v = (torch.randn(1, heads, seq, head_dim, device=device, dtype=dtype)
               for _ in range(3))

    with torch.no_grad():
        samples = _time_call(
            lambda: flash_attention(q, k, v, causal=causal, backend=backend),
            iters, warmup, cuda,
        )
        used = last_backend_used()

    median = samples[len(samples) // 2]
    elements = elements_computed(seq, seq, causal, block_m)
    flops = attention_flops(elements, heads, head_dim)
    tflops = achieved_tflops(flops, median / 1000.0)

    profile = get_profile(profile_name)
    peak = profile.peak_tflops
    bytes_per_elem = torch.empty(0, dtype=dtype).element_size()
    hbm = (hbm_bytes_fused(seq, heads, head_dim, bytes_per_elem)
           if used != "unfused"
           else hbm_bytes_unfused(seq, heads, head_dim, bytes_per_elem, block_n=128))

    return KernelMeasurement(
        backend=used or "unknown",
        seq=seq, heads=heads, head_dim=head_dim, causal=causal,
        dtype=str(dtype), device=str(device),
        median_ms=median, min_ms=samples[0],
        elements=elements, flops=flops, tflops=tflops,
        mfu=mfu(tflops, peak) if peak else 0.0,
        arithmetic_intensity=arithmetic_intensity(flops, hbm),
        peak_tflops=peak, profile=profile.name,
    )


def measure_mfu(
    *, seq: int = 8192, heads: int = 32, head_dim: int = 128, causal: bool = True,
    backend: str | None = None,
) -> float:
    """MFU at one shape. Called by the contract suite to enforce the gate."""
    return benchmark(seq=seq, heads=heads, head_dim=head_dim,
                     causal=causal, backend=backend).mfu


def sweep(
    seqs: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768),
    backends: tuple[str, ...] | None = None,
    **kw,
) -> list[KernelMeasurement]:
    """Every available backend across sequence lengths.

    Backends that cannot express a case hand off rather than failing the sweep;
    a missing row is recorded by its absence, not by a crash.

    OOM is expected and is a *result*, not a defect: the unfused backend
    materializes an [H, S, S] fp32 score tile, which is 34 GiB at S=16384 before
    intermediates, so it drops out of the sweep well before the fused paths do.
    That boundary is the memory argument this project makes. The allocator is
    drained afterwards -- otherwise the failed allocation stays cached and the
    *next* measurement OOMs spuriously, which would look like a much lower limit
    than the hardware actually has.
    """
    chosen = backends or tuple(available_backends())
    out = []
    for backend in chosen:
        for seq in seqs:
            try:
                out.append(benchmark(seq=seq, backend=backend, **kw))
            except torch.cuda.OutOfMemoryError as e:  # pragma: no cover
                torch.cuda.empty_cache()
                print(f"  OOM  {backend} seq={seq} (memory boundary, not a failure)")
                del e
            except (RuntimeError, ValueError) as e:  # pragma: no cover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  skip {backend} seq={seq}: {e}")
    return out
