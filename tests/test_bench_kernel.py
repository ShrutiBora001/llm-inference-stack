"""The FLOP numerator, which decides every MFU number.

Counting the causally-useful entries instead of the entries the kernel actually
computes would inflate MFU by roughly the masked fraction -- about 2x for a
causal kernel. That would be a flattering error, which is the kind worth a test.
"""

from __future__ import annotations

import pytest

from lis.bench_kernel import (
    benchmark,
    elements_computed,
    hbm_bytes_fused,
    hbm_bytes_unfused,
    measure_mfu,
)


def test_non_causal_is_the_full_rectangle():
    assert elements_computed(1024, 2048, causal=False) == 1024 * 2048


def test_causal_is_about_half_but_not_exactly():
    """Roughly the triangle, plus a half-tile of overshoot per row block.

    Exactly S^2/2 would mean the kernel skipped masked entries *within* the
    diagonal tile, which no block-tiled kernel does.
    """
    s, block = 8192, 128
    got = elements_computed(s, s, causal=True, block_m=block)
    assert got > s * s / 2
    assert got == pytest.approx(s * s / 2, rel=0.02)


def test_larger_tiles_compute_more_masked_work():
    """Bigger query tiles overshoot the diagonal further. Real, and it is why
    the count depends on block size rather than being a pure triangle."""
    s = 4096
    assert elements_computed(s, s, True, 256) > elements_computed(s, s, True, 64)


def test_single_tile_degenerates_to_the_rectangle():
    """If the whole sequence is one tile there is nothing to skip."""
    assert elements_computed(128, 128, True, block_m=128) == 128 * 128


def test_fused_hbm_traffic_is_four_tensors():
    """Q, K, V in and O out. The score tile never reaches HBM — that is the
    entire point of fusing."""
    seq, h, d, b = 8192, 32, 128, 2
    assert hbm_bytes_fused(seq, h, d, b) == 4 * seq * h * d * b


def test_unfused_traffic_dwarfs_fused():
    """Quantifies the upstream finding: ~90 GiB measured against a fused ideal
    near 1 GiB at S=32768."""
    args = (32768, 32, 128, 2)
    fused = hbm_bytes_fused(*args)
    unfused = hbm_bytes_unfused(*args, block_n=128)
    assert unfused > 20 * fused


def test_benchmark_runs_on_cpu_and_reports_the_backend_used():
    """Timings are meaningless here; the point is that the harness works end to
    end before a GPU is rented."""
    m = benchmark(seq=256, heads=4, head_dim=32, iters=2, warmup=1)
    assert m.backend == "unfused"
    assert m.tflops > 0
    assert m.elements == elements_computed(256, 256, True, 128)
    assert m.arithmetic_intensity > 0


def test_measurement_serializes_for_results_files():
    import json

    json.dumps(benchmark(seq=128, heads=2, head_dim=32, iters=1, warmup=0).as_dict())


def test_measure_mfu_is_zero_without_a_peak_figure():
    """The CPU profile carries no peak TFLOP/s, so MFU is undefined rather than
    invented. The contract gate is GPU-gated for exactly this reason."""
    assert measure_mfu(seq=128, heads=2, head_dim=32) == 0.0
