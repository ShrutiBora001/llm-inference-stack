"""The anti-decoration contract.

Every technology in this project must be doing something the others cannot do as
well. This suite fails if any of them has degraded into a wrapper around
something else, or is present only to be named.

Each check states what the technology is uniquely good at, and asserts evidence
that it is actually being used that way. A check that cannot run on the current
hardware **skips with a reason** — it never passes vacuously, because a green
suite that silently checked nothing is worse than a red one.

Run with `make contract`.
"""

from __future__ import annotations

import os

import pytest
import torch

CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(not CUDA, reason="needs a CUDA device — GPU-gated contract check")

# Gates chosen BEFORE any measurement, so they cannot be retrofitted to whatever
# the implementation happens to achieve. Raising one requires a written
# justification in the phase report.
MFU_GATE = 0.20            # vs 0.025-0.034 for the unfused upstream loop
BUSBW_FRACTION_GATE = 0.60  # measured NCCL bus bandwidth vs nameplate
DECODE_KERNEL_REDUCTION = 0.50  # CUDA-graph capture must halve launches


# ---------------------------------------------------------------- Triton
# Uniquely best at: authoring fast kernels quickly, with block-level
# abstractions and autotuning. Its failure mode here is silently falling back to
# an eager PyTorch path while the project still claims a Triton kernel.


def test_triton_availability_is_reported_honestly():
    """The project must know whether it has Triton, not assume it."""
    from lis.kernels import triton_available

    available = triton_available()
    assert isinstance(available, bool)
    if not CUDA:
        assert not available, "Triton reported available without a CUDA device"


@gpu_only
def test_triton_kernel_actually_runs_not_a_fallback():
    """Fails if the 'Triton path' is eager PyTorch wearing a hat."""
    from lis.kernels import flash_attention, last_backend_used

    q, k, v = (torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flash_attention(q, k, v, causal=True)
    assert last_backend_used() == "triton", (
        f"expected the Triton kernel, got {last_backend_used()!r} — "
        "the fused path silently fell back"
    )


@gpu_only
def test_triton_autotuner_selects_a_nondefault_config():
    """Autotuning is a large part of why Triton is worth using.

    If the selected configuration is always the first candidate, the autotuner
    is not doing anything and the tile sizes were effectively hardcoded.
    """
    from lis.kernels import autotune_report

    report = autotune_report()
    assert report, "no autotune data recorded — was the kernel ever run?"
    assert any(r["chosen_index"] != 0 for r in report.values()), (
        "autotuner always picked the first config; the search space is not "
        "being explored and Triton's autotuning is decorative"
    )


@gpu_only
def test_triton_kernel_meets_the_mfu_gate():
    """The point of the kernel. Below this gate, fusing bought nothing."""
    from lis.bench_kernel import measure_mfu

    mfu = measure_mfu(seq=8192, heads=32, head_dim=128, causal=True)
    assert mfu >= MFU_GATE, (
        f"achieved {mfu:.1%} MFU, gate is {MFU_GATE:.0%}. The upstream unfused "
        "loop reached 3.3%; a fused kernel that does not clear this gate has "
        "not justified the phase."
    )


# ---------------------------------------------------------------- NCCL
# Uniquely best at: mature, well-tuned bulk collectives. Failure mode: falling
# back to host staging, or moving different bytes than the model predicts.


@gpu_only
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_nccl_moves_the_bytes_the_model_predicts():
    """Measured wire traffic must match the closed-form communication model.

    A mismatch means either the model is wrong or the implementation is moving
    data it should not — both worth failing over.
    """
    from lis.bench_comm import measured_bytes_per_rank, predicted_bytes_per_rank

    got, want = measured_bytes_per_rank(), predicted_bytes_per_rank()
    assert abs(got - want) / want < 0.05, f"moved {got} bytes, model predicts {want}"


@gpu_only
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_nccl_achieves_a_reasonable_fraction_of_nameplate():
    """Guards against P2P being disabled and traffic staging through the host,
    which is the single most common way a multi-GPU box quietly underperforms."""
    from lis.bench_comm import measured_busbw_fraction

    frac = measured_busbw_fraction()
    assert frac >= BUSBW_FRACTION_GATE, (
        f"bus bandwidth is {frac:.0%} of nameplate — suspect PCIe rather than "
        "NVLink, or NCCL_P2P_DISABLE set"
    )


# ---------------------------------------- torch.compile / CUDA graphs ("JIT")
# Uniquely best at: removing launch overhead and fusing elementwise chains.
# Decode is launch-bound, so this is where it pays. Failure mode: graph breaks
# that silently return the model to eager execution.


def test_compile_produces_no_graph_breaks():
    """A graph break means torch.compile fell back to eager for that region.

    Runs on CPU: dynamo's tracing is device-independent, so the most common
    failure is catchable for free.
    """
    from lis.jit import graph_break_report, make_decode_step

    breaks = graph_break_report(make_decode_step(), device="cpu")
    assert not breaks, f"torch.compile introduced graph breaks: {breaks}"


@gpu_only
def test_cuda_graphs_reduce_decode_launches():
    """CUDA graph capture must actually collapse launches, or it is ceremony."""
    from lis.jit import decode_launch_counts

    eager, captured = decode_launch_counts()
    reduction = 1 - captured / eager
    assert reduction >= DECODE_KERNEL_REDUCTION, (
        f"capture reduced launches by only {reduction:.0%} "
        f"({eager} -> {captured}); expected >= {DECODE_KERNEL_REDUCTION:.0%}"
    )


# ---------------------------------------------------------------- vLLM
# Uniquely best at: PagedAttention and mature continuous batching. Failure mode:
# running it in a degenerate single-request, single-block configuration where
# none of that machinery is exercised.


@pytest.mark.skipif("VLLM_TEST" not in os.environ, reason="set VLLM_TEST=1 (needs vLLM installed)")
def test_vllm_block_table_is_nontrivial():
    """A one-block, contiguous block table means paging never happened."""
    from lis.serving.vllm_backend import sample_block_table

    table = sample_block_table()
    assert len(table) > 1, "single block — PagedAttention is not being exercised"
    assert table != sorted(table), (
        "block table is contiguous and in order; the allocator is not actually "
        "paging, so vLLM's central feature is unused"
    )


@pytest.mark.skipif("VLLM_TEST" not in os.environ, reason="set VLLM_TEST=1 (needs vLLM installed)")
def test_vllm_requests_overlap_in_time():
    """Continuous batching means requests share steps. If they run one after
    another, this is a queue with extra steps."""
    from lis.serving.vllm_backend import request_intervals

    intervals = request_intervals()
    overlapped = any(
        a[1] > b[0] for i, a in enumerate(intervals) for b in intervals[i + 1:]
    )
    assert overlapped, "no two requests were in flight together"


# ---------------------------------------------------------------- SGLang
# Uniquely best at: RadixAttention prefix caching. Failure mode: benchmarking on
# a workload with no shared prefixes, where the radix tree is dead weight.


@pytest.mark.skipif("SGLANG_TEST" not in os.environ, reason="set SGLANG_TEST=1 (needs SGLang)")
def test_sglang_prefix_cache_actually_hits():
    """On a shared-prefix workload the radix tree must be doing work.

    A near-zero hit rate means the benchmark chose a workload that cannot
    distinguish SGLang from anything else.
    """
    from lis.serving.sglang_backend import prefix_cache_hit_rate

    rate = prefix_cache_hit_rate()
    assert rate > 0.50, (
        f"prefix cache hit rate {rate:.0%} on a workload built to share prefixes; "
        "RadixAttention is not being exercised"
    )


# ---------------------------------------------------------------- raw CUDA
# Uniquely best at: persistent kernels and device-side scheduling — fusing
# communication into the compute loop, which neither Triton nor CUTLASS
# expresses. Failure mode: a "persistent" kernel relaunched every ring step.


@gpu_only
@pytest.mark.skipif("CUDA_RING" not in os.environ, reason="set CUDA_RING=1 (Phase 3)")
def test_persistent_ring_launches_once_per_traversal():
    from lis.kernels.persistent_ring import launch_count_for_traversal

    world = torch.cuda.device_count()
    assert launch_count_for_traversal(world) == 1, (
        "the persistent kernel is relaunched per step, which is exactly what "
        "raw CUDA was chosen to avoid"
    )


# ---------------------------------------------------------------- NVSHMEM
# Uniquely best at: device-initiated, fine-grained, latency-bound transfers.
# Scoped to decode at P=8 because the upstream report shows communication is
# 1.1% of prefill runtime and would stay hidden even after a 15x speedup.
# Failure mode: using it for prefill, where it cannot win.


@gpu_only
@pytest.mark.skipif("NVSHMEM_TEST" not in os.environ, reason="set NVSHMEM_TEST=1 (Phase 3)")
def test_nvshmem_is_only_used_in_the_regime_where_it_can_win():
    """Guards the scoping decision itself.

    If NVSHMEM appears on the prefill path, someone has reached for it because
    it is impressive rather than because it helps.
    """
    from lis.kernels.nvshmem_path import enabled_regimes

    regimes = enabled_regimes()
    assert "prefill" not in regimes, (
        "NVSHMEM enabled for prefill, where measurement shows communication is "
        "1.1% of runtime and already hidden — this is decoration"
    )
    assert "decode" in regimes


# ---------------------------------------------------------------- CUTLASS
# Uniquely best at: peak GEMM through warp specialization. Its role here is a
# comparison against Triton, so "Triton wins" is a valid outcome — but the
# comparison must have actually been run.


@gpu_only
@pytest.mark.skipif("CUTLASS_TEST" not in os.environ, reason="set CUTLASS_TEST=1 (Phase 3)")
def test_cutlass_comparison_was_measured_not_asserted():
    from lis.kernels.cutlass_gemm import comparison_report

    report = comparison_report()
    for key in ("triton_tflops", "cutlass_tflops", "tensor_core_utilization"):
        assert key in report, f"comparison is missing {key}; it was not measured"
    assert report["tensor_core_utilization"] > 0.5, (
        "tensor cores barely used — the CUTLASS path is not configured for them"
    )


# ---------------------------------------------------------------- meta


def test_contract_reports_what_it_skipped():
    """The suite must never pass vacuously.

    On a laptop most checks skip, and that is correct — but the run has to make
    that visible so a green CPU run is not mistaken for a validated stack.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-q", "--co"],
        capture_output=True, text=True,
    )
    assert "test_triton_kernel_meets_the_mfu_gate" in out.stdout, (
        "GPU-gated checks are not being collected; they would vanish silently "
        "rather than skip visibly"
    )
