# Phase 1 — the kernel is decided

**Date:** 2026-08-20 · **Hardware:** 1 × A100-SXM4-40GB, sm_80 · **Cost:** ~$0.30
**Evidence:** `results/triton_validation.json`, `results/kernel.jsonl`,
`results/environment.txt`, `results/gpu_tests.log`, `results/contract.log`

## Decision

**The Triton kernel is validated and clears the gate. `torch_flash` remains the
default fused path.**

Those are two separate findings and it matters that they are kept apart. The
gate asks *is this kernel real?* — the comparison asks *should it be the
default?* Clearing the first does not settle the second, and the validation
script originally conflated them: it recommended flipping `PREFERENCE` on the
gate alone, which would have demoted a faster kernel in favour of a slower one.
That logic is corrected.

| | MFU @ S=16384 | vs gate (20%) | vs `unfused` |
|---|---|---|---|
| `torch_flash` | **58.8%** | clears | — |
| `triton` | **38.6%** | clears | **24.0×** @ S=8192 |
| `unfused` | 1.5% | fails | — |

`PREFERENCE` in `lis/kernels/__init__.py` is unchanged: `("torch_flash",
"triton", "unfused")`. It was written to outrank Triton *until measured*; the
measurement agrees with it, so nothing moves.

## Full sweep

A100-SXM4-40GB, bf16, causal, 32 heads, head_dim 128, batch 1. Peak 312 TFLOP/s
(dense bf16 — never the sparsity-doubled figure, since attention is dense).

| backend | seq | ms | TFLOP/s | MFU | FLOP/byte |
|---|---|---|---|---|---|
| torch_flash | 2048 | 0.38 | 95.6 | 30.6% | 544 |
| torch_flash | 4096 | 1.12 | 126.9 | 40.7% | 1056 |
| torch_flash | 8192 | 3.83 | 145.9 | 46.8% | 2080 |
| torch_flash | 16384 | 13.86 | 159.9 | 51.3% | 4128 |
| torch_flash | 32768 | 46.83 | **188.6** | **60.4%** | 8224 |
| triton | 2048 | 0.50 | 73.5 | 23.6% | 544 |
| triton | 4096 | 1.43 | 99.2 | 31.8% | 1056 |
| triton | 8192 | 4.91 | 113.6 | 36.4% | 2080 |
| triton | 16384 | 18.41 | 120.4 | 38.6% | 4128 |
| triton | 32768 | 72.24 | 122.2 | **39.2%** | 8224 |
| unfused | 2048 | 7.64 | 4.8 | 1.5% | 21.8 |
| unfused | 4096 | 29.88 | 4.7 | 1.5% | 21.6 |
| unfused | 8192 | 117.71 | 4.7 | 1.5% | 21.4 |
| unfused | 16384 | — | — | OOM | — |
| unfused | 32768 | — | — | OOM | — |

### Reading it

**Arithmetic intensity is the explanation, not the MFU number.** The unfused
path sits at ~21 FLOP/byte and is pinned there across every sequence length —
it is bandwidth-bound, and no amount of tuning moves a memory-bound kernel. Both
fused paths scale intensity linearly with sequence length (544 → 8224) because
the score tile never leaves SRAM. That is the entire mechanism, and it is why
1.5% and 60% are the same algorithm.

**The unfused OOM is a result, not a failure.** Its `[H, S, S]` fp32 score tile
is 32 GiB at S=16384 before intermediates. It falls out of the sweep on any
card, including an 80 GB one. That boundary *is* the memory argument.

**Triton is compared only where both completed.** Comparing each backend's best
row would pit `triton@16384` against `unfused@4096` and report the shape
difference as a speedup.

## Why torch_flash wins

Not a defect in the Triton kernel — it is a competent FA-2 forward that clears
the gate on first compile. `torch_flash` dispatches to a kernel with
hand-tuned pipelining and warp specialization per architecture. Reaching 66% of
it with 228 lines of Triton is close to the expected outcome, and it is the
comparison that makes the Triton work meaningful: *a hand-written kernel that
only beats an unfused Python loop has proven nothing.*

The autotuner is genuinely searching — selected candidates #1, #2, #6, #7 across
shapes, never uniformly #0.

## What the session found beyond the kernel

Three bugs, all invisible on a laptop, all in measurement rather than in the
code being measured. This is the pattern worth noting: **the code under test was
fine; the instruments were wrong.**

**1. The CUDA-graph launch counter measured the wrong quantity.**
`test_cuda_graphs_reduce_decode_launches` failed at `43 -> 43`, 0% reduction. It
counted *device kernel executions*, which graph replay does not reduce — capture
removes host-side submissions, not work. Corrected to count `cudaLaunchKernel` /
`cudaGraphLaunch`: **43 → 1, a 98% reduction.** The capture had been working the
whole time and the metric was reporting it as broken.

**2. The kernel tests built CPU fp32 tensors.** Sixteen failures, all
`Pointer argument cannot be accessed from Triton (cpu tensor?)` or an aten
dispatch error. `torch_flash` accepts only fp16/bf16 on CUDA and Triton needs
device pointers. On a laptop only `unfused` exists and accepts anything, so
these tests had passed for every backend that was never running. Device and
dtype now derive from the backend under test.

**3. TF32.** With fp32 kernel tests running on CUDA for the first time, two
failed on precision alone. Some CUDA builds enable TF32 for fp32 matmul by
default, dropping the mantissa from 24 bits to ~10 — which reads as a
correctness bug rather than a precision setting. Now disabled suite-wide in
`tests/conftest.py`. This is the second time this exact trap has cost time; it
is documented in both repos now.

Final GPU state: **188 passed, 16 skipped, 0 failed**, contract suite 8 passed /
8 skipped with reasons printed.

## Consequences for Phase 2

- Serving numbers can be measured now. They will run on `torch_flash` at
  ~60% MFU rather than the upstream 3%, so they are worth publishing.
- The CP ring, paging and RoPE layers need no change — the dispatcher contract
  held across all three backends, which is what the substitutability tests
  existed to prove.
- Triton stays in the repo as a measured comparison point, not decoration. It is
  also the natural base for the Phase 3 persistent ring kernel, where the
  requirement is device-side control over the ring loop rather than raw GEMM
  throughput — a thing `torch_flash` cannot express at all.

## Not done

- **HBM bandwidth is assumed, not measured.** K2's roofline uses the 1.55 TB/s
  A100 spec figure; the Vast listing advertised 1314 GB/s. This shifts the ridge
  point and nothing else — no gate or decision depends on it — but the figure
  annotates an assumption where it could carry a measurement.
- **No nsys trace of the Triton kernel.** Not needed for a single-GPU decision;
  it becomes relevant in Phase 3 where launch counts per ring traversal are the
  gate.
