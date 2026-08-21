# Resume snippets

Every number here traces to a committed results file. Claims are written so they
survive an interviewer opening the repo.

**One deliberate restraint:** the 60.4% MFU figure belongs to PyTorch's kernel,
not to a kernel written here. The hand-written Triton kernel measured **39.2%**.
Nothing below claims otherwise — selecting the faster path is presented as the
engineering judgement it was, which is a stronger story than an inflated number
that collapses under one follow-up question.

---

## Three bullets — highest density

> **GPU Inference Systems — Context-Parallel Attention & Kernel Optimization**
>
> - Diagnosed causal load imbalance as a critical-path rather than total-work
>   problem and implemented zigzag sequence sharding, **elevating ring-attention
>   throughput 73%** to 99.1% of the theoretical `2−1/P` bound and enabling
>   **1M-token prefill** on 4× A100 where the all-gather baseline exhausted HBM.
> - Authored an autotuned **Triton** FlashAttention kernel returning log-sum-exp
>   for cross-device merging, **raising Model FLOPs Utilization 24×** over an
>   unfused baseline; roofline analysis attributed the gap to a **97% reduction
>   in HBM traffic**, and CUDA-graph capture **cut decode kernel launches 98%**.
> - Built a **vLLM/SGLang** serving benchmark isolating RadixAttention with a
>   cache-disabled control, measuring **6.2× throughput** from prefix caching and
>   locating the SGLang→vLLM crossover at 0.92 shared prefix; **goodput exposed
>   0% SLO attainment at throughputs that appeared healthy**.

---

## Six bullets — fuller project entry

> **Distributed LLM Inference Stack** — PyTorch, CUDA, Triton, NCCL, vLLM, SGLang
>
> - Engineered ring and all-gather context-parallel attention with double-buffered
>   NCCL P2P overlap, validated bit-exact against a single-process reference at
>   world sizes 2/4/8 and **reducing communication to 1.1% of prefill runtime**.
> - Identified that causal masking idles ~50% of devices under contiguous
>   sharding, and delivered a zigzag layout **improving throughput 73%** —
>   99.1% of the analytically derived ceiling.
> - Scaled attention to **1,048,576 tokens** at 32.3 GiB/device by making KV
>   residency independent of world size, **lowering peak memory 50%** against
>   all-gather at P=4.
> - Implemented a three-backend kernel dispatcher behind one contract, enabling
>   the full serving stack to be developed and CI-tested on CPU and executed
>   unchanged on GPU; **validated a Triton kernel at 24× the unfused baseline**
>   and adopted the measured-faster vendor path on evidence.
> - Instrumented MFU, roofline, goodput and SLO attainment as first-class
>   metrics with thresholds fixed before measurement, and **automated 240 tests**
>   including an anti-decoration suite that fails if a technology is present in
>   name only.
> - Benchmarked prefix-cache sensitivity across 63 GPU runs, **isolating a 6.2×
>   RadixAttention gain** via an ablation control and locating the cross-engine
>   crossover, **for under $2 of rented compute**.

---

## Two bullets — space-constrained

> - Delivered zigzag context-parallel attention **elevating throughput 73%** to
>   99.1% of the theoretical bound and enabling **1M-token prefill** on 4× A100;
>   authored an autotuned Triton FlashAttention kernel **raising MFU 24×**.
> - Built a vLLM/SGLang benchmark harness quantifying a **6.2× prefix-caching
>   gain** against an ablation control, surfacing **0% SLO attainment masked by
>   healthy-looking throughput**.

---

## Skills line

> **GPU & Systems:** CUDA, Triton, NCCL, CUDA Graphs, NVSHMEM, CUTLASS, Nsight
> Systems, roofline & MFU analysis, tensor cores, mixed precision (bf16/fp32)
> **Inference & Serving:** vLLM, SGLang, PagedAttention, RadixAttention,
> FlashAttention, context parallelism, KV-cache management, continuous batching,
> goodput/SLO benchmarking, RoPE
> **Distributed:** ring attention, sequence/context parallelism, collective
> communication, overlap scheduling, load balancing, critical-path analysis
> **Engineering:** PyTorch, torch.compile, Python, pytest, mutation testing,
> Docker, reproducible benchmarking

---

## Provenance

| Claim | Source |
|---|---|
| 73% throughput gain, 99.1% of bound | `results/layout_gpu.jsonl`, upstream REPORT.md §4.3 |
| 1,048,576 tokens @ 32.3 GiB/device | `results/sweep.jsonl` |
| Communication 1.1% of runtime | upstream REPORT.md §6 |
| Triton 39.2% MFU, 24× unfused | `results/kernel.jsonl`, `docs/PHASE1.md` |
| 97% HBM traffic reduction | arithmetic intensity 21 → 8224 FLOP/byte, `results/kernel.jsonl` |
| CUDA graphs 98% launch reduction | `lis.jit.decode_launch_counts`, 43 → 1 |
| 6.2× RadixAttention, crossover 0.915 | `results/serving_saturated.jsonl`, `docs/PHASE2.md` |
| 0% SLO at 5.81 req/s throughput | `results/serving_saturated.jsonl`, share 0.50 |
| 240 tests | `make test` |

## Interview follow-ups these invite

Worth rehearsing, because each is the obvious next question:

- *"Why is your Triton kernel slower than PyTorch's?"* — It reaches 66% of a
  vendor kernel with hand-tuned pipelining and per-architecture warp
  specialization, in 228 lines. The comparison is the deliverable; a kernel that
  only beats a Python loop has proven nothing.
- *"Does the 73% speedup survive a fast kernel?"* — Unknown, and it probably
  compresses. Communication was 1.1% of runtime because the kernel was slow; at
  20× faster compute it becomes roughly 18%, so overlap quality starts to
  dominate. That re-measurement is the next experiment.
- *"Why is goodput zero when throughput looks fine?"* — Requests completed, but
  TTFT p99 was 9.8 s against a 2 s target. Throughput counts served requests;
  goodput counts acceptably served ones.
- *"How do you know the 6.2× is caching?"* — Same engine, cache disabled, flat
  at 4.62 req/s across all 21 prefix shares with 5.5% spread.
