# Foundations, without the courses

A build-first path to the basics this project assumes. Five units, roughly
**fourteen evenings and $10 of GPU time**. Two units are free.

The premise: you already have the lab. A course's value is mostly its
assignments, and this repo contains better ones than most courses set — an
unanalyzed `nsys` trace, a kernel that has never been compiled, and a
communication model with no measurement behind it yet. Every exercise below
produces something the project needs, so studying and unblocking Phase 1 are the
same activity.

**Rule for all five units: predict the number before you measure it.** Write the
prediction down. The gap between your prediction and the measurement is the
entire learning signal, and it disappears if you measure first.

---

## What to skip

Time saved is the point, so be explicit about what is *not* on the path:

- **Training.** Backward passes, optimizers, gradient accumulation, ZeRO. This
  project is inference-shaped. Not needed.
- **Most of PMPP.** Chapters 1–6 only. Chapters 7+ are application case studies.
- **CUTLASS / CuTe.** Phase 3, comparison only. Skip until Triton is decided.
- **Compilers, OS, networking theory.** Adjacent, not load-bearing.
- **CUDA C++ beyond the basics.** You will write three small kernels to build
  intuition, then work in Triton. Production kernels here are Triton, not CUDA.

---

## Unit 0 — the mental model
**2 evenings · free · no code**

The one non-negotiable unit. Everything else is mechanics.

1. Read Horace He, **Making Deep Learning Go Brrrr From First Principles**.
   Twice. It is short.
2. On paper, for our attention kernel at `S=8192, H=32, D=128`:
   - FLOPs: `elements × H × 4 × D`  (`lis.bench_kernel.elements_computed`)
   - HBM bytes, if the score tile is materialized in fp32
   - Arithmetic intensity = FLOPs ÷ bytes
   - A100 ridge point = 312 TFLOP/s ÷ 1.55 TB/s ≈ **200 FLOP/byte**
3. Compare. Left of the ridge is memory-bound.

**Checkpoint:** given a shape, say whether it is memory- or compute-bound
*before* running anything — and say what you would change to move it right.

You now understand why this project reports 3% MFU, which is the fact everything
else in it hangs off.

---

## Unit 1 — profiling
**2 evenings · free · uses data already on disk**

`results/ring_striped.nsys-rep` was captured on the A100 session and never
opened. It is 8.4 MB of answers.

```bash
nsys stats results/ring_striped.nsys-rep --report cuda_gpu_kern_sum
```

1. Install Nsight Systems locally (the GUI reads the trace; no GPU needed).
2. Open the timeline. Find the NCCL kernels and the GEMMs.
3. Answer, from the trace:
   - Do NCCL kernels overlap GEMMs, or sit in gaps? (**this is figure K4**)
   - How many kernel launches per ring traversal?
   - What fraction of wall-clock is any kernel resident at all?
4. Learn `ncu` on one kernel — just `--set basic`. Read *Memory Throughput* and
   *Compute Throughput*. One of them is near 100%; that one is your bound.

**Checkpoint:** state the measured overlap fraction, and whether it matches the
54% the report claims from timing alone.

**Deliverable:** figure K4, and the launch count that Phase 3's persistent
kernel must beat.

---

## Unit 2 — the GPU execution model
**4 evenings · ~$4 · 1× A100**

This is ECE408's core, compressed to the part that transfers. Skim PMPP ch. 1–6
as reference, not cover-to-cover — write code first, read when stuck.

Write three kernels in CUDA C, in this order. Do not skip to the third.

| # | Kernel | Teaches | Target |
|---|---|---|---|
| 1 | vector add | grid/block/thread indexing, launch config | should hit ~HBM bandwidth; verify it does |
| 2 | naive matmul | why global-memory access patterns dominate | will be badly slow. Measure how slow |
| 3 | **tiled matmul, shared memory** | the whole idea | should beat #2 by ~10× |

Kernel 3 is the point of the unit. Loading a tile into shared memory once and
reusing it across the block is *exactly* what FlashAttention does with the score
tile — and exactly what our unfused loop fails to do. Once you have written it,
FlashAttention stops being a paper and becomes an obvious idea.

Then: compute kernel 3's arithmetic intensity, plot it on the roofline against
kernels 1 and 2, and explain the ordering.

**Checkpoint:** explain why tiling is faster **in bytes moved**, not in
hand-waving. If your explanation does not contain a number, it is not an
explanation.

---

## Unit 3 — Triton, and the kernel that gates the project
**4 evenings · ~$4 · 1× A100**

1. Work the official Triton tutorials `01` → `06`, in order, on the GPU.
   `03-matrix-multiplication` should feel familiar after Unit 2.
   `06-fused-attention` is the closest published relative of our kernel.
2. Diff `06-fused-attention` against `src/lis/kernels/triton_flash.py`. Ours has
   never compiled — expect real bugs, not typos.
3. Compile it. Fix it. Validate against `backend="unfused"`, never against
   `torch_flash` — the oracle chain has to stay rooted outside the thing under
   test.
4. Run the sweep:

```bash
make bench-kernel && make report
```

**Checkpoint:** the kernel either clears the 20% MFU gate or it does not. Both
are results. Not knowing is the only failure.

**Deliverable:** Phase 1 exit criteria, met. This is the single blocker on
everything downstream.

---

## Unit 4 — collectives
**2 evenings · ~$2 · 2–4× A100**

1. Build and run `nccl-tests` (`all_reduce_perf`, `sendrecv_perf`). Read its
   README for the `busbw = algbw × 2(P−1)/P` convention — it is defined there,
   not in the NCCL docs.
2. Run ours and compare:

```bash
python -c "from lis.bench_comm import measure; ..."   # see the module docstring
```

3. Do the A/B that makes it concrete:

```bash
NCCL_P2P_DISABLE=1 make smoke-gpu NPROC=4
```

Bandwidth should collapse. That is what a PCIe box looks like while reporting
itself healthy — the failure `test_nccl_achieves_a_reasonable_fraction_of_nameplate`
exists to catch.

4. Derive on paper why the ring moves the same total bytes as all-gather but
   needs `P/2` less memory.

**Checkpoint:** explain `busbw` vs `algbw` without looking, and say why we quote
the measured 156.5 GB/s rather than NVLink's 300 GB/s spec.

---

## Sequencing and cost

| Unit | Evenings | GPU | Cost | Unblocks |
|---|---|---|---|---|
| 0 · mental model | 2 | none | $0 | everything |
| 1 · profiling | 2 | none | $0 | figure K4 |
| 2 · execution model | 4 | 1× | ~$4 | Unit 3 |
| 3 · Triton | 4 | 1× | ~$4 | **Phase 1 exit** |
| 4 · collectives | 2 | 2–4× | ~$2 | contract suite |

**~14 evenings, ~$10.** Units 0 and 1 are free and worth doing this week
regardless of when GPU time happens.

Rent one GPU for Units 2 and 3 back to back in a single session — the setup cost
is per-session, not per-hour, and four GPUs cannot make one kernel converge
faster.

---

## How you will know it worked

Not by finishing the list. By being able to answer these cold:

1. Given a kernel and a shape, is it memory- or compute-bound, and what is the
   evidence?
2. Why does tiling reduce HBM traffic? With a number.
3. Why is the causal ring's cost `Σ_steps max_r work(r,s)` rather than
   `Σ_r work(r)`?
4. Why does striping fix it, and where does the `2 − 1/P` ceiling come from?
5. What does an overlapped NCCL exchange look like in a profiler, and what does
   a failed one look like?

Those five are also, near enough, an inference-systems interview.
