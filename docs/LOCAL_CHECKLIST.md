# Local checklist — what to verify before renting anything

The upstream project established that distributed *correctness* needs no GPU;
only performance does. That saved the entire correctness argument from being
debugged at $3/hour. This checklist extends the same discipline.

Rule: **if a question can be answered on the laptop, it must be, before a box is
rented.** Renting to discover a shape bug is the most expensive way to find one.

## Phase 1 — before the single-GPU kernel session

Kernel authoring needs a GPU. Almost nothing else does.

- [x] **RoPE global-position invariant.** `rope-then-shard == shard-then-rope`
      for both layouts at P ∈ {2,4,8}. The bug it prevents is silent: wrong
      relative distances, fluent output, no crash. `tests/test_rope.py`
- [x] **The positional test discriminates.** The deliberately-wrong
      implementation must *fail* the invariant. A test that passes for both the
      correct and broken version is not a test.
- [x] **LSE merge is exact.** Splitting keys anywhere and merging equals a
      single pass. Order-invariant and associative, so blocks may arrive in ring
      order. `tests/test_lse.py`
- [x] **Empty-partial edge cases.** A fully-masked block merges as a no-op;
      two empty partials do not produce NaN. Same class as the upstream `-inf`
      guard.
- [ ] **Tiling spec.** A CPU reference performing exactly the tiling the Triton
      kernel will perform, so the kernel has a precise thing to be wrong against.
- [ ] **MFU harness.** Wired to `dattn.analysis.computed_elements` and
      `attention_flops`, with the gate value chosen before any measurement so it
      cannot be retrofitted to whatever the kernel happens to achieve.
- [ ] **Contract suite skips cleanly.** `make contract` on CPU must report which
      checks are GPU-gated rather than silently passing an empty suite.

## Rent-day rules

- **One GPU for kernel work.** Roughly $1/hr against $3–4/hr for four. Four GPUs
  cannot make a single kernel converge faster.
- **Four GPUs only for distributed runs**, and only once single-GPU MFU has met
  the gate.
- **Eight GPUs only for NVSHMEM**, and only after the microbenchmark decision
  gate in Phase 3.
- Run `make plan PROFILE=<hw>` before every session — predicted limits are free.
- Write the session runbook *before* starting the box.
- Copy results off before destroying it.

## What cannot be checked locally, and why

Recorded so nobody wastes an afternoon trying.

| Question | Why it needs hardware |
|---|---|
| Does the Triton kernel compile and run? | Triton requires an NVIDIA target; the CPU backend is experimental and not representative |
| Achieved TFLOP/s and MFU | Needs tensor cores |
| Autotuner's chosen configuration | Selection is measured on the target device |
| NCCL bandwidth and topology | Needs real interconnect |
| Whether `torch.compile` reduces kernel count | Needs a profiler on a real device |
| vLLM/SGLang paged-KV behaviour under load | CPU backends exist but do not exercise the memory manager realistically |

## Prediction log

The upstream project registered predictions before measuring, which turned two
results into confirmations rather than observations. Continue that. Record here,
before the session, with a date.

| Date | Prediction | Outcome |
|---|---|---|
| 2026-08-04 | A Triton fused kernel reaches ≥ 20% MFU on A100, from 2.5–3.4% today | pending |
| 2026-08-04 | The 1.73× striped speedup survives fusion — it is a scheduling result, not a kernel artifact | pending |
| 2026-08-04 | Communication rises from 1.1% toward ~8% of runtime once compute is ~10× faster | pending |
