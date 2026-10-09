# llm-inference-stack

Long-context LLM inference on multiple GPUs: a fused attention kernel, context
parallelism, and a serving study that measures **where SGLang stops beating
vLLM** instead of asserting it.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![tests](https://img.shields.io/badge/tests-178-brightgreen)
![measured on](https://img.shields.io/badge/measured%20on-A100--SXM4--40GB-76b900)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

<!-- Add once causal-attention-load-imbalance is public, so CI can install dattn:
[![ci](https://github.com/ShrutiBora001/llm-inference-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/ShrutiBora001/llm-inference-stack/actions/workflows/ci.yml)
And once Pages is enabled (Settings -> Pages -> Source: GitHub Actions):
**[Live results dashboard](https://shrutibora001.github.io/llm-inference-stack/)**
-->

## Measured results

Every number below is reproduced by `make report` from the committed files in
[`results/`](results/) — nothing is typed into a figure or a table. CI rebuilds
them on every push, so a hand-edited number fails the build.

| | Result | Evidence |
|---|---|---|
| **Fused Triton attention kernel** | **38.6% MFU** on A100, against a 20% gate set before measuring | [`triton_validation.json`](results/triton_validation.json) |
| **vs the unfused baseline** | **24× faster** at matched shape (1.5% → 36.4% MFU) | [`kernel.jsonl`](results/kernel.jsonl) |
| **Kernel actually shipped** | `torch_flash`, at **58.8% MFU** — Triton reached 66% of it, so it did *not* ship | [`PHASE1.md`](docs/PHASE1.md) |
| **Prefix-cache crossover** | **0.915** — SGLang leads below it, vLLM above | [`serving_saturated.jsonl`](results/serving_saturated.jsonl) |
| **Peak output throughput** | 3,683 tok/s SGLang · 4,265 tok/s vLLM (Qwen2.5-1.5B, 8K context, 64 concurrent) | same |
| **Upstream striping result** | 1.73× at S=131072, 99.1% of the `2−1/P` ceiling | [upstream repo](https://github.com/ShrutiBora001/causal-attention-load-imbalance) |

### The headline finding

![prefix-share crossover](report/figures/S4_crossover.png)

Prefix-cache hit rate is usually reported as a single number from a single
workload. Sweeping it as the independent variable — 21 points, three
configurations, 63 runs — turns it into a curve with a **crossover at
prefix_share ≈ 0.915**.

Below that, SGLang's RadixAttention wins: at zero sharing it is **24% faster**
than vLLM (568 vs 458 tok/s). Above it, vLLM's scheduler wins, and at full
sharing it is **16% faster** (4,265 vs 3,683 tok/s). Picking a framework from
one benchmark point picks the wrong one roughly half the time.

### Why the Triton kernel did not ship

![kernel MFU](report/figures/K1_mfu.png)

It cleared its gate — 38.6% MFU, 24× the unfused loop, correct to 8.5e-3 against
the oracle, autotuner demonstrably exploring the space. It still lost to
`torch_flash` at 58.8%, so `torch_flash` stays the default and the Triton kernel
is reported as a **measured null result** rather than quietly adopted.

That is the outcome the project was built to be able to reach. The gate was
written down first, so "my kernel is slower than the library" was a possible
answer, not an unthinkable one.

## Quick start

No GPU required. The entire correctness argument runs on a laptop.

```bash
make venv && make test && make contract
```

```bash
make report      # rebuild every figure + the dashboard from results/
```

`make contract` is the interesting one — see below.

## The anti-decoration contract

The organising constraint: **no technology is here for its name.** Each must do
something the others cannot do as well, and each has a test that fails if it has
become decorative.

| Technology | Uniquely best at | Test that fails if decorative |
|---|---|---|
| **Triton** | Fast kernel authoring, autotuning | MFU ≥ 20% gate; autotuner picks a non-default config; no silent fallback |
| **NCCL** | Mature bulk collectives | Wire bytes match the closed-form model; busbw ≥ 60% of nameplate |
| **torch.compile / CUDA graphs** | Removing launch overhead | Zero graph breaks; decode launches halve under capture |
| **vLLM** | PagedAttention, continuous batching | Block table non-trivial; requests overlap in time |
| **SGLang** | RadixAttention prefix caching | Hit rate > 50% on a prefix-sharing workload |
| **raw CUDA** | Persistent kernels, device-side scheduling | One launch per ring traversal, not `P` |
| **NVSHMEM** | Device-initiated, latency-bound transfers | Must beat NCCL *in decode*; refuses to enable for prefill |
| **CUTLASS** | Peak GEMM, warp specialization | Comparison measured, not asserted; tensor cores actually used |

GPU-gated checks **skip visibly** on a laptop — a green CPU run is never
mistaken for a validated stack.

Two scoping decisions, stated because they are the ones most likely to be got
wrong:

**NVSHMEM is restricted to decode.** Communication is 1.1% of prefill runtime
and stays hidden by overlap even after a hypothetical 15× kernel speedup, so it
cannot win there. `test_nvshmem_is_only_used_in_the_regime_where_it_can_win`
fails if it appears on the prefill path.

**JAX is excluded**, not forgotten — it would mean maintaining two stacks when
vLLM and SGLang are both PyTorch. See [future_scope.md](future_scope.md).

## How it works

**Parallelism is sequence/context parallelism only.** Each rank owns `S/P`
tokens and the full weights; no tensor, pipeline, or data parallelism. KV either
all-gathers once (`O(S)` resident) or rotates around a ring (`O(S/P)` resident,
independent of world size).

**LSE merging is what makes a stock kernel usable.** A ring-aware kernel would
have to accept and return `(acc, m, l)`. Instead every FlashAttention-style
kernel already returns `(out, lse)`, and two partials merge exactly:

```
lse_ab = logaddexp(lse_a, lse_b)
out_ab = out_a·exp(lse_a − lse_ab) + out_b·exp(lse_b − lse_ab)
```

So the ring never enters the kernel. "Write ring attention in Triton" becomes
"call a flash kernel and merge" — and the merge is proven on CPU before any GPU
is rented.

**RoPE under striping is the silent trap.** A device owns chunks `r` and
`2P−1−r` — two disjoint, non-adjacent position ranges — so no single offset is
correct. Getting it wrong does not crash: the model attends at wrong relative
distances and emits fluent nonsense. `tests/test_rope.py` keeps the
deliberately-broken implementation to assert the suite *detects* it.

## Layout

```
src/lis/
  kernels/      dispatcher + unfused oracle, torch_flash, Triton; Phase 3 stubs
  lse.py        LSE merging — the kernel contract
  rope.py       RoPE with global positions; the striping trap
  cp/           framework-agnostic context-parallel ring + KV provider seam
  serving/      paged-KV mapping, vLLM and SGLang bindings
  metrics.py    every metric defined once, with its gate
  workload.py   prefix-share sweep over vLLM's own samplers
  jit.py        torch.compile + CUDA graph capture, decode-scoped
bench/          kernel.py (MFU/roofline) · serving.py (goodput) · comm
report/         make_figures.py · make_dashboard.py — no number hardcoded
results/        the committed evidence; every figure regenerates from it
deploy/         Dockerfile + one-command remote runner
docs/           PHASE1, PHASE2, PIPELINE, FOUNDATIONS, runbooks
```

178 tests. `src/` is ~3k lines.

## Deploy

```bash
deploy/remote.sh <PORT> <HOST> all     # rented SSH box: sync, run, pull results back
docker build -f deploy/Dockerfile -t lis:latest . && \
  docker run --gpus all --rm -v "$PWD/results:/workspace/lis/results" lis:latest all
```

No credentials ever reach the box — code goes up by `rsync` over the key the
provider already has. Every path runs preflight first, because a wrong
interconnect silently invalidates every timing after it. Full runbook, verbs and
cost discipline: [deploy/README.md](deploy/README.md).

## Status

| Phase | State |
|---|---|
| Correctness: RoPE, LSE merge, CP ring, paging | done — 178 tests, laptop-only |
| **Phase 1** — fused kernel validated and chosen | **done**, on A100-SXM4-40GB |
| **Phase 2** — prefix-share crossover, SGLang vs vLLM | **done**, 63 runs |
| Phase 3 — persistent CUDA ring, NVSHMEM (decode), CUTLASS | scaffolded, contract-gated, not built |

Built on [causal-attention-load-imbalance](https://github.com/ShrutiBora001/causal-attention-load-imbalance)
(frozen at v0.3.0), which established the striped-layout result and the 3%-MFU
limitation that this repo set out to remove.
