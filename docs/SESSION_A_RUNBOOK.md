# Session A — the prefix-share crossover

One A100 40GB, ~2 hours, ~$1.35. Answers one question: **where does prefix
caching start paying for itself, and does SGLang's RadixAttention overtake
vLLM's?**

Nothing here is blocked on anyone. The gated half of Phase 2 — zigzag vs
contiguous CP — needs the SGLang CUDA port and is not part of this session.

## What to rent

| | |
|---|---|
| GPU | 1 × A100 40GB (SXM or PCIe — no interconnect traffic here) |
| **Disk** | **64 GB, set before renting** — cannot be resized after |
| Image | PyTorch NGC |
| Cost | ~$0.70–0.75/hr |

## The plan

| Step | Time | Notes |
|---|---|---|
| sync + setup | 3 min | `deploy/remote.sh <PORT> <HOST> shell` |
| install vLLM + SGLang | 30 min | the dominant cost; see below |
| download Qwen2.5-1.5B | 5 min | ~3 GB, ungated — **no HF token needed** |
| smoke: one run per engine | 5 min | catches config errors before the sweep |
| the sweep, 63 runs | 24 min | 3 frameworks × 21 prefix shares |
| pull results, destroy | 2 min | |

The 1.8× allowance on top is for first contact: neither binding has ever talked
to a live engine.

## Configuration

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` — Apache 2.0, ungated, geometry verified |
| Frameworks | `sglang`, `sglang-no-cache`, `vllm` |
| Prefix share | 0 → 1 in steps of 0.05, **21 points** |
| Requests | 64 per run, Poisson arrivals at 4 QPS |
| Prompt / output | 8192 / 128 tokens, greedy, `ignore_eos` |
| SLO | TTFT p99 ≤ 2000 ms, TPOT p99 ≤ 50 ms |

`sglang-no-cache` is the control. Without it, a throughput gap against vLLM
could be RadixAttention or could be any of a dozen other implementation
differences between two engines.

## Run it

```bash
SSH_KEY=~/.ssh/id_ed25519_github deploy/remote.sh <PORT> <HOST> shell
```

Then on the box:

```bash
pip install vllm sglang[all]
python bench/serving.py --sweep-prefix-share --frameworks sglang sglang-no-cache vllm
```

## What will probably break first

Neither binding has run against a live engine. Expected, not defects:

- **`sample_block_table` / `request_intervals`** are `NotImplementedError` by
  design — they reach into version-specific engine internals. Only the
  `VLLM_TEST=1` contract checks call them; the sweep does not.
- **`engine.async_generate` signature drift** in SGLang. Its streaming API is
  less documented than vLLM's, and `lis/serving/sglang_backend.py` was written
  from the documented shape rather than from installed source.
- **`AsyncEngineArgs(decode_context_parallel_size=...)`** may not exist in the
  installed vLLM version. Only `vllm-dcp` passes it, which Session A does not
  run — but it is constructed in the same dataclass.
- **Prefix-cache telemetry field names.** `cached_tokens_from_usage` accepts
  both `num_cached_tokens` and `cached_tokens`; if an engine reports neither,
  the hit rate reads 0 and S4's lower panel is flat. That is a reporting bug,
  not a caching failure — check it before concluding anything.

## Before destroying the instance

```bash
ls results/serving.jsonl && wc -l results/serving.jsonl   # expect 63
```

Then locally:

```bash
make report
```

S1–S5 draw from the real numbers, and the dashboard picks up the serving view.

## What counts as success

Not "SGLang wins". The deliverable is a **located crossover** — a prefix share
above which caching pays, measured rather than assumed. If the curves never
cross, that is equally publishable and the figure says so rather than inventing
a point.

The one outcome that would mean the session failed is a flat cache-hit curve on
a workload built to share prefixes, because that means the benchmark could not
distinguish the two engines at all.
