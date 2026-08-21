# Phase 2 — the prefix-share crossover

**Date:** 2026-08-21 · **Hardware:** 1 × A100-SXM4-40GB · **Cost:** ~$1.50
**Model:** Qwen/Qwen2.5-1.5B (Apache 2.0, geometry verified)
**Evidence:** `results/serving_saturated.jsonl` (63 runs),
`results/serving.jsonl` (21 paced runs), `results/environment_phase2.txt`

## Headline

**SGLang leads vLLM across almost the entire useful range; vLLM overtakes only
at `prefix_share = 0.915`, where prompts are essentially identical.**

| | sglang | vllm |
|---|---|---|
| Throughput @ share 0.0 | **4.44** | 3.58 |
| Throughput @ share 0.75 | **12.10** | 10.00 |
| Throughput @ share 0.90 | **18.87** | 18.11 |
| Throughput @ share 0.95 | 23.12 | **24.86** |
| Throughput @ share 1.0 | 28.77 | **33.32** |
| First share with any goodput | **0.50** | 0.65 |
| First share at 100% SLO | **0.90** | 0.95 |

The crossover is at **0.915**, interpolated from the two points that bracket it.
That is a narrow and fairly artificial corner: at 0.95 shared prefix, every
request is nearly the same request. Through the range a real workload occupies,
SGLang is ahead on throughput, on goodput, and on how early it starts meeting
the SLO at all.

## The control is what makes this a measurement

`sglang-no-cache` — the same engine with `disable_radix_cache=True` — is **flat
at 4.62 req/s across all 21 prefix shares**, varying by 5.5%.

That single line is the most important one in the study. Without it, a rising
curve is consistent with caching, with scheduler luck, with memory pressure
easing, or with a dozen other differences between two engines. With it, the
entire effect is attributable: **6.22× from RadixAttention alone**, measured
against the same engine with one feature switched off.

## Goodput was the right headline

At `prefix_share = 0.50`, vLLM posts a throughput of 5.81 req/s. That looks like
a working system. Its goodput is **0.00** — every request missed the 2000 ms
TTFT target, with p99 at 9.8 seconds.

| share | vllm throughput | vllm TTFT p99 | vllm SLO | **vllm goodput** |
|---|---|---|---|---|
| 0.00 | 3.58 | 16.7 s | 0% | **0.00** |
| 0.50 | 5.81 | 9.8 s | 0% | **0.00** |
| 0.75 | 10.00 | 5.0 s | 44% | **4.38** |
| 1.00 | 33.32 | 0.7 s | 100% | **33.32** |

A throughput-only report of this sweep would describe a system steadily
improving from 3.6 to 33 req/s. The goodput column says something different and
truer: below 0.65 shared prefix, **this configuration serves nobody
acceptably**.

## A methodology correction, made mid-session

The first sweep used Poisson arrivals at 4 QPS. Every configuration met the SLO
100% of the time and throughput sat near 4 req/s regardless of caching — because
**arrivals, not the engine, were the bottleneck**. Throughput in that regime
measures the load generator.

The sweep was re-run saturated (`--qps 1000`, all requests offered at once), so
the engine is the constraint and throughput measures capacity. Both datasets are
kept: `serving.jsonl` is the paced run, `serving_saturated.jsonl` is the one the
figures use. Every row records its own `qps`, so the two regimes cannot be
confused later.

The paced data is not wasted, but it is not comparative either — only vLLM was
measured before the correction.

## Bugs found on first contact

Both were in the bindings, both invisible without a live engine, and both would
have **corrupted the study rather than crashing it**.

**Prefix-cache telemetry read from the wrong place.** vLLM 0.27 exposes
`num_cached_tokens` as a direct attribute of `RequestOutput`; `metrics` carries
only timing. The binding read `metrics` and reported a 0% hit rate — which is
indistinguishable in the output from a cache that is switched off. A probe
showed 592 of a 600-token shared prefix hitting while the benchmark reported
zero. The runbook had named this exact failure in advance, which is the only
reason it was checked rather than believed.

**A new event loop per run.** `asyncio.run()` closes its loop on return, so the
engine cached across sweep points lost the background handler bound to the first
loop, and point two died with `EngineDeadError`. Fixed in the SGLang binding
pre-emptively rather than paying for a second session to find it twice.

## What is not here

- **S1 (TTFT vs context)** — not drawn. Every run used an 8192-token context, so
  there is no curve. The figure refuses to plot a single x value rather than
  drawing a vertical smear that reads as evidence.
- **S5 (peak memory)** — not drawn. Neither binding implements
  `peak_memory_bytes`, so no row carries it and the figure skips.
- **`vllm-dcp`** — needs ≥2 GPUs for decode context parallelism. Session B.
- **Zigzag vs contiguous CP** — still blocked on the SGLang CUDA port.
- **Concurrency sweep** — held fixed at 64 requests throughout.

## What this says about the CP work

The crossover result is about prefix caching, not context parallelism, and it
does not by itself argue for the zigzag ring. What it does establish is that the
harness measures what it claims to: a control that stays flat, an effect that
tracks its independent variable, and a headline metric that disagrees with
throughput exactly where it should.

That is the instrument Session B needs.
