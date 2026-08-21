# Phase 2 — sizing the GPU session

Run `python scripts/size_session.py` to regenerate any number here. Nothing
below is asserted; it all derives from the benchmark's own defaults, so
changing `--requests` or `--input-len` changes the estimate.

## The headline: this is a 1–2 GPU session, not a 4-GPU one

The ungated study — the prefix-share crossover — needs **no context
parallelism**, so it does not need multiple GPUs for parallelism. It needs
however many GPUs the KV cache requires, and with a 1B model that is one.

| Configuration | KV at peak | Total | GPUs | Cost |
|---|---|---|---|---|
| **1.5B @ 8K × 64 concurrent** | 14.2 GiB | 17.1 GiB | **1** | **$1.09** |
| 1.5B @ 32K × 64 | 56.2 GiB | 59.1 GiB | 2 | $2.18 |
| 8B @ 32K × 64 | 257.0 GiB | 272.0 GiB | 8 | $8.73 |

An earlier estimate in conversation put this at $15–30. That was a guess, and
it was wrong by about an order of magnitude — it assumed 4× A100 because Phase 2
was filed under "the multi-GPU phase", when in fact the multi-GPU requirement
belongs to the *gated* half.

## Where the time goes

At the default sweep — 4 frameworks × 6 prefix shares = 24 runs:

| | |
|---|---|
| install vLLM + SGLang | 30 min |
| model download | 5 min |
| engine startup × 4 | 8 min |
| **measurement** | **9 min** |
| subtotal | 52 min |
| ×1.8 for first-contact fixes | 94 min |

**Measurement is 17% of the session.** Setup dominates completely, and that has
three consequences worth acting on:

1. **One long session beats three short ones.** Every re-rent pays the 35-minute
   install and download again. The measurement itself is nearly free.
2. **Sweep wider than you think you need.** Going from 6 prefix shares to 12
   costs about four minutes. Going back for the extra points costs 40.
3. **Pre-bake the image if this is run more than once.** `deploy/Dockerfile`
   already exists; adding vLLM and SGLang to it converts 30 minutes of install
   into a pull.

## What needs more than one GPU

| Arm | GPUs | Why |
|---|---|---|
| `sglang`, `sglang-no-cache`, `vllm` | 1 | KV fits; no parallelism needed |
| `vllm-dcp` | ≥2 | decode context parallelism is the thing being measured |
| `sglang-zigzag` / `-contiguous` | ≥2 | **blocked** — Ascend-only upstream |

So there are two sensible sessions:

**Session A — the crossover, 1× A100, ~$1.35.** `sglang` vs `sglang-no-cache`
vs `vllm`, sweeping prefix share across 21 points. Produces the headline S4
figure and the cache-hit-rate curve. Nothing here is blocked.

`vllm-dcp` is **not** in Session A: decode context parallelism needs at least
two devices, so it cannot run on one GPU. It belongs to Session B.

Sized concretely: 63 runs, 116 minutes including an 1.8x allowance for
first-contact fixes, of which 24 minutes is measurement.

**Session B — the CP arms, 2–4× A100.** Adds `vllm-dcp`, and the zigzag layout
comparison if the port lands. Worth combining with re-measuring the upstream
1.69×/1.73× layout ratio under a fused kernel, which Phase 1 showed will
compress (comm moves from 1.1% to ~18% of runtime).

Doing A first is strictly better: it is cheap, it is unblocked, and it debugs
the whole harness — workload generation, both bindings, the reduction path,
the figures — on the cheapest possible hardware before B pays 4× per hour for
the same bugs.

## The model: Qwen2.5-1.5B, verified

Settled, and changed from the plan's Llama-3.2-1B for two reasons.

**It is ungated.** `meta-llama/Llama-3.2-1B/config.json` returns **HTTP 401**
without accepting the licence, so the geometry could not be confirmed — and,
more importantly, using it would require putting a Hugging Face token on a
rented box. That is a credential on a machine someone else operates, for no
benefit.

**Its geometry is confirmed from the published config**, so the memory
arithmetic driving this whole estimate is not a guess:

| | |
|---|---|
| layers | 28 |
| attention heads | 12 |
| **KV heads** | **2** (GQA) |
| head_dim | 128 (hidden 1536 / 12) |
| max_position_embeddings | **131072** |
| licence | Apache 2.0, ungated |

`KV per token = 2 × 28 × 2 × 128 × 2 bytes = 28.0 KiB`

`n_kv_heads` is the input that matters: at 2 rather than 12 it is a 6× smaller
KV cache than MHA would give. Get that wrong and the 1-GPU conclusion becomes a
multi-GPU one.

The 128K native context is a bonus the plan wanted and Llama-3.2-1B would also
have provided — it means long-context runs need no RoPE scaling, so a TTFT curve
out to 128K measures the system rather than an extrapolation hack.
