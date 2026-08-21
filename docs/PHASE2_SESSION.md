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
| 1B @ 8K × 64 concurrent | 16.2 GiB | 18.6 GiB | **1** | **$1.09** |
| 1B @ 32K × 64 | 64.2 GiB | 66.6 GiB | 2 | $2.18 |
| 1B @ 128K × 16 | 64.1 GiB | 66.4 GiB | 2 | $1.90 |
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

**Session A — the crossover, 1× A100, ~$1.** `sglang` vs `sglang-no-cache` vs
`vllm`, sweeping prefix share. Produces the headline S4 figure and the
cache-hit-rate curve. Nothing here is blocked.

**Session B — the CP arms, 2–4× A100.** Adds `vllm-dcp`, and the zigzag layout
comparison if the port lands. Worth combining with re-measuring the upstream
1.69×/1.73× layout ratio under a fused kernel, which Phase 1 showed will
compress (comm moves from 1.1% to ~18% of runtime).

Doing A first is strictly better: it is cheap, it is unblocked, and it debugs
the whole harness — workload generation, both bindings, the reduction path,
the figures — on the cheapest possible hardware before B pays 4× per hour for
the same bugs.

## The one number to verify before renting

`scripts/size_session.py` prints a warning because the model geometry is from
memory, not from the model card:

```
llama-3.2-1b: 16 layers, 8 kv heads, head_dim 64  ->  32.0 KiB/token
```

`n_kv_heads` is the one that matters. Llama-3.2-1B uses grouped-query attention,
so `n_kv_heads` (8) is well below `n_heads` (32). If that were wrong and the
model were MHA, KV per token would be **4× larger** and the 1-GPU conclusion
would collapse into a 4-GPU one. Check `config.json` before committing to a
listing — it is a thirty-second check that decides the whole rental.
