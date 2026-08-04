# llm-inference-stack

A fused, load-balanced ring attention kernel and the serving stack around it.

Builds on [causal-attention-load-imbalance](https://github.com/ShrutiBora001/causal-attention-load-imbalance)
(frozen at v0.3.0), which established that striped sequence assignment gives
**1.69–1.73× on 4× A100** — 99.1% of its theoretical ceiling — but also that the
implementation runs at **2.5–3.4% of the A100's bf16 peak**. Every remaining
limitation is downstream of that number: a 1M-token result is academic if nobody
would deploy the kernel that produced it.

This project fixes that, then serves real models with it.

## The anti-decoration contract

The organising constraint: **no technology is present for its name.** Each must
do something the others cannot do as well, and each has a test that fails if it
has become decorative.

| Technology | Uniquely best at | Used for | Test that fails if decorative |
|---|---|---|---|
| **Triton** | Fast kernel authoring, autotuning | Fused attention with LSE output | MFU ≥ 20% gate; autotuner picks a non-default config; assert no silent fallback |
| **NCCL** | Mature bulk collectives | Ring communication | Wire bytes match the closed-form model; busbw ≥ 60% of nameplate |
| **torch.compile / CUDA graphs** | Removing launch overhead | Decode step capture, elementwise fusion | Zero graph breaks; launches per decode step halve |
| **vLLM** | PagedAttention, continuous batching | Attention backend plugin; long context | Block table non-trivial; requests overlap in time |
| **SGLang** | RadixAttention prefix caching | Many small requests, shared prefixes | Prefix-cache hit rate > 50% on a prefix-sharing workload |
| **raw CUDA** | Persistent kernels, device-side scheduling | Ring kernel with in-loop KV fetch | One launch per traversal, not P |
| **NVSHMEM** | Device-initiated, latency-bound transfers | Decode-path KV fetch at P=8 | Must beat NCCL *in decode*; refuses to enable for prefill |
| **CUTLASS** | Peak GEMM, warp specialization | Comparison against Triton | Comparison measured, not asserted; tensor cores actually used |

```bash
make contract   # runs it; GPU-gated checks SKIP visibly, never pass vacuously
```

Two scoping decisions worth stating, because they are the ones most likely to be
got wrong:

**NVSHMEM is restricted to decode.** The upstream report measures communication
at 1.1% of prefill runtime, still hidden by overlap even after a hypothetical 15×
kernel speedup. Using NVSHMEM for prefill would be exactly the decoration this
contract forbids, so `test_nvshmem_is_only_used_in_the_regime_where_it_can_win`
fails if it appears there.

**JAX is excluded**, not forgotten — it would mean maintaining two stacks when
vLLM and SGLang are both PyTorch. See [future_scope.md](future_scope.md).

## Status

| | |
|---|---|
| RoPE under striped layouts | done, 29 tests |
| LSE merge for chunked attention | done, 16 tests |
| Anti-decoration contract suite | done, skips visibly on CPU |
| Triton fused kernel | next — needs a single GPU |
| vLLM / SGLang integration | Phase 2 |
| raw CUDA + NVSHMEM + CUTLASS | Phase 3 |

## Why LSE merging matters

It is the decision that makes the Triton phase tractable. The upstream ring
carries `(acc, m, l)` through its own tiling loop, so a fused kernel would have
to be ring-aware — accepting and returning that state. Instead, every
FlashAttention-style kernel already returns `(out, lse)`, and two partials merge
exactly:

```
lse_ab = logaddexp(lse_a, lse_b)
out_ab = out_a·exp(lse_a − lse_ab) + out_b·exp(lse_b − lse_ab)
```

So a stock kernel is called once per (query chunk, key chunk) pair and combined
in `lis.lse`. The ring never enters the kernel. That turns "write ring attention
in Triton" into "call a flash kernel and merge" — a far smaller surface, and the
merge is proven correct on CPU before any GPU is rented.

## The RoPE trap

Rotary embeddings depend on **absolute** position. Under a striped layout a
device owns chunks `r` and `2P−1−r` — two *disjoint, non-adjacent* ranges — so
there is no single offset that is correct.

Getting this wrong does not crash. No NaN, no shape error. The model attends with
wrong relative distances and emits fluent, confident nonsense, which is far worse
than a crash. `lis.rope` makes global positions the only convenient option, and
`tests/test_rope.py` includes the deliberately-broken implementation to assert
the test suite actually detects it — a positional test that passes for both the
correct and broken version is not testing anything.

## Layout

```
src/lis/
  rope.py       RoPE with global positions; the striping trap
  lse.py        LSE merging — the kernel contract
  jit.py        torch.compile + CUDA graph capture for decode
  kernels/      Triton (Phase 1), persistent CUDA + NVSHMEM (Phase 3)
tests/
  test_rope.py      positional correctness, incl. proof the test discriminates
  test_lse.py       merge exactness, order invariance, empty-partial edges
  test_contract.py  the anti-decoration suite
docs/LOCAL_CHECKLIST.md   what to verify before renting anything
future_scope.md           what was considered and deferred, with reasons
```

## Local first

Everything above runs on a laptop with no GPU. That is deliberate: the upstream
project settled its entire correctness argument on a MacBook and used rented
hardware only for performance, which is the discipline
[docs/LOCAL_CHECKLIST.md](docs/LOCAL_CHECKLIST.md) carries forward — including a
prediction log filled in *before* each session, so results are confirmations
rather than observations.

```bash
make venv && make test && make contract
```
