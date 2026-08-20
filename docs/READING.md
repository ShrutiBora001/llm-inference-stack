# Reading list

Ordered by weight, not by importance. Each tier assumes the one before it, and
every entry says **what in this codebase it explains** — the list exists to make
the code legible, not to be comprehensive.

If you read six things, read the ones marked ★.

> Citations were written from memory and spot-checked, not re-verified
> end-to-end. Titles and authors are reliable; if an arXiv number does not
> resolve, search the title.

---

## Tier 0 — orientation (blog posts, an evening each)

Read these before any paper. They build the intuition the papers assume you
already have.

| # | Read | Why it is first | Explains |
|---|---|---|---|
| 1 | Jay Alammar, **The Illustrated Transformer** | The only genuinely gentle introduction to attention. Pictures, no math. | what `q, k, v` are before you shard them |
| 2 ★ | Horace He, **Making Deep Learning Go Brrrr From First Principles** (`horace.io/brrr_intro.html`) | The single highest-value read here. Compute-bound vs memory-bound vs overhead-bound, explained in three examples. | **why our kernel hits 3% MFU** — [K2_roofline](../report/make_figures.py), and why fusing is the fix rather than a bigger GPU |
| 3 | kipply, **Transformer Inference Arithmetic** (`kipp.ly/transformer-inference-arithmetic/`) | Does the KV-cache and FLOP arithmetic by hand, slowly. | why KV memory drives the context-parallelism argument in `docs/PLAN.md` |
| 4 | NVIDIA, **Matrix Multiplication Background User's Guide** (DL Performance docs) | Tensor cores, tile quantization, why dtype changes throughput by 10×. | the [tensor-core bug](../../Distributed_Attention00/docs/ARCHITECTURE.md) — upcasting Q/K before the matmul bypassed them entirely |
| 5 | vLLM blog, **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention** | Sets up the serving vocabulary in 10 minutes. | `lis/serving/paging.py` |

**Checkpoint:** you should be able to say, without looking, whether attention at
S=128K is limited by FLOPs or by HBM bandwidth, and why the answer changes with
sequence length.

---

## Tier 1 — the numerical core

These four papers *are* `online_softmax.py`. Read them in this order; each is a
small delta on the previous.

| # | Read | Explains |
|---|---|---|
| 6 | Vaswani et al., **Attention Is All You Need** (arXiv:1706.03762) | the baseline. Read §3.2 only; skip the rest |
| 7 ★ | Milakov & Gimelshein, **Online normalizer calculation for softmax** (arXiv:1805.02867) | 8 pages, and it is the whole trick. The running-max rescaling in `SoftmaxState.block_update` is Algorithm 3 verbatim |
| 8 | Rabe & Staats, **Self-attention Does Not Need O(n²) Memory** (arXiv:2112.05682) | that online softmax makes attention **order-invariant**, which is the property that licenses the ring to fold blocks in *rank* order rather than *sequence* order |
| 9 ★ | Dao et al., **FlashAttention** (arXiv:2205.14135) | tiling + IO-awareness. Our `tiled_attention` has FlashAttention's memory profile and numerics but not its speed — this paper is the gap |
| 10 | Dao, **FlashAttention-2** (arXiv:2307.08691) | work partitioning and the causal-mask skip. §3.1 is what `triton_flash.py` implements |
| 11 | Su et al., **RoFormer** (arXiv:2104.09864) | RoPE. §3.4.2 is why `rope.py` must use **global** positions — under striping a device owns two disjoint position ranges, and local positions fail silently |

**Checkpoint:** derive the LSE merge — `lse_ab = logaddexp(lse_a, lse_b)` — on
paper, and explain why a fully-masked block gives `exp(-inf − -inf) = NaN`.
That guard is in `online_softmax.py` because this was learned the hard way.

Optional but load-bearing: Williams, Waterman & Patterson, **Roofline: An
Insightful Visual Performance Model** (CACM 2009). Six pages. It is the axis
system of figure K2.

---

## Tier 2 — parallelism

Where the upstream repo's finding lives.

| # | Read | Explains |
|---|---|---|
| 12 | Shoeybi et al., **Megatron-LM** (arXiv:1909.08053) | tensor parallelism — the axis we are *not* using. Read to know why sequence parallelism is a different thing |
| 13 | Li et al., **Sequence Parallelism** (arXiv:2105.13120) | the first ring-style sequence sharding |
| 14 ★ | Liu, Zaharia & Abbeel, **Ring Attention with Blockwise Transformers** (arXiv:2310.01889) | `dattn/ring_attn.py` and `lis/cp/ring.py`. The overlap argument in §3 is why the exchange is posted before the matmul it hides behind |
| 15 ★ | Brandon et al., **Striped Attention** (arXiv:2311.09431) | **the paper this project reproduces and extends.** Read it after the ring paper and the contrast is obvious. Our contribution over it: the critical-path formulation, a per-step balance test, and 4×A100 measurement at 1M tokens |
| 16 | Jacobs et al., **DeepSpeed Ulysses** (arXiv:2309.14509) | the all-to-all alternative to a ring. Different communication/memory trade-off; worth knowing before claiming the ring is best |
| 17 | Li et al., **DISTFLASHATTN** (arXiv:2310.03294) | an independent attack on the same causal load imbalance, via workload rebalancing rather than assignment. Good adversarial read — it is the strongest "why not do it this way instead" |

**Checkpoint:** explain, without notes, why the causal ring's cost is
`Σ_steps max_r work(r,s)` and not `Σ_r work(r)` — that distinction is the entire
result, and it is the thing to be fluent in for an interview.

---

## Tier 3 — serving systems

This is the tier that matches the inference role most directly.

| # | Read | Explains |
|---|---|---|
| 18 ★ | Kwon et al., **Efficient Memory Management for LLM Serving with PagedAttention** (arXiv:2309.06180, SOSP'23) | vLLM. Block tables ↔ our `Chunk`, which is exactly the mapping `serving/paging.py` implements |
| 19 | Yu et al., **Orca** (OSDI'22) | continuous / iteration-level batching. No arXiv — find the OSDI PDF. The reason `test_vllm_requests_overlap_in_time` exists in the contract suite |
| 20 | Zheng et al., **SGLang / RadixAttention** (arXiv:2312.07104) | prefix caching via a radix tree. The whole reason `prefix_share` is our independent variable — at share 0 the tree is dead weight |
| 21 ★ | Zhong et al., **DistServe** (arXiv:2401.09670, OSDI'24) | **where "goodput" comes from.** §2 is the argument for why throughput alone is misleading; `lis/metrics.py` takes its definition from here |
| 22 | Agrawal et al., **Sarathi-Serve** (arXiv:2403.02310, OSDI'24) | chunked prefill, and the prefill/decode interference that makes TTFT and TPOT trade off |
| 23 | Patel et al., **Splitwise** (arXiv:2311.18677, ISCA'24) | phase splitting across heterogeneous hardware |
| 24 | **Mooncake** (arXiv:2407.00079) | KV-cache-centric disaggregation at production scale. Read last; it assumes 18–23 |

**Checkpoint:** state the difference between throughput and goodput, and
construct a workload where a system wins on the first and loses badly on the
second. That scenario is the argument for our headline metric.

---

## Tier 4 — writing the kernels

Only needed when you touch `kernels/`. Reference material, not narrative — read
the parts you need.

| # | Read | Explains |
|---|---|---|
| 25 | **Triton tutorials**, especially `06-fused-attention.py` | the closest published relative of `triton_flash.py`. Diff ours against it before debugging ours |
| 26 | Tillet, Kung & Cox, **Triton** (MAPL 2019) | the block-level programming model, and why the autotuner matters — `test_triton_autotuner_selects_a_nondefault_config` fails if it is not exploring |
| 27 | Hwu, Kirk & El Hajj, **Programming Massively Parallel Processors** (4th ed.), ch. 1–6 | the only entry here that is a book. Occupancy, coalescing, shared memory. Needed for Phase 3's persistent kernel |
| 28 | **NVIDIA Ampere (A100) architecture whitepaper** | 108 SMs, 312 TFLOP/s dense bf16, 1.55 TB/s HBM, NVLink 3 — every constant in `dattn/profiles.py` |
| 29 | **NCCL documentation**, plus the `nccl-tests` README | the `busbw = algbw × 2(P−1)/P` convention used in `lis/bench_comm.py`. Read the README specifically — the convention is defined there, not in the docs |
| 30 | **NVSHMEM programming guide** | device-initiated one-sided transfer. Phase 3. Read the memory model section before writing anything |
| 31 | **CUTLASS / CuTe docs** | Phase 3 comparison only. Densest thing on this list; skip until Triton is decided |
| 32 | Ye et al., **FlashInfer** (arXiv:2501.01005) | a production attention-kernel library. Useful as a survey of what the current state of the art actually ships |

---

## Tier 5 — this project

Read after Tier 2. The docs assume the reader has the ring paper in hand.

| Read | Contents |
|---|---|
| [`Distributed_Attention00/report/REPORT.md`](../../Distributed_Attention00/report/REPORT.md) | the finding, 11 figures, method → results → conclusion |
| [`Distributed_Attention00/docs/ARCHITECTURE.md`](../../Distributed_Attention00/docs/ARCHITECTURE.md) | module layering, data flow, and the six invariants that fail *silently* |
| [`Distributed_Attention00/docs/EXPERIMENTAL_SETUP.md`](../../Distributed_Attention00/docs/EXPERIMENTAL_SETUP.md) | exact hardware, and §2.2 — why random tensors are the correct input here and where that argument expires |
| [`tests/test_contract.py`](../tests/test_contract.py) | the anti-decoration suite. Reads as a list of claims with their evidence attached |
| [`docs/PLAN.md`](PLAN.md) · [`docs/NVSHMEM.md`](NVSHMEM.md) | what is built, what is next, what was deliberately scoped out |

---

## The short path

If you have a week rather than a season, and the goal is to hold a technical
conversation about this work:

**2 → 7 → 9 → 14 → 15 → 21**

Brrrr, online softmax, FlashAttention, Ring Attention, Striped Attention,
DistServe. Roughly 90 pages. That is enough to explain every design decision in
both repos, and it covers the ground an inference-systems interview actually
walks over: what limits a kernel, why the ring is valid, why causal masking
breaks it, and what "fast" means when you are serving rather than benchmarking.
