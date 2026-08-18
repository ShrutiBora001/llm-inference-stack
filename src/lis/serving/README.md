# Phase 2 — serving

## Entry points

**vLLM.** `vllm.v1.attention.backends.registry.register_backend` is a supported
extension point, so this is a plugin rather than a fork. Subclass
`AttentionBackend`, and have its forward call `lis.kernels.flash_attention` per
(query chunk, key chunk) pair with `lis.lse.merge` combining them.

**SGLang.** Dispatch happens through `forward_batch.attn_backend.forward()`.
Confirm the exact registration path against the installed source — the docs are
thinner here than vLLM's.

## The translation problem

Both frameworks own the KV cache and hand out **paged, non-contiguous** blocks.
This project's layouts assume contiguous per-device shards. The integration is
mostly a mapping between:

    block table (paged, framework-owned)  <->  Chunk(local_start, length, global_start)

`global_start` is what both causal masking and RoPE need, so the mapping must
preserve it. Getting this wrong produces fluent garbage, not a crash — the same
failure mode `lis/rope.py` guards against.

## Testable without a GPU

- block-table construction and the paged<->chunk mapping
- RoPE positions derived from a block table
- scheduling logic
- a 2-layer, `d_model=128` random-weight model exercising the whole path on CPU

## Needs a GPU

- throughput and latency
- prefix-cache hit rates under load
- the vLLM vs SGLang crossover measurement

## Contract checks this branch must satisfy

`tests/test_contract.py`, enabled with `VLLM_TEST=1` / `SGLANG_TEST=1`:

- block table non-trivial (>1 block, non-contiguous) — else paging never happened
- requests provably overlap in time — else continuous batching is a queue
- prefix-cache hit rate > 50% on a shared-prefix workload — else RadixAttention
  is dead weight and the benchmark cannot distinguish SGLang from anything else
