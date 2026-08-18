# Phase 3 — raw CUDA, NVSHMEM, CUTLASS

Three technologies, each with a narrow justification. Two of them may produce
null results, and that is an acceptable outcome provided it is *measured*.

## 1. Persistent CUDA ring kernel

The only piece here that neither Triton nor CUTLASS expresses well: a kernel
that stays resident across all `P` ring steps, fetching the next KV shard from
inside the attention loop, with the online-softmax state `(acc, m, l)` living in
registers the whole time.

It is also the prerequisite for NVSHMEM — device-initiated fetch needs a kernel
to fetch *into*.

**Contract check:** `nsys` must show **one** kernel launch per ring traversal,
not `P`. A "persistent" kernel relaunched per step is exactly what raw CUDA was
chosen to avoid.

## 2. NVSHMEM — decode only, and the ordering is not negotiable

Upstream measurement: communication is **1.1% of prefill runtime**, and stays
hidden even after a hypothetical 15x kernel speedup (per-step compute would fall
to ~4 ms against ~0.35 ms per hop). NVSHMEM cannot win there. Using it for
prefill would be decoration, and `test_nvshmem_is_only_used_in_the_regime_where
_it_can_win` fails if it appears on that path.

Decode is different: one query against a growing cache, small messages, latency-
bound. That is NVSHMEM's regime.

**Order:**

1. Compile and run `nvshmem/ring_bw.cu` in the upstream repo. It has never been
   compiled; expect build-flag fixes.
2. **Decision gate.** If device-initiated `getmem` does not beat NCCL send/recv
   in the small-message regime, **stop and publish the null result.** That is a
   genuine finding and worth more than a half-built kernel.
3. Only if it wins: integrate into the decode path at P=8.

Needs 8x A100/H100 — the ring must be long enough for the effect to exist.

## 3. CUTLASS — a comparison, not a dependency

Hand-tuned QK^T/PV against the Triton kernel on identical shapes. The
deliverable is the comparison and the `ncu` evidence, not necessarily a faster
kernel. **"Triton reaches N% of hand-tuned CUTLASS" is a publishable result.**

Deferred to last because its marginal value collapses if Triton reaches 40% MFU,
and it has the steepest learning curve of anything in the project.

## Validation

Everything here is diffed against `lis.kernels.flash_attention(..., backend=
"unfused")`. Do not validate a hand-written CUDA kernel against another
hand-written CUDA kernel.
