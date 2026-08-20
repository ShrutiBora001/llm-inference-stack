# Comment for SGLang #21788 (CP roadmap — OPEN, High priority)

https://github.com/sgl-project/sglang/issues/21788

---

Hi — the roadmap notes KV-cache sharding by sequence dim is "probably with Ring
Attention". I've built and measured a zigzag ring prefill implementation and
wanted to check whether a CUDA contribution here would be useful before doing
more.

Measured on 4× A100 SXM4 40GB / NVLink:

- Zigzag (rank `r` takes chunks `r` and `2P−1−r`) vs contiguous: **1.69× at
  S=32768, 1.73× at S=131072** — 99.1% of the `2 − 1/P` critical-path bound.
- **1,048,576-token prefill** at 32.3 GiB/device, where an all-gather baseline
  needs ~38 GiB and OOMs.
- Communication is 1.1% of runtime with double-buffered NCCL P2P overlap.
- Verified against a single-process reference at P=2/4/8, `<1e-5`.

Caveat up front: my attention kernel is unfused (~3% MFU), so the absolute
numbers are not competitive — the speedups are ratios between two layouts of the
same kernel, which is what the scheduling claim rests on.

Two questions:

1. Is ring attention still the likely direction for sequence-dim KV sharding, or
   has that been decided otherwise? Don't want to build against a dead path.
2. If a CUDA prefill-CP contribution is wanted, how would you want it scoped?
   #22223 has an Ascend NPU implementation; a CUDA port seems the natural gap,
   but it's a large PR and I'd rather split it the way you prefer.

Happy to share the repo and raw measurements.
