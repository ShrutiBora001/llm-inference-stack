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
- Communication was 1.1% of runtime with double-buffered NCCL P2P overlap.
- Verified against a single-process reference at P=2/4/8, `<1e-5`.

One caveat I'd rather raise myself, because it cuts against my own numbers.
Those measurements used an unfused attention kernel (~3% MFU). I've since
validated a fused path on 1× A100 — 60.4% MFU at S=32768 — which makes compute
roughly 20× faster and leaves communication unchanged. That moves comm from
**1.1% of runtime to an estimated ~18%**, so the layout speedup measured above
will compress once the kernel is fast. I have not yet re-measured the ratio at
4× with the fused kernel; that is the next thing I plan to run.

I mention it because it seems relevant to the roadmap decision rather than only
to my numbers: with a fast kernel, overlap quality starts to matter much more
than it does today, which is an argument for ring over all-gather rather than
against it.

Two questions:

1. Is ring attention still the likely direction for sequence-dim KV sharding, or
   has that been decided otherwise? Don't want to build against a dead path.
2. If a CUDA prefill-CP contribution is wanted, how would you want it scoped?
   #22223 has an Ascend NPU implementation; a CUDA port seems the natural gap,
   but it's a large PR and I'd rather split it the way you prefer.

Happy to share the repo and raw measurements.
