# Comment for SGLang #22223 (PCP zigzag ring — CLOSED as inactive)

https://github.com/sgl-project/sglang/issues/22223

Post this *after* or alongside #21788. Short on purpose — the substantive
discussion belongs on the open roadmap thread.

---

Is anyone still working on this? It's closed as inactive with no linked PRs, so
I assume it's parked.

I have a tested zigzag ring prefill implementation (1.73× over contiguous at
S=131072, 99.1% of the `2 − 1/P` bound, verified at P=2/4/8) and would be glad
to pick up the CUDA side, since the existing implementation is Ascend-only.

Would you be open to reopening this, or would you rather track it under the CP
roadmap (#21788)? Asked there too.
