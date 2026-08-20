# Outreach

Drafts for upstream contact. Kept in the repo because the *decision* they
trigger — whether Phase 2 is a PR or a benchmark study — is a project fact worth
versioning, not a throwaway message.

| File | Target | Status |
|---|---|---|
| `sglang-21788.md` | CP roadmap, **open**, High priority | not sent |
| `sglang-22223.md` | PCP zigzag, **closed** as inactive | not sent |

## Why two

#22223 describes exactly this work but is closed and unwatched; a comment there
alone may reach nobody. #21788 is open, High priority, with five assignees and
PRs in flight — that is where people actually are.

## What the answer decides

- **"Yes, and here's the scope"** → proceed with the CUDA port (plan §2.1–2.4)
- **"Ring isn't the direction"** → drop the port; Phase 2 becomes the benchmark
  study (§2.5), which needs nobody's approval
- **Silence for ~2 weeks** → same as above

### A correction to the plan's scoping

The plan called the serving study "ungated — start here" while naming
`sglang-zigzag` and `sglang-contiguous` as two of its three configurations.
Both require SGLang prefill CP **on CUDA**, which does not exist upstream — the
#22223 implementation is Ascend-NPU only. So those two are gated on the port,
not ungated.

What is genuinely ungated is the more valuable half:

| Measurement | Needs CP? |
|---|---|
| Prefix-share crossover, SGLang vs vLLM | no |
| Prefix-cache hit rate vs `prefix_share` | no |
| TTFT / goodput vs concurrency | no |
| Zigzag vs contiguous CP layout | **yes** |

The crossover was always the headline finding (plan §8.1, S4) and it is a
property of RadixAttention versus vLLM's prefix caching, not of context
parallelism. The study proceeds; only the layout comparison waits.

`lis.serving.sglang_backend.make_client` enforces this in code rather than
leaving it in a document: the CP configurations refuse with the reason, because
constructing a non-CP engine under a CP name would write a results file
describing a system that was never run.

Per the plan, **do not start the port before this is answered.** The benchmark
study proceeds either way.

## Notes on the drafts

Both lead with measurements rather than an offer to help.

`21788` volunteers the caveat that cuts against its own numbers, and Phase 1
changed what that caveat is. It used to be "my kernel is unfused at ~3% MFU, so
absolute numbers are not competitive" — obsolete now that a fused path measures
60.4% MFU. The replacement is sharper and more useful: a 20x faster kernel
leaves communication unchanged, moving it from **1.1% of runtime to ~18%**, so
the 1.69-1.73x layout speedup will compress once the kernel is fast. That has
not been re-measured at 4x yet, and the draft says so.

Volunteering the weakness is what makes the rest credible — a maintainer
computes it in thirty seconds anyway, and finding it themselves after reading
the claims is much worse. It also happens to be the strongest argument in the
message: with a fast kernel, overlap quality starts to matter, which favours
ring over all-gather rather than undermining it.
