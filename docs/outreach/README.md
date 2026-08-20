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

Per the plan, **do not start the port before this is answered.** The benchmark
study proceeds either way.

## Notes on the drafts

Both lead with measurements rather than an offer to help, and both state the
~3% MFU caveat up front. Volunteering the weakness is what makes the rest
credible — a maintainer will compute it in thirty seconds anyway, and finding it
themselves after reading the claims is much worse.
