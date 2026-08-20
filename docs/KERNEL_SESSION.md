# Phase 1 rent-day runbook — validate the Triton fused kernel

One GPU. Budget ~$3 and two hours. The question this session answers, and the
only one: **does `src/lis/kernels/triton_flash.py` clear the 20% MFU gate?**

Both answers are shippable. "It does" flips `PREFERENCE` in the dispatcher.
"It does not" adopts `torch_flash` as the fused path and reports Triton as a
measured null result. Leaving it undecided is the only bad outcome, because
every Phase 2 serving number measured on a 3%-MFU kernel has to be re-run.

## Why one GPU

Four cannot make a single kernel converge faster, and cost 4x. Nothing in this
session is distributed: no NCCL, no ring, no interconnect. The multi-GPU work
is Phase 2 and needs a *decided* kernel first.

## What to rent

| | |
|---|---|
| GPU | **1 × A100** (SXM or PCIe — irrelevant here, there is no interconnect traffic) |
| Fallback | 1 × A10 / L4 / 4090 is fine for correctness; MFU is **not** comparable to the upstream A100 numbers |
| Disk | **64 GB** — the NGC image is ~25 GB and **disk cannot be resized after creation** |
| Image | PyTorch NGC |
| Cost | ~$1.00–1.50/hr |

Ampere or newer (sm_80+) is required for bf16 tensor cores. Step 1 of the
validator warns if you are below that, because every MFU number would be
meaningless.

**Do not rent 4 GPUs for this.** If a listing bundles four, the session still
works — `preflight 1` accepts it — but you are paying 4x for three idle cards.

## Run it

```bash
deploy/remote.sh <PORT> <HOST> kernel
```

That syncs, sets up, runs the six-step validator, runs the full MFU sweep across
every backend, runs the contract suite with the Triton checks live rather than
skipped, and pulls `results/` back.

To work interactively instead — which is what you want the moment something
fails:

```bash
deploy/remote.sh <PORT> <HOST> shell
cd /workspace/lis && python scripts/validate_triton.py --skip-mfu
```

`--skip-mfu` stops after correctness. Use it while iterating on a fix; there is
no point benchmarking a kernel that computes the wrong answer.

## The six steps, and what each rules out

| | Step | Rules out |
|---|---|---|
| 1 | environment | no GPU, no Triton, dispatcher not routing to it |
| 2 | compile | build failure, at the smallest shape so it fails in seconds |
| 3 | correctness | wrong arithmetic, wrong mask, **wrong global-offset handling** |
| 4 | lse contract | an lse the ring cannot merge — silent, and fatal for Phase 2 |
| 5 | autotune | the search never leaving candidate 0 |
| 6 | MFU | the actual question |

Step 3 is the one that matters most. The oracle chain is rooted outside this
codebase — `unfused` is pinned to PyTorch's own SDPA upstream — so passing step 3
compares the kernel to PyTorch, not to itself. The `q_offset`/`k_offset` cases
are there because a kernel that masks against *local* indices produces a
perfectly plausible wrong answer under a striped layout, with no crash and no
shape error.

## Failures to expect on first contact

This kernel has never been compiled. These are anticipated, not defects:

- **`key=[..., "BLOCK_D", "CAUSAL"]` in `@triton.autotune`.** Both are
  `constexpr` and already specialize the kernel. Some Triton versions reject
  constexpr names in the autotune key. Fix: drop them from `key`.
- **The first call takes minutes.** 36 configs × compile, per shape. It looks
  like a hang. It is not.
- **`_attn_fwd.best_config` missing.** The autotuner's attribute name has moved
  between versions. Step 5 fails with that exact message; the capture is in
  `triton_flash.LAST_CONFIG`.
- **`tl.dot` minimum tile size.** Triton requires ≥16 in each dimension on some
  versions; `BLOCK_N=32` is safe but a `head_dim` below 16 is not.
- **Shared-memory overflow** at `BLOCK_M=128, BLOCK_N=128, num_stages=4` with
  `head_dim=128`. Triton usually prunes these itself; if it does not, narrow
  `_configs()`.

## Before you destroy the instance

```bash
ls results/triton_validation.json results/kernel.jsonl
```

Rented storage vanishes when you stop paying. `deploy/remote.sh` pulls results
back automatically; if you worked in `shell`, rsync them yourself first.

Then, on the laptop:

```bash
make report     # K1 and K2 redraw from the real numbers
make contract   # Triton checks go back to skipping, visibly
```

## Recording the decision

Whichever way it goes, it goes in `docs/PHASE1.md` with the number attached, and
`results/triton_validation.json` is committed as the evidence. A gate that moves
after measurement needs a written justification — that is design rule 5, and it
is the rule that stops a gate from becoming whatever the implementation happened
to achieve.
