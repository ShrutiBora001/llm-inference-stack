#!/usr/bin/env python3
"""Validate the Triton fused kernel. Run this FIRST on a rented box.

    python scripts/validate_triton.py

Six steps, ordered by cost. Each one is cheaper than the one after it and rules
out a distinct failure, so a broken kernel fails in seconds rather than after a
ten-minute autotune. The expensive step (MFU) only runs once correctness holds
-- benchmarking a kernel that computes the wrong answer is the most expensive
way to learn nothing.

    1  environment      is there a GPU, and does Triton import
    2  compile          does it build at all, at the smallest possible shape
    3  correctness      does it match the unfused oracle: plain, causal, offset
    4  lse contract     can two halves be merged into the whole
    5  autotune         did the search space get explored
    6  mfu              does it clear the gate

Exit code is 0 only if every step passes. Step 6 failing the gate is reported
but does not fail the run: "Triton did not clear the gate" is a legitimate
measured outcome, and the decision it feeds is `torch_flash` becomes the fused
path. A wrong answer in steps 2-4 is not a legitimate outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

BOLD, RED, GREEN, YELLOW, DIM, OFF = (
    "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
)

RESULTS: dict = {"steps": {}}


def head(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}/6] {title}{OFF}")


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{OFF}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{OFF}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{OFF}  {msg}")


def step(n: int, title: str):
    """Record each step's outcome so the whole run lands in one JSON file."""
    def deco(fn):
        def run(*a, **kw):
            head(n, title)
            t0 = time.perf_counter()
            try:
                fn(*a, **kw)
            except Exception as exc:
                RESULTS["steps"][title] = {
                    "ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.perf_counter() - t0,
                }
                fail(f"{type(exc).__name__}: {exc}")
                print(DIM + traceback.format_exc() + OFF)
                return False
            RESULTS["steps"][title] = {"ok": True, "seconds": time.perf_counter() - t0}
            return True
        return run
    return deco


# ------------------------------------------------------------------ 1
@step(1, "environment")
def check_env() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device. This script is for a rented GPU box; there is "
            "nothing here it can validate."
        )
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    ok(f"{dev}, sm_{cap[0]}{cap[1]}, torch {torch.__version__}")

    import triton

    ok(f"triton {triton.__version__}")
    RESULTS["device"] = dev
    RESULTS["capability"] = f"sm_{cap[0]}{cap[1]}"
    RESULTS["triton"] = triton.__version__
    RESULTS["torch"] = torch.__version__

    if cap[0] < 8:
        warn(f"sm_{cap[0]}{cap[1]} predates Ampere; bf16 tensor cores are absent "
             "and every MFU number below will be meaningless")

    from lis.kernels import available_backends, triton_available

    ok(f"backends: {available_backends()}")
    if not triton_available():
        raise RuntimeError(
            "lis.kernels.triton_available() is False despite Triton importing. "
            "The dispatcher will never route to the kernel, so nothing below "
            "would be testing it."
        )


# ------------------------------------------------------------------ 2
@step(2, "compile at the smallest shape")
def check_compile() -> None:
    """The cheapest possible question: does it build.

    64 queries, 64 keys, head_dim 64, non-causal. Small enough that a
    compilation failure surfaces in seconds, and every autotune config is legal.
    """
    from lis.kernels.triton_flash import flash_attention_triton

    q, k, v = (torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    t0 = time.perf_counter()
    r = flash_attention_triton(q, k, v, causal=False)
    torch.cuda.synchronize()

    ok(f"compiled and ran in {time.perf_counter() - t0:.1f}s "
       f"(first call includes autotune)")
    if not torch.isfinite(r.out).all():
        raise RuntimeError("output contains NaN or inf at the smallest shape")
    if not torch.isfinite(r.lse).all():
        raise RuntimeError("lse contains NaN or inf at the smallest shape")
    ok(f"out {tuple(r.out.shape)} {r.out.dtype}, lse {tuple(r.lse.shape)} {r.lse.dtype}")


# ------------------------------------------------------------------ 3
@step(3, "correctness vs the unfused oracle")
def check_correctness() -> None:
    """The oracle chain: unfused is pinned to PyTorch SDPA upstream, so this
    transitively compares the Triton kernel to PyTorch rather than to itself.

    Three cases, each catching something the previous cannot:
      plain   the arithmetic
      causal  the mask
      offset  global-position masking -- the striped-layout case, where a
              kernel using local indices produces a plausible wrong answer
    """
    from lis.kernels import flash_attention

    cases = [
        ("plain",            dict(causal=False)),
        ("causal",           dict(causal=True)),
        ("causal q_off=256", dict(causal=True, q_offset=256, k_offset=0)),
        ("causal both_off",  dict(causal=True, q_offset=512, k_offset=256)),
        ("fully future",     dict(causal=True, q_offset=0, k_offset=1024)),
    ]

    torch.manual_seed(1234)
    b, h, s, d = 2, 4, 256, 64
    q, k, v = (torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))

    worst = 0.0
    for name, kw in cases:
        got = flash_attention(q, k, v, backend="triton", **kw)
        want = flash_attention(q, k, v, backend="unfused", **kw)

        # A fully-future block legitimately produces empty rows: out 0, lse -inf.
        # Comparing those with a relative tolerance is meaningless, so compare
        # the mask itself and then only the live rows.
        live = torch.isfinite(want.lse)
        if live.any():
            diff = (got.out[live] .float() - want.out[live].float()).abs().max().item()
            ldiff = (got.lse[live] - want.lse[live]).abs().max().item()
        else:
            diff = ldiff = 0.0

        empty_match = torch.equal(torch.isfinite(got.lse), live)
        worst = max(worst, diff, ldiff)

        # bf16 has ~8 mantissa bits; 2e-2 is the tolerance the rest of the suite
        # uses for bf16 comparisons against an fp32-accumulating reference.
        if diff > 2e-2 or ldiff > 2e-2 or not empty_match:
            fail(f"{name}: out {diff:.2e}, lse {ldiff:.2e}, "
                 f"empty-row pattern {'matches' if empty_match else 'DIFFERS'}")
            raise RuntimeError(
                f"{name} disagrees with the oracle. The kernel is wrong; do not "
                "benchmark it. Check global-offset masking first -- that is the "
                "failure that produces plausible-looking wrong answers."
            )
        ok(f"{name}: out {diff:.2e}, lse {ldiff:.2e}")

    RESULTS["max_abs_error"] = worst


# ------------------------------------------------------------------ 4
@step(4, "lse merge contract")
def check_lse_contract() -> None:
    """The property the whole ring depends on.

    Attention over [0, S) must equal merge(attention over [0,S/2),
    attention over [S/2,S)). If this fails the kernel is unusable for context
    parallelism no matter how fast it is -- and the failure is silent, because
    each half looks perfectly reasonable on its own.
    """
    from lis.kernels import flash_attention
    from lis.lse import merge

    torch.manual_seed(1234)
    b, h, s, d = 1, 4, 512, 64
    q, k, v = (torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    half = s // 2

    whole = flash_attention(q, k, v, backend="triton", causal=True)
    a = flash_attention(q, k[:, :, :half], v[:, :, :half],
                        backend="triton", causal=True, q_offset=0, k_offset=0)
    bb = flash_attention(q, k[:, :, half:], v[:, :, half:],
                         backend="triton", causal=True, q_offset=0, k_offset=half)
    merged = merge(a, bb)

    diff = (merged.out.float() - whole.out.float()).abs().max().item()
    ldiff = (merged.lse - whole.lse).abs().max().item()
    if diff > 2e-2 or ldiff > 2e-2:
        raise RuntimeError(
            f"split-then-merge != whole (out {diff:.2e}, lse {ldiff:.2e}). "
            "The kernel's lse is not what merge() expects -- check it is a "
            "natural log and that empty rows carry -inf."
        )
    ok(f"merge(halves) == whole: out {diff:.2e}, lse {ldiff:.2e}")


# ------------------------------------------------------------------ 5
@step(5, "autotuner explored the space")
def check_autotune() -> None:
    """Triton's headline advantage. If the search never moves off candidate
    zero the tile sizes were effectively hardcoded and Triton bought nothing
    over writing the same loop by hand."""
    from lis.kernels import autotune_report, flash_attention

    for s in (512, 2048):
        q, k, v = (torch.randn(1, 8, s, 64, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        flash_attention(q, k, v, backend="triton", causal=True)
    torch.cuda.synchronize()

    report = autotune_report()
    if not report:
        raise RuntimeError(
            "autotune_report() is empty. best_config was not captured -- the "
            "Triton API for it moved. Fix triton_flash.LAST_CONFIG capture."
        )
    for key, entry in report.items():
        ok(f"{key}: {entry['config']} (candidate #{entry['chosen_index']})")

    RESULTS["autotune"] = report
    if all(e["chosen_index"] == 0 for e in report.values()):
        warn("autotuner always chose candidate 0 across every shape. Either the "
             "space is badly ordered or the search is not running; the contract "
             "test will fail on this.")


# ------------------------------------------------------------------ 6
@step(6, "MFU against the gate")
def check_mfu(seqs: list[int]) -> None:
    """The question the rental exists to answer."""
    from lis.bench_kernel import sweep
    from lis.metrics import MFU_GATE

    backends = ["unfused", "torch_flash", "triton"]
    rows = sweep(seqs=tuple(seqs), backends=tuple(backends),
                 heads=32, head_dim=128, causal=True, iters=10, warmup=3)

    print(f"\n  {'backend':<14}{'seq':>8}{'ms':>10}{'TFLOP/s':>10}{'MFU':>8}{'FLOP/B':>9}")
    print("  " + "-" * 59)
    for r in rows:
        print(f"  {r.backend:<14}{r.seq:>8}{r.median_ms:>10.2f}{r.tflops:>10.2f}"
              f"{r.mfu:>7.1%}{r.arithmetic_intensity:>9.1f}")

    RESULTS["mfu_rows"] = [r.as_dict() for r in rows]
    tri = [r for r in rows if r.backend == "triton"]
    if not tri:
        raise RuntimeError("the sweep produced no Triton rows")

    best = max(tri, key=lambda r: r.mfu)
    RESULTS["triton_best_mfu"] = best.mfu
    RESULTS["gate"] = MFU_GATE

    # Compare at matched sequence lengths only. The unfused backend OOMs out of
    # the sweep at long sequences (its score tile is 34 GiB at S=16384), so
    # comparing each backend's best row would silently pit triton@16384 against
    # unfused@4096 and report a "speedup" that is mostly the shape difference.
    print()
    by_seq: dict[int, dict[str, float]] = {}
    for r in rows:
        by_seq.setdefault(r.seq, {})[r.backend] = r.mfu

    comparisons: dict[str, dict] = {}
    for name in ("torch_flash", "unfused"):
        shared = sorted(s for s, m in by_seq.items() if name in m and "triton" in m)
        if not shared:
            warn(f"no sequence length where both triton and {name} completed; "
                 "no like-for-like comparison is possible")
            continue
        at = max(shared)
        t, o = by_seq[at]["triton"], by_seq[at][name]
        ratio = t / o if o > 0 else float("inf")
        comparisons[name] = {"seq": at, "triton_mfu": t, "other_mfu": o, "ratio": ratio}
        print(f"  @ seq={at:<7} triton {t:>6.1%}  vs  {name} {o:>6.1%}  ({ratio:.2f}x)")

    dropped = {n: sorted(set(seqs) - {s for s in by_seq if n in by_seq[s]})
               for n in ("unfused", "torch_flash", "triton")}
    for name, missing in dropped.items():
        if missing:
            print(f"  {DIM}{name} did not complete at {missing} (OOM or unsupported){OFF}")

    RESULTS["comparisons"] = comparisons
    RESULTS["incomplete"] = {k: v for k, v in dropped.items() if v}

    if best.mfu >= MFU_GATE:
        ok(f"triton {best.mfu:.1%} >= gate {MFU_GATE:.0%} @ seq={best.seq}")
        # Clearing the gate proves the kernel is real, not that it should be the
        # default. Those are separate questions, and an earlier version of this
        # script conflated them -- it recommended flipping PREFERENCE on the
        # gate alone, which would have demoted a faster kernel for a slower one.
        tf = comparisons.get("torch_flash")
        if tf and tf["ratio"] < 1.0:
            RESULTS["decision"] = (
                f"triton clears the gate at {best.mfu:.1%} and is validated, but "
                f"torch_flash is faster at every matched shape "
                f"({tf['other_mfu']:.1%} vs {tf['triton_mfu']:.1%} @ seq={tf['seq']}). "
                "KEEP PREFERENCE as is; torch_flash stays the default fused path."
            )
        else:
            RESULTS["decision"] = (
                f"triton clears the gate at {best.mfu:.1%} and is at least as "
                "fast as torch_flash; flipping PREFERENCE is justified."
            )
        print(f"  {BOLD}decision:{OFF} {RESULTS['decision']}")
    else:
        warn(f"triton {best.mfu:.1%} < gate {MFU_GATE:.0%}. This is a valid "
             "measured outcome, not a script failure.")
        tf = others.get("torch_flash", 0.0)
        RESULTS["decision"] = (
            "triton below gate; adopt torch_flash as the fused path and report "
            f"triton as a measured null result (torch_flash reached {tf:.1%})"
        )
        print(f"  {BOLD}decision:{OFF} {RESULTS['decision']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seqs", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "triton_validation.json")
    p.add_argument("--skip-mfu", action="store_true",
                   help="stop after correctness; use while iterating on a fix")
    args = p.parse_args()

    print(f"{BOLD}Triton fused kernel validation{OFF}")

    passed = check_env() and check_compile() and check_correctness() \
        and check_lse_contract() and check_autotune()

    if passed and not args.skip_mfu:
        check_mfu(args.seqs)
    elif passed:
        warn("skipping MFU (--skip-mfu)")

    RESULTS["all_passed"] = all(s["ok"] for s in RESULTS["steps"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(RESULTS, indent=2))

    print(f"\n{BOLD}{'=' * 62}{OFF}")
    for name, s in RESULTS["steps"].items():
        mark = f"{GREEN}ok{OFF}" if s["ok"] else f"{RED}FAILED{OFF}"
        print(f"  {mark:<18} {name}  {DIM}{s['seconds']:.1f}s{OFF}")
    if "decision" in RESULTS:
        print(f"\n  {BOLD}DECISION:{OFF} {RESULTS['decision']}")
    print(f"\n  written to {args.out}")

    if not RESULTS["all_passed"]:
        print(f"\n  {RED}Kernel is not validated. Do not report any number "
              f"from this run.{OFF}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
