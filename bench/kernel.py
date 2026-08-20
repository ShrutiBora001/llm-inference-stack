#!/usr/bin/env python3
"""Kernel MFU / roofline sweep.

    python bench/kernel.py --out results/kernel.jsonl

Sweeps every available backend across sequence lengths and records TFLOP/s,
MFU and arithmetic intensity. On a laptop only `unfused` exists and the timings
are not meaningful -- the run is still useful for checking the harness end to
end before paying for a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lis.bench_kernel import sweep  # noqa: E402
from lis.kernels import available_backends  # noqa: E402
from lis.metrics import MFU_GATE  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seqs", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--no-causal", dest="causal", action="store_false", default=True)
    p.add_argument("--out", type=Path, default=Path("results/kernel.jsonl"))
    args = p.parse_args()

    print(f"backends available: {available_backends()}")
    rows = sweep(
        seqs=tuple(args.seqs), backends=tuple(args.backends) if args.backends else None,
        heads=args.heads, head_dim=args.head_dim, causal=args.causal,
        iters=args.iters, warmup=args.warmup,
    )
    if not rows:
        raise SystemExit("no measurements produced")

    print(f"\n{'backend':<14}{'seq':>8}{'ms':>10}{'TFLOP/s':>10}{'MFU':>8}{'intensity':>11}")
    print("-" * 61)
    for r in rows:
        print(f"{r.backend:<14}{r.seq:>8}{r.median_ms:>10.2f}{r.tflops:>10.2f}"
              f"{r.mfu:>7.1%}{r.arithmetic_intensity:>11.1f}")

    best = max(rows, key=lambda r: r.mfu)
    print(f"\nbest MFU: {best.mfu:.1%} ({best.backend} @ seq={best.seq}) "
          f"vs gate {MFU_GATE:.0%}")
    if best.peak_tflops and best.mfu < MFU_GATE:
        print("  -> below gate. See report section on the roofline: if arithmetic")
        print("     intensity is low the kernel is memory-bound, not compute-bound.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r.as_dict()) + "\n")
    print(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
