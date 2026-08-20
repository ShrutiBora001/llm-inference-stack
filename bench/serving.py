#!/usr/bin/env python3
"""Serving benchmark: sweep shared-prefix length, report goodput.

    python bench/serving.py --sweep-prefix-share --framework sglang-zigzag
    python bench/serving.py --dry-run          # shape check, no engine needed

The independent variable is `prefix_share` -- how much of each prompt is shared
with its neighbours. It is the variable that decides whether a prefix cache is
doing anything, and it is the one existing benchmarks fix rather than sweep.

**Goodput is the headline**, not throughput: a system can post excellent
throughput while missing latency targets on most requests. Goodput cannot.

Without a serving engine installed this refuses to invent numbers -- `--dry-run`
exercises the workload and reduction path and prints the shapes, which is what
is checkable on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lis.metrics import SLO, RequestRecord, summarize  # noqa: E402
from lis.workload import WorkloadSpec, generate, sweep  # noqa: E402

FRAMEWORKS = ("sglang-zigzag", "sglang-contiguous", "vllm-dcp")
PREFIX_SHARES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def build_client(framework: str):
    """Resolve a framework name to a request driver.

    Deliberately raises rather than falling back to a simulation. A serving
    number produced without a serving engine is not a measurement, and the
    distinction is invisible once it reaches a plot.
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework {framework!r}; expected one of {FRAMEWORKS}")

    module = "sglang_backend" if framework.startswith("sglang") else "vllm_backend"
    try:
        mod = __import__(f"lis.serving.{module}", fromlist=["make_client"])
        return mod.make_client(framework)
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"{framework}: lis.serving.{module}.make_client is a Phase 2 stub "
            f"({exc}). Install the engine and implement the binding, or use "
            "--dry-run to check the workload path without one. This refuses to "
            "simulate: a serving number produced without a serving engine is "
            "not a measurement, and that distinction vanishes once it reaches "
            "a plot."
        ) from exc


def run_one(client, spec: WorkloadSpec, slo: SLO) -> dict:
    """One (framework, prefix_share) cell."""
    workload = generate(spec)
    t0 = time.perf_counter()
    records: list[RequestRecord] = list(client.send(workload))
    wall = time.perf_counter() - t0

    summary = summarize(records, wall_seconds=wall, slo=slo)
    return {
        **summary.as_dict(),
        "framework": client.name,
        "prefix_share_requested": spec.prefix_share,
        "prefix_share_realized": workload.realized_prefix_share(),
        "workload_source": workload.source,
        "n_requests": len(records),
        "wall_seconds": wall,
    }


def specs_for(args) -> list[WorkloadSpec]:
    base = WorkloadSpec(n_requests=args.requests, prompt_tokens=args.input_len,
                        output_tokens=args.output_len, qps=args.qps, seed=args.seed)
    return sweep(base, shares=tuple(args.prefix_shares))


def dry_run(args) -> int:
    """Everything except the engine: generation, shapes, realized prefix share.

    The realized column is the one to read. A sampler that silently ignores the
    requested prefix would still produce a plausible curve; measuring the ratio
    it actually produced is what makes the x-axis trustworthy.
    """
    print(f"{'prefix_share':>13}  {'realized':>9}  {'requests':>8}  {'prompt tok':>11}  source")
    for spec in specs_for(args):
        w = generate(spec)
        total = sum(r.prompt_len for r in w.requests)
        print(f"{spec.prefix_share:13.2f}  {w.realized_prefix_share():9.3f}  "
              f"{len(w.requests):8d}  {total:11d}  {w.source}")
    print("\ndry run — no engine contacted, no serving metrics produced.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frameworks", nargs="+", default=["sglang-zigzag"], choices=FRAMEWORKS)
    p.add_argument("--sweep-prefix-share", action="store_true")
    p.add_argument("--prefix-shares", nargs="+", type=float, default=list(PREFIX_SHARES),
                   dest="prefix_shares")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--input-len", type=int, default=8192)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ttft-p99-ms", type=float, default=2000.0)
    p.add_argument("--tpot-p99-ms", type=float, default=50.0)
    p.add_argument("--out", default="results/serving.jsonl")
    p.add_argument("--dry-run", action="store_true",
                   help="generate workloads and print shapes; contact no engine")
    args = p.parse_args()

    if not args.sweep_prefix_share:
        args.prefix_shares = args.prefix_shares[:1]

    if args.dry_run:
        return dry_run(args)

    slo = SLO(ttft_p99_ms=args.ttft_p99_ms, tpot_p99_ms=args.tpot_p99_ms)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out.open("a") as fh:
        for framework in args.frameworks:
            client = build_client(framework)
            for spec in specs_for(args):
                row = run_one(client, spec, slo)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                print(f"{framework:20s} prefix_share={row['prefix_share_realized']:.2f}  "
                      f"goodput={row['goodput_rps']:.2f} req/s  "
                      f"throughput={row['throughput_rps']:.2f}  "
                      f"TTFT p99={row['ttft_p99_ms']:.0f} ms  "
                      f"SLO {row['slo_attainment']:.0%}")

    print(f"\nappended {len(rows)} rows to {args.out}")
    if rows:
        best = max(rows, key=lambda r: r["goodput_rps"])
        print(f"best goodput: {best['goodput_rps']:.2f} req/s "
              f"({best['framework']} @ prefix_share={best['prefix_share_realized']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
