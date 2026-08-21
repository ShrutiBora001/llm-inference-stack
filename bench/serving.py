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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lis.metrics import SLO, RequestRecord, summarize  # noqa: E402
from lis.workload import WorkloadSpec, generate, sweep  # noqa: E402

# Runnable today. The crossover finding -- where SGLang's RadixAttention
# overtakes vLLM as prefixes become more shared -- needs no context parallelism
# at all, so it does not wait on the CUDA port.
FRAMEWORKS = (
    "sglang",           # RadixAttention on
    "sglang-no-cache",  # the A/B that isolates what RadixAttention contributes
    "vllm",             # vLLM's own prefix caching
    "vllm-dcp",         # vLLM's shipped CP -- decode-only, scope it honestly
)

# Blocked on the SGLang CUDA port, which is gated on a maintainer reply.
# Listed so `--frameworks` accepts them and the refusal explains itself, rather
# than reporting them as a typo.
GATED_FRAMEWORKS = ("sglang-zigzag", "sglang-contiguous")

# 0 to 1 in steps of 0.05. Wider than it looks necessary on purpose: each extra
# point costs about 40 seconds of measurement now, while coming back for it
# later costs a fresh 40-minute install. The crossover location is the headline
# number, and a coarse sweep can straddle it -- interpolating across a 0.25-wide
# gap reports a point that was never near a measurement.
PREFIX_SHARES = tuple(round(i * 0.05, 2) for i in range(21))


def build_client(framework: str):
    """Resolve a framework name to a request driver.

    Deliberately raises rather than falling back to a simulation. A serving
    number produced without a serving engine is not a measurement, and the
    distinction is invisible once it reaches a plot.
    """
    if framework not in FRAMEWORKS + GATED_FRAMEWORKS:
        raise ValueError(
            f"unknown framework {framework!r}; expected one of "
            f"{FRAMEWORKS + GATED_FRAMEWORKS}"
        )

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
    """One (framework, prefix_share) cell.

    The run duration comes from the records themselves, not from a wall clock
    around the call: `summarize` measures first-arrival to last-token, which
    excludes engine startup and teardown. Timing the call instead would fold
    warmup into throughput and make a slow-loading engine look like a slow one.
    """
    workload = generate(spec)
    records: list[RequestRecord] = list(client.send(workload))
    summary = summarize(records, slo=slo)

    row = summary.as_dict()
    # Flatten the percentile dicts: one JSON row per run, one key per number, so
    # a figure can name exactly what it reads. Nested dicts would force every
    # consumer to know the shape.
    for name in ("ttft_ms", "tpot_ms"):
        for pct, value in row.pop(name).items():
            row[f"{name.split('_')[0]}_{pct}_ms"] = value

    row.update({
        "framework": client.name,
        "prefix_share_requested": spec.prefix_share,
        "prefix_share_realized": workload.realized_prefix_share(),
        "workload_source": workload.source,
        # The independent variables. Without these in the row, a figure plotting
        # against context length or concurrency would have to hardcode them,
        # which is exactly what design rule 4 forbids -- and a hardcoded x-axis
        # silently mislabels every point if the sweep parameters ever change.
        "context_len": spec.prompt_tokens,
        "output_len": spec.output_tokens,
        "concurrency": spec.n_requests,
        "qps": spec.qps,
        # The thresholds the SLO bars are drawn against. Recorded per row so a
        # results file always carries the policy its attainment was judged by;
        # otherwise a later SLO change would silently redraw old data.
        "slo_ttft_ms": slo.ttft_p99_ms,
        "slo_tpot_ms": slo.tpot_p99_ms,
        # Peak device memory, when the binding can report it. Optional because
        # not every engine exposes it, and a missing value must read as absent
        # rather than as zero.
        "peak_memory_bytes": _peak_memory(client),
    })
    return row


def _peak_memory(client) -> float | None:
    """Peak device memory for the run, if the binding can report it.

    Returns None rather than 0 when unavailable: zero is a measurement, absence
    is not, and S5 must not plot a flat line at the origin for an engine that
    simply did not tell us.
    """
    hook = getattr(client, "peak_memory_bytes", None)
    if hook is None:
        return None
    try:
        return float(hook())
    except Exception:
        return None


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
    p.add_argument("--frameworks", nargs="+", default=["sglang", "vllm"],
                   choices=FRAMEWORKS + GATED_FRAMEWORKS,
                   help="default runs the ungated crossover comparison")
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
                      f"TTFT p99={row.get('ttft_p99_ms', float('nan')):.0f} ms  "
                      f"cache {row['prefix_cache_hit_rate']:.0%}  "
                      f"SLO {row['slo_attainment']:.0%}")

    print(f"\nappended {len(rows)} rows to {args.out}")
    if rows:
        best = max(rows, key=lambda r: r["goodput_rps"])
        print(f"best goodput: {best['goodput_rps']:.2f} req/s "
              f"({best['framework']} @ prefix_share={best['prefix_share_realized']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
