#!/usr/bin/env python3
"""Static figures, regenerated from results/*.jsonl.

    python report/make_figures.py

Rule inherited from upstream and worth restating: **no number is hardcoded in a
figure.** Everything derives from the committed results files through the same
functions the reports and the dashboard use, so a fabricated table disagrees
with its own figure. `tests/test_figures.py` enforces that every figure declares
the file and keys it reads, and that they exist.

Figures whose inputs are missing are skipped with a printed reason rather than
drawn from placeholder data -- a plot of nothing is worse than no plot.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lis.metrics import MFU_GATE  # noqa: E402

RESULTS = ROOT / "results"
FIGDIR = ROOT / "report" / "figures"

# Same palette as the upstream report, so figures from both repos read as one
# body of work.
C_PROBLEM = "#c44e52"   # red   — the baseline / the problem
C_FIX = "#4c72b0"       # blue  — the improvement
C_ACCENT = "#55a868"    # green — reference lines and gates
C_THIRD = "#dd8452"     # orange — a third series

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

# A100 bf16 dense. Quoted dense, never the sparsity-doubled marketing number:
# attention is dense, so the sparse figure would halve the apparent shortfall.
FLASH_ATTENTION_BAND = (120, 180)


@dataclass(frozen=True)
class FigureSpec:
    """What a figure needs, declared so tests can verify it without drawing."""

    name: str
    source: str          # results file it reads
    keys: tuple[str, ...]  # record keys it depends on
    describe: str


SPECS: dict[str, FigureSpec] = {
    "K1_mfu": FigureSpec(
        "K1_mfu", "kernel.jsonl",
        ("backend", "seq", "tflops", "mfu", "peak_tflops"),
        "MFU vs sequence length per backend, against the gate",
    ),
    "K2_roofline": FigureSpec(
        "K2_roofline", "kernel.jsonl",
        ("backend", "arithmetic_intensity", "tflops", "peak_tflops"),
        "Roofline: is the kernel memory-bound or compute-bound",
    ),
    "S1_ttft": FigureSpec(
        "S1_ttft", "serving.jsonl",
        ("framework", "context_len", "ttft_p99_ms", "slo_ttft_ms"),
        "TTFT vs context length, against the SLO",
    ),
    "S2_goodput": FigureSpec(
        "S2_goodput", "serving.jsonl",
        ("framework", "concurrency", "throughput_rps", "goodput_rps"),
        "Throughput AND goodput vs concurrency -- throughput alone misleads",
    ),
    "S3_percentiles": FigureSpec(
        "S3_percentiles", "serving.jsonl",
        ("framework", "ttft_p50_ms", "ttft_p99_ms", "tpot_p99_ms", "slo_tpot_ms"),
        "Latency percentiles against the SLO lines",
    ),
    "S4_crossover": FigureSpec(
        "S4_crossover", "serving.jsonl",
        ("framework", "prefix_share_realized", "throughput_rps",
         "prefix_cache_hit_rate"),
        "THE HEADLINE: where RadixAttention overtakes vLLM as prefixes are shared",
    ),
    "S5_memory": FigureSpec(
        "S5_memory", "serving.jsonl",
        ("framework", "context_len", "peak_memory_bytes"),
        "Peak KV memory vs context -- the memory argument",
    ),
}

# Framework colours. Held in one place so a framework is the same colour in
# every figure and in the dashboard; a reader should never have to re-learn the
# legend between plots.
FRAMEWORK_COLORS = {
    "sglang": C_FIX,
    "sglang-no-cache": C_PROBLEM,
    "sglang-zigzag": C_ACCENT,
    "sglang-contiguous": C_PROBLEM,
    "vllm": C_THIRD,
    "vllm-dcp": "#8172b3",
}


def fcolor(name: str) -> str:
    return FRAMEWORK_COLORS.get(name, "gray")


def by_framework(rows: list[dict], xkey: str) -> dict[str, list[dict]]:
    """Group and sort by the x variable, dropping rows missing either field.

    Dropping is deliberate: a row without the x value cannot be placed, and
    substituting a default would put a real measurement at a fictional
    coordinate.
    """
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r.get(xkey) is None or r.get("framework") is None:
            continue
        out.setdefault(r["framework"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r[xkey])
    return out


def crossing_point(a: list[tuple[float, float]], b: list[tuple[float, float]]):
    """Where series `a` overtakes series `b`, by linear interpolation.

    Returns (x, y) or None. Only the *first* sign change is reported: two curves
    that weave across each other have no single crossover, and inventing one
    would be the whole finding fabricated from noise.
    """
    xs = sorted({x for x, _ in a} & {x for x, _ in b})
    if len(xs) < 2:
        return None
    da, db = dict(a), dict(b)
    diffs = [(x, da[x] - db[x]) for x in xs]
    for (x0, d0), (x1, d1) in zip(diffs, diffs[1:]):
        if d0 == 0:
            return x0, da[x0]
        if (d0 < 0) != (d1 < 0):
            t = -d0 / (d1 - d0)
            x = x0 + t * (x1 - x0)
            return x, da[x0] + t * (da[x1] - da[x0])
    return None


def load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _series(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seq"])
    return out


def fig_k1_mfu(rows: list[dict]) -> Path | None:
    """Is the kernel competitive? The gate line makes the answer unambiguous."""
    if not rows:
        return None
    peak = next((r["peak_tflops"] for r in rows if r.get("peak_tflops")), None)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"unfused": C_PROBLEM, "torch_flash": C_FIX, "triton": C_THIRD}
    for backend, rs in _series(rows, "backend").items():
        ax.plot([r["seq"] for r in rs], [r["mfu"] * 100 for r in rs],
                "o-", color=colors.get(backend, "gray"), label=backend)

    ax.axhline(MFU_GATE * 100, ls="--", lw=1.2, color=C_ACCENT)
    ax.text(rows[0]["seq"], MFU_GATE * 100 * 1.05,
            f"gate {MFU_GATE:.0%}", fontsize=7.5, color=C_ACCENT)
    if peak:
        lo, hi = (b / peak * 100 for b in FLASH_ATTENTION_BAND)
        ax.axhspan(lo, hi, color="gray", alpha=0.15)
        ax.text(rows[0]["seq"], (lo + hi) / 2, "FlashAttention", fontsize=7.5, color="gray")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("MFU (% of dense peak)")
    ax.set_title("K1 — achieved utilization per backend", loc="left")
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "K1_mfu.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_k2_roofline(rows: list[dict]) -> Path | None:
    """Why the number is what it is.

    A kernel that materializes its score tile has low arithmetic intensity and
    lands on the bandwidth slope; fusing moves it right, toward the compute
    ceiling. This is the difference between reporting 3% and explaining it.
    """
    if not rows:
        return None
    peak = next((r["peak_tflops"] for r in rows if r.get("peak_tflops")), None) or 312.0
    hbm_tb_s = 1.55  # A100 80GB-class HBM; annotated on the plot as an assumption

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [2 ** i for i in range(-2, 12)]
    ax.plot(xs, [min(peak, hbm_tb_s * 1e3 * x / 1e3) for x in xs],
            "-", color="black", lw=1.4, label=f"roofline (peak {peak:.0f} TF/s)")

    colors = {"unfused": C_PROBLEM, "torch_flash": C_FIX, "triton": C_THIRD}
    for backend, rs in _series(rows, "backend").items():
        ax.scatter([r["arithmetic_intensity"] for r in rs], [r["tflops"] for r in rs],
                   color=colors.get(backend, "gray"), label=backend, s=40, zorder=3)

    ridge = peak / (hbm_tb_s * 1e3) * 1e3
    ax.axvline(ridge, ls=":", lw=1, color=C_ACCENT)
    ax.text(ridge * 1.1, peak * 0.3, "ridge point\nleft = memory-bound",
            fontsize=7.5, color=C_ACCENT)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"K2 — roofline (HBM assumed {hbm_tb_s} TB/s)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGDIR / "K2_roofline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out




# ------------------------------------------------------------- Phase 2 serving


def fig_s1_ttft(rows: list[dict]) -> Path | None:
    """Does context parallelism help where it is supposed to?

    TTFT is prefill latency, so it is where a sequence-parallel prefill should
    show up and where a decode-only scheme cannot.
    """
    series = by_framework(rows, "context_len")
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, rs in series.items():
        ax.plot([r["context_len"] for r in rs], [r["ttft_p99_ms"] for r in rs],
                "o-", color=fcolor(name), label=name)

    slos = {r["slo_ttft_ms"] for r in rows if r.get("slo_ttft_ms")}
    if len(slos) == 1:
        slo = slos.pop()
        ax.axhline(slo, ls="--", lw=1.2, color=C_ACCENT)
        ax.text(min(r["context_len"] for r in rows), slo * 1.05,
                f"SLO {slo:.0f} ms", fontsize=7.5, color=C_ACCENT)
    elif len(slos) > 1:
        # Rows judged under different policies cannot share one line. Say so
        # rather than drawing whichever happened to come first.
        ax.set_title("S1 — TTFT p99 (mixed SLOs in data; no line drawn)", loc="left")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("TTFT p99 (ms)")
    if len(slos) <= 1:
        ax.set_title("S1 — time to first token vs context", loc="left")
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "S1_ttft.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_s2_goodput(rows: list[dict]) -> Path | None:
    """Throughput and goodput on the same axes.

    The gap between the two lines is the point of the figure: it is requests the
    system served but served too slowly to count. A throughput curve alone hides
    that entirely, and hides it *most* exactly where load is highest.
    """
    series = by_framework(rows, "concurrency")
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, rs in series.items():
        x = [r["concurrency"] for r in rs]
        ax.plot(x, [r["throughput_rps"] for r in rs], "o--", color=fcolor(name),
                alpha=0.45, label=f"{name} throughput")
        ax.plot(x, [r["goodput_rps"] for r in rs], "o-", color=fcolor(name),
                label=f"{name} goodput")

    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("requests / s")
    ax.set_title("S2 — goodput is the solid line; the gap is served-but-too-slow",
                 loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = FIGDIR / "S2_goodput.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_s3_percentiles(rows: list[dict]) -> Path | None:
    """Percentile bars against the SLO lines.

    Means are useless for latency: a system can post a fine average while a
    quarter of requests time out. p99 is what an SLO is written against, so it
    is what gets drawn.
    """
    frameworks = sorted({r["framework"] for r in rows if r.get("framework")})
    if not frameworks:
        return None

    def best(name, key):
        vals = [r[key] for r in rows if r["framework"] == name and r.get(key) is not None]
        return max(vals) if vals else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8))
    width, xs = 0.35, range(len(frameworks))

    for ax, metric, label in ((ax1, "ttft", "TTFT"), (ax2, "tpot", "TPOT")):
        p50 = [best(f, f"{metric}_p50_ms") for f in frameworks]
        p99 = [best(f, f"{metric}_p99_ms") for f in frameworks]
        ax.bar([x - width / 2 for x in xs], p50, width, label="p50",
               color=[fcolor(f) for f in frameworks], alpha=0.45)
        ax.bar([x + width / 2 for x in xs], p99, width, label="p99",
               color=[fcolor(f) for f in frameworks])

        slos = {r[f"slo_{metric}_ms"] for r in rows if r.get(f"slo_{metric}_ms")}
        if len(slos) == 1:
            slo = slos.pop()
            ax.axhline(slo, ls="--", lw=1.2, color=C_ACCENT)
            ax.text(-0.4, slo * 1.03, f"SLO {slo:.0f} ms", fontsize=7,
                    color=C_ACCENT)

        ax.set_xticks(list(xs))
        ax.set_xticklabels(frameworks, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"{label} (ms)")
        ax.set_title(f"S3 — {label} p50 vs p99", loc="left", fontsize=10)
        ax.legend(fontsize=7.5)

    fig.tight_layout()
    out = FIGDIR / "S3_percentiles.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_s4_crossover(rows: list[dict]) -> Path | None:
    """The headline.

    Throughput against how much of each prompt is shared. At `prefix_share` 0
    there is nothing to cache and RadixAttention cannot help; as sharing rises
    it should overtake. The crossover point is the finding, and it is annotated
    from the data rather than asserted.

    The x-axis is the *realized* prefix share, not the requested one: a sampler
    that quietly ignored the request would otherwise produce a confident curve
    against a fictional axis.
    """
    series = by_framework(rows, "prefix_share_realized")
    if not series:
        return None

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})

    curves = {}
    for name, rs in series.items():
        x = [r["prefix_share_realized"] for r in rs]
        y = [r["throughput_rps"] for r in rs]
        curves[name] = list(zip(x, y))
        ax.plot(x, y, "o-", color=fcolor(name), label=name)

    # Annotate the first crossing between an SGLang curve and a vLLM one.
    sg = next((n for n in curves if n.startswith("sglang")), None)
    vl = next((n for n in curves if n.startswith("vllm")), None)
    if sg and vl:
        pt = crossing_point(curves[sg], curves[vl])
        if pt:
            ax.plot(*pt, "o", ms=11, mfc="none", mec="black", mew=1.6)
            ax.annotate(f"crossover\nprefix_share={pt[0]:.2f}", pt,
                        textcoords="offset points", xytext=(12, -22), fontsize=8,
                        arrowprops=dict(arrowstyle="->", lw=0.9))
        else:
            ax.text(0.02, 0.94, "no single crossover in this range",
                    transform=ax.transAxes, fontsize=8, color=C_PROBLEM)

    ax.set_ylabel("throughput (req/s)")
    ax.set_title("S4 — where prefix caching starts paying", loc="left")
    ax.legend(fontsize=8)

    for name, rs in series.items():
        ax2.plot([r["prefix_share_realized"] for r in rs],
                 [r["prefix_cache_hit_rate"] * 100 for r in rs],
                 "o-", color=fcolor(name))
    ax2.set_xlabel("realized prefix share")
    ax2.set_ylabel("cache hit (%)")
    ax2.set_ylim(-5, 105)

    fig.tight_layout()
    out = FIGDIR / "S4_crossover.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_s5_memory(rows: list[dict]) -> Path | None:
    """Peak device memory vs context — the memory argument, measured.

    Rows whose engine did not report memory are absent, not zero. Plotting a
    missing value as zero would draw a flat line at the origin and read as an
    engine that used no memory.
    """
    usable = [r for r in rows if r.get("peak_memory_bytes")]
    if not usable:
        return None

    series = by_framework(usable, "context_len")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, rs in series.items():
        ax.plot([r["context_len"] for r in rs],
                [r["peak_memory_bytes"] / 2**30 for r in rs],
                "o-", color=fcolor(name), label=name)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("peak device memory (GiB)")
    ax.set_title("S5 — peak memory vs context", loc="left")
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "S5_memory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


BUILDERS = {
    "K1_mfu": fig_k1_mfu,
    "K2_roofline": fig_k2_roofline,
    "S1_ttft": fig_s1_ttft,
    "S2_goodput": fig_s2_goodput,
    "S3_percentiles": fig_s3_percentiles,
    "S4_crossover": fig_s4_crossover,
    "S5_memory": fig_s5_memory,
}


def build_all(verbose: bool = True) -> list[Path]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    for name, spec in SPECS.items():
        rows = load(spec.source)
        if not rows:
            if verbose:
                print(f"  skip {name}: no {spec.source} — {spec.describe}")
            continue
        path = BUILDERS[name](rows)
        if path:
            made.append(path)
            if verbose:
                print(f"  wrote {path.relative_to(ROOT)}")
    return made


def main() -> None:
    print("figures:")
    made = build_all()
    if not made:
        print("\nnothing drawn. Run bench/kernel.py first, or copy results/ "
              "back from a GPU box.")
    else:
        print(f"\n{len(made)} figure(s)")


if __name__ == "__main__":
    main()
