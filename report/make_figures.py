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
}


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


BUILDERS = {"K1_mfu": fig_k1_mfu, "K2_roofline": fig_k2_roofline}


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
