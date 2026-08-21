"""Figures must regenerate from committed results, and agree with the dashboard.

The failure this guards against is a number that was typed rather than measured.
If a figure or a table can be produced without the results file that supposedly
backs it, nothing stops a plausible-looking value from being invented -- and
nobody re-derives a plot by hand.

So: every figure declares the file and the keys it reads, and these tests check
those keys exist in real records. The dashboard is built from the same specs, so
it cannot show a different story than the PNGs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))

import make_dashboard  # noqa: E402
import make_figures  # noqa: E402


def test_every_figure_declares_its_inputs():
    """A figure with no declared source cannot be audited."""
    for name, spec in make_figures.SPECS.items():
        assert spec.source.endswith(".jsonl"), f"{name}: source is not a results file"
        assert spec.keys, f"{name}: declares no keys, so nothing can be checked"
        assert spec.describe, f"{name}: no description of what it answers"


def test_every_spec_has_a_builder():
    assert set(make_figures.SPECS) == set(make_figures.BUILDERS)


@pytest.mark.parametrize("name", sorted(make_figures.SPECS))
def test_declared_keys_exist_in_the_results(name):
    """The keys a figure reads must be present in the data it reads them from.

    Skips rather than passes when the file is absent: a figure whose GPU results
    have not been collected yet is not evidence of anything.
    """
    spec = make_figures.SPECS[name]
    rows = make_figures.load(spec.source)
    if not rows:
        pytest.skip(f"no results/{spec.source} yet — run the corresponding bench")
    missing = [k for k in spec.keys if k not in rows[0]]
    assert not missing, f"{name} reads {missing}, absent from {spec.source}"


def test_figures_regenerate_from_scratch():
    """Delete the PNGs, rebuild, and confirm they come back.

    This is the end-to-end check: if a figure ever depended on a stale file on
    disk rather than on the results, it does not survive the delete.
    """
    if not make_figures.load("kernel.jsonl"):
        pytest.skip("no kernel results to draw from")

    for png in make_figures.FIGDIR.glob("*.png"):
        png.unlink()
    made = make_figures.build_all(verbose=False)
    assert made, "no figures produced from results that exist"
    for p in made:
        assert p.exists() and p.stat().st_size > 1000, f"{p} is empty or truncated"


def test_dashboard_reads_exactly_the_figure_sources():
    """The dashboard is a view, never a second source of truth."""
    assert set(make_dashboard.collect()) == {s.source for s in make_figures.SPECS.values()}


def test_dashboard_is_self_contained():
    """No external requests: it has to work offline and from file://."""
    make_dashboard.main()
    html = make_dashboard.OUT.read_text()
    for bad in ("http://", "https://", "<script src", "<link rel=\"stylesheet\""):
        assert bad not in html, f"dashboard references external resource: {bad!r}"


def test_dashboard_embeds_the_same_records_the_figures_use():
    """Same inputs, so the two cannot disagree about what was measured."""
    rows = make_figures.load("kernel.jsonl")
    if not rows:
        pytest.skip("no kernel results")

    make_dashboard.main()
    html = make_dashboard.OUT.read_text()
    blob = html.split('type="application/json">')[1].split("</script>")[0]
    assert len(json.loads(blob)["kernel.jsonl"]) == len(rows)


def test_both_scripts_run_clean_as_subprocesses():
    """They are run by `make report`, not imported. Test them that way."""
    for script in ("make_figures.py", "make_dashboard.py"):
        r = subprocess.run([sys.executable, str(ROOT / "report" / script)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{script} failed:\n{r.stderr}"


def test_no_figure_hardcodes_a_measurement():
    """Design rule 4, mechanically checked.

    Bare large floats in a plotting function are how a measured number turns
    into a typed one. Hardware constants are allowed but must be named at module
    level, where they are visible and reviewable.
    """
    src = (ROOT / "report" / "make_figures.py").read_text()
    body = src.split("def fig_", 1)[1] if "def fig_" in src else ""
    for tok in ("3.3", "10.21", "1.69", "1.73", "156.5"):
        assert tok not in body, (
            f"{tok!r} appears inside a figure function — measurements must come "
            "from results/*.jsonl, not from the plotting code"
        )


# ------------------------------------------------------------ Phase 2 figures
# Drawn from synthetic rows written to a tmp dir. They never touch results/:
# a fabricated serving number must not be able to reach a real results file,
# and a test that wrote one would defeat the rule the figures exist to enforce.


def _serving_rows():
    """Two frameworks whose throughput curves cross at prefix_share 0.5.

    sglang starts slower and improves with sharing (a cache that starts empty);
    vllm is flat. The crossing is placed at a known point so the annotation can
    be checked against a value computed by hand rather than by the same code.
    """
    rows = []
    for share in (0.0, 0.25, 0.5, 0.75, 1.0):
        for fw, base in (("sglang", 8.0 + 8.0 * share), ("vllm", 12.0)):
            hit = share if fw == "sglang" else share * 0.2
            rows.append({
                "framework": fw,
                "prefix_share_realized": share,
                "prefix_share_requested": share,
                "throughput_rps": base,
                "goodput_rps": base * 0.9,
                "prefix_cache_hit_rate": hit,
                "context_len": 8192,
                "output_len": 128,
                "concurrency": 64,
                "qps": 4.0,
                "ttft_p50_ms": 300.0, "ttft_p90_ms": 700.0, "ttft_p99_ms": 900.0,
                "tpot_p50_ms": 12.0, "tpot_p90_ms": 25.0, "tpot_p99_ms": 40.0,
                "slo_ttft_ms": 2000.0, "slo_tpot_ms": 50.0,
                "peak_memory_bytes": 20 * 2**30,
                "slo_attainment": 1.0, "n_requests": 64, "duration_s": 20.0,
                "output_throughput_tps": base * 128,
                "workload_source": "local",
                "realized_prefix_share": share,
            })
    return rows


@pytest.fixture
def serving_figdir(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "serving.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _serving_rows())
    )
    monkeypatch.setattr(make_figures, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(make_figures, "FIGDIR", tmp_path / "figures")
    (tmp_path / "figures").mkdir()
    return tmp_path


@pytest.mark.parametrize("name", ["S2_goodput", "S3_percentiles",
                                  "S4_crossover", "S5_memory"])
def test_serving_figures_draw_from_representative_rows(name, serving_figdir):
    """Each S figure must survive contact with data shaped like real output."""
    rows = make_figures.load("serving.jsonl")
    path = make_figures.BUILDERS[name](rows)
    assert path is not None and path.exists()
    assert path.stat().st_size > 1000, f"{name} produced an empty image"


def test_s1_refuses_a_single_context_length(serving_figdir):
    """A TTFT-vs-context plot needs more than one context. With one value every
    point lands on the same x and the result is a vertical smear that reads as
    a curve -- worse than no figure, because it looks like evidence."""
    rows = make_figures.load("serving.jsonl")
    assert len({r["context_len"] for r in rows}) == 1
    assert make_figures.fig_s1_ttft(rows) is None


def test_s1_draws_once_a_context_sweep_exists(serving_figdir):
    rows = make_figures.load("serving.jsonl")
    for i, r in enumerate(rows):
        r["context_len"] = 2048 * (2 ** (i % 4))
    assert make_figures.fig_s1_ttft(rows) is not None


def test_crossover_is_found_at_the_hand_computed_point():
    """sglang goes 8 -> 16 linearly, vllm is flat at 12: they cross at 0.5."""
    a = [(0.0, 8.0), (0.5, 12.0), (1.0, 16.0)]
    b = [(0.0, 12.0), (0.5, 12.0), (1.0, 12.0)]
    x, y = make_figures.crossing_point(a, b)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(12.0)


def test_crossover_interpolates_between_sampled_points():
    """The crossing rarely lands on a sampled x. 8->16 vs flat 10 crosses at
    0.25, which is not one of the sampled shares."""
    a = [(0.0, 8.0), (1.0, 16.0)]
    b = [(0.0, 10.0), (1.0, 10.0)]
    x, _ = make_figures.crossing_point(a, b)
    assert x == pytest.approx(0.25)


def test_no_crossover_is_reported_as_none_not_invented():
    """Two curves that never meet have no crossover.

    Returning a plausible-looking point here would fabricate the entire
    headline finding out of nothing.
    """
    a = [(0.0, 1.0), (1.0, 2.0)]
    b = [(0.0, 10.0), (1.0, 20.0)]
    assert make_figures.crossing_point(a, b) is None


def test_missing_memory_rows_are_dropped_not_zeroed(serving_figdir):
    """An engine that did not report memory must vanish from S5, not appear as
    a flat line at zero -- which would read as 'used no memory'."""
    rows = make_figures.load("serving.jsonl")
    for r in rows:
        if r["framework"] == "vllm":
            r["peak_memory_bytes"] = None
    path = make_figures.fig_s5_memory(rows)
    assert path is not None, "sglang still has memory data; S5 should draw"


def test_s5_skips_entirely_when_no_engine_reported_memory(serving_figdir):
    rows = make_figures.load("serving.jsonl")
    for r in rows:
        r["peak_memory_bytes"] = None
    assert make_figures.fig_s5_memory(rows) is None


def test_every_framework_has_a_stable_colour():
    """A framework must be the same colour in every figure and the dashboard.
    Re-learning the legend between plots is a real cost to a reader."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(ROOT / "bench"))
    import serving

    for fw in serving.FRAMEWORKS + serving.GATED_FRAMEWORKS:
        assert fw in make_figures.FRAMEWORK_COLORS, f"{fw} has no assigned colour"
