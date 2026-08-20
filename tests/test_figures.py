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
