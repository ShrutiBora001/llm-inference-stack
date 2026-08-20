"""The serving harness, minus the engine.

What is checkable on a laptop is the part most likely to be quietly wrong: that
the sweep produces the prefix shares it claims, and that the harness refuses to
produce numbers when no engine is present rather than falling back to something
that looks like a measurement.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import serving  # noqa: E402


class Args:
    requests, input_len, output_len, qps, seed = 8, 512, 16, 4.0, 1234
    prefix_shares = list(serving.PREFIX_SHARES)


def test_sweep_covers_every_requested_share():
    assert [s.prefix_share for s in serving.specs_for(Args())] == Args.prefix_shares


def test_realized_share_matches_the_requested_one():
    """If it did not, the crossover plot's x-axis would be fiction."""
    from lis.workload import generate

    for spec in serving.specs_for(Args()):
        w = generate(spec)
        assert w.realized_prefix_share() == pytest.approx(spec.prefix_share, abs=0.02)


def test_prompt_length_is_constant_across_the_sweep():
    """Prefix share must be the *only* thing changing.

    If total prompt length moved with it, the curve would confound cache hits
    with prefill cost and the finding would be uninterpretable.
    """
    from lis.workload import generate

    totals = {sum(r.prompt_len for r in generate(s).requests)
              for s in serving.specs_for(Args())}
    assert len(totals) == 1, f"prompt token count varies across the sweep: {totals}"


@pytest.mark.parametrize("framework", serving.FRAMEWORKS)
def test_missing_engine_refuses_rather_than_simulating(framework):
    with pytest.raises(NotImplementedError, match="refuses to simulate"):
        serving.build_client(framework)


def test_unknown_framework_is_rejected():
    with pytest.raises(ValueError, match="unknown framework"):
        serving.build_client("tensorrt-llm")


def test_dry_run_exits_clean_and_contacts_nothing():
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench" / "serving.py"), "--dry-run",
         "--sweep-prefix-share", "--requests", "4", "--input-len", "256"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert "no engine contacted" in r.stdout
    assert "goodput" not in r.stdout, "dry run emitted a serving metric"
