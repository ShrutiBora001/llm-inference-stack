"""Metric definitions, checked against hand-computed values.

These are arithmetic, so the tests are hand-computed rather than
property-based: if `mfu` is ever silently redefined, a number in a report moves
and nothing else complains. Pinning the arithmetic is the only defence.

The goodput tests matter most. Goodput is the headline serving number precisely
because it cannot be gamed by serving many requests badly, and a bug that let
SLO-violating requests count would destroy that property while still producing
plausible-looking output.
"""

from __future__ import annotations

import math

import pytest

from lis.metrics import (
    DEFAULT_SLO,
    MFU_GATE,
    RANK_SPREAD_GATE,
    RequestRecord,
    SLO,
    achieved_tflops,
    arithmetic_intensity,
    comm_fraction,
    comm_seconds,
    memory_model_accuracy,
    mfu,
    overlap_recovery,
    percentile,
    rank_spread,
    summarize,
)

# ------------------------------------------------------------------- kernel


def test_achieved_tflops_hand_computed():
    # 2.47e12 FLOP in 242 ms -> 10.2 TFLOP/s. The upstream S=32768 striped run.
    assert achieved_tflops(2.47e12, 0.242) == pytest.approx(10.21, rel=1e-2)


def test_mfu_reproduces_the_upstream_number():
    """The measured 3.3% that gates the whole project."""
    assert mfu(10.21, 312.0) == pytest.approx(0.0327, rel=1e-2)


def test_mfu_gate_is_far_above_the_unfused_baseline():
    """Sanity on the gate itself: it must demand a real improvement."""
    unfused = mfu(10.21, 312.0)
    assert MFU_GATE > unfused * 5, "gate is too close to the baseline to mean anything"


def test_arithmetic_intensity_falls_when_scores_hit_hbm():
    """The roofline explanation for 3% MFU.

    Same FLOPs, but an unfused kernel writes the score tile to HBM, so bytes
    moved rise by an order of magnitude and intensity collapses.
    """
    flops = 2.47e12
    fused = arithmetic_intensity(flops, 5e9)
    unfused = arithmetic_intensity(flops, 90e9)   # ~90 GiB measured upstream
    assert unfused < fused / 10


@pytest.mark.parametrize("bad", [0, -1])
def test_zero_time_rejected(bad):
    with pytest.raises(ValueError):
        achieved_tflops(1e12, bad)


# -------------------------------------------------------------- distributed


def test_comm_seconds_uses_measured_bandwidth():
    """384 MiB at the measured 156.5 GB/s -> 2.57 ms.

    At the 300 GB/s nameplate it would read 1.34 ms, understating comms by 2x.
    That mistake was made once in the upstream report.
    """
    ms = comm_seconds(384 * 2**20, 156.5) * 1000
    assert ms == pytest.approx(2.57, rel=0.02)


def test_comm_fraction_matches_the_reported_share():
    assert comm_fraction(0.00257, 0.242) == pytest.approx(0.0106, rel=0.05)


def test_overlap_recovery_is_a_fraction_of_comm_not_of_runtime():
    """243.16 -> 241.78 ms saved 1.38 ms of a 2.57 ms comm cost = 54%."""
    assert overlap_recovery(243.16, 241.78, 2.57) == pytest.approx(0.537, rel=0.02)


def test_rank_spread_detects_the_contiguous_staircase():
    """Upstream P=4 measured compute: 9.5 / 17.5 / 25.6 / 33.1 ms."""
    contiguous = rank_spread([9.5, 17.5, 25.6, 33.1])
    striped = rank_spread([24.2, 23.8, 23.9, 23.8])
    assert contiguous < 0.3
    assert striped > RANK_SPREAD_GATE


def test_rank_spread_perfect_balance():
    assert rank_spread([10.0] * 4) == 1.0


# -------------------------------------------------------------------- memory


def test_memory_model_accuracy_direction():
    """<= 1.0 is safe (over-predicted); > 1.0 would have OOMed unexpectedly."""
    assert memory_model_accuracy(32.31, 41.60) == pytest.approx(0.777, rel=0.01)
    assert memory_model_accuracy(41.60, 32.31) > 1.0


# ---------------------------------------------------------------- percentile


def test_percentile_interpolates():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 1.0) == 4.0
    assert percentile(vals, 0.5) == 2.5


def test_percentile_single_value():
    assert percentile([7.0], 0.99) == 7.0


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


# ------------------------------------------------------------------- serving


def rec(arrival, first, last, prompt=100, output=10, cached=0):
    return RequestRecord(arrival, first, last, prompt, output, cached)


def test_ttft_and_tpot_definitions():
    r = rec(arrival=0.0, first=0.5, last=1.4, output=10)
    assert r.ttft_ms == pytest.approx(500.0)
    # 900 ms spread over 9 gaps, not 10 -- the first token's cost is TTFT
    assert r.tpot_ms == pytest.approx(100.0)


def test_single_token_response_has_no_tpot():
    assert rec(0.0, 0.5, 0.5, output=1).tpot_ms == 0.0


def test_out_of_order_timeline_rejected():
    with pytest.raises(ValueError, match="out of order"):
        RequestRecord(1.0, 0.5, 2.0, 100, 10)


def test_goodput_excludes_slo_violators():
    """The property that makes goodput worth reporting.

    Four requests finish in 1 s, so throughput is 4 req/s. Two blew the TTFT
    target. Goodput must be 2 req/s -- a system serving many requests badly
    should not look good.
    """
    slo = SLO(ttft_p99_ms=600.0, tpot_p99_ms=200.0)
    records = [
        rec(0.0, 0.5, 1.0, output=10),   # ttft 500 ok
        rec(0.0, 0.5, 1.0, output=10),   # ok
        rec(0.0, 0.9, 1.0, output=10),   # ttft 900 -> violation
        rec(0.0, 0.95, 1.0, output=10),  # violation
    ]
    m = summarize(records, slo)
    assert m.throughput_rps == pytest.approx(4.0)
    assert m.goodput_rps == pytest.approx(2.0)
    assert m.slo_attainment == pytest.approx(0.5)


def test_goodput_equals_throughput_when_all_pass():
    # 0.4 s of streaming over 9 gaps -> 44 ms TPOT, inside the 50 ms default.
    # Written explicitly because the obvious 1.0 s here fails the default SLO,
    # which is a fair reminder that the default is strict.
    records = [rec(0.0, 0.1, 0.5, output=10) for _ in range(4)]
    m = summarize(records, DEFAULT_SLO)
    assert m.goodput_rps == pytest.approx(m.throughput_rps)
    assert m.slo_attainment == 1.0


def test_tpot_violation_alone_fails_the_slo():
    """Both targets must be met, not either."""
    slo = SLO(ttft_p99_ms=1000.0, tpot_p99_ms=50.0)
    fast_start_slow_stream = rec(0.0, 0.1, 5.0, output=10)  # tpot ~544 ms
    assert not fast_start_slow_stream.meets(slo)
    assert summarize([fast_start_slow_stream], slo).goodput_rps == 0.0


def test_prefix_cache_hit_rate():
    records = [rec(0.0, 0.1, 1.0, prompt=100, cached=60),
               rec(0.0, 0.1, 1.0, prompt=100, cached=40)]
    assert summarize(records).prefix_cache_hit_rate == pytest.approx(0.5)


def test_output_throughput_counts_tokens_not_requests():
    records = [rec(0.0, 0.1, 1.0, output=50) for _ in range(2)]
    m = summarize(records)
    assert m.output_throughput_tps == pytest.approx(100.0)
    assert m.throughput_rps == pytest.approx(2.0)


def test_percentiles_present_and_ordered():
    records = [rec(0.0, 0.1 * i, 1.0 + i, output=10) for i in range(1, 21)]
    m = summarize(records)
    assert m.ttft_ms["p50"] <= m.ttft_ms["p90"] <= m.ttft_ms["p99"]


def test_empty_run_rejected():
    with pytest.raises(ValueError, match="no records"):
        summarize([])


def test_metrics_serialize_for_results_files():
    m = summarize([rec(0.0, 0.1, 1.0, output=10)])
    d = m.as_dict()
    assert set(d) >= {"goodput_rps", "throughput_rps", "slo_attainment", "ttft_ms"}
    import json
    json.dumps(d)  # must round-trip into results/*.jsonl
