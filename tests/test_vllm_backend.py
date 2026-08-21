"""The vLLM binding's testable half.

vLLM needs CUDA and cannot run on the development machine, so the engine call
itself is only exercised on a rented box. What *can* be tested here is the half
most likely to be wrong: arrival pacing, record construction, timing
arithmetic, and the reduction path that turns records into published numbers.

That last one matters. `bench/serving.py` was written against an imagined
`summarize()` signature and would have failed the first time an engine existed
-- the dry-run and refusal tests never touched the reduction path. A fake
in-process client fixes that permanently.

The fake is a **test double, never a fallback**. It exists only in this file,
`bench/serving.py` cannot reach it, and no path writes its output to results/.
Simulated numbers must never be able to reach a plot.
"""

from __future__ import annotations

import pytest

from lis.metrics import SLO, RequestRecord, summarize
from lis.serving.vllm_backend import (
    TokenStream,
    available,
    cached_tokens_from_usage,
    make_client,
    pace_arrivals,
    records_from_streams,
)
from lis.workload import Request, WorkloadSpec, generate

needs_vllm = pytest.mark.skipif(not available(), reason="vLLM not installed here")


# ------------------------------------------------------------------- pacing


def test_arrivals_preserve_the_workloads_offsets():
    """Every framework must see the same arrival trace.

    Regenerating arrivals per framework would make part of any throughput
    difference an artifact of a different random process rather than of the
    system under test.
    """
    reqs = [Request([1, 2, 3], 4, off) for off in (0.0, 0.5, 1.25)]
    assert pace_arrivals(reqs, t0=100.0) == [100.0, 100.5, 101.25]


def test_arrivals_are_monotonic_for_a_generated_workload():
    w = generate(WorkloadSpec(n_requests=16, prompt_tokens=128, output_tokens=8))
    times = pace_arrivals(w.requests, t0=0.0)
    assert times == sorted(times), "arrival trace is out of order"


# ------------------------------------------------------------------ records


def test_streams_become_records_with_correct_latencies():
    """Hand-computed: 120 ms TTFT, and 9 gaps over 900 ms is 100 ms TPOT."""
    s = TokenStream(arrival_s=1.0, first_token_s=1.12, last_token_s=2.02,
                    prompt_tokens=512, output_tokens=10)
    (rec,) = records_from_streams([s])
    assert rec.ttft_ms == pytest.approx(120.0)
    assert rec.tpot_ms == pytest.approx(100.0)


def test_a_malformed_timeline_is_rejected_not_averaged():
    """A first token before arrival is impossible. Silently keeping it would
    contribute a negative TTFT to a percentile and quietly lower it."""
    bad = TokenStream(arrival_s=2.0, first_token_s=1.0, last_token_s=3.0,
                      prompt_tokens=8, output_tokens=4)
    with pytest.raises(ValueError, match="timeline out of order"):
        records_from_streams([bad])


# ------------------------------------------------------------ cache reporting


def test_cached_tokens_come_from_the_engine_not_the_workload():
    assert cached_tokens_from_usage({"num_cached_tokens": 300}, 1000) == 300


def test_absent_usage_reports_no_cache_hits():
    """Missing telemetry must read as zero, never as 'assume it worked'."""
    assert cached_tokens_from_usage(None, 1000) == 0
    assert cached_tokens_from_usage({}, 1000) == 0


def test_cached_tokens_cannot_exceed_the_prompt():
    """A hit rate above 100% would be nonsense and would inflate the headline
    prefix-cache figure the SGLang comparison rests on."""
    assert cached_tokens_from_usage({"num_cached_tokens": 5000}, 1000) == 1000


# --------------------------------------------------------------- refusal


def test_make_client_refuses_without_vllm():
    if available():
        pytest.skip("vLLM is installed here")
    with pytest.raises(NotImplementedError, match="refuses to simulate"):
        make_client("vllm-dcp")


@needs_vllm
def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="unknown vLLM configuration"):
        make_client("vllm-quantum")


# ------------------------------------------------- the reduction path, end to end


class FakeClient:
    """A test double for the client protocol. Never used outside tests.

    Emits a deterministic timeline so the arithmetic in `bench/serving.py` can
    be checked against hand-computed numbers.
    """

    name = "fake"

    def __init__(self, ttft_s=0.1, tpot_s=0.01, cached_frac=0.0):
        self.ttft_s, self.tpot_s, self.cached_frac = ttft_s, tpot_s, cached_frac

    def send(self, workload):
        out = []
        for r in workload.requests:
            arrival = r.arrival_offset_s
            first = arrival + self.ttft_s
            last = first + self.tpot_s * (r.output_tokens - 1)
            out.append(RequestRecord(
                arrival_s=arrival, first_token_s=first, last_token_s=last,
                prompt_tokens=r.prompt_len, output_tokens=r.output_tokens,
                cached_prefix_tokens=int(r.prompt_len * self.cached_frac),
            ))
        return out


def test_run_one_produces_every_key_the_results_file_needs():
    """The regression test for the bug this file was written after.

    bench/serving.py called summarize() with a parameter it does not take and
    read percentile keys it does not return. Both are only reachable with a
    client, so no existing test touched them.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    import serving

    spec = WorkloadSpec(n_requests=8, prompt_tokens=256, output_tokens=16,
                        prefix_share=0.5)
    row = serving.run_one(FakeClient(), spec, SLO())

    for key in ("goodput_rps", "throughput_rps", "slo_attainment",
                "prefix_cache_hit_rate", "ttft_p99_ms", "tpot_p99_ms",
                "framework", "prefix_share_realized", "workload_source"):
        assert key in row, f"results row is missing {key!r}"
    assert not isinstance(row["ttft_p99_ms"], dict), "percentiles were not flattened"


def test_run_one_latencies_match_the_fake_clients_timeline():
    """100 ms TTFT and 10 ms TPOT go in; the same must come out."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    import serving

    spec = WorkloadSpec(n_requests=8, prompt_tokens=256, output_tokens=16)
    row = serving.run_one(FakeClient(ttft_s=0.1, tpot_s=0.01), spec, SLO())
    assert row["ttft_p99_ms"] == pytest.approx(100.0, rel=1e-6)
    assert row["tpot_p99_ms"] == pytest.approx(10.0, rel=1e-6)


def test_goodput_falls_below_throughput_when_the_slo_is_missed():
    """The headline metric must actually discriminate.

    A 200 ms TPOT against a 50 ms SLO fails every request, so goodput must be
    zero while throughput stays positive. If goodput tracked throughput here,
    it would be measuring nothing.
    """
    spec = WorkloadSpec(n_requests=8, prompt_tokens=256, output_tokens=16)
    records = FakeClient(ttft_s=0.1, tpot_s=0.2).send(generate(spec))
    m = summarize(records, slo=SLO(ttft_p99_ms=2000, tpot_p99_ms=50))
    assert m.throughput_rps > 0
    assert m.goodput_rps == 0.0
    assert m.slo_attainment == 0.0


def test_cache_hit_rate_is_reported_from_the_records():
    spec = WorkloadSpec(n_requests=8, prompt_tokens=1000, output_tokens=4)
    records = FakeClient(cached_frac=0.3).send(generate(spec))
    assert summarize(records).prefix_cache_hit_rate == pytest.approx(0.3, abs=0.01)


# ------------------------------------------- the field-location regression
# vLLM 0.27 puts num_cached_tokens on RequestOutput itself; `metrics` holds only
# timing. Reading `metrics` returned 0, which is indistinguishable in the output
# from a cache that is switched off — and it produced a 0% hit rate on a
# workload whose cache was demonstrably working.


class FakeVLLMOutput:
    """Shaped like vLLM 0.27's RequestOutput: the count is a direct attribute."""

    def __init__(self, cached):
        self.num_cached_tokens = cached
        self.metrics = type("Stats", (), {"num_generation_tokens": 4})()


class FakeSGLangChunk(dict):
    """SGLang puts it inside meta_info."""


def test_cached_tokens_read_from_a_direct_attribute():
    from lis.serving.client import cached_tokens_from_output

    assert cached_tokens_from_output(FakeVLLMOutput(592), 1024) == 592


def test_timing_only_metrics_do_not_masquerade_as_zero_hits():
    """The exact first-contact bug: metrics exists but carries no cache field."""
    from lis.serving.client import cached_tokens_from_output

    out = FakeVLLMOutput(592)
    assert not hasattr(out.metrics, "num_cached_tokens")
    assert cached_tokens_from_output(out, 1024) == 592


def test_cached_tokens_read_from_a_nested_meta_info():
    from lis.serving.client import cached_tokens_from_output

    assert cached_tokens_from_output({"cached_tokens": 300}, 1024) == 300


def test_no_cache_field_anywhere_reads_as_zero():
    from lis.serving.client import cached_tokens_from_output

    assert cached_tokens_from_output(object(), 1024) == 0


def test_the_client_reuses_one_event_loop_across_runs():
    """asyncio.run() closes its loop on return, which orphans the background
    tasks of an engine cached across calls -- the second sweep point then fails
    with EngineDeadError. One loop per client is what makes a sweep possible at
    all; the alternative reloads the model per point."""
    from lis.serving.sglang_backend import SGLangClient
    from lis.serving.vllm_backend import VLLMClient

    for cls in (VLLMClient, SGLangClient):
        c = cls()
        first = c.loop()
        assert c.loop() is first, f"{cls.__name__} made a second loop"
        assert not first.is_closed()
        c.close()
