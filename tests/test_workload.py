"""Workload generation: does it produce the sharing it claims to?

prefix_share is the independent variable of the whole serving study, so a
generator that quietly produces a different ratio than requested would move
every point on the crossover curve while looking entirely healthy. These tests
measure what was produced rather than trusting the parameter.
"""

from __future__ import annotations

import pytest

from lis.workload import (
    IGNORE_EOS,
    TEMPERATURE,
    Request,
    WorkloadSpec,
    generate,
    generate_local,
    longest_common_prefix,
    poisson_arrivals,
    realized_prefix_share,
    sampling_params,
    sweep,
    vllm_available,
)

SHARES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


# ------------------------------------------------------------------- the knob


@pytest.mark.parametrize("share", SHARES)
def test_realized_prefix_share_matches_request(share):
    """The property the entire crossover curve rests on."""
    w = generate_local(WorkloadSpec(n_requests=16, prompt_tokens=200, prefix_share=share))
    assert w.realized_prefix_share() == pytest.approx(share, abs=0.01)


def test_zero_share_produces_no_common_prefix():
    """ρ=0 must be genuinely unique, not merely low — it is the control point,
    and it is what vLLM's `random` sampler produces."""
    w = generate_local(WorkloadSpec(n_requests=16, prompt_tokens=200, prefix_share=0.0))
    assert longest_common_prefix([r.prompt_token_ids for r in w.requests]) == 0


def test_full_share_makes_every_request_identical():
    w = generate_local(WorkloadSpec(n_requests=8, prompt_tokens=64, prefix_share=1.0))
    first = w.requests[0].prompt_token_ids
    assert all(r.prompt_token_ids == first for r in w.requests)


def test_shared_prefix_is_byte_identical_across_requests():
    """A prefix cache keys on token identity. Near-identical is a total miss,
    so 'shared' has to mean exactly equal, not statistically similar."""
    w = generate_local(WorkloadSpec(n_requests=8, prompt_tokens=100, prefix_share=0.5))
    n = w.spec.shared_tokens
    heads = {tuple(r.prompt_token_ids[:n]) for r in w.requests}
    assert len(heads) == 1


def test_unique_suffixes_actually_differ():
    """Otherwise ρ<1 would silently behave like ρ=1."""
    w = generate_local(WorkloadSpec(n_requests=8, prompt_tokens=100, prefix_share=0.5))
    n = w.spec.shared_tokens
    tails = {tuple(r.prompt_token_ids[n:]) for r in w.requests}
    assert len(tails) == len(w.requests)


def test_prompt_length_is_exact():
    spec = WorkloadSpec(n_requests=8, prompt_tokens=333, prefix_share=0.3)
    for r in generate_local(spec).requests:
        assert r.prompt_len == 333


# --------------------------------------------------------------- reproducible


def test_same_seed_is_byte_reproducible():
    a = generate_local(WorkloadSpec(n_requests=8, seed=7))
    b = generate_local(WorkloadSpec(n_requests=8, seed=7))
    assert [r.prompt_token_ids for r in a.requests] == [r.prompt_token_ids for r in b.requests]
    assert [r.arrival_offset_s for r in a.requests] == [r.arrival_offset_s for r in b.requests]


def test_different_seed_differs():
    a = generate_local(WorkloadSpec(n_requests=8, seed=1))
    b = generate_local(WorkloadSpec(n_requests=8, seed=2))
    assert [r.prompt_token_ids for r in a.requests] != [r.prompt_token_ids for r in b.requests]


# ------------------------------------------------------------------- arrivals


def test_arrivals_are_monotonic_and_start_at_zero():
    offs = poisson_arrivals(50, qps=10.0, rng=__import__("random").Random(0))
    assert offs[0] == 0.0
    assert offs == sorted(offs)


def test_arrival_rate_is_approximately_qps():
    """Poisson, not uniform: bursts are what make queueing and continuous
    batching matter, so a uniform pattern would flatter the scheduler."""
    import random as _r

    n, qps = 4000, 20.0
    offs = poisson_arrivals(n, qps, _r.Random(3))
    observed = (n - 1) / offs[-1]
    assert observed == pytest.approx(qps, rel=0.1)


# -------------------------------------------------------------------- pinning


def test_output_length_is_fixed_not_model_determined():
    """Fixed output length is what makes throughput comparable across systems.
    If the model decided, output length would vary with sampling and the
    comparison would not be of the same work."""
    spec = WorkloadSpec(n_requests=8, output_tokens=77)
    assert all(r.output_tokens == 77 for r in generate_local(spec).requests)


def test_sampling_is_pinned_greedy():
    p = sampling_params()
    assert p["temperature"] == TEMPERATURE == 0.0
    assert p["ignore_eos"] == IGNORE_EOS is True


# --------------------------------------------------------------------- sweep


def test_sweep_covers_the_curve_and_changes_nothing_else():
    base = WorkloadSpec(n_requests=32, prompt_tokens=8192, qps=6.0, seed=99)
    specs = sweep(base, SHARES)
    assert [s.prefix_share for s in specs] == list(SHARES)
    for s in specs:
        assert (s.n_requests, s.prompt_tokens, s.qps, s.seed) == (32, 8192, 6.0, 99)


# ---------------------------------------------------------------- provenance


def test_source_is_recorded_honestly():
    """A local-generator number is not comparable to a published vLLM
    benchmark. Every results row must say which produced it."""
    w = generate_local(WorkloadSpec(n_requests=4))
    assert w.source == "local"
    assert generate(WorkloadSpec(n_requests=4), prefer_vllm=False).source == "local"


def test_vllm_path_refuses_rather_than_guessing():
    """The vLLM sampler integration is unimplemented on purpose: writing it
    against a remembered API would import cleanly and be wrong."""
    from lis.workload import _generate_vllm

    if vllm_available():
        pytest.skip("vLLM installed — implement and test the real path")
    with pytest.raises(NotImplementedError, match="Phase 2 work"):
        _generate_vllm(WorkloadSpec())


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_out_of_range_share_rejected(bad):
    with pytest.raises(ValueError, match="prefix_share"):
        WorkloadSpec(prefix_share=bad)


def test_nonpositive_counts_rejected():
    with pytest.raises(ValueError, match="positive"):
        WorkloadSpec(n_requests=0)


def test_realized_share_of_empty_is_zero():
    assert realized_prefix_share([]) == 0.0


def test_longest_common_prefix_handles_unequal_lengths():
    assert longest_common_prefix([[1, 2, 3], [1, 2]]) == 2
