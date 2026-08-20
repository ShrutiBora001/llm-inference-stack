"""The SGLang binding's testable half, and the scoping it enforces.

The most important thing tested here is a refusal. SGLang's prefill context
parallelism is Ascend-NPU only, so `sglang-zigzag` and `sglang-contiguous`
cannot be constructed on CUDA. Silently building a non-CP engine under a CP name
would write a results file describing a system that was never run — and nothing
downstream could detect it, because every number in that file would be real.
"""

from __future__ import annotations

import pytest

from lis.serving.sglang_backend import CP_CONFIGS, available, make_client
from lis.serving.sglang_backend import prefix_cache_hit_rate

needs_sglang = pytest.mark.skipif(not available(), reason="SGLang not installed here")


@pytest.mark.parametrize("framework", CP_CONFIGS)
def test_cp_configurations_are_refused_with_the_reason(framework):
    """The scoping correction, enforced in code rather than left in a doc."""
    with pytest.raises(NotImplementedError, match="Ascend"):
        make_client(framework)


def test_the_refusal_points_at_what_can_be_run_instead():
    """A refusal that does not say what to do instead just stops the work."""
    with pytest.raises(NotImplementedError, match="sglang-no-cache|need no CP"):
        make_client("sglang-zigzag")


def test_cp_refusal_does_not_depend_on_sglang_being_installed():
    """It is a statement about upstream, not about this machine.

    If it were ordered after the install check, someone with SGLang present
    would get "not installed" for a config that will never work, and someone
    without it would never learn the config is unavailable at all.
    """
    with pytest.raises(NotImplementedError, match="Ascend"):
        make_client("sglang-contiguous")


def test_non_cp_configurations_refuse_only_for_a_missing_engine():
    if available():
        pytest.skip("SGLang is installed here")
    with pytest.raises(NotImplementedError, match="refuses to simulate"):
        make_client("sglang")


@needs_sglang
def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="unknown SGLang configuration"):
        make_client("sglang-warpdrive")


def test_prefix_cache_hit_rate_requires_an_engine():
    if available():
        pytest.skip("SGLang is installed here")
    with pytest.raises(NotImplementedError, match="refuses to simulate"):
        prefix_cache_hit_rate()


# --------------------------------------------- shared timing, one implementation


def test_both_bindings_share_the_timing_code():
    """A TTFT computed differently per framework would invalidate the whole
    comparison in a way no test catches, because each side would be internally
    consistent. There must be exactly one implementation."""
    from lis.serving import sglang_backend, vllm_backend
    from lis.serving.client import pace_arrivals, records_from_streams

    assert sglang_backend.pace_arrivals is pace_arrivals
    assert vllm_backend.pace_arrivals is pace_arrivals
    assert sglang_backend.records_from_streams is records_from_streams
    assert vllm_backend.records_from_streams is records_from_streams


def test_cached_token_keys_cover_both_engines():
    """vLLM reports num_cached_tokens, SGLang reports cached_tokens."""
    from lis.serving.client import cached_tokens_from_usage

    assert cached_tokens_from_usage({"num_cached_tokens": 100}, 500) == 100
    assert cached_tokens_from_usage({"cached_tokens": 100}, 500) == 100
    assert cached_tokens_from_usage({"totally_different": 100}, 500) == 0


def test_clients_satisfy_the_declared_protocol():
    """The seam stays narrow: a name, and the ability to run a workload."""
    from lis.serving.client import ServingClient
    from lis.serving.sglang_backend import SGLangClient
    from lis.serving.vllm_backend import VLLMClient

    for cls in (SGLangClient, VLLMClient):
        assert isinstance(cls(), ServingClient), f"{cls.__name__} broke the protocol"
