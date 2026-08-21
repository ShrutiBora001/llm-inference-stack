"""What every serving binding shares.

Both bindings answer the same question -- when did each token arrive -- and
differ only in which engine they ask. Keeping the timing arithmetic here rather
than in each backend means it is written once, tested once, and cannot drift
between frameworks. A TTFT computed slightly differently for vLLM than for
SGLang would make the comparison meaningless in a way no test would catch,
because each side would be internally consistent.

Neither engine runs on the development machine, so everything in this module is
pure: timestamps in, records out. That is deliberate. The parts most likely to
be wrong are the arithmetic and the record construction, not the engine call,
and those are exactly the parts that can be tested for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..metrics import RequestRecord
from ..workload import Request, Workload

# Keys the engines use for prefix-cache hits. vLLM reports num_cached_tokens;
# SGLang reports cached_tokens in meta_info. Listed rather than hardcoded per
# backend so a renamed field fails the same way in both.
CACHED_TOKEN_KEYS = ("num_cached_tokens", "cached_tokens")


@runtime_checkable
class ServingClient(Protocol):
    """What `bench/serving.py` requires of a binding. Nothing more.

    The seam stays this narrow on purpose: a binding supplies a name and the
    ability to run a workload. Anything wider gives each binding somewhere for
    bugs to hide, and makes the two harder to compare.
    """

    name: str

    def send(self, workload: Workload) -> list[RequestRecord]: ...


@dataclass
class TokenStream:
    """One request's observed timeline, as an engine reported it.

    Absolute seconds from a common clock. `first_token_s` is when the first
    token became visible -- the only thing a user experiences as "did it hang".
    """

    arrival_s: float
    first_token_s: float
    last_token_s: float
    prompt_tokens: int
    output_tokens: int
    cached_prefix_tokens: int = 0


def pace_arrivals(requests: list[Request], t0: float) -> list[float]:
    """Absolute send times from the workload's arrival offsets.

    The offsets are a Poisson process generated once and reused across every
    framework, so all configurations see the *same* arrival trace. Regenerating
    them per framework would make part of any throughput difference an artifact
    of a different random process rather than of the system under test.
    """
    return [t0 + r.arrival_offset_s for r in requests]


def records_from_streams(streams: list[TokenStream]) -> list[RequestRecord]:
    """Convert observed timelines into the metric layer's record type.

    Validation lives in `RequestRecord.__post_init__`, so a malformed stream
    raises here rather than contributing a negative TTFT to a percentile and
    quietly lowering it.
    """
    return [
        RequestRecord(
            arrival_s=s.arrival_s,
            first_token_s=s.first_token_s,
            last_token_s=s.last_token_s,
            prompt_tokens=s.prompt_tokens,
            output_tokens=s.output_tokens,
            cached_prefix_tokens=s.cached_prefix_tokens,
        )
        for s in streams
    ]


def cached_tokens_from_output(obj, prompt_tokens: int) -> int:
    """Prefix-cache hits from an engine's own result object.

    Engines report this in different places and the difference is invisible in
    the output: reading the wrong one gives 0, which is indistinguishable from
    a cache that is switched off. That happened on first contact -- vLLM 0.27
    exposes `num_cached_tokens` as a direct attribute of `RequestOutput`, while
    `metrics` carries only timing, so reading `metrics` reported a 0% hit rate
    on a workload whose cache was in fact working perfectly.

    So: look in every place an engine is known to put it, and treat "found
    nowhere" as absent rather than as zero hits.
    """
    for key in CACHED_TOKEN_KEYS:
        value = getattr(obj, key, None)
        if value:
            return min(int(value), prompt_tokens)

    meta = obj if isinstance(obj, dict) else getattr(obj, "meta_info", None)
    return cached_tokens_from_usage(meta, prompt_tokens)


def cached_tokens_from_usage(usage: dict | None, prompt_tokens: int) -> int:
    """Prefix-cache hits, read from the engine rather than inferred.

    Inferring this from the workload's *requested* prefix share would make the
    metric a restatement of the input: it would report a perfect hit rate even
    with caching disabled, which is precisely the failure the SGLang contract
    check exists to catch.

    Clamped to the prompt length because a hit rate above 100% is nonsense and
    would inflate the headline figure the crossover finding rests on.
    """
    if not usage:
        return 0
    for key in CACHED_TOKEN_KEYS:
        if key in usage and usage[key]:
            return min(int(usage[key]), prompt_tokens)
    return 0
