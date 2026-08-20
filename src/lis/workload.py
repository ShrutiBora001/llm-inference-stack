"""Request generation for serving benchmarks.

Deliberately a **thin driver over vLLM's samplers**, not a reimplementation.
`vllm.benchmarks.datasets` ships `random`, `sonnet`, `sharegpt`, `burstgpt` and
`hf` with an established request format and arrival model. Reusing them buys
numbers directly comparable to published vLLM benchmarks, and a lot of tested
code we do not have to debug. It also means we redistribute no dataset.

What we add is one thing: **shared-prefix length becomes the independent
variable** rather than a fixed setting. `sonnet` already parameterizes a prefix;
we sweep it into a curve, because the crossover point -- where SGLang's
RadixAttention overtakes vLLM -- is the finding, and a single workload gives one
point instead of a curve.

When vLLM is absent (the laptop), a local generator produces requests in the
same shape so the whole serving path stays CI-testable. The local path is for
tests and shape-checking, never for published numbers; `Workload.source`
records which was used and the benchmark writes it into every results row.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Sampling is pinned, never swept. Temperature does not change how much work
# attention does, but it changes *when* EOS is emitted, which changes decode
# step count, which moves throughput for reasons unrelated to the system under
# test. Fixing output length removes that variance entirely.
TEMPERATURE = 0.0
IGNORE_EOS = True


@dataclass(frozen=True)
class WorkloadSpec:
    """A reproducible benchmark workload.

    prefix_share is the independent variable: the fraction of each prompt that
    is a prefix shared with every other request. 0.0 means every request is
    unique (equivalent to vLLM's `random` sampler); 1.0 means all requests are
    identical.
    """

    n_requests: int = 64
    prompt_tokens: int = 4096
    output_tokens: int = 128
    prefix_share: float = 0.0
    qps: float = 4.0
    seed: int = 1234
    vocab_size: int = 32000

    def __post_init__(self) -> None:
        if not 0.0 <= self.prefix_share <= 1.0:
            raise ValueError(f"prefix_share must be in [0,1], got {self.prefix_share}")
        if self.n_requests < 1 or self.prompt_tokens < 1 or self.output_tokens < 1:
            raise ValueError("counts must be positive")
        if self.qps <= 0:
            raise ValueError("qps must be positive")

    @property
    def shared_tokens(self) -> int:
        return int(round(self.prefix_share * self.prompt_tokens))

    @property
    def unique_tokens(self) -> int:
        return self.prompt_tokens - self.shared_tokens


@dataclass
class Request:
    """One request, in the shape a serving client needs."""

    prompt_token_ids: list[int]
    output_tokens: int
    arrival_offset_s: float

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


@dataclass
class Workload:
    requests: list[Request]
    spec: WorkloadSpec
    source: str = "local"
    """Which generator produced this: "vllm" or "local". Recorded in every
    results row, because a number from the local generator is not comparable to
    a published vLLM benchmark and the file should say so."""

    meta: dict = field(default_factory=dict)

    def realized_prefix_share(self) -> float:
        """Measured, not assumed.

        The requested value and the produced one can differ through rounding, or
        through a sampler that does not honour the request. Every run records
        the realized value so the x-axis of the crossover plot is what actually
        happened.
        """
        return realized_prefix_share(self.requests)


def vllm_available() -> bool:
    """Whether vLLM's benchmark samplers can be imported here."""
    try:
        import vllm.benchmarks.datasets  # noqa: F401
    except Exception:
        return False
    return True


def poisson_arrivals(n: int, qps: float, rng: random.Random) -> list[float]:
    """Arrival offsets from a Poisson process at `qps`.

    Exponential inter-arrival times. Poisson rather than uniform because a
    uniform arrival pattern never produces the bursts that make queueing and
    continuous batching matter, so it would flatter the scheduler.
    """
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(qps)
    return out


def longest_common_prefix(sequences: list[list[int]]) -> int:
    """Length of the token prefix every sequence shares."""
    if not sequences:
        return 0
    shortest = min(len(s) for s in sequences)
    for i in range(shortest):
        tok = sequences[0][i]
        if any(s[i] != tok for s in sequences):
            return i
    return shortest


def realized_prefix_share(requests: list[Request]) -> float:
    """Shared prefix tokens as a fraction of mean prompt length."""
    if not requests:
        return 0.0
    shared = longest_common_prefix([r.prompt_token_ids for r in requests])
    mean_len = sum(r.prompt_len for r in requests) / len(requests)
    return shared / mean_len if mean_len else 0.0


def generate_local(spec: WorkloadSpec) -> Workload:
    """Local generator, matching vLLM's `random` sampler in shape.

    Token ids are drawn uniformly. That is exactly what vLLM's `random` sampler
    does and is correct here: prefill cost depends on token *count*, and the
    prefix cache keys on token *identity*, neither of which needs the ids to be
    meaningful text.
    """
    rng = random.Random(spec.seed)

    # One shared prefix, reused verbatim, so a prefix cache can actually hit.
    shared = [rng.randrange(spec.vocab_size) for _ in range(spec.shared_tokens)]

    requests = []
    for offset in poisson_arrivals(spec.n_requests, spec.qps, rng):
        unique = [rng.randrange(spec.vocab_size) for _ in range(spec.unique_tokens)]
        requests.append(Request(
            prompt_token_ids=shared + unique,
            output_tokens=spec.output_tokens,
            arrival_offset_s=offset,
        ))

    return Workload(requests=requests, spec=spec, source="local")


def generate(spec: WorkloadSpec, prefer_vllm: bool = True) -> Workload:
    """Produce a workload, preferring vLLM's samplers when available.

    The vLLM path is wired in Phase 2 once the framework is installed; until
    then this falls back and says so through `Workload.source`, rather than
    silently producing local data that a reader might mistake for a comparable
    benchmark.
    """
    if prefer_vllm and vllm_available():
        return _generate_vllm(spec)
    return generate_local(spec)


def _generate_vllm(spec: WorkloadSpec) -> Workload:  # pragma: no cover - needs vLLM
    """Drive vLLM's sampler at a fixed prefix share.

    Deliberately unimplemented until vLLM is installed and its sampler
    signatures can be read rather than guessed. Writing this against a
    remembered API would produce code that imports cleanly and is wrong.
    """
    raise NotImplementedError(
        "vLLM sampler integration is Phase 2 work: read "
        "vllm.benchmarks.datasets and drive its prefix-length parameter. "
        "Use generate_local(spec) for CPU tests."
    )


def sweep(
    base: WorkloadSpec,
    shares: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0),
) -> list[WorkloadSpec]:
    """The prefix-share sweep: one spec per point on the crossover curve."""
    return [WorkloadSpec(**{**base.__dict__, "prefix_share": s}) for s in shares]


def sampling_params() -> dict:
    """Pinned sampling settings, for the serving client and the results row.

    Recorded in output so a reader can see that output length was fixed rather
    than model-determined -- otherwise throughput comparisons across systems are
    not comparing the same work.
    """
    return {"temperature": TEMPERATURE, "ignore_eos": IGNORE_EOS}
