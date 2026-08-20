"""Every metric defined once.

The reason this is a module rather than arithmetic scattered through benchmark
scripts: a metric that is computed in two places will eventually mean two
things, and the disagreement surfaces as an unexplainable number in a report.
Defining each here means the static figures, the dashboard and the contract
tests all consume the same function.

Thresholds are set HERE, before measurement. Raising one requires a written
justification in the phase report -- otherwise a gate becomes whatever the
implementation happened to achieve, which is not a gate.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

# --------------------------------------------------------------------- gates
# Recorded 2026-08-17, before any Phase 1/2 measurement exists.

MFU_GATE = 0.20
"""Fused kernel must reach 20% of dense peak. The unfused upstream loop reached
2.5-3.4%; a fused kernel that does not clear this has not justified the phase."""

BUSBW_FRACTION_GATE = 0.60
"""Measured NCCL bus bandwidth vs nameplate. Below this, suspect PCIe rather
than NVLink, or P2P disabled -- both silently invalidate every timing."""

RANK_SPREAD_GATE = 0.95
"""min/max per-rank compute under the striped layout. The whole point of
striping is that this approaches 1.0."""


@dataclass(frozen=True)
class SLO:
    """Latency targets a request must meet to count toward goodput.

    Defaults chosen for long-context prefill: TTFT is dominated by the O(S^2)
    prefill at 128K, so 2 s is realistic rather than aspirational. TPOT is the
    steady-state decode rate a user perceives as fluent.
    """

    ttft_p99_ms: float = 2000.0
    tpot_p99_ms: float = 50.0


DEFAULT_SLO = SLO()


# -------------------------------------------------------------------- kernel


def achieved_tflops(flops: float, seconds: float) -> float:
    """Measured throughput. `flops` must come from `analysis.attention_flops`
    over `computed_elements` -- what the tensor cores were actually asked for,
    including masked-away entries, not the causally-useful count."""
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    return flops / seconds / 1e12


def mfu(achieved: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization: achieved / dense peak.

    Dense, never the sparsity-doubled marketing figure -- attention is dense, so
    quoting against the sparse number would halve the apparent shortfall.
    """
    if peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive")
    return achieved / peak_tflops


def arithmetic_intensity(flops: float, hbm_bytes: float) -> float:
    """FLOPs per byte moved to/from HBM -- the x-axis of a roofline plot.

    This is the number that explains *why* an unfused kernel is slow: it
    materializes the score tile to HBM, so intensity collapses and the kernel
    lands on the memory-bound slope rather than the compute ceiling.
    """
    if hbm_bytes <= 0:
        raise ValueError("hbm_bytes must be positive")
    return flops / hbm_bytes


def roofline_ceiling(intensity: float, peak_tflops: float, hbm_tbs: float) -> float:
    """Best achievable TFLOP/s at this arithmetic intensity.

    min(compute ceiling, bandwidth * intensity). Where a measurement sits
    relative to this says whether to optimize math or memory traffic.
    """
    return min(peak_tflops, hbm_tbs * 1e3 * intensity / 1e3)


# --------------------------------------------------------------- distributed


def comm_seconds(comm_bytes: float, bandwidth_gb_s: float) -> float:
    """Time to move bytes at *measured* bandwidth.

    Always measured, never nameplate. On the A100 box the ring achieved
    156.5 GB/s against a 300 GB/s spec; using the spec would have understated
    communication time by 2x.
    """
    if bandwidth_gb_s <= 0:
        raise ValueError("bandwidth must be positive")
    return comm_bytes / (bandwidth_gb_s * 1e9)


def comm_fraction(comm_s: float, total_s: float) -> float:
    """Communication as a share of runtime. NOT `ring - local`.

    An earlier version of the upstream benchmark reported `ring - local` as
    communication cost. It is not: `local` attends only its own shard while
    `ring` attends the whole sequence, so that difference is overwhelmingly
    extra compute. That mistake reached a report before being caught.
    """
    if total_s <= 0:
        raise ValueError("total_s must be positive")
    return comm_s / total_s


def overlap_recovery(no_overlap_ms: float, overlap_ms: float, comm_ms: float) -> float:
    """Fraction of communication time that overlap actually hid.

    1.0 means fully hidden. Measured 0.54 on NVLink -- the remainder is likely
    NCCL's copy engine already running concurrently with host-side Python even
    in the 'no overlap' path, so the two configurations differ less than the
    flag suggests.
    """
    if comm_ms <= 0:
        raise ValueError("comm_ms must be positive")
    return (no_overlap_ms - overlap_ms) / comm_ms


def rank_spread(per_rank_times: list[float]) -> float:
    """min/max across ranks. 1.0 is perfect balance.

    The ring synchronizes every step, so the slowest rank sets the pace; this
    is the direct measurement of the imbalance striping exists to fix.
    """
    if not per_rank_times:
        raise ValueError("no per-rank times")
    hi = max(per_rank_times)
    if hi <= 0:
        raise ValueError("per-rank times must be positive")
    return min(per_rank_times) / hi


# -------------------------------------------------------------------- memory


def memory_model_accuracy(measured_bytes: float, predicted_bytes: float) -> float:
    """measured / predicted. **Must stay <= 1.0.**

    Asymmetric on purpose: over-predicting costs a rerun, under-predicting costs
    a CUDA OOM partway through a paid session. A regression test pins this
    against 13 real GPU configurations.
    """
    if predicted_bytes <= 0:
        raise ValueError("predicted_bytes must be positive")
    return measured_bytes / predicted_bytes


# ------------------------------------------------------------------- serving


@dataclass
class RequestRecord:
    """One request's timeline. Absolute seconds from a common clock."""

    arrival_s: float
    first_token_s: float
    last_token_s: float
    prompt_tokens: int
    output_tokens: int
    cached_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        if not (self.arrival_s <= self.first_token_s <= self.last_token_s):
            raise ValueError(
                f"timeline out of order: arrival={self.arrival_s} "
                f"first={self.first_token_s} last={self.last_token_s}"
            )
        if self.output_tokens < 1:
            raise ValueError("a request must emit at least one token")

    @property
    def ttft_ms(self) -> float:
        """Time to first token: what a user experiences as 'did it hang'."""
        return (self.first_token_s - self.arrival_s) * 1000.0

    @property
    def tpot_ms(self) -> float:
        """Mean inter-token latency *after* the first.

        Divides by (output_tokens - 1) because the first token's cost is TTFT
        and counting it twice flatters the number. A single-token response has
        no TPOT.
        """
        if self.output_tokens < 2:
            return 0.0
        return (self.last_token_s - self.first_token_s) * 1000.0 / (self.output_tokens - 1)

    def meets(self, slo: SLO = DEFAULT_SLO) -> bool:
        return self.ttft_ms <= slo.ttft_p99_ms and self.tpot_ms <= slo.tpot_p99_ms


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1].

    Written out rather than using numpy's default so the convention is explicit
    and identical between the figures and the dashboard.
    """
    if not values:
        raise ValueError("no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class ServingMetrics:
    """Everything derived from a completed run."""

    n_requests: int
    duration_s: float
    ttft_ms: dict[str, float] = field(default_factory=dict)
    tpot_ms: dict[str, float] = field(default_factory=dict)
    throughput_rps: float = 0.0
    output_throughput_tps: float = 0.0
    goodput_rps: float = 0.0
    slo_attainment: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    realized_prefix_share: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_requests": self.n_requests,
            "duration_s": self.duration_s,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "throughput_rps": self.throughput_rps,
            "output_throughput_tps": self.output_throughput_tps,
            "goodput_rps": self.goodput_rps,
            "slo_attainment": self.slo_attainment,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
            "realized_prefix_share": self.realized_prefix_share,
        }


def summarize(records: list[RequestRecord], slo: SLO = DEFAULT_SLO) -> ServingMetrics:
    """Reduce a run to its metrics.

    **Goodput, not throughput, is the headline.** A system can post excellent
    throughput while missing latency targets on most requests -- serving more
    users badly. Goodput counts only requests that met both SLOs, so it cannot
    be gamed that way.
    """
    if not records:
        raise ValueError("no records to summarize")

    start = min(r.arrival_s for r in records)
    end = max(r.last_token_s for r in records)
    duration = end - start
    if duration <= 0:
        raise ValueError("run has zero duration")

    ttfts = [r.ttft_ms for r in records]
    tpots = [r.tpot_ms for r in records if r.output_tokens >= 2]
    met = [r for r in records if r.meets(slo)]

    prompt_total = sum(r.prompt_tokens for r in records)
    cached_total = sum(r.cached_prefix_tokens for r in records)

    return ServingMetrics(
        n_requests=len(records),
        duration_s=duration,
        ttft_ms={f"p{int(q*100)}": percentile(ttfts, q) for q in (0.5, 0.9, 0.99)},
        tpot_ms={f"p{int(q*100)}": percentile(tpots, q) for q in (0.5, 0.9, 0.99)}
        if tpots else {},
        throughput_rps=len(records) / duration,
        output_throughput_tps=sum(r.output_tokens for r in records) / duration,
        goodput_rps=len(met) / duration,
        slo_attainment=len(met) / len(records),
        prefix_cache_hit_rate=cached_total / prompt_total if prompt_total else 0.0,
    )
