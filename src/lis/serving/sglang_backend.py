"""SGLang binding: the prefix-caching arm of the Phase 2 comparison.

SGLang is uniquely good at **RadixAttention prefix caching** — a radix tree over
KV blocks that lets requests sharing a prefix skip recomputing it. The failure
mode this binding must avoid is benchmarking on a workload with no shared
prefixes, where the radix tree is dead weight and SGLang is indistinguishable
from anything else. `tests/test_contract.py` fails if the hit rate is near zero
on a workload built to share prefixes.

## What is measurable today, and what is not

This matters more than it looks, and it corrects the project plan.

SGLang's **prefill context parallelism is Ascend-NPU only** (issue #22223).
Neither `sglang-zigzag` nor `sglang-contiguous` can be constructed on CUDA until
the port exists, and the port is gated on a maintainer reply. The plan listed the
serving study as "ungated — start here" while naming those two configurations,
which cannot both be true.

What *is* ungated, and is the more valuable half:

| Measurement | Needs CP? | Status |
|---|---|---|
| Prefix-share crossover, SGLang vs vLLM | no | **runnable now** |
| Prefix-cache hit rate vs `prefix_share` | no | **runnable now** |
| TTFT / goodput vs concurrency | no | **runnable now** |
| Zigzag vs contiguous CP layout | yes | blocked on the port |

The crossover figure (plan §8.1, S4) was always the headline finding, and it
needs no context parallelism at all — it is a property of RadixAttention versus
vLLM's prefix caching. So the study proceeds; only the layout comparison waits.

`make_client` therefore refuses the CP configurations with an explanation rather
than silently constructing a non-CP engine under a CP name, which would produce
a plausible results file describing a system that was never run.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from ..metrics import RequestRecord
from ..workload import Workload
from .client import (
    TokenStream,
    cached_tokens_from_usage,
    pace_arrivals,
    records_from_streams,
)

DEFAULT_MODEL = os.environ.get("LIS_MODEL", "meta-llama/Llama-3.2-1B")

# Configurations that require prefill CP on CUDA, which does not exist upstream.
CP_CONFIGS = ("sglang-zigzag", "sglang-contiguous")


def available() -> bool:
    """Whether SGLang can actually be imported here. Reported, never assumed."""
    try:
        import sglang  # noqa: F401
    except Exception:
        return False
    return True


def _require_sglang():
    if not available():
        raise NotImplementedError(
            "SGLang is not installed. This binding refuses to simulate: a "
            "serving number produced without a serving engine is not a "
            "measurement, and the distinction disappears once it reaches a "
            "plot. Install SGLang on a GPU box, or use "
            "`bench/serving.py --dry-run`."
        )


@dataclass
class SGLangClient:
    """Drives a live SGLang engine and returns one record per request."""

    name: str = "sglang"
    model: str = DEFAULT_MODEL
    tp_size: int = 1
    disable_radix_cache: bool = False
    _engine: object = field(default=None, repr=False)

    def engine(self):
        """Construct the engine on first use.

        `sgl.Engine` is the in-process entry point. The HTTP server would add
        network latency to every TTFT measurement, which is a real cost in
        production but not the thing under comparison here — and it would be
        added unevenly, since the two engines' servers differ.
        """
        if self._engine is not None:
            return self._engine
        _require_sglang()

        import sglang as sgl

        self._engine = sgl.Engine(
            model_path=self.model,
            tp_size=self.tp_size,
            disable_radix_cache=self.disable_radix_cache,
        )
        return self._engine

    def send(self, workload: Workload) -> list[RequestRecord]:
        import asyncio

        return records_from_streams(asyncio.run(self._run(workload)))

    async def _run(self, workload: Workload) -> list[TokenStream]:
        import asyncio

        engine = self.engine()
        t0 = time.perf_counter()
        send_at = pace_arrivals(workload.requests, t0)

        tasks = [
            asyncio.create_task(self._one(engine, req, at))
            for req, at in zip(workload.requests, send_at)
        ]
        return list(await asyncio.gather(*tasks))

    async def _one(self, engine, req, send_at: float) -> TokenStream:
        import asyncio

        from ..workload import IGNORE_EOS, TEMPERATURE

        delay = send_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        arrival = time.perf_counter()
        params = {
            "temperature": TEMPERATURE,
            "max_new_tokens": req.output_tokens,
            "ignore_eos": IGNORE_EOS,
        }

        first = last = None
        emitted = 0
        cached = 0
        stream = await engine.async_generate(
            input_ids=req.prompt_token_ids, sampling_params=params, stream=True
        )
        async for chunk in stream:
            now = time.perf_counter()
            meta = chunk.get("meta_info") or {}
            emitted = meta.get("completion_tokens", emitted + 1)
            if first is None:
                first = now
            last = now
            cached = cached_tokens_from_usage(meta, req.prompt_len) or cached

        if first is None or last is None:
            raise RuntimeError(
                f"request produced no tokens. With ignore_eos=True and "
                f"max_new_tokens={req.output_tokens} it must emit exactly that "
                "many; an empty result means the engine rejected the prompt."
            )

        return TokenStream(
            arrival_s=arrival, first_token_s=first, last_token_s=last,
            prompt_tokens=req.prompt_len, output_tokens=max(emitted, 1),
            cached_prefix_tokens=cached,
        )


def make_client(framework: str = "sglang") -> SGLangClient:
    """Resolve a framework name to a configured client.

    Refuses the CP configurations explicitly. Constructing a non-CP engine under
    a CP name would write a results file describing a system that was never run,
    and nothing downstream could detect it.
    """
    if framework in CP_CONFIGS:
        raise NotImplementedError(
            f"{framework!r} requires SGLang prefill context parallelism on CUDA, "
            "which does not exist upstream — the implementation in issue #22223 "
            "is Ascend-NPU only, and the CUDA port is Phase 2's gated track. "
            "This refuses to simulate rather than construct a non-CP engine "
            "under a CP name. The ungated measurements — prefix-share "
            "crossover, cache hit rate, goodput vs concurrency — need no CP: "
            "use 'sglang' or 'sglang-no-cache'."
        )

    _require_sglang()
    import torch

    n = torch.cuda.device_count() or 1
    if framework == "sglang":
        return SGLangClient(name=framework, tp_size=n)
    if framework == "sglang-no-cache":
        # The A/B that isolates RadixAttention's contribution. Without it, a
        # throughput difference against vLLM could be caching or could be a
        # dozen other implementation differences.
        return SGLangClient(name=framework, tp_size=n, disable_radix_cache=True)
    raise ValueError(f"unknown SGLang configuration {framework!r}")


# ------------------------------------------------------- contract-check hooks


def prefix_cache_hit_rate(prefix_share: float = 0.75, n_requests: int = 16) -> float:
    """Measured hit rate on a workload deliberately built to share prefixes.

    A near-zero result means the benchmark chose a workload that cannot
    distinguish SGLang from anything else, so the whole comparison is measuring
    something other than what it claims.

    Runs a real workload rather than querying a counter: the counter can be
    non-zero from warmup traffic, whereas this ties the number to a known input.
    """
    _require_sglang()

    from ..metrics import summarize
    from ..workload import WorkloadSpec, generate

    client = make_client("sglang")
    spec = WorkloadSpec(n_requests=n_requests, prompt_tokens=2048,
                        output_tokens=8, prefix_share=prefix_share)
    records = client.send(generate(spec))
    return summarize(records).prefix_cache_hit_rate
