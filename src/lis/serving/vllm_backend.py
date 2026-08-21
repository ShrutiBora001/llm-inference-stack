"""vLLM binding: the baseline every comparison in Phase 2 is measured against.

vLLM is uniquely good at **PagedAttention and mature continuous batching**, so
that is what this binding has to actually exercise. The failure mode it must
avoid is running the engine in a degenerate single-request, single-block
configuration where none of that machinery is engaged and the comparison
measures nothing. `tests/test_contract.py` asserts against exactly that: a block
table with more than one block, non-contiguous, and requests that overlap in
time.

Scope, stated honestly: **vLLM's shipped context parallelism is decode-only**
(`decode_context_parallel_size`). It is not a prefill-CP implementation, so it
is not a like-for-like alternative to the zigzag ring — it is the shipped
baseline that a prefill-CP contribution would have to justify itself against.
Presenting it as an equivalent would misrepresent both.

## Structure

The timing logic and the engine calls are deliberately separate:

  - `records_from_streams` and `pace_arrivals` are pure functions over
    timestamps and are fully tested on a laptop
  - `VLLMClient` is a thin adapter that calls the engine and feeds those
    functions

This split exists because vLLM needs CUDA and cannot run on the development
machine at all. Without it, every line here would be untested until a rental,
and the parts most likely to be wrong are the timing arithmetic and the record
construction, not the engine call.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from ..workload import Request, Workload
from .client import (
    TokenStream,
    cached_tokens_from_usage,
    pace_arrivals,
    records_from_streams,
)

__all__ = [
    "TokenStream", "VLLMClient", "available", "cached_tokens_from_usage",
    "make_client", "pace_arrivals", "records_from_streams",
    "request_intervals", "sample_block_table",
]

# Llama-3.2-1B: small weights mean KV dominates device memory sooner, so a
# small model reaches *longer* context on the same hardware -- roughly 1M tokens
# on 4x40GB versus ~256K for an 8B. That is directly on-message for a
# context-parallelism study and makes a ~70-run sweep affordable.
DEFAULT_MODEL = os.environ.get("LIS_MODEL", "Qwen/Qwen2.5-1.5B")


def available() -> bool:
    """Whether vLLM can actually be imported here. Reported, never assumed."""
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


def _require_vllm():
    if not available():
        raise NotImplementedError(
            "vLLM is not installed. This binding refuses to simulate: a serving "
            "number produced without a serving engine is not a measurement, and "
            "the distinction disappears once it reaches a plot. "
            "Install vLLM on a GPU box, or use `bench/serving.py --dry-run`."
        )


# Timing helpers are shared with the SGLang binding and live in `client.py`,
# so a TTFT is computed identically for both engines. Computing it even slightly
# differently per framework would invalidate the comparison in a way no test
# catches, because each side would be internally consistent.


# --------------------------------------------------------------- the adapter


@dataclass
class VLLMClient:
    """Drives a live vLLM engine and returns one record per request."""

    name: str = "vllm"
    model: str = DEFAULT_MODEL
    tensor_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    enable_prefix_caching: bool = True
    max_model_len: int | None = None
    _engine: object = field(default=None, repr=False)

    def engine(self):
        """Construct the engine on first use.

        Async, because TTFT requires per-token arrival times and the offline
        `LLM.generate` API returns only completed outputs -- with it, TTFT and
        TPOT are not merely inaccurate, they are unobservable.
        """
        if self._engine is not None:
            return self._engine
        _require_vllm()

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        args = AsyncEngineArgs(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            decode_context_parallel_size=self.decode_context_parallel_size,
            enable_prefix_caching=self.enable_prefix_caching,
            max_model_len=self.max_model_len,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)
        return self._engine

    def send(self, workload: Workload) -> list[RequestRecord]:
        """Run the whole workload and return one record per request."""
        import asyncio

        return records_from_streams(asyncio.run(self._run(workload)))

    async def _run(self, workload: Workload) -> list[TokenStream]:
        import asyncio

        engine = self.engine()
        t0 = time.perf_counter()
        send_at = pace_arrivals(workload.requests, t0)

        tasks = [
            asyncio.create_task(self._one(engine, req, at, i))
            for i, (req, at) in enumerate(zip(workload.requests, send_at))
        ]
        return list(await asyncio.gather(*tasks))

    async def _one(self, engine, req: Request, send_at: float, idx: int) -> TokenStream:
        import asyncio

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from ..workload import IGNORE_EOS, TEMPERATURE

        delay = send_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        arrival = time.perf_counter()
        params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=req.output_tokens,
            ignore_eos=IGNORE_EOS,
        )
        stream = engine.generate(
            TokensPrompt(prompt_token_ids=req.prompt_token_ids),
            params,
            request_id=f"{self.name}-{idx}",
        )

        first = last = None
        emitted = 0
        cached = 0
        async for out in stream:
            now = time.perf_counter()
            emitted = len(out.outputs[0].token_ids) if out.outputs else 0
            if first is None and emitted > 0:
                first = now
            if emitted > 0:
                last = now
            cached = cached_tokens_from_usage(
                getattr(out, "metrics", None) and out.metrics.__dict__, req.prompt_len
            ) or cached

        if first is None or last is None:
            raise RuntimeError(
                f"request {idx} produced no tokens. With ignore_eos=True and "
                f"max_tokens={req.output_tokens} it must emit exactly that many; "
                "an empty result means the engine rejected the prompt."
            )

        return TokenStream(
            arrival_s=arrival, first_token_s=first, last_token_s=last,
            prompt_tokens=req.prompt_len, output_tokens=emitted,
            cached_prefix_tokens=cached,
        )


def make_client(framework: str = "vllm") -> VLLMClient:
    """Resolve a framework name to a configured client.

    `vllm-dcp` enables decode context parallelism across the visible devices.
    It is decode-only -- see the module docstring.
    """
    _require_vllm()
    import torch

    n = torch.cuda.device_count() or 1
    if framework == "vllm-dcp":
        return VLLMClient(name=framework, tensor_parallel_size=n,
                          decode_context_parallel_size=n)
    if framework in ("vllm", "vllm-baseline"):
        return VLLMClient(name=framework, tensor_parallel_size=n)
    raise ValueError(f"unknown vLLM configuration {framework!r}")


# ------------------------------------------------------- contract-check hooks
# Called only under VLLM_TEST=1, against a live engine.


def sample_block_table() -> list[int]:
    """Block IDs from a live paged allocation.

    The contract check fails if this is a single contiguous run, which would
    mean the allocator never actually paged and vLLM's central feature is
    unexercised by the benchmark.
    """
    _require_vllm()
    raise NotImplementedError(
        "Reach into the live KV cache manager for a request's block table. The "
        "path is version-specific (vLLM v1: engine core -> KVCacheManager -> "
        "per-request block IDs), so it must be written against the installed "
        "version rather than guessed here. Run with VLLM_TEST=1 on the box."
    )


def request_intervals() -> list[tuple[float, float]]:
    """(start, end) per request, to prove requests were in flight together.

    Without overlap this is a queue with extra steps rather than continuous
    batching, and any throughput number describes a different system than the
    one being claimed.
    """
    _require_vllm()
    raise NotImplementedError(
        "Derive from a completed run's RequestRecords: (arrival_s, last_token_s) "
        "per request. Requires one real run, so it is written on the box."
    )
