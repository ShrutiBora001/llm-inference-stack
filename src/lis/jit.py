"""torch.compile and CUDA graphs, applied where they actually pay.

These are the "JIT" leg of the stack, and the easiest to use decoratively —
wrapping a model in `torch.compile` and declaring victory is common, and often
buys nothing because a graph break silently returns the hot region to eager
execution.

Two places they genuinely earn their keep:

**Elementwise fusion on the non-attention path.** RMSNorm, RoPE, and residual
adds are memory-bound chains of small ops. Attention itself goes to a fused
kernel (see `lis.kernels`), so what remains around it is exactly what inductor
is good at collapsing.

**Launch overhead in decode.** A decode step does very little arithmetic per
token, so wall-clock time is dominated by kernel launches rather than compute.
CUDA graph capture replays the whole step as one submission. This does not help
prefill, which is compute-bound — applying it there would be ceremony.

The contract suite asserts both: zero graph breaks, and a measured reduction in
launches. Neither is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .rope import apply_rope


class RMSNorm(nn.Module):
    """Llama-style RMSNorm. Memory-bound; a fusion target, not a compute one."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class DecodeStep(nn.Module):
    """The elementwise work surrounding attention in one decode step.

    Deliberately excludes attention: that goes to a fused kernel, and mixing the
    two would make it impossible to attribute any speedup to compilation.

    What remains — norm, projections, RoPE, residual — is a chain of small
    memory-bound ops, which is precisely what inductor collapses well.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = self.norm(x)
        q = self.wq(h).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        # Global positions, for the same reason as everywhere else in this
        # project: under a striped layout a device owns disjoint position ranges.
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        merged = (q + k).transpose(1, 2).reshape(b, s, -1)
        return x + self.wo(merged)


def make_decode_step(d_model: int = 512, n_heads: int = 8) -> DecodeStep:
    return DecodeStep(d_model, n_heads).eval()


@dataclass
class GraphBreak:
    reason: str
    user_stack: str = ""

    def __repr__(self) -> str:  # keeps assertion output readable
        return f"GraphBreak({self.reason!r})"


def graph_break_report(
    module: nn.Module, *, device: str = "cpu", batch: int = 1, seq: int = 1
) -> list[GraphBreak]:
    """Graph breaks dynamo hits while tracing `module`.

    A break means compilation gave up on that region and fell back to eager, so
    the compiled model is partly uncompiled — usually without anyone noticing,
    since it still produces correct output.

    Tracing is device-independent, so this runs on a laptop and catches the most
    common failure for free.
    """
    import torch._dynamo as dynamo

    dynamo.reset()
    module = module.to(device)
    x = torch.randn(batch, seq, module.norm.weight.shape[0], device=device)
    positions = torch.arange(seq, device=device)

    explanation = dynamo.explain(module)(x, positions)

    reasons = getattr(explanation, "break_reasons", []) or []
    out = []
    for r in reasons:
        out.append(GraphBreak(
            reason=getattr(r, "reason", str(r)),
            user_stack="".join(str(s) for s in getattr(r, "user_stack", []) or []),
        ))
    return out


def compiled_decode_step(module: nn.Module | None = None, **kw) -> nn.Module:
    """The compiled decode path. `fullgraph=True` turns a graph break into an
    error rather than a silent fallback — the whole point."""
    module = module if module is not None else make_decode_step(**kw)
    return torch.compile(module, fullgraph=True, dynamic=False)


def decode_launch_counts(
    *, d_model: int = 512, n_heads: int = 8, iters: int = 20
) -> tuple[int, int]:
    """Kernel launches per decode step: eager versus CUDA-graph replay.

    Returns (eager, captured). The contract suite requires capture to at least
    halve the count; decode is launch-bound, so if it does not, capture is not
    doing its job.

    CUDA only — there is nothing to capture on CPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("decode_launch_counts requires CUDA")

    module = make_decode_step(d_model, n_heads).cuda()
    x = torch.randn(1, 1, d_model, device="cuda")
    positions = torch.arange(1, device="cuda")

    def count_launches(fn) -> int:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(3):  # warm up allocator and any lazy init
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sum(
            1 for e in prof.events()
            if getattr(e, "device_type", None) is not None and e.self_device_time_total > 0
        )

    eager = count_launches(lambda: module(x, positions))

    # Capture the step into a graph and replay it as a single submission.
    static_x, static_pos = x.clone(), positions.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(static_x, static_pos)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module(static_x, static_pos)

    captured = count_launches(graph.replay)
    return eager, captured
