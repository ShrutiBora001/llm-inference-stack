"""Prefill context parallelism: zigzag ring attention, framework-agnostic.

This is the algorithm SGLang issue #22223 asks for, implemented independently of
any serving framework. That separation is deliberate:

  - it is testable on a laptop over gloo, so correctness costs nothing
  - it can be bound to SGLang, vLLM or a standalone server without change
  - a framework binding becomes a thin adapter rather than a fork

`adapter.py` defines the narrow interface a framework must satisfy, with an
in-memory implementation used by the tests. Binding to a real framework means
implementing that interface against its KV cache and process groups, not
reimplementing the ring.
"""

from .adapter import CPContext, InMemoryKVProvider, KVProvider
from .ring import ring_attention_cp, ring_attention_cp_paged

__all__ = [
    "CPContext",
    "InMemoryKVProvider",
    "KVProvider",
    "ring_attention_cp",
    "ring_attention_cp_paged",
]
