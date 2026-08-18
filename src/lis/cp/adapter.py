"""The seam between the ring algorithm and a serving framework.

Kept deliberately narrow. A framework binding supplies KV storage and a process
group; it does not supply attention, masking, position arithmetic or the ring
schedule. Anything wider would make each binding a place bugs can hide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from dattn.layout import Chunk, ShardLayout, make_layout


@dataclass
class CPContext:
    """Everything the ring needs to know about its place in the CP group.

    `layout` is the zigzag assignment. Both SGLang's RFC and this project arrive
    at the same construction -- the sequence split into 2*cp_size chunks with
    rank r taking chunk r and chunk 2*cp_size-1-r -- so a binding should assert
    equality against the framework's own partition rather than assume it.
    """

    rank: int
    world_size: int
    layout: ShardLayout
    causal: bool = True
    block_size: int = 512
    backend: str | None = None  # None = best available; force for A/B testing

    @classmethod
    def create(
        cls,
        rank: int,
        world_size: int,
        seq_len: int,
        *,
        layout: str = "striped",
        **kw,
    ) -> "CPContext":
        return cls(rank=rank, world_size=world_size,
                   layout=make_layout(layout, seq_len, world_size), **kw)

    def local_chunks(self) -> list[Chunk]:
        return sorted(self.layout.chunks(self.rank), key=lambda c: c.local_start)

    def chunks_of(self, rank: int) -> list[Chunk]:
        return sorted(self.layout.chunks(rank), key=lambda c: c.local_start)

    @property
    def local_len(self) -> int:
        return self.layout.local_len


class KVProvider(Protocol):
    """How the ring obtains this rank's K and V.

    A framework implementation reads its paged cache; the test implementation
    slices a tensor. The ring does not care which, which is the point.
    """

    def local_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's K and V, [B, H, S_local, Dh], in local-buffer order."""
        ...


@dataclass
class InMemoryKVProvider:
    """Contiguous in-memory KV. Used by tests and by the standalone server."""

    k: torch.Tensor
    v: torch.Tensor

    def local_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k, self.v
