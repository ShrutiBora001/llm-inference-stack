"""KV provider backed by a framework's paged cache.

Bridges `lis.serving.paging` (block table -> spans -> contiguous) to the
`KVProvider` interface the ring consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..serving.paging import gather_spans, global_positions, layout_to_spans
from .adapter import CPContext


@dataclass
class PagedKVProvider:
    """Reads this rank's K and V out of paged storage.

    k_cache, v_cache: [num_blocks, page_size, n_heads, head_dim]
    block_table:      logical page -> physical block, per the framework
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: list[int]
    page_size: int
    ctx: CPContext

    def _spans(self):
        return layout_to_spans(
            self.ctx.layout, self.ctx.rank, self.block_table, self.page_size
        )

    def local_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        spans = self._spans()
        return gather_spans(self.k_cache, spans), gather_spans(self.v_cache, spans)

    def positions(self, device=None) -> torch.Tensor:
        """Global positions for this rank's tokens, in local-buffer order.

        Hand these to `lis.rope.apply_rope`. Deriving them from a local index is
        the silent failure this project keeps guarding against.
        """
        return global_positions(self._spans(), device=device)
