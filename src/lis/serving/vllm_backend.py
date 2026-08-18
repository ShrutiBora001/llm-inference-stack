"""vllm_backend — Phase 2 scaffold.

Not implemented. The contract suite imports these symbols only when
vllm_TEST=1 is set, so the suite stays coherent on a laptop while the
branch is empty.

See README.md in this package for the entry point and the paged<->chunk
mapping that is the real work.
"""

from __future__ import annotations

_NOT_IMPLEMENTED = (
    "Phase 2 scaffold: implement against the installed framework. "
    "See src/lis/serving/README.md"
)


def sample_block_table() -> list[int]:
    """Block IDs from a live paged allocation. Used by the contract check that
    fails if the table is a single contiguous block — i.e. paging never
    happened and PagedAttention is unexercised."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def request_intervals() -> list[tuple[float, float]]:
    """(start, end) per request. The contract check requires overlap; without it
    this is a queue with extra steps rather than continuous batching."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
