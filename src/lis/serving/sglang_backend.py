"""sglang_backend — Phase 2 scaffold.

Not implemented. The contract suite imports these symbols only when
sglang_TEST=1 is set, so the suite stays coherent on a laptop while the
branch is empty.

See README.md in this package for the entry point and the paged<->chunk
mapping that is the real work.
"""

from __future__ import annotations

_NOT_IMPLEMENTED = (
    "Phase 2 scaffold: implement against the installed framework. "
    "See src/lis/serving/README.md"
)


def prefix_cache_hit_rate() -> float:
    """Radix-tree hit rate on a shared-prefix workload. Near zero means the
    benchmark chose a workload that cannot distinguish SGLang from anything
    else, which is the failure this check exists to catch."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
