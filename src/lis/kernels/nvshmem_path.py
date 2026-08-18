"""nvshmem_path — Phase 3 scaffold. Not implemented.

The contract suite imports these only under the matching env flag, so the suite
stays coherent on a laptop. See PHASE3.md in this package.
"""

from __future__ import annotations

_NOT_IMPLEMENTED = "Phase 3 scaffold. See src/lis/kernels/PHASE3.md"


def enabled_regimes() -> set[str]:
    """Where NVSHMEM is wired in. Must never contain "prefill".

    Measurement says communication is 1.1% of prefill runtime and already
    half-hidden by overlap, so NVSHMEM cannot win there. The contract check
    fails if it appears, which is the guard against reaching for it because it
    is impressive rather than because it helps.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)
