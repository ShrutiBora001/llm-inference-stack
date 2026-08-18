"""persistent_ring — Phase 3 scaffold. Not implemented.

The contract suite imports these only under the matching env flag, so the suite
stays coherent on a laptop. See PHASE3.md in this package.
"""

from __future__ import annotations

_NOT_IMPLEMENTED = "Phase 3 scaffold. See src/lis/kernels/PHASE3.md"


def launch_count_for_traversal(world_size: int) -> int:
    """Kernel launches for one full ring traversal, from an nsys trace.

    Must be 1. If it is `world_size`, the kernel is being relaunched per step
    and is not persistent — the one thing raw CUDA was chosen for.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)
