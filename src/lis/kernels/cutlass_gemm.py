"""cutlass_gemm — Phase 3 scaffold. Not implemented.

The contract suite imports these only under the matching env flag, so the suite
stays coherent on a laptop. See PHASE3.md in this package.
"""

from __future__ import annotations

_NOT_IMPLEMENTED = "Phase 3 scaffold. See src/lis/kernels/PHASE3.md"


def comparison_report() -> dict:
    """Measured Triton vs CUTLASS on identical shapes.

    Must carry triton_tflops, cutlass_tflops and tensor_core_utilization — the
    contract check fails if any is missing, because a comparison that was
    asserted rather than measured is not a comparison.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)
