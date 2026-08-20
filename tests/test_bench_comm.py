"""Communication accounting. Bandwidth needs GPUs; the byte model does not.

The byte model is the half worth testing on a laptop: if the implementation's
traffic ever diverges from the closed form, that is either a wrong model or an
implementation moving data it should not, and neither shows up as a wrong
answer.
"""

from __future__ import annotations

import pytest
import torch

from lis.bench_comm import measured_bytes_per_rank, predicted_bytes_per_rank

CFG = dict(seq=32768, world_size=4, n_heads=32, head_dim=128, bytes_per_elem=2)


def test_measured_matches_the_closed_form():
    """The contract check, runnable without hardware."""
    got, want = measured_bytes_per_rank(**CFG), predicted_bytes_per_rank(**CFG)
    assert abs(got - want) / want < 0.05


def test_reproduces_the_upstream_384_mib():
    """Upstream reported 384 MiB received per rank per iteration at S=32768."""
    assert measured_bytes_per_rank(**CFG) / 2**20 == pytest.approx(384.0, rel=0.01)


def test_traffic_scales_linearly_with_sequence():
    a = measured_bytes_per_rank(**{**CFG, "seq": 32768})
    b = measured_bytes_per_rank(**{**CFG, "seq": 65536})
    assert b == pytest.approx(2 * a)


def test_more_ranks_move_more_bytes_per_rank():
    """(P-1) hops of an S/P shard: total per rank grows toward S as P rises,
    which is why the ring's advantage is memory rather than traffic."""
    p2 = measured_bytes_per_rank(**{**CFG, "world_size": 2})
    p8 = measured_bytes_per_rank(**{**CFG, "world_size": 8})
    assert p8 > p2


def test_bandwidth_helpers_require_a_process_group():
    from dattn.comms import DistEnv
    from lis.bench_comm import allreduce_bandwidth, ring_bandwidth

    solo = DistEnv(rank=0, world_size=1, local_rank=0,
                   device=torch.device("cpu"), backend="gloo")
    with pytest.raises(RuntimeError, match="process group"):
        ring_bandwidth(solo)
    with pytest.raises(RuntimeError, match="process group"):
        allreduce_bandwidth(solo)
