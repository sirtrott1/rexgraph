"""The TemporalRex constructor takes connectivity, not complexes.

`TemporalRex(snapshots)` carries connectivity only: reconstructed snapshots have
w_E=None and signs=None. `append_snapshot(rex)` is the path that keeps edge
attribution. Passing RexGraph objects to the constructor would therefore drop w_E and
signs, so it is refused at construction rather than failing later inside `at()`.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex


def _rex(n_extra=0):
    src = [0, 1, 2] + [3 + i for i in range(n_extra)]
    tgt = [1, 2, 0] + [4 + i for i in range(n_extra)]
    return RexGraph(sources=np.asarray(src, np.int32), targets=np.asarray(tgt, np.int32))


def test_connectivity_tuples_are_the_constructor_input():
    graphs = [_rex(i) for i in range(3)]
    t = TemporalRex([(g.sources, g.targets) for g in graphs])
    assert t.T == 3
    assert int(t.at(0).nE) == 3


def test_passing_complexes_is_refused():
    with pytest.raises(TypeError, match="takes connectivity tuples"):
        TemporalRex([_rex(0), _rex(1)])


def test_the_error_points_at_append_snapshot():
    with pytest.raises(TypeError, match="append_snapshot"):
        TemporalRex([_rex(0)])


def test_the_error_names_the_offending_index():
    with pytest.raises(TypeError, match=r"snapshots\[1\]"):
        TemporalRex([(_rex(0).sources, _rex(0).targets), _rex(1)])


def test_general_mode_names_the_boundary_form():
    with pytest.raises(TypeError, match="boundary_ptr, boundary_idx"):
        TemporalRex([_rex(0)], general=True)


def test_append_snapshot_takes_complexes():
    """The documented path for full complexes, which keeps w_E and signs."""
    t = TemporalRex([])
    for i in range(3):
        t.append_snapshot(_rex(i))
    assert t.T == 3
    assert int(t.at(2).nE) == 5


def test_an_empty_store_is_still_valid():
    assert TemporalRex([]).T == 0
