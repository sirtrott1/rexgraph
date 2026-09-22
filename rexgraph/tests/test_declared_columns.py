"""A declared head and a declared share are components of the grade 1 column.

The general one head constructor is b = s - h, with positive rational tail shares
summing to one; the equal share 1/(k-1) is the SPECIALISATION of it. These tests pin three things: the specialisation is unchanged, a
declared column is carried exactly by every reader that claims to read the column, and
every carrier that can only hold the canonical column refuses rather than flattening.
"""
from fractions import Fraction

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.native_rank import primary_columns

DECLARED = [(0, -1), (1, Fraction(1, 4)), (2, Fraction(1, 2)), (3, Fraction(1, 4))]


def declared_rex():
    return RexGraph.from_cells([4, [DECLARED]])


def test_equal_share_is_unchanged_and_declares_nothing():
    plain = RexGraph.from_cells([4, [[0, 1, 2, 3]]])
    assert primary_columns(plain) == [{0: Fraction(-1), 1: Fraction(1, 3),
                                       2: Fraction(1, 3), 3: Fraction(1, 3)}]
    assert not plain.declares_columns
    # an explicit orientation still chooses the head and still stores it at slot zero
    oriented = RexGraph.from_cells([4, [[(1, -1), (0, 1), (2, 1), (3, 1)]]])
    assert oriented.boundary_idx.tolist() == [1, 0, 2, 3]
    assert not oriented.declares_columns
    # and so does a head declared over equal shares through the low level constructor
    head = RexGraph(boundary_ptr=np.array([0, 4]), boundary_idx=np.array([0, 1, 2, 3]),
                    head_slot=np.array([2], np.int32))
    assert head.boundary_idx.tolist() == [2, 0, 1, 3]
    assert not head.declares_columns


def test_declared_share_is_the_column_every_exact_reader_returns():
    rex = declared_rex()
    column = {0: Fraction(-1), 1: Fraction(1, 4), 2: Fraction(1, 2), 3: Fraction(1, 4)}
    assert rex.declares_columns
    assert primary_columns(rex) == [column]
    assert sum(column.values()) == 0                      # still a boundary
    assert np.allclose(rex.B1[:, 0], [-1.0, 0.25, 0.5, 0.25])
    from rexgraph.faces import _exact_b1_block
    assert _exact_b1_block(rex, [0]) == [column]
    # the quadrance is 1 + sum s_i^2 = 11/8, not the equal share's k/(k-1) = 4/3
    assert sum(c * c for c in column.values()) == Fraction(11, 8)


def test_the_geometry_and_the_composite_read_the_declared_column():
    from rexgraph.cells import Cell, composite_binary
    from rexgraph.geometry import relation_quadrance
    rex = RexGraph.from_cells([4, [[0, 1, 2, 3], DECLARED]])
    assert relation_quadrance(rex, 0) == Fraction(4, 3)     # the equal share, k/(k-1)
    assert relation_quadrance(rex, 1) == Fraction(11, 8)    # 1 + sum s_i^2
    declared = composite_binary(Cell(rex, 1, 1))
    assert [str(v) for v in declared.share.values] == ["0", "1/4", "1/2", "1/4"]
    # the integer representative clears the column's own denominator, not k-1
    assert declared.integer_boundary.values.tolist() == [-4, 1, 2, 1]


def test_declared_share_changes_the_cycle_space():
    """Two relations on one support are dependent at equal shares and independent once
    one of them declares a different one, so the share is boundary data, not a metric."""
    same = RexGraph.from_cells([4, [[0, 1, 2, 3], [0, 1, 2, 3]]])
    declared = RexGraph.from_cells([4, [[0, 1, 2, 3], DECLARED]])
    from rexgraph.native_rank import boundary_rank
    assert boundary_rank(same, 1) == 1
    assert boundary_rank(declared, 1) == 2


def test_declared_column_round_trips_through_the_state():
    rex = declared_rex()
    state = to_state(rex)
    assert state.header["format_version"] == 9
    assert {"column_head", "column_share_num", "column_share_den"} <= set(state.tensors)
    back = from_state(state)
    assert back.declares_columns
    assert primary_columns(back) == primary_columns(rex)
    # a complex that declares nothing writes no such tensor and keeps its version
    plain = to_state(RexGraph.from_cells([4, [[0, 1, 2, 3]]]))
    assert "column_head" not in plain.tensors
    assert plain.header["format_version"] < 9


def test_an_inadmissible_declaration_is_refused_by_its_own_constraint():
    with pytest.raises(ValueError, match="summing to"):
        RexGraph.from_cells([4, [[(0, -1), (1, Fraction(1, 4)), (2, Fraction(1, 4)),
                                  (3, Fraction(1, 4))]]])
    # an explicit cell cannot even spell a zero coefficient, so the zero tail is refused
    # where it can be written: on the constructor that takes the share vector itself
    with pytest.raises(ValueError, match="s_i > 0"):
        RexGraph(boundary_ptr=np.array([0, 3]), boundary_idx=np.array([0, 1, 2]),
                 head_slot=np.array([0], np.int32), shares=[0, 1, 0])
    with pytest.raises(ValueError, match="1\\^T s = 1"):
        RexGraph(boundary_ptr=np.array([0, 3]), boundary_idx=np.array([0, 1, 2]),
                 head_slot=np.array([0], np.int32),
                 shares=[0, Fraction(1, 3), Fraction(1, 3)])
    with pytest.raises(ValueError, match="s_h = 0"):
        RexGraph(boundary_ptr=np.array([0, 3]), boundary_idx=np.array([0, 1, 2]),
                 head_slot=np.array([0], np.int32),
                 shares=[Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)])
    with pytest.raises(ValueError, match="supported head"):
        RexGraph(boundary_ptr=np.array([0, 3]), boundary_idx=np.array([0, 1, 2]),
                 head_slot=np.array([5], np.int32))


def test_carriers_that_hold_only_the_canonical_column_refuse():
    rex = declared_rex()
    with pytest.raises(ValueError, match="canonical column"):
        rex._require_canonical_columns("a probe")
    with pytest.raises(ValueError, match="canonical column"):
        assert rex.clique_expansion is None      # the read is what refuses
    from rexgraph.graph import TemporalRex
    with pytest.raises(ValueError, match="canonical column"):
        TemporalRex([]).append_snapshot(rex, at=0.0)
    from rexgraph.joins import join
    with pytest.raises(ValueError, match="canonical column"):
        join(rex, rex, labels_r=["a", "b", "c", "d"], labels_s=["a", "b", "c", "d"])


def test_every_channel_reading_carries_a_declared_column():
    """All three readings answer with the declared column: T is the quadrance
    1 + sum s_i^2 = 11/8, not the equal share's k/(k-1) = 4/3."""
    from rexgraph.rational_trig import exact_channel_diagonals
    from rexgraph.sparse_character import channel_diagonals, channel_diagonals_integer
    rex = declared_rex()
    exact, names = exact_channel_diagonals(rex)
    assert exact[names[0]][0] == Fraction(11, 8)
    plain, plain_names = exact_channel_diagonals(RexGraph.from_cells([4, [[0, 1, 2, 3]]]))
    assert plain[plain_names[0]][0] == Fraction(4, 3)
    # the compiled tower, which reads one coefficient per incidence
    assert channel_diagonals(rex)["L1_down"][0] == pytest.approx(11 / 8)
    # and the integer carrier, over its own common denominator
    numerators, scale = channel_diagonals_integer(rex)
    assert Fraction(int(numerators["L1_down"][0]), scale) == Fraction(11, 8)


def test_the_compiled_tower_agrees_with_the_exact_reading_and_leaves_canonical_alone():
    from rexgraph.core._channel_tower import channel_diagonals_any_arity
    from rexgraph.rational_trig import exact_channel_diagonals
    from rexgraph.sparse_character import channel_diagonals
    # a complex mixing declared relations, a pair and a witness, sharing vertices
    rex = RexGraph.from_cells([6, [
        [(0, -1), (1, Fraction(1, 6)), (2, Fraction(1, 3)), (3, Fraction(1, 2))],
        [(2, -1), (3, Fraction(2, 5)), (4, Fraction(3, 5))],
        [0, 4], [5],
    ]])
    fast = channel_diagonals(rex)
    exact, names = exact_channel_diagonals(rex)
    for key, name in zip(("L1_down", "L_O", "L_SG", "L_C"), names, strict=True):
        assert np.asarray(fast[key]) == pytest.approx(
            [float(v) for v in exact[name]], abs=1e-12)
    # the canonical reading is untouched: same arrays, threaded or serial
    plain = RexGraph.from_cells([5, [[0, 1, 2, 3], [0, 4], [2, 3, 4]]])
    bp = np.asarray(plain._boundary_ptr, np.int32)
    bi = np.asarray(plain._boundary_idx, np.int32)
    serial = channel_diagonals_any_arity(bp, bi, int(plain.nV), None, 1)
    threaded = channel_diagonals_any_arity(bp, bi, int(plain.nV), None, 4)
    for a, b in zip(serial, threaded, strict=True):
        assert a.tobytes() == b.tobytes()
