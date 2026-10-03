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
    assert rex._exact_column_norms_B1() == [Fraction(11, 8)]
    assert rex.trace_T == Fraction(11, 8)


def test_declared_column_drives_the_full_exact_coupling_tower():
    """The coupling invariants consume the declared column, never an arity template."""
    rex = RexGraph.from_cells([4, [DECLARED, DECLARED], [[0, 1]]])
    assert rex.trace_T == Fraction(11, 4)
    assert rex.trace_L1 == Fraction(2)
    assert rex.c0_squared == Fraction(8, 11)
    assert rex.c2_E == Fraction(64, 121)
    assert rex.c2_H == Fraction(1)
    assert rex.c2_E * rex.c2_H == rex.c0_squared * rex.c0_squared


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
    assert state.header["format_version"] == 10
    assert {"column_head", "column_share_num", "column_share_den"} <= set(state.tensors)
    back = from_state(state)
    assert back.declares_columns
    assert primary_columns(back) == primary_columns(rex)
    # Absence is explicit in the same native version; no column is invented.
    plain = to_state(RexGraph.from_cells([4, [[0, 1, 2, 3]]]))
    assert "column_head" not in plain.tensors
    assert plain.header["format_version"] == 10


def test_declared_column_round_trips_through_temporal_state_and_signal():
    from rexgraph.graph import TemporalRex
    from rexgraph.io.temporal_state import from_temporal_state, to_temporal_state
    from rexgraph.temporal_signal import temporal_signal

    first = declared_rex()
    second = RexGraph.from_cells([4, [[
        (0, -1),
        (1, Fraction(1, 2)),
        (2, Fraction(1, 4)),
        (3, Fraction(1, 4)),
    ]]])
    history = TemporalRex([])
    history.append_snapshot(first, at=1.0)
    history.append_snapshot(second, at=2.0)

    # A delta does not yet carry unequal rational shares, so declaration bearing
    # transitions are exact full checkpoints rather than lossy arity reconstructions.
    assert history._index_cp_times.tolist() == [0, 1]
    assert history._index_deltas == [None, None]
    assert primary_columns(history.at(0)) == primary_columns(first)
    assert primary_columns(history.reconstruct_at(1)) == primary_columns(second)

    state = to_temporal_state(history)
    assert state.header["temporal_state_version"] == 4
    assert "checkpoint/0/column_share_num" in state.tensors
    restored = from_temporal_state(state)
    assert primary_columns(restored.reconstruct_at(0)) == primary_columns(first)
    assert primary_columns(restored.reconstruct_at(1)) == primary_columns(second)

    signal = temporal_signal(restored, 1)
    assert signal.event((0, 1, 2, 3)).boundary_changed
    assert signal.source_field("structural").values.tolist() == [
        Fraction(0), Fraction(1, 4), Fraction(-1, 4), Fraction(0)
    ]


def test_temporal_append_bundle_preserves_unequal_declared_shares(tmp_path):
    from rexgraph.graph import TemporalRex
    from rexgraph.io import load, save
    first = RexGraph.from_cells([4, [[
        (0, -1), (1, Fraction(1, 2)), (2, Fraction(1, 4)), (3, Fraction(1, 4)),
    ]]])
    second = declared_rex()
    history = TemporalRex([])
    history.append_snapshot(first, at=1.0)
    history.append_snapshot(second, at=2.0)
    path = tmp_path / "declared.rcbd"
    save(path, history)
    restored = load(path)
    assert primary_columns(restored.reconstruct_at(0)) == primary_columns(first)
    result = restored.reconstruct_at(1)
    assert primary_columns(result) == primary_columns(second)
    assert result._exact_column_norms_B1() == [Fraction(11, 8)]


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


def test_a_declared_share_follows_its_own_relation_through_an_edit():
    """Removing a relation must drop its head slot AND its incidence slots.

    `head_slot` is one entry per relation and `share_num`/`share_den` one per incidence
    in CSR order, so both have to follow the renumbering the boundary takes. They did
    not: the arrays kept their old length and offsets while the boundary was shortened,
    so the surviving relation read the REMOVED relation's shares, silently, because the
    only length check is at construction.
    """
    from fractions import Fraction as F

    from rexgraph import native_rank
    from rexgraph.graph import RexGraph

    def build():
        return RexGraph(
            boundary_ptr=np.array([0, 3, 6], dtype=np.int64),
            boundary_idx=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            w_E=np.asarray([F(10), F(20)], dtype=object),
            head_slot=np.asarray([0, 0], dtype=np.int32),
            shares=[F(0), F(1, 4), F(3, 4), F(0), F(1, 2), F(1, 2)])

    # the weight identifies which relation survived, and its own shares must follow it
    own = {F(10): [F(1, 4), F(3, 4)], F(20): [F(1, 2), F(1, 2)]}
    for mask in ([True, False], [False, True]):
        rex = build()
        rex.remove_edges(np.asarray(mask))
        rex._ensure_clean()
        weight = list(rex.edge_metric_exact)[0]
        column = native_rank.primary_columns(rex)[0]
        tails = sorted(value for value in column.values() if value > 0)
        assert tails == sorted(own[weight]), (
            f"survivor carries weight {weight} but shares {tails}, which belong to the "
            "relation that was removed")
        assert sorted(column.values())[0] == F(-1), "the head must still carry -1"


def test_a_declared_complex_still_declares_after_an_edit():
    """The declaration survives as a declaration, not as a silently canonical column."""
    from fractions import Fraction as F

    from rexgraph.graph import RexGraph
    rex = RexGraph(
        boundary_ptr=np.array([0, 3, 6], dtype=np.int64),
        boundary_idx=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        w_E=np.asarray([F(1), F(1)], dtype=object),
        head_slot=np.asarray([0, 0], dtype=np.int32),
        shares=[F(0), F(1, 4), F(3, 4), F(0), F(1, 2), F(1, 2)])
    assert rex.declares_columns
    rex.remove_edges(np.asarray([True, False]))
    rex._ensure_clean()
    assert rex.declares_columns, "the surviving declared relation still declares"
    assert len(rex._declaration.head_slot) == rex.nE
    assert len(rex._declaration.share_den) == len(np.asarray(rex._boundary_idx))
