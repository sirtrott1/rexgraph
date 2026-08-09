"""One row per cell, carrying what that cell actually is.

The per-cell table was the useful half of the old dashboard: a cell with its readings
attached, so a question about one cell has one place to look. Two things about its shape
were wrong.

An edge row named a `source` and a `target`, which is the arity-2 coordinate of a
relation rather than the relation. A branching column of arity k had k-2 of its boundary
nowhere in the row. And channels were positional, so `L1_down` and `L_O` came back as
"channel 0" and "channel 1", two numbers that are equal on an unweighted complex for a
structural reason and read as a coincidence.
"""
from __future__ import annotations

import numpy as np
import pytest
from agent.cell_view import cells, edge_rows, vertex_rows

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    g = RexGraph(sources=np.array([0, 1, 2, 2, 3, 4], dtype=np.int32),
                 targets=np.array([1, 2, 0, 3, 4, 2], dtype=np.int32))
    autoface(g, 3)
    return g


@pytest.fixture
def branching():
    """One 4-ary relation and one 2-ary, so arity is visible in the rows."""
    return RexGraph.from_hypergraph(np.array([0, 4, 6], dtype=np.int32),
                                    np.array([0, 1, 2, 3, 0, 1], dtype=np.int32))


#### a relation carries its whole boundary


def test_a_branching_relation_is_one_row(branching):
    rows = edge_rows(branching, positions=False)
    assert len(rows) == branching.nE == 2


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_a_k_ary_relation_reports_all_k_boundary_cells(k):
    """The defect this replaces: source/target could hold two of them."""
    g = RexGraph.from_hypergraph(np.array([0, k], dtype=np.int32),
                                 np.arange(k, dtype=np.int32))
    row = edge_rows(g, positions=False)[0]
    assert row["arity"] == k
    assert len(row["boundary"]) == k
    assert len(row["boundary_index"]) == k


def test_the_boundary_matches_the_complex(branching):
    rows = edge_rows(branching, positions=False)
    branching._ensure_clean()
    bp = np.asarray(branching._boundary_ptr)
    bi = np.asarray(branching._boundary_idx)
    for e, row in enumerate(rows):
        assert row["boundary_index"] == [int(v) for v in bi[bp[e]:bp[e + 1]]]


def test_labels_are_used_for_the_boundary(branching):
    row = edge_rows(branching, labels=["a", "b", "c", "d"], positions=False)[0]
    assert row["boundary"] == ["a", "b", "c", "d"]


#### channels are named


def test_channel_shares_are_keyed_by_name(rex):
    out = cells(rex, positions=False, limit=1)
    assert out["channels"]
    assert set(out["vertices"][0]["phi"]) == set(out["channels"])
    assert set(out["relations"][0]["chi"]) == set(out["channels"])


def test_the_named_channels_expose_a_structural_equality(rex):
    """L1_down and L_O share a diagonal on an unweighted complex. Named, that reads as
    the fact it is; positional, it reads as two unrelated numbers that happen to agree."""
    chi = edge_rows(rex, positions=False, limit=1)[0]["chi"]
    assert chi["L1_down"] == chi["L_O"]


def test_the_shares_of_a_cell_sum_to_one(rex):
    """chi and phi live on the simplex over the channels."""
    for row in cells(rex, positions=False)["relations"]:
        assert sum(row["chi"].values()) == pytest.approx(1.0, abs=1e-5)


#### both coherence towers


def test_a_vertex_carries_both_coherences(rex):
    """The exact per-vertex kappa against the global Green's function, and the O(nnz)
    local companion. Disagreement is a fact about the vertex, not an error."""
    row = vertex_rows(rex, positions=False, limit=1)[0]
    assert row["coherence"] is not None
    assert row["local_coherence"] is not None


#### the signal split


def test_a_signal_is_split_per_relation(rex):
    rows = edge_rows(rex, signal=np.ones(int(rex.nE)), positions=False)
    for row in rows:
        assert set(row["hodge"]) == {"gradient", "curl", "harmonic"}


def test_the_hodge_parts_reconstruct_the_signal(rex):
    """Orthogonal by the chain condition, so the three parts add back to the input."""
    sig = np.arange(1.0, rex.nE + 1.0)
    rows = edge_rows(rex, signal=sig, positions=False)
    total = np.array([sum(r["hodge"].values()) for r in rows])
    assert np.allclose(total, sig, atol=1e-5)


def test_divergence_appears_on_vertices_when_a_signal_is_given(rex):
    rows = vertex_rows(rex, signal=np.ones(int(rex.nE)), positions=False)
    assert all("divergence" in r for r in rows)


def test_no_signal_means_no_signal_columns(rex):
    assert "hodge" not in edge_rows(rex, positions=False)[0]
    assert "divergence" not in vertex_rows(rex, positions=False)[0]


#### what is deliberately absent


@pytest.mark.parametrize("gone", ["fiedler", "fiedlerLO", "partL0", "partLO",
                                  "partL1a", "pagerank", "betweenness",
                                  "clustering", "community", "source", "target"])
def test_the_retired_columns_are_not_here(rex, gone):
    """Fiedler and its partitions report where a cut fell; the standard baselines are
    the comparison column; source/target is the arity-2 assumption."""
    out = cells(rex, positions=False, limit=1)
    assert gone not in out["vertices"][0]
    assert gone not in out["relations"][0]


#### shape


def test_positions_agree_with_the_coordinate_layer(rex):
    from agent.graph_view import character_positions
    rows = vertex_rows(rex, limit=0)
    at = character_positions(rex, grade="vertex", dim=3)["positions"]
    assert np.allclose([r["at"] for r in rows], np.round(at, 6), atol=1e-6)


def test_a_limit_truncates_without_changing_the_rows(rex):
    full = cells(rex, positions=False)["relations"]
    short = cells(rex, positions=False, limit=2)["relations"]
    assert short == full[:2]


def test_an_unknown_grade_is_refused(rex):
    with pytest.raises(ValueError, match="grade must be"):
        cells(rex, grade="faces")
