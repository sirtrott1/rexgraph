"""One call that says what a complex is, with its sections checked against each other.

A report that prints Euler in one place and Betti in another has stated two numbers that
must agree and not compared them. Both are integers over the integers here, so the
comparison is exact and a disagreement is a defect rather than something to interpret.
That cross-checking is the part a section-by-section summary cannot have, and it is why
this is assembled rather than a list of calls.

The arity distribution is the other reason. `nE` alone does not distinguish a complex
whose relations are all pairwise from one carrying 4-ary relations, and that is the
first thing worth knowing about a relational complex.
"""
from __future__ import annotations

import numpy as np
import pytest
from agent.overview import consistency_of, overview, shape_of

from rexgraph.faces import auto_hyperface, autoface
from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    g = RexGraph(sources=np.array([0, 1, 2, 2, 3, 4], dtype=np.int32),
                 targets=np.array([1, 2, 0, 3, 4, 2], dtype=np.int32))
    autoface(g, 3)
    return g


@pytest.fixture
def branching():
    g = RexGraph.from_hypergraph(
        np.array([0, 4, 6, 8, 10, 12], dtype=np.int32),
        np.array([0, 1, 2, 3, 0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32))
    auto_hyperface(g)
    return g


#### arity is a first-class reading


def test_the_arity_distribution_is_reported(branching):
    shape = shape_of(branching)
    assert shape["arity"] == {"2": 4, "4": 1}
    assert shape["has_branching"] is True
    assert shape["n_branching"] == 1
    assert shape["max_arity"] == 4


def test_a_pairwise_complex_says_so(rex):
    shape = shape_of(rex)
    assert set(shape["arity"]) == {"2"}
    assert shape["has_branching"] is False


def test_a_dropped_face_is_counted_not_absorbed():
    """A face that arrived and does not bound is excluded from the homology, so the gap
    between declared and surviving is worth stating."""
    g = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                 targets=np.array([1, 2, 0, 0], dtype=np.int32))
    g.add_faces([[0, 1, 3]], [[1.0, 1.0, 1.0]])
    shape = shape_of(g)
    assert shape["nF_declared"] == 1
    assert shape["nF"] == 0
    assert shape["faces_dropped"] == 1


#### the cross-checks


def test_the_identities_hold_on_a_valid_complex(rex):
    out = overview(rex, cells=False)["consistency"]
    assert out["euler_agrees"] is True
    assert out["harmonic_equals_betti"] is True
    assert out["chain_valid"] is True
    assert out["ok"] is True


def test_the_identities_hold_on_a_branching_complex(branching):
    out = overview(branching, cells=False)["consistency"]
    assert out["ok"] is True


def test_a_face_that_does_not_bound_is_named(rex):
    """The check earning its place: it fails, and says which face."""
    g = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                 targets=np.array([1, 2, 0, 0], dtype=np.int32))
    g.add_faces([[0, 1, 3]], [[1.0, 1.0, 1.0]])
    out = overview(g, cells=False)["consistency"]
    assert out["chain_valid"] is False
    assert out["unbounded_faces"] == [0]
    assert out["ok"] is False


def test_euler_is_computed_two_ways_and_compared(rex):
    out = overview(rex, cells=False)["consistency"]
    assert out["euler_from_counts"] == out["euler_from_betti"]


def test_the_harmonic_dimensions_are_the_betti_numbers(rex):
    tower = rex.rank_tower()
    harmonic = [g["harmonic"] for g in tower["grades"]]
    assert harmonic[:len(rex.betti)] == [int(b) for b in rex.betti]
    assert consistency_of(rex, tower)["harmonic_equals_betti"] is True


#### flow


def test_without_a_signal_the_dimensions_are_still_reported(rex):
    """How many independent directions each part has is structural and exact, where the
    split of a particular signal is a fact about that signal."""
    flow = overview(rex, cells=False)["flow"]
    assert set(flow["dimensions"]) == {"gradient", "curl", "harmonic"}
    assert "energy" not in flow


def test_with_a_signal_the_parts_are_additive(rex):
    """The chain condition makes the cross terms vanish, so a residual would mean the
    decomposition is wrong rather than that the data is awkward."""
    flow = overview(rex, signal=np.arange(1.0, rex.nE + 1.0), cells=False)["flow"]
    assert flow["cross_residual"] == pytest.approx(0.0, abs=1e-6)
    assert sum(flow["share"].values()) == pytest.approx(1.0, abs=1e-6)


def test_a_signal_of_the_wrong_length_is_refused(rex):
    with pytest.raises(ValueError, match="entries for"):
        overview(rex, signal=np.ones(rex.nE + 3), cells=False)


#### character and shape of the answer


def test_the_character_is_keyed_by_channel_name(rex):
    ch = overview(rex, cells=False)["character"]
    assert ch["channels"]
    assert set(ch["relations"]["mean"]) == set(ch["channels"])
    assert ch["relations"]["dominant"] in ch["channels"]


def test_the_cells_are_included_on_request(rex):
    out = overview(rex, labels=list("abcde"), cells=True, limit=2, positions=False)
    assert len(out["cells"]["relations"]) == 2
    assert out["cells"]["vertices"][0]["label"] == "a"


def test_the_cells_can_be_left_out(rex):
    assert "cells" not in overview(rex, cells=False)


@pytest.mark.parametrize("section", [
    "shape", "homology", "character", "flow", "curvature", "consistency"])
def test_every_section_is_present(rex, section):
    assert section in overview(rex, cells=False)


@pytest.mark.parametrize("gone", ["fiedler", "partitions", "standard_metrics",
                                  "pagerank", "spectra"])
def test_the_retired_sections_are_not_here(rex, gone):
    assert gone not in overview(rex, cells=False)
