"""The pipeline separates multiplicity from topology when it reports dim_H.

beta_1 alone overstates the topology of a document. Two relations can be distinct
and still have the same boundary column, and their difference is a cycle that
records an occurrence count rather than a hole. Measured on the Gutenberg store it
is 37% to 85% of dim_H, so reporting dim_H without the split reports mostly
occurrence counting.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from agent.pipeline import AnalysisPipeline

from rexgraph.graph import RexGraph


def _rex(src, tgt):
    r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
    r._ensure_clean()
    return r


@pytest.mark.parametrize("src,tgt,dim_h,mult,genuine", [
    # a triangle (one genuine hole) plus a doubled relation (one multiplicity cycle)
    ([0, 1, 2, 0, 0], [1, 2, 0, 1, 1], 3, 2, 1),
    # a bare triangle: all topology
    ([0, 1, 2], [1, 2, 0], 1, 0, 1),
    # nothing but a repeated relation: all bookkeeping
    ([0, 0, 0], [1, 1, 1], 2, 2, 0),
])
def test_the_stage_reports_the_split(src, tgt, dim_h, mult, genuine):
    res = AnalysisPipeline(_rex(src, tgt))._stage_hodge()
    assert "dim_H_multiplicity_error" not in res, res.get("dim_H_multiplicity_error")
    assert res["dim_H"] == dim_h
    assert res["dim_H_multiplicity"] == mult
    assert res["dim_H_simple"] == genuine
    assert res["dim_H_multiplicity"] + res["dim_H_simple"] == res["dim_H"]


def test_a_simple_complex_reports_no_multiplicity():
    """K5 has no repeated relation, so the split must not invent one and dim_H is
    entirely genuine."""
    a, b = zip(*itertools.combinations(range(5), 2), strict=False)
    res = AnalysisPipeline(_rex(list(a), list(b)))._stage_hodge()
    assert res["dim_H_multiplicity"] == 0
    assert res["dim_H_simple"] == res["dim_H"] == 6


def test_the_split_survives_branching_relations():
    """Arity-general, which matters because the corpus complexes are branching:
    the same 3-ary relation twice is multiplicity, a re-headed one is not."""
    ptr = np.array([0, 3, 6, 9], np.int64)
    idx = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2], np.int64)
    r = RexGraph.from_hypergraph(ptr, idx)
    r._ensure_clean()
    res = AnalysisPipeline(r)._stage_hodge()
    assert "dim_H_multiplicity_error" not in res
    assert res["dim_H_multiplicity"] == 1
    assert res["dim_H_simple"] == res["dim_H"] - 1


def test_the_reading_survives_a_failure_in_the_split():
    """The split is reported next to dim_H, so it must never cost the rest of the
    stage: a failure records itself and leaves every other key intact."""

    r = _rex([0, 1, 2], [1, 2, 0])
    good = AnalysisPipeline(r)._stage_hodge()

    import rexgraph.harmonic_sparse as hs
    original = hs.multiplicity_dimension

    def boom(*a, **k):
        raise RuntimeError("forced")

    hs.multiplicity_dimension = boom
    try:
        bad = AnalysisPipeline(r)._stage_hodge()
    finally:
        hs.multiplicity_dimension = original

    assert "RuntimeError: forced" in bad["dim_H_multiplicity_error"]
    assert bad["dim_H"] == good["dim_H"]
    assert "hodge_fractions" in bad or set(good) - set(bad) <= {
        "dim_H_multiplicity", "dim_H_simple"}


def test_the_split_sums_even_when_a_face_fills_a_multiplicity_cycle():
    """The case that made `dim_H_genuine` the wrong name AND the wrong number: a
    bigon with a face on it has beta_1 = 0, while the chain-level multiplicity
    subspace still has dimension 1. dim_H_simple is a quotient, so the two parts
    still sum to dim_H instead of being clamped at zero."""
    r = _rex([0, 0], [1, 1])
    r.add_faces([[0, 1]], signs=[[1.0, -1.0]])
    r._ensure_clean()
    res = AnalysisPipeline(r)._stage_hodge()
    assert "dim_H_multiplicity_error" not in res
    assert res["dim_H"] == 0
    assert res["dim_H_simple"] == 0
    assert res["dim_H_multiplicity"] == 0
    assert res["dim_H_multiplicity"] + res["dim_H_simple"] == res["dim_H"]


def test_neither_part_is_ever_negative_or_clamped():
    """A clamp would hide exactly the disagreement worth seeing, so the invariant is
    checked rather than enforced."""
    cases = [
        ([0, 1, 2, 0, 0], [1, 2, 0, 1, 1], [[0, 1, 2]], [[1.0, 1.0, 1.0]]),
        ([0, 1, 2, 0, 0], [1, 2, 0, 1, 1], [[3, 4]], [[1.0, -1.0]]),
        ([0, 0, 0], [1, 1, 1], [[0, 1]], [[1.0, -1.0]]),
    ]
    for src, tgt, faces, signs in cases:
        r = _rex(src, tgt)
        r.add_faces(faces, signs=signs)
        r._ensure_clean()
        res = AnalysisPipeline(r)._stage_hodge()
        assert res["dim_H_multiplicity"] >= 0, (src, res)
        assert res["dim_H_simple"] >= 0, (src, res)
        assert res["dim_H_multiplicity"] + res["dim_H_simple"] == res["dim_H"]
