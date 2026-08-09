"""The analysis surface off its dense operators, checked against them.

`analyze` was cubic: 37.8s at nE=2396 and about four minutes at nE=4797. The cost was
not the channel character or the spectral bundle, both of which are already sparse. It
was three places that took a whole spectrum to report a bounded reading, and one that
counted eigenvalues under a cutoff to get an integer the rank tower already has.

Each is checked against the dense route rather than assumed equal to it. The dense
routes stay reachable, which is what makes these tests possible.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    """Two triangles sharing a vertex, one filled."""
    g = RexGraph(sources=np.array([0, 1, 2, 2, 3, 4], dtype=np.int32),
                 targets=np.array([1, 2, 0, 3, 4, 2], dtype=np.int32))
    autoface(g, 3)
    return g


#### the graded state


@pytest.mark.parametrize("t", [0.0, 0.5, 1.7, 3.0])
def test_the_graded_state_matches_the_mode_sum(rex, t):
    """`e^{-iDt}` applied to one state is a Chebyshev matvec; the mode sum needs the
    whole spectrum to answer the same question."""
    from rexgraph.core import _dirac
    psi0 = np.ascontiguousarray(
        _dirac.canonical_collapse(rex.B1, rex.nV, rex.nE, rex.nF_hodge, 0), dtype=float)
    re_fast, im_fast = rex.graded_state(t=t)
    evals, evecs = rex._dirac_eigen
    re_slow, im_slow = _dirac.schrodinger_evolve(evals, evecs, psi0, t)
    assert np.allclose(re_fast, re_slow, atol=1e-8)
    assert np.allclose(im_fast, im_slow, atol=1e-8)


def test_the_graded_state_is_norm_preserving(rex):
    """e^{-iDt} is unitary, so amplitude moves between grades without being lost."""
    re0, im0 = rex.graded_state(t=0.0)
    re1, im1 = rex.graded_state(t=2.0)
    n0 = float(np.sum(re0 ** 2 + im0 ** 2))
    n1 = float(np.sum(re1 ** 2 + im1 ** 2))
    assert n1 == pytest.approx(n0, rel=1e-6)


#### bounded spectra


def test_the_dirac_low_spectrum_matches_the_dense_one(rex):
    """Compared by MAGNITUDE, which is the well-defined question.

    D's spectrum is symmetric about zero and degenerate, so "the k nearest zero" can be
    satisfied by different signed picks when the cut falls inside a tied group: here
    {0, +/-1, +/-1.732 twice}, where k=6 takes three of four tied at 1.732. Both routes
    are right and the signed lists still differ, so asserting on them would be pinning
    an arbitrary tie-break rather than the spectrum.
    """
    from rexgraph.analysis import _dirac_low_spectrum
    k = 6
    fast = np.sort(np.abs(np.asarray(_dirac_low_spectrum(rex, k), dtype=float)))
    dense = np.asarray(rex.dirac_eigenvalues, dtype=float)
    slow = np.sort(np.abs(dense[np.argsort(np.abs(dense))[:k]]))
    assert np.allclose(fast, slow, atol=1e-8)


def test_the_field_frequencies_match_the_dense_ones(rex):
    from rexgraph.analysis import _low_frequencies
    from rexgraph.field_propagator import assemble_field_operator
    M = assemble_field_operator(rex)
    fast = np.asarray(_low_frequencies(M, 4), dtype=float)
    _evals, _evecs, freqs = rex.field_eigen
    assert np.allclose(fast, np.sort(np.asarray(freqs, dtype=float))[:4], atol=1e-6)


def test_the_sparse_field_operator_is_the_dense_one(rex):
    from rexgraph.field_propagator import assemble_field_operator
    dense = np.asarray(rex.field_operator[0], dtype=float)
    assert np.allclose(assemble_field_operator(rex).toarray(), dense, atol=1e-9)


#### the kernel count is an integer, not a cutoff


def test_the_dirac_kernel_count_is_the_betti_sum(rex):
    """dim ker(D) = sum of the Betti numbers. Counting |eval| < 1e-8 asks the same
    question of a dense spectrum and answers it with a magnitude."""
    from rexgraph.analysis import analyze
    section = analyze(rex).get("dirac") or {}
    assert section, "the dirac section is missing"
    assert section["n_zero"] == sum(int(b) for b in rex.betti)


#### what is reported by default


def test_the_standard_baselines_are_off_by_default(rex):
    """PageRank, betweenness, clustering and Louvain are the comparison baselines
    rather than this library's readings, and betweenness alone is O(nV * nE)."""
    from rexgraph.analysis import analyze
    off = analyze(rex)["analysis"]["standard_metrics"]
    assert off["n_communities"] == 0
    on = analyze(rex, standard_metrics=True)["analysis"]["standard_metrics"]
    assert on["n_communities"] >= 1


def test_the_fiedler_value_is_off_by_default(rex):
    """An approximate slice, and 9.4s of a 38s call at nE=2400."""
    from rexgraph.analysis import analyze
    assert analyze(rex)["coupling"]["fiedler_L1"] == 0.0
    assert analyze(rex, spectral_extras=True)["coupling"]["fiedler_L1"] >= 0.0


def test_the_analysis_still_produces_its_whole_contract(rex):
    """Gating a reading must not drop the key: a consumer reading it should see a
    value it can recognise as absent, not a KeyError."""
    from rexgraph.analysis import analyze
    data = analyze(rex)
    for section in ("meta", "vertices", "edges", "topology", "coupling",
                    "spectra", "hodge", "energy", "analysis", "dirac"):
        assert section in data, f"{section} went missing"
    for section in ("standard_metrics", "partitions", "field", "structure"):
        assert section in data["analysis"], f"analysis.{section} went missing"
