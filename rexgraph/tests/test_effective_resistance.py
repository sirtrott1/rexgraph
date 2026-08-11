"""Effective resistance on a complex whose L0 has a kernel, and Foster's identity.

R_eff(e) = b_e^T L0^+ b_e was computed by running CG on the raw singular L0. A
boundary column lies in the range so that converges in exact arithmetic, but the
solver clamped a zero curvature up to 1e-300 instead of treating it as "no progress
possible", and rz/1e-300 is an overflow. On a 66-component ontology slice 348 of 385
relations came back NaN, and `agentic_reading` ranks load_bearing on this.

The fix is the harmonic-regularized inverse L0^+ = (L0 + P_H)^-1 - P_H, with P_H the
projector onto the known kernel (the component indicators). b_e is orthogonal to that
kernel, so P_H b_e = 0 and R_eff = b_e^T (L0 + P_H)^-1 b_e, on an SPD operator.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.graph import RexGraph


def _path(n):
    return RexGraph(sources=np.arange(n - 1, dtype=np.int32),
                    targets=np.arange(1, n, dtype=np.int32))


def _triangle():
    return RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 0], dtype=np.int32))


def _k4():
    return RexGraph(sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
                    targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32))


def _disjoint(k, per=4):
    """k disjoint paths: beta0 = k, so L0 has a k-dimensional kernel."""
    s, t = [], []
    for c in range(k):
        base = c * per
        s.extend(range(base, base + per - 1))
        t.extend(range(base + 1, base + per))
    return RexGraph(sources=np.array(s, dtype=np.int32),
                    targets=np.array(t, dtype=np.int32))


def _reff(rex):
    return np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))


def test_every_relation_of_a_tree_is_a_bridge():
    assert np.allclose(_reff(_path(6)), 1.0)


def test_closed_forms():
    assert np.allclose(_reff(_triangle()), 2.0 / 3.0)
    assert np.allclose(_reff(_k4()), 0.5)


@pytest.mark.parametrize("k", [2, 5, 20])
def test_a_kernel_of_any_dimension_is_finite(k):
    """The regression: many components meant many NaNs."""
    r = _reff(_disjoint(k))
    assert np.all(np.isfinite(r)), f"{int((~np.isfinite(r)).sum())} NaN with beta0={k}"
    assert np.allclose(r, 1.0)          # disjoint paths are all bridges


@pytest.mark.parametrize("k", [1, 3, 8])
def test_foster_identity(k):
    """sum_e R_eff(e) = nV - beta0, exactly. An integer, and a free self-test: it is
    the invariant the NaNs violated."""
    rex = _disjoint(k, per=5)
    total = float(_reff(rex).sum())
    assert total == pytest.approx(int(rex.nV) - int(rex.betti[0]), abs=1e-9)


def test_foster_holds_on_a_complex_with_cycles():
    rex = _k4()
    assert float(_reff(rex).sum()) == pytest.approx(int(rex.nV) - int(rex.betti[0]),
                                                    abs=1e-9)


def test_matches_the_dense_pseudoinverse():
    """Ground truth, on a complex that has both cycles and several components."""
    s = np.array([0, 1, 2, 0, 4, 5, 6, 4, 8, 9], dtype=np.int32)
    t = np.array([1, 2, 0, 2, 5, 6, 4, 6, 9, 8], dtype=np.int32)
    rex = RexGraph(sources=s, targets=t)
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    truth = np.einsum("ve,vw,we->e", B1, np.linalg.pinv(B1 @ B1.T), B1)
    assert np.allclose(_reff(rex), truth, atol=1e-9)


def test_the_mean_carries_nothing_the_counts_do_not():
    """Foster again, read as a statement about information: the MEAN is
    (nV - beta0)/nE and needs no solve at all. Only the DISTRIBUTION is content."""
    for k in (1, 4):
        rex = _disjoint(k, per=6)
        r = _reff(rex)
        predicted = (int(rex.nV) - int(rex.betti[0])) / int(rex.nE)
        assert float(r.mean()) == pytest.approx(predicted, abs=1e-9)


def test_block_cg_freezes_degenerate_directions():
    """Directly: a right-hand side already solved must not blow the column up."""
    from rexgraph.sparse_character import _block_cg
    A = sp.diags([2.0, 2.0, 0.0]).tocsr()          # deliberately singular
    B = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    d = A.diagonal()
    dinv = np.where(d > 1e-30, 1.0 / np.where(d > 1e-30, d, 1.0), 1.0)
    X = _block_cg(lambda P: A @ P, B, dinv, tol=1e-12, maxit=50)
    assert np.all(np.isfinite(X))
    assert np.allclose(A @ X, B, atol=1e-9)


def test_resistance_decomposes_the_rank_and_its_complement_is_the_cycle_space():
    """The identity that makes R_eff a SIGNIFICANCE and not just a number.

        sum_e R_eff(e)       = rank(B1)
        sum_e (1 - R_eff(e)) = dim ker(B1) = n1 - rank(B1)

    So R_eff(e) is the relation's exact share of the boundary operator's rank (how
    much of the structure only it carries) and 1 - R_eff(e) is its exact share of the
    cycle space (how much independent alternative path corroborates it). Both totals
    are integers fixed by the complex, so the per-relation values are normalised by
    the structure itself rather than by a chosen threshold. Verified to 0.00e+00 on
    real Gene Ontology slices from 385 to 2315 relations.
    """
    for rex in (_k4(), _path(7), _disjoint(3, per=5)):
        rex._ensure_clean()
        r = _reff(rex)
        B1 = np.asarray(rex.B1)
        rank = int(np.linalg.matrix_rank(B1))
        assert float(r.sum()) == pytest.approx(rank, abs=1e-9)
        assert float((1.0 - r).sum()) == pytest.approx(int(rex.nE) - rank, abs=1e-9)


def test_a_bridge_carries_all_of_its_own_rank_and_none_of_the_cycle_space():
    """The two extremes, which is what makes the reading auditable: a relation with
    no alternative path reads exactly 1, one inside a cycle reads strictly less."""
    tree = _path(5)
    assert np.allclose(_reff(tree), 1.0)                 # every relation irreplaceable
    cyc = _triangle()
    assert np.all(_reff(cyc) < 1.0)                      # every relation corroborated


def test_the_complement_is_the_cycle_space_and_not_betti_once_faces_exist():
    """beta1 = n1 - rank(B1) - rank(B2), so it parts company with the resistance sum
    the moment anything is filled. The resistance reads B1 at its own grade and cannot
    see the grade above; a cycle a face has closed is still a second route."""
    import itertools as _it

    from rexgraph.faces import auto_hyperface
    groups = [[0, 1, 2], [2, 3, 4], [4, 5, 0]]
    rels = [list(g) for g in groups]
    for g in groups:
        rels.extend([a, b] for a, b in _it.combinations(sorted(g), 2))
    ptr, idx = [0], []
    for rl in rels:
        idx.extend(rl)
        ptr.append(len(idx))
    rex = RexGraph.from_hypergraph(np.array(ptr, np.int32), np.array(idx, np.int32))
    rex._ensure_clean()
    rank1 = int(np.linalg.matrix_rank(np.asarray(rex.B1)))
    unfilled = float(_reff(rex).sum())
    assert unfilled == pytest.approx(rank1, abs=1e-9)
    assert int(rex.betti[1]) == int(rex.nE) - rank1          # agree with no faces

    assert auto_hyperface(rex) > 0
    r = _reff(rex)
    assert float(r.sum()) == pytest.approx(rank1, abs=1e-9)
    assert float((1.0 - r).sum()) == pytest.approx(int(rex.nE) - rank1, abs=1e-9)
    assert int(rex.betti[1]) < int(rex.nE) - rank1           # and part company with them


def test_the_invariant_is_asserted_not_merely_true():
    """The self-test fires rather than returning a number that does not close."""
    from rexgraph.graph import _check_resistance_closes
    rex = _triangle()
    rex._ensure_clean()
    with pytest.raises(ValueError, match="does not close"):
        _check_resistance_closes(rex, np.arange(3), np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        _check_resistance_closes(rex, np.arange(3), np.array([np.nan, 1.0, 1.0]))
    # a subset carries no total, so it must NOT fire there
    _check_resistance_closes(rex, np.arange(2), np.array([1.0, 1.0]))


def test_arity_needs_the_leverage_reading_not_a_deflated_solve():
    """At arity the kernel of L0 is beta_0, not the component count, so the indicator
    basis is incomplete and an iterative solve walks into what it missed. On 400 human
    protein complexes the basis held 1519 of 2047 directions. The row-space projector
    needs no kernel at all."""
    # one 3-ary relation: three vertices, rank one, so ker(L0) is 2-dimensional
    # while the graph has a single component. The indicator basis holds one of two.
    ptr, idx = [0, 3, 7], [0, 1, 2, 2, 3, 4, 5]
    ptr = [0, 3, 7]
    rex = RexGraph.from_hypergraph(np.array(ptr, np.int32), np.array(idx, np.int32))
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    L0 = B1 @ B1.T
    ncomp = int(sp.csgraph.connected_components(sp.csr_matrix(L0), directed=False)[0])
    assert int(rex.betti[0]) > ncomp                   # kernel bigger than components
    truth = np.einsum("ve,vw,we->e", B1, np.linalg.pinv(L0), B1)
    got = _reff(rex)
    assert np.allclose(got, truth, atol=1e-9)
    assert float(got.sum()) == pytest.approx(int(rex.nV) - int(rex.betti[0]), abs=1e-9)


def test_the_two_readings_agree_and_the_gate_picks_one():
    """Leverage and deflated CG compute the same thing; the routing is on whether the
    dense nV x nE fits, not on arity, because a decomposition is exact and its cost is
    not set by conditioning while CG's iteration count is."""
    from rexgraph.core._common import check_dense_allocation, configure_memory
    s = np.array([0, 1, 2, 0, 4, 5, 6, 4, 8, 9], dtype=np.int32)
    t = np.array([1, 2, 0, 2, 5, 6, 4, 6, 9, 8], dtype=np.int32)
    rex = RexGraph(sources=s, targets=t)
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    truth = np.einsum("ve,vw,we->e", B1, np.linalg.pinv(B1 @ B1.T), B1)
    leverage = _reff(rex)                                # fits, so takes the leverage
    assert np.allclose(leverage, truth, atol=1e-9)
    try:                                                 # force the matrix-free path
        configure_memory(max_dense_allocation=1)
        with pytest.raises(Exception):
            check_dense_allocation("t", int(rex.nV), int(rex.nE))
        cg = _reff(rex)
        assert np.allclose(cg, truth, atol=1e-9)
        assert np.allclose(cg, leverage, atol=1e-9)
    finally:
        configure_memory(max_dense_allocation=8_000_000_000)
