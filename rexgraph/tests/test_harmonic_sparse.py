"""Eigen-free harmonic plane == dense eigendecomposition (the tower thesis).

The combinatorial, low-rank harmonic projector in ``harmonic_sparse`` (spanning-tree
cycle basis + reduced null-space, NO dense nE x nE eigendecomposition) must produce
exactly the same projector onto ker(L1) as the dense ``_harmonic.harmonic_projectors``
(which forms hb @ hbᵀ from a full eigensolve). If these agree, the eigen-free path can
replace the dense one with no loss - which is the whole point of the eigen-free tower.
"""
import numpy as np
import pytest

from rexgraph import harmonic_sparse as hs
from rexgraph.core import _harmonic
from rexgraph.graph import RexGraph


def _P_dense(rex):
    B1 = np.asarray(rex.B1, dtype=np.float64)
    B2 = np.asarray(rex.B2, dtype=np.float64)
    if B2.ndim != 2 or B2.shape[1] == 0:
        B2 = np.zeros((B1.shape[1], 0), dtype=np.float64)
    d = _harmonic.harmonic_projectors(B1, B2)
    return np.asarray(d["P_harm"]), int(d["dim_H"])


def _P_sparse(rex):
    H = hs.harmonic_basis(rex)
    Hd = np.asarray(H.todense()) if hasattr(H, "todense") else np.asarray(H)
    dim = Hd.shape[1]
    if dim == 0:
        return np.zeros((Hd.shape[0], Hd.shape[0])), 0
    # low-rank exact-rational projector H (HᵀH)⁻¹ Hᵀ (materialized only for the test)
    return Hd @ np.linalg.inv(Hd.T @ Hd) @ Hd.T, dim


def _cycle(nv):
    return RexGraph.from_graph(np.arange(nv), (np.arange(nv) + 1) % nv)


def _two_cycles_one_filled():
    # 6 vertices, 7 edges (two 4-cycles sharing edge 1-2); fill cycle A (0-1-2-3-0),
    # whose boundary is e0+e1+e2-e3 -> exactly one harmonic mode (cycle B) remains.
    return RexGraph.from_cells([6,
        [[0, 1], [1, 2], [2, 3], [0, 3], [1, 4], [4, 5], [2, 5]],
        [[(0, 1), (1, 1), (2, 1), (3, -1)]]])


def _tetra_shell():
    return RexGraph.from_simplicial(
        np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3]),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))


CASES = {
    "C3": _cycle(3), "C5": _cycle(5), "C8": _cycle(8),
    "two-cycles-one-filled": _two_cycles_one_filled(),
    "tetra-shell": _tetra_shell(),
}


@pytest.mark.parametrize("name", list(CASES))
def test_sparse_harmonic_projector_equals_dense(name):
    rex = CASES[name]
    Pd, dim_d = _P_dense(rex)
    Ps, dim_s = _P_sparse(rex)
    assert dim_s == dim_d, f"{name}: harmonic dim sparse={dim_s} dense={dim_d}"
    assert np.allclose(Pd, Ps, atol=1e-9), \
        f"{name}: projector mismatch max|Δ|={np.abs(Pd - Ps).max():.2e}"


def test_projector_is_idempotent_and_annihilates_gradient():
    """The eigen-free projector is a true Hodge projector: P²=P and P·B1ᵀ = 0
    (harmonic ⟂ gradient), computed with no eigensolve."""
    rex = _two_cycles_one_filled()
    Ps, dim = _P_sparse(rex)
    assert dim == 1
    assert np.allclose(Ps @ Ps, Ps, atol=1e-9)          # idempotent
    B1 = np.asarray(rex.B1, dtype=np.float64)
    assert np.allclose(Ps @ B1.T, 0.0, atol=1e-9)       # kills gradients


def test_filled_complex_does_not_crash():
    """Regression: harmonic_basis on a complex WITH faces (the C @ null_space path)
    previously raised AttributeError('.tocsc' on ndarray)."""
    H = hs.harmonic_basis(_two_cycles_one_filled())
    assert H.shape[1] == 1


@pytest.mark.parametrize("name", list(CASES))
def test_harmonic_basis_from_boundaries_matches_rex(name):
    """The rex-free `harmonic_basis_from_boundaries(B1, B2)` (reused by _void /
    _quotient) spans exactly the same harmonic plane as `harmonic_basis(rex)`:
    same dim_H, annihilates B1, and its column span contains the rex basis."""
    from rexgraph.core._sparse import to_scipy_csr
    rex = CASES[name]
    B1 = to_scipy_csr(rex._B1_dual).tocsc()
    B2 = hs._b2_csr(rex)
    H_rex = hs.harmonic_basis(rex)
    H_bnd = hs.harmonic_basis_from_boundaries(B1, B2)
    assert H_bnd.shape[1] == H_rex.shape[1]
    if H_bnd.shape[1] == 0:
        return
    Hb = np.asarray(H_bnd.todense())
    assert np.abs(B1 @ Hb).max() < 1e-9                 # B1 · H = 0
    Hr = np.asarray(H_rex.todense())
    Q, _ = np.linalg.qr(Hb)                             # span(H_rex) ⊆ span(H_bnd)
    assert np.linalg.norm(Hr - Q @ (Q.T @ Hr)) < 1e-9


def test_harmonic_basis_from_boundaries_stays_in_ker_b1_on_branching():
    """The rex-free core must validate its combinatorial cycle basis like cycle_basis
    does. Without that, the endpoint reduction invents cycles on branching hyperedges
    and returns vectors outside ker(B1), which _void and _quotient then consume."""
    import scipy.sparse as sp

    from rexgraph.graph import RexGraph
    from rexgraph.harmonic_sparse import cycle_basis, harmonic_basis, harmonic_basis_from_boundaries

    # mixed arity 1, 2, 3, 4; the cycle space is empty (betti_1 == 0)
    h = RexGraph.from_hypergraph(np.array([0, 1, 3, 6, 10], dtype=np.int32),
                                 np.array([0, 0, 1, 1, 2, 3, 0, 2, 3, 4], dtype=np.int32))
    assert int(h.betti[1]) == 0

    B1 = sp.csr_matrix(np.asarray(h.B1, dtype=float))
    ref = cycle_basis(h)
    got = harmonic_basis_from_boundaries(B1, None)

    # same dimension as the validated basis, and genuinely in ker(B1)
    assert got.shape[1] == ref.shape[1] == 0
    dense = got.toarray() if sp.issparse(got) else np.asarray(got)
    if dense.size:
        assert float(np.abs(B1 @ dense).max()) < 1e-9

    # and it must agree with the rex-taking wrapper
    assert harmonic_basis(h).shape[1] == got.shape[1]


def test_harmonic_basis_from_boundaries_matches_cycle_basis_on_a_branching_cycle():
    """A branching complex that does carry cycles: the rex-free core must return a
    basis of the right dimension that B1 annihilates."""
    import scipy.sparse as sp

    from rexgraph.graph import RexGraph
    from rexgraph.harmonic_sparse import cycle_basis, harmonic_basis_from_boundaries

    h = RexGraph.from_hypergraph(np.array([0, 3, 6, 9, 12, 15], dtype=np.int32),
                                 np.array([0, 1, 2, 1, 2, 3, 2, 3, 0,
                                           0, 3, 1, 1, 0, 2], dtype=np.int32))
    B1 = sp.csr_matrix(np.asarray(h.B1, dtype=float))
    ref = cycle_basis(h)
    got = harmonic_basis_from_boundaries(B1, None)
    assert got.shape[1] == ref.shape[1]
    dense = got.toarray() if sp.issparse(got) else np.asarray(got)
    if dense.size:
        assert float(np.abs(B1 @ dense).max()) < 1e-9
