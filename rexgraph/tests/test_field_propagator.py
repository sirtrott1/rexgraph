"""Matrix-free graded field evolution == dense field eigendecomposition (tower thesis
for the coupled (edge, face) field).

The field operator M = [[RL1,-gB2],[-gB2ᵀ,L2]] on the graded space C1(+)C2 is evolved
by a Chebyshev polynomial of the SPARSE M (never the dense (nE+nF)² matrix, no
eigensolve). heat e^{-tM} and wave cos(t√M) must match the dense eigendecomposition
apply, and a tensor-shaped field (a block of components) must propagate as a single
spmm - the shape the parallel/GPU backend batches over.
"""
import numpy as np
import pytest

from rexgraph import field_propagator as fp
from rexgraph.graph import RexGraph


def _tetra():
    return RexGraph.from_simplicial(
        np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3]),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))


def _octa_solid():
    from rexgraph.graded_boundary import solid_octahedron_3rex
    return RexGraph.from_cells(solid_octahedron_3rex())


def _dense(M):
    Md = np.asarray(M.todense())
    w, V = np.linalg.eigh(Md)
    return w, V


def _dense_heat(w, V, F0, t):
    return V @ (np.exp(-t * w) * (V.T @ F0))


def _dense_wave(w, V, F0, t):
    return V @ (np.cos(t * np.sqrt(np.maximum(w, 0.0))) * (V.T @ F0))


CASES = {"tetra-2rex": _tetra}


def test_field_operator_is_sparse_not_dense():
    g = _tetra()
    M = fp.assemble_field_operator(g)
    import scipy.sparse as sp
    assert sp.issparse(M)
    N = g.nE + g.nF_hodge
    assert M.shape == (N, N)
    # symmetric PSD graded operator
    assert np.allclose(M.todense(), M.todense().T, atol=1e-12)


@pytest.mark.parametrize("t", [0.1, 0.5, 2.0])
def test_field_heat_matches_dense(t):
    g = _tetra()
    M = fp.assemble_field_operator(g)
    w, V = _dense(M)
    rng = np.random.default_rng(0)
    F0 = rng.standard_normal(M.shape[0])
    got = fp.field_heat(g, F0, t)
    np.testing.assert_allclose(got, _dense_heat(w, V, F0, t), atol=1e-10)


@pytest.mark.parametrize("t", [0.1, 0.5, 2.0])
def test_field_wave_matches_dense(t):
    g = _tetra()
    M = fp.assemble_field_operator(g)
    w, V = _dense(M)
    rng = np.random.default_rng(1)
    F0 = rng.standard_normal(M.shape[0])
    got = fp.field_wave(g, F0, t)
    np.testing.assert_allclose(got, _dense_wave(w, V, F0, t), atol=1e-9)


def test_field_heat_trajectory_shares_matvecs():
    g = _tetra()
    M = fp.assemble_field_operator(g)
    w, V = _dense(M)
    rng = np.random.default_rng(2)
    F0 = rng.standard_normal(M.shape[0])
    times = np.array([0.05, 0.25, 1.0, 3.0])
    traj = fp.field_heat_trajectory(g, F0, times)
    want = np.stack([_dense_heat(w, V, F0, t) for t in times])
    np.testing.assert_allclose(traj, want, atol=1e-10)


def test_tensor_field_block_propagates_as_spmm():
    """A block field (N, m) - m tensor components - propagates in one spmm and equals
    component-by-component evolution."""
    g = _tetra()
    N = g.nE + g.nF_hodge
    rng = np.random.default_rng(3)
    Fblk = rng.standard_normal((N, 5))
    blk = fp.field_heat(g, Fblk, 0.7)
    cols = np.stack([fp.field_heat(g, Fblk[:, j], 0.7) for j in range(5)], axis=1)
    assert blk.shape == (N, 5)
    np.testing.assert_allclose(blk, cols, atol=1e-12)


def test_edge_signal_lifts_to_graded_state():
    g = _tetra()
    N = g.nE + g.nF_hodge
    f_edge = np.arange(g.nE, dtype=float)
    out = fp.field_heat(g, f_edge, 0.3)
    assert out.shape == (N,)


def test_graded_3rex_field_runs_and_matches_dense():
    """The field operator and its evolution generalize to a grade-3 rex (octahedron
    solid) - the graded space is C1(+)C2 here regardless of higher grades."""
    g = _octa_solid()
    M = fp.assemble_field_operator(g)
    w, V = _dense(M)
    rng = np.random.default_rng(4)
    F0 = rng.standard_normal(M.shape[0])
    np.testing.assert_allclose(fp.field_heat(g, F0, 0.4), _dense_heat(w, V, F0, 0.4), atol=1e-10)


#### tensor METRIC: weighted graded inner product W (sqrt-w default, SPD override)
from scipy.linalg import expm as _expm


def _ref_metric_heat(Md, W, F0, t):
    return _expm(-t * np.linalg.solve(W, Md)) @ F0


def test_unweighted_default_metric_is_identity():
    """The default metric on an unweighted complex is the identity: field_heat reduces
    to e^{-tM} exactly (back-compat)."""
    g = _tetra()
    M = fp.assemble_field_operator(g); Md = np.asarray(M.todense())
    rng = np.random.default_rng(0); F0 = rng.standard_normal(M.shape[0])
    np.testing.assert_allclose(fp.field_heat(g, F0, 0.5), _expm(-0.5 * Md) @ F0, atol=1e-10)


@pytest.mark.parametrize("t", [0.2, 0.8])
def test_diagonal_metric_heat_matches_generalized(t):
    """e^{-t W^{-1}M} F under a diagonal tensor metric equals the dense reference."""
    g = _tetra()
    M = fp.assemble_field_operator(g); Md = np.asarray(M.todense()); N = M.shape[0]
    rng = np.random.default_rng(1)
    d = rng.uniform(0.4, 2.5, N); F0 = rng.standard_normal(N)
    got = fp.field_heat(g, F0, t, W=d)
    np.testing.assert_allclose(got, _ref_metric_heat(Md, np.diag(d), F0, t), atol=1e-10)


def test_full_spd_metric_heat_matches_generalized():
    """A full (non-diagonal) SPD tensor metric via Cholesky conjugation matches."""
    g = _tetra()
    M = fp.assemble_field_operator(g); Md = np.asarray(M.todense()); N = M.shape[0]
    rng = np.random.default_rng(2)
    A = rng.standard_normal((N, N)); W = A @ A.T + N * np.eye(N)
    F0 = rng.standard_normal(N)
    got = fp.field_heat(g, F0, 0.5, W=W)
    np.testing.assert_allclose(got, _ref_metric_heat(Md, W, F0, 0.5), atol=1e-9)


def test_metric_wave_matches_generalized():
    """cos(t·sqrt(W^{-1}M)) F under a diagonal metric matches the conjugated reference."""
    g = _tetra()
    M = fp.assemble_field_operator(g); Md = np.asarray(M.todense()); N = M.shape[0]
    rng = np.random.default_rng(3)
    d = rng.uniform(0.5, 2.0, N); F0 = rng.standard_normal(N)
    L = np.linalg.cholesky(np.diag(d))
    S = np.linalg.solve(L, np.linalg.solve(L, Md).T).T
    ws, Vs = np.linalg.eigh(S)
    ref = np.linalg.solve(L.T, Vs @ (np.cos(0.6 * np.sqrt(np.maximum(ws, 0))) * (Vs.T @ (L.T @ F0))))
    np.testing.assert_allclose(fp.field_wave(g, F0, 0.6, W=d), ref, atol=1e-9)


def test_metric_trajectory_and_tensor_block():
    g = _tetra()
    M = fp.assemble_field_operator(g); Md = np.asarray(M.todense()); N = M.shape[0]
    rng = np.random.default_rng(4)
    d = rng.uniform(0.4, 2.5, N); F0 = rng.standard_normal(N)
    times = np.array([0.1, 0.5, 1.0])
    traj = fp.field_heat_trajectory(g, F0, times, W=d)
    ref = np.stack([_ref_metric_heat(Md, np.diag(d), F0, t) for t in times])
    np.testing.assert_allclose(traj, ref, atol=1e-9)
    Fblk = rng.standard_normal((N, 3))
    blk = fp.field_heat(g, Fblk, 0.4, W=d)
    cols = np.stack([fp.field_heat(g, Fblk[:, j], 0.4, W=d) for j in range(3)], axis=1)
    np.testing.assert_allclose(blk, cols, atol=1e-12)
