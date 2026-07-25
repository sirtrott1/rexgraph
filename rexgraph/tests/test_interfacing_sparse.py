"""Parity tests for the eigen-free / scale-free interfacing bundle
(``rexgraph.sparse_interfacing.build_interfacing_bundle_sparse``) against the dense
Cython oracle (``rexgraph.core._interfacing.build_interfacing_bundle``, reached via
``RexGraph.interfacing_vector`` on small graphs).

Small-graph fixtures (cycle, K4, two triangles sharing an edge) mirror
``test_interfacing.py``.

Tolerances
----------
* EIGEN-FREE fields (rho, psi, signal_magnitude, scores T/G/F, iv[:3], efficiency)
  are exact to ~1e-8 - they are matrix-free reproductions of the dense contraction.
* On small graphs the sparse path REUSES the exact dense RL eigenbasis for the
  genuinely-spectral schrodinger/coverage, so those (and hence sphere_pos, confidence)
  are also exact to ~1e-8.
* When the scale-free path is FORCED (eigen_dense_limit lowered so no dense RL is
  built) schrodinger/coverage come from a bounded eigsh surrogate; only the eigen-free
  fields keep tight parity, the spectral fields are checked loosely (documented).
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.core import _interfacing
from rexgraph.sparse_interfacing import (
    build_interfacing_bundle_sparse,
    pinv_bilinear_form,
)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                           dtype=np.int32),
    )


@pytest.fixture
def cycle():
    # 5-cycle 0-1-2-3-4-0 (no faces).
    return RexGraph.from_graph([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])


@pytest.fixture
def two_triangles():
    # Triangles (0,1,2) and (1,2,3) sharing edge (1,2).
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
    )


def _dense_bundle(rex, ti, tw, target, vw=None):
    """Dense oracle via the Cython kernel (same inputs graph.interfacing_vector uses)."""
    if vw is None:
        deg = rex.degree.astype(np.float64)
        vw = 1.0 / np.log(deg + np.e)
    sb = rex.spectral_bundle
    evals_rl, evecs_rl = rex._rl_eigen
    return _interfacing.build_interfacing_bundle(
        ti.astype(np.int32),
        np.ascontiguousarray(tw, dtype=np.float64),
        np.ascontiguousarray(vw, dtype=np.float64),
        rex.B1,
        sb['evals_L0'],
        np.ascontiguousarray(sb['evecs_L0'], dtype=np.float64),
        rex.L_overlap,
        rex.L_frustration,
        evals_rl, evecs_rl,
        np.ascontiguousarray(target, dtype=np.float64),
        rex.nV, rex.nE,
    )


def _cases(rex):
    """A few (target_indices, target_weights, target_signal) probes."""
    rng = np.random.RandomState(7)
    return [
        (np.array([0, 1], dtype=np.int32),
         np.array([1.0, 1.0]), np.ones(rex.nE)),
        (np.array([0], dtype=np.int32),
         np.array([2.5]), rng.randn(rex.nE)),
        (np.array([0, 2], dtype=np.int32),
         np.array([1.0, 3.0]), rng.randn(rex.nE)),
    ]


# --- exact-reuse parity (dense RL spectrum available) -----------------------

@pytest.mark.parametrize("fixture", ["k4", "cycle", "two_triangles"])
def test_full_bundle_parity(fixture, request):
    rex = request.getfixturevalue(fixture)
    # Sparse is now the universal path (no dense-vs-sparse size cutoff). Parity still
    # holds exactly here: both the dense oracle (_dense_bundle -> rex._rl_eigen) and the
    # sparse bundle eigendecompose the SAME dense-on-demand RL4 for the full-spectrum
    # schrodinger/coverage, since nE is within the mode budget (_RL_SURROGATE_K).
    assert rex._use_sparse_character
    for ti, tw, target in _cases(rex):
        dense = _dense_bundle(rex, ti, tw, target)
        sparse = build_interfacing_bundle_sparse(rex, ti, tw, target)

        # eigen-free fields: tight
        np.testing.assert_allclose(sparse['rho'], dense['rho'], atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(sparse['psi'], dense['psi'], atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(sparse['scores'], dense['scores'],
                                   atol=1e-8, rtol=1e-7)
        assert abs(sparse['signal_magnitude'] - dense['signal_magnitude']) < 1e-8
        assert abs(sparse['efficiency'] - dense['efficiency']) < 1e-10

        # spectral fields: exact because the sparse path reuses the dense RL basis
        assert abs(sparse['schrodinger'] - dense['schrodinger']) < 1e-8
        assert abs(sparse['coverage'] - dense['coverage']) < 1e-10

        # assembled (raw, well-defined everywhere)
        np.testing.assert_allclose(sparse['iv'], dense['iv'], atol=1e-8, rtol=1e-7)

        # sphere_pos = iv / ||iv|| and confidence's phi_T = sphere_pos[0] are only
        # well-conditioned when iv is above rounding noise: the dense oracle divides
        # even a ~1e-16 iv by its norm, so its DIRECTION is pure noise there (this
        # happens on the symmetric cycle when all channel scores vanish). Compare the
        # normalized direction / confidence only when iv carries real magnitude.
        if np.linalg.norm(dense['iv']) > 1e-6:
            np.testing.assert_allclose(sparse['sphere_pos'], dense['sphere_pos'],
                                       atol=1e-8, rtol=1e-7)
            assert sparse['confidence'] == dense['confidence']


def test_matches_graph_method(k4):
    """graph.interfacing_vector on a small graph == the direct sparse bundle for the
    eigen-free fields (it routes to the dense oracle, which the sparse path reproduces)."""
    ti = np.array([0, 1], dtype=np.int32)
    tw = np.array([1.0, 1.0])
    target = np.ones(k4.nE)
    dense = k4.interfacing_vector(ti, tw, target)
    sparse = build_interfacing_bundle_sparse(k4, ti, tw, target)
    np.testing.assert_allclose(sparse['scores'], dense['scores'], atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(sparse['psi'], dense['psi'], atol=1e-8, rtol=1e-7)


def test_bilinear_identity(k4):
    """Cross-check the I_T shortcut ``(B1 target)^T y`` against the literal two-solve
    bilinear ``pinv_bilinear_form(L0, B1 target, B1 psi)``."""
    ti = np.array([0, 1], dtype=np.int32)
    tw = np.array([1.0, 1.0])
    target = np.ones(k4.nE)
    sparse = build_interfacing_bundle_sparse(k4, ti, tw, target)
    B1 = k4.B1
    L0 = k4.L0_sparse
    u = B1 @ target
    w = B1 @ sparse['psi']
    I_T_literal = pinv_bilinear_form(L0, u, w)
    assert abs(I_T_literal - sparse['scores'][0]) < 1e-7


def test_keys_and_shapes(k4):
    ti = np.array([0], dtype=np.int32)
    tw = np.array([1.0])
    target = np.ones(k4.nE)
    b = build_interfacing_bundle_sparse(k4, ti, tw, target)
    for key in ['rho', 'psi', 'scores', 'schrodinger', 'iv', 'sphere_pos',
                'signal_magnitude', 'coverage', 'efficiency', 'confidence']:
        assert key in b
    assert b['rho'].shape == (k4.nV,)
    assert b['psi'].shape == (k4.nE,)
    assert b['scores'].shape == (3,)
    assert b['iv'].shape == (4,)
    assert b['sphere_pos'].shape == (4,)


# --- forced scale-free path (bounded-spectrum surrogate) --------------------

def test_forced_sparse_path_eigenfree_parity():
    """Force the scale-free route (no dense RL) by lowering eigen_dense_limit, and
    confirm the eigen-free fields still match the dense oracle tightly; the bounded
    eigsh schrodinger/coverage are only sanity-checked (finite, coverage in [0,1])."""
    from rexgraph.core import _common

    ti = np.array([0, 1], dtype=np.int32)
    tw = np.array([1.0, 1.0])

    # dense oracle graph (normal limit)
    dense_rex = RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                           dtype=np.int32),
    )
    target = np.ones(dense_rex.nE)
    dense = dense_rex.interfacing_vector(ti, tw, target)

    saved = _common.get_algorithm_config().get('eigen_dense_limit', 2000)
    try:
        _common.configure_algorithms(eigen_dense_limit=1)  # force sparse everywhere
        rex = RexGraph.from_simplicial(
            sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
            targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
            triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                               dtype=np.int32),
        )
        assert rex._use_sparse_character
        sparse = rex.interfacing_vector(ti, tw, target)  # routes to sparse bundle
    finally:
        _common.configure_algorithms(eigen_dense_limit=int(saved))

    # eigen-free fields: tight vs dense oracle
    np.testing.assert_allclose(sparse['rho'], dense['rho'], atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(sparse['psi'], dense['psi'], atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(sparse['scores'], dense['scores'], atol=1e-8, rtol=1e-7)
    assert abs(sparse['signal_magnitude'] - dense['signal_magnitude']) < 1e-8
    assert abs(sparse['efficiency'] - dense['efficiency']) < 1e-10

    # bounded-spectrum surrogate fields: sane, not exact
    assert np.isfinite(sparse['schrodinger'])
    assert sparse['schrodinger'] >= -1e-10
    assert 0.0 <= sparse['coverage'] <= 1.0
