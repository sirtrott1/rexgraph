"""Exact integer/weighted curvature identities - the integer tower, no float drift.

The Lagrangian and weighted-curvature quantities have closed-form or exact-integer
oracles; these guard the sparse/integer kernels against silent regression.
"""
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.core import _curvature


def _rex(edges):
    s = np.array([e[0] for e in edges], dtype=np.int32)
    t = np.array([e[1] for e in edges], dtype=np.int32)
    return RexGraph.from_graph(s, t)


def _complete(k):
    return _rex([(i, j) for i in range(k) for j in range(i + 1, k)])


@pytest.mark.parametrize("k", [4, 5, 6])
def test_lagrangian_L_T_integer_identity(k):
    """L_T = tr((B1^T B1)^2) = sum(deg^2) + 2*nE, exact from the degree sequence (script 08)."""
    rex = _complete(k)
    deg = np.abs(np.asarray(rex.B1, dtype=float)).sum(axis=1)
    want = int((deg ** 2).sum() + 2 * rex.nE)
    got = _curvature.lagrangian_L_T_integer(rex.sources, rex.targets, rex.nV)
    assert int(got) == want
    # closed form for the complete graph K_k: L_T = k^2 (k-1)
    assert want == k * k * (k - 1)


def test_weighted_curvature_signature_unweighted_is_flat():
    """Unweighted (W = I) has zero curvature residual and n_eff = nE (script 20)."""
    rex = _rex([(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (4, 5), (2, 5)])
    sig = rex.weighted_curvature_signature()
    assert sig["total_curvature"] == pytest.approx(0.0, abs=1e-9)
    assert sig["n_eff"] == pytest.approx(rex.nE)


def test_weighted_curvature_signature_weighting_concentrates_n_eff():
    """Non-uniform weights lower n_eff = (sum w)^2 / sum w^2, and with faces present
    raise the curvature residual R = B1 (W - I) B2 above zero (script 20)."""
    rex = RexGraph.from_simplicial(                                  # tetrahedron: 4 faces
        np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32))
    w = np.array([1, 1, 1, 2, 2, 3], dtype=float)
    sig = rex.weighted_curvature_signature(w)
    assert sig["n_eff"] == pytest.approx((w.sum() ** 2) / (w ** 2).sum())   # 5.0
    assert sig["n_eff"] < rex.nE
    assert sig["total_curvature"] > 0.0
