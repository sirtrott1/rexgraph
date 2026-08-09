"""Dimensions that come from ranks, not from counting eigenvalues under a cutoff.

The harmonic space, the nullity of a Laplacian and the Fiedler value all turn on the
same question: which modes are zero. A magnitude cutoff answers it by asking how small
is small, so a nearly-degenerate mode moves the reported topology, and the reading is a
different number on a different machine. The rank tower answers it with an integer.

These pin the exact path against the dense spectral one, which stays available as the
oracle. Agreement is the point: the sparse combinatorial route is not an approximation
of the eigendecomposition, it is the same space by a route that does not have to guess.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.faces import autoface
from rexgraph.graph import RexGraph


def _two_triangles():
    """Sharing one vertex: beta_1 = 2."""
    return RexGraph(sources=np.array([0, 1, 2, 2, 3, 4], dtype=np.int32),
                    targets=np.array([1, 2, 0, 3, 4, 2], dtype=np.int32))


def _square():
    return RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                    targets=np.array([1, 2, 3, 0], dtype=np.int32))


def _tree():
    return RexGraph(sources=np.array([0, 0, 1], dtype=np.int32),
                    targets=np.array([1, 2, 3], dtype=np.int32))


def _filled_triangle():
    g = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                 targets=np.array([1, 2, 0], dtype=np.int32))
    autoface(g, 3)
    return g


CASES = [("two triangles", _two_triangles), ("square", _square),
         ("tree", _tree), ("filled triangle", _filled_triangle)]


@pytest.mark.parametrize("name,build", CASES, ids=[c[0] for c in CASES])
def test_the_harmonic_dimension_is_beta_1(name, build):
    """Not a count of eigenvalues under 1e-10."""
    g = build()
    assert g.harmonic_space.shape[0] == int(g.betti[1])


@pytest.mark.parametrize("name,build", CASES, ids=[c[0] for c in CASES])
def test_the_harmonic_basis_agrees_with_the_spectral_oracle(name, build):
    """Same SPACE, compared through orthogonal projectors so the comparison does not
    depend on which representative each route happens to return."""
    g = build()
    H, D = g.harmonic_space, g.harmonic_space_dense
    assert H.shape[0] == D.shape[0], f"{name}: dimensions disagree"
    if H.shape[0] == 0:
        return
    q, _ = np.linalg.qr(D.T)
    assert np.allclose(H.T @ H, q @ q.T, atol=1e-8), f"{name}: different subspace"


@pytest.mark.parametrize("name,build", CASES, ids=[c[0] for c in CASES])
def test_harmonic_vectors_lie_exactly_in_the_kernel(name, build):
    """The combinatorial basis satisfies B_1 H = 0 by construction, so the residual is
    zero rather than small."""
    g = build()
    H = g.harmonic_space
    if H.shape[0] == 0:
        return
    assert float(np.abs(g.L1 @ H.T).max()) < 1e-12


def test_the_harmonic_rows_are_orthonormal():
    """The contract this property has always advertised."""
    H = _two_triangles().harmonic_space
    assert np.allclose(H @ H.T, np.eye(H.shape[0]), atol=1e-10)


def test_a_face_that_does_not_bound_is_named_when_merging():
    """The merge path densified B1 and B2 to multiply them and compared against a
    cutoff. The complex's own predicate is exact and says which face."""
    g = RexGraph(sources=np.array([0, 1, 2, 3], dtype=np.int32),
                 targets=np.array([1, 2, 0, 0], dtype=np.int32))
    g.add_faces([[0, 1, 3]], [[1.0, 1.0, 1.0]])
    assert g.self_loop_face_indices == [0]
    assert g.nF_hodge == 0


def test_a_face_that_bounds_is_not_named():
    g = _filled_triangle()
    assert g.self_loop_face_indices == []
    assert g.nF_hodge == g.nF
