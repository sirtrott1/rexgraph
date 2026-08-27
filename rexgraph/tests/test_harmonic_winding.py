"""The winding: the exact, counted reading of the harmonic sector.

Pairing a cochain against a harmonic cycle sees the harmonic part and nothing else,
because the other two sectors pair to exactly zero. So the winding is not one
component among three, it is the whole of what a cycle can read, and on an integer
frame it is an integer count rather than a measurement.
"""
import itertools

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.harmonic_sparse import harmonic_basis, harmonic_coordinates, harmonic_winding


def _graph(edges):
    r = RexGraph(sources=np.array([a for a, b in edges], np.int32),
                 targets=np.array([b for a, b in edges], np.int32))
    r._ensure_clean()
    return r


def _kn(k):
    return _graph(list(itertools.combinations(range(k), 2)))


ZOO = {
    "K4": _kn(4),
    "K5": _kn(5),
    "C6": _graph([(i, (i + 1) % 6) for i in range(6)]),
    "prism": _graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                     (0, 3), (1, 4), (2, 5)]),
    "house": _graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 2)]),
    "petersen": _graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]),
}


@pytest.mark.parametrize("name", list(ZOO))
def test_a_gradient_signal_has_no_winding(name):
    """PROVED: <B1^T phi, z> = <phi, B1 z> = 0 for any cycle z. A potential is
    path-independent, so it winds around nothing."""
    rex = ZOO[name]
    H = harmonic_basis(rex)
    B1 = np.asarray(rex.B1_dense, float)
    rng = np.random.default_rng(0)
    for _ in range(5):
        phi = rng.integers(-9, 10, B1.shape[0]).astype(float)
        assert np.abs(harmonic_winding(H, B1.T @ phi)).max() < 1e-9


@pytest.mark.parametrize("name", list(ZOO))
def test_integer_data_gives_an_integer_count(name):
    """The frame is integer, so the pairing is exact arithmetic and the result is a
    COUNT: a number of turns, with no angle and no transcendental in it."""
    rex = ZOO[name]
    H = harmonic_basis(rex)
    if H.shape[1] == 0:
        pytest.skip("no cycles")
    rng = np.random.default_rng(1)
    f = rng.integers(-7, 8, int(rex.nE))
    w = harmonic_winding(H, f)
    assert np.issubdtype(w.dtype, np.integer), w.dtype
    assert w.shape == (H.shape[1],)


@pytest.mark.parametrize("name", list(ZOO))
def test_the_winding_sees_only_the_harmonic_part(name):
    """Which is what makes it the complete reading rather than a lossy one: the
    winding of the whole cochain equals the winding of its harmonic projection, so
    nothing is discarded by not projecting first."""
    from rexgraph.harmonic_sparse import harmonic_projection
    rex = ZOO[name]
    H = harmonic_basis(rex)
    if H.shape[1] == 0:
        pytest.skip("no cycles")
    rng = np.random.default_rng(2)
    f = rng.normal(size=int(rex.nE))
    assert np.allclose(np.asarray(harmonic_winding(H, f), dtype=float),
                       np.asarray(harmonic_winding(H, harmonic_projection(H, f)),
                                  dtype=float), atol=1e-9)


def test_a_face_kills_the_winding_of_the_cycle_it_bounds():
    """And the winding tracks structure, not just signal. Filling a triangle of K4
    removes it from the harmonic sector, so a flow around that triangle stops being
    visible to the frame: dim_H drops and the bounded cycle no longer registers."""
    rex = _kn(4)
    before = harmonic_basis(rex).shape[1]
    ei = {e: i for i, e in enumerate(itertools.combinations(range(4), 2))}
    loop = np.zeros(int(rex.nE))
    loop[ei[(0, 1)]] = 1.0
    loop[ei[(1, 2)]] = 1.0
    loop[ei[(0, 2)]] = -1.0
    assert np.abs(np.asarray(rex.B1_dense, float) @ loop).max() < 1e-12
    assert np.abs(harmonic_winding(harmonic_basis(rex), loop)).max() > 0

    rex.add_faces([[ei[(0, 1)], ei[(1, 2)], ei[(0, 2)]]], signs=[[1.0, 1.0, -1.0]])
    rex._ensure_clean()
    after = harmonic_basis(rex)
    assert after.shape[1] == before - 1, (before, after.shape[1])
    assert np.abs(harmonic_winding(after, loop)).max() < 1e-9


@pytest.mark.parametrize("name", list(ZOO))
def test_the_coordinates_are_the_winding_through_the_metric(name):
    """The two readings, kept straight. coords = (H^T H)^-1 * winding, so the winding
    is the exact numerator and the Gram solve is what makes it float. The harmonic
    metric is H^T H and NOT the identity, which is exactly why they differ."""
    rex = ZOO[name]
    H = harmonic_basis(rex)
    if H.shape[1] == 0:
        pytest.skip("no cycles")
    rng = np.random.default_rng(3)
    f = rng.normal(size=int(rex.nE))
    w = np.asarray(harmonic_winding(H, f), dtype=float)
    gram = np.asarray((H.T @ H).todense())
    assert np.allclose(harmonic_coordinates(H, f), np.linalg.solve(gram, w), atol=1e-8)
    if H.shape[1] > 1:
        assert not np.allclose(gram, np.eye(gram.shape[0])), "metric is not identity"


def test_the_method_on_the_graph_matches_the_function():
    rex = ZOO["petersen"]
    f = np.arange(int(rex.nE)) % 5 - 2
    assert np.array_equal(rex.harmonic_winding(f),
                          harmonic_winding(harmonic_basis(rex), f))


#### reading the holonomy around CHOSEN cycles, which is the only affordable way at scale
def test_a_rings_mask_becomes_a_signed_chain():
    """rings returns UNSIGNED masks: which relations a ring uses, and nothing about
    how it closes. That is the set-theoretic encoding, and it does not land in
    ker(B1). `cycle_vector` orients the walk, so the result is a genuine 1-cycle."""
    from rexgraph.rings import cycle_vector, cycle_vectors, shortest_cycles
    rex = ZOO["K5"]
    B1 = np.asarray(rex.B1_dense, float)
    cycles = shortest_cycles(rex)
    assert cycles, "K5 has triangles"
    for c in cycles:
        v = cycle_vector(rex, c)
        assert set(np.unique(v).tolist()) <= {-1.0, 0.0, 1.0}
        assert int(np.count_nonzero(v)) == c[0], "support is the mask"
        assert np.abs(B1 @ v).max() < 1e-12, "and it closes"
    C = np.asarray(cycle_vectors(rex, cycles).todense())
    assert np.abs(B1 @ C).max() < 1e-12


def test_an_unsigned_mask_would_not_have_closed():
    """The control. Taking the mask as a 0/1 vector leaves a nonzero boundary, which
    is the whole reason the orientation step exists."""
    from rexgraph.rings import shortest_cycles
    rex = ZOO["K5"]
    B1 = np.asarray(rex.B1_dense, float)
    w, mask = shortest_cycles(rex)[0]
    unsigned = np.array([(mask >> e) & 1 for e in range(int(rex.nE))], dtype=float)
    assert np.abs(B1 @ unsigned).max() > 0.5, "unsigned support is not a cycle"


@pytest.mark.parametrize("name", ["K5", "prism", "house"])
def test_winding_around_chosen_cycles_matches_the_full_basis_on_gradients(name):
    """Whatever cycles are chosen, a gradient still winds around nothing: the
    exactness does not depend on picking a basis, only on the cycles closing."""
    from rexgraph.rings import relevant_cycles
    rex = ZOO[name]
    B1 = np.asarray(rex.B1_dense, float)
    cyc = relevant_cycles(rex)
    rng = np.random.default_rng(4)
    for _ in range(3):
        g = B1.T @ rng.integers(-9, 10, B1.shape[0]).astype(float)
        assert np.abs(rex.harmonic_winding(g, cycles=cyc)).max() < 1e-9


def test_the_cycles_argument_accepts_a_matrix_and_skips_the_basis():
    """The scale path. At nE ~ 2e6 with beta_1 ~ 1.8e6 the full frame is neither a
    feature vector nor affordable to build, so the caller supplies the cycles it
    cares about and pays one matvec against those alone."""
    import scipy.sparse as sp

    from rexgraph.rings import cycle_vectors, shortest_cycles
    rex = ZOO["K5"]
    C = cycle_vectors(rex, shortest_cycles(rex))
    assert sp.issparse(C)
    f = np.arange(int(rex.nE)) % 7 - 3
    w = rex.harmonic_winding(f, cycles=C)
    assert w.shape == (C.shape[1],)
    assert np.issubdtype(w.dtype, np.integer)
