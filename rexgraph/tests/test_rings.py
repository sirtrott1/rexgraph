"""Rings without choosing a basis.

A cycle basis holds beta_1 cycles and most structures have more rings than that,
so every basis drops some and nothing in the notion says which. The lattice
questions are basis-free: `shortest_cycles` is the minimal-vector set,
`relevant_cycles` is the cycles that are not a sum of strictly shorter ones,
which is the union of every minimum cycle basis.
"""

import itertools

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.rings import (
    cycle_candidates,
    minimum_cycle_basis,
    relevant_cycles,
    ring_sizes,
    shortest_cycles,
)


def _rex(edges):
    r = RexGraph(sources=np.array([a for a, _ in edges], np.int32),
                 targets=np.array([b for _, b in edges], np.int32))
    r._ensure_clean()
    return r


def _ring(n):
    return _rex([(i, (i + 1) % n) for i in range(n)])


def _cubane():
    """The 3-cube: eight atoms, twelve bonds, six square faces."""
    V = list(itertools.product([0, 1], repeat=3))
    ix = {v: i for i, v in enumerate(V)}
    return _rex([(ix[a], ix[b]) for a in V for b in V
                 if sum(x != y for x, y in zip(a, b, strict=False)) == 1 and ix[a] < ix[b]])


def _c60():
    from rexgraph.graded_boundary import build_graded_boundaries, truncated_icosahedron_3rex
    B1 = build_graded_boundaries(truncated_icosahedron_3rex())[0].toarray().astype(int)
    return _rex([tuple(int(v) for v in np.nonzero(B1[:, j])[0])
                 for j in range(B1.shape[1])])


#### the simple cases
def test_a_single_ring_has_one_ring():
    r = _ring(6)
    assert ring_sizes(r) == {6: 1}
    assert len(minimum_cycle_basis(r)) == 1


def test_two_fused_rings():
    """Naphthalene. Two hexagons sharing a bond, and the ten-cycle around the
    outside is their sum, so it is not relevant."""
    r = _rex([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
              (2, 6), (6, 7), (7, 8), (8, 9), (9, 3)])
    assert ring_sizes(r) == {6: 2}


def test_a_tree_has_no_rings():
    r = _rex([(0, 1), (1, 2), (2, 3), (1, 4)])
    assert relevant_cycles(r) == []
    assert shortest_cycles(r) == []
    assert ring_sizes(r) == {}


#### the cases a basis cannot answer
def test_cubane_keeps_all_six_faces():
    """beta_1 is five, so every basis drops a face and no rule says which. All six
    are relevant, and here they are also the minimal vectors."""
    r = _cubane()
    assert len(minimum_cycle_basis(r)) == 5
    assert len(relevant_cycles(r)) == 6
    assert ring_sizes(r) == {4: 6}


def test_c60_returns_twelve_pentagons_and_twenty_hexagons():
    """The real case. beta_1 is 31 against 32 faces, and the answer is derived
    from the bond graph, not read off the construction."""
    r = _c60()
    assert (r.nV, r.nE) == (60, 90)
    assert len(minimum_cycle_basis(r)) == 31
    assert ring_sizes(r) == {5: 12, 6: 20}


def test_shortest_is_not_relevant_and_c60_is_why():
    """A hexagon is one longer than a pentagon, so it is not a minimal vector.
    Reporting minimal vectors as the rings would return twelve where chemistry
    wants thirty-two."""
    r = _c60()
    assert len(shortest_cycles(r)) == 12
    assert len(relevant_cycles(r)) == 32


#### the properties that make these canonical
def test_the_basis_size_is_beta_1_and_relevant_is_at_least_that():
    from rexgraph.hodge_coords import harmonic_frame
    for build in (lambda: _ring(5), _cubane, _c60):
        r = build()
        b1 = harmonic_frame(r).shape[1]
        assert len(minimum_cycle_basis(r)) == b1
        assert len(relevant_cycles(r)) >= b1


def test_the_minimum_basis_weight_is_an_invariant_even_though_the_basis_is_not():
    """Which cycles get picked depends on the order candidates are seen; the total
    weight does not. That is exactly why the basis is the wrong thing to report."""
    r = _cubane()
    cands = cycle_candidates(r)
    base = sum(w for w, _ in minimum_cycle_basis(r, cands))
    rng = np.random.default_rng(0)
    for _ in range(5):
        shuffled = sorted(cands, key=lambda wm: (wm[0], rng.random()))
        assert sum(w for w, _ in minimum_cycle_basis(r, shuffled)) == base


def test_every_relevant_cycle_is_a_real_cycle_in_the_kernel():
    """A ring has to be a cycle: its boundary vanishes."""
    r = _cubane()
    B1 = np.asarray(r.B1_dense, float)
    for w, m in relevant_cycles(r):
        v = np.zeros(r.nE)
        for e in range(r.nE):
            if m >> e & 1:
                v[e] = 1.0
        assert int(v.sum()) == w
        # signs are the orientation; the support closes, so some signing is in ker
        idx = np.nonzero(v)[0]
        sub = B1[:, idx]
        assert np.linalg.matrix_rank(sub) == len(idx) - 1


#### the boundary of what this is defined on
def test_branching_relations_are_refused_rather_than_guessed():
    """A walk through a k-ary relation is not defined without saying which
    participant it leaves by."""
    r = RexGraph.from_hypergraph(np.array([0, 3], np.int64),
                                 np.array([0, 1, 2], np.int64))
    with pytest.raises(ValueError, match="2-ary"):
        relevant_cycles(r)


#### the diagnostics read the sparse dual, not the dense operator
def test_flow_diagnostics_agree_with_the_exact_rank():
    """gradient_dim and curl_dim are integers, so they are settled over the
    rationals rather than by thresholding singular values."""

    from rexgraph.flow.hyperflow import FlowComplex, _b1_csr
    from rexgraph.graded_boundary import _sparse_rank
    rng = np.random.default_rng(0)
    for nV in (40, 120):
        m = nV * 3
        s = rng.integers(0, nV, m).astype(np.int32)
        t = rng.integers(0, nV, m).astype(np.int32)
        k = s != t
        r = RexGraph(sources=s[k], targets=t[k])
        r._ensure_clean()
        assert FlowComplex(r).gradient_dim == _sparse_rank(_b1_csr(r))
    # and a branching complex, where the share 1/(k-1) makes the float form fragile
    r = RexGraph.from_hypergraph(np.array([0, 40, 42], np.int64),
                                 np.array(list(range(40)) + [0, 1], np.int64))
    r._ensure_clean()
    assert FlowComplex(r).gradient_dim == _sparse_rank(_b1_csr(r))


def test_the_diagnostics_never_materialise_the_dense_boundary():
    """`rex.B1` is a cached_property returning a dense nV x nE array and keeping
    it. The diagnostics read the dual instead, so asking for a dimension must not
    leave a dense B1 behind."""
    from rexgraph.flow.hyperflow import FlowComplex
    rng = np.random.default_rng(1)
    nV, m = 400, 1200
    s = rng.integers(0, nV, m).astype(np.int32)
    t = rng.integers(0, nV, m).astype(np.int32)
    k = s != t
    r = RexGraph(sources=s[k], targets=t[k])
    r._ensure_clean()
    fc = FlowComplex(r)
    assert "B1" not in r.__dict__, "fixture already cached it"
    _ = fc.gradient_dim, fc.curl_dim, fc.chain_residual
    assert "B1" not in r.__dict__, "a diagnostic materialised the dense B1"
    assert "B2" not in r.__dict__, "a diagnostic materialised the dense B2"


#### rings read the cycle space, not homology
def test_a_cycle_basis_counts_dim_z1_and_not_betti_one():
    """The distinction the module docstring used to blur. rings never consults B2,
    so with faces present a minimum cycle basis has dim Z1 = nE - rank(B1) members,
    which is strictly more than beta_1 once a face fills something."""
    import itertools

    import numpy as np

    from rexgraph.graph import RexGraph
    from rexgraph.rings import cycle_vectors, minimum_cycle_basis

    a, b = zip(*itertools.combinations(range(4), 2), strict=False)
    r = RexGraph(sources=np.array(a, np.int32), targets=np.array(b, np.int32))
    r._ensure_clean()
    bare = len(minimum_cycle_basis(r))
    assert bare == int(r.betti[1]) == 3, "face-free, the two agree"

    r.add_faces([[0, 1, 3]], signs=[[1.0, -1.0, 1.0]])
    r._ensure_clean()
    with_face = minimum_cycle_basis(r)
    assert int(r.betti[1]) == 2, "the face killed one class"
    assert len(with_face) == 3, "but the cycle space is unchanged"

    nE = int(r.nE)
    B1 = np.asarray(r.B1_dense, float)
    dim_z1 = nE - int(np.linalg.matrix_rank(B1))
    assert len(with_face) == dim_z1 == 3
    C = np.asarray(cycle_vectors(r, with_face).todense())
    assert np.abs(B1 @ C).max() < 1e-12, "still genuine cycles"
    assert int(np.linalg.matrix_rank(C)) == 3, "and still independent"


def test_a_filled_bigon_still_has_a_ring():
    """The sharpest case: beta_1 is zero and the ring is still there, because a ring
    is a fact about the 1-skeleton."""
    import numpy as np

    from rexgraph.graph import RexGraph
    from rexgraph.rings import minimum_cycle_basis

    r = RexGraph(sources=np.array([0, 0], np.int32), targets=np.array([1, 1], np.int32))
    r.add_faces([[0, 1]], signs=[[1.0, -1.0]])
    r._ensure_clean()
    assert int(r.betti[1]) == 0
    assert len(minimum_cycle_basis(r)) == 1
