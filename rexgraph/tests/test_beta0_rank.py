"""beta_0 is n_0 - rank(B_1), which is a component count only on a pairwise graph.

`betti_numbers` took beta_0 from a union-find over components and its docstring called
that "equivalently ``n_0 - rank(B_1)``". The two agree when every relation has arity 2
and part otherwise: rank(B_1) = n_0 - c is a GRAPH identity, and a relation of arity k
touches k vertices while contributing rank one, so reaching a new vertex stops meaning
reaching a new direction. Only the second is what beta_0 counts.

The arbiter is the Euler characteristic, beta_0 - beta_1 + beta_2 = n_0 - n_1 + n_2,
which needs no convention to settle. Under the component count it fails on the first
relation of arity greater than two.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _branching(ptr, idx):
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int32), np.asarray(idx, np.int32))


# (name, boundary_ptr, boundary_idx, expected beta_0)
FIXTURES = [
    ("pairwise triangle",          [0, 2, 4, 6], [0, 1, 1, 2, 2, 0], 1),
    ("two disjoint edges",         [0, 2, 4],    [0, 1, 2, 3],       2),
    ("lone k=3",                   [0, 3],       [0, 1, 2],          2),
    ("lone k=4",                   [0, 4],       [0, 1, 2, 3],       3),
    ("lone k=6",                   [0, 6],       [0, 1, 2, 3, 4, 5], 5),
    ("double-T",                   [0, 3, 6],    [0, 1, 2, 0, 1, 3], 2),
    ("three hyperedges on a pair", [0, 3, 6, 9], [0, 1, 2, 0, 1, 3, 0, 1, 4], 2),
    ("hyperedge = mean of two",    [0, 3, 5, 7], [0, 1, 2, 0, 1, 0, 2], 1),
    # arity-3 contributes rank 1, the disjoint triangle rank 2, so beta_0 = 6 - 3
    ("k=3 beside a triangle",      [0, 3, 5, 7, 9],
                                   [0, 1, 2, 3, 4, 4, 5, 5, 3], 3),
]


@pytest.mark.parametrize("name,ptr,idx,want", FIXTURES)
def test_beta0_is_rank_based(name, ptr, idx, want):
    rex = _branching(ptr, idx)
    assert int(rex.betti[0]) == want, name


@pytest.mark.parametrize("name,ptr,idx,want", FIXTURES)
def test_euler_closes(name, ptr, idx, want):
    """The property that fixes the convention. nF is 0 in every fixture here."""
    rex = _branching(ptr, idx)
    b0, b1, b2 = (int(x) for x in rex.betti)
    chi = int(rex.nV) - int(rex.nE) + int(rex.nF_hodge)
    assert b0 - b1 + b2 == chi, f"{name}: betti={(b0, b1, b2)} chi={chi}"


@pytest.mark.parametrize("name,ptr,idx,want", FIXTURES)
def test_beta0_equals_nullity_of_b1_transpose(name, ptr, idx, want):
    """The definition itself, computed independently of the implementation."""
    import scipy.sparse as sp

    rex = _branching(ptr, idx)
    B1 = rex.B1
    B1 = B1.toarray() if sp.issparse(B1) else np.asarray(B1)
    assert int(rex.betti[0]) == B1.shape[0] - np.linalg.matrix_rank(B1), name


def test_a_component_count_would_disagree_on_branching():
    """Pins the distinction rather than assuming it: a lone arity-4 relation is one
    connected component and has beta_0 = 3."""
    rex = _branching([0, 4], [0, 1, 2, 3])
    assert int(rex.betti[0]) == 3
    assert int(rex.nV) == 4        # one component by any traversal


def test_pairwise_graphs_are_unchanged():
    """Where the graph identity holds, nothing moves."""
    # a triangle plus a disjoint edge: two components, one cycle
    rex = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                   targets=np.array([1, 2, 0, 4], np.int32))
    assert tuple(int(x) for x in rex.betti) == (2, 1, 0)
    chi = int(rex.nV) - int(rex.nE) + int(rex.nF_hodge)
    b0, b1, b2 = (int(x) for x in rex.betti)
    assert b0 - b1 + b2 == chi


#### the rank itself
def test_a_self_loop_does_not_inflate_the_rank():
    """A self-loop stores -1 and +1 at the same (row, col), so duplicates must be
    summed before the column is read. An unsummed duplicate overwrites instead of
    cancelling, leaving a zero column that registers a spurious pivot: rank 2 on a
    matrix of rank 1, which puts both beta_0 and beta_1 out by one."""
    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.graded_boundary import _sparse_rank

    rex = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([0, 2], np.int32))
    A = to_scipy_csr(rex._B1_dual)
    assert _sparse_rank(A) == np.linalg.matrix_rank(A.toarray()) == 1
    assert tuple(int(x) for x in rex.betti) == (2, 1, 0)


def test_the_self_loop_cycle_is_counted():
    """A self-loop's column is zero, so it lies in ker(B1) and is an independent cycle.
    The parallel pair contributes another."""
    rex = RexGraph(sources=np.array([0, 0, 1, 1], np.int32),
                   targets=np.array([1, 1, 1, 2], np.int32))
    b = [int(x) for x in rex.betti]
    assert b[1] == 2                                    # parallel pair + self-loop
    assert b[0] - b[1] + b[2] == int(rex.nV) - int(rex.nE) + int(rex.nF_hodge)


@pytest.mark.parametrize("ptr,idx,want_rank", [
    ([0, 3], [0, 1, 2], 1),
    ([0, 3, 5], [0, 1, 2, 0, 1], 2),
    ([0, 3, 5, 7], [0, 1, 2, 0, 1, 0, 2], 2),           # h is the mean of p1, p2
    ([0, 7], list(range(7)), 1),
    ([0, 4, 6, 8], [0, 1, 2, 3, 3, 4, 3, 5], 3),
])
def test_branching_rank_stays_on_the_integer_path(ptr, idx, want_rank):
    """A branching complex must not leave the integer tower to have its rank taken.

    The stored column carries the share 1/(k-1) and so looks rational, but it has an
    exact INTEGER representative: scaling by (k-1) gives (-(k-1), +1, ..., +1), still
    zero-sum and still (-1, +1) at k=2. Rank is invariant under column scaling, so the
    rank path is handed that and no fraction ever enters. Reconstructing 1/3 from its
    nearest double would be answering a question that should not be asked.
    """
    from rexgraph.graded_boundary import _exact_rank_reduction, _is_integer_matrix

    rex = _branching(ptr, idx)
    M = rex._integer_B1()
    assert _is_integer_matrix(M), "the rank path was handed a non-integer boundary"
    assert np.asarray(M.sum(axis=0)).ravel().tolist() == [0] * int(rex.nE)
    got = _exact_rank_reduction(M)
    assert got is not None, "declined the exact path on an integer boundary"
    assert got == want_rank == np.linalg.matrix_rank(M.toarray())


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_the_integer_representative_is_the_share_cleared(k):
    """(-1, 1/(k-1), ...) scaled by (k-1). At k=2 the two coincide exactly."""
    rex = _branching([0, k], list(range(k)))
    col = rex._integer_B1().toarray()[:, 0]
    assert sorted(col.tolist()) == sorted([-(k - 1)] + [1] * (k - 1))
    assert col.sum() == 0


def test_the_stored_boundary_keeps_the_share():
    """The channels are not scale-free: T and G weight a wide relation less per leg than
    a narrow one, and that is the content of the share there. Only the rank path, where
    scale is free, uses the cleared form."""
    import scipy.sparse as sp

    rex = _branching([0, 4], [0, 1, 2, 3])
    B = rex.B1
    B = B.toarray() if sp.issparse(B) else np.asarray(B)
    assert np.allclose(sorted(B[:, 0]), sorted([-1.0, 1 / 3, 1 / 3, 1 / 3]))


def test_the_exact_path_declines_a_genuine_float():
    """It must not pretend. An entry that is not a small rational falls back."""
    import scipy.sparse as sp

    from rexgraph.graded_boundary import _exact_rank_reduction, _sparse_rank

    M = sp.csc_matrix(np.array([[np.pi, 0.0], [0.0, np.sqrt(2)]]))
    assert _exact_rank_reduction(M) is None
    assert _sparse_rank(M) == 2                          # the float path still answers
