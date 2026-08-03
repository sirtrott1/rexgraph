"""A branching boundary column must be signed AND sum to zero.

`_build_B1_general` wrote -1 at the first boundary entry and +1 at every other, so a
column of arity k summed to k-2. That is the star, which is the existence tensor: a
consistent object, but not a boundary. The condition that makes a column a boundary is
that it is signed and sums to zero, and the share 1/(k-1) is what delivers that at every
arity, with the ordinary edge as the k=2 case where the share is 1.

Four consequences are pinned here, each of which fails under the star.

ZERO COLUMN SUM, the definition itself.

LEVEL LINKING. sum(c) = 0 implies c c^T 1 = c (c^T 1) = 0, so zero column sums of B1
give zero row sums of L0 and put the constant vector in ker L0. Under the star the row
sums are nonzero and the constant vector is outside the kernel, which is the same
failure the unsigned set-theoretic encoding has.

ARITY IS RECOVERABLE. For the rank-one contribution P_e = c_e c_e^T of one relation,
P_e[d,d]/P_e[i,i] = (k-1)^2, so k = 1 + sqrt(ratio). Under the star every modulus is 1,
the ratio is 1, and the reading returns 2 whatever the arity.

A GENUINE LINEAR DEPENDENCY SURVIVES. A relation that IS the mean of two pairwise
relations is a real element of ker B1. The star destroys that dependency and with it a
class in H_1.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.graph import RexGraph


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _branching(ptr, idx):
    return RexGraph.from_hypergraph(np.asarray(ptr, np.int32), np.asarray(idx, np.int32))


def _lone(k):
    return _branching([0, k], list(range(k)))


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8, 12])
def test_a_branching_column_sums_to_zero(k):
    col = _dense(_lone(k).B1)[:, 0]
    assert abs(col.sum()) < 1e-12, f"k={k} column sums to {col.sum()}"


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_the_share_is_one_over_k_minus_one(k):
    col = _dense(_lone(k).B1)[:, 0]
    nz = col[col != 0]
    assert (nz < 0).sum() == 1, "exactly one distinguished entry"
    assert np.isclose(nz[nz < 0][0], -1.0)
    assert np.allclose(nz[nz > 0], 1.0 / (k - 1))


def test_the_ordinary_edge_is_the_k_equals_2_case():
    """The share is 1/(2-1) = 1, so a pairwise column is exactly (-1, +1) and nothing
    about the standard path moves."""
    r = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    B1 = _dense(r.B1)
    for e in range(B1.shape[1]):
        nz = B1[:, e][B1[:, e] != 0]
        assert sorted(nz) == [-1.0, 1.0]


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_level_linking_puts_the_constant_vector_in_the_kernel(k):
    B1 = _dense(_lone(k).B1)
    L0 = B1 @ B1.T
    assert np.allclose(L0.sum(axis=1), 0.0, atol=1e-12), "row sums of L0 must vanish"
    assert np.allclose(L0 @ np.ones(L0.shape[0]), 0.0, atol=1e-12)


def test_level_linking_holds_on_a_mixed_complex():
    """Not only in isolation: a complex mixing arities still has 1 in ker L0."""
    r = _branching([0, 4, 6, 9], [0, 1, 2, 3, 3, 4, 4, 5, 6])
    B1 = _dense(r.B1)
    L0 = B1 @ B1.T
    assert np.allclose(L0 @ np.ones(L0.shape[0]), 0.0, atol=1e-12)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_arity_is_readable_from_the_rank_one_diagonal(k):
    """k = 1 + sqrt(P[d,d]/P[i,i]) on the relation's own column."""
    col = _dense(_lone(k).B1)[:, 0]
    nz = col[col != 0]
    d0 = nz[nz < 0][0] ** 2
    di = nz[nz > 0][0] ** 2
    assert np.isclose(1.0 + np.sqrt(d0 / di), k)


def test_arity_reading_survives_inside_a_complex():
    """The identity is about the per-relation rank-one term, so other relations being
    present must not disturb it."""
    r = _branching([0, 4, 6, 8], [0, 1, 2, 3, 3, 4, 3, 5])   # k=4 plus two pairwise legs
    col = _dense(r.B1)[:, 0]
    nz = col[col != 0]
    d0 = nz[nz < 0][0] ** 2
    di = nz[nz > 0][0] ** 2
    assert np.isclose(1.0 + np.sqrt(d0 / di), 4)


def test_the_mean_relation_is_a_real_kernel_element():
    """h over {a,b,c} distinguished at a, with pairwise a-b and a-c: under the share
    c_h = (c_p1 + c_p2)/2 exactly, so rank(B1) = 2 rather than 3."""
    r = _branching([0, 3, 5, 7], [0, 1, 2, 0, 1, 0, 2])
    B1 = _dense(r.B1)
    h, p1, p2 = B1[:, 0], B1[:, 1], B1[:, 2]
    assert np.allclose(h, 0.5 * (p1 + p2), atol=1e-12)
    assert np.linalg.matrix_rank(B1) == 2


def test_the_witness_column_is_the_deliberate_exception():
    """A single entry cannot sum to zero and is not a boundary."""
    col = _dense(_branching([0, 1], [0]).B1)[:, 0]
    assert col[col != 0].tolist() == [1.0]


def test_the_channels_still_partition_unity_at_every_arity():
    """The share must not break the character's normalisation."""
    for k in (3, 4, 5, 6):
        r = _branching([0, k, k + 2], list(range(k)) + [k - 1, k])
        chi = np.asarray(r.structural_character)
        assert np.allclose(chi.sum(axis=1), 1.0, atol=1e-9), k


def test_the_gramian_carries_the_share():
    """Step 1's helper is where the share reaches the unsigned side: K = |B1|^T|B1|
    must now weight a wide relation less than a narrow one."""
    r = _branching([0, 3, 6], [0, 1, 2, 0, 1, 3])
    B1 = _dense(r.B1)
    K = r.overlap_gramian_sparse.toarray()
    assert np.allclose(K, np.abs(B1).T @ np.abs(B1), atol=1e-12)
