"""The exact integer-rank reduction is memoized on exact matrix content so the same
boundary map (e.g. a shared B1 across two complexes in one monitor step) is reduced once.
These tests pin that the memo returns byte-exact results and never collides across
distinct matrices."""
import numpy as np
import scipy.sparse as sp

import rexgraph.graded_boundary as gb
from rexgraph.graded_boundary import _exact_rank_reduction


def _rand_int_sparse(m, n, density=0.2, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((m, n)) < density
    vals = rng.choice(np.array([-2, -1, 1, 2, 3]), size=(m, n))
    A = np.where(mask, vals, 0).astype(np.int64)
    return sp.csc_matrix(A)


def test_memo_returns_exact_rank_and_matches_dense():
    for seed in range(5):
        M = _rand_int_sparse(20, 25, seed=seed)
        exact = _exact_rank_reduction(M)
        dense = int(np.linalg.matrix_rank(M.toarray()))
        assert exact == dense


def test_identical_matrices_hit_the_memo_once():
    gb._RANK_MEMO.clear()
    M = _rand_int_sparse(30, 40, seed=1)
    r1 = _exact_rank_reduction(M)
    size_after_first = len(gb._RANK_MEMO)
    # a fresh, content-identical copy must hit the cache (no new entry)
    M2 = sp.csc_matrix((M.data.copy(), M.indices.copy(), M.indptr.copy()), shape=M.shape)
    r2 = _exact_rank_reduction(M2)
    assert r1 == r2
    assert len(gb._RANK_MEMO) == size_after_first == 1


def test_distinct_matrices_do_not_collide():
    gb._RANK_MEMO.clear()
    A = _rand_int_sparse(25, 25, seed=2)
    B = _rand_int_sparse(25, 25, seed=3)
    ra, rb = _exact_rank_reduction(A), _exact_rank_reduction(B)
    assert ra == int(np.linalg.matrix_rank(A.toarray()))
    assert rb == int(np.linalg.matrix_rank(B.toarray()))
    assert len(gb._RANK_MEMO) == 2  # two distinct keys


def test_memo_is_bounded():
    gb._RANK_MEMO.clear()
    for i in range(gb._RANK_MEMO_MAX + 20):
        _exact_rank_reduction(_rand_int_sparse(12, 14, seed=100 + i))
    assert len(gb._RANK_MEMO) <= gb._RANK_MEMO_MAX
