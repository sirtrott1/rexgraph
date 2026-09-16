"""Exact rank adapters, structural fast paths, full tower homology and no SciPy."""
import builtins
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.core import _sparse
from rexgraph.graded_boundary import (
    _exact_column_rank_reduction,
    _exact_rank_reduction,
    _integer_columns,
    _rank_integer_columns,
)
from rexgraph.linear_operator import boundary_operator, coboundary_operator
from rexgraph.native_rank import (
    betti_from_rex,
    boundary_rank,
    exact_tower,
    primary_columns,
    tower_chain_residual,
)
from rexgraph.native_sparse import NativeSparse, empty_native


@pytest.fixture
def no_scipy(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            pytest.fail("native rank attempted a SciPy import")
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("native rank exported a SciPy matrix")
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(_sparse, "to_scipy_csr", forbidden)


def _rex(supports):
    return RexGraph(boundary_ptr=np.cumsum([0] + [len(s) for s in supports]),
                    boundary_idx=[v for s in supports for v in s])


def _oracle_rank(shape, columns):
    # Deliberately small, independent rational row reduction reference oracle.
    rows = [[Q(columns[j].get(i, 0)) for j in range(shape[1])] for i in range(shape[0])]
    rank = 0
    for j in range(shape[1]):
        pivot = next((i for i in range(rank, shape[0]) if rows[i][j]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][j]
        rows[rank] = [v / scale for v in rows[rank]]
        for i in range(rank + 1, shape[0]):
            scale = rows[i][j]
            rows[i] = [a - scale*b for a, b in zip(rows[i], rows[rank], strict=True)]
        rank += 1
    return rank


@pytest.mark.parametrize("supports,expected", [
    ([], (0, 0)), ([[0, 0]], (1, 1)), ([[0]], (0, 0)),
    ([[0, 1], [1, 2], [2, 0]], (1, 1)),
    ([[0, 1], [2, 3]], (2, 0)),
    ([[0, 1, 2, 3]], (3, 0)),
    ([[0, 1, 2], [0, 1], [0, 2]], (1, 1)),
    ([[0, 0], [1, 0]], (1, 1)),
    ([[0, 1], [0, 1], [1]], (0, 1)),
])
def test_native_source_rank_betti_and_transpose(supports, expected, no_scipy):
    rex = _rex(supports)
    rank = boundary_rank(rex, 1)
    assert betti_from_rex(rex) == expected
    assert rex.betti == (*expected, 0)
    assert rank == rex.nV - expected[0]
    assert rank == rex.nE - expected[1]
    assert boundary_operator(rex, 1).exact_rank_factory()[0] == rank
    assert coboundary_operator(rex, 0).exact_rank_factory()[0] == rank
    assert coboundary_operator(rex, 1).exact_rank_factory() == (0, "integer-zero")


@pytest.mark.parametrize("branching", [False, True])
def test_structural_shortcuts_do_not_eliminate_or_populate_rank_memo(branching, monkeypatch):
    import rexgraph.graded_boundary as gb
    supports = [[v, v+1] for v in range(4000)]
    if branching:
        supports += [list(range(100)), list(range(3900, 4001))]
    rex = _rex(supports)
    monkeypatch.setattr(gb, "_exact_column_rank_reduction", lambda *a, **k: pytest.fail("elimination"))
    gb._RANK_MEMO.clear()
    rank, method = boundary_rank(rex, 1, return_info=True)
    assert rank == 4000
    assert method == ("spanned-branching-union-find" if branching else "pairwise-union-find")
    assert len(gb._RANK_MEMO) == 0


def test_fragmented_and_non_zero_sum_columns_refuse_structural_shortcut():
    for columns in ([{0:-1, 1:1}, {0:-3, 1:1, 2:1, 3:1}],
                    [{0:-1, 1:1}, {1:-1, 2:1}, {0:1, 1:1, 2:1}]):
        rank, method = _rank_integer_columns((4, len(columns)), columns)
        assert method == "sparse-integer-elimination"
        assert rank == _oracle_rank((4, len(columns)), columns)


def test_random_integer_ranks_and_pivot_rows_against_independent_rational_oracle():
    rng = np.random.default_rng(391)
    for _ in range(100):
        m, n = map(int, rng.integers(0, 9, size=2))
        dense = rng.integers(-5, 6, size=(m, n))
        dense[rng.random((m, n)) < .6] = 0
        columns = [{i:int(dense[i, j]) for i in range(m) if dense[i, j]} for j in range(n)]
        expected = _oracle_rank((m, n), columns)
        rank, pivots = _exact_column_rank_reduction((m, n), columns, with_pivots=True)
        assert rank == expected == _rank_integer_columns((m, n), columns)[0]
        assert len(pivots) == rank
        restricted = [{i:col.get(row, 0) for i, row in enumerate(pivots) if col.get(row, 0)} for col in columns]
        assert _oracle_rank((rank, n), restricted) == rank


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_duplicate_integer_entries_coalesce_in_Z(format, dtype):
    import scipy.sparse as sp
    # Both int64 overflow and floating cancellation would turn a nonzero into zero.
    big = 2**62 if dtype == np.int64 else 2**53
    vals = [big, big, big, big] if dtype == np.int64 else [big, 1, -big]
    n = len(vals)
    if format == "coo":
        matrix = sp.coo_matrix((np.array(vals, dtype=dtype), ([0]*n, [0]*n)), shape=(1, 1))
    else:
        constructor = sp.csr_matrix if format == "csr" else sp.csc_matrix
        matrix = constructor((np.array(vals, dtype=dtype), [0]*n, [0, n]), shape=(1, 1))
    assert _integer_columns(matrix) == [{0:sum(map(int, vals))}]
    assert _exact_rank_reduction(matrix) == 1


def test_native_duplicate_cancellation_and_memo_cross_carrier():
    import scipy.sparse as sp

    import rexgraph.graded_boundary as gb
    vals = np.array([2**53, 1, -2**53], float)
    dual = _sparse.dual_from_csr(_sparse.CSRMatrix(np.array([0, 3], np.int64),
        np.zeros(3, np.int64), vals, 1, 1))
    assert _integer_columns(NativeSparse(dual)) == [{0:1}]
    gb._RANK_MEMO.clear()
    assert _exact_rank_reduction(NativeSparse(dual)) == 1
    assert _exact_rank_reduction(sp.csc_matrix([[1]])) == 1
    assert len(gb._RANK_MEMO) == 1
    assert _exact_column_rank_reduction((1, 1), [{0:2**100}]) == 1
    assert len(gb._RANK_MEMO) == 2


def test_rank_memo_concurrent_hits_and_evictions_keep_exact_results():
    from concurrent.futures import ThreadPoolExecutor

    import rexgraph.graded_boundary as gb
    def read(i):
        return gb._exact_column_rank_reduction((1, 1), [{0:i % 100 + 1}])
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(read, range(1000))) == [1]*1000
    assert len(gb._RANK_MEMO) <= gb._RANK_MEMO_MAX


def test_chain_law_uses_original_shares_not_rank_scaling(no_scipy):
    rex = _rex([[0, 1, 2], [0, 1], [0, 2]])
    rex._nF = 1
    rex._B2_dual = _sparse.dual_from_coo([0, 1, 2], [0, 0, 0], [2., -1., -1.], 3, 1)
    assert rex.nF_hodge == 1
    shapes, columns = exact_tower(rex)
    assert tower_chain_residual(shapes, columns) == 0
    columns[0] = primary_columns(rex, integer=True)
    assert tower_chain_residual(shapes, columns) == 2
    assert betti_from_rex(rex) == (1, 0, 0)


def test_full_tower_with_empty_middle_grade(no_scipy):
    rex = _rex([[0, 1]])
    rex._graded_duals = [empty_native((0, 3)).dual,
        _sparse.dual_from_coo([0, 1], [0, 0], [-1., 1.], 3, 1)]
    assert betti_from_rex(rex) == (1, 0, 0, 2, 0)
    assert rex.betti == (1, 0, 0)


@pytest.mark.parametrize("bad", [False, True])
def test_upper_chain_law_not_assumed(bad, no_scipy):
    rex = _rex([[0, 1], [1, 2], [2, 0]])
    rex._nF = 2
    rex._B2_dual = _sparse.dual_from_coo([0, 1, 2]*2, [0]*3+[1]*3, [1.]*6, 3, 2)
    rex._graded_duals = [_sparse.dual_from_coo([0, 1], [0, 0], [1., 1. if bad else -1.], 2, 3)]
    if bad:
        with pytest.raises(ValueError, match="chain condition"):
            betti_from_rex(rex)
    else:
        assert betti_from_rex(rex) == (1, 0, 0, 2)


def test_higher_stored_large_integers_do_not_pass_through_float_view():
    import scipy.sparse as sp
    rex = _rex([[0, 0]])
    rex._nF = 2
    rex._B2_dual = empty_native((1, 2)).dual
    n = 2**60
    rex._graded_duals = [sp.csr_matrix(np.array([[n, n+1], [n-1, n]], dtype=np.int64))]
    assert boundary_rank(rex, 3) == 2  # determinant 1, lost by binary64 conversion
    assert boundary_operator(rex, 3).exact_rank_factory()[0] == 2
    assert betti_from_rex(rex) == (1, 1, 0, 0)


def test_operator_rank_certificate_retains_construction_time_coefficients():
    rex = _rex([[0, 1], [1, 2]])
    op = boundary_operator(rex, 1)
    # Storage mutation is unsupported for live handles; the captured exact action
    # and its rank must nevertheless describe the SAME construction, not a reread.
    rex._boundary_idx[:] = 0
    assert op.exact_rank_factory()[0] == 2
    assert boundary_rank(rex, 1) == 0


@pytest.mark.parametrize("value", [np.pi, 1.0000001, np.inf, np.nan])
def test_measured_upper_coefficients_are_not_rationalized(value):
    rex = _rex([[0, 0]])
    rex._nF = 1
    rex._B2_dual = empty_native((1, 1)).dual
    rex._graded_duals = [_sparse.dual_from_coo([0], [0], [value], 1, 1)]
    with pytest.raises(ValueError, match="integer higher"):
        boundary_rank(rex, 3)
    with pytest.raises(ValueError, match="integer higher"):
        betti_from_rex(rex)


@pytest.mark.parametrize("method", ["auto", "dense", "sparse"])
def test_compiled_rank_dispatch_is_native_for_integers(method, no_scipy):
    from rexgraph.core import _boundary
    dual = _sparse.dual_from_coo([0, 1, 1, 2], [0, 0, 1, 1], [-1., 1., -1., 1.], 3, 2)
    assert _boundary.compute_rank(dual, method=method) == 2
    assert _boundary.compute_rank(empty_native((0, 5)).dual, method=method) == 0


@pytest.mark.parametrize("dtype", [np.int64, np.float64, object])
def test_native_higher_state_roundtrip_preserves_exact_coefficients_and_digest(dtype, no_scipy):
    from rexgraph.io.catalog import object_digest
    from rexgraph.io.rex_state import from_state, to_state
    from rexgraph.native_sparse import csr_carrier
    # Upper determinant = 1, even above binary64/integer machine word range.
    n = 2**60 if dtype == np.int64 else 2**100 if dtype is object else 2**50
    rex = RexGraph.from_graph([0], [1])
    rex._graded_duals = [empty_native((0, 2)).dual, csr_carrier(np.array([0, 2, 4], np.int32),
        np.array([0, 1, 0, 1], np.int32), np.array([n, n+1, n-1, n], dtype=dtype), (2, 2))]
    restored = from_state(to_state(rex))
    assert object_digest(restored) == object_digest(rex)
    assert restored._graded_duals[1].vals.dtype == dtype
    assert boundary_rank(restored, 4) == 2
    assert restored.betti_tower == (1, 0, 0, 0, 0)
    np.testing.assert_array_equal(restored._graded_duals[1].vals, rex._graded_duals[1].vals)


@pytest.mark.parametrize("ptr,indices,data,shape", [
    ([1, 1], [], [], (1, 1)), ([0, 2], [0], [1], (1, 1)),
    ([0, 1], [-1], [1], (1, 1)), ([0, 1], [1], [1], (1, 1)),
    ([0, 2, 1], [0], [1], (2, 1)), ([0], [], [], (-1, 0)),
])
def test_native_restored_csr_addresses_are_checked(ptr, indices, data, shape):
    from rexgraph.native_sparse import csr_carrier
    with pytest.raises(ValueError, match="CSR carrier"):
        csr_carrier(np.array(ptr, np.int64), np.array(indices, np.int64), np.array(data), shape)
