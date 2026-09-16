"""Native actions, immutable carriers and explicit optional SciPy exports."""
import builtins
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.core import _sparse
from rexgraph.linear_operator import boundary_operator, coboundary_operator, hodge_operator
from rexgraph.native_sparse import NativeSparse, empty_native, native_boundaries


@pytest.fixture
def no_scipy(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy forbidden in native action")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    # Cython may cache imported callables; forbid the bridge itself as well.
    def no_export(*args, **kwargs):
        raise ImportError("SciPy forbidden in native action")
    monkeypatch.setattr(_sparse, "to_scipy_csr", no_export)
    return guarded


@pytest.mark.parametrize("indices", [np.int32, np.int64])
@pytest.mark.parametrize("coefficients", [np.float32, np.float64])
def test_cython_actions_accept_readonly_inputs_and_storage(indices, coefficients):
    dual = _sparse.dual_from_coo(np.array([0, 1, 1], dtype=indices),
        np.array([0, 0, 2], dtype=indices), np.array([2, -3, 4], dtype=coefficients), 2, 3)
    for array in (dual.row_ptr, dual.col_idx, dual.vals, dual.col_ptr, dual.row_idx, dual.vals_csc):
        array.setflags(write=False)
    x = np.array([1, 0, 2], dtype=coefficients)
    y = np.array([1, 2], dtype=coefficients)
    x.setflags(write=False)
    y.setflags(write=False)
    np.testing.assert_array_equal(_sparse.matvec(dual, x), [2, 5])
    np.testing.assert_array_equal(_sparse.rmatvec(dual, y), [-4, 0, 8])


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (4, 3)])
@pytest.mark.parametrize("block", [None, 0, 2])
def test_native_empty_shapes_preserve_complex_vector_and_block_axes(shape, block, no_scipy):
    matrix = empty_native(shape)
    tail = () if block is None else (block,)
    x, y = np.ones((shape[1], *tail), complex), np.ones((shape[0], *tail), complex)
    assert matrix.apply(x).shape == y.shape
    assert matrix.transpose_apply(y).shape == x.shape
    np.testing.assert_array_equal(matrix.apply(x), np.zeros_like(y))
    assert np.iscomplexobj(matrix.apply(x))


def test_numeric_complex_noncontiguous_action_uses_both_real_kernels(monkeypatch, no_scipy):
    matrix = NativeSparse(_sparse.dual_from_coo([0, 1, 1], [0, 0, 2], [2., -3., 4.], 2, 3))
    calls = []
    original = _sparse.matvec

    def counted(dual, vector):
        calls.append(vector.copy())
        return original(dual, vector)

    monkeypatch.setattr(_sparse, "matvec", counted)
    x = (np.arange(12).reshape(3, 4) * (2+3j))[:, ::2]
    np.testing.assert_array_equal(matrix.apply(x), np.array([[2, 0, 0], [-3, 0, 4]]) @ x)
    assert len(calls) == 4 and all(not np.iscomplexobj(v) for v in calls)
    with pytest.raises(ValueError, match="axis"):
        matrix.apply(np.zeros(4))
    with pytest.raises(TypeError, match="booleans"):
        matrix.apply(np.ones(3, bool))
    with pytest.raises(FloatingPointError):
        matrix.apply(np.full(3, np.inf))
    with pytest.raises(ImportError, match="SciPy forbidden"):
        matrix.as_scipy()


def test_branching_exact_action_and_numerical_adjoint_need_no_scipy(no_scipy):
    rex = RexGraph.from_hypergraph([0, 4, 6], [0, 1, 2, 3, 1, 0])
    b = boundary_operator(rex, 1)
    x = np.array([Q(3), Q(1)], dtype=object)
    np.testing.assert_array_equal(b.apply(x, exact=True), [Q(-2), Q(0), Q(1), Q(1)])
    z = np.arange(4) + 1j*np.arange(4)[::-1]
    cob = coboundary_operator(rex, 0)
    np.testing.assert_allclose(cob.apply(z), b.transpose_apply(z))
    assert b.matrix is None and cob.matrix is None
    np.testing.assert_allclose(hodge_operator(rex, 1).apply(np.ones(2)),
                               b.transpose_apply(b.apply(np.ones(2))))
    assert coboundary_operator(rex, 1).apply(np.ones(2)).shape == (0,)


def test_native_face_filter_and_exact_grade2_sheaf(no_scipy):
    from rexgraph.chain_map import CoordinateComplex, GradedMap
    from rexgraph.sheaf import ExactSheaf, Sheaf
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    # One nonbounding decorative face and one genuine cycle.
    rex._nF = 2
    rex._B2_dual = _sparse.dual_from_coo([0, 0, 1, 2], [0, 1, 1, 1], [1., 1., 1., 1.], 3, 2)
    assert rex.nF_hodge == 1
    np.testing.assert_array_equal(rex._chain_col_bounds, [False, True])
    bounds = native_boundaries(rex)
    assert [b.shape for b in bounds] == [(3, 3), (3, 1)]
    assert bounds[0].dual is rex._B1_dual
    np.testing.assert_array_equal(bounds[0].apply(bounds[1].apply(np.ones(1))), np.zeros(3))
    # Exact coordinate capture retains raw faces rather than the filtered view.
    c = CoordinateComplex.from_rex(rex)
    assert c.sizes == (3, 3, 2)
    identity = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    with pytest.raises(ValueError, match="chain law failed"):
        identity.verify()
    sh = ExactSheaf(rex, grade=2)
    assert sh.n_cells == 1 and sh.check_section().compatible
    assert Sheaf(rex, grade=2).n_cells == 1


def test_large_integral_face_cancellation_is_decided_exactly(no_scipy):
    # All three B1 columns are equal. Float summation can lose the middle 1
    # in 2**53 + 1 - 2**53, but the face does NOT bound.
    rex = RexGraph.from_graph([0, 0, 0], [1, 1, 1])
    rex._nF = 1
    rex._B2_dual = _sparse.dual_from_coo([0, 1, 2], [0, 0, 0],
                                      [float(2**53), 1., -float(2**53)], 3, 1)
    np.testing.assert_array_equal(rex._chain_col_bounds, [False])
    assert rex.nF_hodge == 0


def test_native_explicit_empty_middle_grade_keeps_the_tower(no_scipy):
    rex = RexGraph.from_graph([0], [1])
    rex._graded_duals = [empty_native((0, 2)).dual]
    assert [b.shape for b in native_boundaries(rex)] == [(2, 1), (1, 0), (0, 2)]
    assert boundary_operator(rex, 3).apply(np.ones(2)).shape == (0,)


@pytest.mark.parametrize("g", ["raw", "normalized"])
@pytest.mark.parametrize("c", ["share", "count"])
@pytest.mark.parametrize("weighted", [False, True])
def test_all_channel_actions_and_diagonals_without_scipy(g, c, weighted, monkeypatch):
    from rexgraph.channel_operator import channel_operator
    from rexgraph.sparse_character import build_sparse_channels
    rex = RexGraph(boundary_ptr=[0, 4, 6, 7], boundary_idx=[0, 1, 2, 3, 1, 0, 2],
                   g_channel=g, c_channel=c, w_E=np.array([2., 0., 3.]) if weighted else None)
    # Traditional assembled oracle is built BEFORE the import prohibition.
    oracle = dict(build_sparse_channels(rex))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise AssertionError("native channel imported scipy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    def no_export(*args, **kwargs):
        raise AssertionError("native channel exported scipy")
    monkeypatch.setattr(_sparse, "to_scipy_csr", no_export)
    values = np.array([[1+2j, 3], [2-1j, 4], [5j, 1]])
    for name, key in zip("TGFC", ("L1_down", "L_O", "L_SG", "L_C"), strict=True):
        op = channel_operator(rex, name)
        expected = oracle[key]
        np.testing.assert_allclose(op.apply(values), expected @ values, atol=1e-12)
        np.testing.assert_allclose(op.diagonal(), expected.diagonal(), atol=1e-12)
