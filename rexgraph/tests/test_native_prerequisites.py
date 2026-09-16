"""Native contracts; dense matrices appear only in independent test oracles."""
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.dirac_propagator import SparseDirac
from rexgraph.graph import RexGraph
from rexgraph.rational_trig import exact_channel_diagonals
from rexgraph.sparse_character import build_sparse_channels, channel_diagonals, channel_diagonals_integer


def _forbidden(*args, **kwargs):
    raise AssertionError("native path reached a dense/eigen oracle")


def test_block_cg_reports_unconverged_result():
    from rexgraph.sparse_character import _block_cg
    A = sp.diags([1., 2., 7.], format="csr")
    rhs = np.ones((3, 1))
    with pytest.raises(ArithmeticError, match="residual tolerance"):
        _block_cg(lambda x: A @ x, rhs, np.ones(3), maxit=1)
    answer = _block_cg(lambda x: A @ x, rhs, np.ones(3))
    assert np.linalg.norm(A @ answer-rhs) < 1e-10


@pytest.mark.parametrize("sizes", [(1, 1), (2, 3, 2), (0, 0)])
def test_dirac_bound_is_incidence_only(monkeypatch, sizes):
    boundaries = [sp.csr_matrix(np.ones((a, b))) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
    D = SparseDirac(boundaries)
    expected = np.asarray(abs(D.to_scipy()).sum(axis=1)).max(initial=0.0)
    monkeypatch.setattr(np.linalg, "eigvalsh", _forbidden)
    monkeypatch.setattr(sp.linalg, "eigsh", _forbidden)
    monkeypatch.setattr(D, "to_scipy", _forbidden)
    assert D.spectral_radius() == pytest.approx(max(expected, 1e-12))


@pytest.mark.parametrize("t", [-300., 0., 300.])
def test_default_dirac_light_resolves_both_time_directions(monkeypatch, t):
    D = SparseDirac([sp.csr_matrix([[-1., 0., 1.], [1., -1., 0.], [0., 1., -1.]])])
    w, V = np.linalg.eigh(D.to_scipy().toarray())
    psi = np.arange(6, dtype=float)
    want_re = V @ (np.cos(t*w) * (V.T @ psi))
    want_im = V @ (-np.sin(t*w) * (V.T @ psi))
    monkeypatch.setattr(np.linalg, "eigh", _forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", _forbidden)
    monkeypatch.setattr(sp.linalg, "eigsh", _forbidden)
    monkeypatch.setattr(D, "to_scipy", _forbidden)
    re, im = D.light(psi, t)
    np.testing.assert_allclose(re, want_re, atol=1e-10)
    np.testing.assert_allclose(im, want_im, atol=1e-10)


def test_dirac_rejects_mismatched_grades_and_invalid_polynomial():
    with pytest.raises(ValueError, match="grade shapes"):
        SparseDirac([sp.csr_matrix((2,3)), sp.csr_matrix((4,1))])
    D = SparseDirac([sp.csr_matrix([[-1.], [1.]])])
    with pytest.raises(ValueError, match="order"):
        D.light(np.ones(3), 1., order=1)
    with pytest.raises(ValueError, match="lam_max"):
        D.light(np.ones(3), 1., order=24, lam_max=0.)


def test_default_field_assembly_does_not_read_dense_accessor(monkeypatch):
    from rexgraph.field_propagator import assemble_field_operator
    rex = RexGraph.from_simplicial(np.array([0, 0, 1]), np.array([1, 2, 2]),
                                   np.array([[0, 1, 2]]))
    expected = assemble_field_operator(rex).toarray()
    monkeypatch.setattr(RexGraph, "relational_laplacian", property(_forbidden))
    monkeypatch.setattr(sp.csr_matrix, "toarray", _forbidden)
    actual = assemble_field_operator(rex)
    assert sp.issparse(actual)
    assert np.allclose(actual @ np.ones(4), expected @ np.ones(4))


def test_second_trace_uses_primary_rationals():
    rex = RexGraph.from_hypergraph(np.array([0, 4, 10, 18]),
                                   np.array(list(range(4))+list(range(6))+list(range(8))))
    arities = [4, 6, 8]
    columns = [[Fraction(-1) if i == 0 else Fraction(1, k-1) if i < k else Fraction(0)
                for i in range(8)] for k in arities]
    expected = sum((sum(a*b for a, b in zip(c, d, strict=True))**2 for c in columns for d in columns), Fraction(0))
    assert rex._second_traces() == (expected, Fraction(0))


def test_sparse_kernel_keeps_zero_columns_and_large_integer_coordinates():
    from rexgraph.graded_boundary import _exact_kernel_columns
    big = 2**60 + 1
    columns = [{4: Fraction(big)}, {4: Fraction(big+1)}, {}]
    basis = list(_exact_kernel_columns(columns))
    assert len(basis) == 2
    assert basis[0] == {0: -Fraction(big+1, big), 1: Fraction(1)}
    assert basis[1] == {2: Fraction(1)}


def test_branching_native_kernel_never_expands_coordinate_lists(monkeypatch):
    from rexgraph import faces, harmonic_sparse
    rex = RexGraph(boundary_ptr=np.array([0,3,5,7]), boundary_idx=np.array([0,1,2,0,1,0,2]))
    monkeypatch.setattr(faces, "_cycle_basis_kernel", _forbidden)
    monkeypatch.setattr(faces, "cycle_basis", _forbidden)
    monkeypatch.setattr(sp.csc_matrix, "toarray", _forbidden)
    basis = harmonic_sparse.cycle_basis(rex)
    assert basis.shape == (3, 1)
    assert basis.nnz == 3
    from rexgraph.core._sparse import to_scipy_csr
    assert (to_scipy_csr(rex.B1_sparse) @ basis).nnz == 0
    assert [s.tolist() for s in faces.cycle_supports(rex)] == [[0,1,2]]


def test_native_harmonic_reduction_is_sparse_and_never_falls_back(monkeypatch):
    from rexgraph.harmonic_sparse import _face_reduced_frame
    import scipy.linalg
    C = sp.eye(3, format="csc")
    M = sp.csr_matrix([[1., 1., 0.]])
    monkeypatch.setattr(sp.csr_matrix, "todense", _forbidden)
    monkeypatch.setattr(sp.csc_matrix, "todense", _forbidden)
    monkeypatch.setattr(scipy.linalg, "null_space", _forbidden)
    H = _face_reduced_frame(C, M)
    assert H.shape == (3,2) and H.nnz == 3
    assert (M @ H).nnz == 0
    with pytest.raises(ValueError, match="integral flux"):
        _face_reduced_frame(C, sp.csr_matrix([[0.5, 1., 0.]]))


def test_harmonic_flux_is_not_rounded_before_the_exact_kernel():
    from rexgraph.harmonic_sparse import _face_reduced_frame
    C = sp.csc_matrix([[float(2**53), 1.], [1., 0.]])
    B2 = sp.csc_matrix([[1.], [1.]])
    # Exact flux is [2**53+1, 1]. A float product rounds away that +1 and
    # invents a representable primitive kernel with a nonzero true residual.
    with pytest.raises(ValueError, match="primitive coordinates"):
        _face_reduced_frame(C, B2=B2)


def test_coboundary_determinant_keeps_primary_integers(monkeypatch):
    from rexgraph import coboundary_volume as module
    rex = RexGraph(sources=np.array([0]), targets=np.array([1]))
    big = 2**53 + 1
    B = sp.csc_matrix((np.array([big,2], dtype=np.int64), ([0,1],[0,1])), shape=(2,2))
    monkeypatch.setattr(module, "_integer_coboundary", lambda rex, grade: B)
    monkeypatch.setattr(sp.csr_matrix, "todense", _forbidden)
    monkeypatch.setattr(sp.csc_matrix, "todense", _forbidden)
    assert module.coboundary_volume(rex, 1) == (2*big)**2


@pytest.mark.parametrize("reading", ["share", "count"])
def test_all_character_readers_honor_c_channel(reading):
    rex = RexGraph(boundary_ptr=np.array([0, 4, 7, 9]),
                   boundary_idx=np.array([0, 1, 2, 3, 1, 2, 3, 0, 1]), c_channel=reading)
    expected = dict(build_sparse_channels(rex))["L_C"].diagonal()
    exact = np.array([float(v) for v in exact_channel_diagonals(rex)[0]["L_C"]])
    assert np.allclose(exact, expected)
    assert np.allclose(channel_diagonals(rex)["L_C"], expected)
    integers, scale = channel_diagonals_integer(rex)
    assert np.allclose(integers["L_C"] / scale, expected)


@pytest.mark.parametrize("reading", ["share", "count"])
def test_integer_character_includes_exact_relation_weights(reading):
    rex = RexGraph(boundary_ptr=np.array([0,3,5]), boundary_idx=np.array([0,1,2,0,1]),
                   w_E=np.array([Fraction(3,2),Fraction(1,3)], dtype=object), c_channel=reading)
    integers, scale = channel_diagonals_integer(rex)
    expected, names = exact_channel_diagonals(rex)
    assert integers is not None
    for name in names:
        assert [Fraction(int(value),scale) for value in integers[name]] == expected[name]


def test_diagonal_readers_distinguish_normalized_g_from_raw():
    rex = RexGraph(sources=np.array([0,1]), targets=np.array([1,2]), g_channel="normalized")
    np.testing.assert_allclose(channel_diagonals(rex)['L_O'], [1/3, 1/3])
    assert channel_diagonals_integer(rex) == (None,None)
    assert exact_channel_diagonals(rex)[0]['L_O'] == [Fraction(1,3)] * 2


def test_exact_coupling_caches_follow_mutations():
    rex = RexGraph(sources=np.array([0,1,2]), targets=np.array([1,2,0]))
    assert (rex.trace_T,rex.trace_L1,rex.c2_E,rex.c2_H,rex.c0_squared) == (6,0,0,0,0)
    assert rex.relational_laplacian_sparse.shape == (3,3)
    rex.add_faces([[0,1,2]], [[1.,1.,1.]])
    assert (rex.trace_T,rex.trace_L1,rex.c2_E,rex.c2_H,rex.c0_squared) == (6,3,Fraction(1,2),Fraction(1,2),Fraction(1,2))
    rex.add_edges(np.array([3]), np.array([4]))
    assert rex.trace_T == 8
    assert rex.c2_E == Fraction(9,22)
    assert rex.c2_H == Fraction(11,32)
    assert rex.c0_squared == Fraction(3,8)
    assert rex.relational_laplacian_sparse.shape == (4,4)
