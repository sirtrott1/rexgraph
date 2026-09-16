"""The exact rational path is primary; the float tower is its oracle.

Fixtures are the worked examples of *Exact Relational Field Calculus*, so a failure
here is a disagreement with the paper, not with a previous run of this code.
"""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.core._common import configure_algorithms, get_algorithm_config
from rexgraph.core._linalg import (
    eigh,
    frame_project,
    frame_projector,
    harmonic_pinv_matvec,
    least_quadrance,
    metric_cg,
    pinv_spectral,
)
from rexgraph.sparse_interfacing import _component_frame, _component_projector

TRIANGLE_PTR = np.asarray([0, 2, 4, 6], np.int64)
TRIANGLE_IDX = np.asarray([0, 1, 1, 2, 2, 0], np.int64)


def triangle(filled=False, **kw):
    rex = RexGraph(boundary_ptr=TRIANGLE_PTR, boundary_idx=TRIANGLE_IDX, **kw)
    if filled:
        rex.add_faces([[0, 1, 2]])
    return rex


def frac(values):
    return tuple(Q(v) for v in values)


class TestPaperGreenFixtures:
    """Section "an open cycle and a filled face" and its neighbours."""

    def test_open_and_filled_triangle(self):
        source = frac([1, 0, 0])
        assert triangle().green(source, exact=True) == frac(["1/2", "1/4", "1/4"])
        assert triangle(True).green(source, exact=True) == frac(["1/4", 0, 0])

    def test_three_vertex_path(self):
        path = RexGraph(boundary_ptr=np.asarray([0, 2, 4], np.int64),
                        boundary_idx=np.asarray([0, 1, 1, 2], np.int64))
        assert path.green(frac([1, 0]), exact=True) == frac(["3/8", "1/8"])

    def test_four_leaf_star(self):
        star = RexGraph(boundary_ptr=np.asarray([0, 2, 4, 6, 8], np.int64),
                        boundary_idx=np.asarray([0, 1, 0, 2, 0, 3, 0, 4], np.int64))
        assert star.green(frac([1, 0, 0, 0]), exact=True) == frac(
            ["5/12", "-1/12", "-1/12", "-1/12"])

    def test_multiplicity_is_a_real_harmonic_mode(self):
        """Two primary relations may share a boundary column. (1,-1) is then a nonzero
        field with zero boundary -- not a duplicate to collapse."""
        both = RexGraph(boundary_ptr=np.asarray([0, 2, 4], np.int64),
                        boundary_idx=np.asarray([0, 1, 0, 1], np.int64))
        assert both.green(frac([1, 0]), exact=True) == frac(["3/5", "-2/5"])
        harm = both.hodge(frac([1, 0]), exact=True)[2]
        assert harm == frac(["1/2", "-1/2"])

    @pytest.mark.parametrize("faces,expected", [
        ([[0, 1, 3], [0, 2, 4], [1, 2, 5]], [0, 0, 0, "3/10", "-1/10", "1/10"]),
        ([[0, 1, 3], [0, 2, 4], [1, 2, 5], [3, 4, 5]], [0, 0, 0, "1/5", 0, 0]),
    ])
    def test_tetrahedron_shell_changes_the_response_at_equal_betti(self, faces, expected):
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        rex = RexGraph(
            boundary_ptr=np.asarray([0] + [2 * (i + 1) for i in range(6)], np.int64),
            boundary_idx=np.asarray([v for e in edges for v in e], np.int64))
        rex.add_faces([list(f) for f in faces])
        source = [Q(0)] * 6
        source[3] = Q(1)
        assert rex.green(source, exact=True) == frac(expected)


class TestExactHodgeIsPrimary:
    def test_the_face_moves_the_cycle_from_harmonic_to_curl(self):
        source = frac([1, 0, 0])
        g_o, c_o, h_o = triangle().hodge(source, exact=True)
        g_f, c_f, h_f = triangle(True).hodge(source, exact=True)
        assert g_o == g_f == frac(["2/3", "-1/3", "-1/3"])
        assert h_o == frac(["1/3", "1/3", "1/3"]) and c_o == frac([0, 0, 0])
        assert c_f == frac(["1/3", "1/3", "1/3"]) and h_f == frac([0, 0, 0])

    def test_parts_reconstruct_the_field_exactly(self):
        field = frac([1, 2, 0])
        parts = triangle().hodge(field, exact=True)
        assert tuple(a + b + c for a, b, c in zip(*parts, strict=True)) == field

    def test_default_returns_floats_of_the_exact_values(self):
        rex = triangle()
        floats = rex.hodge(np.asarray([1.0, 2.0, 0.0]))
        exact = rex.hodge(frac([1, 2, 0]), exact=True)
        assert all(isinstance(part, np.ndarray) for part in floats)
        for got, want in zip(floats, exact, strict=True):
            assert np.allclose(got, [float(v) for v in want], atol=0, rtol=0)

    def test_green_preserves_the_harmonic_sector(self):
        """L_k annihilates it, so (I + lam L_k)^-1 leaves it alone."""
        rex = triangle()
        harm = rex.hodge(frac([1, 2, 0]), exact=True)[2]
        assert rex.green(harm, exact=True) == harm


class TestTheMetricIsNotOptional:
    """B_2^dagger = M_2^-1 B_2^* M_1. The float routine takes no metric at all, so
    under a nonidentity grade metric it answers the unweighted question instead."""

    def setup_method(self):
        self.rex = triangle(w_E=[Q(1), Q(2), Q(3)])
        self.field = frac([1, 2, 0])
        self.metric = [Q(1), Q(2), Q(3)]

    def test_exact_and_oracle_disagree_by_far_more_than_rounding(self):
        exact = self.rex.hodge(self.field, exact=True)
        oracle = self.rex.hodge(np.asarray([1.0, 2.0, 0.0]), exact=False)
        gap = max(abs(float(a) - b) for a, b in zip(exact[0], oracle[0], strict=True))
        assert gap > 0.1

    def test_sectors_are_orthogonal_in_the_declared_metric(self):
        grad, curl, harm = self.rex.hodge(self.field, exact=True)
        def pair(u, v):
            return sum(m * a * b for m, a, b in zip(self.metric, u, v, strict=True))

        assert pair(grad, harm) == 0 and pair(grad, curl) == 0
        assert sum(a * b for a, b in zip(grad, harm, strict=True)) != 0

    def test_parts_still_reconstruct(self):
        parts = self.rex.hodge(self.field, exact=True)
        assert tuple(a + b + c for a, b, c in zip(*parts, strict=True)) == self.field


class TestGreenOracle:
    """Above the exact ceiling the float tower produces. It is a real solve of the same
    operator, not the normal equations system the paper warns about."""

    def _oracle(self, rex, source):
        before = get_algorithm_config()["exact_field_limit"]
        try:
            configure_algorithms(exact_field_limit=0)
            return rex.green(source)
        finally:
            configure_algorithms(exact_field_limit=before)

    @pytest.mark.parametrize("kw,faces", [
        ({}, None),
        ({}, [[0, 1, 2]]),
        ({"w_E": [Q(1), Q(2), Q(3)]}, None),
        ({"w_E": [Q(1), Q(2), Q(3)]}, [[0, 1, 2]]),
    ])
    def test_it_agrees_with_the_exact_path_including_under_weighting(self, kw, faces):
        rex = RexGraph(boundary_ptr=TRIANGLE_PTR, boundary_idx=TRIANGLE_IDX, **kw)
        if faces:
            rex.add_faces(faces)
        source = frac([1, 0, 0])
        exact = rex.green(source, exact=True)
        oracle = self._oracle(rex, np.asarray([1.0, 0.0, 0.0]))
        assert np.allclose([float(v) for v in exact], oracle, atol=1e-12)

    def test_it_solves_the_operator_it_claims_to(self):
        from rexgraph.native_sparse import NativeSparse
        rng = np.random.default_rng(3)
        nE = 600
        src, tgt = rng.integers(0, 400, nE), rng.integers(0, 400, nE)
        keep = src != tgt
        src, tgt = src[keep], tgt[keep]
        nE = src.size
        idx = np.empty(2 * nE, dtype=np.int64)
        idx[0::2], idx[1::2] = src, tgt
        rex = RexGraph(boundary_ptr=np.arange(0, 2 * nE + 1, 2, dtype=np.int64),
                       boundary_idx=idx)
        j = rng.standard_normal(nE)
        g = self._oracle(rex, j)
        B1 = NativeSparse(rex._B1_dual)
        assert np.allclose(g + B1.transpose_apply(B1.apply(g)), j, atol=1e-9)

    def test_a_negative_lam_is_refused_rather_than_iterated(self):
        rex = triangle()
        before = get_algorithm_config()["exact_field_limit"]
        try:
            configure_algorithms(exact_field_limit=0)
            with pytest.raises(ValueError, match="non-negative"):
                rex.green(np.asarray([1.0, 0.0, 0.0]), -1)
        finally:
            configure_algorithms(exact_field_limit=before)


class TestExactPathPolicy:
    def test_the_ceiling_is_declared_and_movable(self):
        before = get_algorithm_config()["exact_field_limit"]
        try:
            configure_algorithms(exact_field_limit=0)
            assert get_algorithm_config()["exact_field_limit"] == 0
            rex = triangle()
            assert isinstance(rex.hodge(np.asarray([1.0, 0.0, 0.0]))[0], np.ndarray)
            assert np.allclose(rex.green(np.asarray([1.0, 0.0, 0.0])),
                               [0.5, 0.25, 0.25])
        finally:
            configure_algorithms(exact_field_limit=before)

    def test_exact_true_answers_above_the_ceiling_rather_than_refusing(self):
        before = get_algorithm_config()["exact_field_limit"]
        try:
            configure_algorithms(exact_field_limit=0)
            assert triangle().hodge(frac([1, 0, 0]), exact=True)[2] == frac(
                ["1/3", "1/3", "1/3"])
        finally:
            configure_algorithms(exact_field_limit=before)

    def test_a_negative_ceiling_is_refused(self):
        with pytest.raises(ValueError):
            configure_algorithms(exact_field_limit=-1)

    def test_a_float_field_enters_at_its_exact_binary_value(self):
        """The codebase convention for edge_metric_exact, applied to fields."""
        rex = triangle()
        assert rex.hodge([0.5, 0.25, 0.0], exact=True)[2] == frac(["1/4"] * 3)

    def test_a_non_finite_coordinate_is_refused(self):
        with pytest.raises(ValueError):
            triangle().hodge([float("inf"), 0.0, 0.0], exact=True)


class TestNonSpectralSolveContracts:
    """Section 12: no eigensolve is part of these algorithms."""

    def setup_method(self):
        B1 = np.array([[-1.0, 0, 1], [1, -1, 0], [0, 1, -1]])
        self.L = B1.T @ B1
        self.H = np.ones((3, 1))

    def test_harmonic_pinv_matches_the_spectral_pseudoinverse_it_replaces(self):
        evals, evecs = eigh(self.L)
        spectral = pinv_spectral(evals, evecs)
        for x in (np.array([1.0, 0, 0]), np.array([2.0, -3, 5]), np.ones(3)):
            got = harmonic_pinv_matvec(self.L, self.H, x)
            assert np.allclose(got, spectral @ x, atol=1e-10)

    def test_it_satisfies_the_pseudoinverse_identity(self):
        x = np.array([1.0, 0, 0])
        y = harmonic_pinv_matvec(self.L, self.H, x)
        assert np.allclose(self.L @ y, x - frame_project(self.H, x), atol=1e-10)

    def test_a_metric_self_adjoint_operator_is_solved_in_its_own_pairing(self):
        M = np.array([1.0, 2.0, 3.0])
        L = np.diag(1.0 / M) @ self.L          # B_1^dagger B_1, M_0 = I
        assert not np.allclose(L, L.T)         # not symmetric in coordinates
        assert np.allclose(np.diag(M) @ L, (np.diag(M) @ L).T)   # symmetric in M
        x = np.array([1.0, 0, 0])
        y = harmonic_pinv_matvec(L, self.H, x, metric=M)
        assert np.allclose(L @ y, x - frame_project(self.H, x, M), atol=1e-10)

    def test_no_harmonic_frame_is_a_plain_solve(self):
        A = np.array([[3.0, -1], [-1, 3]])
        b = np.array([1.0, 0])
        y, _, resid = metric_cg(A, b)
        assert np.allclose(y, np.array([3 / 8, 1 / 8]), atol=1e-12)
        assert resid < 1e-10

    def test_an_indefinite_operator_is_refused_not_iterated(self):
        with pytest.raises(ArithmeticError):
            metric_cg(np.array([[1.0, 0], [0, -1.0]]), np.array([1.0, 1.0]))


class TestFrameProjector:
    def setup_method(self):
        self.F = np.array([[1.0, 0], [1, 1], [0, 1]])
        self.M = np.array([1.0, 2.0, 3.0])

    def test_idempotent_and_metric_self_adjoint(self):
        project = frame_projector(self.F, self.M)
        u, v = np.array([2.0, -1, 5]), np.array([3.0, 1, 4])
        assert np.allclose(project(project(v)), project(v))
        assert np.isclose(project(u) @ (self.M * v), u @ (self.M * project(v)))

    def test_an_empty_frame_is_the_zero_projector(self):
        assert np.allclose(frame_project(np.zeros((3, 0)), np.ones(3)), 0.0)

    def test_a_redundant_frame_is_refused_rather_than_regularized(self):
        redundant = np.column_stack([self.F, self.F[:, 0] + self.F[:, 1]])
        with pytest.raises(ValueError, match="image frame"):
            frame_projector(redundant)

    def test_the_paper_worked_projector(self):
        """Example "Noncommuting subspace selections": M = diag(2,3), line (1,1)."""
        got = frame_project(np.array([1.0, 1.0]), np.array([1.0, 0.0]),
                            np.array([2.0, 3.0]))
        assert np.allclose(got, [2 / 5, 2 / 5])


class TestLeastQuadrance:
    def test_the_paper_square_fixture(self):
        """Example "Forest support and Hodge purity differ"."""
        B1 = np.array([[-1.0, 0, 0, 1], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
        p = np.array([1.0, 1, 0, 0])
        star = least_quadrance(p, np.ones(4))
        assert np.allclose(star, [0.5, 0.5, -0.5, -0.5])
        assert np.isclose(star @ star, 1.0)
        assert np.allclose(B1 @ star, B1 @ p)      # still feasible

    def test_an_empty_kernel_leaves_the_realization_alone(self):
        p = np.array([1.0, 2.0])
        assert np.allclose(least_quadrance(p, np.zeros((2, 0))), p)


class TestComponentProjector:
    """The kernel of a graph Laplacian is written down, not solved for."""

    def _laplacian(self, nV, seed):
        import scipy.sparse as sp
        rng = np.random.default_rng(seed)
        src, tgt = rng.integers(0, nV, nV * 3), rng.integers(0, nV, nV * 3)
        keep = src != tgt
        src, tgt = src[keep], tgt[keep]
        nE = src.size
        cols = np.arange(nE)
        B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)],
                            (np.r_[src, tgt], np.r_[cols, cols])), shape=(nV, nE))
        return (B1 @ B1.T).tocsr()

    def test_it_agrees_with_the_dense_indicator_frame(self):
        L0 = self._laplacian(300, 11)
        frame = _component_frame(L0)
        cheap, dense = _component_projector(L0), frame_projector(frame)
        x = np.random.default_rng(3).standard_normal(300)
        assert np.allclose(cheap(x), dense(x), atol=1e-12)

    def test_the_frame_really_spans_the_kernel(self):
        L0 = self._laplacian(300, 11)
        frame = _component_frame(L0)
        assert np.allclose(L0 @ frame, 0.0)
        assert np.linalg.matrix_rank(frame) == frame.shape[1]

    def test_the_pinv_identity_holds_on_a_disconnected_complex(self):
        from rexgraph.sparse_interfacing import _l0_pinv_matvec
        L0 = self._laplacian(300, 11)
        project = _component_projector(L0)
        b = np.random.default_rng(5).standard_normal(300)
        y = _l0_pinv_matvec(L0, b)
        assert np.allclose(L0 @ y, b - project(b), atol=1e-9)
        assert np.allclose(project(y), 0.0, atol=1e-9)   # deflated onto range(L0)


class TestProjectorConsolidation:
    """One projector, shared. `Pi_H = H (H* M H)^-1 H* M` had four implementations."""

    def test_the_prepared_projector_handles_a_block(self):
        rng = np.random.default_rng(4)
        F = rng.standard_normal((60, 7))
        project = frame_projector(F)
        block = rng.standard_normal((60, 5))
        got = project(block)
        for j in range(block.shape[1]):
            assert np.allclose(got[:, j], project(block[:, j]))

    def test_its_diagonal_matches_the_assembled_projector(self):
        rng = np.random.default_rng(9)
        F = rng.standard_normal((40, 6))
        M = rng.random(40) + 0.5
        project = frame_projector(F, M)
        dense = np.column_stack([project(np.eye(40)[:, i]) for i in range(40)])
        assert np.allclose(project.diagonal(), np.diag(dense))

    def test_the_deflated_green_diagonal_agrees_column_by_column(self):
        """greens_character_edge is diag(L1^+) via (L1 + P_H)^-1 - P_H, the same
        identity harmonic_pinv_matvec applies to a vector. On the open triangle
        diag(L1^+) = 2/9: L1 acts by 3 on the gradient plane and 0 on the cycle."""
        from rexgraph.core._linalg import harmonic_pinv_matvec
        from rexgraph.harmonic_sparse import harmonic_basis
        from rexgraph.scale_propagator import greens_diagonal_deflated
        rex = triangle()
        L1 = rex.L1_sparse.tocsr()
        H = np.asarray(harmonic_basis(rex).todense())
        assert np.allclose(L1 @ H, 0.0)              # the frame really is the kernel
        diag = greens_diagonal_deflated(L1, H)
        assert np.allclose(diag, 2 / 9)
        for i in range(len(diag)):
            column = harmonic_pinv_matvec(L1, H, np.eye(len(diag))[:, i])
            assert np.isclose(column[i], diag[i], atol=1e-8)

    def test_a_prepared_projector_does_not_write_into_its_own_frame(self):
        """LAPACK solves in place. A single row or single column right hand side is both
        C- and F-contiguous, so an unguarded `asfortranarray` hands back the caller's
        array and the solve overwrites the frame. Only the degenerate shapes show it."""
        for shape in ((3, 1), (1, 1), (6, 1), (4, 4)):
            F = np.ones(shape) if shape[1] == 1 else np.eye(shape[0])
            before = F.copy()
            project = frame_projector(F)
            project(np.ones(shape[0]))
            project.diagonal()
            assert np.array_equal(F, before), f"frame mutated at shape {shape}"


class TestTheOracleWillNotAnswerADifferentQuestion:
    def test_a_weighted_fallback_raises_rather_than_return_the_unweighted_split(self):
        rex = triangle(w_E=[Q(1), Q(2), Q(3)])
        before = get_algorithm_config()["exact_field_limit"]
        try:
            configure_algorithms(exact_field_limit=0)
            with pytest.raises(ValueError, match="unweighted decomposition"):
                rex.hodge(np.asarray([1.0, 2.0, 0.0]))
        finally:
            configure_algorithms(exact_field_limit=before)

    def test_an_explicit_oracle_request_is_still_honoured(self):
        """exact=False asks for the approximation tower by name, so it gets it."""
        rex = triangle(w_E=[Q(1), Q(2), Q(3)])
        parts = rex.hodge(np.asarray([1.0, 2.0, 0.0]), exact=False)
        assert np.allclose(parts[0], [0.0, 1.0, -1.0])

    def test_an_unweighted_fallback_is_fine(self):
        rex = triangle()
        before = get_algorithm_config()["exact_field_limit"]
        try:
            configure_algorithms(exact_field_limit=0)
            grad, _, harm = rex.hodge(np.asarray([1.0, 0.0, 0.0]))
            assert np.allclose(harm, 1 / 3) and np.allclose(grad, [2 / 3, -1 / 3, -1 / 3])
        finally:
            configure_algorithms(exact_field_limit=before)


class TestProjectorSubstitutability:
    """A cheap structured projector and the dense frame must be interchangeable."""

    def test_the_component_projector_carries_the_same_contract(self):
        import scipy.sparse as sp
        rng = np.random.default_rng(11)
        nV = 300
        src, tgt = rng.integers(0, nV, nV * 3), rng.integers(0, nV, nV * 3)
        keep = src != tgt
        src, tgt = src[keep], tgt[keep]
        nE = src.size
        cols = np.arange(nE)
        B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)],
                            (np.r_[src, tgt], np.r_[cols, cols])), shape=(nV, nE))
        L0 = (B1 @ B1.T).tocsr()
        cheap = _component_projector(L0)
        dense = frame_projector(_component_frame(L0))
        x = rng.standard_normal(nV)
        block = rng.standard_normal((nV, 4))
        assert np.allclose(cheap(x), dense(x))
        assert np.allclose(cheap(block), dense(block))
        assert np.allclose(cheap.diagonal(), dense.diagonal())

    def test_an_empty_frame_still_carries_a_diagonal(self):
        empty = frame_projector(np.zeros((5, 0)))
        assert np.allclose(empty(np.ones(5)), 0.0)
        assert np.allclose(empty.diagonal(), 0.0)
