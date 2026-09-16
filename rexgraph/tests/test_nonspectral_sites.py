"""Sites that used to need a full spectrum or SVD to answer a solve.

Each test pins the converted path against the spectral one it replaced. A failure here
means the two disagree, which is a wrong answer and not a slower one.
"""
import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.core import _character, _query
from rexgraph.core._linalg import (
    eigh,
    frame_projector,
    pinv_spectral,
    spd_inverse_diagonal,
    spd_solve,
)


def pairwise(nV, nE, seed):
    rng = np.random.default_rng(seed)
    src, tgt = rng.integers(0, nV, nE), rng.integers(0, nV, nE)
    keep = src != tgt
    src, tgt = src[keep], tgt[keep]
    m = src.size
    idx = np.empty(2 * m, dtype=np.int64)
    idx[0::2], idx[1::2] = src, tgt
    return RexGraph(boundary_ptr=np.arange(0, 2 * m + 1, 2, dtype=np.int64),
                    boundary_idx=idx)


def branching(nV, nE, seed):
    rng = np.random.default_rng(seed)
    ptr, idx = [0], []
    for _ in range(nE):
        k = int(rng.integers(2, 7))
        idx += list(rng.choice(nV, size=k, replace=False))
        ptr.append(len(idx))
    return RexGraph(boundary_ptr=np.asarray(ptr, dtype=np.int64),
                    boundary_idx=np.asarray(idx, dtype=np.int64))


class TestSpdSolve:
    def setup_method(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((50, 50))
        self.A = A @ A.T + 50 * np.eye(50)
        self.b = rng.standard_normal(50)

    def test_it_matches_the_spectral_pseudoinverse(self):
        evals, evecs = eigh(self.A)
        assert np.allclose(spd_solve(self.A, self.b), pinv_spectral(evals, evecs) @ self.b)

    def test_the_inverse_diagonal_matches_too(self):
        evals, evecs = eigh(self.A)
        assert np.allclose(spd_inverse_diagonal(self.A),
                           np.diag(pinv_spectral(evals, evecs)))

    def test_an_indefinite_matrix_declines_rather_than_answers(self):
        assert spd_solve(np.diag([1.0, 2.0, -3.0]), np.ones(3)) is None
        assert spd_inverse_diagonal(np.diag([1.0, -1.0])) is None

    def test_strict_raises_instead(self):
        with pytest.raises(ArithmeticError):
            spd_solve(np.diag([1.0, -1.0]), np.ones(2), strict=True)

    def test_an_empty_system_is_an_empty_answer(self):
        assert spd_solve(np.zeros((0, 0)), np.zeros(0)).shape == (0,)

    def test_a_non_symmetric_matrix_raises_rather_than_solving_something_else(self):
        """`dpotrf_` reads one triangle. Symmetry is structural, so this is a caller
        error and is tested exactly -- not inferred from a residual tolerance."""
        broken = self.A.copy()
        broken[0, 5] += 7.0                     # upper triangle only
        with pytest.raises(ValueError, match="symmetric"):
            spd_solve(broken, self.b)

    def test_a_symmetric_solve_is_accurate(self):
        solved = spd_solve(self.A, self.b)
        assert solved is not None
        assert np.abs(self.A @ solved - self.b).max() < 1e-10


class TestRelationalLaplacianIsPositiveDefinite:
    """The premise every conversion in `_query` and `_character` rests on. If this ever
    fails, `RL^+ = RL^-1` stops holding and those sites need their spectra back."""

    @pytest.mark.parametrize("rex", [pairwise(40, 90, 1), pairwise(30, 120, 2),
                                     branching(40, 60, 3)])
    def test_rl_is_symmetric_and_positive_definite(self, rex):
        RL = np.asarray(rex.RL, dtype=np.float64)
        assert np.allclose(RL, RL.T)
        assert np.linalg.eigvalsh(RL).min() > 0.0
        assert spd_solve(RL, np.ones(RL.shape[0])) is not None


class TestQuerySites:
    def setup_method(self):
        self.rex = pairwise(60, 150, 1)
        self.nE = int(self.rex.nE)
        self.RL = self.rex.RL
        self.rcf = self.rex._rcf_bundle
        evals, evecs = eigh(self.RL)
        self.RLp = pinv_spectral(evals, evecs)
        rng = np.random.default_rng(6)
        self.source, self.target = rng.standard_normal(self.nE), rng.standard_normal(self.nE)

    def test_spectral_propagate_matches_the_spectral_score(self):
        got = _query.spectral_propagate(self.RL, self.rcf['hats'], self.rcf['nhats'],
                                        self.source, self.target, self.nE)
        want = float(self.RLp @ self.source @ self.target) / (
            np.linalg.norm(self.source) * np.linalg.norm(self.target))
        assert np.isclose(got['score'], want, rtol=0, atol=1e-11)
        assert got['coverage'] == 1.0          # positive definite: no null space

    def test_explain_edge_effective_resistance_matches(self):
        got = _query.explain_edge(self.rex.B1, np.zeros((self.nE, 0)),
                                  np.zeros((self.nE, self.nE)), self.RL,
                                  self.rcf['hats'], self.rcf['nhats'],
                                  3, int(self.rex.nV), self.nE, 0)
        assert np.isclose(got['effective_resistance'], self.RLp[3, 3], atol=1e-11)

    def test_effective_resistance_is_finite_above_the_dense_eigen_gate(self):
        """It returned nan there before: the dense pseudoinverse was unaffordable, but
        one diagonal entry is one solve and never needed it."""
        big = pairwise(1200, 3000, 2)
        nE = int(big.nE)
        rcf = big._rcf_bundle
        got = _query.explain_edge(big.B1, np.zeros((nE, 0)), np.zeros((nE, nE)), big.RL,
                                  rcf['hats'], rcf['nhats'], 5, int(big.nV), nE, 0)
        assert np.isfinite(got['effective_resistance'])
        assert got['effective_resistance'] > 0.0

    def test_signal_impute_matches_the_spectral_imputation(self):
        rng = np.random.default_rng(11)
        mask = np.ones(self.nE, dtype=bool)
        mask[rng.choice(self.nE, 20, replace=False)] = False
        signal = rng.standard_normal(self.nE) * mask
        got = _query.signal_impute(self.RL, signal, mask.astype(np.uint8), self.nE)
        mis, obs = np.where(~mask)[0], np.where(mask)[0]
        RL_d = np.asarray(self.RL, dtype=np.float64)
        want = -np.linalg.pinv(RL_d[np.ix_(mis, mis)]) @ RL_d[np.ix_(mis, obs)] @ signal[obs]
        assert np.allclose(got['imputed'][mis], want, atol=1e-10)


class TestVertexCharacterBlockSolve:
    def test_the_block_solve_matches_the_per_vertex_loop(self):
        """One factorization and nV right hand sides, against nV factorizations of the
        same system."""
        rex = pairwise(30, 70, 4)
        nV, nE = int(rex.nV), int(rex.nE)
        rcf = rex._rcf_bundle
        block = _character.compute_phi(rex.B1, rex.RL, rcf['hats'], rcf['nhats'],
                                       nV, nE, None)
        loop = np.zeros((nV, rcf['nhats']), dtype=np.float64)
        for v in range(nV):
            loop[v, :] = _character.compute_phi_sparse_single(
                rex.RL, rcf['hats'], rcf['nhats'], rex.B1, v, nV, nE)
        assert np.allclose(block, loop, atol=1e-9)


class TestQuotientSpanResidual:
    def test_the_projector_residual_matches_least_squares(self):
        from rexgraph.core._linalg import lstsq
        rng = np.random.default_rng(7)
        basis = rng.standard_normal((40, 6))
        d = rng.standard_normal(40)
        sol, _rank = lstsq(np.ascontiguousarray(basis), np.ascontiguousarray(d))
        want = float(np.linalg.norm(d - basis @ sol))
        got = float(np.linalg.norm(d - frame_projector(basis)(d)))
        assert np.isclose(got, want, atol=1e-10)

    def test_congruence_is_unchanged_on_a_real_complex(self):
        from rexgraph.core._quotient import congruent_edges
        rex = pairwise(20, 45, 8)
        B1 = np.ascontiguousarray(rex.B1)
        mask = np.zeros(int(rex.nE), dtype=np.uint8)
        mask[:8] = 1
        same, residual = congruent_edges(9, 10, B1, mask)
        assert isinstance(same, (bool, np.bool_)) and residual >= 0.0


class TestCommonUnionFindKernel:
    """`_common.pxd` has carried a path compressed union find all along. Component
    labelling was being derived again in Python instead of using it."""

    def _laplacian(self, nV, seed, isolated=0):
        import scipy.sparse as sp
        rng = np.random.default_rng(seed)
        live = nV - isolated
        src, tgt = rng.integers(0, live, live * 3), rng.integers(0, live, live * 3)
        keep = src != tgt
        src, tgt = src[keep], tgt[keep]
        nE = src.size
        cols = np.arange(nE)
        B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)],
                            (np.r_[src, tgt], np.r_[cols, cols])), shape=(nV, nE))
        return (B1 @ B1.T).tocsr()

    def test_it_agrees_with_a_reference_traversal(self):
        from rexgraph.core._sparse import connected_components
        L0 = self._laplacian(400, 3)
        labels, count = connected_components(L0.indptr, L0.indices, L0.shape[0])
        indptr, indices = np.asarray(L0.indptr), np.asarray(L0.indices)
        ref = np.full(L0.shape[0], -1, dtype=np.int64)
        seen = 0
        for start in range(L0.shape[0]):
            if ref[start] >= 0:
                continue
            stack = [start]
            ref[start] = seen
            while stack:
                v = stack.pop()
                for k in range(indptr[v], indptr[v + 1]):
                    w = int(indices[k])
                    if ref[w] < 0:
                        ref[w] = seen
                        stack.append(w)
            seen += 1
        assert count == seen
        # Same partition, independent of the label numbering.
        assert len(set(zip(labels.tolist(), ref.tolist(), strict=True))) == count

    def test_labels_are_dense_from_zero(self):
        """A frame or a bincount indexes by label, so they must not have gaps."""
        from rexgraph.core._sparse import connected_components
        L0 = self._laplacian(300, 5)
        labels, count = connected_components(L0.indptr, L0.indices, L0.shape[0])
        assert set(labels.tolist()) == set(range(count))

    def test_isolated_vertices_are_their_own_components(self):
        from rexgraph.core._sparse import connected_components
        L0 = self._laplacian(300, 7, isolated=40)
        labels, count = connected_components(L0.indptr, L0.indices, L0.shape[0])
        assert count >= 40
        assert len(set(labels[-40:].tolist())) == 40

    def test_an_empty_pattern_is_answered_not_raised(self):
        from rexgraph.core._sparse import connected_components
        labels, count = connected_components(np.zeros(1, dtype=np.int64),
                                             np.zeros(0, dtype=np.int64), 0)
        assert labels.shape == (0,) and count == 0

    def test_the_frame_it_produces_really_is_the_kernel(self):
        from rexgraph.sparse_interfacing import _component_frame
        L0 = self._laplacian(400, 9)
        frame = _component_frame(L0)
        assert np.abs(L0 @ frame).max() == 0.0          # exactly, not to a tolerance
        assert np.linalg.matrix_rank(frame) == frame.shape[1]

    def test_the_cheap_projector_still_matches_the_dense_frame(self):
        from rexgraph.sparse_interfacing import _component_frame, _component_projector
        L0 = self._laplacian(400, 9)
        cheap, dense = _component_projector(L0), frame_projector(_component_frame(L0))
        x = np.random.default_rng(1).standard_normal(400)
        assert np.allclose(cheap(x), dense(x), atol=1e-12)
        assert np.allclose(cheap.diagonal(), dense.diagonal(), atol=1e-12)


class TestCommonIsExactWhereItsInputsAre:
    """`_common`'s kernels take integer supports. Where the answer is a ratio of two
    counts it is a rational, and forcing it through a double loses it at the boundary."""

    def test_support_jaccard_is_an_exact_fraction(self):
        from fractions import Fraction

        from rexgraph.core._sparse import support_jaccard
        assert support_jaccard([0, 1, 2, 5], [1, 2, 3]) == Fraction(2, 5)
        assert support_jaccard([0, 1], [2, 3]) == Fraction(0)
        assert support_jaccard([3, 1], [1, 3]) == Fraction(1)
        assert support_jaccard([], []) == Fraction(0)

    def test_it_treats_its_inputs_as_sets(self):
        from fractions import Fraction

        from rexgraph.core._sparse import support_jaccard
        assert support_jaccard([5, 2, 1, 0, 2], [3, 1, 2]) == Fraction(2, 5)

    def test_the_float_reading_agrees_with_the_exact_one(self):
        from rexgraph.core._sparse import support_jaccard
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.choice(40, size=int(rng.integers(1, 20)), replace=False)
            b = rng.choice(40, size=int(rng.integers(1, 20)), replace=False)
            assert np.isclose(float(support_jaccard(a, b)),
                              support_jaccard(a, b, exact=False))

    def test_a_third_is_exact_not_the_nearest_double(self):
        from fractions import Fraction

        from rexgraph.core._sparse import support_jaccard
        got = support_jaccard([0], [0, 1, 2])          # 1 shared of 3 total
        assert got == Fraction(1, 3)
        assert got * 3 == 1                            # exact; the double does not


class TestSupportIsStructuralNotNoise:
    """The nonzero pattern of a boundary tensor IS its arity, degree and local parity,
    so a conversion must not delete a small coefficient."""

    def test_a_small_weight_survives_the_dense_round_trip(self):
        from rexgraph.core._sparse import from_dense_f64, to_scipy_csr
        D = np.array([[-1.0, 0.0],
                      [1.0, -1e-11],
                      [0.0, 1e-11]])
        out = to_scipy_csr(from_dense_f64(D)).toarray()
        assert np.count_nonzero(out[:, 1]) == 2        # the relation still participates
        assert np.allclose(out, D, rtol=0, atol=0)

    def test_exact_zeros_are_still_dropped(self):
        from rexgraph.core._sparse import from_dense_f64
        D = np.array([[1.0, 0.0], [0.0, 2.0]])
        assert from_dense_f64(D).nnz == 2

    def test_an_explicit_threshold_is_still_honoured(self):
        """`_boundary` converts a binary incidence with tol=0.5 -- that use is real."""
        from rexgraph.core._sparse import from_dense_f64
        D = np.array([[1.0, 0.3], [0.2, 1.0]])
        assert from_dense_f64(D, tol=0.5).nnz == 2


class TestNoArbitraryThresholds:
    """The exact path takes no tolerance, and the float paths that replaced a spectrum
    do not invent one either. Where a test is structural it is tested exactly."""

    def test_rl_is_positive_definite_on_every_degenerate_shape(self):
        """The premise `_query` and `_character` now assert outright. If any of these
        stops holding, those sites need their fallbacks back."""
        from fractions import Fraction

        shapes = {
            "single relation": ([0, 2], [0, 1], {}),
            "parallel relations": ([0, 2, 4], [0, 1, 0, 1], {}),
            "arity-1 witness": ([0, 1], [0], {}),
            "arity-12 relation": ([0, 12], list(range(12)), {}),
            "zero weight": ([0, 2, 4], [0, 1, 1, 2], {"w_E": [Fraction(0), Fraction(1)]}),
            "weight 1e-12": ([0, 2, 4], [0, 1, 1, 2],
                             {"w_E": [Fraction(1, 10 ** 12), Fraction(1)]}),
            "weight 1e12": ([0, 2, 4], [0, 1, 1, 2],
                            {"w_E": [Fraction(10 ** 12), Fraction(1)]}),
            "disconnected": ([0, 2, 4], [0, 1, 2, 3], {}),
        }
        for label, (ptr, idx, kw) in shapes.items():
            rex = RexGraph(boundary_ptr=np.asarray(ptr, dtype=np.int64),
                           boundary_idx=np.asarray(idx, dtype=np.int64), **kw)
            RL = np.asarray(rex.RL, dtype=np.float64)
            assert np.linalg.eigvalsh(RL).min() > 0.0, label
            assert spd_solve(RL, np.ones(RL.shape[0])) is not None, label

    def test_rl_is_exactly_symmetric(self):
        """Which is why spd_solve can test symmetry with == and not a tolerance."""
        rng = np.random.default_rng(3)
        for nV, nE in ((60, 150), (200, 500)):
            rex = pairwise(nV, nE, int(rng.integers(1, 99)))
            RL = np.asarray(rex.RL, dtype=np.float64)
            assert np.array_equal(RL, RL.T)

    def test_the_laplacian_component_frame_is_exactly_in_the_kernel(self):
        """So the harmonic complement route needs no tolerance to verify its frame."""
        import scipy.sparse as sp

        from rexgraph.sparse_interfacing import _component_frame
        rng = np.random.default_rng(5)
        nV = 400
        src, tgt = rng.integers(0, nV, nV * 3), rng.integers(0, nV, nV * 3)
        keep = src != tgt
        src, tgt = src[keep], tgt[keep]
        nE = src.size
        cols = np.arange(nE)
        B1 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)],
                            (np.r_[src, tgt], np.r_[cols, cols])), shape=(nV, nE))
        L0 = (B1 @ B1.T).tocsr()
        assert np.abs(L0 @ _component_frame(L0)).max() == 0.0

    def test_spectral_propagate_reports_full_coverage(self):
        """Positive definite means no null space, so coverage is one by construction
        rather than by counting modes above a cutoff."""
        rex = pairwise(40, 100, 2)
        nE = int(rex.nE)
        rcf = rex._rcf_bundle
        rng = np.random.default_rng(8)
        got = _query.spectral_propagate(rex.RL, rcf['hats'], rcf['nhats'],
                                        rng.standard_normal(nE),
                                        rng.standard_normal(nE), nE)
        assert got['coverage'] == 1.0
