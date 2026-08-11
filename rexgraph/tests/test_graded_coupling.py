"""The between-grade coupling, L_gb as a spread, and what arity does to cycles.

The coupling between adjacent grades is the one rank they share, L_gb is that same
spread taken on spectra instead of ranks, and a branching relation carries no cycle
until its pairwise measurements are carried with it.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from rexgraph.core._l_gb import l_gb_scalar
from rexgraph.faces import auto_hyperface
from rexgraph.fiedler import kernel_basis
from rexgraph.graph import RexGraph
from rexgraph.sparse_character import _block_cg


def _g(s, t):
    return RexGraph(sources=np.asarray(s, dtype=np.int32),
                    targets=np.asarray(t, dtype=np.int32))


def _cycle(n):
    return _g(range(n), np.roll(np.arange(n), -1))


def _complete(n):
    s, t = zip(*[(i, j) for i in range(n) for j in range(i + 1, n)], strict=True)
    return _g(s, t)


def _hyper(groups):
    ptr, idx = [0], []
    for grp in groups:
        idx.extend(grp)
        ptr.append(len(idx))
    return RexGraph.from_hypergraph(np.array(ptr, np.int32), np.array(idx, np.int32))


def _mixed(groups):
    """Both grades: each group as a branching relation AND its pairs as 2-ary ones."""
    rels = [list(g) for g in groups]
    for g in groups:
        rels.extend([a, b] for a, b in itertools.combinations(sorted(g), 2))
    return _hyper(rels)


def _coupling(n_k, n_k1, rank_k1):
    return Fraction(1) - Fraction(rank_k1 * rank_k1, n_k * n_k1)


def _reff_of(Bk):
    """R_eff for each column of any boundary operator, via the regularized inverse."""
    Bk = sp.csc_matrix(Bk)
    if Bk.shape[1] == 0:
        return np.zeros(0)
    L = (Bk @ Bk.T).tocsr()
    U, _ = kernel_basis(L)
    Bc = np.ascontiguousarray(np.asarray(Bk.todense()))
    d = np.asarray(L.diagonal(), float) + (U * U).sum(axis=1)
    dinv = np.where(d > 1e-30, 1.0 / d, 1.0)
    X = _block_cg(lambda P: L @ P + U @ (U.T @ P), Bc, dinv, tol=1e-12, maxit=500)
    return np.einsum("ve,ve->e", Bc, X)


# --- Theorem 18: L_gb is a spread ------------------------------------------------

def test_l_gb_is_the_spread_of_the_two_spectra():
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = np.abs(rng.standard_normal(int(rng.integers(1, 10))))
        b = np.abs(rng.standard_normal(int(rng.integers(1, 10))))
        n = max(a.size, b.size)
        A, B = np.pad(a, (0, n - a.size)), np.pad(b, (0, n - b.size))
        spread = 1.0 - float(A @ B) ** 2 / (float(A @ A) * float(B @ B))
        assert l_gb_scalar(a, b) == pytest.approx(np.sqrt(2.0 * spread), abs=1e-6)


# --- the graded coupling ---------------------------------------------------------

def test_the_coupling_is_an_exact_rational_in_three_integers():
    """Cycles read (2n-1)/n^2 and paths 1/n, off the rank tower alone."""
    for n in (4, 5, 6, 7, 8):
        rex = _cycle(n)
        rex._ensure_clean()
        rank1 = int(rex.nV) - int(rex.betti[0])
        assert _coupling(int(rex.nV), int(rex.nE), rank1) == Fraction(2 * n - 1, n * n)
    for n in (4, 5, 6, 7):
        rex = _g(range(n - 1), range(1, n))
        rex._ensure_clean()
        rank1 = int(rex.nV) - int(rex.betti[0])
        assert _coupling(int(rex.nV), int(rex.nE), rank1) == Fraction(1, n)


def test_the_coupling_tracks_the_filling_one_grade_up():
    """A tetrahedron reads 5/6 at one face, 2/3 at two, 5/8 at four."""
    s, t = [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]
    faces = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]]
    seen = []
    for k, want in ((1, Fraction(5, 6)), (2, Fraction(2, 3)), (4, Fraction(5, 8))):
        rex = _g(s, t)
        rex.add_faces(np.array(faces[:k], np.int32))
        rex._ensure_clean()
        B2 = np.asarray(rex.B2_hodge)
        rank2 = int(np.linalg.matrix_rank(B2))
        got = _coupling(int(rex.nE), int(rex.nF_hodge), rank2)
        assert got == want
        seen.append(got)
    assert seen[0] > seen[1] > seen[2]                  # monotone as it seals


def test_the_seal_is_where_beta_2_appears():
    s, t = [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]
    faces = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]]
    partial = _g(s, t)
    partial.add_faces(np.array(faces[:3], np.int32))
    assert int(partial.betti[2]) == 0
    full = _g(s, t)
    full.add_faces(np.array(faces, np.int32))
    assert int(full.betti[2]) == 1


# --- the identity is grade-general -----------------------------------------------

def test_the_rank_identity_holds_at_grade_two():
    """sum R_eff_k = rank(B_k) with the grade's own boundary operator."""
    s, t = [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]
    rex = _g(s, t)
    rex.add_faces(np.array([[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]], np.int32))
    rex._ensure_clean()
    for B in (np.asarray(rex.B1), np.asarray(rex.B2_hodge)):
        r = _reff_of(B)
        rank = int(np.linalg.matrix_rank(B))
        assert float(r.sum()) == pytest.approx(rank, abs=1e-9)
        assert float((1.0 - r).sum()) == pytest.approx(B.shape[1] - rank, abs=1e-9)


# --- Theorem 19 and the mixed construction ---------------------------------------

@pytest.mark.parametrize("groups", [
    [[0, 1, 2]],
    [[0, 1, 2], [2, 3, 4]],
    [[0, 1, 2], [2, 3, 4], [4, 5, 0]],          # looks like a triangle, is not
    [[0, 1, 2, 3]],
    [[0, 1, 2, 3, 4]],
])
def test_wide_relations_do_not_close_on_a_shared_vertex(groups):
    """Sharing a vertex is not enough to make branching columns dependent. It is NOT
    true that wide relations never carry cycles: see the overlap test below."""
    rex = _hyper(groups)
    rex._ensure_clean()
    assert int(rex.betti[1]) == 0


def test_enough_overlap_does_make_wide_relations_dependent():
    """The correction: real protein complexes share subunits and 260 of them carry 52
    k-ary cycles with no pairwise relation present. Reproduced here in miniature."""
    # every 3-subset of five proteins: far more relations than the rank can support
    groups = list(itertools.combinations(range(5), 3))
    rex = _hyper([list(g) for g in groups])
    rex._ensure_clean()
    rank = int(np.linalg.matrix_rank(np.asarray(rex.B1)))
    assert int(rex.nE) > rank                       # dependent columns
    assert int(rex.betti[1]) == int(rex.nE) - rank
    assert int(rex.betti[1]) > 0                    # cycles, from wide relations alone


def test_the_boundary_column_is_zero_sum_at_every_arity():
    rex = _hyper([[0, 1, 2], [2, 3, 4, 5], [0, 5, 6, 7, 8]])
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    assert np.max(np.abs(B1.sum(axis=0))) < 1e-12


def test_beta0_is_the_rank_convention_not_the_component_count():
    """Two disjoint 3-ary relations: two components, but beta_0 = n0 - rank = 4."""
    rex = _hyper([[0, 1, 2], [3, 4, 5]])
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    assert int(rex.betti[0]) == int(rex.nV) - int(np.linalg.matrix_rank(B1))
    assert int(rex.betti[0]) == 4


def test_the_mixed_construction_is_what_creates_the_class():
    """Wide alone has nothing; carried with its pairs, one class survives filling."""
    groups = [[0, 1, 2], [2, 3, 4], [4, 5, 0]]
    assert int(_hyper(groups).betti[1]) == 0
    rex = _mixed(groups)
    rex._ensure_clean()
    assert int(rex.betti[1]) > 0
    assert auto_hyperface(rex) > 0
    assert int(rex.betti[1]) == 1                       # the loop of the wide relations


def test_two_wide_relations_sharing_an_edge_enclose_a_void():
    """A branching relation bounds a face, and pairwise enumeration finds neither."""
    rex = _mixed([[0, 1, 2, 3], [2, 3, 4, 5]])
    rex._ensure_clean()
    assert auto_hyperface(rex) > 0
    assert int(rex.betti[2]) == 1


# --- section 6j: what a circle is ------------------------------------------------

def _lone(*groups):
    rex = _hyper([list(g) for g in groups])
    rex._ensure_clean()
    return rex


def test_a_self_loop_is_a_cycle_on_its_own():
    """Its column is identically zero, so it is in ker(B1) by being zero rather than
    by cancelling against anything."""
    rex = _lone([0, 0])
    B1 = np.asarray(rex.B1)
    assert B1.shape[1] == 1
    assert np.count_nonzero(B1) == 0
    assert int(rex.betti[1]) == 1
    assert int(np.linalg.matrix_rank(B1)) == 0


def test_the_loop_and_the_rectangle_agree_on_homology_and_not_on_geometry():
    loop = _lone([0, 0])
    rect = _g([0, 1, 2, 3], [1, 2, 3, 0])
    rect._ensure_clean()
    assert int(loop.betti[1]) == int(rect.betti[1]) == 1
    assert np.allclose(_reff_of(np.asarray(loop.B1)), 0.0)
    assert np.allclose(_reff_of(np.asarray(rect.B1)), 0.75)


def test_nothing_fills_a_self_loop():
    """A face bounds a cycle VECTOR, and there is none here to bound."""
    rex = _lone([0, 0])
    assert auto_hyperface(rex) == 0
    assert int(rex.betti[1]) == 1


@pytest.mark.parametrize("groups,want", [
    (([0],), 1.0),                 # witness: the only non-zero-sum column
    (([0, 0],), 0.0),              # self-loop: carries none of the boundary
    (([0, 1],), 1.0),              # edge
    (([0, 1, 2],), 1.0),           # branching
])
def test_the_arity_spectrum(groups, want):
    rex = _lone(*groups)
    got = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    assert np.allclose(got, want, atol=1e-9)


def test_a_loop_beside_an_edge_separates_circulation_from_boundary():
    rex = _lone([0, 0], [0, 1])
    got = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    assert np.allclose(np.sort(got), [0.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("groups", [
    (([0],)),
    (([0], [0, 1])),
    (([0], [1])),
    (([0], [1, 2], [2, 3], [3, 1])),
])
def test_a_witness_breaks_the_laplacian_premise(groups):
    """A witness column does not sum to zero, so L0 is not a Laplacian and the
    component indicators are not its kernel. R_eff must still match the pseudoinverse."""
    rex = _lone(*groups)
    B1 = np.asarray(rex.B1)
    L0 = B1 @ B1.T
    truth = np.einsum("ve,vw,we->e", B1, np.linalg.pinv(L0), B1)
    got = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    assert np.allclose(got, truth, atol=1e-9)
    kept, _n = kernel_basis(sp.csr_matrix(L0))
    if kept.shape[1]:
        assert np.allclose(L0 @ kept, 0.0, atol=1e-9)      # only genuine kernel vectors


def test_arity_and_degree_are_opposite_axes():
    rex = _lone([0, 1, 2], [0, 3], [0, 4])
    B1 = np.asarray(rex.B1)
    arity = (B1 != 0).sum(axis=0)
    degree = (B1 != 0).sum(axis=1)
    assert arity.tolist() == [3, 2, 2]
    assert degree.tolist() == [3, 1, 1, 1, 1]
    assert len(arity) == int(rex.nE) and len(degree) == int(rex.nV)


# --- sections 6k and 6l ----------------------------------------------------------

def _leverage(M):
    """diag of the projector onto row(M): Theorem 21."""
    M = np.asarray(M, dtype=float)
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros(0)
    _u, sv, vt = np.linalg.svd(M, full_matrices=False)
    keep = sv > max(M.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    V = vt[keep]
    return np.einsum("ie,ie->e", V, V)


def _tetra(nfaces):
    rex = _g([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
    faces = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]][:nfaces]
    rex.add_faces(np.array(faces, np.int32))
    rex._ensure_clean()
    return rex


def test_the_binary_reading_exists_above_grade_one():
    """Corollary 21.2: the walk is grade-1 only, the READING is not."""
    full = _leverage(np.asarray(_tetra(4).B2_hodge))
    assert np.allclose(full, 0.75)
    assert int((full > 1 - 1e-9).sum()) == 0            # every face corroborated
    lone = _leverage(np.asarray(_tetra(1).B2_hodge))
    assert np.allclose(lone, 1.0)                       # the only face is load-bearing


def test_the_leverage_form_gives_the_rank_at_grade_two():
    rex = _tetra(4)
    B2 = np.asarray(rex.B2_hodge)
    assert float(_leverage(B2).sum()) == pytest.approx(
        int(np.linalg.matrix_rank(B2)), abs=1e-9)


def test_phi_comes_off_one_decomposition():
    """Theorem 22: no per-vertex solve."""
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    rex._ensure_clean()
    ch = rex._sparse_character
    RL = np.asarray(ch["RL"].todense())
    hats = [np.asarray(h.todense()) for h in ch["hats"]]
    B1 = np.asarray(rex.B1)
    RLp = np.linalg.pinv(RL)
    ref = []
    for v in range(int(rex.nV)):
        b = B1[v, :]
        den = float(b @ RLp @ b)
        ref.append([float(b @ RLp @ h @ RLp @ b) / den if den > 1e-12 else 0.0
                    for h in hats])
    ref = np.asarray(ref)
    w, U = np.linalg.eigh(RL)
    keep = w > max(RL.shape) * np.finfo(float).eps * max(abs(w).max(), 1.0)
    Uk, wk = U[:, keep], w[keep]
    Y = (B1 @ Uk) / wk
    den = np.einsum("vi,vi->v", Y, B1 @ Uk)
    got = np.stack([np.einsum("vi,ij,vj->v", Y, Uk.T @ h @ Uk, Y) for h in hats], axis=1)
    got = np.where(den[:, None] > 1e-12, got / np.maximum(den[:, None], 1e-300), 0.0)
    assert np.allclose(got, ref, atol=1e-9)
    assert np.allclose(got.sum(axis=1), 1.0)


def test_the_leverage_partitions_the_rank():
    """Theorem 23 on a mixed partition: sections sum to the whole, each under its own
    rank, and the section ranks over-count by exactly the overlap."""
    groups = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6]]
    rex = _mixed(groups)
    rex._ensure_clean()
    B1 = sp.csc_matrix(np.asarray(rex.B1))
    reff = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    arity = np.diff(B1.indptr)
    rank_all = int(np.linalg.matrix_rank(np.asarray(B1.todense())))
    total, rank_sum = 0.0, 0
    for k in sorted(set(arity.tolist())):
        cols = np.flatnonzero(arity == k)
        mass = float(reff[cols].sum())
        rk = int(np.linalg.matrix_rank(np.asarray(B1[:, cols].todense())))
        assert mass <= rk + 1e-9                       # a section cannot exceed its span
        total += mass
        rank_sum += rk
    assert total == pytest.approx(rank_all, abs=1e-9)  # sections sum to the whole
    assert rank_sum >= rank_all                        # subadditive, deficit = overlap


def test_a_subsets_own_cycles_bound_its_global_share():
    """Theorem 24, tight over the whole set."""
    rex = _g([0, 1, 2, 0], [1, 2, 0, 2])
    rex._ensure_clean()
    B1 = sp.csc_matrix(np.asarray(rex.B1))
    reff = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = int(rng.integers(1, int(rex.nE) + 1))
        S = rng.choice(int(rex.nE), k, replace=False)
        own = k - int(np.linalg.matrix_rank(np.asarray(B1[:, S].todense())))
        share = float((1.0 - reff[S]).sum())
        assert own <= share + 1e-9
    whole = np.arange(int(rex.nE))
    own = int(rex.nE) - int(np.linalg.matrix_rank(np.asarray(B1.todense())))
    assert float((1.0 - reff[whole]).sum()) == pytest.approx(own, abs=1e-9)
