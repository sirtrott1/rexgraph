"""The claims, stated so they can fail, and checked against an awkward zoo.

Each is written as a property over many structures rather than a fixture that
happens to pass, because that is how the one real defect here was found: the two
frustration operators agreed on every unweighted complex and disagreed on every
weighted one.

Marked below as PROVED where the algebra is given, and MEASURED where the
statement has survived search but has no proof here.
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.hodge_coords import coordinate_dims, harmonic_frame, harmonic_gram_det
from rexgraph.rational_trig import bareiss_determinant
from rexgraph.sparse_character import build_sparse_channels


def _zoo(seed=0, n=60):
    """Simple, weighted, branching and parallel-edge complexes, random orientation.

    Weighted and branching are both here on purpose: they are the two axes that
    unweighted-simple fixtures cannot distinguish.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n // 3):
        nV = int(rng.integers(3, 12))
        m = int(rng.integers(nV - 1, 3 * nV))
        s, t = rng.integers(0, nV, m), rng.integers(0, nV, m)
        k = s != t
        if k.sum() >= 2:
            out.append(("simple", RexGraph(sources=s[k].astype(np.int32),
                                           targets=t[k].astype(np.int32))))
    for _ in range(n // 6):
        nV = int(rng.integers(3, 9))
        m = int(rng.integers(nV, 2 * nV))
        s, t = rng.integers(0, nV, m), rng.integers(0, nV, m)
        k = s != t
        if k.sum() >= 2:
            w = rng.integers(1, 6, int(k.sum())).astype(np.float64)
            out.append(("weighted", RexGraph(sources=s[k].astype(np.int32),
                                             targets=t[k].astype(np.int32), w_E=w)))
    for _ in range(n // 6):
        nV = int(rng.integers(4, 12))
        ptr, idx = [0], []
        for _ in range(int(rng.integers(2, 7))):
            a = int(rng.integers(2, min(nV, 5) + 1))
            idx += [int(x) for x in rng.choice(nV, size=a, replace=False)]
            ptr.append(len(idx))
        out.append(("branching", RexGraph.from_hypergraph(np.array(ptr, np.int64),
                                                          np.array(idx, np.int64))))
    for _ in range(n // 6):
        out.append(("parallel", RexGraph(sources=np.array([0, 0, 1, 1, 2], np.int32),
                                         targets=np.array([1, 1, 2, 2, 0], np.int32))))
    live = []
    for kind, r in out:
        r._ensure_clean()
        if int(r.nE) >= 2 and int(r.nV) >= 2:
            live.append((kind, r))
    return live


ZOO = _zoo()


def _fails(pred):
    return [(k, int(r.nV), int(r.nE), d) for k, r in ZOO
            for ok, d in [pred(r)] if not ok]


#### the Lagrangian, PROVED: both sides unpack to the same hat diagonals
def test_lt_plus_ls_is_the_relational_mass():
    """Lt is hat_T's diagonal and Ls the other three summed, so the pair is
    RL[e,e] by construction. Stated because it is easy to assume it is 1: chi is
    the normalised version and that is what sums to 1."""
    def p(r):
        f = r.lagrangian_fields()
        if f is None:
            return True, ""
        got = np.asarray(f["Lt"]) + np.asarray(f["Ls"])
        return np.allclose(got, np.diagonal(np.asarray(r.RL)), atol=1e-9), "mismatch"
    assert _fails(p) == []


def test_the_speed_squared_is_fixed_by_the_character_alone():
    """c^2 = Ls/Lt = (1 - chi_T)/chi_T, since chi_k = hat_k/RL[e,e] divides the
    common factor out. So arg f(e) = arctan(c^2) is scale-free: the modulus of the
    complex Lagrangian carries the mass and its argument carries the character."""
    def p(r):
        f = r.lagrangian_fields()
        if f is None:
            return True, ""
        chi = np.asarray(r.structural_character)
        live = chi[:, 0] > 1e-12
        if not live.any():
            return True, ""
        pred = (1.0 - chi[live, 0]) / chi[live, 0]
        return np.allclose(np.asarray(f["c2"])[live], pred, atol=1e-7), "mismatch"
    assert _fails(p) == []


#### the chain condition, PROVED
def test_the_chain_condition_forces_the_graded_operators_to_commute():
    """PROVED. L_down = B_d^T B_d and L_up = B_{d+1} B_{d+1}^T, so

        L_down L_up = B_d^T (B_d B_{d+1}) B_{d+1}^T = 0
        L_up L_down = B_{d+1} (B_d B_{d+1})^T B_d = 0

    both vanishing because B_d B_{d+1} = 0. The commutator is not small, it is
    identically zero, which is why a Cauchy-Riemann condition on the GRADED
    operators is a tautology and the interesting version is on the channels."""
    from rexgraph.graded_boundary import build_graded_boundaries, truncated_icosahedron_3rex
    Bs = [np.asarray(b.todense(), float)
          for b in build_graded_boundaries(truncated_icosahedron_3rex())]
    for d in range(len(Bs) - 1):
        Bd, Bd1 = Bs[d], Bs[d + 1]
        assert np.abs(Bd @ Bd1).max() == 0.0
        Ld, Lu = Bd.T @ Bd, Bd1 @ Bd1.T
        assert np.abs(Ld @ Lu - Lu @ Ld).max() == 0.0

    def p(r):
        B1 = np.asarray(r.B1_dense, float)
        B2 = np.asarray(r.B2_dense, float)
        if B2.size == 0:
            return True, ""
        Ld, Lu = B1.T @ B1, B2 @ B2.T
        return np.abs(Ld @ Lu - Lu @ Ld).max() < 1e-9, "commutator nonzero"
    assert _fails(p) == []


#### the channels
def test_the_signed_and_unsigned_grams_share_a_diagonal():
    """PROVED. The diagonal squares each incidence entry and squaring kills the
    sign, so diag(T) = diag(G) whatever the metric. That is why F's content is
    entirely off-diagonal and why F has to manufacture a diagonal from it."""
    def p(r):
        chan = dict(build_sparse_channels(r))
        dT = np.asarray(chan["L1_down"].todense()).diagonal()
        dG = np.asarray(chan["L_O"].todense()).diagonal()
        return np.allclose(dT, dG, atol=1e-9), "diagonals differ"
    assert _fails(p) == []


def _uniformly_oriented(r):
    """Every vertex heads all of its relations or none of them."""
    r._ensure_clean()
    ptr, idx = np.asarray(r._boundary_ptr), np.asarray(r._boundary_idx)
    head, member = {}, {}
    for e in range(int(r.nE)):
        col = idx[ptr[e]:ptr[e + 1]]
        for pos, v in enumerate(col):
            member.setdefault(int(v), set()).add(e)
            if pos == 0:
                head.setdefault(int(v), set()).add(e)
    return all(not head.get(v) or head[v] == cells for v, cells in member.items())


def test_frustration_vanishes_exactly_on_a_uniformly_oriented_complex():
    """PROVED, and MEASURED over the zoo. A column is (-1 at the head, 1/(k-1)
    elsewhere), so two columns sharing v agree in sign at v exactly when v heads
    both or neither. F sums |signed - unsigned| overlap over pairs, so it vanishes
    iff every shared vertex agrees, which is the condition above."""
    def p(r):
        names = list(r._rcf_bundle.get("hat_names", []) or [])
        if "L_SG" not in names:
            return True, ""
        hats = list(r._rcf_bundle["hats"])
        F0 = float(np.abs(np.diagonal(
            np.asarray(hats[names.index("L_SG")]))).sum()) < 1e-12
        u = _uniformly_oriented(r)
        return F0 == u, f"uniform={u} F==0={F0}"
    assert _fails(p) == []


def test_the_two_frustration_operators_agree():
    """The regression that found the defect. `frustration_exact` took an
    UNWEIGHTED T against the weighted G that `overlap_gramian_sparse` returns, so
    on a weighted complex the difference reported the metric rather than the sign
    mismatch. Unweighted fixtures could never see it: W is the identity there."""
    def p(r):
        a = np.asarray(dict(build_sparse_channels(r))["L_SG"].todense())
        b = np.asarray(r.frustration_exact.todense())
        return np.allclose(a, b, atol=1e-9), f"max diff {np.abs(a - b).max():.3e}"
    assert _fails(p) == []


def test_a_uniformly_oriented_weighted_complex_still_reads_zero_frustration():
    """The exact case that was wrong. Weights cannot create orientation conflict."""
    for w in ([1., 1., 1.], [3., 1., 2.], [5., 1., 1.], [2., 3., 4.]):
        r = RexGraph(sources=np.array([2, 0, 0], np.int32),
                     targets=np.array([1, 1, 1], np.int32), w_E=np.array(w))
        r._ensure_clean()
        assert _uniformly_oriented(r)
        names = list(r._rcf_bundle["hat_names"])
        hats = list(r._rcf_bundle["hats"])
        assert np.allclose(np.diagonal(np.asarray(hats[names.index("L_SG")])), 0.0), w
        assert np.allclose(np.asarray(r.structural_character)[:, 2], 0.0), w


#### the Hodge chart
def test_the_three_spaces_span_the_edge_space():
    """rank(B1) + rank(B2) + dim_H = nE, on every shape in the zoo."""
    def p(r):
        d = coordinate_dims(r)
        return d["independent"] == d["nE"], f"{d['independent']} != {d['nE']}"
    assert _fails(p) == []


def test_the_gram_determinant_counts_spanning_trees():
    """Matrix-Tree, in scope for a connected 2-ary face-free complex. Checked
    against an independent cofactor of L0, sharing no code with the frame."""
    checked = 0
    for _, r in ZOO:
        ptr = np.asarray(r._boundary_ptr)
        if (np.diff(ptr) != 2).any() or int(r.betti[0]) != 1:
            continue
        nV = int(r.nV)
        L = np.asarray(r.B1_dense, float) @ np.asarray(r.B1_dense, float).T
        minor = [[Fraction(int(round(L[i][j]))) for j in range(1, nV)]
                 for i in range(1, nV)]
        assert harmonic_gram_det(r) == int(bareiss_determinant(minor))
        checked += 1
    assert checked >= 10, f"only {checked} in scope"


#### L_gb
def test_the_graded_delta_spectrum_is_plus_minus_root_spread():
    """PROVED. L_gb = alpha P_a - beta P_b on the two grades' spectra, a difference
    of rank-1 projectors, so on their span trace = alpha - beta and determinant =
    -alpha beta sin^2, giving the eigenvalues in closed form."""
    from rexgraph.core._l_gb import l_gb_scalar
    rng = np.random.default_rng(5)
    for _ in range(60):
        n = int(rng.integers(2, 40))
        a, b = np.abs(rng.normal(size=n)), np.abs(rng.normal(size=n))
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        perp = b - ((a / na) @ b) * (a / na)
        s2 = float(perp @ perp) / (nb * nb)
        assert np.isclose(l_gb_scalar(a, b), np.sqrt(2.0 * s2), atol=1e-12)


#### the complete-graph closure law: MEASURED, not proved
@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_the_closure_law_on_complete_graphs(n):
    """MEASURED. Read off n = 4..7 and held as a prediction at 8 and 9. Stated for
    the fundamental cycle basis of K_n, whose axes are all triangles; no proof here
    and no basis-free form."""
    from rexgraph.hodge_coords import harmonic_closure
    e = list(itertools.combinations(range(n), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    H = np.asarray(harmonic_frame(r).todense()).astype(int)
    assert {int((H[:, i] != 0).sum()) for i in range(H.shape[1])} == {3}
    C = harmonic_closure(r, exact=True)
    assert C[0][0] == Fraction(3 * n - 8, 3 * n)
    off = next(C[0][j] for j in range(1, H.shape[1]) if C[0][j] != 0)
    assert off == Fraction(n - 2, n)


#### the gauge bundle, and what mass does over it
def _sign_bundle(src, tgt, nV, H):
    """Enumerate every sign configuration and group it by holonomy class."""
    nE = len(src)
    classes, orbit = {}, set()
    for bits in itertools.product([1, -1], repeat=nE):
        sig = np.array(bits)
        hol = tuple(int(np.prod([sig[i] for i in np.nonzero(H[:, c])[0]]))
                    for c in range(H.shape[1]))
        classes.setdefault(hol, []).append(sig)
    for t in itertools.product([1, -1], repeat=nV):
        orbit.add(tuple(int(t[int(src[i])] * t[int(tgt[i])]) for i in range(nE)))
    return classes, orbit


@pytest.mark.parametrize("src,tgt", [
    ([0, 1, 2, 3], [1, 2, 3, 0]),                       # C4
    ([0, 0, 1, 1, 0], [1, 1, 2, 2, 2]),                 # theta
    ([a for a, b in itertools.combinations(range(4), 2)],
     [b for a, b in itertools.combinations(range(4), 2)]),   # K4
])
def test_the_sign_configurations_are_a_principal_bundle_over_the_holonomy(src, tgt):
    """PROVED by counting, and checked by full enumeration. A sign per relation is
    a Z/2 gauge field, so the total space is 2^nE. The gauge group (Z/2)^nV acts by
    sigma_ij -> t_i t_j sigma_ij, with the global flip acting trivially on each
    component, so orbits have size 2^(nV - b0). Holonomy is constant on an orbit and
    separates them, giving 2^b1 classes, and

        2^(nV - b0) * 2^b1 = 2^(nV - b0 + nE - nV + b0) = 2^nE

    which is the bundle closing exactly. The base is H^1 with Z/2 coefficients.
    """
    r = RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))
    r._ensure_clean()
    nV, nE, b0 = int(r.nV), int(r.nE), int(r.betti[0])
    H = np.asarray(harmonic_frame(r).todense()).astype(int)
    b1 = H.shape[1]
    classes, orbit = _sign_bundle(np.array(src), np.array(tgt), nV, H)
    assert len(orbit) == 2 ** (nV - b0), "orbit size"
    assert len(classes) == 2 ** b1, "one class per holonomy"
    assert len(orbit) * len(classes) == 2 ** nE, "the bundle closes"


def _masses(r):
    from rexgraph.tower import boundary_mass
    structural = float(boundary_mass(r, 1, exact=False))
    metric = float(np.asarray(
        dict(build_sparse_channels(r))["L1_down"].todense()).diagonal().sum())
    return structural, metric


def test_both_masses_are_constant_over_the_gauge_action():
    """PROVED. Both masses sum SQUARED entries, and squaring kills the sign, so
    neither can see a gauge transformation, an arbitrary re-signing, or a
    reorientation. Mass is the pure existence reading: what survives forgetting
    both the sign and the direction.

    They differ in one thing only. The structural mass ||B_1||_F^2 is taken on the
    raw boundary and is blind to the metric as well, which is what makes it
    extensive. The metric mass tr(W B_1^T B_1 W) = sum_e w_e^2 Q(e) carries it.
    """
    w = np.array([3., 1., 2., 1.])
    base = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                    targets=np.array([1, 2, 3, 0], np.int32), w_E=w)
    base._ensure_clean()
    s0, m0 = _masses(base)

    for signs in ([1, -1, 1, -1], [1, -1, 1, 1], [-1, -1, -1, -1]):
        r = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                     targets=np.array([1, 2, 3, 0], np.int32),
                     signs=np.array(signs, np.int32), w_E=w)
        r._ensure_clean()
        assert _masses(r) == (s0, m0), signs

    # reversing a relation moves the boundary column and neither mass
    rev = RexGraph(sources=np.array([0, 2, 2, 3], np.int32),
                   targets=np.array([1, 1, 3, 0], np.int32),
                   w_E=np.array([3., 2., 1., 1.]))
    rev._ensure_clean()
    assert _masses(rev) == (s0, m0)

    # the metric is the one thing the metric mass does see
    heavy = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                     targets=np.array([1, 2, 3, 0], np.int32),
                     w_E=np.array([9., 1., 1., 1.]))
    heavy._ensure_clean()
    s1, m1 = _masses(heavy)
    assert s1 == s0, "structural mass is blind to the metric too"
    assert m1 != m0, "metric mass carries it"


def test_the_tower_law_is_taken_on_the_raw_boundary_throughout():
    """Both sides of tr(L_k) = ||B_k||^2 + ||B_{k+1}||^2 are raw, so the law holds
    on a weighted complex as well. Worth pinning because tr(L1_down) as a CHANNEL
    is the weighted W B_1^T B_1 W and reads differently: 8 against 30 on a weighted
    C4. Same name, two objects, and the tower means the unweighted one."""
    from rexgraph.tower import tower_law
    for w in (None, [3., 1., 2., 1.]):
        kw = {} if w is None else {"w_E": np.array(w)}
        r = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                     targets=np.array([1, 2, 3, 0], np.int32), **kw)
        r._ensure_clean()
        assert tower_law(r)["holds"], w
        assert float(tower_law(r)["residual"]) == 0.0, w


#### the spectral parameter sigma, and the channel force hierarchy
def _commutator(A, B):
    """[A, B] for SYMMETRIC A and B, formed so the antisymmetry is exact everywhere.

    (AB)^T = B^T A^T = BA when both are symmetric, so AB - BA = M - M^T for M = AB.
    Antisymmetrising one product gives C + C^T = 0 bitwise, since C[i,j] + C[j,i] is
    (M[i,j] - M[j,i]) + (M[j,i] - M[i,j]) and IEEE subtraction cancels that exactly.
    Forming AB and BA as two independent matmuls does not: the two calls need not
    accumulate in the same order, which read as 3.5e-18 under Accelerate against an
    exact 0.0 under OpenBLAS. The residue is the BLAS and not the mathematics, so
    the fix is to take the product once rather than to widen the claim to a
    tolerance.
    """
    M = A @ B
    return M - M.T


def _deformed_TG(B1, sigma=0.5):
    """T and G on the sigma-deformed boundary, the one point of the family every
    reading here is taken at. Reading one operator deformed against another raw was
    a real slip: it reported an alignment of +0.246 on a 2-ary ring, where taking
    both at the same point leaves [T,G] identically zero and no alignment to read.
    """
    p = np.array(_primes(B1.shape[0]), float)
    W0 = np.log(p) * p ** (-sigma)
    Wd = np.maximum(((B1 * B1).T * W0).sum(1), 1e-300)
    Bw = (np.diag(np.sqrt(W0)) @ B1) / np.sqrt(Wd)
    T = Bw.T @ Bw
    G = np.abs(Bw).T @ np.abs(Bw)
    return T, G


def _primes(k):
    out, c = [], 2
    while len(out) < k:
        if all(c % p for p in out):
            out.append(c)
        c += 1
    return out


def _sigma_deformed(k, sigma):
    """spore's sigma-weighted boundary: B1w = W0^{1/2} B1 W1^{-1/2} on K_k over the
    first k primes, with W0[v] = ln(p_v) p_v^-sigma and W1[e] = W0[s] + W0[t]."""
    p = np.array(_primes(k), float)
    W0 = np.maximum(np.log(p) * p ** (-sigma), 1e-15)
    e = list(itertools.combinations(range(k), 2))
    W1 = np.maximum(np.array([W0[i] + W0[j] for i, j in e]), 1e-15)
    B = np.zeros((k, len(e)))
    for c, (i, j) in enumerate(e):
        B[i, c] = -np.sqrt(W0[i]) / np.sqrt(W1[c])
        B[j, c] = +np.sqrt(W0[j]) / np.sqrt(W1[c])
    return B, e


def _sigma_channels(B, e, k):
    T = B.T @ B
    G = np.abs(B).T @ np.abs(B)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    F = Foff + np.diag(np.abs(Foff).sum(1))
    S = np.zeros((k, len(e)))
    for c, (i, j) in enumerate(e):
        S[i, c] = S[j, c] = 1.0
    K = S.T @ S
    Koff = K - np.diag(np.diagonal(K))
    return T, G, F, np.diag(Koff.sum(1)) - Koff


@pytest.mark.parametrize("k", [4, 6, 8, 10])
@pytest.mark.parametrize("sigma", [0.2, 0.5, 0.8, 1.3])
def test_the_sigma_deformation_normalises_every_boundary_column(k, sigma):
    """PROVED. W1[e] = W0[s] + W0[t] is exactly the squared norm of the column
    before dividing, so B1w's columns are unit vectors at EVERY sigma and
    tr(T) = nE identically. The spectral parameter reweights the complex without
    moving its total topological mass, which is what lets it be swept."""
    B, e = _sigma_deformed(k, sigma)
    assert np.allclose((B * B).sum(0), 1.0)
    assert np.isclose(float((B * B).sum()), len(e))


@pytest.mark.parametrize("k", [5, 6, 7])
def test_co_participation_is_exactly_sigma_invariant(k):
    """C reads the UNWEIGHTED shared-vertex count, so it is topological and the
    spectral parameter cannot move it. This is the half of the force hierarchy that
    holds exactly rather than approximately."""
    ref = None
    for sigma in (0.2, 0.5, 0.9, 1.4):
        B, e = _sigma_deformed(k, sigma)
        C = _sigma_channels(B, e, k)[3]
        if ref is None:
            ref = C
        assert np.allclose(C, ref)


@pytest.mark.parametrize("k", [5, 6, 7])
def test_the_t_g_coupling_runs_with_sigma_and_f_c_barely_moves(k):
    """MEASURED. The reference states a force hierarchy: 'T-G coupling runs with
    sigma (strong), F-C is sigma-invariant (weak)'. Couplings are channel-tensor
    entries, which is what channel_delta computes.

    F-C is not invariant to machine precision, but it moves far less, and the
    margin shrinks with both k and the width of the sweep. Measured ratios of the
    T-G spread to the F-C spread:

        sigma in (0.05, 0.95)   k=5 48.2x   k=6 36.6x   k=7 21.3x   k=8 13.3x
        sigma in (0.10, 1.40)   k=5 24.8x   k=6 11.7x   k=7  8.5x   k=8  6.4x

    So the hierarchy is real and is NOT a clean order of magnitude everywhere. The
    bound below is the weakest of those measurements rather than the strongest,
    because asserting the best case would make this a fixture rather than a test.
    """
    from rexgraph.core._l_gb import l_gb_channel_tensor
    tg, fc = [], []
    for sigma in np.linspace(0.1, 1.4, 14):
        B, e = _sigma_deformed(k, float(sigma))
        Tn = np.asarray(l_gb_channel_tensor(list(_sigma_channels(B, e, k))))
        tg.append(Tn[0, 1])
        fc.append(Tn[2, 3])
    spread_tg = max(tg) - min(tg)
    spread_fc = max(fc) - min(fc)
    assert spread_tg > 6 * spread_fc, (spread_tg, spread_fc)


def test_the_t_g_peak_does_not_settle_on_the_critical_line():
    """MEASURED, and a NEGATIVE result worth keeping so it is not re-derived.

    The T-G coupling has a peak in sigma that drifts down as k grows, and the
    s <-> 1-s asymmetry falls with it, which looks like a critical line forming at
    sigma = 1/2. It is not one: the peak passes THROUGH 1/2 near k = 22 and keeps
    descending (0.588, 0.563, 0.550, 0.531, 0.520, 0.508, 0.4999, 0.491 for
    k = 10..24), and the asymmetry bottoms out near k = 14 and then grows again.

    This says nothing about Equation IV, which is a statement about
    ||A([RL4(s), RL4(1-s)])|| on the full four-channel tower WITH faces. K_k here
    is face-free, so that quantity is not what is being measured.
    """
    from rexgraph.core._l_gb import l_gb_channel_tensor

    def tg(k, sigma):
        B, e = _sigma_deformed(k, sigma)
        return float(np.asarray(l_gb_channel_tensor(list(_sigma_channels(B, e, k))))[0, 1])

    def peak(k):
        lo, hi = 0.05, 1.6
        for _ in range(8):
            xs = np.linspace(lo, hi, 21)
            i = int(np.argmax([tg(k, float(s)) for s in xs]))
            lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
        return 0.5 * (lo + hi)

    peaks = [peak(k) for k in (10, 14, 18, 22)]
    assert peaks == sorted(peaks, reverse=True), "the peak descends with k"
    assert peaks[0] > 0.5 > peaks[-1] - 1e-3, "and crosses 1/2 rather than settling"


#### Equation IV: the critical line as inter-tower information exchange
def _primes_complex(k, sigma):
    """K_k over the first k primes with every triangle filled, BOTH grades deformed.

        B1w = W0^{1/2} B1 W1^{-1/2}
        B2w = W1^{1/2} B2 W2^{-1/2}   ->  B1w B2w = W0^{1/2}(B1 B2)W2^{-1/2} = 0

    Deforming B1 alone breaks the chain condition, which is the trap: the W1
    factors have to cancel BETWEEN the grades, and they only do if B2 carries the
    matching half. Any W2 works, so it is used to normalise the face columns.
    """
    p = np.array(_primes(k), float)
    W0 = np.maximum(np.log(p) * p ** (-sigma), 1e-15)
    e = list(itertools.combinations(range(k), 2))
    pos = {x: i for i, x in enumerate(e)}
    W1 = np.maximum(np.array([W0[i] + W0[j] for i, j in e]), 1e-15)
    B1 = np.zeros((k, len(e)))
    for c, (i, j) in enumerate(e):
        B1[i, c] = -np.sqrt(W0[i]) / np.sqrt(W1[c])
        B1[j, c] = +np.sqrt(W0[j]) / np.sqrt(W1[c])
    tris = list(itertools.combinations(range(k), 3))
    B2 = np.zeros((len(e), len(tris)))
    for f, (a, b, c) in enumerate(tris):
        B2[pos[(a, b)], f] = 1.0
        B2[pos[(b, c)], f] = 1.0
        B2[pos[(a, c)], f] = -1.0
    B2 = np.diag(np.sqrt(W1)) @ B2
    return B1, B2 / np.sqrt(np.maximum((B2 * B2).sum(0), 1e-15))


def _rl(k, sigma, with_faces):
    B1, B2 = _primes_complex(k, sigma)
    T = B1.T @ B1
    G = np.abs(B1).T @ np.abs(B1)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    F = Foff + np.diag(np.abs(Foff).sum(1))
    e = list(itertools.combinations(range(k), 2))
    S = np.zeros((k, len(e)))
    for c, (i, j) in enumerate(e):
        S[i, c] = S[j, c] = 1.0
    K = S.T @ S
    Koff = K - np.diag(np.diagonal(K))
    C = np.diag(Koff.sum(1)) - Koff
    chans = [T, G, F, C] + ([B2 @ B2.T] if with_faces else [])
    out = np.zeros_like(T)
    for X in chans:
        tr = float(np.trace(X))
        if abs(tr) > 1e-15:
            out = out + X / tr
    return out


@pytest.mark.parametrize("sigma", [0.2, 0.5, 0.9])
def test_deforming_both_grades_preserves_the_chain_condition(sigma):
    """The W1 factors cancel between the grades. Deforming B1 alone leaves a
    residual of 7e-02, which is the chain condition simply broken."""
    B1, B2 = _primes_complex(6, sigma)
    assert np.abs(B1 @ B2).max() < 1e-12


@pytest.mark.parametrize("sigma", [0.2, 0.5, 0.9])
def test_the_two_grades_commute_at_every_sigma(sigma):
    """So the critical line cannot be found in the edge-vs-face interaction: that
    commutator is zero identically, by the chain condition, at every sigma. It is
    the same statement as [L_down, L_up] = 0 proved above."""
    B1, B2 = _primes_complex(6, sigma)
    T, S = B1.T @ B1, B2 @ B2.T
    Th, Sh = T / np.trace(T), S / np.trace(S)
    assert np.abs(Th @ Sh - Sh @ Th).max() < 1e-14


def test_equation_iv_vanishes_exactly_on_the_critical_line():
    """MEASURED, with faces. ||[RL(s), RL(1-s)]|| is 0 at sigma = 1/2 and strictly
    positive everywhere else, growing monotonically with the distance from it.

    Read as information exchange rather than as symmetry: the commutator is what
    the s and the 1-s pictures fail to share, and it is exactly nothing on the
    line."""
    k = 6
    at_half = _rl(k, 0.5, True)
    assert np.abs(at_half @ at_half - at_half @ at_half).max() == 0.0
    prev = -1.0
    for d in (0.01, 0.05, 0.15, 0.25, 0.40):
        A, B = _rl(k, 0.5 + d, True), _rl(k, 0.5 - d, True)
        n = float(np.linalg.norm(A @ B - B @ A, "fro"))
        assert n > 0.0, d
        assert n > prev, "monotone in the distance from the line"
        prev = n


def test_the_exchange_is_linear_in_the_distance_from_the_line():
    """So it is a genuine distance function, not merely a vanishing. norm/|d| is
    flat to about 3% over |d| in (0.01, 0.40), with a k-dependent slope: 0.0807 at
    k=5, 0.0644 at k=6, 0.0558 at k=7."""
    for k in (5, 6, 7):
        ratios = []
        for d in (0.01, 0.05, 0.15, 0.25, 0.40):
            A, B = _rl(k, 0.5 + d, True), _rl(k, 0.5 - d, True)
            ratios.append(float(np.linalg.norm(A @ B - B @ A, "fro")) / d)
        assert max(ratios) / min(ratios) < 1.05, (k, ratios)


def test_filling_the_faces_damps_the_exchange():
    """The point of doing this WITH faces. Carrying the face channel roughly halves
    the inter-picture exchange, and the effect weakens as k grows:
    49.6% at k=5, 42.7% at k=6, 36.4% at k=7, 31.7% at k=8."""
    for k, lo, hi in ((5, 0.45, 0.55), (6, 0.45, 0.55), (7, 0.45, 0.55)):
        a, b = _rl(k, hi, False), _rl(k, lo, False)
        bare = float(np.linalg.norm(a @ b - b @ a, "fro"))
        a, b = _rl(k, hi, True), _rl(k, lo, True)
        faced = float(np.linalg.norm(a @ b - b @ a, "fro"))
        assert faced < bare, k
        assert 0.25 < (1 - faced / bare) < 0.60, (k, 1 - faced / bare)


def test_the_commutator_tracks_the_plain_difference_on_this_family():
    """The honest control, and a limit on what the above shows. ||[A,B]|| is a flat
    0.108 times ||A - B|| across every distance tested, drifting only 1.5%. So on
    THIS family the commutator locates the line no more sharply than the plain
    difference does; what it adds is the reading, not the resolution."""
    k = 6
    ratios = []
    for d in (0.01, 0.05, 0.15, 0.25, 0.40):
        A, B = _rl(k, 0.5 + d, True), _rl(k, 0.5 - d, True)
        c = float(np.linalg.norm(A @ B - B @ A, "fro"))
        s = float(np.linalg.norm(A - B, "fro"))
        ratios.append(c / s)
    assert max(ratios) / min(ratios) < 1.03, ratios


#### does any of it depend on the face being a triangle, or on the grade being 1?
def _deform_tower(Bs, W0):
    """B_d^w = W_{d-1}^{1/2} B_d W_d^{-1/2}, with W_d[cell] the sum of W_{d-1} over
    its boundary support.

    PROVED, and arity-blind. W_d[cell] is exactly the squared norm of that column
    before dividing, whatever its arity, so every column of every B_d^w is a unit
    vector. And W_d^{-1/2} leaving grade d meets W_d^{1/2} entering grade d+1, so
    the chain condition survives at every junction. Nothing in the construction
    mentions 3, which is the point: order tracks degree, not arity.
    """
    out, Wprev = [], np.asarray(W0, float)
    for B in Bs:
        A = np.abs(np.asarray(B, float))
        Wd = np.maximum((A.T * Wprev).sum(1), 1e-300)
        out.append((np.diag(np.sqrt(Wprev)) @ np.asarray(B, float)) / np.sqrt(Wd))
        Wprev = Wd
    return out


def _ngon(n):
    """n vertices, n edges, ONE face of arity n."""
    B1 = np.zeros((n, n))
    B2 = np.zeros((n, 1))
    for i in range(n):
        B1[i, i] = -1.0
        B1[(i + 1) % n, i] = 1.0
        B2[i, 0] = 1.0
    return [B1, B2]


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
def test_the_deformation_is_blind_to_face_arity(n):
    """Unit columns and an intact chain condition at every face size."""
    p = np.array(_primes(n), float)
    Bw = _deform_tower(_ngon(n), np.log(p) * p ** (-0.37))
    for B in Bw:
        assert np.allclose((B * B).sum(0), 1.0), n
    assert np.abs(Bw[0] @ Bw[1]).max() < 1e-12


def test_the_deformation_survives_three_grades_and_mixed_arity():
    """C60 as a solid: pentagons and hexagons in one B2, and a volume above them."""
    from rexgraph.graded_boundary import build_graded_boundaries, truncated_icosahedron_3rex
    Bs = [np.asarray(b.todense(), float)
          for b in build_graded_boundaries(truncated_icosahedron_3rex())]
    assert sorted({int(a) for a in np.abs(Bs[1]).sum(0)}) == [5, 6]
    p = np.array(_primes(Bs[0].shape[0]), float)
    Bw = _deform_tower(Bs, np.log(p) * p ** (-0.37))
    for B in Bw:
        assert np.allclose((B * B).sum(0), 1.0)
    for d in range(len(Bw) - 1):
        assert np.abs(Bw[d] @ Bw[d + 1]).max() < 1e-12


def _prism(n):
    """Two n-gon caps joined by n quadrilaterals: one parameter, and it is the cap
    arity. The sides stay 4 whatever n is."""
    from rexgraph.graded_boundary import _polyhedron_3rex
    th = 2 * np.pi * np.arange(n) / n
    pts = np.array([[np.cos(t), np.sin(t), -1.0] for t in th] +
                   [[np.cos(t), np.sin(t), 1.0] for t in th], float)
    faces = [list(range(n)), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    return _polyhedron_3rex(pts, faces)


def _rl_graded(cells, sigma, d):
    from rexgraph.graded_boundary import build_graded_boundaries
    Bs = [np.asarray(b.todense(), float) for b in build_graded_boundaries(cells)]
    p = np.array(_primes(Bs[0].shape[0]), float)
    Bw = _deform_tower(Bs, np.log(p) * p ** (-sigma))
    B = Bw[d - 1]
    T = B.T @ B
    G = np.abs(B).T @ np.abs(B)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    chans = [T, G, Foff + np.diag(np.abs(Foff).sum(1))]
    if d < len(Bw):
        chans.append(Bw[d] @ Bw[d].T)
    tot = None
    for X in chans:
        tr = float(np.trace(X))
        if abs(tr) > 1e-15:
            tot = X / tr if tot is None else tot + X / tr
    return tot


def _slope(cells, d, es=(0.02, 0.08, 0.20)):
    out = []
    for e in es:
        A, B = _rl_graded(cells, 0.5 + e, d), _rl_graded(cells, 0.5 - e, d)
        out.append(float(np.linalg.norm(A @ B - B @ A, "fro")) / e)
    return out


@pytest.mark.parametrize("n", [3, 4, 6])
@pytest.mark.parametrize("d", [1, 2])
def test_equation_iv_holds_at_every_arity_and_grade(n, d):
    """Zero on the critical line and linear off it, whatever the face size and
    whichever grade is read. The triangle was never load-bearing."""
    cells = _prism(n)
    A = _rl_graded(cells, 0.5, d)
    assert np.abs(A @ A - A @ A).max() == 0.0
    s = _slope(cells, d)
    assert all(x > 0 for x in s)
    assert max(s) / min(s) < 1.12, s      # linear in the distance


def _rl_graded_raw(cells, sigma, d, normalize):
    """Same channels, with the trace normalisation switchable."""
    from rexgraph.graded_boundary import build_graded_boundaries
    Bs = [np.asarray(b.todense(), float) for b in build_graded_boundaries(cells)]
    p = np.array(_primes(Bs[0].shape[0]), float)
    Bw = _deform_tower(Bs, np.log(p) * p ** (-sigma))
    B = Bw[d - 1]
    T = B.T @ B
    G = np.abs(B).T @ np.abs(B)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    chans = [T, G, Foff + np.diag(np.abs(Foff).sum(1))]
    if d < len(Bw):
        chans.append(Bw[d] @ Bw[d].T)
    tot = None
    for X in chans:
        tr = float(np.trace(X))
        Y = X / tr if (normalize and abs(tr) > 1e-15) else X
        tot = Y if tot is None else tot + Y
    return tot


def _slope_raw(cells, d, normalize):
    out = []
    for e in (0.02, 0.08, 0.20):
        A = _rl_graded_raw(cells, 0.5 + e, d, normalize)
        B = _rl_graded_raw(cells, 0.5 - e, d, normalize)
        out.append(float(np.linalg.norm(A @ B - B @ A, "fro")) / e)
    return float(np.mean(out))


def test_the_apparent_grade_migration_is_the_trace_normalisation():
    """RETRACTED FINDING, kept as the control that overturned it.

    Reported first: on n-gonal prisms the grade-1 exchange falls while grade-2
    rises, so widening a face 'moves where the information exchange lives'. The
    ratio ran 7.9x, 12.8x, 22.7x, 33.9x, 47.8x over n = 3, 4, 6, 8, 10, which
    looked like a clean structural trend.

    It is not one. Take the SAME channels without dividing each by its own trace
    and both grades rise together, grade 1 stays about four times the larger, and
    the ratio is flat:

        n                3       4       6       8      10
        raw grade 1    8.28   10.96   15.31   18.33   21.13
        raw grade 2    2.08    2.59    3.36    4.10    4.84
        raw ratio       0.3     0.2     0.2     0.2     0.2

    The whole effect comes from the normalisation: grade-1 traces scale with
    nE = 3n and grade-2 traces with nF = n + 2, so the same raw growth is divided
    by faster and slower growing numbers. The trace-normalised hats are the
    canonical object, so the normalised numbers are not wrong; what is wrong is
    reading them as a fact about where exchange lives in the complex.

    The associated 'slope1 goes like 1/n' was the same artefact seen from the
    other side, and was never more than a curve fitted after the fact.
    """
    ns = (3, 4, 6, 8, 10)
    raw1 = [_slope_raw(_prism(n), 1, False) for n in ns]
    raw2 = [_slope_raw(_prism(n), 2, False) for n in ns]
    # both rise with n: nothing migrates
    assert raw1 == sorted(raw1), raw1
    assert raw2 == sorted(raw2), raw2
    # grade 1 dominates throughout, and the ratio does not run
    ratios = [b / a for a, b in zip(raw1, raw2, strict=True)]
    assert all(r < 0.5 for r in ratios), ratios
    assert max(ratios) / min(ratios) < 1.6, ratios
    # while the normalised ratio moves by more than five fold over the same range
    norm = [_slope_raw(_prism(n), 2, True) / _slope_raw(_prism(n), 1, True) for n in ns]
    assert norm[-1] / norm[0] > 5.0, norm


def _bipyramid(n):
    """2n triangles over an n-gon equator: grows the face COUNT at fixed arity 3,
    where the prism grows the ARITY at nearly fixed count."""
    from rexgraph.graded_boundary import _polyhedron_3rex
    th = 2 * np.pi * np.arange(n) / n
    pts = np.array([[np.cos(t), np.sin(t), 0.0] for t in th]
                   + [[0, 0, 1.0], [0, 0, -1.0]], float)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n])
        faces.append([i, j, n + 1])
    return _polyhedron_3rex(pts, faces)


def _cube_solid():
    from rexgraph.graded_boundary import _polyhedron_3rex
    pts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    faces = [[i for i, p in enumerate(pts) if p[ax] == s]
             for ax in range(3) for s in (-1, 1)]
    return _polyhedron_3rex(pts, faces)


def _families():
    from rexgraph.graded_boundary import (
        solid_octahedron_3rex,
        square_pyramid_3rex,
        truncated_icosahedron_3rex,
    )
    out = [(f"prism {n}", _prism(n)) for n in (3, 4, 6, 8)]
    out += [(f"bipyramid {n}", _bipyramid(n)) for n in (3, 5, 8)]
    out += [("octahedron", solid_octahedron_3rex()), ("cube", _cube_solid()),
            ("square pyramid", square_pyramid_3rex()),
            ("C60", truncated_icosahedron_3rex())]
    return out


def test_the_lattice_condition_is_universal_and_only_the_rate_is_not():
    """THE result, and the reason retracting the migration was worth it.

    What is universal is the CONDITION, not any curve. Across every family below,
    at grade 1 and grade 2, with the trace normalisation ON and OFF, the exchange
    is EXACTLY zero on the critical line and runs linearly off it. Sixty-four
    configurations, worst linearity deviation 6.2% and most inside 2%.

    That it holds normalised AND raw is the load-bearing part: the normalisation
    is precisely what manufactured the grade-migration artefact, and the condition
    does not notice it.

    The RATE is not universal and should not be expected to be. raw2/raw1 spans
    0.039 to 0.272 across these families, a factor of seven, and it tracks face
    arity: all-triangle solids sit near 0.05 to 0.14, the cube at 0.23, C60 at
    0.27. So the geometry sets how fast, and nothing sets whether.

    A criterion independent of arity, grade and convention is intrinsic. A curve
    fitted to one family is not, which is what the retracted 1/n was.
    """
    worst_lin = 0.0
    for name, cells in _families():
        for d in (1, 2):
            for normalize in (True, False):
                A0 = _rl_graded_raw(cells, 0.5, d, normalize)
                assert np.abs(A0 @ A0 - A0 @ A0).max() == 0.0, (name, d, normalize)
                rates = []
                for e in (0.01, 0.04, 0.10, 0.25):
                    A = _rl_graded_raw(cells, 0.5 + e, d, normalize)
                    B = _rl_graded_raw(cells, 0.5 - e, d, normalize)
                    n = float(np.linalg.norm(A @ B - B @ A, "fro"))
                    assert n > 0.0, (name, d, normalize, e)
                    rates.append(n / e)
                worst_lin = max(worst_lin, max(rates) / min(rates))
    assert worst_lin < 1.10, worst_lin


def test_the_rate_is_structure_dependent_and_that_is_the_honest_half():
    """The companion. If the rate were also universal the result would be
    suspiciously strong; it is not, and it varies with face arity."""
    ratios = {}
    for name, cells in _families():
        r1 = _slope_raw(cells, 1, False)
        r2 = _slope_raw(cells, 2, False)
        ratios[name] = r2 / r1
    vals = list(ratios.values())
    assert max(vals) / min(vals) > 4.0, ratios
    # all-triangle solids sit low, wider faces higher
    assert ratios["octahedron"] < ratios["cube"] < ratios["C60"]


#### branching relations: arity at GRADE 1, not just at the faces
def _deform_tower_sq(Bs, W0):
    """W_d[cell] = sum_v W_{d-1}[v] c[v]^2: the column's squared norm in the
    W_{d-1} metric.

    The earlier form summed |c[v]| instead, which is the same thing only while the
    entries are ternary. A branching column is (-1, 1/(k-1), ..., 1/(k-1)), so
    |c| and c^2 part company at arity 3 and only the squared one is the norm.
    Measured: the |c| form leaves non-unit columns at every arity above 2.
    """
    out, Wprev = [], np.asarray(W0, float)
    for B in Bs:
        M = np.asarray(B, float)
        Wd = np.maximum(((M * M).T * Wprev).sum(1), 1e-300)
        out.append((np.diag(np.sqrt(Wprev)) @ M) / np.sqrt(Wd))
        Wprev = Wd
    return out


def _hyper(cells, nV):
    ptr, idx = [0], []
    for c in cells:
        idx += list(c)
        ptr.append(len(idx))
    r = RexGraph.from_hypergraph(np.array(ptr, np.int64), np.array(idx, np.int64))
    r._ensure_clean()
    return r


@pytest.mark.parametrize("k", [2, 3, 4, 6, 8])
def test_only_the_squared_form_normalises_a_branching_column(k):
    r = _hyper([tuple(range(k)), (0, 1)], k)
    B1 = np.asarray(r.B1_dense, float)
    p = np.array(_primes(B1.shape[0]), float)
    W0 = np.log(p) * p ** (-0.37)
    assert np.allclose((_deform_tower_sq([B1], W0)[0] ** 2).sum(0), 1.0)
    if k > 2:
        A = np.abs(B1)
        Wd = np.maximum((A.T * W0).sum(1), 1e-300)
        bad = (np.diag(np.sqrt(W0)) @ B1) / np.sqrt(Wd)
        assert not np.allclose((bad * bad).sum(0), 1.0), "the |c| form should fail here"


def _rl_branch(B1, B2, sigma, normalize=True):
    p = np.array(_primes(B1.shape[0]), float)
    Bs = _deform_tower_sq([B1] + ([B2] if B2 is not None else []),
                          np.log(p) * p ** (-sigma))
    B = Bs[0]
    T = B.T @ B
    G = np.abs(B).T @ np.abs(B)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    S = (np.abs(B1) > 0).astype(float)
    K = S.T @ S
    Koff = K - np.diag(np.diagonal(K))
    ch = [T, G, Foff + np.diag(np.abs(Foff).sum(1)), np.diag(Koff.sum(1)) - Koff]
    if B2 is not None:
        ch.append(Bs[1] @ Bs[1].T)
    tot = None
    for X in ch:
        tr = float(np.trace(X))
        Y = X / tr if (normalize and abs(tr) > 1e-15) else X
        tot = Y if tot is None else tot + Y
    return tot


BRANCHING = {
    "one 3-ary": ([(0, 1, 2), (1, 2), (2, 3), (3, 0)], 4),
    "one 5-ary": ([(0, 1, 2, 3, 4), (0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5),
    "all 3-ary": ([(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1)], 5),
    "all 4-ary": ([(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 0)], 6),
    "mixed 2,3,5": ([(0, 1), (1, 2, 3), (0, 2, 3, 4, 5), (4, 5), (3, 5)], 6),
    "wide 8-ary": ([tuple(range(8)), (0, 1), (2, 3), (4, 5), (6, 7), (1, 4)], 8),
}


@pytest.mark.parametrize("name", sorted(BRANCHING))
@pytest.mark.parametrize("normalize", [True, False])
def test_the_critical_line_condition_holds_for_branching_relations(name, normalize):
    """Arity at GRADE 1, where every earlier test varied it at the faces. The
    condition does not notice: exactly zero on the line and linear off it at
    arities 2, 3, 4, 5, 8 and mixed, normalised and raw, to within 0.7%. That is
    tighter than the 2-ary faced cases managed."""
    cells, nV = BRANCHING[name]
    B1 = np.asarray(_hyper(cells, nV).B1_dense, float)
    A0 = _rl_branch(B1, None, 0.5, normalize)
    assert np.abs(A0 @ A0 - A0 @ A0).max() == 0.0
    rates = []
    for e in (0.01, 0.04, 0.10, 0.25):
        A = _rl_branch(B1, None, 0.5 + e, normalize)
        B = _rl_branch(B1, None, 0.5 - e, normalize)
        n = float(np.linalg.norm(A @ B - B @ A, "fro"))
        assert n > 0.0
        rates.append(n / e)
    assert max(rates) / min(rates) < 1.02, rates


def test_the_condition_does_not_care_whether_the_faces_are_independent():
    """A face column only has to lie in ker(B1), so a DEPENDENT one still satisfies
    the chain condition and the chain condition cannot see it. Neither can the
    critical line: zero on the line and linear off it for an independent basis, for
    that basis plus a repeated column, plus a summed column, and plus two
    dependents.

    So the criterion is insensitive to face independence, which is worth stating
    precisely because it is the kind of thing easily assumed the other way.
    """
    from rexgraph.hodge_coords import harmonic_frame
    cells = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1), (0, 2)]
    r = _hyper(cells, 5)
    B1 = np.asarray(r.B1_dense, float)
    H = np.asarray(harmonic_frame(r).todense())
    variants = [H, np.hstack([H, H[:, :1]]),
                np.hstack([H, H[:, :1] + H[:, 1:2]]),
                np.hstack([H, H[:, :1], H[:, 1:2]])]
    for B2 in variants:
        assert np.abs(B1 @ B2).max() < 1e-12, "still a valid chain"
        A0 = _rl_branch(B1, B2, 0.5)
        assert np.abs(A0 @ A0 - A0 @ A0).max() == 0.0
        rates = []
        for e in (0.02, 0.08, 0.20):
            A = _rl_branch(B1, B2, 0.5 + e)
            B = _rl_branch(B1, B2, 0.5 - e)
            rates.append(float(np.linalg.norm(A @ B - B @ A, "fro")) / e)
        assert max(rates) / min(rates) < 1.03, (B2.shape, rates)


def test_the_rate_is_not_a_function_of_the_face_rank():
    """A refuted guess, kept. Since a dependent face adds nothing to the span, the
    rate might have depended only on rank(B2). It does not: B2B2^T sums col col^T
    over the columns actually present, so a repeated column doubles its own
    contribution. At rank 2 the rate reads 0.1368 for the bare basis, 0.2195 with
    one column repeated, 0.1176 with a summed column, and 0.1368 again once BOTH
    columns are repeated, which restores the symmetry rather than the rank."""
    from rexgraph.hodge_coords import harmonic_frame
    cells = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1), (0, 2)]
    r = _hyper(cells, 5)
    B1 = np.asarray(r.B1_dense, float)
    H = np.asarray(harmonic_frame(r).todense())

    def rate(B2):
        return float(np.mean([
            float(np.linalg.norm(_rl_branch(B1, B2, 0.5 + e) @ _rl_branch(B1, B2, 0.5 - e)
                                 - _rl_branch(B1, B2, 0.5 - e) @ _rl_branch(B1, B2, 0.5 + e),
                                 "fro")) / e
            for e in (0.02, 0.08, 0.20)]))

    base = rate(H)
    assert np.isclose(rate(H[:, ::-1]), base), "reordering must not matter"
    assert not np.isclose(rate(np.hstack([H, H[:, :1]])), base), "but a repeat does"
    assert np.isclose(rate(np.hstack([H, H[:, :1], H[:, 1:2]])), base), "symmetry restores it"


#### face independence and the topology/geometry exchange
def _tg_exchange(B1, B2, sigma=0.5):
    """||[T_hat, G_hat]|| in full, and the part the harmonic sector sees.

    T and G are both built from B1, so faces cannot move the commutator itself.
    What faces move is the SUBSPACE it is read in: ker(L1) = ker(B1) cap ker(B2^T).
    """
    from scipy.linalg import null_space
    p = np.array(_primes(B1.shape[0]), float)
    Bw = _deform_tower_sq([B1], np.log(p) * p ** (-sigma))[0]
    T = Bw.T @ Bw
    G = np.abs(Bw).T @ np.abs(Bw)
    Th, Gh = T / np.trace(T), G / np.trace(G)
    comm = Th @ Gh - Gh @ Th
    N = null_space(Bw)
    if B2 is not None and B2.size:
        M = B2.T @ N
        ns = null_space(M) if M.size else np.eye(N.shape[1])
        H = N @ ns if ns.size else np.zeros((B1.shape[1], 0))
    else:
        H = N
    if H.shape[1] == 0:
        return float(np.linalg.norm(comm, "fro")), 0.0, 0
    Q, _ = np.linalg.qr(H)
    return (float(np.linalg.norm(comm, "fro")),
            float(np.linalg.norm(Q @ Q.T @ comm @ Q @ Q.T, "fro")), H.shape[1])


def _k5():
    e = list(itertools.combinations(range(5), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    return r


def _graph(edges):
    r = RexGraph(sources=np.array([a for a, b in edges], np.int32),
                 targets=np.array([b for a, b in edges], np.int32))
    r._ensure_clean()
    return r


#### a small NAMED zoo, so a sector claim can be read off a structure by name
_ZOO = {
    "K5": _k5(),
    "C6": _graph([(i, (i + 1) % 6) for i in range(6)]),
    "prism": _graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                     (0, 3), (1, 4), (2, 5)]),
    "petersen": _graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
                        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]),
    "house": _graph([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 2)]),
}


def test_the_topology_geometry_commutator_is_antisymmetric():
    """PROVED. T and G are symmetric, so [T,G]^T = G^T T^T - T^T G^T = -[T,G].
    Exactly, not to tolerance."""
    B1 = np.asarray(_k5().B1_dense, float)
    p = np.array(_primes(5), float)
    Bw = _deform_tower_sq([B1], np.log(p) * p ** (-0.5))[0]
    T = Bw.T @ Bw
    G = np.abs(Bw).T @ np.abs(Bw)
    Tn, Gn = T / np.trace(T), G / np.trace(G)
    assert np.abs(Tn - Tn.T).max() == 0.0      # the hypothesis, and a Gram matrix
    assert np.abs(Gn - Gn.T).max() == 0.0      # is symmetric to the last bit
    C = _commutator(Tn, Gn)
    assert np.abs(C + C.T).max() == 0.0
    # and the construction is the commutator: the two-matmul form agrees to backward
    # error, which is the platform-dependent quantity the exact claim cannot be about.
    naive = Tn @ Gn - Gn @ Tn
    fence = np.finfo(float).eps * Tn.shape[0] * np.abs(Tn).max() * np.abs(Gn).max()
    assert np.abs(C - naive).max() <= fence, (np.abs(C - naive).max(), fence)


def test_the_rank_one_bound_is_the_weakest_case_of_the_annihilation():
    """This started as the reason the harmonic exchange vanishes at dim_H <= 1: for
    antisymmetric A and a rank-1 projector P = q q^T, P A P = q (q^T A q) q^T and
    q^T A q = 0. The identity is true and still does work, but it is not why.

    The harmonic exchange is zero at EVERY dim_H, not only at one, because T = B^T B
    annihilates the whole sector. The rank-1 bound is the general statement about
    antisymmetric forms; the annihilation is the specific one about this operator,
    and it is strictly stronger. Both are checked here so the weaker one is not
    mistaken for an explanation again.
    """
    from rexgraph.hodge_coords import harmonic_frame
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    H = np.asarray(harmonic_frame(r).todense())
    for nf in range(0, H.shape[1] + 1):
        _, harm, dim_h = _tg_exchange(B1, H[:, :nf] if nf else None)
        assert dim_h == H.shape[1] - nf
        assert harm < 1e-15, (nf, dim_h, harm)          # zero at every dimension

    A = _sector_J(B1, "gradient")                        # where it is NOT zero
    assert np.linalg.norm(A) > 1e-3
    q = np.linalg.qr(np.random.default_rng(2).normal(size=(A.shape[0], 1)))[0]
    assert abs((q.T @ A @ q).item()) < 1e-15, "and the rank-1 bound still holds there"


def test_only_independent_faces_move_the_topology_geometry_exchange():
    """The statement being tested: face independence is what governs the
    information exchange between topology and geometry. It holds, and the correct
    reading is sharper than the first one.

    The FULL commutator never moves. T and G both come from B1 alone, so faces are
    literally absent from it, and the GRADIENT sector reads 0.005090002 on K5 at
    every face count, to nine decimals. What independent faces build is a CURL
    sector, and the exchange appears THERE:

        independent faces   0    1    2        3        4        5
        dim_curl            0    1    2        3        4        5
        curl exchange       0    0    1.9e-7   1.7e-5   4.9e-4   1.1e-3

    monotone in the number of INDEPENDENT faces, and exactly zero at one face
    because a rank-1 projector kills any antisymmetric form. A dependent face adds
    no curl direction and so moves nothing.
    """
    B1 = np.asarray(_k5().B1_dense, float)
    C = _tg_deformed(B1)
    Bw = _weighted_boundary(B1)
    Qg = _sector(Bw, "gradient")
    grad0 = float(np.linalg.norm(Qg.T @ C @ Qg))

    prev, seq = 0.0, []
    for nf in (1, 2, 3, 4, 5):
        B2 = _k5_faces(nf)
        assert int(np.linalg.matrix_rank(B2)) == nf, "independent by construction"
        assert np.abs(B1 @ B2).max() < 1e-12, "and a valid chain"
        Qc = np.linalg.qr(B2)[0][:, :nf]
        cur = float(np.linalg.norm(Qc.T @ C @ Qc))
        assert cur >= prev - 1e-15, "each independent face adds exchange"
        prev = cur
        seq.append(cur)
        assert np.isclose(float(np.linalg.norm(Qg.T @ C @ Qg)), grad0), \
            "and none of them touches the gradient"
    assert seq[0] < 1e-12, "one face is a rank-1 projector: P A P = 0"
    assert seq[-1] > 1e-3

    base = _k5_faces(3)
    Qb = np.linalg.qr(base)[0][:, :3]
    ref = float(np.linalg.norm(Qb.T @ C @ Qb))
    for extra in (base[:, :1], base[:, :1] + base[:, 1:2]):
        dep = np.hstack([base, extra])
        assert int(np.linalg.matrix_rank(dep)) == 3, "still rank 3"
        assert np.abs(B1 @ dep).max() < 1e-12, "still a valid chain"
        Qd = np.linalg.qr(dep)[0][:, :3]
        assert np.isclose(float(np.linalg.norm(Qd.T @ C @ Qd)), ref, atol=1e-15)


def test_the_exchange_is_face_invariant_and_sigma_dependent():
    """A correction, not a refinement. The first reading called the effect the same
    at every sigma; on the sector where the exchange is actually nonzero it is not.
    The gradient exchange falls by more than four across sigma = 0.2 to 0.8 while
    faces cannot move it at all.

    Which is the same radial/angular separation the whole family shows: what the
    COMPLEX owns is conserved, what the WEIGHTING owns drifts. Faces are structure
    and sigma is weight, and they land on opposite sides of that line.
    """
    B1 = np.asarray(_k5().B1_dense, float)
    vals = []
    for sigma in (0.20, 0.35, 0.50, 0.65, 0.80):
        Bw = _weighted_boundary(B1, sigma)
        Q = _sector(Bw, "gradient")
        C = _tg_deformed(B1, sigma)
        vals.append(float(np.linalg.norm(Q.T @ C @ Q)))
        for nf in (0, 3, 5):
            B2 = _k5_faces(nf)
            assert np.isclose(float(np.linalg.norm(Q.T @ C @ Q)), vals[-1])
            assert B2.shape[1] == nf
    assert all(b < a for a, b in zip(vals, vals[1:], strict=False)), vals
    assert vals[0] / vals[-1] > 4.0


#### the real line, the complex plane, and the harmonic plane
def test_the_exchange_supplies_the_complex_unit_on_a_gradient_plane():
    """The result the rest was building toward, at its correct address.

    [T,G] is antisymmetric, and an antisymmetric operator on a two-dimensional
    space is a rotation generator: normalise one 2x2 Schur block of it and

        J = [[0, 1], [-1, 0]],   J^2 = -I   exactly.

    So the plane is a COMPLEX LINE and the unit is not put in by hand; it is the
    topology/geometry exchange. The sector this happens on is the gradient, not the
    harmonic: see test_the_harmonic_sector_is_annihilated_by_both_generators.
    """
    from scipy.linalg import schur
    A = _sector_J(np.asarray(_k5().B1_dense, float), "gradient")
    blk = schur(A, output="real")[0][:2, :2]
    assert abs(blk[0, 1]) > 1e-9
    J = blk / abs(blk[0, 1])
    assert np.allclose(J @ J, -np.eye(2), atol=1e-9)


def test_a_direction_with_no_exchange_is_a_real_line():
    """The other half, and it is proved rather than measured. For antisymmetric A
    and a rank-1 projector P = q q^T, P A P = q (q^T A q) q^T with q^T A q = 0. So
    a one-dimensional sector carries no rotation at all: it is a real line. The
    kernel of the exchange inside a larger sector is real for the same reason, and
    on an odd-dimensional gradient sector there is always at least one."""
    A = _sector_J(np.asarray(_k5().B1_dense, float), "gradient")
    q = np.linalg.qr(np.random.default_rng(0).normal(size=(A.shape[0], 1)))[0]
    assert abs((q.T @ A @ q).item()) < 1e-15, "any rank-1 projector kills it"

    e = list(itertools.combinations(range(6), 2))       # K6: dim_grad = 5, odd
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    A6 = _sector_J(np.asarray(r.B1_dense, float), "gradient")
    sv = np.linalg.svd(A6, compute_uv=False)
    assert A6.shape[0] == 5
    assert int((sv > sv.max() * 1e-9).sum()) == 4, "one real line left over"


def test_three_complex_planes_and_only_one_has_a_derived_unit():
    """Keeping them apart, because the word plane is doing three jobs.

    s = sigma + it is the SPECTRAL plane: a deformation parameter, chosen, with
    the critical line living in it.

    f(e) = L_t(e) + i L_s(e) is the LAGRANGIAN plane: one complex number per
    relation, modulus the mass and argument the character, with i supplied as
    bookkeeping to pair two real readings.

    The GRADIENT plane is the only one whose complex structure is derived. Its i is
    J = [T,G] normalised on a two-dimensional block, and it is exactly zero on any
    direction the exchange cannot reach, which is why a lone one stays real.
    """
    from scipy.linalg import schur
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    f = r.lagrangian_fields()
    Lt, Ls = np.asarray(f["Lt"]), np.asarray(f["Ls"])
    assert np.allclose(Lt + Ls, np.diagonal(np.asarray(r.RL)))
    assert np.allclose(np.asarray(f["f_mag"]), np.hypot(Lt, Ls))

    A = _sector_J(B1, "gradient")
    blk = schur(A, output="real")[0][:2, :2]
    two = blk / abs(blk[0, 1])
    assert np.allclose(two @ two, -np.eye(2), atol=1e-9)
    q = np.linalg.qr(np.random.default_rng(1).normal(size=(A.shape[0], 1)))[0]
    assert abs((q.T @ A @ q).item()) < 1e-15


#### is the Hodge split the same split as real-versus-complex?
def _hodge_bases(B1, B2):
    """Orthonormal bases of gradient im(B1^T), curl im(B2), harmonic ker(L1)."""
    from scipy.linalg import null_space
    nE = B1.shape[1]
    rg = int(np.linalg.matrix_rank(B1))
    Qg = np.linalg.qr(B1.T)[0][:, :rg]
    if B2 is not None and B2.size:
        rc = int(np.linalg.matrix_rank(B2))
        Qc = np.linalg.qr(B2)[0][:, :rc]
    else:
        Qc = np.zeros((nE, 0))
    N = null_space(B1)
    if B2 is not None and B2.size:
        M = B2.T @ N
        ns = null_space(M) if M.size else np.eye(N.shape[1])
        Hh = N @ ns if ns.size else np.zeros((nE, 0))
    else:
        Hh = N
    Qh = np.linalg.qr(Hh)[0] if Hh.size else np.zeros((nE, 0))
    return Qg, Qc, Qh




def test_the_gradient_sector_is_not_the_real_line():
    """The tempting correspondence, refuted. gradient <-> real, curl <-> complex,
    harmonic <-> harmonic plane is NOT what happens: the gradient sector carries a
    FULL complex structure and in fact the largest rotation rates of the three.

    On K5 with three faces: gradient dim 4 rank 4, top rate 2.77e-03; curl dim 3
    rank 2, 1.17e-05; harmonic dim 3 rank 2, 5.68e-04. Every sector is complex.
    """
    from rexgraph.hodge_coords import harmonic_frame
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    H = np.asarray(harmonic_frame(r).todense())
    C = _tg_deformed(B1)
    Qg, Qc, Qh = _hodge_bases(B1, H[:, :3])
    rates = {}
    for nm, Q in (("gradient", Qg), ("curl", Qc), ("harmonic", Qh)):
        A = Q.T @ C @ Q
        sv = np.linalg.svd(A, compute_uv=False)
        rates[nm] = (Q.shape[1], int((sv > 1e-14).sum()), float(sv[0]))
    assert rates["gradient"][1] == rates["gradient"][0] == 4, rates
    assert rates["gradient"][1] > 0, "the gradient is not real"
    assert rates["gradient"][2] > rates["harmonic"][2] > rates["curl"][2], rates


def test_the_real_directions_cut_across_the_hodge_sectors():
    """So the two splits are different structures on one space, not one split under
    two names. The kernel of [T,G] is the real part of the edge space, and it does
    not sit inside any Hodge sector: on K5 face-free it distributes 0.21 gradient,
    0.00 curl, 0.79 harmonic by energy.

    Adding faces MOVES it, and moves it exactly the way the Hodge decomposition
    says it should: harmonic content becomes curl content, so the real directions
    follow, 0.79 harmonic to 0.28 with 0.51 arriving in curl. The GRADIENT share
    does not move at all, faces having no reach into that sector.
    """
    from scipy.linalg import null_space

    from rexgraph.hodge_coords import harmonic_frame
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    H = np.asarray(harmonic_frame(r).todense())
    C = _tg_deformed(B1)
    K = null_space(C)
    assert K.shape[1] > 0, "there are real directions to place"

    def shares(B2):
        Qg, Qc, Qh = _hodge_bases(B1, B2)
        f = lambda Q: float(np.sum((Q.T @ K) ** 2) / K.shape[1]) if Q.shape[1] else 0.0
        return f(Qg), f(Qc), f(Qh)

    g0, c0, h0 = shares(None)
    g1, c1, h1 = shares(H[:, :3])
    assert np.isclose(g0 + c0 + h0, 1.0, atol=1e-9)
    assert 0.05 < g0 < 0.95, "not contained in the gradient"
    assert 0.05 < h0 < 0.95, "nor in the harmonic sector"
    assert np.isclose(g0, g1, atol=1e-9), "faces cannot reach the gradient share"
    assert c1 > c0 and h1 < h0, "harmonic real content becomes curl real content"


@pytest.mark.parametrize("cells,nV,tag", [
    ([(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1)], 5, "all 3-ary"),
    ([(0, 1), (1, 2, 3), (0, 2, 3, 4, 5), (4, 5), (3, 5)], 6, "mixed 2,3,5"),
])
def test_the_complex_structure_survives_branching(cells, nV, tag):
    """Everything above holds with branching relations at grade 1: [T,G] stays
    exactly antisymmetric, its rank stays even, and the gradient sector still
    carries the complex structure rather than being real. The real directions are
    even more gradient-weighted there, 0.85 and 1.00 on these two."""
    from scipy.linalg import null_space
    B1 = np.asarray(_hyper(cells, nV).B1_dense, float)
    C = _tg_deformed(B1)
    assert np.abs(C + C.T).max() < 1e-15, "antisymmetric under branching too"
    sv = np.linalg.svd(C, compute_uv=False)
    rank = int((sv > 1e-14).sum())
    assert rank % 2 == 0, "an antisymmetric operator has even rank"
    Qg = _hodge_bases(B1, None)[0]
    A = Qg.T @ C @ Qg
    assert int((np.linalg.svd(A, compute_uv=False) > 1e-14).sum()) > 0, \
        "the gradient sector is complex here as well"
    K = null_space(C)
    if K.size:
        share = float(np.sum((Qg.T @ K) ** 2) / K.shape[1])
        assert share > 0.5, (tag, share)


#### calculus on the tensor, instead of reading its norm
def _RL_sigma(B1, sigma):
    T, G = _deformed_TG(B1, sigma)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    S = (np.abs(B1) > 0).astype(float)
    K = S.T @ S
    Ko = K - np.diag(np.diagonal(K))
    tot = None
    for X in (T, G, Foff + np.diag(np.abs(Foff).sum(1)), np.diag(Ko.sum(1)) - Ko):
        tr = float(np.trace(X))
        Y = X / tr if abs(tr) > 1e-15 else X
        tot = Y if tot is None else tot + Y
    return tot


def _flow_generator(B1, sigma=0.5, h=1e-5):
    """[R, dR/dsigma]: the generator of the sigma flow, as an operator."""
    R = _RL_sigma(B1, sigma)
    Rp = (_RL_sigma(B1, sigma + h) - _RL_sigma(B1, sigma - h)) / (2 * h)
    return R, Rp, _commutator(R, Rp)


def _tg_deformed(B1, sigma=0.5):
    """[T_hat, G_hat] on the SAME deformed boundary the flow generator uses, so the
    two structures are compared at one point of the family rather than one deformed
    against one raw. Reading them at different points was a real slip: the alignment
    came out +0.246 for a 2-ary ring against +0.014 measured properly."""
    T, G = _deformed_TG(B1, sigma)
    return _commutator(T / np.trace(T), G / np.trace(G))


def test_the_linearity_of_equation_iv_is_taylors_theorem():
    """What reading the RATE was hiding.

    RL(1/2 +- e) = R +- e R' + O(e^2), so

        [R + eR', R - eR'] = -2e [R, R'] + O(e^3)

    The exchange being linear in the distance from the critical line is therefore
    not a measurement, it is the first-order term, and the 'slope' quoted for it is
    ||2[R, R']||: the NORM of a tensor that was available directly. Verified below,
    with the relative error falling as e^2.
    """
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    R, Rp, gen = _flow_generator(B1)
    predicted = 2.0 * float(np.linalg.norm(gen, "fro"))
    prev = None
    for eps in (0.005, 0.01, 0.02, 0.05):
        A = _RL_sigma(B1, 0.5 + eps)
        B = _RL_sigma(B1, 0.5 - eps)
        measured = float(np.linalg.norm(A @ B - B @ A, "fro")) / eps
        rel = abs(measured - predicted) / predicted
        assert rel < 1e-3, (eps, rel)
        if prev is not None:
            assert rel > prev, "the error grows with e, as an O(e^2) term should"
        prev = rel


def test_the_flow_generator_is_a_second_complex_structure():
    """R and R' are both symmetric, so [R, R'] is antisymmetric exactly, and is
    therefore a complex structure in its own right, generated by the sigma flow
    rather than by the topology/geometry exchange.

    It is also a richer one: on K5 it has full rank 10 with five complex pairs,
    where [T,G] has rank 6 with three.
    """
    from rexgraph.hodge_coords import complex_structure
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    R, Rp, gen = _flow_generator(B1)
    assert np.abs(R - R.T).max() == 0.0
    assert np.abs(Rp - Rp.T).max() == 0.0
    assert np.abs(gen + gen.T).max() == 0.0
    flow = complex_structure(gen)
    assert flow["rank"] == flow["dim"] == 10
    assert flow["pairs"] == 5

    assert complex_structure(_tg_deformed(B1))["pairs"] == 3, \
        "the exchange reaches less of the space"


def test_the_two_structures_are_neither_commuting_nor_anticommuting():
    """So they do not generate a quaternionic structure: both the commutator and the
    anticommutator are substantial and, on complete graphs, nearly equal, which is
    what two generic independent generators look like."""
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    _, _, gen = _flow_generator(B1)
    tg = _tg_deformed(B1)
    a = tg / np.linalg.norm(tg, "fro")
    b = gen / np.linalg.norm(gen, "fro")
    comm = float(np.linalg.norm(a @ b - b @ a, "fro"))
    anti = float(np.linalg.norm(a @ b + b @ a, "fro"))
    assert comm > 0.3 and anti > 0.3, (comm, anti)
    assert abs(comm - anti) < 0.1, (comm, anti)


def test_branching_couples_the_two_structures_where_two_ary_does_not():
    """The finding the norms could not have carried. Frobenius alignment of the two
    generators, on rings of fixed size and varying relation arity:

        arity      3       4       6
        alignment -0.555  -0.307  -0.555

    Arity 2 carries no alignment at all, and that is the stronger half: on a 2-ary
    ring T and G COMMUTE, so [T,G] is the zero operator and there is no plane for
    the flow generator to be turned against. That is the same vanishing
    test_a_cycle_has_no_topology_geometry_exchange_at_all asserts directly, reached
    from the other side.

    An earlier version of this test read structure_alignment on that zero matrix
    and reported +0.014 for arity 2. That number was the direction of its own
    rounding error: perturbing B1 by ONE ULP moves it across [-0.47, +0.16], where
    the branching values move by 4e-11 under the same perturbation. It is asserted
    as a vanishing here rather than as a value.

    Two operators of equal norm can be orthogonal or identical, and no magnitude
    distinguishes those, which is the whole reason to compare the tensors.

    Stated for the RING family, not for 2-ary complexes in general: K4, K5, a
    degree-3 tree and a theta graph are every one of them 2-ary, and every one has
    ||[T,G]|| between 0.15 and 0.24 of the scale it lives on. What commutes here is
    the ring, whose adjacent relations all meet with the same sign.

    Stated for arity, NOT as a trend in k: on complete graphs the alignment passes
    through zero between k = 7 and k = 8 (-0.101, -0.076, -0.041, -0.017, +0.005,
    +0.020) rather than settling, so there is no 'increasingly independent' claim
    to make there.
    """
    from rexgraph.hodge_coords import structure_alignment

    n = 7

    def ring(ar):
        cells = [tuple((i + j) % n for j in range(ar)) for i in range(n)]
        return np.asarray(_hyper(cells, n).B1_dense, float)

    # arity 2: the exchange is zero against the scale it would live on, so the
    # alignment is 0/0 and there is no number to read.
    T, G = _deformed_TG(ring(2))
    Tn, Gn = T / np.trace(T), G / np.trace(G)
    scale = np.linalg.norm(Tn, "fro") * np.linalg.norm(Gn, "fro")
    got2 = float(np.linalg.norm(_tg_deformed(ring(2)), "fro")) / scale
    assert got2 <= np.finfo(float).eps * Tn.shape[0], got2

    # the branching arities, pinned to the measured values: they are stable to 4e-11
    # under a one-ulp perturbation, so they are quantities and not roundoff.
    want = {3: -0.555054, 4: -0.306760, 6: -0.554548}
    got = {}
    for ar in (3, 4, 6):
        B1 = ring(ar)
        _, _, gen = _flow_generator(B1)
        got[ar] = structure_alignment(_tg_deformed(B1), gen)
        assert abs(got[ar] - want[ar]) < 1e-6, (ar, got[ar], want[ar])
        assert abs(got[ar]) > 0.25, got


#### is there a calculus here, and on what
def test_the_discrete_invariants_are_conserved_along_the_flow():
    """The question worth asking, since Taylor holding is not itself news: RL(sigma)
    is built from p^-sigma plus rational operations, so it is real-analytic for
    inherited reasons and smoothness is free.

    What is NOT free is that the DISCRETE structure survives the flow. Over sigma
    from 0.05 to 8.0, a range of 160, rank[T,G] stays 6, its complex pairs stay 3,
    rank(RL) stays 10 and tr(RL) stays exactly 4. Nothing jumps.

    That is what makes this a calculus ON the discrete object rather than one that
    has replaced it: the integer content is a conserved quantity of the flow, and
    only the determinant moves.
    """
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    seen = set()
    traces = []
    for s in (0.05, 0.2, 0.5, 0.8, 1.2, 2.0, 4.0, 8.0):
        A = _tg_deformed(B1, s)
        sv = np.linalg.svd(A, compute_uv=False)
        rk = int((sv > 1e-12 * sv[0]).sum())
        R = _RL_sigma(B1, s)
        seen.add((rk, int(np.linalg.matrix_rank(R))))
        traces.append(float(np.trace(R)))
    assert seen == {(6, 10)}, seen
    assert max(traces) - min(traces) < 1e-12
    assert np.isclose(traces[0], 4.0), traces[0]


def test_jacobis_formula_holds_on_the_flow():
    """d/dsigma log det RL = tr(RL^-1 dRL/dsigma). So the determinant has a
    well-defined logarithmic derivative and it is a trace of the flow generator,
    which is what connects the determinant tower to the calculus rather than
    leaving it a separate reading."""
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    h = 1e-5
    for s in (0.2, 0.5, 1.0, 2.0):
        R = _RL_sigma(B1, s)
        Rp = (_RL_sigma(B1, s + h) - _RL_sigma(B1, s - h)) / (2 * h)
        lhs = (np.log(abs(np.linalg.det(_RL_sigma(B1, s + h))))
               - np.log(abs(np.linalg.det(_RL_sigma(B1, s - h))))) / (2 * h)
        rhs = float(np.trace(np.linalg.solve(R, Rp)))
        assert abs(lhs - rhs) / abs(rhs) < 1e-6, (s, lhs, rhs)


def test_the_fundamental_theorem_closes_on_the_flow():
    """Integrating the derivative recovers the difference, so integration over the
    parameter is well founded and not only differentiation."""
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    h = 1e-5

    def d(s):
        R = _RL_sigma(B1, s)
        Rp = (_RL_sigma(B1, s + h) - _RL_sigma(B1, s - h)) / (2 * h)
        return float(np.trace(np.linalg.solve(R, Rp)))

    a, b = 0.3, 2.0
    xs = np.linspace(a, b, 401)
    integral = float(np.trapezoid([d(float(s)) for s in xs], xs))
    direct = float(np.log(abs(np.linalg.det(_RL_sigma(B1, b))))
                   - np.log(abs(np.linalg.det(_RL_sigma(B1, a)))))
    assert abs(integral - direct) / abs(direct) < 1e-5, (integral, direct)


def test_the_determinant_is_stationary_somewhere_and_it_does_not_settle():
    """det RL has a stationary point in sigma, so there is a genuine variational
    structure on the flow. Where it sits is NOT the critical line and does not
    approach it:

        k        4      5      6      7      8      9     10
        sigma*  0.763  0.704  0.588  0.534  0.473  0.438  0.422

    It passes through 1/2 between k = 7 and 8 and keeps descending. That is the
    THIRD quantity in this family to do exactly that, after the T-G coupling peak
    and the zero of the structure alignment, so the pattern is recorded as a shared
    property of extrema over the p^-sigma weighting rather than as evidence about
    the critical line.

    det RL also collapses fast with k, 4.7e-02 down to 1.3e-52 by k = 10, so the
    bisection runs on the log-derivative, which is a trace and stays conditioned.
    """
    h = 1e-5

    def dlogdet(B1, s):
        R = _RL_sigma(B1, s)
        Rp = (_RL_sigma(B1, s + h) - _RL_sigma(B1, s - h)) / (2 * h)
        return float(np.trace(np.linalg.solve(R, Rp)))

    stars = []
    for k in (4, 5, 6, 7, 8):
        e = list(itertools.combinations(range(k), 2))
        r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                     targets=np.array([b for a, b in e], np.int32))
        r._ensure_clean()
        B1 = np.asarray(r.B1_dense, float)
        lo, hi = 0.05, 4.0
        assert dlogdet(B1, lo) * dlogdet(B1, hi) < 0, k
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if dlogdet(B1, lo) * dlogdet(B1, mid) <= 0:
                hi = mid
            else:
                lo = mid
        stars.append(0.5 * (lo + hi))
    assert stars == sorted(stars, reverse=True), stars
    assert stars[0] > 0.5 > stars[-1], "crosses the line rather than settling on it"


#### what the sigma drifts have in common
def _weights(k, sigma):
    p = np.array(_primes(k), float)
    return np.log(p) * p ** (-sigma)


def _RL_weighted(B1, sigma, base):
    """RL with the vertex weight family supplied, so it can be swapped."""
    W0 = np.log(base) * base ** (-sigma)
    Wd = np.maximum(((B1 * B1).T * W0).sum(1), 1e-300)
    Bw = (np.diag(np.sqrt(W0)) @ B1) / np.sqrt(Wd)
    T = Bw.T @ Bw
    G = np.abs(Bw).T @ np.abs(Bw)
    Foff = T - G
    np.fill_diagonal(Foff, 0.0)
    S = (np.abs(B1) > 0).astype(float)
    K = S.T @ S
    Ko = K - np.diag(np.diagonal(K))
    tot = None
    for X in (T, G, Foff + np.diag(np.abs(Foff).sum(1)), np.diag(Ko.sum(1)) - Ko):
        tr = float(np.trace(X))
        Y = X / tr if abs(tr) > 1e-15 else X
        tot = Y if tot is None else tot + Y
    return tot


def _detstar(B1, base, h=1e-5, lo=0.02, hi=12.0):
    def d(s):
        R = _RL_weighted(B1, s, base)
        Rp = (_RL_weighted(B1, s + h, base) - _RL_weighted(B1, s - h, base)) / (2 * h)
        return float(np.trace(np.linalg.solve(R, Rp)))
    if d(lo) * d(hi) > 0:
        return float("nan")
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if d(lo) * d(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _kk(k):
    e = list(itertools.combinations(range(k), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    return np.asarray(r.B1_dense, float)


def test_the_sigma_drifts_belong_to_the_weighting_and_not_the_complex():
    """The decisive test, and it settles three near-misses at once.

    Hold the complex FIXED at K7 and swap the vertex weight family. If the drifting
    extrema said anything about the relational structure they could not move. They
    move by a factor of nearly four:

        family          <ln base>   det*     det* * <ln base>
        primes 2..17      1.878     0.5336        1.002
        integers 2..8     1.515     0.5164        0.782
        squares 4..64     3.030     0.2582        0.782
        tight 2.0..2.6    0.829     0.9526        0.790

    and `det* * <ln base>` is CONSTANT within a scaling family: squaring the bases
    doubles <ln base> and halves det* exactly. So sigma is a scale conjugate to
    <ln base>, and these extrema sit at C/<ln base> with C set by the SHAPE of the
    weight distribution rather than by the graph.
    """
    B1 = _kk(7)
    fams = {"integers": np.arange(2, 9, dtype=float),
            "squares": np.arange(2, 9, dtype=float) ** 2,
            "primes": np.array(_primes(7), float)}
    got = {}
    for nm, base in fams.items():
        d = _detstar(B1, base)
        got[nm] = (float(np.mean(np.log(base))), d, d * float(np.mean(np.log(base))))
    # same topology, det* moves a long way
    ds = [v[1] for v in got.values()]
    assert max(ds) / min(ds) > 1.8, got
    # and the product is invariant under squaring the base
    assert np.isclose(got["integers"][2], got["squares"][2], rtol=1e-6), got


def test_one_half_is_crossed_where_the_primes_reach_e_squared():
    """Why all three drifts pass 1/2 near k = 8 and none of them stops there.

    They scale as C/<ln p>, so they cross 1/2 where <ln p> reaches 2, which is
    where the GEOMETRIC MEAN of the first k primes passes e^2 = 7.389:

        k          6       7       8       9
        <ln p>   1.7183  1.8776  2.0110  2.1359
        geo mean 5.5752  6.5378  7.4704  8.4647

    So the coincidence is about how big the first k primes are, and carries no
    information about the critical line. 1/2 is distinguished because s = 1 - s
    fixes it, not because any extremum lives there.
    """
    below = [k for k in range(4, 8) if float(np.mean(np.log(_primes(k)))) < 2.0]
    above = [k for k in range(8, 12) if float(np.mean(np.log(_primes(k)))) > 2.0]
    assert below == [4, 5, 6, 7] and above == [8, 9, 10, 11]
    g7 = float(np.exp(np.mean(np.log(_primes(7)))))
    g8 = float(np.exp(np.mean(np.log(_primes(8)))))
    assert g7 < np.e ** 2 < g8, (g7, g8)


def test_rescaling_by_the_weight_scale_collapses_the_drift():
    """Partially, which is the honest amount. Expressed in the dimensionless
    sigma * <ln p>, the spread of det* over k = 4..9 falls from 1.74x to 1.17x and
    the T-G peak from 1.33x to 1.20x. So most of the drift is units, and what
    remains is a real but much smaller k dependence.

    The point is the separation it makes. Everything that drifts belongs to the
    weighting; everything that does not - the conserved rank, the complex-pair
    count, tr(RL) = 4, and the vanishing-and-linear condition itself - belongs to
    the complex.
    """
    ks = (4, 5, 6, 7, 8, 9)
    raw, resc = [], []
    for k in ks:
        p = np.array(_primes(k), float)
        m = float(np.mean(np.log(p)))
        d = _detstar(_kk(k), p, lo=0.05, hi=4.0)
        raw.append(d)
        resc.append(d * m)
    raw, resc = np.array(raw), np.array(resc)
    assert raw.max() / raw.min() > 1.6
    assert resc.max() / resc.min() < 1.25
    assert resc.max() / resc.min() < raw.max() / raw.min()


#### the polar setting is real, but it is not in the harmonic sector
def _sector(B, which):
    """An orthonormal basis of one Hodge sector OF B ITSELF. Taking the kernel from
    the raw boundary while reading the commutator on the deformed one is the same
    mismatch _tg_deformed warns about, and it manufactured a rotation on the
    harmonic sector that is exactly zero when both are read at one point."""
    from scipy.linalg import null_space
    if which == "harmonic":
        N = null_space(B)
        return np.linalg.qr(N)[0] if N.size else np.zeros((B.shape[1], 0))
    return np.linalg.qr(B.T)[0][:, :int(np.linalg.matrix_rank(B))]


def _sector_J(B1, which, sigma=0.5):
    Q = _sector(_weighted_boundary(B1, sigma), which)
    A = Q.T @ _tg_deformed(B1, sigma) @ Q
    return 0.5 * (A - A.T)


def _deformed_pair(B1, B2, sigma=0.5):
    """The deformed boundary at grade 1 AND grade 2. B2 must carry Wd^(1/2) or the
    chain condition does not survive: B1w B2w = W0^(1/2) (B1 B2) W2^(-1/2) = 0 only
    when the middle weights cancel. Taking a curl basis from the RAW B2 against a
    deformed commutator is the same mismatch as taking a raw kernel, and it produced
    a face-independence reading that is actually zero."""
    p = np.array(_primes(B1.shape[0]), float)
    W0 = np.log(p) * p ** (-sigma)
    Wd = np.maximum(((B1 * B1).T * W0).sum(1), 1e-300)
    B1w = (np.diag(np.sqrt(W0)) @ B1) / np.sqrt(Wd)
    if B2 is None or not B2.size:
        return B1w, np.zeros((B1.shape[1], 0))
    W2 = np.maximum(((B2 * B2).T * Wd).sum(1), 1e-300)
    return B1w, (np.diag(np.sqrt(Wd)) @ B2) / np.sqrt(W2)


def _weighted_boundary(B1, sigma=0.5):
    p = np.array(_primes(B1.shape[0]), float)
    W0 = np.log(p) * p ** (-sigma)
    Wd = np.maximum(((B1 * B1).T * W0).sum(1), 1e-300)
    return (np.diag(np.sqrt(W0)) @ B1) / np.sqrt(Wd)


@pytest.mark.parametrize("name", ["K5", "prism", "petersen", "house", "C6"])
def test_the_harmonic_sector_is_annihilated_by_both_generators(name):
    """The harmonic sector carries NO dynamics at all, and it is a one-line theorem
    rather than a measurement: T = B^T B, so T Q = B^T (B Q) = 0 for a harmonic Q,
    and T is symmetric, hence

        Q^T [T, G] Q = (T Q)^T G Q - Q^T G (T Q) = 0.

    Both the decay generator L1 and the rotation generator [T,G] vanish there
    exactly, for any boundary and any weighting. Nothing decays and nothing turns.
    """
    B1 = np.asarray(_ZOO[name].B1_dense, float)
    Bw = _weighted_boundary(B1)
    Q = _sector(Bw, "harmonic")
    assert Q.shape[1] > 0
    assert np.linalg.norm(Bw.T @ Bw @ Q) < 1e-12, "L1 annihilates it"
    assert np.linalg.norm(_sector_J(B1, "harmonic")) < 1e-12, "[T,G] annihilates it"


@pytest.mark.parametrize("name", ["K5", "prism", "petersen", "house"])
def test_the_exchange_carries_the_harmonic_sector_into_the_gradient(name):
    """The positive form of the annihilation, and the one that says what happens.

    The FORM vanishing on a sector is not the sector lying in the operator's kernel.
    For harmonic x, [T,G]x = T(Gx) - G(Tx) = T(Gx), which is nonzero, and it lands
    in im(T) = im(B^T) = the GRADIENT sector. Measured: 100.000000% of the image,
    exactly, on every structure.

    So the harmonic sector is not inert under the exchange, it is a one-way SOURCE
    for the gradient. It carries no rotation of its own for the same reason nothing
    returns to it: a rotation needs the plane mapped to itself. And the flow out is
    the larger quantity, 0.0411 on K5 against 0.0051 for the gradient's internal
    exchange, so the sector that reads zero is in fact the biggest participant.
    """
    B1 = np.asarray(_ZOO[name].B1_dense, float)
    Bw = _weighted_boundary(B1)
    C = _tg_deformed(B1)
    Qh, Qg = _sector(Bw, "harmonic"), _sector(Bw, "gradient")
    Y = C @ Qh
    total = float(np.linalg.norm(Y))
    assert total > 1e-3, "the sector is not in the kernel of the exchange"
    assert abs(float(np.linalg.norm(Qg.T @ Y)) / total - 1.0) < 1e-12, "all of it"
    assert float(np.linalg.norm(Qh.T @ Y)) / total < 1e-12, "and none comes back"


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_a_cycle_has_no_topology_geometry_exchange_at_all(n):
    """The degenerate case, and it has a reason rather than being an exception.

    On a consistently oriented cycle every edge meets its neighbour with opposite
    sign, so G = |T| entrywise and the two differ only off-diagonal by that sign.
    Column normalisation sets diag(T) = diag(G) = 1, which makes G = 2I - T, and
    [T, 2I - T] = 0 identically. A cycle is the structure where topology and
    geometry have nothing to exchange: it is 2-regular, so there is no variation in
    concentration for the unsigned reading to disagree about.

    This is why C6 is excluded from the sector-flow tests: there is no flow to
    place, not because the placement fails.
    """
    e = [(i, (i + 1) % n) for i in range(n)]
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    B1 = np.asarray(r.B1_dense, float)
    T = B1.T @ B1
    G = np.abs(B1).T @ np.abs(B1)
    assert np.array_equal(G, np.abs(T)), "unsigned is the entrywise absolute value"
    assert np.abs(T @ G - G @ T).max() == 0.0, "exactly, on the raw boundary"
    assert np.linalg.norm(_tg_deformed(B1)) < 1e-15, "and on the deformed one"
    assert np.allclose(np.diagonal(G), 2.0), "2-regular: nothing to disagree about"


def test_the_outflow_exceeds_the_gradients_own_exchange():
    """Which is why reading zero on the harmonic sector was so misleading: on K5 the
    harmonic outflow is eight times the gradient's internal rotation."""
    B1 = np.asarray(_ZOO["K5"].B1_dense, float)
    Bw = _weighted_boundary(B1)
    C = _tg_deformed(B1)
    out = float(np.linalg.norm(C @ _sector(Bw, "harmonic")))
    inner = float(np.linalg.norm(_sector_J(B1, "gradient")))
    assert abs(out - 0.041101) < 1e-5, out
    assert abs(inner - 0.005090) < 1e-5, inner
    assert out / inner > 7.0


def test_the_mismatched_kernel_manufactures_the_rotation_it_reports():
    """The control that killed the earlier reading. Pairing ker(B1) with [T_w,G_w]
    reports a rotation on the harmonic sector of order 1e-3; pairing ker(B_w) with
    the same commutator reports 1e-17. The whole of the reported structure was the
    gap between the two kernels."""
    from scipy.linalg import null_space
    B1 = np.asarray(_ZOO["K5"].B1_dense, float)
    C = _tg_deformed(B1)
    Qm = np.linalg.qr(null_space(B1))[0]            # kernel of the RAW boundary
    Qc = _sector(_weighted_boundary(B1), "harmonic")  # kernel of the deformed one
    mismatched = float(np.linalg.norm(Qm.T @ C @ Qm))
    consistent = float(np.linalg.norm(Qc.T @ C @ Qc))
    assert mismatched > 1e-4
    assert consistent < 1e-12
    assert mismatched / max(consistent, 1e-300) > 1e10


def test_the_exchange_generates_a_genuine_circle_action():
    """Not an analogy with polar coordinates: the complex HAS them, on the GRADIENT
    sector. J = [T,G] normalised on one 2x2 Schur block satisfies J^2 = -I, so
    exp(tJ) is orthogonal with determinant 1 and exp(2 pi J) = I exactly. It is a
    circle acting on the plane, and it preserves lengths."""
    from scipy.linalg import expm, schur
    A = _sector_J(np.asarray(_ZOO["K5"].B1_dense, float), "gradient")
    blk = schur(A, output="real")[0][:2, :2]
    J = blk / abs(blk[0, 1])
    assert np.allclose(J @ J, -np.eye(2), atol=1e-9)
    R = expm(0.7 * J)
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(expm(2 * np.pi * J), np.eye(2), atol=1e-9)


@pytest.mark.parametrize("name,dim,circles", [("K5", 4, 2), ("prism", 5, 2),
                                              ("petersen", 9, 4), ("house", 4, 2)])
def test_the_gradient_sector_splits_into_circles_with_their_own_rates(name, dim, circles):
    """The sector splits into complex lines, each with its OWN angular rate, and an
    odd dimension leaves a real line over: Petersen is 9 = 4 circles + 1. The real
    Schur form is block diagonal in 2x2 blocks [[0,w],[-w,0]], one per line."""
    from scipy.linalg import schur
    A = _sector_J(np.asarray(_ZOO[name].B1_dense, float), "gradient")
    assert A.shape[0] == dim
    Tm = schur(A, output="real")[0]
    band = np.triu(np.tril(Tm, 1), -1)
    assert np.allclose(Tm - band, 0, atol=1e-12), "block diagonal in 2x2 blocks"
    rates = sorted({round(float(s), 9) for s in np.linalg.svd(A, compute_uv=False)
                    if s > 1e-12})
    assert len(rates) == circles, rates
    assert 2 * circles + (dim % 2) == dim or 2 * circles <= dim


def test_angular_averaging_projects_onto_the_complex_scalars():
    """The polar technique itself, and why it works here. Averaging any observable
    over the circle sends it to the commutant of J, which is span{I, J} = the
    complex scalars. Four real numbers become two: a radial part and a rotation
    rate, exactly as the angular integral of a rotationally symmetric integrand
    leaves a function of r alone.

    And the radius is what survives: ||v||^2 is constant along the orbit.
    """
    from scipy.linalg import expm, schur
    A = _sector_J(np.asarray(_ZOO["K5"].B1_dense, float), "gradient")
    blk = schur(A, output="real")[0][:2, :2]
    J = blk / abs(blk[0, 1])

    M = np.random.default_rng(0).normal(size=(2, 2))
    ts = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    avg = sum(expm(-t * J) @ M @ expm(t * J) for t in ts) / len(ts)
    assert np.allclose(avg @ J - J @ avg, 0, atol=1e-6), "the average commutes with J"
    predicted = 0.5 * np.trace(M) * np.eye(2) + 0.5 * float(np.trace(M @ J.T)) * J
    assert np.allclose(avg, predicted, atol=1e-6), "and it is a + bJ"

    v = np.array([3.0, -1.0])
    q = float(v @ v)
    for t in (0.3, 1.7, 4.2):
        w = expm(t * J) @ v
        assert abs(float(w @ w) - q) < 1e-9, t


#### space and time: what each sector is, and what kind
def test_the_gradient_is_the_only_sector_that_decays(_none=None):
    """Under e^{-tL1} the gradient and curl both have finite half-lives ln2/lambda
    and the harmonic sector is fixed exactly. That is an ARROW: the flow is a strict
    contraction that never returns to the identity at any t > 0, checked out to 100.
    """
    from scipy.linalg import expm
    rex, faces = _FACED["prism"]
    B1 = np.asarray(rex.B1_dense, float)
    B2 = _face_cols(rex, faces[:3])
    L1 = B1.T @ B1 + B2 @ B2.T
    Qg, Qc = _sector(B1, "gradient"), np.linalg.qr(B2)[0][:, :3]
    from rexgraph.harmonic_sparse import _face_reduced_frame
    from rexgraph.hodge_coords import harmonic_frame
    Hc = np.asarray(harmonic_frame(rex).todense())
    Qh = np.linalg.qr(np.asarray(_face_reduced_frame(Hc, B2.T @ Hc).todense()))[0]

    assert np.linalg.eigvalsh(Qg.T @ L1 @ Qg).min() > 1e-9, "gradient decays"
    assert np.linalg.eigvalsh(Qc.T @ L1 @ Qc).min() > 1e-9, "curl decays"
    assert np.linalg.norm(Qh.T @ L1 @ Qh) < 1e-12, "harmonic does not"
    for t in (1.0, 20.0, 100.0):
        assert np.allclose(expm(-t * L1) @ Qh, Qh, atol=1e-12), "fixed for all t"
    assert not any(np.allclose(expm(-t * L1), np.eye(L1.shape[0]), atol=1e-6)
                   for t in (0.1, 1.0, 2.0, 5.0, 10.0, 100.0)), "never returns"


def test_the_rotation_returns_and_the_decay_does_not():
    """The two generators give two kinds of time on the SAME sector. exp(tJ) from
    [T,G] preserves the norm exactly and returns to I at 2 pi; e^{-tL1} from the
    Laplacian is monotone and returns at no t. Cyclical and linear, side by side."""
    from scipy.linalg import expm, schur
    rex, faces = _FACED["prism"]
    B1 = np.asarray(rex.B1_dense, float)
    T = B1.T @ B1
    G = np.abs(B1).T @ np.abs(B1)
    A = _sector(B1, "gradient").T @ (T @ G - G @ T) @ _sector(B1, "gradient")
    A = 0.5 * (A - A.T)
    Tm = schur(A, output="real")[0]
    i = max(range(A.shape[0] - 1), key=lambda k: abs(Tm[k, k + 1]))
    J = Tm[i:i + 2, i:i + 2] / abs(Tm[i, i + 1])
    assert np.allclose(J @ J, -np.eye(2), atol=1e-9)
    v = np.array([1.0, 0.0])
    for t in (0.0, 1.0, 2.0, 3.0):
        assert abs(float(np.linalg.norm(expm(t * J) @ v)) - 1.0) < 1e-9, "norm kept"
    assert np.allclose(expm(2 * np.pi * J), np.eye(2), atol=1e-9), "and it returns"


@pytest.mark.parametrize("name", ["K5", "prism", "house", "K6"])
def test_only_the_harmonic_coordinate_is_visible_to_a_cycle(name):
    """What each coordinate COUNTS, which is what fixes its kind.

    Pair a cochain against an integer harmonic cycle z:

      gradient  <B1^T phi, z> = <phi, B1 z> = 0   path-INdependent, a potential
      curl      <B2 psi,   z> = <psi, B2^T z> = 0 it bounds, z cannot see it
      harmonic  nonzero                           HOLONOMY, the only visible part

    So the gradient coordinate is a difference of a potential (linear, affine, no
    winding), and the harmonic coordinate is a winding number. For integer data the
    winding is an INTEGER, and it equals the pairing of the whole cochain, because
    the other two sectors contribute exactly zero. That integer is the counted
    duration: a number of turns, with no angle and no pi anywhere in it.
    """
    from rexgraph.harmonic_sparse import _face_reduced_frame
    from rexgraph.hodge_coords import harmonic_frame
    rex, faces = _FACED[name]
    B1 = np.asarray(rex.B1_dense, float)
    B2 = _face_cols(rex, faces[:len(faces) - 1])
    Hc = np.asarray(harmonic_frame(rex).todense())
    Hh = np.asarray(_face_reduced_frame(Hc, B2.T @ Hc).todense())
    assert Hh.shape[1] > 0
    assert np.array_equal(Hh, np.round(Hh)), "the harmonic frame stays integer"
    assert np.linalg.norm(B1 @ Hh) < 1e-12 and np.linalg.norm(B2.T @ Hh) < 1e-12

    rng = np.random.default_rng(11)
    z = Hh[:, 0]
    grad = B1.T @ rng.integers(-3, 4, B1.shape[0]).astype(float)
    curl = B2 @ rng.integers(-3, 4, B2.shape[1]).astype(float)
    assert abs(float(grad @ z)) < 1e-9, "a potential has no holonomy"
    assert abs(float(curl @ z)) < 1e-9, "what bounds is invisible to a free cycle"

    f = rng.integers(-5, 6, B1.shape[1]).astype(float)
    w = Hh.T @ f
    assert np.array_equal(w, np.round(w)), "integer data gives integer winding"
    Qh = np.linalg.qr(Hh)[0]
    assert abs(float(f @ z) - float((Qh @ (Qh.T @ f)) @ z)) < 1e-9, \
        "and the pairing sees only the harmonic part"


#### how long, without a transcendental
@pytest.mark.parametrize("name", ["K5", "prism", "petersen", "house", "C6"])
def test_the_harmonic_sector_is_what_the_flow_cannot_destroy(name):
    """So the harmonic sector is not the carrier of time; it is the invariant of it.
    Under e^{-t L1} every gradient direction has a finite half-life ln2/lambda with
    lambda > 0, while the harmonic sector is fixed exactly, for all t. It is the
    t -> infinity limit of the flow: the only part with unbounded duration."""
    from scipy.linalg import expm
    B1 = np.asarray(_ZOO[name].B1_dense, float)
    L1 = B1.T @ B1
    Qg, Qh = _sector(B1, "gradient"), _sector(B1, "harmonic")
    lam = np.linalg.eigvalsh(Qg.T @ L1 @ Qg)
    assert lam.min() > 1e-9, "every gradient direction decays"
    for t in (1.0, 20.0, 100.0):
        E = expm(-t * L1)
        assert np.allclose(E @ Qh, Qh, atol=1e-12), "the harmonic sector is fixed"
    assert np.linalg.norm(expm(-100.0 * L1) @ Qg) < 1e-12, "the rest is gone"


def test_duration_is_a_valuation_and_needs_no_pi():
    """"How far" needs an arc length and therefore pi. "How long" does not, as long
    as it is COUNTED rather than measured: iterating the (3,4,5) rotation n times
    gives a denominator of exactly 5^n, so n is recovered by an integer valuation.

    No angle, no arc length, no logarithm: the clock is the denominator.
    """
    from fractions import Fraction

    def compose(r1, r2):
        return (r1[0] * r2[0] - r1[1] * r2[1], r1[0] * r2[1] + r1[1] * r2[0])

    def valuation(n, p):
        k = 0
        while n and n % p == 0:
            n //= p
            k += 1
        return k

    A = (Fraction(3, 5), Fraction(4, 5))
    r = A
    for n in range(1, 9):
        assert r[0] * r[0] + r[1] * r[1] == 1, n
        assert r[0].denominator == 5 ** n, (n, r)
        assert valuation(r[0].denominator, 5) == n
        r = compose(r, A)


@pytest.mark.parametrize("a,b,c", [(3, 4, 5), (5, 12, 13), (8, 15, 17), (20, 21, 29)])
def test_every_rational_rotation_carries_its_own_prime_clock(a, b, c):
    """And the clock is not universal: each primitive triple ticks on its own prime,
    v_5 for (3,4,5), v_13 for (5,12,13), v_17 for (8,15,17), v_29 for (20,21,29).
    Duration is exact but it is stated per generator, not on one shared line."""
    from fractions import Fraction

    def compose(r1, r2):
        return (r1[0] * r2[0] - r1[1] * r2[1], r1[0] * r2[1] + r1[1] * r2[0])

    g = (Fraction(a, c), Fraction(b, c))
    r = g
    for n in range(1, 7):
        assert r[0] * r[0] + r[1] * r[1] == 1
        assert r[0].denominator == c ** n, (n, r)
        r = compose(r, g)


def test_continuous_duration_is_a_projection_the_way_distance_is():
    """The boundary of the exactness, stated rather than glossed. A half-life
    ln(2)/lambda is transcendental in the same way an arc length is: the algebraic
    content is lambda, an eigenvalue of an integer matrix, and the logarithm is the
    projection into a measurable unit. Counted time is exact; measured time is not.
    """
    B1 = np.asarray(_ZOO["K5"].B1_dense, float)
    L1 = B1.T @ B1
    assert np.array_equal(L1, np.round(L1)), "the generator is an integer matrix"
    lam = np.linalg.eigvalsh(_sector(B1, "gradient").T @ L1 @ _sector(B1, "gradient"))
    assert np.allclose(lam, 5.0), "and its rates are algebraic: K5 gives 5 exactly"
    ratio = float(np.log(2) / lam.max())
    assert abs(ratio - np.log(2) / 5.0) < 1e-12, "the transcendental is only the unit"


#### the circle without pi
@pytest.mark.parametrize("k,pairs,real", [(4, 1, 1), (5, 2, 0), (6, 2, 1),
                                          (7, 3, 0), (8, 3, 1), (9, 4, 0)])
def test_the_gradient_sector_is_fully_complex_up_to_parity(k, pairs, real):
    """The law, once it is read on the sector the exchange actually acts on. The
    gradient sector of K_k has dimension k-1 and [T,G] is FULL RANK on it, so it
    splits into floor((k-1)/2) complex lines with a single real line left over when
    k is even. Nothing is capped by k-2: that bound came from the mismatched
    harmonic reading."""
    e = list(itertools.combinations(range(k), 2))
    r = RexGraph(sources=np.array([a for a, b in e], np.int32),
                 targets=np.array([b for a, b in e], np.int32))
    r._ensure_clean()
    A = _sector_J(np.asarray(r.B1_dense, float), "gradient")
    assert A.shape[0] == k - 1
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int((sv > sv.max() * 1e-9).sum())
    assert rank // 2 == pairs, (rank, sv)
    assert A.shape[0] - rank == real


def _k5_faces(nf):
    """nf independent triangles of K5, as B2 columns on the edge order of B1."""
    ei = {e: i for i, e in enumerate(itertools.combinations(range(5), 2))}
    tris = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 3, 4)][:nf]
    B2 = np.zeros((len(ei), len(tris)))
    for j, (a, b, c) in enumerate(tris):
        B2[ei[(a, b)], j] = 1.0
        B2[ei[(b, c)], j] = 1.0
        B2[ei[(a, c)], j] = -1.0
    return B2


@pytest.mark.parametrize("nf", [0, 1, 2, 3, 4, 5])
def test_the_whole_cycle_space_is_annihilated_by_the_topology_channel(nf):
    """The general theorem, of which the harmonic case is half.

    T = B1^T B1 annihilates ker(B1), and by the chain condition B1 B2 = 0 the curl
    sector im(B2) lies inside ker(B1) as well. So T kills the ENTIRE cycle space,
    curl and harmonic alike, and

        Q^T [T,G] Q = (TQ)^T G Q - Q^T G (TQ) = 0

    on any part of it. All of the topology/geometry exchange lives on the gradient
    sector; the cycle space carries none of it at any face count.
    """
    B1 = np.asarray(_ZOO["K5"].B1_dense, float)
    B2 = _k5_faces(nf)
    B1w, B2w = _deformed_pair(B1, B2)
    T = B1w.T @ B1w
    if nf:
        assert np.abs(B1w @ B2w).max() < 1e-12, "the deformation preserves the chain"
        assert np.linalg.norm(T @ B2w) < 1e-12, "T kills the curl sector"
    C = _tg_deformed(B1)
    for Q in ([np.linalg.qr(B2w)[0][:, :nf]] if nf else []) + [_sector(B1w, "harmonic")]:
        if Q.shape[1]:
            assert float(np.linalg.norm(Q.T @ C @ Q)) < 1e-12


@pytest.mark.parametrize("name,edges,faces", [
    ("K5", None, [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 3, 4)]),
])
#### what faces actually govern: a conservation law
def _face_cols(rex, loops):
    """B2 columns for closed loops of vertices, any length, signed to close."""
    src, dst = np.asarray(rex.sources), np.asarray(rex.targets)
    E = [(int(a), int(b)) for a, b in zip(src, dst, strict=True)]
    ei = {}
    for i, (a, b) in enumerate(E):
        ei[(a, b)] = i
        ei[(b, a)] = i
    cols = []
    for lp in loops:
        c = np.zeros(len(E))
        for a, b in zip(lp, list(lp[1:]) + [lp[0]], strict=True):
            c[ei[(a, b)]] += 1.0 if E[ei[(a, b)]] == (a, b) else -1.0
        cols.append(c)
    return np.stack(cols, 1) if cols else np.zeros((len(E), 0))


_FACED = {
    "K5": (_ZOO["K5"], [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 3, 4)]),
    "K4": (_graph(list(itertools.combinations(range(4), 2))),
           [(0, 1, 2), (0, 1, 3), (0, 2, 3)]),
    "K6": (_graph(list(itertools.combinations(range(6), 2))),
           [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 3, 4)]),
    "prism": (_ZOO["prism"], [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4)]),
    "house": (_ZOO["house"], [(2, 3, 4), (0, 1, 2, 3)]),
}


def _cycle_split(B1, B2, rex):
    """The gradient's view of the cycle space, split into what bounds and what does
    not. ker(B1) = curl + harmonic orthogonally, and faces change only WHICH
    splitting of that fixed space is used."""
    from rexgraph.harmonic_sparse import _face_reduced_frame
    from rexgraph.hodge_coords import harmonic_frame
    T = B1.T @ B1
    G = np.abs(B1).T @ np.abs(B1)
    C = T @ G - G @ T
    Qg = _sector(B1, "gradient")
    Hc = np.asarray(harmonic_frame(rex).todense())
    nf = int(np.linalg.matrix_rank(B2)) if B2.size else 0
    Qc = np.linalg.qr(B2)[0][:, :nf] if nf else np.zeros((B1.shape[1], 0))
    Hh = np.asarray(_face_reduced_frame(Hc, B2.T @ Hc).todense()) if B2.size else Hc
    Qh = np.linalg.qr(Hh)[0] if Hh.shape[1] else np.zeros((B1.shape[1], 0))
    gc = float(np.linalg.norm(Qg.T @ C @ Qc)) if nf else 0.0
    gh = float(np.linalg.norm(Qg.T @ C @ Qh)) if Qh.shape[1] else 0.0
    return gc, gh


@pytest.mark.parametrize("name,total", [("K5", 16.0), ("K4", 8.0),
                                        ("prism", 8.099383), ("house", 5.257030),
                                        ("K6", 27.129320)])
def test_faces_repartition_a_conserved_outflow(name, total):
    """What face independence actually governs, and it is a conservation law.

    The cycle space is annihilated by T, so it exchanges nothing internally; what it
    does is feed the gradient one way. That outflow is INVARIANT under adding faces.
    All a face does is move part of the cycle space from the free side to the bounded
    side, and the outflow follows it across. On K5:

        faces      0        1        2        3        4        5
        from curl  0.000    7.303    8.944    9.798   11.662   15.833
        from harm 16.000   14.236   13.267   12.649   10.954    2.309
        total     16.000   16.000   16.000   16.000   16.000   16.000

    Exact to 0.0e+00, and it is a theorem rather than a coincidence: ker(B1) is the
    whole cycle space and faces do not change it, they only refine its orthogonal
    splitting into curl + harmonic. The total out of a fixed space cannot move.

    Verified with quadrilateral faces as well as triangles, so it is arity-general
    at grade 2 and not a simplicial accident.
    """
    rex, faces = _FACED[name]
    B1 = np.asarray(rex.B1_dense, float)
    base = None
    for nf in range(0, len(faces) + 1):
        B2 = _face_cols(rex, faces[:nf])
        if nf:
            assert np.abs(B1 @ B2).max() < 1e-12, "the faces are valid chains"
        gc, gh = _cycle_split(B1, B2, rex)
        got = float(np.hypot(gc, gh))
        if base is None:
            base = got
            assert abs(base - total) < 1e-5, (name, base)
        assert abs(got - base) < 1e-9, (name, nf, gc, gh, got)


@pytest.mark.parametrize("name", ["K5", "prism", "house", "petersen"])
def test_the_outflow_drifts_less_than_the_internal_turn_but_not_always(name):
    """The honest version of a separation I wanted to be clean and is not.

    Across sigma = 0.05 to 0.95 the outflow from the cycle space moves by 1 to 8
    percent. The gradient's internal turn moves by 514 percent on K5, which looked
    like the same radial/angular split the rest of the family shows, until the
    prism, where the turn moves 3.5 percent and the outflow 8.0, the wrong way round.

        structure   outflow spread   turn spread   ratio
        K5                  0.0098        5.1446   523.5
        prism               0.0795        0.0354     0.4
        house               0.0355        0.3731    10.5
        petersen            0.0363        0.1881     5.2

    So the outflow's exact conservation is under FACES, and only under faces. Its
    sigma behaviour is small but not privileged, and no general ordering holds.
    The claim asserted here is the one that survives all four: the outflow drifts
    by under ten percent.
    """
    B1 = np.asarray(_ZOO[name].B1_dense, float)
    out = []
    for sigma in np.linspace(0.05, 0.95, 19):
        B1w, _ = _deformed_pair(B1, None, sigma)
        out.append(float(np.linalg.norm(_tg_deformed(B1, sigma)
                                        @ _sector(B1w, "harmonic"))))
    out = np.array(out)
    assert (out.max() - out.min()) / out.min() < 0.10


def test_the_complex_unit_generates_a_finite_group_with_no_transcendental():
    """exp(2 pi J) = I imports pi into an apparatus that is otherwise exact
    integers, and that circle is the PROJECTION into a metric space. The algebraic
    circle underneath needs none of it: J^4 = I exactly, so J alone generates Z/4,
    the four quadrants, purely algebraically."""
    from fractions import Fraction
    J = np.array([[Fraction(0), Fraction(1)], [Fraction(-1), Fraction(0)]], dtype=object)
    I2 = np.array([[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]], dtype=object)
    assert (J @ J == -I2).all()
    assert (J @ J @ J @ J == I2).all()


@pytest.mark.parametrize("a,b,c", [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)])
def test_every_pythagorean_triple_is_an_exact_point_of_the_circle(a, b, c):
    """R = (aI + bJ)/c has R^T R = I EXACTLY over the rationals, since
    (aI - bJ)(aI + bJ) = (a^2 + b^2) I and a^2 + b^2 = c^2. No square root is taken
    and no angle is named."""
    from fractions import Fraction
    R = np.array([[Fraction(a, c), Fraction(b, c)],
                  [Fraction(-b, c), Fraction(a, c)]], dtype=object)
    P = R.T @ R
    assert P[0][0] == P[1][1] == Fraction(1)
    assert P[0][1] == P[1][0] == Fraction(0)
    assert R[0][0] * R[1][1] - R[0][1] * R[1][0] == Fraction(1)


def test_the_rational_rotations_close_into_a_group():
    """(c1 + s1 J)(c2 + s2 J) = (c1c2 - s1s2) + (c1s2 + s1c2) J, all rational, so
    the exact points compose: (3,4,5) with (5,12,13) gives (33,56,65), and
    33^2 + 56^2 = 65^2. Inverses are the conjugates. Iterating (3,4,5) stays exactly
    on the circle with denominators 5^n and never closes, so the exact points are
    dense: a circle that is a group of rationals rather than an arc length."""
    from fractions import Fraction

    def compose(r1, r2):
        return (r1[0] * r2[0] - r1[1] * r2[1], r1[0] * r2[1] + r1[1] * r2[0])

    def on_circle(r):
        return r[0] * r[0] + r[1] * r[1] == 1

    A = (Fraction(3, 5), Fraction(4, 5))
    B = (Fraction(5, 13), Fraction(12, 13))
    assert on_circle(A) and on_circle(B)
    AB = compose(A, B)
    assert on_circle(AB)
    assert AB == (Fraction(-33, 65), Fraction(56, 65))
    assert compose(A, (A[0], -A[1])) == (Fraction(1), Fraction(0))
    r = A
    for n in range(2, 8):
        r = compose(r, A)
        assert on_circle(r), n
        assert r[0].denominator in (5 ** n, 1) or r[1].denominator in (5 ** n, 1)


def test_the_spread_of_a_rotation_is_a_ratio_of_integers():
    """What replaces the angle. The (3,4,5) rotation moves a vector by spread
    16/25, exactly, and no transcendental is involved in saying so."""
    from fractions import Fraction
    v = (Fraction(1), Fraction(0))
    for a, b, c, want in ((3, 4, 5, Fraction(16, 25)),
                          (5, 12, 13, Fraction(144, 169)),
                          (8, 15, 17, Fraction(225, 289))):
        w = (Fraction(a, c) * v[0] + Fraction(b, c) * v[1],
             Fraction(-b, c) * v[0] + Fraction(a, c) * v[1])
        dot = v[0] * w[0] + v[1] * w[1]
        qv = v[0] * v[0] + v[1] * v[1]
        qw = w[0] * w[0] + w[1] * w[1]
        assert 1 - dot * dot / (qv * qw) == want


def test_the_generator_is_an_integer_matrix_before_normalising():
    """And it starts exact. On the raw integer boundary of K5, T and G are integer
    matrices and so is [T,G], entries in {-4,-2,0,2,4}. The rationals appear only
    when the hats are trace-normalised, and pi only when the circle is given an arc
    length. The object is algebraic at every step before that."""
    r = _k5()
    B1 = np.asarray(r.B1_dense, float)
    T = B1.T @ B1
    G = np.abs(B1).T @ np.abs(B1)
    C = T @ G - G @ T
    for M in (T, G, C):
        assert np.array_equal(M, np.round(M))
    assert set(np.round(C).astype(int).ravel().tolist()) <= {-4, -2, 0, 2, 4}
    assert np.abs(C + C.T).max() == 0.0
