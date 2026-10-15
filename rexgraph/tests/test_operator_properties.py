"""Operator properties across weighted, branching and parallel relation fixtures.

The tests compare assembled matrices, exact counts and numerical responses
with independent calculations. Randomized fixtures vary relation weights,
arity and orientation so those inputs are exercised separately.
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
    """Simple, weighted, branching and parallel edge complexes, random orientation.

    Weighted and branching are both here on purpose: they are the two axes that
    unweighted simple fixtures cannot distinguish.
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


# Lagrangian fields and channel diagonals
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
    """c^2 = Ls/Lt = (1: chi_T)/chi_T, since chi_k = hat_k/RL[e,e] divides the
    common factor out. So arg f(e) = arctan(c^2) is scale free: the modulus of the
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


# Boundary closure and Hodge products
def test_the_chain_condition_forces_the_graded_operators_to_commute():
    """L_down = B_d^T B_d and L_up = B_{d+1} B_{d+1}^T, so

        L_down L_up = B_d^T (B_d B_{d+1}) B_{d+1}^T = 0
        L_up L_down = B_{d+1} (B_d B_{d+1})^T B_d = 0

    both vanishing because B_d B_{d+1} = 0. The commutator is not small, it is
    identically zero, which is why a Cauchy Riemann condition on the GRADED
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
    """The diagonal squares each incidence entry and squaring kills the
    sign, so diag(T) = diag(G) whatever the metric. That is why F's content is
    entirely off diagonal and why F has to manufacture a diagonal from it."""
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
    """A column is (-1 at the head, 1/(k-1)
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
    """Matrix Tree, in scope for a connected 2 ary face free complex. Checked
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
    """L_gb = alpha P_a - beta P_b on the two grades' spectra, a difference
    of rank 1 projectors, so on their span trace = alpha - beta and determinant =
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


# Closure values in the complete graph cycle basis
@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_the_closure_law_on_complete_graphs(n):
    """Check closure entries in the fundamental cycle basis of complete graphs.

    The fixture uses triangle cycle columns. The diagonal and nonzero off
    diagonal entries are compared with explicit rational expected values.
    """
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
    """Checked by full enumeration. A sign per relation is
    a Z/2 gauge field, so the total space is 2^nE. The gauge group (Z/2)^nV acts by
    sigma_ij -> t_i t_j sigma_ij, with the global flip acting trivially on each
    component, so orbits have size 2^(nV: b0). Holonomy is constant on an orbit and
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
    """Both masses sum SQUARED entries, and squaring kills the sign, so
    neither can see a gauge transformation, an arbitrary re signing, or a
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
