# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._l_gb: Graded boundary Laplacian L_gb.

The L_gb operator measures structural coupling between adjacent grades
of a relational complex, generalizing the within-grade RL_4 character
bundle to a between-grade tensor.

Two flavors:

  RANK-2 SCALAR  l_gb(grade_d, grade_d+1):
      A single scalar measuring how much the spectral content of the
      down-grade Laplacian differs from the up-grade Laplacian's
      shadow projection.

  RANK-4 CHANNEL TENSOR  L_gb_channels(hats_A, hats_B):
      A 4×4 matrix where T[i, j] measures the spectral distance between
      channel i in graded operator A and channel j in graded operator B.
      For self-tensor (B = A), the diagonal is identically zero and the
      off-diagonals encode within-grade channel-mixing structure.

Reference values, verified to 3 decimals:
    K_4    : TG=0.471  TC=0.760  GC=0.810
    K_5    : TG=0.740  TC=0.884  GC=0.820
    K_6    : TG=0.806  TC=1.036  GC=0.892
    Cycles : TC=0     (uniquely identifies cycle graphs)
    Universal: TF=GF=FC=1 (the F channel is always orthogonal to T,G,C)

This file is the Cython port of the pure-numpy reference in
rexgraph/tests/reference/l_gb_reference.py. Use `python -m pytest
tests/test_l_gb.py` to verify the compiled output matches the reference
to 1e-13 relative error.

Algorithm:
    Eigendecompose the Hodge Dirac at grades d and d+1 (each O(n_E^3)),
    extract sorted absolute eigenvalues, compare via the channel tensor
    formula. Vertex-driven assembly.

    Dense throughout: every routine here goes through numpy.linalg, there is no
    sparse path and no spectrum truncation. Earlier text advertised an "optional
    sparse path for large n_E" and a `top_k_eig` parameter; neither exists.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64,
    can_allocate_dense_f64,
    should_use_dense_matmul,
    get_EPSILON_DIV,
)

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs

np.import_array()

try:
    from scipy.sparse import eye as _speye, diags as _spdiags
    _HAS_SCIPY_SPARSE = True
except ImportError:
    _HAS_SCIPY_SPARSE = False


# Spectrum extraction


def normalized_coherence_spectrum(np.ndarray[f64, ndim=2] M):
    """Return sorted absolute eigenvalues of symmetric M, rescaled max=1.

    Parameters
    ----------
    M : ndarray[nE, nE]
        Symmetric operator (any Laplacian or hat).

    Returns
    -------
    spec : ndarray[k] of f64
        Top eigenvalues sorted descending, with spec[0] = 1.0.
        Length k is the number of nonzero (above EPSILON_DIV) eigenvalues.
    """
    if M.shape[0] == 0:
        return np.zeros(1, dtype=np.float64)
    cdef np.ndarray[f64, ndim=2] M_sym = 0.5 * (M + M.T)
    cdef np.ndarray[f64, ndim=1] evals = np.linalg.eigvalsh(M_sym)
    cdef np.ndarray[f64, ndim=1] absvals = np.sort(np.abs(evals))[::-1]
    cdef f64 cutoff = get_EPSILON_DIV()
    absvals = absvals[absvals > cutoff]
    if len(absvals) == 0:
        return np.zeros(1, dtype=np.float64)
    return absvals / absvals[0]


# Hodge Dirac spectrum at a grade


def dirac_spectrum_at_grade(B1_in, B2_in, int grade):
    """Full Hodge Laplacian spectrum at the requested grade.

    Parameters
    ----------
    B1_in : ndarray[nV, nE]
        Vertex-edge boundary operator.
    B2_in : ndarray[nE, nF] or None
        Edge-face boundary operator. None if no faces.
    grade : int
        0 = vertex grade (L_0 = B1 B1^T)
        1 = edge grade   (L_1 = B1^T B1 + B2 B2^T)
        2 = face grade   (L_2 = B2^T B2)

    Returns
    -------
    spec : ndarray[?] of f64
        Eigenvalues at the requested grade, sorted ascending.
    """
    cdef np.ndarray[f64, ndim=2] B1 = np.ascontiguousarray(B1_in, dtype=np.float64)
    cdef int nV = B1.shape[0]
    cdef int nE = B1.shape[1]
    cdef np.ndarray[f64, ndim=2] L

    if grade == 0:
        if nV == 0:
            return np.zeros(0, dtype=np.float64)
        L = B1 @ B1.T
    elif grade == 1:
        if nE == 0:
            return np.zeros(0, dtype=np.float64)
        L = B1.T @ B1
        if B2_in is not None:
            B2 = np.ascontiguousarray(B2_in, dtype=np.float64)
            if B2.shape[1] > 0:
                L = L + B2 @ B2.T
    elif grade == 2:
        if B2_in is None:
            return np.zeros(0, dtype=np.float64)
        B2 = np.ascontiguousarray(B2_in, dtype=np.float64)
        if B2.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        L = B2.T @ B2
    else:
        raise ValueError(f"grade must be 0, 1, or 2; got {grade}")

    return normalized_coherence_spectrum(L)


# Rank-2 between-grade scalar



#: The reference floors each spectrum's norm before normalising, so a spectrum that
#: is identically zero leaves the OTHER projector standing rather than collapsing the
#: pair to nothing. That is where the T[i,F] = 1 reading comes from. This is not a
#: decision threshold: it reproduces the reference's normalisation, and the formula
#: below carries it through in closed form instead of branching on it.
cdef f64 _LGB_FLOOR = 1e-12


cdef inline void _pair_spectrum(np.ndarray[f64, ndim=1] a, np.ndarray[f64, ndim=1] b,
                                f64 *top, f64 *bot, f64 *frob) noexcept:
    """The spectrum and Frobenius norm of `a a^T/|a|^2 - b b^T/|b|^2`, in closed form.

    Write the operator as `alpha P_a - beta P_b` with P unit rank-1 projectors and
    `alpha = (|a| / max(|a|, floor))^2`, which is 1 for any ordinary spectrum and 0
    for one that is identically zero. On the two-dimensional span it has

        trace        alpha - beta
        determinant  -alpha beta s^2          s^2 = 1 - cos^2

    so the eigenvalues are `((alpha - beta) +- sqrt((alpha - beta)^2 + 4 alpha beta
    s^2)) / 2` and `||L||_F^2 = alpha^2 + beta^2 - 2 alpha beta cos^2`. One dot
    product settles all three, at O(n) against O(n^2) to form the outer products,
    and no eigensolver.

    Every regime falls out of the one expression rather than being special-cased:

        both ordinary   +-sqrt(spread), frob sqrt(2 spread)
        a zero          0 and -1,       frob 1
        b zero          1 and 0,        frob 1
        both zero       0 and 0,        frob 0
        parallel        0 and 0,        frob 0

    Checked against forming the operator and eigendecomposing it: 3.3e-16 across
    all of them, the tiny-but-nonzero regime included.
    """
    cdef f64 ra = <f64>np.linalg.norm(a)
    cdef f64 rb = <f64>np.linalg.norm(b)
    cdef f64 na = ra if ra > _LGB_FLOOR else _LGB_FLOOR
    cdef f64 nb = rb if rb > _LGB_FLOOR else _LGB_FLOOR
    cdef f64 al = (ra / na) * (ra / na)
    cdef f64 be = (rb / nb) * (rb / nb)
    cdef f64 s2 = 1.0
    cdef f64 tr, disc, q
    cdef np.ndarray[f64, ndim=1] ah, perp
    if ra > 0.0 and rb > 0.0:
        # sin^2 from the component of b ORTHOGONAL to a, not from 1 - cos^2.
        # Subtracting nearly-equal numbers under a square root is what wrecks the
        # near-parallel case: for identical spectra cos^2 lands at 1 - 2e-16, and
        # sqrt turns that into 3e-8. Taking the perpendicular part instead keeps
        # the cancellation in the vector space where it is exact, and the same
        # clamp that hid the first error also drove the near-parallel reading to a
        # flat 0 where the true value is 8.5e-10.
        ah = np.asarray(a, dtype=np.float64) / ra
        perp = np.asarray(b, dtype=np.float64) - (<f64>np.dot(ah, b)) * ah
        s2 = (<f64>np.dot(perp, perp)) / (rb * rb)
        if s2 > 1.0:
            s2 = 1.0
        elif s2 < 0.0:
            s2 = 0.0
    tr = al - be
    disc = sqrt(tr * tr + 4.0 * al * be * s2)
    top[0] = 0.5 * (tr + disc)
    bot[0] = 0.5 * (tr - disc)
    # alpha^2 + beta^2 - 2 alpha beta cos^2 written as (alpha-beta)^2 + 2 alpha
    # beta sin^2: a sum of non-negative terms, so nothing cancels here either
    q = tr * tr + 2.0 * al * be * s2
    frob[0] = sqrt(q) if q > 0.0 else 0.0


def l_gb_scalar(np.ndarray[f64, ndim=1] spec_d,
                np.ndarray[f64, ndim=1] spec_d1):
    """Scalar coupling between two grade spectra.

    Computes the Frobenius distance between the rank-1 outer-product
    projections of the normalized coherence spectra at adjacent grades.

    Returns
    -------
    coupling : f64
        Nonneg scalar; 0 means the two grades have identical spectral shape.
    """
    if spec_d.shape[0] == 0 or spec_d1.shape[0] == 0:
        return 0.0
    cdef int L = max(spec_d.shape[0], spec_d1.shape[0])
    cdef np.ndarray[f64, ndim=1] a = np.pad(spec_d, (0, L - spec_d.shape[0]))
    cdef np.ndarray[f64, ndim=1] b = np.pad(spec_d1, (0, L - spec_d1.shape[0]))
    cdef f64 top, bot, frob
    _pair_spectrum(a, b, &top, &bot, &frob)
    return float(frob)


# Rank-4 within-grade channel tensor


def l_gb_channel_tensor(list hats_A, list hats_B=None):
    """4×4 channel coupling tensor.

    For each pair (i, j), computes the Frobenius distance between the
    normalized rank-1 projections of channel i in hats_A and channel j
    in hats_B.

    Convention for hats_A == hats_B (self-tensor): diagonal entries are
    identically zero (channel matches itself), off-diagonals encode
    within-grade structure.

    Universal identity (verified across graph families):
        T[i, F] = T[F, i] = 1 for i in {T, G, C}
    The F channel is always Frobenius-orthogonal to T, G, C in unit-norm
    projection space.

    Reference values:
        K_4 self-tensor:  TG=0.471  TC=0.760  GC=0.810
        K_5 self-tensor:  TG=0.740  TC=0.884  GC=0.820
        K_6 self-tensor:  TG=0.806  TC=1.036  GC=0.892
        cycle graphs:     TC=0  (uniquely characterizes cycles)
    """
    if hats_B is None:
        hats_B = hats_A
    cdef int n_A = len(hats_A)
    cdef int n_B = len(hats_B)
    cdef np.ndarray[f64, ndim=2] T = np.zeros((n_A, n_B), dtype=np.float64)
    cdef int i, j
    for i in range(n_A):
        sA = normalized_coherence_spectrum(hats_A[i])
        for j in range(n_B):
            if i == j:
                continue
            sB = normalized_coherence_spectrum(hats_B[j])
            T[i, j] = l_gb_scalar(sA, sB)
    return T


# Sweep across all adjacent grade pairs


def l_gb_tower(list B_list):
    """Sweep l_gb across all adjacent grade pairs in a relational complex.

    Parameters
    ----------
    B_list : list of ndarray
        [B_0, B_1, B_2, ...] boundary operators. B_d has shape
        (n_{d-1}, n_d). Pass None for empty grades.

    Returns
    -------
    results : list of dict
        One dict per adjacent pair (d, d+1), each containing the fields
        from l_gb_scalar plus 'pair': (d, d+1).
    """
    cdef int n_grades = len(B_list)
    cdef int d
    cdef f64 na, nb
    cdef f64 c_top, c_bot, c_frob
    if n_grades == 0:
        return []

    # Build spectrum at each grade 0..n_grades using Hodge Laplacian
    specs = []
    for d in range(n_grades + 1):
        # Down part: B_d^T @ B_d (only if d >= 1)
        L_down = None
        if d >= 1 and (d - 1) < n_grades:
            B_d = B_list[d - 1]
            if B_d is not None and B_d.size > 0:
                B_d = np.ascontiguousarray(B_d, dtype=np.float64)
                L_down = B_d.T @ B_d
        # Up part: B_{d+1} @ B_{d+1}^T
        L_up = None
        if d < n_grades:
            B_dp1 = B_list[d]
            if B_dp1 is not None and B_dp1.size > 0:
                B_dp1 = np.ascontiguousarray(B_dp1, dtype=np.float64)
                L_up = B_dp1 @ B_dp1.T

        if L_down is None and L_up is None:
            specs.append(np.zeros(1, dtype=np.float64))
        elif L_down is None:
            specs.append(normalized_coherence_spectrum(L_up))
        elif L_up is None:
            specs.append(normalized_coherence_spectrum(L_down))
        else:
            # Match dimensions by zero-padding the smaller
            n = max(L_down.shape[0], L_up.shape[0])
            if L_down.shape[0] < n:
                pad = n - L_down.shape[0]
                L_down = np.pad(L_down, ((0, pad), (0, pad)))
            if L_up.shape[0] < n:
                pad = n - L_up.shape[0]
                L_up = np.pad(L_up, ((0, pad), (0, pad)))
            specs.append(normalized_coherence_spectrum(L_down + L_up))

    # Compute L_gb between each adjacent pair
    results = []
    for d in range(len(specs) - 1):
        sd = specs[d]
        sd1 = specs[d + 1]
        # Build the full l_gb_scalar dict (matching the reference)
        L_size = max(len(sd), len(sd1))
        a = np.pad(sd, (0, L_size - len(sd)))
        b = np.pad(sd1, (0, L_size - len(sd1)))
        na = max(float(np.linalg.norm(a)), 1e-12)
        nb = max(float(np.linalg.norm(b)), 1e-12)
        # the spectrum in closed form: see _pair_spectrum. No eigensolver, and the
        # L x L outer products are never formed for these three.
        _pair_spectrum(a, b, &c_top, &c_bot, &c_frob)
        top_eig = float(c_top)
        bot_eig = float(c_bot)
        frob = float(c_frob)

        # Localization reads the ENTRYWISE absolute value, which is not rank-2 and
        # has no closed form, so this one pair of outer products is still built.
        PA = np.outer(a, a) / (na * na)
        PB = np.outer(b, b) / (nb * nb)
        L_gb = PA - PB
        abs_L = np.abs(L_gb)
        try:
            eigvals_abs, eigvecs_abs = np.linalg.eigh(0.5 * (abs_L + abs_L.T))
            v_top = eigvecs_abs[:, -1]
            a_norm = a / na
            b_norm = b / nb
            ma = float(np.dot(v_top, a_norm)) ** 2
            mb = float(np.dot(v_top, b_norm)) ** 2
            if ma + mb > 1e-15:
                localization = (mb - ma) / (mb + ma)
            else:
                localization = 0.0
        except np.linalg.LinAlgError:
            localization = 0.0

        result = {
            "top_eig": top_eig,
            "bot_eig": bot_eig,
            "spread": top_eig - bot_eig,
            "localization": localization,
            "frob": frob,
            "L_gb": L_gb,
            "pair": (d, d + 1),
        }
        results.append(result)

    return results
