# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._l_gb: Graded boundary Laplacian L_gb.

The L_gb operator measures structural coupling between adjacent grades
of a relational complex, generalizing the within grade RL_4 character
bundle to a between grade tensor.

Two forms:

  RANK 2 SCALAR  l_gb(grade_d, grade_d+1):
      A single scalar measuring how much the spectral content of the
      down-grade Laplacian differs from the up-grade Laplacian's
      shadow projection.

  RANK 4 CHANNEL TENSOR  L_gb_channels(hats_A, hats_B):
      A 4×4 matrix where T[i, j] measures the spectral distance between
      channel i in graded operator A and channel j in graded operator B.
      For self-tensor (B = A), the diagonal is identically zero and the
      off-diagonals encode within-grade channel-mixing structure.

Full coherence spectra use native LAPACK without eigenvectors. Sparse operators
retain their storage and split spectral work by connected support. Pairwise projector
distances use a closed form without an eigensolver. Tower localization reads
the leading eigenvector of the entrywise absolute projector difference.
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

# Spectrum extraction


def normalized_coherence_spectrum(M):
    """Return sorted absolute eigenvalues of symmetric M, rescaled max=1.

    Parameters

    M : ndarray[nE, nE] or sparse matrix
        Operator read through its symmetric part.

    Returns

    spec : ndarray[k] of f64
        All retained eigenvalue magnitudes sorted descending, with spec[0] = 1.0.
        Length k is the number of nonzero (above EPSILON_DIV) eigenvalues.
        A zero operator returns a single zero.
    """
    from rexgraph.core._linalg import eigvalsh, symmetric_sparse_spectrum
    cdef np.ndarray[f64, ndim=1] evals
    if hasattr(M, 'tocsr') or hasattr(M, 'dual') or hasattr(M, 'row_ptr'):
        evals = symmetric_sparse_spectrum(M)
    else:
        a = np.asarray(M)
        if np.iscomplexobj(a):
            raise TypeError("coherence spectrum requires a real matrix")
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("coherence spectrum requires a square matrix")
        evals = eigvalsh(0.5 * a + 0.5 * a.T)
    if evals.shape[0] == 0:
        return np.zeros(1, dtype=np.float64)
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

    B1_in : ndarray[nV, nE]
        Vertex-edge boundary operator.
    B2_in : ndarray[nE, nF] or None
        Edge-face boundary operator. None if no faces.
    grade : int
        0 = vertex grade (L_0 = B1 B1^T)
        1 = edge grade   (L_1 = B1^T B1 + B2 B2^T)
        2 = face grade   (L_2 = B2^T B2)

    Returns

    spec : ndarray[?] of f64
        Nonzero absolute eigenvalues, sorted descending and rescaled to max=1.
    """
    from rexgraph.native_sparse import as_native
    if (hasattr(B1_in, 'tocsr') or hasattr(B1_in, 'dual') or hasattr(B1_in, 'row_ptr')
            or (B2_in is not None and (hasattr(B2_in, 'tocsr') or hasattr(B2_in, 'dual')
                                      or hasattr(B2_in, 'row_ptr')))):
        b1 = as_native(B1_in)
        b2 = as_native(B2_in) if B2_in is not None else None
        if b2 is not None and b2.shape[0] != b1.shape[1]:
            raise ValueError("adjacent boundary axes do not match")
        if grade == 0:
            return normalized_coherence_spectrum(b1.product(b1.T)) if b1.shape[0] else np.empty(0)
        if grade == 1:
            if not b1.shape[1]:
                return np.empty(0)
            operator = b1.T.product(b1)
            if b2 is not None:
                operator = operator.add(b2.product(b2.T))
            return normalized_coherence_spectrum(operator)
        if grade == 2:
            return (normalized_coherence_spectrum(b2.T.product(b2))
                    if b2 is not None and b2.shape[1] else np.empty(0))
        raise ValueError(f"grade must be 0, 1, or 2; got {grade}")
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


# Rank 2 between grade scalar



#: Spectrum norms are floored before normalization. A zero spectrum contributes a
#: zero projector; the other spectrum retains its normalized projector.
cdef f64 _LGB_FLOOR = 1e-12


cdef inline f64 _stable_norm(const f64[::1] x) noexcept:
    """Overflow/underflow-resistant Euclidean norm without Python or BLAS dispatch."""
    cdef Py_ssize_t k, n = x.shape[0]
    cdef f64 scale = 0.0
    cdef f64 ssq = 1.0
    cdef f64 ax, ratio
    for k in range(n):
        ax = fabs(x[k])
        if ax == 0.0:
            continue
        if scale < ax:
            ratio = scale / ax
            ssq = 1.0 + ssq * ratio * ratio
            scale = ax
        else:
            ratio = ax / scale
            ssq += ratio * ratio
    return scale * sqrt(ssq) if scale > 0.0 else 0.0


cdef inline f64 _dot_prefix(const f64[::1] a, const f64[::1] b,
                             Py_ssize_t n) noexcept:
    cdef Py_ssize_t k
    cdef f64 total = 0.0
    for k in range(n):
        total += a[k] * b[k]
    return total


cdef inline void _pair_spectrum(const f64[::1] a, const f64[::1] b,
                                f64 *top, f64 *bot, f64 *frob) noexcept:
    """Closed form spectrum/Frobenius norm of two normalized rank 1 projectors.

    Contiguous memoryviews may have different lengths. The shorter spectrum is
    implicitly zero padded without allocating a padded vector.
    """
    cdef Py_ssize_t n_a = a.shape[0]
    cdef Py_ssize_t n_b = b.shape[0]
    cdef Py_ssize_t n_common = n_a if n_a < n_b else n_b
    cdef Py_ssize_t k
    cdef f64 ra = _stable_norm(a)
    cdef f64 rb = _stable_norm(b)
    cdef f64 na = ra if ra > _LGB_FLOOR else _LGB_FLOOR
    cdef f64 nb = rb if rb > _LGB_FLOOR else _LGB_FLOOR
    cdef f64 al = (ra / na) * (ra / na)
    cdef f64 be = (rb / nb) * (rb / nb)
    cdef f64 s2 = 1.0
    cdef f64 tr, disc, q, dot_ab, coeff, perp, perp2

    if ra > 0.0 and rb > 0.0:
        # Compute ||b - proj_a(b)||^2 directly without allocating normalized vectors
        # or a residual array. Unequal lengths contribute the zero padded tails.
        dot_ab = _dot_prefix(a, b, n_common)
        coeff = dot_ab / (ra * ra)
        perp2 = 0.0
        for k in range(n_common):
            perp = b[k] - coeff * a[k]
            perp2 += perp * perp
        for k in range(n_common, n_b):
            perp2 += b[k] * b[k]
        for k in range(n_common, n_a):
            perp = -coeff * a[k]
            perp2 += perp * perp
        s2 = perp2 / (rb * rb)
        if s2 > 1.0:
            s2 = 1.0
        elif s2 < 0.0:
            s2 = 0.0

    tr = al - be
    disc = sqrt(tr * tr + 4.0 * al * be * s2)
    top[0] = 0.5 * (tr + disc)
    bot[0] = 0.5 * (tr - disc)
    # alpha^2 + beta^2 - 2 alpha beta cos^2, rearranged as a sum of
    # nonnegative terms to avoid cancellation near parallel spectra.
    q = tr * tr + 2.0 * al * be * s2
    frob[0] = sqrt(q) if q > 0.0 else 0.0


def l_gb_scalar(np.ndarray[f64, ndim=1] spec_d,
                np.ndarray[f64, ndim=1] spec_d1):
    """Scalar coupling between two grade spectra.

    Computes the Frobenius distance between the rank 1 outer product
    projections of the normalized coherence spectra at adjacent grades.

    Returns

    coupling : f64
        Nonneg scalar; 0 means the two grades have identical spectral shape.
    """
    if spec_d.shape[0] == 0 or spec_d1.shape[0] == 0:
        return 0.0
    cdef f64 top, bot, frob
    cdef const f64[::1] a = spec_d
    cdef const f64[::1] b = spec_d1
    _pair_spectrum(a, b, &top, &bot, &frob)
    return float(frob)


# Rank 4 within grade channel tensor


def l_gb_channel_tensor(list hats_A, list hats_B=None):
    """4×4 channel coupling tensor.

    For each pair (i, j), computes the Frobenius distance between the
    normalized rank 1 projections of channel i in hats_A and channel j
    in hats_B.

    Convention for hats_A == hats_B (self tensor): diagonal entries are
    identically zero (channel matches itself), off diagonals encode
    within grade structure.

    A zero spectrum retains a zero projector. Comparing it with a unit
    projector gives distance 1; two zero projectors give distance 0.
    """
    cdef bint self_tensor = hats_B is None or hats_B is hats_A
    if hats_B is None:
        hats_B = hats_A
    cdef int n_A = len(hats_A)
    cdef int n_B = len(hats_B)
    cdef np.ndarray[f64, ndim=2] T = np.zeros((n_A, n_B), dtype=np.float64)
    cdef int i, j

    # Compute each channel spectrum once and reuse it across channel pairs.
    specs_A = [normalized_coherence_spectrum(hats_A[i]) for i in range(n_A)]
    specs_B = specs_A if self_tensor else [
        normalized_coherence_spectrum(hats_B[j]) for j in range(n_B)]

    if self_tensor and n_A == n_B:
        # Frobenius projector distance is symmetric, so a self tensor needs only the
        # strict upper triangle. The diagonal stays exactly zero by construction.
        for i in range(n_A):
            for j in range(i + 1, n_B):
                T[i, j] = l_gb_scalar(specs_A[i], specs_B[j])
                T[j, i] = T[i, j]
    else:
        for i in range(n_A):
            for j in range(n_B):
                # Preserve the historical cross tensor convention: matching channel
                # indices are left at zero even when hats_B is a different list.
                if i == j:
                    continue
                T[i, j] = l_gb_scalar(specs_A[i], specs_B[j])
    return T


# Sweep across all adjacent grade pairs


def l_gb_tower(list B_list):
    """Sweep l_gb across all adjacent grade pairs in a relational complex.

    Parameters

    B_list : list of ndarray or sparse matrix
        [B_0, B_1, B_2, ...] boundary operators. B_d has shape
        (n_{d-1}, n_d). Pass None for empty grades.

    Returns

    results : list of dict
        One dict per adjacent pair, with top_eig, bot_eig, spread, frob,
        localization, L_gb and pair.
    """
    cdef int n_grades = len(B_list)
    cdef int d
    cdef f64 na, nb
    cdef f64 c_top, c_bot, c_frob
    cdef Py_ssize_t i, j, L_size
    cdef f64 dena, denb, value, dot_a, dot_b
    cdef const f64[::1] av, bv, vv
    cdef f64[:, ::1] difference, absolute
    if n_grades == 0:
        return []
    from rexgraph.native_sparse import as_native
    from rexgraph.core._linalg import largest_eigenpair
    sparse = any(b is not None and (hasattr(b, 'tocsr') or hasattr(b, 'dual')
                                   or hasattr(b, 'row_ptr')) for b in B_list)
    if sparse:
        B_list = [as_native(b) if b is not None else None for b in B_list]

    # Build spectrum at each grade 0..n_grades using Hodge Laplacian
    specs = []
    for d in range(n_grades + 1):
        # Down part: B_d^T @ B_d (only if d >= 1)
        L_down = None
        if d >= 1 and (d - 1) < n_grades:
            B_d = B_list[d - 1]
            if B_d is not None and all(B_d.shape):
                if sparse:
                    L_down = B_d.T.product(B_d)
                else:
                    B_d = np.ascontiguousarray(B_d, dtype=np.float64)
                    L_down = B_d.T @ B_d
        # Up part: B_{d+1} @ B_{d+1}^T
        L_up = None
        if d < n_grades:
            B_dp1 = B_list[d]
            if B_dp1 is not None and all(B_dp1.shape):
                if sparse:
                    L_up = B_dp1.product(B_dp1.T)
                else:
                    B_dp1 = np.ascontiguousarray(B_dp1, dtype=np.float64)
                    L_up = B_dp1 @ B_dp1.T

        if L_down is None and L_up is None:
            specs.append(np.zeros(1, dtype=np.float64))
        elif L_down is None:
            specs.append(normalized_coherence_spectrum(L_up))
        elif L_up is None:
            specs.append(normalized_coherence_spectrum(L_down))
        elif sparse:
            if L_down.shape != L_up.shape:
                raise ValueError("adjacent boundary axes do not match")
            specs.append(normalized_coherence_spectrum(L_down.add(L_up)))
        else:
            # Match dimensions by zero padding the smaller
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
        if not can_allocate_dense_f64(L_size, L_size):
            raise MemoryError("grade projector difference exceeds the dense allocation budget")
        a = np.pad(sd, (0, L_size - len(sd)))
        b = np.pad(sd1, (0, L_size - len(sd1)))
        na = max(_stable_norm(a), _LGB_FLOOR)
        nb = max(_stable_norm(b), _LGB_FLOOR)
        # the spectrum in closed form: see _pair_spectrum. No eigensolver, and the
        # L x L outer products are never formed for these three.
        _pair_spectrum(a, b, &c_top, &c_bot, &c_frob)
        top_eig = float(c_top)
        bot_eig = float(c_bot)
        frob = float(c_frob)

        # Localization reads the ENTRYWISE absolute value, which is not rank 2 and
        # has no closed form, so this one pair of outer products is still built.
        L_gb = np.empty((L_size, L_size), dtype=np.float64)
        abs_L = np.empty((L_size, L_size), dtype=np.float64)
        av = a; bv = b; difference = L_gb; absolute = abs_L
        dena = na * na; denb = nb * nb
        with nogil:
            for i in range(L_size):
                for j in range(i + 1):
                    value = av[i] * av[j] / dena - bv[i] * bv[j] / denb
                    difference[i, j] = value
                    difference[j, i] = value
                    absolute[i, j] = fabs(value)
                    absolute[j, i] = fabs(value)
        try:
            _top, v_top = largest_eigenpair(abs_L)
            vv = v_top
            dot_a = _dot_prefix(vv, av, L_size) / na
            dot_b = _dot_prefix(vv, bv, L_size) / nb
            ma = dot_a * dot_a
            mb = dot_b * dot_b
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
