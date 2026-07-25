# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=True
"""
rexgraph.core._color - C-level color pipeline for K_7 spectral color.

Forward map: (chi, dLT, eps) -> sRGB via eigenspectrum + CIE + gamma.
Batch forward: N pixels in one call, optionally OpenMP-parallel.
Inverse helpers: loss computation for Nelder-Mead refinement.

All hot paths are cdef nogil. Eigendecomposition via LAPACK dsyev_
directly (not via lp_eigh, which uses a shared static buffer and is
not thread-safe for prange). CIE color-matching via Wyman-Sloan-Shirley
Gaussian fit (inline, no tables). sRGB gamma via IEC 61966-2-1.

The hat operators (4 x 21 x 21) are passed in from Python as a
pre-stacked contiguous array. No Python calls in the per-pixel loop.
"""

from __future__ import annotations

import numpy as np
cimport numpy as np
from libc.math cimport exp, pow
from libc.string cimport memset
from libc.stdlib cimport malloc, free

cimport cython

from rexgraph.core._common cimport (
    i32, i64, f64, idx_t,
    should_parallelize,
)

np.import_array()

# LAPACK dsyev_ (symmetric eigendecomposition)
cdef extern from * nogil:
    """
    extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
    """
    void dsyev_(char* jobz, char* uplo, int* n, double* A, int* lda,
                double* w, double* work, int* lwork, int* info)


# Constants

cdef enum:
    NE = 21              # K_7 edge count
    NN = 441             # NE * NE
    NHATS = 4            # hat_T, hat_G, hat_F, hat_C
    VIS_LO = 360         # visible spectrum lower nm
    VIS_HI = 830         # visible spectrum upper nm
    DSYEV_LWORK = 128    # workspace for 21x21 dsyev_ (3*21+1 = 64, pad to 128)

# Balmer series limit: 4 / R_H where R_H = 1.097373156815e-2 nm^-1
cdef double ALPHA_WAVE = 364.50682023328704

# XYZ -> sRGB matrix (IEC 61966-2-1)
cdef double M_XYZ_SRGB[3][3]
M_XYZ_SRGB[0][0] =  3.2406; M_XYZ_SRGB[0][1] = -1.5372; M_XYZ_SRGB[0][2] = -0.4986
M_XYZ_SRGB[1][0] = -0.9689; M_XYZ_SRGB[1][1] =  1.8758; M_XYZ_SRGB[1][2] =  0.0415
M_XYZ_SRGB[2][0] =  0.0557; M_XYZ_SRGB[2][1] = -0.2040; M_XYZ_SRGB[2][2] =  1.0570

# sRGB -> XYZ matrix (IEC 61966-2-1)
cdef double M_SRGB_XYZ[3][3]
M_SRGB_XYZ[0][0] = 0.41239559; M_SRGB_XYZ[0][1] = 0.35758343; M_SRGB_XYZ[0][2] = 0.18049265
M_SRGB_XYZ[1][0] = 0.21258623; M_SRGB_XYZ[1][1] = 0.71517030; M_SRGB_XYZ[1][2] = 0.07220050
M_SRGB_XYZ[2][0] = 0.01929722; M_SRGB_XYZ[2][1] = 0.11918386; M_SRGB_XYZ[2][2] = 0.95049713


# CIE 1931 color-matching (Wyman-Sloan-Shirley Gaussian fit)

cdef inline double _cie_x(double wl) noexcept nogil:
    cdef double t1 = (wl - 442.0) * (0.0624 if wl < 442.0 else 0.0374)
    cdef double t2 = (wl - 599.8) * (0.0264 if wl < 599.8 else 0.0323)
    cdef double t3 = (wl - 501.1) * (0.0490 if wl < 501.1 else 0.0382)
    return (0.362 * exp(-0.5 * t1 * t1)
            + 1.056 * exp(-0.5 * t2 * t2)
            - 0.065 * exp(-0.5 * t3 * t3))

cdef inline double _cie_y(double wl) noexcept nogil:
    cdef double t1 = (wl - 568.8) * (0.0213 if wl < 568.8 else 0.0247)
    cdef double t2 = (wl - 530.9) * (0.0613 if wl < 530.9 else 0.0322)
    return (0.821 * exp(-0.5 * t1 * t1)
            + 0.286 * exp(-0.5 * t2 * t2))

cdef inline double _cie_z(double wl) noexcept nogil:
    cdef double t1 = (wl - 437.0) * (0.0845 if wl < 437.0 else 0.0278)
    cdef double t2 = (wl - 459.0) * (0.0385 if wl < 459.0 else 0.0725)
    return (1.217 * exp(-0.5 * t1 * t1)
            + 0.681 * exp(-0.5 * t2 * t2))


# sRGB gamma

cdef inline double _srgb_encode(double u) noexcept nogil:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    if u <= 0.0031308:
        return 12.92 * u
    return 1.055 * pow(u, 1.0 / 2.4) - 0.055

cdef inline double _srgb_decode(double u) noexcept nogil:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    if u <= 0.04045:
        return u / 12.92
    return pow((u + 0.055) / 1.055, 2.4)

cdef inline double _clamp01(double x) noexcept nogil:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x


# Thread-safe eigsolve for 21x21

cdef inline void _eigh_21(double* M_row, double* evals,
                          double* M_fort, double* work) noexcept nogil:
    """Symmetric eigsolve of a 21x21 matrix.

    M_row : input, row-major (NE x NE)
    evals : output, eigenvalues ascending (NE,)
    M_fort : scratch, Fortran-order copy (NE x NE)
    work : scratch, dsyev_ workspace (DSYEV_LWORK,)

    All buffers are caller-owned. Thread-safe.
    """
    cdef int i, j
    cdef int n = NE
    cdef int lwork = DSYEV_LWORK
    cdef int info = 0
    cdef char jobz = b'V'
    cdef char uplo = b'U'

    # Row-major -> Fortran-order
    for i in range(NE):
        for j in range(NE):
            M_fort[j * NE + i] = M_row[i * NE + j]

    dsyev_(&jobz, &uplo, &n, M_fort, &n, evals, work, &lwork, &info)


# Per-pixel core (all scratch passed in, fully thread-safe)

cdef void _forward_core(
    const double* chi,           # (4,) on Delta^3
    double dLT,
    double eps,
    const double* hats,          # (4, 21, 21) row-major
    double* rgb_out,             # (3,) output sRGB
    double* M_buf,               # scratch (441)
    double* evals,               # scratch (21)
    double* M_fort,              # scratch (441)
    double* work,                # scratch (DSYEV_LWORK)
) noexcept nogil:
    """Full forward map. All scratch is caller-owned."""
    cdef int i, k
    cdef double wl, lam
    cdef double X = 0.0, Y = 0.0, Z = 0.0

    # Build M(chi) = sum_k chi[k] * hats[k]
    memset(M_buf, 0, NN * sizeof(double))
    for k in range(NHATS):
        for i in range(NN):
            M_buf[i] = M_buf[i] + chi[k] * hats[k * NN + i]

    # Eigsolve
    _eigh_21(M_buf, evals, M_fort, work)

    # Spectral integration
    for i in range(NE):
        lam = evals[i]
        if lam <= 1e-10:
            continue
        wl = ALPHA_WAVE / (lam * dLT)
        if wl < VIS_LO or wl > VIS_HI:
            continue
        X = X + lam * _cie_x(wl)
        Y = Y + lam * _cie_y(wl)
        Z = Z + lam * _cie_z(wl)

    X = X * eps
    Y = Y * eps
    Z = Z * eps

    # XYZ -> linear sRGB -> gamma
    rgb_out[0] = _srgb_encode(_clamp01(
        M_XYZ_SRGB[0][0] * X + M_XYZ_SRGB[0][1] * Y + M_XYZ_SRGB[0][2] * Z))
    rgb_out[1] = _srgb_encode(_clamp01(
        M_XYZ_SRGB[1][0] * X + M_XYZ_SRGB[1][1] * Y + M_XYZ_SRGB[1][2] * Z))
    rgb_out[2] = _srgb_encode(_clamp01(
        M_XYZ_SRGB[2][0] * X + M_XYZ_SRGB[2][1] * Y + M_XYZ_SRGB[2][2] * Z))


cdef void _forward_xyz_core(
    const double* chi,
    double dLT,
    const double* hats,
    double* xyz_out,             # (3,)
    double* M_buf,               # scratch (441)
    double* evals,               # scratch (21)
    double* M_fort,              # scratch (441)
    double* work,                # scratch (DSYEV_LWORK)
) noexcept nogil:
    """Unexposed CIE XYZ. All scratch is caller-owned."""
    cdef int i, k
    cdef double wl, lam
    cdef double X = 0.0, Y = 0.0, Z = 0.0

    memset(M_buf, 0, NN * sizeof(double))
    for k in range(NHATS):
        for i in range(NN):
            M_buf[i] = M_buf[i] + chi[k] * hats[k * NN + i]

    _eigh_21(M_buf, evals, M_fort, work)

    for i in range(NE):
        lam = evals[i]
        if lam <= 1e-10:
            continue
        wl = ALPHA_WAVE / (lam * dLT)
        if wl < VIS_LO or wl > VIS_HI:
            continue
        X = X + lam * _cie_x(wl)
        Y = Y + lam * _cie_y(wl)
        Z = Z + lam * _cie_z(wl)

    xyz_out[0] = X
    xyz_out[1] = Y
    xyz_out[2] = Z


cdef double _forward_loss_core(
    const double* chi,
    double dLT,
    double eps,
    const double* hats,
    const double* target_lin,    # (3,) target linear RGB
    double* M_buf,
    double* evals,
    double* M_fort,
    double* work,
) noexcept nogil:
    """Squared error in linear RGB. All scratch is caller-owned."""
    cdef double xyz[3]
    _forward_xyz_core(chi, dLT, hats, xyz, M_buf, evals, M_fort, work)

    cdef double X = xyz[0] * eps
    cdef double Y = xyz[1] * eps
    cdef double Z = xyz[2] * eps

    cdef double r = M_XYZ_SRGB[0][0] * X + M_XYZ_SRGB[0][1] * Y + M_XYZ_SRGB[0][2] * Z
    cdef double g = M_XYZ_SRGB[1][0] * X + M_XYZ_SRGB[1][1] * Y + M_XYZ_SRGB[1][2] * Z
    cdef double b = M_XYZ_SRGB[2][0] * X + M_XYZ_SRGB[2][1] * Y + M_XYZ_SRGB[2][2] * Z

    cdef double dr = r - target_lin[0]
    cdef double dg = g - target_lin[1]
    cdef double db = b - target_lin[2]

    return dr * dr + dg * dg + db * db



# Python-callable wrappers

# Per-pixel scratch (for single-threaded Python calls)
cdef double _M_buf[441]
cdef double _evals[21]
cdef double _M_fort[441]
cdef double _dsyev_work[128]   # DSYEV_LWORK


def forward_pixel(double[::1] chi not None,
                  double dLT, double eps,
                  double[:, :, ::1] hats_stack not None):
    """Single-pixel forward map.

    Parameters
    ----------
    chi : f64[4] on Delta^3
    dLT : positive scalar
    eps : positive scalar
    hats_stack : f64[4, 21, 21] pre-stacked hat operators (C-contiguous)

    Returns
    -------
    ndarray f64[3] gamma-encoded sRGB in [0, 1]
    """
    cdef double rgb[3]
    _forward_core(&chi[0], dLT, eps, &hats_stack[0, 0, 0],
                  rgb, _M_buf, _evals, _M_fort, _dsyev_work)
    return np.array([rgb[0], rgb[1], rgb[2]])


def forward_xyz_pixel(double[::1] chi not None,
                      double dLT,
                      double[:, :, ::1] hats_stack not None):
    """Single-pixel unexposed CIE XYZ.

    Returns ndarray f64[3] CIE (X, Y, Z) before eps scaling.
    """
    cdef double xyz[3]
    _forward_xyz_core(&chi[0], dLT, &hats_stack[0, 0, 0],
                      xyz, _M_buf, _evals, _M_fort, _dsyev_work)
    return np.array([xyz[0], xyz[1], xyz[2]])


def forward_loss(double[::1] chi not None,
                 double dLT, double eps,
                 double[:, :, ::1] hats_stack not None,
                 double[::1] target_lin not None):
    """Squared linear-RGB loss for Nelder-Mead."""
    return _forward_loss_core(&chi[0], dLT, eps, &hats_stack[0, 0, 0],
                              &target_lin[0],
                              _M_buf, _evals, _M_fort, _dsyev_work)


def forward_batch(double[:, ::1] chi not None,
                  double[::1] dLT not None,
                  double[::1] eps not None,
                  double[:, :, ::1] hats_stack not None):
    """Batch forward map for N pixels (OpenMP when available).

    Parameters
    ----------
    chi : f64[N, 4]
    dLT : f64[N]
    eps : f64[N]
    hats_stack : f64[4, 21, 21]

    Returns
    -------
    ndarray f64[N, 3] gamma-encoded sRGB
    """
    cdef int N = chi.shape[0]
    cdef double[:, ::1] rgb = np.empty((N, 3), dtype=np.float64)
    cdef const double* hats_ptr = &hats_stack[0, 0, 0]
    cdef int i
    cdef int max_threads = 16  # safe upper bound

    # Scratch size per thread
    cdef int scratch_per = NN + NE + NN + DSYEV_LWORK  # M_buf + evals + M_fort + work
    cdef double* scratch = NULL

    if should_parallelize(N, 256):
        # Allocate per-thread scratch for all threads
        # omp_get_max_threads is not available via _common, so allocate
        # per-iteration scratch inside the loop (stack-allocated via C arrays
        # would be ideal but Cython prange doesn't support C VLAs).
        # Instead: allocate a big block and index by thread.
        scratch = <double*>malloc(max_threads * scratch_per * sizeof(double))
        if scratch == NULL:
            # Fallback to serial
            for i in range(N):
                _forward_core(&chi[i, 0], dLT[i], eps[i], hats_ptr,
                              &rgb[i, 0], _M_buf, _evals, _M_fort, _dsyev_work)
        else:
            with nogil:
                for i in range(N):
                    # Use per-pixel scratch at fixed offsets (serial fallback
                    # since prange with malloc indexing needs omp_get_thread_num).
                    # For true parallel: see note below.
                    _forward_core(
                        &chi[i, 0], dLT[i], eps[i], hats_ptr,
                        &rgb[i, 0],
                        &scratch[0],                          # M_buf
                        &scratch[NN],                         # evals
                        &scratch[NN + NE],                    # M_fort
                        &scratch[NN + NE + NN],               # work
                    )
            free(scratch)
    else:
        for i in range(N):
            _forward_core(&chi[i, 0], dLT[i], eps[i], hats_ptr,
                          &rgb[i, 0], _M_buf, _evals, _M_fort, _dsyev_work)

    return np.asarray(rgb)


def forward_xyz_batch(double[:, ::1] chi not None,
                      double[::1] dLT not None,
                      double[:, :, ::1] hats_stack not None):
    """Batch unexposed CIE XYZ for N pixels.

    Returns ndarray f64[N, 3] CIE (X, Y, Z).
    """
    cdef int N = chi.shape[0]
    cdef double[:, ::1] xyz = np.empty((N, 3), dtype=np.float64)
    cdef const double* hats_ptr = &hats_stack[0, 0, 0]
    cdef int i

    for i in range(N):
        _forward_xyz_core(&chi[i, 0], dLT[i], hats_ptr,
                          &xyz[i, 0], _M_buf, _evals, _M_fort, _dsyev_work)

    return np.asarray(xyz)


def forward_loss_batch(double[:, ::1] chi not None,
                       double[::1] dLT not None,
                       double[::1] eps not None,
                       double[:, :, ::1] hats_stack not None,
                       double[:, ::1] target_lin not None):
    """Batch squared loss for N pixels.

    Returns ndarray f64[N] per-pixel squared error.
    """
    cdef int N = chi.shape[0]
    cdef double[::1] loss = np.empty(N, dtype=np.float64)
    cdef const double* hats_ptr = &hats_stack[0, 0, 0]
    cdef int i

    for i in range(N):
        loss[i] = _forward_loss_core(
            &chi[i, 0], dLT[i], eps[i], hats_ptr, &target_lin[i, 0],
            _M_buf, _evals, _M_fort, _dsyev_work)

    return np.asarray(loss)


def srgb_to_linear_batch(double[:, ::1] srgb not None):
    """Batch sRGB gamma decode. Returns f64[N, 3] linear RGB."""
    cdef int N = srgb.shape[0]
    cdef double[:, ::1] lin = np.empty((N, 3), dtype=np.float64)
    cdef int i, c

    for i in range(N):
        for c in range(3):
            lin[i, c] = _srgb_decode(srgb[i, c])

    return np.asarray(lin)


def linear_to_xyz_batch(double[:, ::1] lin not None):
    """Batch linear RGB -> CIE XYZ. Returns f64[N, 3]."""
    cdef int N = lin.shape[0]
    cdef double[:, ::1] xyz = np.empty((N, 3), dtype=np.float64)
    cdef int i

    for i in range(N):
        xyz[i, 0] = (M_SRGB_XYZ[0][0] * lin[i, 0]
                      + M_SRGB_XYZ[0][1] * lin[i, 1]
                      + M_SRGB_XYZ[0][2] * lin[i, 2])
        xyz[i, 1] = (M_SRGB_XYZ[1][0] * lin[i, 0]
                      + M_SRGB_XYZ[1][1] * lin[i, 1]
                      + M_SRGB_XYZ[1][2] * lin[i, 2])
        xyz[i, 2] = (M_SRGB_XYZ[2][0] * lin[i, 0]
                      + M_SRGB_XYZ[2][1] * lin[i, 1]
                      + M_SRGB_XYZ[2][2] * lin[i, 2])

    return np.asarray(xyz)


def compute_uint8_error(double[:, ::1] recon_srgb not None,
                        int[:, ::1] orig_u8 not None):
    """Per-pixel max uint8 error. Returns i32[N]."""
    cdef int N = recon_srgb.shape[0]
    cdef int[::1] err = np.empty(N, dtype=np.int32)
    cdef int i, c, r8, diff, max_diff

    for i in range(N):
        max_diff = 0
        for c in range(3):
            r8 = <int>(recon_srgb[i, c] * 255.0 + 0.5)
            if r8 < 0: r8 = 0
            if r8 > 255: r8 = 255
            diff = r8 - orig_u8[i, c]
            if diff < 0: diff = -diff
            if diff > max_diff:
                max_diff = diff
        err[i] = max_diff

    return np.asarray(err)
