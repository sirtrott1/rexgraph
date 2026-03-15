# rexgraph/core/_linalg.pxd
# cython: language_level=3
"""
LAPACK/BLAS interface declarations for the rexgraph Cython layer.

Every module that needs eigensolve, SVD, lstsq, or matrix multiply
cimports from this file. All functions are noexcept nogil.

Workspace is statically allocated for matrices up to MAX_DIM.
For larger matrices, dynamic allocation is used.
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport fabs, sqrt

ctypedef double f64

# Maximum dimension for static workspace allocation.
# Matrices larger than this use malloc'd workspace.
cdef enum:
    MAX_DIM = 2048
    MAX_DIM_SQ = 4194304  # 2048^2
    WORK_SIZE = 8192       # dsyev workspace (3*MAX_DIM + padding)
    IWORK_SIZE = 8192      # dgelsd iwork

# LAPACK extern declarations

cdef extern from * nogil:
    """
    extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
    extern void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);
    extern void dgelsd_(int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*, int*);
    extern void dpotrf_(char*, int*, double*, int*, int*);
    extern void dpotrs_(char*, int*, int*, double*, int*, double*, int*, int*);
    """
    # Symmetric eigensolve
    void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
                double* w, double* work, int* lwork, int* info)
    # General SVD
    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
                 double* s, double* u, int* ldu, double* vt, int* ldvt,
                 double* work, int* lwork, int* info)
    # Least squares via SVD
    void dgelsd_(int* m, int* n, int* nrhs, double* a, int* lda,
                 double* b, int* ldb, double* s, double* rcond, int* rank,
                 double* work, int* lwork, int* iwork, int* liwork, int* info)
    # Cholesky factorize
    void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info)
    # Cholesky solve
    void dpotrs_(char* uplo, int* n, int* nrhs, double* a, int* lda,
                 double* b, int* ldb, int* info)


# BLAS extern declarations

cdef extern from * nogil:
    """
    extern void cblas_dgemm(int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    extern void cblas_dgemv(int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    extern void cblas_dsymv(int, int, int, double, const double*, int, const double*, int, double, double*, int);
    extern double cblas_ddot(int, const double*, int, const double*, int);
    extern double cblas_dnrm2(int, const double*, int);
    extern void cblas_dscal(int, double, double*, int);
    extern void cblas_daxpy(int, double, const double*, int, double*, int);
    extern void cblas_dcopy(int, const double*, int, double*, int);
    """
    # Matrix-matrix: C = alpha*op(A)*op(B) + beta*C
    void cblas_dgemm(int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb,
                     double beta, double* C, int ldc) nogil
    # Matrix-vector: y = alpha*op(A)*x + beta*y
    void cblas_dgemv(int Order, int Trans,
                     int M, int N,
                     double alpha, const double* A, int lda,
                     const double* x, int incx,
                     double beta, double* y, int incy) nogil
    # Symmetric matrix-vector: y = alpha*A*x + beta*y
    void cblas_dsymv(int Order, int Uplo, int N,
                     double alpha, const double* A, int lda,
                     const double* x, int incx,
                     double beta, double* y, int incy) nogil
    # Dot product
    double cblas_ddot(int N, const double* x, int incx,
                      const double* y, int incy) nogil
    # 2-norm
    double cblas_dnrm2(int N, const double* x, int incx) nogil
    # Scale: x = alpha*x
    void cblas_dscal(int N, double alpha, double* x, int incx) nogil
    # AXPY: y = alpha*x + y
    void cblas_daxpy(int N, double alpha, const double* x, int incx,
                     double* y, int incy) nogil
    # Copy: y = x
    void cblas_dcopy(int N, const double* x, int incx,
                     double* y, int incy) nogil

# CBLAS constants
cdef enum:
    CblasRowMajor = 101
    CblasColMajor = 102
    CblasNoTrans = 111
    CblasTrans = 112
    CblasUpper = 121
    CblasLower = 122


# Inline wrappers — zero-overhead calls from any cimporting module

cdef inline void lp_eigh(double* A, double* evals, int n) noexcept nogil:
    """Symmetric eigendecomposition. A overwritten with eigenvectors (column-major).
    evals sorted ascending. A must be Fortran-order (column-major)."""
    cdef char jobz = b'V'
    cdef char uplo = b'U'
    cdef int info = 0
    cdef int lwork
    cdef double work_query
    cdef double* work

    # Workspace query
    lwork = -1
    dsyev_(&jobz, &uplo, &n, A, &n, evals, &work_query, &lwork, &info)
    lwork = <int>work_query
    if lwork < 3 * n + 1:
        lwork = 3 * n + 1

    if lwork <= WORK_SIZE:
        # Use module-level static buffer (declared in .pyx)
        dsyev_(&jobz, &uplo, &n, A, &n, evals, _lp_work, &lwork, &info)
    else:
        work = <double*>malloc(lwork * sizeof(double))
        if work != NULL:
            dsyev_(&jobz, &uplo, &n, A, &n, evals, work, &lwork, &info)
            free(work)


cdef inline void lp_svd(double* A, double* S, double* U, double* Vt,
                         int m, int n) noexcept nogil:
    """General SVD: A = U * diag(S) * Vt. A is m x n column-major, overwritten.
    S has min(m,n) entries. U is m x m, Vt is n x n."""
    cdef char jobu = b'A'
    cdef char jobvt = b'A'
    cdef int info = 0
    cdef int lwork
    cdef double work_query
    cdef double* work
    cdef int mn = m if m < n else n

    lwork = -1
    dgesvd_(&jobu, &jobvt, &m, &n, A, &m, S, U, &m, Vt, &n,
            &work_query, &lwork, &info)
    lwork = <int>work_query
    if lwork < 1:
        lwork = 5 * (m + n)

    work = <double*>malloc(lwork * sizeof(double))
    if work != NULL:
        dgesvd_(&jobu, &jobvt, &m, &n, A, &m, S, U, &m, Vt, &n,
                work, &lwork, &info)
        free(work)


cdef inline int lp_lstsq(double* A, double* B, int m, int n, int nrhs,
                          double* S, int* rank_out) noexcept nogil:
    """Least squares via SVD: min ||A*X - B||. A is m x n, B is m x nrhs.
    Both column-major. Solution overwrites B. Returns info."""
    cdef double rcond = -1.0  # machine precision
    cdef int info = 0
    cdef int lwork
    cdef double work_query
    cdef double* work
    cdef int liwork
    cdef int iwork_query
    cdef int* iwork
    cdef int mn = m if m < n else n

    # Workspace query
    lwork = -1
    liwork = -1
    dgelsd_(&m, &n, &nrhs, A, &m, B, &m, S, &rcond, rank_out,
            &work_query, &lwork, &iwork_query, &liwork, &info)
    lwork = <int>work_query
    liwork = iwork_query

    work = <double*>malloc(lwork * sizeof(double))
    iwork = <int*>malloc(liwork * sizeof(int))
    if work != NULL and iwork != NULL:
        dgelsd_(&m, &n, &nrhs, A, &m, B, &m, S, &rcond, rank_out,
                work, &lwork, iwork, &liwork, &info)
    if work != NULL: free(work)
    if iwork != NULL: free(iwork)
    return info


cdef inline void bl_gemm_nn(const double* A, const double* B, double* C,
                             int M, int N, int K) noexcept nogil:
    """C = A @ B. All row-major. C must be pre-allocated M x N."""
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N)


cdef inline void bl_gemm_nt(const double* A, const double* B, double* C,
                             int M, int N, int K) noexcept nogil:
    """C = A @ B^T. All row-major. C must be pre-allocated M x N."""
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0, A, K, B, K, 0.0, C, N)


cdef inline void bl_gemm_tn(const double* A, const double* B, double* C,
                             int M, int N, int K) noexcept nogil:
    """C = A^T @ B. All row-major. C must be pre-allocated M x N."""
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0, A, M, B, N, 0.0, C, N)


cdef inline void bl_gemv_n(const double* A, const double* x, double* y,
                            int M, int N) noexcept nogil:
    """y = A @ x. A is M x N row-major."""
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1)


cdef inline void bl_gemv_t(const double* A, const double* x, double* y,
                            int M, int N) noexcept nogil:
    """y = A^T @ x. A is M x N row-major, result is N-vector."""
    cblas_dgemv(CblasRowMajor, CblasTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1)


cdef inline void bl_symv(const double* A, const double* x, double* y,
                          int N) noexcept nogil:
    """y = A @ x where A is symmetric N x N row-major."""
    cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0, A, N, x, 1, 0.0, y, 1)


cdef inline double bl_dot(const double* x, const double* y, int N) noexcept nogil:
    """Dot product x . y."""
    return cblas_ddot(N, x, 1, y, 1)


cdef inline double bl_nrm2(const double* x, int N) noexcept nogil:
    """Euclidean norm ||x||."""
    return cblas_dnrm2(N, x, 1)


cdef inline void bl_axpy(double alpha, const double* x, double* y, int N) noexcept nogil:
    """y = alpha*x + y."""
    cblas_daxpy(N, alpha, x, 1, y, 1)


cdef inline void bl_scal(double alpha, double* x, int N) noexcept nogil:
    """x = alpha * x."""
    cblas_dscal(N, alpha, x, 1)


# High-level composed operations

cdef inline void spectral_pinv(const double* evals, const double* evecs,
                                double* out, int n, double tol) noexcept nogil:
    """RL^+ = sum_{lam>tol} (1/lam) v v^T. out must be n x n, zeroed.

    evecs is n x n row-major where evecs[i*n+k] = component i of eigenvector k
    (i.e., after column-major dsyev_ output is reinterpreted row-major,
    evecs[k, :] is eigenvector k in Fortran layout = column k in C layout).

    For dsyev_ Fortran output stored column-major: A[i + j*n] = evecs[i][j]
    When read as row-major: row i, col j = A[i*n + j] = Fortran A[j + i*n] = evecs[j][i]
    So row-major evecs[:, k] = Fortran column k = eigenvector k. Correct.
    """
    cdef int k, i, j
    cdef double inv_lam, vi, vj
    for k in range(n):
        if evals[k] > tol:
            inv_lam = 1.0 / evals[k]
            for i in range(n):
                vi = evecs[i * n + k] * inv_lam
                for j in range(i, n):
                    vj = evecs[j * n + k]
                    out[i * n + j] += vi * vj
                    if i != j:
                        out[j * n + i] += vi * vj


cdef inline void spectral_pinv_matvec(const double* evals, const double* evecs,
                                       const double* x, double* out,
                                       int n, double tol) noexcept nogil:
    """out = RL^+ @ x via spectral decomposition. No matrix materialization."""
    cdef int k, i
    cdef double coeff
    memset(out, 0, n * sizeof(double))
    for k in range(n):
        if evals[k] > tol:
            # Project x onto eigenvector k
            coeff = 0.0
            for i in range(n):
                coeff += evecs[i * n + k] * x[i]
            coeff /= evals[k]
            for i in range(n):
                out[i] += coeff * evecs[i * n + k]


cdef inline int compute_rank_svd(double* A, int m, int n, double tol) noexcept nogil:
    """Matrix rank via SVD. A is m x n column-major, overwritten."""
    cdef int mn = m if m < n else n
    cdef double* S = <double*>malloc(mn * sizeof(double))
    cdef double* U = <double*>malloc(m * m * sizeof(double))
    cdef double* Vt = <double*>malloc(n * n * sizeof(double))
    cdef int rank = 0
    cdef int k

    if S == NULL or U == NULL or Vt == NULL:
        if S != NULL: free(S)
        if U != NULL: free(U)
        if Vt != NULL: free(Vt)
        return -1

    lp_svd(A, S, U, Vt, m, n)

    for k in range(mn):
        if S[k] > tol:
            rank += 1

    free(S)
    free(U)
    free(Vt)
    return rank


# Trace of square matrix (row-major)
cdef inline double mat_trace(const double* A, int n) noexcept nogil:
    cdef double tr = 0.0
    cdef int i
    for i in range(n):
        tr += A[i * n + i]
    return tr


# Diagonal extraction (row-major)
cdef inline void mat_diag(const double* A, double* d, int n) noexcept nogil:
    cdef int i
    for i in range(n):
        d[i] = A[i * n + i]


# Static workspace buffer (defined in _linalg.pyx)
cdef double* _lp_work
