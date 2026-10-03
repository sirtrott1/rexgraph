# rexgraph/core/_linalg.pxd
# cython: language_level=3
"""
LAPACK/BLAS interface declarations for the rexgraph Cython layer.

Numerical kernels release the GIL and allocate a private LAPACK workspace.
LAPACK drivers return their status; check_lapack_info raises at the Python boundary.
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.math cimport fabs, sqrt
from numpy.linalg import LinAlgError

ctypedef double f64
ctypedef double complex c128

# LAPACK extern declarations

cdef extern from * nogil:
    """
    extern void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
    extern void dsyevr_(char*, char*, char*, int*, double*, int*, double*, double*, int*, int*, double*, int*, double*, double*, int*, int*, double*, int*, int*, int*, int*);
    extern void dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);
    extern void dgelsd_(int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*);
    extern double dlamch_(char*);
    extern void zheev_(char*, char*, int*, void*, int*, double*, void*, int*, double*, int*);
    extern void zgesvd_(char*, char*, int*, int*, void*, int*, double*, void*, int*, void*, int*, void*, int*, double*, int*);
    extern void dpotrf_(char*, int*, double*, int*, int*);
    extern void dpotrs_(char*, int*, int*, double*, int*, double*, int*, int*);
    extern void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);
    extern void dorgqr_(int*, int*, int*, double*, int*, double*, double*, int*, int*);
    extern void dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
    """
    # Symmetric eigensolve
    void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
                double* w, double* work, int* lwork, int* info)
    void dsyevr_(char* jobz, char* range_, char* uplo, int* n, double* a, int* lda,
                 double* vl, double* vu, int* il, int* iu, double* abstol,
                 int* m, double* w, double* z, int* ldz, int* isuppz,
                 double* work, int* lwork, int* iwork, int* liwork, int* info)
    # General SVD
    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
                 double* s, double* u, int* ldu, double* vt, int* ldvt,
                 double* work, int* lwork, int* info)
    # Least squares via SVD
    void dgelsd_(int* m, int* n, int* nrhs, double* a, int* lda,
                 double* b, int* ldb, double* s, double* rcond, int* rank,
                 double* work, int* lwork, int* iwork, int* info)
    double dlamch_(char* cmach)
    void zheev_(char* jobz, char* uplo, int* n, void* a, int* lda,
                double* w, void* work, int* lwork, double* rwork, int* info)
    void zgesvd_(char* jobu, char* jobvt, int* m, int* n, void* a, int* lda,
                 double* s, void* u, int* ldu, void* vt, int* ldvt,
                 void* work, int* lwork, double* rwork, int* info)
    # Cholesky factorize
    void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info)
    # Cholesky solve
    void dpotrs_(char* uplo, int* n, int* nrhs, double* a, int* lda,
                 double* b, int* ldb, int* info)
    void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau,
                 double* work, int* lwork, int* info)
    void dorgqr_(int* m, int* n, int* k, double* a, int* lda, double* tau,
                 double* work, int* lwork, int* info)
    void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
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
    # Matrix matrix: C = alpha*op(A)*op(B) + beta*C
    void cblas_dgemm(int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb,
                     double beta, double* C, int ldc) nogil
    # Matrix vector: y = alpha*op(A)*x + beta*y
    void cblas_dgemv(int Order, int Trans,
                     int M, int N,
                     double alpha, const double* A, int lda,
                     const double* x, int incx,
                     double beta, double* y, int incy) nogil
    # Symmetric matrix vector: y = alpha*A*x + beta*y
    void cblas_dsymv(int Order, int Uplo, int N,
                     double alpha, const double* A, int lda,
                     const double* x, int incx,
                     double beta, double* y, int incy) nogil
    # Dot product
    double cblas_ddot(int N, const double* x, int incx,
                      const double* y, int incy) nogil
    # 2 norm
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


# Inline wrappers: zero overhead calls from any cimporting module

cdef inline void check_lapack_info(int info) except *:
    """Raise for workspace allocation failure, invalid arguments or nonconvergence."""
    if info == -1000:
        raise MemoryError("native LAPACK workspace allocation failed")
    if info < 0:
        raise ValueError(f"native LAPACK argument {-info} is invalid")
    if info > 0:
        raise LinAlgError(f"native LAPACK failed to converge (info={info})")


cdef inline int lp_eigh_mode(double* A, double* evals, int n,
                             bint vectors, char uplo) noexcept nogil:
    """Symmetric spectrum of column major A; vectors=True overwrites A with vectors."""
    cdef char jobz = b'V' if vectors else b'N'
    cdef int info = 0
    cdef int lwork
    cdef double work_query
    cdef double* work

    if n == 0:
        return 0
    lwork = -1
    dsyev_(&jobz, &uplo, &n, A, &n, evals, &work_query, &lwork, &info)
    if info != 0:
        return info
    lwork = <int>work_query
    if lwork < 3 * n + 1:
        lwork = 3 * n + 1

    work = <double*>malloc(lwork * sizeof(double))
    if work == NULL:
        return -1000
    dsyev_(&jobz, &uplo, &n, A, &n, evals, work, &lwork, &info)
    free(work)
    return info


cdef inline int lp_eigh(double* A, double* evals, int n) noexcept nogil:
    """Symmetric eigendecomposition, using the upper triangle of column major A."""
    return lp_eigh_mode(A, evals, n, True, b'U')


cdef inline int lp_svd_mode(double* A, double* S, double* U, double* Vt,
                            int m, int n, char job) noexcept nogil:
    """Column major SVD; job is A for full vectors, S for reduced vectors, N for values."""
    cdef char jobu = job
    cdef char jobvt = job
    cdef int info = 0
    cdef int lwork
    cdef double work_query
    cdef double* work
    cdef int mn = m if m < n else n
    cdef int ldvt = n if job == b'A' else (mn if job == b'S' else 1)
    cdef int ldu = m if job != b'N' else 1

    if mn == 0:
        return 0
    lwork = -1
    dgesvd_(&jobu, &jobvt, &m, &n, A, &m, S, U, &ldu, Vt, &ldvt,
            &work_query, &lwork, &info)
    if info != 0:
        return info
    lwork = <int>work_query
    if lwork < 1:
        lwork = 5 * (m + n)

    work = <double*>malloc(lwork * sizeof(double))
    if work == NULL:
        return -1000
    dgesvd_(&jobu, &jobvt, &m, &n, A, &m, S, U, &ldu, Vt, &ldvt,
            work, &lwork, &info)
    free(work)
    return info


cdef inline int lp_svd(double* A, double* S, double* U, double* Vt,
                       int m, int n) noexcept nogil:
    """Full real SVD of column major A."""
    return lp_svd_mode(A, S, U, Vt, m, n, b'A')


cdef inline int lp_heev(void* A, double* evals, int n,
                        bint vectors, char uplo) noexcept nogil:
    """Hermitian spectrum of column major complex128 A."""
    cdef char jobz = b'V' if vectors else b'N'
    cdef int info = 0, lwork = -1
    cdef c128 query
    cdef c128* work
    cdef double* rwork
    if n == 0:
        return 0
    rwork = <double*>malloc((3 * <size_t>n + 1) * sizeof(double))
    if rwork == NULL:
        return -1000
    zheev_(&jobz, &uplo, &n, A, &n, evals, &query, &lwork, rwork, &info)
    if info != 0:
        free(rwork)
        return info
    lwork = <int>query.real
    work = <c128*>malloc(lwork * sizeof(c128))
    if work == NULL:
        free(rwork)
        return -1000
    zheev_(&jobz, &uplo, &n, A, &n, evals, work, &lwork, rwork, &info)
    free(work)
    free(rwork)
    return info


cdef inline int lp_zsvd(void* A, double* S, void* U, void* Vh,
                        int m, int n, char job) noexcept nogil:
    """Complex column major SVD with full, reduced or omitted singular vectors."""
    cdef char jobu = job, jobvt = job
    cdef int info = 0, lwork = -1
    cdef int mn = m if m < n else n
    cdef int ldu = m if job != b'N' else 1
    cdef int ldvt = n if job == b'A' else (mn if job == b'S' else 1)
    cdef c128 query
    cdef c128* work
    cdef double* rwork
    if mn == 0:
        return 0
    rwork = <double*>malloc(5 * <size_t>mn * sizeof(double))
    if rwork == NULL:
        return -1000
    zgesvd_(&jobu, &jobvt, &m, &n, A, &m, S, U, &ldu, Vh, &ldvt,
            &query, &lwork, rwork, &info)
    if info != 0:
        free(rwork)
        return info
    lwork = <int>query.real
    work = <c128*>malloc(lwork * sizeof(c128))
    if work == NULL:
        free(rwork)
        return -1000
    zgesvd_(&jobu, &jobvt, &m, &n, A, &m, S, U, &ldu, Vh, &ldvt,
            work, &lwork, rwork, &info)
    free(work)
    free(rwork)
    return info


cdef inline int lp_lstsq(double* A, double* B, int m, int n, int nrhs,
                          double* S, int* rank_out,
                          double rcond=-1.0) noexcept nogil:
    """Least squares via SVD: min ||A*X - B||. B has max(m,n) rows.
    Both column major. Solution overwrites B. Returns info."""
    cdef int info = 0
    cdef int lwork
    cdef double work_query
    cdef double* work
    cdef int liwork
    cdef int iwork_query
    cdef int* iwork
    cdef int mn = m if m < n else n
    cdef int ldb = m if m > n else n

    lwork = -1
    if mn == 0:
        rank_out[0] = 0
        return 0
    dgelsd_(&m, &n, &nrhs, A, &m, B, &ldb, S, &rcond, rank_out,
            &work_query, &lwork, &iwork_query, &info)
    if info != 0:
        return info
    lwork = <int>work_query
    liwork = iwork_query

    work = <double*>malloc(lwork * sizeof(double))
    iwork = <int*>malloc(liwork * sizeof(int))
    if work == NULL or iwork == NULL:
        if work != NULL: free(work)
        if iwork != NULL: free(iwork)
        return -1000
    dgelsd_(&m, &n, &nrhs, A, &m, B, &ldb, S, &rcond, rank_out,
            work, &lwork, iwork, &info)
    if work != NULL: free(work)
    if iwork != NULL: free(iwork)
    return info


cdef inline void bl_gemm_nn(const double* A, const double* B, double* C,
                             int M, int N, int K) noexcept nogil:
    """C = A @ B. All row major. C must be pre allocated M x N."""
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C, N)


cdef inline void bl_gemm_nt(const double* A, const double* B, double* C,
                             int M, int N, int K) noexcept nogil:
    """C = A @ B^T. All row major. C must be pre allocated M x N."""
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0, A, K, B, K, 0.0, C, N)


cdef inline void bl_gemm_tn(const double* A, const double* B, double* C,
                             int M, int N, int K) noexcept nogil:
    """C = A^T @ B. All row major. C must be pre allocated M x N."""
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0, A, M, B, N, 0.0, C, N)


cdef inline void bl_gemv_n(const double* A, const double* x, double* y,
                            int M, int N) noexcept nogil:
    """y = A @ x. A is M x N row major."""
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1)


cdef inline void bl_gemv_t(const double* A, const double* x, double* y,
                            int M, int N) noexcept nogil:
    """y = A^T @ x. A is M x N row major, result is N-vector."""
    cblas_dgemv(CblasRowMajor, CblasTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1)


cdef inline void bl_symv(const double* A, const double* x, double* y,
                          int N) noexcept nogil:
    """y = A @ x where A is symmetric N x N row major."""
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


# High level composed operations

cdef inline void spectral_pinv(const double* evals, const double* evecs,
                                double* out, int n, double tol) noexcept nogil:
    """RL^+ = sum_{lam>tol} (1/lam) v v^T. out must be n x n, zeroed.

    evecs is a contiguous row major n x n array with eigenvectors in columns.
    evecs[i*n+k] is component i of eigenvector k. Column major LAPACK output
    must be copied to row major storage before calling this function.
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
    """Matrix rank via SVD. A is m x n column major, overwritten."""
    cdef int mn = m if m < n else n
    cdef double* S = <double*>malloc(mn * sizeof(double))
    cdef double dummy_u, dummy_vt
    cdef int rank = 0
    cdef int k

    if mn == 0:
        if S != NULL: free(S)
        return 0
    if S == NULL:
        if S != NULL: free(S)
        return -1

    cdef int info = lp_svd_mode(A, S, &dummy_u, &dummy_vt, m, n, b'N')
    if info != 0:
        free(S)
        return -1

    for k in range(mn):
        if S[k] > tol:
            rank += 1

    free(S)
    return rank


# Trace of square matrix (row major)
cdef inline double mat_trace(const double* A, int n) noexcept nogil:
    cdef double tr = 0.0
    cdef int i
    for i in range(n):
        tr += A[i * n + i]
    return tr


# Diagonal extraction (row major)
cdef inline void mat_diag(const double* A, double* d, int n) noexcept nogil:
    cdef int i
    for i in range(n):
        d[i] = A[i * n + i]
