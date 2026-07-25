"""rexgraph.sparse_interfacing - eigen-free / scale-free interfacing-vector bundle.

Mirrors ``rexgraph.core._interfacing.build_interfacing_bundle`` field-for-field,
but NEVER materializes the dense response operator ``S_T = B1^T L0^+ B1`` (nE x nE)
nor a dense ``L0^+`` / RL eigendecomposition for the parts that do not genuinely
need the full spectrum. This is the path that removes the arbitrary size ceiling on
the interfacing vector - the dense ceiling was an implementation choice, not the math.

Field taxonomy
--------------
EIGEN-FREE (tight parity to the dense oracle, ~1e-8 or better):
  ``rho``               - weighted vertex source (scatter-add, no linear algebra).
  ``psi = B1^T L0^+ rho`` - one Jacobi-free LSQR solve of the SPARSE graph Laplacian
                          ``L0 = B1 B1^T``; LSQR returns the minimum-norm least-squares
                          solution = ``L0^+ rho``, deflating the per-connected-component
                          constant nullspace of L0 EXACTLY (unlike CG/MINRES, which
                          diverge on the kernel component). No nV x nV pseudoinverse.
  ``signal_magnitude``  - ``||psi||``.
  ``scores[0]`` (I_T)   - topological channel. The dense contraction is
                          ``target^T S_T psi = (B1 target)^T L0^+ (B1 psi)``, a matrix-free
                          bilinear ``u^T L0^+ v`` (LSQR seam ``pinv_bilinear_form``).
                          Because ``psi = B1^T L0^+ rho`` and ``L0 L0^+`` is the projector
                          onto ``range(L0)``, ``L0^+ (B1 psi) = L0^+ L0 (L0^+ rho) = L0^+ rho = y``,
                          so ``I_T = (B1 target)^T y`` with ``y = L0^+ rho`` already in hand -
                          exact and solve-free (see parity test, which also cross-checks
                          the literal two-solve ``pinv_bilinear_form`` form).
  ``scores[1]`` (I_G)   - geometric: ``target^T L_O psi``  (sparse normalized-overlap matvec).
  ``scores[2]`` (I_F)   - frustration: ``target^T L_SG psi`` (sparse integer F = T - G matvec).
  ``iv[:3]``, ``sphere_pos``, ``efficiency`` - assembled from the above (efficiency counts
                          activating/deactivating boundary entries straight from sparse B1 rows).

GENUINELY SPECTRAL (per-mode densities over the edge RL spectrum; no matrix-free reduction):
  ``schrodinger = sum_j (v_j.psi)^2 (v_j.target)^2`` over ``lambda_j > 0`` - degree-4 in the
                          eigenvectors, not a trace / bilinear, so it genuinely needs the modes.
  ``coverage``          - fraction of positive RL modes activated by psi (per-mode count).
  These reuse the EXACT dense RL eigenbasis when it exists (small graphs: ``rex._rl_eigen``)
  -> exact parity. Above the dense limit (``_use_sparse_character`` True, no dense RL) they
  fall back to a BOUNDED ``scipy.sparse.linalg.eigsh`` on the sparse RL4 (k largest-magnitude
  modes) - a documented surrogate, since a fixed-k basis cannot reproduce a full-basis
  degree-4 density exactly. This is the design's option (ii): compute the RL spectrum only
  when small, else a bounded spectrum.
"""
from __future__ import annotations

import numpy as np

_f64 = np.float64

# Largest-magnitude modes kept by the bounded-spectrum eigsh surrogate (schrodinger /
# coverage) on the scale-free path where no dense RL spectrum exists.
_RL_SURROGATE_K = 1024


def _b1_csr(rex):
    """B1 (nV x nE, -1 source / +1 target) as scipy CSR."""
    from rexgraph.core._sparse import to_scipy_csr
    return to_scipy_csr(rex._B1_dual).tocsr()


def _normalized_overlap_sparse(rex):
    """Sparse normalized overlap Laplacian ``L_O = I - D^{-1/2} K D^{-1/2}``,
    ``K = |B1|^T |B1|``, ``D = rowsum(K)`` (incl. diagonal). Identical to the dense
    ``rex.L_overlap`` the dense bundle consumes, built without densifying nE x nE."""
    import scipy.sparse as sp
    nE = int(rex.nE)
    K = rex.overlap_gramian_sparse.tocsr()
    d = np.asarray(K.sum(axis=1)).ravel()
    inv_sqrt = np.zeros(nE, dtype=_f64)
    nz = d > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
    Dis = sp.diags(inv_sqrt)
    return (sp.identity(nE, format='csr', dtype=_f64) - (Dis @ K @ Dis)).tocsr()


def pinv_bilinear_form(A, u, v, atol=1e-13, btol=1e-13, iter_lim=20000):
    """``u^T A^+ v`` for a symmetric PSD sparse ``A`` (possibly SINGULAR), matrix-free
    via LSQR: ``x = A^+ v`` is the minimum-norm least-squares solution, so LSQR projects
    off ``ker(A)`` exactly and ``u^T A^+ v = u^T x``. The bilinear generalization of
    ``sparse_character.pinv_quadratic_form`` (``A^+`` symmetric). Equals the dense
    eigenmode pseudoinverse ``sum_{lambda_j>0} <p_j,u><p_j,v>/lambda_j`` to machine
    precision - no eigendecomposition, no explicit kernel projection."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    u = np.ascontiguousarray(u, dtype=_f64).ravel()
    v = np.ascontiguousarray(v, dtype=_f64).ravel()
    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=_f64))
    x = sla.lsqr(A, v, atol=atol, btol=btol, iter_lim=iter_lim)[0]
    return float(u @ x)


def _l0_pinv_matvec(L0, b, atol=1e-13, btol=1e-13, iter_lim=20000):
    """``L0^+ b`` for the singular symmetric PSD graph Laplacian ``L0``, matrix-free
    via LSQR (min-norm least-squares = pseudoinverse, exact nullspace deflation)."""
    import scipy.sparse.linalg as sla
    b = np.ascontiguousarray(b, dtype=_f64).ravel()
    return sla.lsqr(L0, b, atol=atol, btol=btol, iter_lim=iter_lim)[0]


def _rl_spectrum(rex):
    """(evals_RL, evecs_RL) for the edge RL spectrum used by schrodinger / coverage.

    schrodinger / coverage are degree-4 FULL-spectrum functionals of RL4 with no exact
    matrix-free form. When the whole spectrum is affordable (nE within the mode budget)
    compute it EXACTLY via a dense eigh of RL4 - dense ONLY where the full spectrum is
    genuinely needed and cheap, keyed on affordability, NOT on a global bundle cutoff.
    This is the identical basis the dense oracle uses (``rex._rl_eigen`` eigendecomposes
    the same dense-on-demand RL4), so parity stays exact. Above the budget the full
    spectrum is unaffordable, so fall back to a BOUNDED largest-magnitude ``eigsh``
    surrogate (documented approximation - a fixed-k basis cannot reproduce the full
    degree-4 density; the caller sees it via the loosened spectral tolerance)."""
    nE = int(rex.nE)
    if nE == 0:
        return np.zeros(0, dtype=_f64), np.zeros((0, 0), dtype=_f64)
    if nE <= _RL_SURROGATE_K:
        # full spectrum affordable -> exact dense eigh of the dense-on-demand RL4.
        return rex._rl_eigen
    RL = rex._rl4_sparse.tocsr()
    k = min(nE - 1, _RL_SURROGATE_K)
    if k < 1:
        return np.zeros(0, dtype=_f64), np.zeros((nE, 0), dtype=_f64)
    import scipy.sparse.linalg as sla
    evals, evecs = sla.eigsh(RL, k=k, which='LM')
    return np.ascontiguousarray(evals, dtype=_f64), np.ascontiguousarray(evecs, dtype=_f64)


def _schrodinger_and_coverage(psi, target, evals_rl, evecs_rl, probe_floor,
                              eval_floor=1e-10):
    """Vectorized replica of the dense ``_schrodinger_score`` / ``_coverage`` kernels.

    schrodinger = sum_j (v_j.psi)^2 (v_j.target)^2  over lambda_j > eval_floor.
    coverage    = #{j : |v_j.psi| > probe_floor} / #{j : lambda_j > eval_floor}."""
    if evecs_rl.shape[1] == 0:
        return 0.0, 0.0
    c = evecs_rl.T @ np.ascontiguousarray(psi, dtype=_f64)      # per-mode psi projection
    t = evecs_rl.T @ np.ascontiguousarray(target, dtype=_f64)   # per-mode target projection
    active = evals_rl >= eval_floor
    sch = float(np.sum((c[active] ** 2) * (t[active] ** 2)))
    total = int(np.count_nonzero(active))
    if total == 0:
        return sch, 0.0
    cov = float(np.count_nonzero(np.abs(c[active]) > probe_floor)) / float(total)
    return sch, cov


def _confidence_flags(coverage_val, efficiency, phi_T):
    """Pure-Python replica of ``_interfacing.confidence_flags`` (no densification)."""
    pf = 1.0 - np.exp(-1.0)
    reasons = []
    if coverage_val < pf:
        reasons.append('LOW_SIGNAL')
    if efficiency < 0.5 and phi_T < 2.0 / 3.0:
        reasons.append('CHANNEL_CONFLICT')
    if not reasons:
        return {'flag': 'CONFIDENT', 'reasons': []}
    return {'flag': reasons[0], 'reasons': reasons}


def _source_efficiency_sparse(B1_csr, target_indices, nV):
    """Fraction of activating (positive) boundary entries incident to the target
    vertices, straight from sparse B1 rows. Equals ``_interfacing.source_efficiency``
    without densifying B1 (nV x nE)."""
    n_pos = 0
    n_neg = 0
    indptr = B1_csr.indptr
    data = B1_csr.data
    for v in np.asarray(target_indices).ravel():
        v = int(v)
        if v < 0 or v >= nV:
            continue
        row = data[indptr[v]:indptr[v + 1]]
        n_pos += int(np.count_nonzero(row > 1e-15))
        n_neg += int(np.count_nonzero(row < -1e-15))
    total = n_pos + n_neg
    if total == 0:
        return 0.5
    return float(n_pos) / float(total)


def build_interfacing_bundle_sparse(rex, target_indices, target_weights,
                                    target_signal, vertex_weights=None):
    """Eigen-free interfacing-vector bundle - same dict keys/shapes as the dense
    ``_interfacing.build_interfacing_bundle``.

    Parameters
    ----------
    rex : RexGraph
    target_indices : int array   - source vertex indices.
    target_weights : f64 array   - per-target weights.
    target_signal  : f64[nE]     - target/phenotype edge vector.
    vertex_weights : f64[nV], optional - defaults to IDF ``1 / ln(deg + e)``.

    Returns
    -------
    dict with rho, psi, scores, schrodinger, iv, sphere_pos, signal_magnitude,
    coverage, efficiency, confidence.
    """
    nV, nE = int(rex.nV), int(rex.nE)

    target = np.ascontiguousarray(target_signal, dtype=_f64).ravel()
    ti = np.asarray(target_indices).ravel().astype(np.int64)
    tw = np.ascontiguousarray(target_weights, dtype=_f64).ravel()
    if vertex_weights is None:
        deg = rex.degree.astype(_f64)
        vertex_weights = 1.0 / np.log(deg + np.e)
    vw = np.ascontiguousarray(vertex_weights, dtype=_f64).ravel()

    B1 = _b1_csr(rex)                      # nV x nE signed incidence (sparse)
    L0 = rex.L0_sparse.tocsr()             # nV x nV graph Laplacian B1 B1^T

    # --- rho: weighted vertex source (scatter-add, valid indices only) -----------
    rho = np.zeros(nV, dtype=_f64)
    valid = (ti >= 0) & (ti < nV)
    np.add.at(rho, ti[valid], tw[valid] * vw[ti[valid]])

    # --- psi = B1^T L0^+ rho : one LSQR solve, exact nullspace deflation ----------
    y = _l0_pinv_matvec(L0, rho)           # y = L0^+ rho  (in range(L0))
    psi = B1.T @ y                          # nE
    sig_mag = float(np.linalg.norm(psi))

    # --- channel scores ----------------------------------------------------------
    # I_T = target^T S_T psi = (B1 target)^T L0^+ (B1 psi) = (B1 target)^T y
    # (since L0^+ (B1 psi) = L0^+ L0 y = y; y already deflated onto range(L0)).
    u = B1 @ target                         # nV
    I_T = float(u @ y)
    # I_G, I_F : sparse channel-operator contractions target^T L psi.
    L_O = _normalized_overlap_sparse(rex)   # normalized overlap Laplacian (== rex.L_overlap)
    L_SG = rex.frustration_exact.tocsr()    # integer frustration F = T - G (== rex.L_frustration)
    I_G = float(target @ (L_O @ psi))
    I_F = float(target @ (L_SG @ psi))

    # --- schrodinger + coverage (genuinely spectral over the RL edge spectrum) ----
    evals_rl, evecs_rl = _rl_spectrum(rex)
    pf_val = 1.0 / (float(nV) ** 3) if nV > 0 else 1e-10
    sch, cov = _schrodinger_and_coverage(psi, target, evals_rl, evecs_rl, pf_val)

    # --- assemble ---------------------------------------------------------------
    iv_raw = np.array([I_T, I_G, I_F, sch], dtype=_f64)
    norm = float(np.linalg.norm(iv_raw))
    sp_pos = iv_raw / norm if norm > 1e-30 else iv_raw.copy()

    eff = _source_efficiency_sparse(B1, ti, nV)
    conf = _confidence_flags(cov, eff, float(sp_pos[0]) if sp_pos.shape[0] > 0 else 0.0)

    return {
        'rho': rho,
        'psi': np.ascontiguousarray(psi, dtype=_f64),
        'scores': np.array([I_T, I_G, I_F], dtype=_f64),
        'schrodinger': float(sch),
        'iv': iv_raw,
        'sphere_pos': sp_pos,
        'signal_magnitude': sig_mag,
        'coverage': float(cov),
        'efficiency': float(eff),
        'confidence': conf,
    }
