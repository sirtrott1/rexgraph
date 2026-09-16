"""Native eigenfree interfacing and explicitly basis dependent mode diagnostics.

Native scores preserve the three contractions in the dense reference, without
materializing ``S_T = B1^T L0^+ B1``, a dense pseudoinverse, or an eigenbasis.
The five legacy mode dependent fields are None unless explicitly requested from
the oracle. channel_direction is a separate three score normalization.

Field taxonomy

EIGEN FREE (tight parity to the dense oracle, ~1e-8 or better):
  ``rho``               - weighted vertex source (scatter add, no linear algebra).
  ``psi = B1^T L0^+ rho`` - one CG solve of ``L0 + Pi_h`` on the SPARSE graph Laplacian
                          ``L0 = B1 B1^T``, where ``Pi_h`` projects onto the component
                          frame. ``L0^# = (L0 + Pi_h)^-1 - Pi_h`` is the harmonic-
                          complement inverse, equal to ``L0^+`` under the identity
                          metric. The kernel is written down from the components rather
                          than found by a solver or a threshold. This is numerical, not
                          exact arithmetic. No nV x nV pseudoinverse is constructed.
  ``signal_magnitude``  - ``||psi||``.
  ``scores[0]`` (I_T)   - topological channel. The dense contraction is
                          ``target^T S_T psi = (B1 target)^T L0^+ (B1 psi)``, a matrix free
                          bilinear ``u^T L0^+ v`` (LSQR seam ``pinv_bilinear_form``).
                          Because ``psi = B1^T L0^+ rho`` and ``L0 L0^+`` is the projector
                          onto ``range(L0)``, ``L0^+ (B1 psi) = L0^+ L0 (L0^+ rho) = L0^+ rho = y``,
                          so ``I_T = (B1 target)^T y`` with ``y = L0^+ rho`` already in hand -
                          exact and solve-free (see parity test, which also cross-checks
                          the literal two-solve ``pinv_bilinear_form`` form).
  ``scores[1]`` (I_G)   - normalized G-channel: ``target^T L_O psi`` (down sector).
  ``scores[2]`` (I_F)   - frustration: ``target^T L_SG psi`` (sparse integer F = T - G matvec).
  ``channel_direction``, ``efficiency`` - assembled from the above (efficiency counts
                          activating/deactivating boundary entries straight from sparse B1 rows).

GENUINELY SPECTRAL (per mode densities over the edge RL spectrum; no matrix free reduction):
  ``schrodinger = sum_j (v_j.psi)^2 (v_j.target)^2`` over ``lambda_j > 0`` - degree 4 in the
                          eigenvectors, not a trace / bilinear, so it genuinely needs the modes.
  ``coverage``          - fraction of positive RL modes activated by psi (per mode count).
  These are available ONLY through ``build_interfacing_bundle_oracle``. They
  depend on the chosen basis within repeated eigenspaces, not merely the operator.
  Full dense and explicitly requested partial spectra carry distinct metadata.
  No graph size threshold silently substitutes a partial reading for a full one.
"""
from __future__ import annotations

import numpy as np

_f64 = np.float64


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
    """``u^T A^+ v`` for a symmetric PSD sparse ``A`` (possibly SINGULAR), matrix free
    via LSQR: ``x = A^+ v`` is the minimum norm least squares solution, so LSQR projects
    off ``ker(A)`` exactly and ``u^T A^+ v = u^T x``. The bilinear generalization of
    ``sparse_character.pinv_quadratic_form`` (``A^+`` symmetric). Equals the dense
    eigenmode pseudoinverse ``sum_{lambda_j>0} <p_j,u><p_j,v>/lambda_j`` to machine
    precision: no eigendecomposition, no explicit kernel projection."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    u = np.ascontiguousarray(u, dtype=_f64).ravel()
    v = np.ascontiguousarray(v, dtype=_f64).ravel()
    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=_f64))
    x = sla.lsqr(A, v, atol=atol, btol=btol, iter_lim=iter_lim)[0]
    return float(u @ x)


def _component_labels(L0):
    """Component id per vertex, from the union find kernel in ``_common.pxd``.

    ``L0 = B1 B1^T`` has zero row sums, so every component indicator is in its kernel,
    and for a graph Laplacian those indicators SPAN it -- ``x^T L0 x = 0`` forces ``x``
    constant along every relation. So the kernel is a pattern traversal, not an
    eigensolve with a threshold deciding which eigenvalue counts as zero.

    The traversal is ``_sparse.connected_components``, over ``_common``'s union find.
    """
    from rexgraph.core._sparse import connected_components
    labels, _count = connected_components(L0.indptr, L0.indices, L0.shape[0])
    return np.asarray(labels)


def _component_frame(L0):
    """The kernel of a graph Laplacian as a dense indicator frame.

    Columns are orthogonal by construction because components are disjoint. Prefer
    ``_component_projector``: this materializes ``nV x beta_0`` doubles to say
    "average within each part".
    """
    label = _component_labels(L0)
    n = L0.shape[0]
    count = int(label.max()) + 1 if n else 0
    frame = np.zeros((n, count), dtype=_f64)
    if n:
        frame[np.arange(n), label] = 1.0
    return frame


def _component_projector(L0):
    """`Pi_h` for a graph Laplacian's kernel, in O(nV) time and O(nV) memory.

    The kernel frame is an indicator matrix, so the projector is "replace each
    coordinate by its component mean" -- one bincount, rather than the `O(nV beta_0)`
    a dense frame would cost to store and apply.
    """
    label = _component_labels(L0)
    count = int(label.max()) + 1 if label.size else 0
    size = np.bincount(label, minlength=count).astype(_f64) if count else np.zeros(0)

    def project(x):
        block = np.asarray(x, dtype=_f64)
        if block.ndim == 1:
            return (np.bincount(label, weights=block, minlength=count) / size)[label]
        return np.column_stack([project(block[:, j]) for j in range(block.shape[1])])

    # diag(P_H)[i] = 1 / |component(i)|, carrying the same contract as the dense
    # `frame_projector` so the two are substitutable at a Green diagonal.
    project.diagonal = lambda: 1.0 / size[label]
    project.labels = label
    return project


def _l0_pinv_matvec(L0, b, atol=1e-13, btol=1e-13, iter_lim=20000):
    """``L0^+ b`` for the singular symmetric PSD graph Laplacian ``L0``, matrix free.

    ``L0^# = (L0 + Pi_h)^-1 - Pi_h`` over the component frame: on the kernel the shifted
    operator is the identity, off it ``L0`` is positive, so one CG answers and the
    identity ``L0 L0^# = I - Pi_h`` holds by construction. Under the identity metric
    this is the Moore Penrose action, so the contract here is unchanged.

    ``atol``, ``btol`` and ``iter_lim`` are accepted for callers that pass them: the
    first is the CG tolerance and the last its iteration cap.
    """
    from rexgraph.core._linalg import harmonic_pinv_matvec
    b = np.ascontiguousarray(b, dtype=_f64).ravel()
    return harmonic_pinv_matvec(L0, _component_projector(L0), b,
                                tol=max(atol, 1e-15), maxiter=int(iter_lim))


def _rl_spectrum(rex, mode_count=None):
    """Explicit full dense or requested partial RL4 eigenbasis for oracle callers.

    This is numerical and basis dependent; no size gate changes the definition.
    """
    from numbers import Integral
    nE = int(rex.nE)
    if mode_count is not None and (isinstance(mode_count, (bool, np.bool_))
            or not isinstance(mode_count, Integral) or not 0 < mode_count < nE):
        raise ValueError("partial mode_count must be an integer in [1, nE)")
    if nE == 0:
        return np.zeros(0, dtype=_f64), np.zeros((0, 0), dtype=_f64)
    if mode_count is None:
        # Explicit full spectrum oracle request; never reached by native bundles.
        return rex._rl_eigen
    RL = rex._rl4_sparse.tocsr()
    k = int(mode_count)
    if k < 1:
        return np.zeros(0, dtype=_f64), np.zeros((nE, 0), dtype=_f64)
    import scipy.sparse.linalg as sla
    evals, evecs = sla.eigsh(RL, k=k, which='LM', v0=np.ones(nE))
    return np.ascontiguousarray(evals, dtype=_f64), np.ascontiguousarray(evecs, dtype=_f64)


def _schrodinger_and_coverage(psi, target, evals_rl, evecs_rl, probe_floor,
                              eval_floor=1e-10):
    """Vectorized replica of the dense ``_schrodinger_score`` / ``_coverage`` kernels.

    schrodinger = sum_j (v_j.psi)^2 (v_j.target)^2  over lambda_j > eval_floor.
    coverage    = #{j : |v_j.psi| > probe_floor} / #{j : lambda_j > eval_floor}."""
    if evecs_rl.shape[1] == 0:
        return 0.0, 0.0
    c = evecs_rl.T @ np.ascontiguousarray(psi, dtype=_f64)      # per mode psi projection
    t = evecs_rl.T @ np.ascontiguousarray(target, dtype=_f64)   # per mode target projection
    active = evals_rl >= eval_floor
    sch = float(np.sum((c[active] ** 2) * (t[active] ** 2)))
    total = int(np.count_nonzero(active))
    if total == 0:
        return sch, 0.0
    cov = float(np.count_nonzero(np.abs(c[active]) > probe_floor)) / float(total)
    return sch, cov


def _confidence_flags(coverage_val, efficiency, phi_T):
    """Pure Python replica of ``_interfacing.confidence_flags`` (no densification)."""
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
    """Native interfacing: three channel scores, no implicit eigenbasis.

    Parameters

    rex : RexGraph
    target_indices : int array   - source vertex indices.
    target_weights : f64 array   - per target weights.
    target_signal  : f64[nE]     - target/phenotype edge vector.
    vertex_weights : f64[nV], optional - defaults to IDF ``1 / ln(deg + e)``.

    Returns

    Mode dependent schrodinger/coverage/iv/sphere_pos/confidence are None.
    channel_direction normalizes only the three native scores, not the legacy
    four component vector. mode_diagnostics states why no mode scores exist.
    """
    nV, _nE = int(rex.nV), int(rex.nE)

    # target_signal=None means "score psi against itself": the self interfacing
    # reading, resolved below once psi exists. Callers used to get it by running the
    # whole bundle twice (a throwaway call with a zero target purely to obtain psi,
    # then a real one) paying two L0^+ solves for one reading.
    self_target = target_signal is None
    def real_vector(value, name, size=None):
        raw = np.asarray(value)
        if raw.ndim != 1 or np.iscomplexobj(raw):
            raise ValueError(f"{name} must be a finite real vector")
        out = np.ascontiguousarray(raw, dtype=_f64)
        if not np.isfinite(out).all() or (size is not None and out.size != size):
            raise ValueError(f"{name} must be finite with length {size}")
        return out
    target = None if self_target else real_vector(target_signal, "target signal", _nE)
    indices = np.asarray(target_indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("target indices must be an integer vector")
    ti = indices.astype(np.int64)
    tw = real_vector(target_weights, "target weights", ti.size)
    if vertex_weights is None:
        deg = rex.degree.astype(_f64)
        vertex_weights = 1.0 / np.log(deg + np.e)
    vw = real_vector(vertex_weights, "vertex weights", nV)

    B1 = _b1_csr(rex)                      # nV x nE signed incidence (sparse)
    L0 = rex.L0_sparse.tocsr()             # nV x nV graph Laplacian B1 B1^T

    #### rho: weighted vertex source (scatter add, valid indices only)
    rho = np.zeros(nV, dtype=_f64)
    valid = (ti >= 0) & (ti < nV)
    np.add.at(rho, ti[valid], tw[valid] * vw[ti[valid]])

    #### psi = B1^T L0^+ rho : one LSQR solve, exact nullspace deflation
    y = _l0_pinv_matvec(L0, rho)           # y = L0^+ rho  (in range(L0))
    psi = B1.T @ y                          # nE
    sig_mag = float(np.linalg.norm(psi))
    if self_target:
        target = psi

    #### channel scores
    # I_T = target^T S_T psi = (B1 target)^T L0^+ (B1 psi) = (B1 target)^T y
    # (since L0^+ (B1 psi) = L0^+ L0 y = y; y already deflated onto range(L0)).
    u = B1 @ target                         # nV
    I_T = float(u @ y)
    # I_G, I_F : sparse channel operator contractions target^T L psi.
    L_O = _normalized_overlap_sparse(rex)   # normalized overlap Laplacian (== rex.L_overlap)
    L_SG = rex.frustration_exact.tocsr()    # integer frustration F = T - G (== rex.L_frustration)
    I_G = float(target @ (L_O @ psi))
    I_F = float(target @ (L_SG @ psi))

    #### assemble
    scores = np.array([I_T, I_G, I_F], dtype=_f64)
    norm = float(np.linalg.norm(scores))
    direction = scores / norm if norm > 0 else scores.copy()

    eff = _source_efficiency_sparse(B1, ti, nV)

    return {
        'rho': rho,
        'psi': np.ascontiguousarray(psi, dtype=_f64),
        'scores': scores,
        'channel_direction': direction,
        'schrodinger': None,
        'iv': None,
        'sphere_pos': None,
        'signal_magnitude': sig_mag,
        'coverage': None,
        'efficiency': float(eff),
        'confidence': None,
        'mode_diagnostics': {'status': 'not-requested', 'basis_dependent': True,
                             'api': 'interfacing_vector_oracle'},
    }


def build_interfacing_bundle_oracle(rex, target_indices, target_weights,
                                    target_signal, vertex_weights=None, *,
                                    mode_count=None, eigenbasis=None):
    """Legacy mode diagnostics under an explicit numerical eigenbasis contract.

    None mode_count requests the full dense RL4 eigenbasis, with O(nE^2) storage.
    An explicit k requests only k largest magnitude modes; this is a partial
    reading, not an error bounded approximation of the full basis dependent score.
    eigenbasis=(values,vectors) supplies a checked full or partial RL4 eigenbasis.
    The selected basis is identified by its content hash, never called canonical.
    """
    import hashlib
    result = build_interfacing_bundle_sparse(
        rex, target_indices, target_weights, target_signal, vertex_weights)
    if eigenbasis is None:
        values, vectors = _rl_spectrum(rex, mode_count)
        method = 'dense-eigh' if mode_count is None else 'partial-eigsh-LM'
    else:
        if mode_count is not None:
            raise ValueError("choose eigenbasis or mode_count, not both")
        values, vectors = map(np.asarray, eigenbasis)
        n = int(rex.nE)
        if (np.iscomplexobj(values) or np.iscomplexobj(vectors) or values.ndim != 1
                or vectors.shape != (n, values.size) or values.size > n
                or not np.isfinite(values).all() or not np.isfinite(vectors).all()):
            raise ValueError("oracle basis must have finite real compatible dimensions")
        values = np.ascontiguousarray(values, dtype=_f64)
        vectors = np.ascontiguousarray(vectors, dtype=_f64)
        if not np.allclose(vectors.T @ vectors, np.eye(values.size), atol=1e-10, rtol=1e-10):
            raise ValueError("oracle basis must be orthonormal")
        RL = rex._rl4_sparse
        residual = RL @ vectors - vectors * values
        scale = max(float(abs(RL).sum(axis=1).max()), 1.) if n else 1.
        if np.linalg.norm(residual) > 1e-9 * scale * max(1, np.sqrt(values.size)):
            raise ValueError("oracle basis must diagonalize this source's RL4")
        method = 'supplied-eigenbasis'
    psi = result['psi']
    target = psi if target_signal is None else np.asarray(target_signal, dtype=_f64)
    floor = 1. / float(rex.nV) ** 3 if rex.nV else 1e-10
    sch, coverage = _schrodinger_and_coverage(psi, target, values, vectors, floor)
    iv = np.append(result['scores'], sch)
    norm = np.linalg.norm(iv)
    sphere = iv / norm if norm > 1e-30 else iv.copy()
    digest = hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()
                            + np.asarray(vectors, dtype='<f8').tobytes()).hexdigest()
    result.update(schrodinger=sch, coverage=coverage, iv=iv, sphere_pos=sphere,
                  confidence=_confidence_flags(coverage, result['efficiency'], float(sphere[0])),
                  mode_diagnostics={'status': 'observed', 'basis_dependent': True,
                                    'method': method, 'mode_count': len(values),
                                    'complete': len(values) == int(rex.nE),
                                    'basis_digest': digest, 'probe_floor': floor,
                                    'eval_floor': 1e-10})
    return result
