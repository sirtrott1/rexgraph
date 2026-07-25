"""rexgraph.scale_propagator - the character engine as moments of f(RL4).

The character/coherence layer is a set of **moments of a sparse matrix function
f(RL4)** - not per-vertex Green's solves and not an eigendecomposition. The
identities here are guarded in rexgraph/tests/test_eigenfree.py and
test_scale_bridge.py.

  * Local energy character  = O(nnz) row-norms  diag(RL4²)_e = ‖RL4[e,:]‖²
    (the short-time t² moment of the heat propagator; Part C.3 / script 14).
  * Resolvent diagonal      = diag(RL4⁻¹) EXACT via block-CG solves of RL4·X = I
    to a fixed tolerance (Part A.3 / script 11) - one algorithm at every scale,
    no eigendecomposition, no size-gated approximation.
  * Harmonic log            = eigen-free **Rényi-2** (collision) entropy
    H₂(X) = -log(tr(X²)/tr(X)²), with the H₂-H₃/Shannon gap as a free
    **varentropy reliability flag** (Part D / scripts 18, 19).

All quantities here are O(nnz) trace/row reductions or exact fixed-tolerance
solves; none forms a dense nE×nE operator, calls an eigensolver, or branches to a
stochastic estimate by size. (The general-f Chebyshev matrix-function diagonal -
diag(e^{-tL}) for arbitrary t has no exact O(nnz) form and is a dense-or-stochastic
estimator - lives in rexgraph._experimental, off every live path.)
"""
from __future__ import annotations

import numpy as np

_f64 = np.float64

# GPU auto-gate: a Chebyshev apply goes to the GPU only when the work n*order*columns
# clears this bound (below it, host<->device transfer outweighs the on-device speedup).
# Default ~4.2M is the measured CPU/GPU crossover on the Strix Halo iGPU; tune per host
# via the REXGRAPH_GPU_MIN_WORK env var. The RESULT is identical either way - this is a
# pure performance gate, never a correctness one.
import os as _os
_GPU_MIN_WORK = int(_os.environ.get("REXGRAPH_GPU_MIN_WORK", 1 << 22))


def _csr(X):
    import scipy.sparse as sp
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=_f64))


def energy_character(RL4):
    """Local per-edge energy character diag(RL4²)_e = ‖RL4[e,:]‖² (row-norms),
    O(nnz). The short-time (t²) moment of the heat propagator e^{-tRL4} - the
    local end of the scale profile (Part C.3, script 14). Returns f64[nE]."""
    R = _csr(RL4)
    return np.asarray(R.multiply(R).sum(axis=1)).ravel()


def trace_power(X, a):
    """tr(X^a) for symmetric sparse X, eigen-free. a=2 uses the Frobenius identity
    tr(X²)=‖X‖_F² (no matmul); a>=3 uses a-1 sparse matmuls. Returns float."""
    X = _csr(X)
    if a == 1:
        return float(X.diagonal().sum())
    if a == 2:
        return float(X.multiply(X).sum())
    Xa = X
    for _ in range(a - 2):
        Xa = (Xa @ X).tocsr()
    return float((Xa @ X).diagonal().sum())


def trace_moments(X, a_max):
    """[tr(X), tr(X²), ..., tr(X^a_max)] for symmetric sparse X, eigen-free, from ONE incremental
    power walk (X^k = X^{k-1} @ X, a_max-1 matmuls total, X^k shared across every order) instead of
    recomputing each power from scratch. tr(X²) uses the Frobenius identity ‖X‖_F². This is the
    integer-order moment engine (scripts 16/18/19): the whole Rényi curve H_a = 1/(1-a)·log(
    tr(X^a)/tr(X)^a) reads straight off these moments, so the order sweep costs a-1 matmuls, not
    Σ(a-1). Returns a list of a_max floats."""
    X = _csr(X)
    tr = [float(X.diagonal().sum())]                    # tr(X¹)
    if a_max >= 2:
        tr.append(float(X.multiply(X).sum()))           # tr(X²) = ‖X‖_F², no matmul
    Xk = X                                              # running X^{k-1}
    for _k in range(3, a_max + 1):
        Xk = (Xk @ X).tocsr()                           # advance to X^{k-1}
        tr.append(float((Xk @ X).diagonal().sum()))     # tr(X^k)
    return tr


def renyi_from_moments(tr, a):
    """H_a = 1/(1-a)·log(tr(X^a)/tr(X)^a) from precomputed trace moments (tr[k-1] = tr(X^k))."""
    trX = tr[0]
    if trX <= 0 or tr[a - 1] <= 0:
        return 0.0
    return float((1.0 / (1 - a)) * np.log(tr[a - 1] / trX ** a))


def renyi_entropy(X, a=2):
    """Integer-order Rényi entropy of the normalized spectrum of symmetric PSD X,
    eigen-free (trace moments): H_a = 1/(1-a) · log(tr(X^a)/tr(X)^a). a=2 is the
    collision entropy / harmonic log (the cheap default). O(nnz) for a=2."""
    trX = trace_power(X, 1)
    if trX <= 0:
        return 0.0
    trXa = trace_power(X, a)
    if trXa <= 0:
        return 0.0
    return float((1.0 / (1 - a)) * np.log(trXa / trX ** a))


def harmonic_entropy(X):
    """Harmonic log H₂(X) = -log(tr(X²)/tr(X)²) = Rényi-2 collision entropy
    (Part D.1). e^{H₂} = 1/Σpᵢ² is the effective mode count. O(nnz)."""
    return renyi_entropy(X, 2)


def reliability_gap(X):
    """Varentropy reliability flag (Part D.4 / script 19): the gap between the
    trace-norm entropy H₂ and Shannon H₁, where H₁ is extrapolated eigen-free from
    the integer-order Rényi curve {H₂,H₃,H₄,H₅}. ~0 on flat/unweighted spectra
    (the cheap H₂ is exact); grows with weight-induced non-uniformity (H₂ is a
    looser summary). Returns {'H2', 'H3', 'shannon_est', 'gap'}; 'gap' certifies
    when the cheap value suffices."""
    orders = np.array([2, 3, 4, 5])
    # one shared power walk gives every order's moment (script 16/18/19); the Rényi curve reads off
    # it - no per-order recomputation, no need to parallelize redundant work.
    tr = trace_moments(X, int(orders.max()))
    Ha = np.array([renyi_from_moments(tr, int(a)) for a in orders])
    # quadratic fit of the Rényi curve H_a vs a, extrapolated to a->1 (Shannon)
    shannon_est = float(np.polyval(np.polyfit(orders, Ha, 2), 1.0))
    H2 = float(Ha[0])
    return {'H2': H2, 'H3': float(Ha[1]),
            'shannon_est': shannon_est, 'gap': float(shannon_est - H2)}


# -- multi-GPU column tiling (embarrassingly parallel over RHS columns) -----------
# The GPU propagators/solvers apply ONE shared sparse operator to a BLOCK of RHS columns
# (state is nE x ncols). Splitting the COLUMN block across GPUs is exact and independent:
# the operator is IDENTICAL on every device (replicated), only the RHS columns are
# partitioned; each device runs the SAME on-device kernel on its tile, and the tiles are
# concatenated back. This is a pure, size-gated extension of the single-GPU path: when
# fewer than 2 GPUs are usable (this host), or the work is below the multi-GPU gate, or
# there is a single column, the dispatch keeps the EXISTING single-device kernel unchanged
# (bit-identical), because the multi-GPU plan below returns None and the caller falls
# through to the original `_matfunc_gpu` / inline block-CG / `_greens_diagonal_gpu` call.

def _torch_device(device):
    """Resolve `device` (None -> the current CUDA device, exactly as before; an int -> that
    CUDA/ROCm index; or a torch.device passed through) to a torch.device. `None` reproduces
    the prior `torch.device("cuda")` so the single-GPU path is unchanged."""
    import torch
    if device is None:
        return torch.device("cuda")
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda", int(device))


def _torch_csr(R, device):
    """Build an on-device torch sparse-CSR operator from a scipy CSR `R`. Shared by every
    GPU kernel (single- and multi-device) so the operator construction lives in one place;
    the tensor is identical on whatever device it is placed."""
    import warnings
    import torch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # torch sparse-CSR beta notice
        return torch.sparse_csr_tensor(
            torch.as_tensor(R.indptr, dtype=torch.int64),
            torch.as_tensor(R.indices, dtype=torch.int64),
            torch.as_tensor(R.data, dtype=torch.float64),
            size=R.shape, device=device)


def _partition_columns(ncols, nparts):
    """Balanced CONTIGUOUS partition of range(ncols) into <=nparts (start, stop) tiles.
    Contiguous + order-preserving so concatenating the tiles reconstructs the columns
    exactly. Never empties a tile: nparts is clamped to [1, ncols]."""
    nparts = max(1, min(int(nparts), int(ncols)))
    base, extra = divmod(int(ncols), nparts)
    bounds, s = [], 0
    for i in range(nparts):
        w = base + (1 if i < extra else 0)
        bounds.append((s, s + w))
        s += w
    return bounds


def _tile_columns_across_gpus(kernel, ncols, devices, axis=1):
    """Partition range(ncols) into len(devices) balanced contiguous tiles, invoke
    `kernel(device, col_start, col_stop)` for each tile - each on its own GPU - CONCURRENTLY
    (torch device ops release the GIL, so a thread pool gives real per-device parallelism),
    then concatenate the per-tile numpy results along `axis`. Exact by construction: the
    operator is identical on every device, the column tiles are disjoint and order-preserving,
    so the concatenation equals the single untiled call. `kernel` is device-agnostic - it may
    ignore `device` and run on the CPU, which is how the tiling math is validated without 2
    physical GPUs (see rexgraph/tests/test_multigpu_dispatch.py)."""
    parts = _partition_columns(ncols, len(devices))

    def _run(job):
        idx, (s, e) = job
        return idx, kernel(devices[idx], s, e)

    from rexgraph import compute as _compute
    results = _compute.parallel_map(_run, list(enumerate(parts)), threads=len(parts))
    results.sort(key=lambda r: r[0])                       # restore column order
    return np.concatenate([r[1] for r in results], axis=axis)


def _multi_gpu_plan(work, ncols):
    """The device-index list for a multi-GPU column tiling, or None to keep the single-device
    path. Returns >=2 device indices ONLY when: >1 column to split, >=2 usable GPUs
    (compute.gpu_devices(), capped by REXGRAPH_MAX_GPUS), AND the work clears the multi-GPU
    gate (compute.multi_gpu_min_work(), larger than _GPU_MIN_WORK). Otherwise None, so the
    caller runs the existing single-GPU kernel unchanged. Never raises."""
    if ncols <= 1:
        return None
    try:
        from rexgraph import compute as _compute
        devs = _compute.gpu_devices()
        if len(devs) >= 2 and work >= _compute.multi_gpu_min_work():
            return devs
    except Exception:
        pass
    return None


def _block_cg_gpu(Rt, B, dinv, tol=1e-10, maxit=1000):
    """Jacobi-preconditioned block CG on the GPU (torch) - solve Rt X = B for all
    columns at once, every vector on-device. Mirrors sparse_character._block_cg."""
    import torch
    X = torch.zeros_like(B)
    R = B - torch.sparse.mm(Rt, X)
    Z = dinv[:, None] * R
    P = Z.clone()
    rz = (R * Z).sum(0)
    bnorm = torch.clamp(torch.linalg.norm(B, dim=0), min=1e-300)
    for _ in range(maxit):
        AP = torch.sparse.mm(Rt, P)
        alpha = rz / torch.clamp((P * AP).sum(0), min=1e-300)
        X = X + alpha * P
        R = R - alpha * AP
        if float(torch.max(torch.linalg.norm(R, dim=0) / bnorm)) < tol:
            break
        Z = dinv[:, None] * R
        rz_new = (R * Z).sum(0)
        P = Z + (rz_new / torch.clamp(rz, min=1e-300)) * P
        rz = rz_new
    return X


def _greens_diagonal_gpu(R, dinv, n, step, tol, device=None, col_range=None):
    """diag(R^{-1}) via GPU-resident block-CG: R (torch sparse CSR) stays on-device;
    identity column tiles are solved on the GPU and only the diagonal entries come
    back. Result identical to the CPU tiling (~1e-9). `col_range=(lo, hi)` restricts the
    work to identity columns [lo, hi) and returns only that slice of the diagonal (length
    hi-lo) - used by the multi-GPU dispatch to give each device a sub-range; the default
    (None) computes the full length-n diagonal exactly as before. `device` selects the
    GPU (None -> the current device, unchanged single-GPU behavior)."""
    import torch
    dev = _torch_device(device)
    lo, hi = (0, n) if col_range is None else col_range
    Rt = _torch_csr(R, dev)
    dinvt = torch.as_tensor(dinv, dtype=torch.float64, device=dev)
    diag = np.zeros(hi - lo, dtype=_f64)
    for s in range(lo, hi, step):
        e = min(s + step, hi)
        cols = torch.arange(e - s, device=dev)
        rows = torch.arange(s, e, device=dev)
        E = torch.zeros((n, e - s), dtype=torch.float64, device=dev)
        E[rows, cols] = 1.0
        X = _block_cg_gpu(Rt, E, dinvt, tol=tol)
        diag[s - lo:e - lo] = X[rows, cols].cpu().numpy()
    return diag


def _greens_diagonal_multi(R, dinv, n, step, tol, devices):
    """Multi-GPU diag(R^{-1}): partition the n identity columns across `devices`, solve each
    device's sub-range on-device via `_greens_diagonal_gpu`, and concatenate the diagonal
    slices. Each device replicates the same operator; the identity columns are disjoint, so
    the concatenation reconstructs the full diagonal (~1e-9, the block-CG tolerance)."""
    def kernel(dev, s, e):
        return _greens_diagonal_gpu(R, dinv, n, step, tol, device=dev, col_range=(s, e))
    return _tile_columns_across_gpus(kernel, n, devices, axis=0)


def block_cg_solve(L, B, dinv, tol=1e-10, maxit=1000, backend=None):
    """Solve L X = B (L SPD sparse, B a dense block) by Jacobi-preconditioned block CG,
    on CPU (scipy matvec) or - when a GPU backend is active and the work n*cols clears
    the auto-gate - GPU-resident (operator + block on-device). Returns X (numpy).
    Reusable by any matrix-free Green's/resistance solve; CPU is the exact fallback."""
    B = np.ascontiguousarray(np.asarray(B, dtype=_f64))
    n = L.shape[0]
    ncols = B.reshape(B.shape[0], -1).shape[1]
    if n * ncols >= _GPU_MIN_WORK and _resolve_backend(backend) == "gpu":
        try:
            import torch
            R = _csr(L)
            dv = np.asarray(dinv, dtype=_f64)
            plan = _multi_gpu_plan(n * ncols, ncols)
            if plan is not None:                          # >=2 GPUs, work over the multi gate
                return _block_cg_gpu_multi(R, B, dv, plan, tol, maxit)
            dev = _torch_device(None)                     # single-GPU path (unchanged)
            Rt = _torch_csr(R, dev)
            Bt = torch.as_tensor(B, dtype=torch.float64, device=dev)
            dinvt = torch.as_tensor(dv, dtype=torch.float64, device=dev)
            return _block_cg_gpu(Rt, Bt, dinvt, tol=tol, maxit=maxit).cpu().numpy()
        except Exception:
            pass
    from rexgraph.sparse_character import _block_cg
    return _block_cg(lambda P: _csr(L) @ P, B, np.asarray(dinv, dtype=_f64),
                     tol=tol, maxit=maxit)


def _block_cg_gpu_multi(R, B, dinv, devices, tol, maxit):
    """Multi-GPU block-CG: replicate the operator R to each device, partition the RHS columns
    of B across `devices`, run `_block_cg_gpu` on each device's column tile, and concatenate.
    Every device solves an independent block of columns against the same operator, so the
    concatenation equals the single-device solve (to the block-CG tolerance)."""
    import torch

    def kernel(dev, s, e):
        d = _torch_device(dev)
        Rt = _torch_csr(R, d)
        Bt = torch.as_tensor(np.ascontiguousarray(B[:, s:e]), dtype=torch.float64, device=d)
        dinvt = torch.as_tensor(dinv, dtype=torch.float64, device=d)
        return _block_cg_gpu(Rt, Bt, dinvt, tol=tol, maxit=maxit).cpu().numpy()

    return _tile_columns_across_gpus(kernel, B.shape[1], devices, axis=1)


def greens_diagonal(RL4, tol=1e-10, chunk=512, backend=None):
    """diag(RL4⁻¹) EXACT via block-CG solves of RL4·X = I (RL4/RL3 is SPD, trace-
    normalized and well-conditioned). One algorithm at every scale: each solve runs
    to a FIXED residual tolerance - accuracy is scale-independent, and only the
    iteration count and the number of identity columns grow with size (more
    compute/memory, never less accuracy). `chunk` tiles the columns purely to bound
    peak memory; it does not change the result. Returns f64[nE]. (The resolvent as a
    Chebyshev matrix-function - a general-f estimator, not exact - is preserved in
    rexgraph._experimental.)"""
    from rexgraph.sparse_character import _block_cg
    R = _csr(RL4)
    n = R.shape[0]
    if n == 0:
        return np.zeros(0, dtype=_f64)
    rl_diag = R.diagonal()
    dinv = np.where(np.abs(rl_diag) > 1e-30, 1.0 / rl_diag, 1.0)   # Jacobi precond
    apply_rl = lambda P: R @ P
    diag = np.zeros(n, dtype=_f64)
    step = max(1, min(n, int(chunk)))
    # GPU path: solve the block-CG on-device when a GPU backend is active and the work
    # (n columns * tile) clears the auto-gate. Identical result to the CPU tiling.
    if n * step >= _GPU_MIN_WORK and _resolve_backend(backend) == "gpu":
        try:
            plan = _multi_gpu_plan(n * step, n)          # n identity columns to split
            if plan is not None:                         # >=2 GPUs, work over the multi gate
                return _greens_diagonal_multi(R, dinv, n, step, tol, plan)
            return _greens_diagonal_gpu(R, dinv, n, step, tol)   # single-GPU (unchanged)
        except Exception:
            pass                                        # any GPU issue -> CPU tiling
    bounds = [(s, min(s + step, n)) for s in range(0, n, step)]

    def _solve(b):                                     # one column tile: RL^-1 e_i, exact to tol
        start, stop = b
        E = np.zeros((n, stop - start), dtype=_f64)
        for i in range(start, stop):
            E[i, i - start] = 1.0
        X = _block_cg(apply_rl, E, dinv, tol=tol)
        return start, np.array([X[i, i - start] for i in range(start, stop)], dtype=_f64)

    # the tiles are independent CG solves (GIL-releasing); run them across threads. Cap the worker
    # count because each tile holds an n x step dense RHS, so peak memory is workers * n * step.
    if len(bounds) > 1:
        from rexgraph import compute as _compute
        results = _compute.parallel_map(_solve, bounds, threads=min(8, len(bounds)))
    else:
        results = [_solve(bounds[0])]
    for start, vals in results:
        diag[start:start + vals.shape[0]] = vals
    return diag


def greens_diagonal_deflated(L, H, tol=1e-10, chunk=512):
    """diag(L⁺) for a SINGULAR symmetric PSD L whose kernel is spanned by the columns
    of H (the combinatorial harmonic / cycle basis, nE × k), via the canonical
    harmonic-projector regularization (oracle 09; CANONICAL_SPARSE_MATH_REFERENCE
    Part VI):

        L⁺ = (L + P_H)⁻¹ − P_H,   P_H = H (HᵀH)⁻¹ Hᵀ   (projector onto ker L)

    (L + P_H) is SPD so ordinary sparse block-CG solves apply; P_H is applied LOW-RANK
    through H (never densified) and diag(P_H) is formed exactly. Eigen-free: no dense
    spectrum, no nE × nE inverse. Returns f64[n] = diag(L⁺). For a full-rank operator
    (empty H) this reduces to greens_diagonal(L) = diag(L⁻¹).

    This is the seam behind Green's character / coherence for SINGULAR operators - the
    individual channel hats and the edge Laplacian L1, whose kernel is the harmonic /
    cycle space (rexgraph.harmonic_sparse.harmonic_basis / cycle_basis). The full-rank
    SPD RL4 needs no deflation and uses greens_diagonal directly."""
    import scipy.sparse as sp
    from rexgraph.sparse_character import _block_cg
    L = _csr(L)
    n = L.shape[0]
    if n == 0:
        return np.zeros(0, dtype=_f64)
    if H is None or (hasattr(H, 'shape') and H.shape[1] == 0):
        return greens_diagonal(L, tol=tol, chunk=chunk)   # full-rank: L⁺ = L⁻¹
    Hd = np.ascontiguousarray(
        (H.toarray() if sp.issparse(H) else np.asarray(H)), dtype=_f64)   # n × k, k small
    # VALIDATE the kernel basis: the deflation is only correct when H ⊆ ker(L). The
    # combinatorial cycle basis reduces branching hyperedges to pairwise endpoints and
    # can invent "cycles" that are NOT in ker(B1) (‖L·H‖ ≠ 0); deflating against them is
    # wrong. When H is not a valid kernel basis fall back to the kernel-robust LSQR
    # pseudoinverse diagonal, which needs no explicit kernel and is exact for any L.
    hnorm = float(np.linalg.norm(Hd)) or 1.0
    if float(np.linalg.norm(L @ Hd)) > 1e-7 * hnorm:
        return _greens_diagonal_lsqr(L, tol=tol)
    GHinv = np.linalg.inv(Hd.T @ Hd)                       # (HᵀH)⁻¹, k × k
    # low-rank projector apply: P_H @ P = H (HᵀH)⁻¹ (Hᵀ P); M = L + P_H is SPD
    apply_M = lambda P: (L @ P) + Hd @ (GHinv @ (Hd.T @ P))
    # exact diag(P_H)[e] = H[e,:] (HᵀH)⁻¹ H[e,:]ᵀ
    Y = GHinv @ Hd.T                                       # k × n
    diag_PH = np.einsum('ea,ae->e', Hd, Y)
    mdiag = L.diagonal() + diag_PH                         # diag(M) for Jacobi precond
    dinv = np.where(np.abs(mdiag) > 1e-30, 1.0 / mdiag, 1.0)
    diagM = np.zeros(n, dtype=_f64)
    step = max(1, min(n, int(chunk)))
    for start in range(0, n, step):
        stop = min(start + step, n)
        E = np.zeros((n, stop - start), dtype=_f64)
        for i in range(start, stop):
            E[i, i - start] = 1.0
        X = _block_cg(apply_M, E, dinv, tol=tol)           # (L+P_H)⁻¹ e_i
        for i in range(start, stop):
            diagM[i] = X[i, i - start]
    return diagM - diag_PH                                 # diag(L⁺) = diag(M⁻¹) − diag(P_H)


def _greens_diagonal_lsqr(L, tol=1e-10):
    """diag(L⁺) for a symmetric PSD L (possibly SINGULAR) with NO kernel basis needed:
    per column, x = lsqr(L, e_i) is the minimum-norm least-squares solution = L⁺ e_i
    (LSQR projects off ker(L) exactly), so diag(L⁺)[i] = x_i. Exact to LSQR tolerance and
    kernel-robust - the correctness fallback when a supplied harmonic basis is invalid
    (e.g. the pairwise cycle basis on branching hyperedges). O(n) solves; the deflation
    path is preferred when a VALID kernel basis is available (block solves, cheaper)."""
    import scipy.sparse.linalg as sla
    L = _csr(L)
    n = L.shape[0]
    diag = np.zeros(n, dtype=_f64)
    for i in range(n):
        e = np.zeros(n, dtype=_f64); e[i] = 1.0
        x = sla.lsqr(L, e, atol=tol, btol=tol, iter_lim=20000)[0]
        diag[i] = x[i]
    return diag


# -- Malaugh action <-> moment calculus across a scale tower (oracle 16) ----------

def action_moment(X):
    """The Malaugh calculus: action <-> moment across a scale tower (oracle 16;
    rcfe_final-5 sec 21.3-21.6). Given a tower of scale-indexed moments
    X = [X(0), X(1), ...] (scalar per step, or a vector/array per step along axis 0):

        moment  DX(k) = X(k+1) - X(k)      (per-step difference = discrete derivative)
        action  S(k)  = sum_{j<=k} X(j)    (cumulative integral = Lagrangian)

    conjugate by the FTC across scale: S(k) - S(k-1) = X(k), and differencing the action
    returns the tower (np.diff(S) == X[1:]) while the partial sums of the moment telescope
    back to X (X[k] = X[0] + sum_{j<k} DX(j)). Watching both across the tower shows the
    transformation the doc describes: the energy action ACCELERATES (moment grows), the
    entropy action CONVERGES (moment shrinks). Pure O(len), no eigendecomposition - the
    X(k) are the edge-space trace / harmonic-log moments from `malaugh_quantities`.
    Returns {'moment': DX (len-1 along axis 0), 'action': S (same length as X)}."""
    X = np.asarray(X, dtype=_f64)
    return {'moment': np.diff(X, axis=0), 'action': np.cumsum(X, axis=0)}


def malaugh_quantities(rex):
    """The per-complex edge-space tower moments X(k) for one rex (oracle 16), all O(nnz)
    trace / harmonic-log moments of the boundary operators - no eigendecomposition:

        L_T = tr(T^2)/tr(T)^2,   L_S = tr(L1_up^2)/tr(L1_up)^2   (collision / IPR)
        c2   = L_S / L_T                 coupling  (= (k-2)/2 on K_k)
        c2_E = tr(L1_up^2)/tr(T^2)       energy coupling (raw trace ratio)
        H_T  = -log(L_T),  H_S = -log(L_S)   harmonic-log (collision) entropies

    with T = B1^T B1 (topology/gradient) and L1_up = B2 B2^T (geometry/curl). Map this
    over a sequence of complexes to build a Malaugh tower, then feed each quantity's
    tower to `action_moment`. NaN for a quantity whose trace is 0 (e.g. c2/H_S with no
    faces)."""
    import scipy.sparse as sp
    from rexgraph.core._laplacians import build_L1_down_sparse, build_L1_up_sparse
    T = build_L1_down_sparse(rex._B1_dual).tocsr()
    trT = float(T.diagonal().sum()); trT2 = float(T.multiply(T).sum())
    if int(rex.nF_hodge) > 0 and rex._B2_hodge_dual is not None:
        Lu = build_L1_up_sparse(rex._B2_hodge_dual).tocsr()
        trL = float(Lu.diagonal().sum()); trL2 = float(Lu.multiply(Lu).sum())
    else:
        trL = trL2 = 0.0
    L_T = trT2 / trT ** 2 if trT > 0 else float('nan')
    L_S = trL2 / trL ** 2 if trL > 0 else float('nan')
    return {
        'c2': (L_S / L_T) if (trT > 0 and trL > 0) else float('nan'),
        'c2_E': (trL2 / trT2) if trT2 > 0 else float('nan'),
        'H_T': float(-np.log(L_T)) if L_T == L_T and L_T > 0 else float('nan'),
        'H_S': float(-np.log(L_S)) if L_S == L_S and L_S > 0 else float('nan'),
    }


# -- eigen-free heat propagation of SIGNALS (Chebyshev matrix-vector) -------------
# Applying e^{-tL} to a signal is O(nnz*K) via a Chebyshev polynomial of L -- exact
# to the polynomial order, ANY t, NO eigendecomposition. (Only the DIAGONAL of
# e^{-tL} lacks an exact O(nnz) form; the vector/state apply does not -- same insight
# as the sparse Dirac. This is the reusable heat primitive for _signal, the field
# evolvers, and the NN/LM layer.) Shape: L is symmetric PSD sparse (a Hodge/graph
# Laplacian); f is (n,) or a block (n, m). Every step is spmv/spmm + a small gemm --
# the ideal multi-core / GPU shape; the mat-vec is `L @ x`, swappable for a
# compute.dispatch('spmv', ...) backend later without touching callers.

def _gershgorin_bound(L):
    """Upper bound on lambda_max(L) for symmetric L via Gershgorin (max absolute row
    sum), O(nnz), NO eigensolve. Chebyshev only needs an upper bound on the spectrum;
    a loose bound costs a few extra polynomial terms, never correctness."""
    R = _csr(L)
    if R.shape[0] == 0:
        return 1.0
    return float(np.asarray(np.abs(R).sum(axis=1)).ravel().max())


def _cheb_basis(lam_max, order):
    """Chebyshev sampling on [0, lam_max]: the node spectrum `lam` and the DCT basis
    `cos_kj` so coefficients of any func are `(2/order) * (func(lam) @ cos_kj.T)` with
    c_0 halved. Precomputed once and reused across all timesteps."""
    j = np.arange(order)
    nodes = np.cos(np.pi * (j + 0.5) / order)              # Chebyshev nodes in [-1,1]
    lam = (nodes + 1.0) * lam_max / 2.0                    # mapped to [0, lam_max]
    cos_kj = np.cos(np.pi * np.outer(j, j + 0.5) / order)  # (order, order)
    return lam, cos_kj


def _cheb_coeffs(func_vals, cos_kj, order):
    """Chebyshev coefficients from precomputed func values at the nodes."""
    c = (2.0 / order) * (cos_kj @ np.asarray(func_vals, dtype=_f64))
    c[0] *= 0.5
    return c


def _cheb_vectors(L, f, lam_max, order):
    """V_k = T_k(L~) f for k=0..order-1, where L~ = 2L/lam_max - I maps the spectrum
    into [-1,1]. `order` sparse mat-vecs (spmv, or spmm when f is a block). Returns an
    (order,)+f.shape array. This is the parallel/GPU-hot core: each L @ x is an spmv."""
    R = _csr(L)
    scale = 2.0 / lam_max
    f = np.asarray(f, dtype=_f64)

    def Ltil(x):
        return scale * (R @ x) - x                          # rescaled sparse mat-vec

    V = np.empty((order,) + f.shape, dtype=_f64)
    V[0] = f
    if order > 1:
        V[1] = Ltil(f)
    for k in range(2, order):
        V[k] = 2.0 * Ltil(V[k - 1]) - V[k - 2]
    return V


def _heat_order(t, lam_max, given):
    if given is not None:
        return int(given)
    return int(max(16, min(600, 1.5 * float(t) * lam_max + 16)))


def matfunc_apply(L, f, func, order, lam_max=None, backend=None):
    """Apply a GENERAL matrix function func(L) to a signal/block f via a Chebyshev
    polynomial of L -- O(nnz*order) sparse mat-vecs, NO eigendecomposition. `func` maps
    eigenvalues (samples in [0, lam_max]) to scalars: e.g. `lambda l: exp(-t*l)` (heat),
    `lambda l: cos(t*sqrt(l))` (wave), `lambda l: 1/(l+s)` (shifted resolvent). L is
    symmetric with spectrum in [0, lam_max]; f is (n,) or a block (n, m). This is the
    reusable f(L)·state primitive underneath heat/wave/field evolution. The mat-vecs
    are spmv/spmm and the combine is a gemm - the SAME computation runs on CPU (scipy)
    or, when a GPU backend is active (`backend`, or the compute default), entirely
    on-device (a GPU-resident Chebyshev). CPU is the always-available fallback."""
    order = int(order)
    if lam_max is None:
        lam_max = _gershgorin_bound(L) * 1.0001 + 1e-30
    lam, cos_kj = _cheb_basis(lam_max, order)
    c = _cheb_coeffs(np.asarray(func(lam), dtype=_f64), cos_kj, order)
    # Size auto-gate: the GPU wins only when the work (n * order * columns) is big enough
    # to amortize host<->device transfer; smaller problems stay on CPU (identical result,
    # just faster). `backend='gpu'/'auto'` still honors the gate; the CPU path is exact.
    fa = np.asarray(f)
    ncols = int(fa.reshape(fa.shape[0], -1).shape[1])
    work = int(fa.shape[0]) * order * ncols
    if work >= _GPU_MIN_WORK and _resolve_backend(backend) == "gpu":
        try:
            plan = _multi_gpu_plan(work, ncols)          # >=2 GPUs, work over the multi gate
            if plan is not None:
                return _matfunc_gpu_multi(L, f, c, lam_max, order, plan)
            return _matfunc_gpu(L, f, c, lam_max, order)  # single-GPU path (unchanged)
        except Exception:
            pass                                        # any GPU issue -> CPU fallback
    V = _cheb_vectors(L, f, lam_max, order)
    return np.tensordot(c, V, axes=([0], [0]))


def _resolve_backend(backend):
    """Resolve the compute backend for a Chebyshev apply to 'gpu' or 'cpu'. `backend`:
    None -> the compute layer's active default (GPU only if explicitly selected via
    set_default_backend / REXGRAPH_BACKEND); 'gpu'/'cuda'/'rocm'/'mps'/'auto' -> GPU if
    torch reports a device; 'cpu' -> CPU. Falls back to CPU whenever a GPU path is
    unavailable, so callers never break on a CPU-only host."""
    name = backend
    if name is None:
        try:
            from rexgraph import compute as _compute
            name = _compute.get_default_backend()
            if name is None:                            # no explicit default -> DYNAMIC:
                name = _compute.recommended_backend()   # the best backend for THIS host
        except Exception:
            name = None
    if name in (None, "cpu", "openmp"):
        return "cpu"
    if name in ("gpu", "cuda", "rocm", "mps", "auto"):
        try:
            import torch
            if torch.cuda.is_available():
                return "gpu"
        except Exception:
            pass
    return "cpu"


def _matfunc_gpu(L, f, coeffs, lam_max, order, device=None):
    """GPU-RESIDENT Chebyshev apply: the operator and Chebyshev vectors stay on-device
    for the whole recurrence (order sparse mat-muls), and the coefficient combine is a
    single on-device contraction; only the tiny coefficients go up and the result comes
    back. Bit-comparable to the CPU path (~1e-15). `device` selects the GPU (None -> the
    current device, unchanged single-GPU behavior); the multi-GPU dispatch passes explicit
    device indices, one per column tile."""
    import torch
    dev = _torch_device(device)
    R = _csr(L)
    At = _torch_csr(R, dev)
    f = np.asarray(f, dtype=_f64)
    X0 = torch.as_tensor(f.reshape(f.shape[0], -1), dtype=torch.float64, device=dev)  # (n, m)
    scale = 2.0 / lam_max

    def Ltil(x):
        return scale * torch.sparse.mm(At, x) - x

    ct = torch.as_tensor(np.asarray(coeffs, dtype=_f64), dtype=torch.float64, device=dev)
    acc = ct[0] * X0
    if order > 1:
        Tkm1, Tk = X0, Ltil(X0)
        acc = acc + ct[1] * Tk
        for k in range(2, order):
            Tkp1 = 2.0 * Ltil(Tk) - Tkm1
            acc = acc + ct[k] * Tkp1
            Tkm1, Tk = Tk, Tkp1
    return acc.cpu().numpy().reshape(f.shape)


def _matfunc_gpu_multi(L, f, coeffs, lam_max, order, devices):
    """Multi-GPU Chebyshev apply: replicate the operator to each device, partition the column
    block of f across `devices`, run the GPU-resident `_matfunc_gpu` on each device's column
    tile, and concatenate. The Chebyshev polynomial is a fixed, data-independent recurrence per
    column, so the tiled result is BIT-IDENTICAL to the single-device call - only the columns
    are partitioned; the operator and coefficients are the same on every device."""
    R = _csr(L)
    fa = np.asarray(f, dtype=_f64)
    f2 = fa.reshape(fa.shape[0], -1)                       # (n, m); tiling needs ncols > 1

    def kernel(dev, s, e):
        return _matfunc_gpu(R, np.ascontiguousarray(f2[:, s:e]), coeffs, lam_max, order, device=dev)

    out = _tile_columns_across_gpus(kernel, f2.shape[1], devices, axis=1)
    return out.reshape(fa.shape)


def matfunc_trajectory(L, f, funcs, order, lam_max=None):
    """[func(L) f for func in funcs] sharing ONE set of Chebyshev vectors V_k=T_k(L~)f
    (order sparse mat-vecs TOTAL). Each func is a coefficient combination Sum_k c_k V_k
    -- a (len(funcs) x order) @ (order x ...) gemm. Returns (len(funcs),) + f.shape."""
    if lam_max is None:
        lam_max = _gershgorin_bound(L) * 1.0001 + 1e-30
    order = int(order)
    V = _cheb_vectors(L, f, lam_max, order)                     # shared mat-vecs
    lam, cos_kj = _cheb_basis(lam_max, order)
    C = np.stack([_cheb_coeffs(np.asarray(fn(lam), dtype=_f64), cos_kj, order)
                  for fn in funcs], axis=0)                     # (nfuncs, order)
    return np.tensordot(C, V, axes=([1], [0]))                  # (nfuncs,) + f.shape


def _schrodinger_order(t, lam_max, given):
    if given is not None:
        return int(given)
    return int(max(24, min(400, 1.5 * float(t) * float(lam_max) + 24)))


def schrodinger_apply(L, psi, t, order=None, lam_max=None):
    """Unitary evolution ``e^{-iLt} psi`` for a real-symmetric PSD sparse ``L``,
    matrix-free: ``e^{-iLt} = cos(tL) - i sin(tL)``, both applied from ONE shared set
    of Chebyshev matvecs. ``psi`` may be real or complex; returns complex. Equals the
    dense mode-sum ``V diag(e^{-i lambda t}) Vᵀ psi`` to ~1e-10, no eigendecomposition."""
    psi = np.asarray(psi)
    if lam_max is None:
        lam_max = _gershgorin_bound(L) * 1.0001 + 1e-30
    order = _schrodinger_order(t, lam_max, order)
    t = float(t)
    funcs = [lambda l: np.cos(t * l), lambda l: np.sin(t * l)]
    a = np.ascontiguousarray(psi.real, dtype=_f64)
    if not np.iscomplexobj(psi):
        cs = matfunc_trajectory(L, a, funcs, order, lam_max=lam_max)   # (2, nE)
        return cs[0] - 1j * cs[1]
    b = np.ascontiguousarray(psi.imag, dtype=_f64)
    block = np.stack([a, b], axis=1)                                  # (nE, 2)
    cs = matfunc_trajectory(L, block, funcs, order, lam_max=lam_max)   # (2, nE, 2)
    cos_ab, sin_ab = cs[0], cs[1]
    real = cos_ab[:, 0] + sin_ab[:, 1]         # cos(tL)a + sin(tL)b
    imag = cos_ab[:, 1] - sin_ab[:, 0]         # cos(tL)b - sin(tL)a
    return real + 1j * imag


def schrodinger_trajectory(L, psi, times, order=None, lam_max=None):
    """``[e^{-iLt} psi for t in times]`` sharing ONE set of Chebyshev matvecs on ``L``
    (order sparse mat-vecs TOTAL, independent of len(times)). ``psi`` real or complex;
    returns a complex array ``(len(times),) + psi.shape``."""
    psi = np.asarray(psi)
    tvec = np.asarray(times, dtype=_f64).ravel()
    if lam_max is None:
        lam_max = _gershgorin_bound(L) * 1.0001 + 1e-30
    tmax = float(tvec.max()) if tvec.size else 1.0
    order = _schrodinger_order(tmax, lam_max, order)
    funcs = []
    for tt in tvec:
        funcs.append(lambda l, s=tt: np.cos(s * l))
        funcs.append(lambda l, s=tt: np.sin(s * l))
    a = np.ascontiguousarray(psi.real, dtype=_f64)
    if not np.iscomplexobj(psi):
        cs = matfunc_trajectory(L, a, funcs, order, lam_max=lam_max)   # (2T, nE)
        return np.stack([cs[2 * i] - 1j * cs[2 * i + 1]
                         for i in range(tvec.size)], axis=0)
    b = np.ascontiguousarray(psi.imag, dtype=_f64)
    block = np.stack([a, b], axis=1)                                  # (nE, 2)
    cs = matfunc_trajectory(L, block, funcs, order, lam_max=lam_max)   # (2T, nE, 2)
    out = np.empty((tvec.size,) + psi.shape, dtype=np.complex128)
    for i in range(tvec.size):
        cos_ab, sin_ab = cs[2 * i], cs[2 * i + 1]
        out[i] = (cos_ab[:, 0] + sin_ab[:, 1]) + 1j * (cos_ab[:, 1] - sin_ab[:, 0])
    return out


def heat_apply(L, f, t, order=None, lam_max=None, backend=None):
    """e^{-tL} f via a Chebyshev polynomial of L -- O(nnz*order) sparse mat-vecs, no
    eigendecomposition, any t >= 0. L symmetric PSD sparse; f is (n,) or (n, m).
    Matches the dense e^{-tL} apply to Chebyshev tolerance. Runs on CPU or (via
    `backend` / the compute default) GPU-resident. Returns f-shaped array.
    (A heat-specialized wrapper over :func:`matfunc_apply`.)"""
    if lam_max is None:
        lam_max = _gershgorin_bound(L) * 1.0001 + 1e-30
    order = _heat_order(t, lam_max, order)
    return matfunc_apply(L, f, lambda l: np.exp(-float(t) * l), order,
                         lam_max=lam_max, backend=backend)


def heat_trajectory(L, f, times, order=None, lam_max=None):
    """[e^{-tL} f for t in times] sharing ONE set of Chebyshev vectors V_k = T_k(L~)f:
    `order` sparse mat-vecs TOTAL (not per-t), then each timestep is a coefficient
    combination Sum_k c_k(t) V_k. The mat-vecs are spmv/spmm (multi-core/GPU) and the
    per-t combination is a (T x order) @ (order x ...) gemm -- both dispatchable.
    Returns an array of shape (len(times),) + f.shape."""
    times = np.asarray(times, dtype=_f64).ravel()
    if lam_max is None:
        lam_max = _gershgorin_bound(L) * 1.0001 + 1e-30
    tmax = float(times.max()) if times.size else 1.0
    order = _heat_order(tmax, lam_max, order)
    V = _cheb_vectors(L, f, lam_max, order)                     # shared mat-vecs
    lam, cos_kj = _cheb_basis(lam_max, order)
    fvals = np.exp(-np.outer(times, lam))                       # (T, order) node values
    C = (2.0 / order) * (fvals @ cos_kj.T)                      # (T, order) coeffs
    C[:, 0] *= 0.5
    return np.tensordot(C, V, axes=([1], [0]))                  # (T,) + f.shape
