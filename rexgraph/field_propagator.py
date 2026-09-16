"""rexgraph.field_propagator: matrix free evolution of the coupled (edge, face)
field on the graded vector space C_1 (+) C_2.

The field operator

    M = [[ RL1,     -g B2 ],
         [ -g B2ᵀ,   L2   ]]

acts on a GRADED VECTOR SPACE: the edge block C_1 stacked with the face block C_2.
A field here is not merely a vector: a block ``F`` of shape ``(nE+nF, m)`` carries a
TENSOR SHAPE (m components), and the boundary weighting W supplies a TENSOR METRIC
(the inner product on the graded space). So the field IS a dynamic tensor object -
static structure = the graded operator M itself, dynamic evolution = a matrix
function of M applied to the field.

Evolution is numerical and eigenfree, via a Chebyshev polynomial of M (assembled SPARSE, never
the dense (nE+nF)^2 matrix):

    heat   e^{-tM} F        decays on positive modes, grows on negative modes
    wave   C_t(M) F         cos(t sqrt(lambda)) / cosh(t sqrt(-lambda))

Sparse spmv/spmm, NO eigendecomposition. Long growth intervals are stepped;
overflow is reported. Full SPD metrics use sparse triangular solves, whose factors
may fill in. Real single function actions with identity/diagonal metrics retain
the scale backend's GPU dispatch. Other actions use the streaming CPU recurrence.
The dense ``core._field`` spectral evolvers remain explicit reference oracles.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from rexgraph import scale_propagator as _spg

_f64 = np.float64

__all__ = [
    "assemble_field_operator",
    "field_coupling",
    "field_heat",
    "field_heat_trajectory",
    "field_wave",
    "field_wave_trajectory",
    "field_wave_full",
    "field_metric",
]


def field_coupling(rex):
    """The coupling g = 1/max(||B2||_F, 1) (matches core._field / graph.field_coupling_psd)."""
    from rexgraph.core._sparse import to_scipy_csr
    if int(rex.nF_hodge) == 0 or rex._B2_hodge_dual is None:
        return 1.0
    B2 = to_scipy_csr(rex._B2_hodge_dual).tocsr()
    b2f = float(np.sqrt(B2.multiply(B2).sum()))
    return 1.0 / (b2f if b2f > 1.0 else 1.0)


def _field_blocks(rex, g=None):
    """(RL1, L2, B2, g, nE, nF) as scipy CSR - the sparse pieces of M. RL1 is the
    relational Laplacian if built, else L1 (same fallback as graph.field_coupling_psd)."""
    from rexgraph.core._sparse import to_scipy_csr
    nE = int(rex.nE)
    nF = int(rex.nF_hodge)
    if g is None:
        g = field_coupling(rex)
    RL1 = rex.relational_laplacian_sparse
    RL1 = RL1.tocsr() if RL1 is not None else rex.L1_sparse.tocsr()
    if nF > 0 and rex._B2_hodge_dual is not None:
        B2 = to_scipy_csr(rex._B2_hodge_dual).tocsr()          # nE x nF
        L2 = rex.L2_sparse.tocsr()
    else:
        B2 = sp.csr_matrix((nE, 0), dtype=_f64)
        L2 = sp.csr_matrix((0, 0), dtype=_f64)
    if isinstance(g, (bool, np.bool_)) or not np.isfinite(float(g)):
        raise ValueError("field coupling must be a finite real scalar")
    return RL1, L2, B2, float(g), nE, nF


def assemble_field_operator(rex, g=None):
    """The field operator M as a SPARSE (nE+nF) x (nE+nF) CSR block matrix - O(nnz),
    never the dense form. Symmetric, but not necessarily PSD. The defined coupling is not clipped."""
    RL1, L2, B2, g, nE, nF = _field_blocks(rex, g)
    if nF == 0:
        return RL1.tocsr()
    return sp.bmat([[RL1, (-g) * B2], [(-g) * B2.T, L2]], format="csr")


def _as_field_block(F, nE, nF):
    """Numerical real/complex tensor state; the leading axis is the cell space."""
    raw = np.asarray(F)
    F = np.asarray(raw, dtype=np.complex128 if np.iscomplexobj(raw) else _f64)
    if F.ndim < 1 or not np.all(np.isfinite(F)):
        raise ValueError("field state must have a cell axis and finite coefficients")
    N = nE + nF
    if F.shape[0] == N:
        return F
    if F.shape[0] == nE and nF > 0:
        return np.concatenate([F, np.zeros((nF,) + F.shape[1:], dtype=F.dtype)], axis=0)
    raise ValueError(f"field state has length {F.shape[0]}, expected {nE} or {N}")


def _symmetric_csr(value, N, name):
    """Validate a real numerical form, admitting only roundoff sized asymmetry."""
    raw = value if sp.issparse(value) else np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real symmetric")
    R = sp.csr_matrix(raw, dtype=_f64)
    if R.shape != (N, N) or not np.all(np.isfinite(R.data)):
        raise ValueError(f"{name} must be finite with shape {(N, N)}")
    R.sum_duplicates()
    R.eliminate_zeros()
    difference = R - R.T
    tolerance = 32 * np.finfo(float).eps * np.max(abs(R.data), initial=0.)
    if np.max(abs(difference.data), initial=0.) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    if difference.nnz:
        R = (R + R.T) * 0.5
    return R


def field_metric(rex, W=None):
    """Identity, positive diagonal, or sparse full SPD form on C1 + C2.

    Default edge metric entries are abs(w_E), face entries one. Zero weights do
    not define an invertible metric. A full form is validated/factored on use;
    no dense matrix or eigensolve is needed. Near one diagonals are not erased.
    """
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    N = nE + nF
    if W is None:
        w_E = getattr(rex, "w_E", None)
        if w_E is None:
            return "identity", None
        d = np.ones(N, dtype=_f64)
        raw = np.asarray(w_E)
        if np.iscomplexobj(raw) or raw.shape != (nE,):
            raise ValueError("default field metric needs one real weight per edge")
        d[:nE] = abs(np.asarray(raw, dtype=_f64))
    elif sp.issparse(W):
        if W.shape in ((1, N), (N, 1)) and W.shape != (N, N):
            if np.iscomplexobj(W):
                raise ValueError("field metric must be real")
            vector = W.tocoo()
            d = np.zeros(N, dtype=_f64)
            np.add.at(d, vector.col if W.shape[0] == 1 else vector.row, vector.data)
        else:
            return "full", _symmetric_csr(W, N, "field metric")
    else:
        raw = np.asarray(W)
        if raw.ndim != 1:
            return "full", _symmetric_csr(raw, N, "field metric")
        if np.iscomplexobj(raw):
            raise ValueError("field metric must be real")
        d = np.asarray(raw, dtype=_f64)
    if d.shape != (N,) or not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError(f"field metric diagonal must have {N} finite positive entries")
    return ("identity", None) if np.all(d == 1) else ("diag", d)


def _interval(M):
    """Two sided Gershgorin enclosure, including negative modes, in O(nnz)."""
    if M.shape[0] == 0:
        return 0., 0.
    diagonal = M.diagonal()
    off_diagonal = M.copy()
    off_diagonal.setdiag(0.)
    off_diagonal.eliminate_zeros()
    radius = np.asarray(abs(off_diagonal).sum(axis=1)).ravel()
    lo, hi = float(np.min(diagonal - radius)), float(np.max(diagonal + radius))
    if not off_diagonal.nnz and lo == hi:
        return lo, hi
    padding = 32 * np.finfo(float).eps * max(abs(lo), abs(hi))
    return np.nextafter(lo - padding, -np.inf), np.nextafter(hi + padding, np.inf)


class _FactoredMetric:
    """Sparse W = P.T L D L.T P; apply its conjugation without forming S.

    SuperLU's symmetric ordering and disabled row pivoting give an LDL.T
    factorization for SPD input. Check permutations, positive pivots and the
    U = D L.T residual; a failed numerical factorization is never silently used.
    Sparse factors can fill in. No claim of O(nnz(W)) factorization/storage.
    """

    def __init__(self, W):
        from scipy.sparse.linalg import splu, spsolve_triangular
        self.solve = spsolve_triangular
        n = W.shape[0]
        try:
            lu = splu(W.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.,
                      options={"SymmetricMode": True, "Equil": False})
        except RuntimeError as exc:
            raise ValueError("field metric must be positive definite and numerically factorable") from exc
        diagonal = lu.U.diagonal()
        if (not np.array_equal(lu.perm_r, lu.perm_c)
                or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0)):
            raise ValueError("field metric must be positive definite")
        self.L = lu.L.tocsr()
        self.LT = self.L.T.tocsr()
        defect = lu.U - sp.diags(diagonal) @ self.LT
        scale = np.max(abs(lu.U.data), initial=0.)
        if np.max(abs(defect.data), initial=0.) > 64 * np.finfo(float).eps * max(1, n) * scale:
            raise ValueError("field metric symmetric factorization residual is too large")
        self.P = sp.csr_matrix((np.ones(n), (lu.perm_r, np.arange(n))), shape=(n, n))
        self.root = np.sqrt(diagonal)
        self.inverse_root = 1. / self.root

        # |L^-1| <= (I - |strict_lower(L)|)^-1, a finite triangular
        # Neumann series. These two sparse solves bound ||C^-1||inf and
        # ||C^-1||1, C = P.T L sqrt(D); no inverse matrix is constructed.
        comparison = sp.eye(n, format="csr") - abs(sp.tril(self.L, -1, format="csr"))
        row_mass = self.solve(comparison, np.ones(n), lower=True, unit_diagonal=True)
        col_mass = self.solve(comparison.T.tocsr(), self.inverse_root,
                              lower=False, unit_diagonal=True)
        self.inverse_norm_squared_bound = float(
            np.max(self.inverse_root * row_mass) * np.max(col_mass))
        if not np.isfinite(self.inverse_norm_squared_bound):
            raise ValueError("field metric inverse bound exceeds numerical range")

    def encode(self, X):
        return self.root[:, None] * (self.LT @ (self.P @ X))

    def decode(self, X):
        return self.P.T @ self.solve(self.LT, self.inverse_root[:, None] * X,
                                     lower=False, unit_diagonal=True)

    def action(self, M, X):
        Y = self.P @ (M @ self.decode(X))
        return self.inverse_root[:, None] * self.solve(self.L, Y, lower=True,
                                                       unit_diagonal=True)


def _prepare(rex, F, g, M, W):
    nE, nF = int(rex.nE), int(rex.nF_hodge)
    Fb = _as_field_block(F, nE, nF)
    N = nE + nF
    M = _symmetric_csr(assemble_field_operator(rex, g) if M is None else M,
                       N, "field operator")
    kind, metric = field_metric(rex, W)
    if N == 0:
        kind = "identity"
    if kind == "identity":
        return Fb, M.__matmul__, _interval(M), lambda X: X, lambda X: X
    if kind == "diag":
        root = np.sqrt(metric)
        inv = 1. / root
        D = sp.diags(inv)
        S = (D @ M @ D).tocsr()
        return (Fb, S.__matmul__, _interval(S), lambda X: root[:, None] * X,
                lambda X: inv[:, None] * X)
    factor = _FactoredMetric(metric)
    lo, hi = _interval(M)
    inverse_bound = factor.inverse_norm_squared_bound
    metric_bound = _spg._gershgorin_bound(metric)
    lo = lo * inverse_bound if lo < 0 else lo / metric_bound
    hi = hi * inverse_bound if hi > 0 else hi / metric_bound
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("field operator bound exceeds numerical range under this metric")
    padding = 64 * np.finfo(float).eps * max(1, N) * max(abs(lo), abs(hi))
    return (Fb, lambda X: factor.action(M, X), (lo - padding, hi + padding),
            factor.encode, factor.decode)


def _times(times):
    raw = np.asarray(times)
    if raw.ndim != 1 or np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError("times must be a one-dimensional real sequence")
    out = np.asarray(raw, dtype=_f64)
    if not np.all(np.isfinite(out)):
        raise ValueError("times must be finite")
    return out


def _time(t):
    from numbers import Real
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, Real) or not np.isfinite(float(t)):
        raise ValueError("time must be a finite real scalar")
    return float(t)


def _order(times, interval, given, wave):
    from numbers import Integral
    if given is not None:
        if isinstance(given, (bool, np.bool_)) or not isinstance(given, Integral) or given < 2:
            raise ValueError("Chebyshev order must be an integer at least two")
        return int(given)
    size = max(abs(interval[0]), abs(interval[1]))
    phase = float(np.max(abs(times), initial=0.)) * (np.sqrt(size) if wave else size)
    if not np.isfinite(phase):
        raise ValueError("field evolution scale exceeds numerical range")
    # No silent degree cap. This is a numerical polynomial order heuristic, not
    # an exactness or tolerance certificate. Callers may specify their own order.
    return max(32, int(np.ceil(2. * phase)) + 32)


def _wave_values(lam, t, kind):
    """Entire wave functions C_t, S_t and C'_t; no spectral projection/clipping."""
    rate = np.sqrt(abs(lam))
    z = t * rate
    positive = lam >= 0.
    out = np.empty_like(lam)
    if kind == "cos":
        out[positive] = np.cos(z[positive])
        out[~positive] = np.cosh(z[~positive])
    elif kind == "sin":
        out[positive] = t * np.sinc(z[positive] / np.pi)
        out[~positive] = np.sinh(z[~positive]) / rate[~positive]
    else:  # derivative of C_t
        out[positive] = -rate[positive] * np.sin(z[positive])
        out[~positive] = rate[~positive] * np.sinh(z[~positive])
    return out


def _functions(action, X, interval, functions, order):
    """One streaming sparse recurrence for every requested scalar function.

    O(order * action_cost + outputs * order * state_size) work, three recurrence
    states rather than an order by state history. DCT II replaces the quadratic
    cosine table. This CPU path also handles factored metrics and complex states.
    """
    from scipy.fft import dct
    lo, hi = interval
    if not functions:
        return np.empty((0,) + X.shape, dtype=X.dtype)
    if X.size == 0:
        return np.zeros((len(functions),) + X.shape, dtype=X.dtype)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if lo == hi:
            factors = np.array([fn(np.array([lo]))[0] for fn in functions])
            return factors[:, None, None] * X[None, :, :]
        center, radius = lo / 2 + hi / 2, hi / 2 - lo / 2
        nodes = center + radius * np.cos(np.pi * (np.arange(order) + .5) / order)
        values = np.array([fn(nodes) for fn in functions])
        coefficients = dct(values, type=2, axis=1, norm="backward") / order
        coefficients[:, 0] *= .5
        # Reuse the live scale GPU kernels and placement policy. Shifting S by
        # lo maps the two sided interval to their [0, hi lo] contract without
        # changing this polynomial. Complex/factored and multi function actions
        # stay on the shared CPU recurrence (the GPU kernel is real only).
        matrix = getattr(action, "__self__", None)
        work = X.size * order
        if (len(functions) == 1 and not np.iscomplexobj(X) and sp.issparse(matrix)
                and work >= _spg._GPU_MIN_WORK and _spg._resolve_backend(None) == "gpu"):
            try:
                shifted = (matrix - lo * sp.eye(matrix.shape[0], format="csr")).tocsr()
                plan = _spg._multi_gpu_plan(work, X.shape[1])
                if plan is None:
                    accelerated = _spg._matfunc_gpu(shifted, X, coefficients[0], hi - lo, order)
                else:
                    accelerated = _spg._matfunc_gpu_multi(
                        shifted, X, coefficients[0], hi - lo, order, plan)
                if accelerated.shape == X.shape and np.isfinite(accelerated).all():
                    return accelerated[None, :, :]
            except Exception:
                pass  # Preserve the scale backend's CPU fallback on GPU failure.
        previous = X
        current = (action(X) - center * X) / radius
        out = coefficients[:, 0, None, None] * previous
        out += coefficients[:, 1, None, None] * current
        for k in range(2, order):
            following = 2. * (action(current) - center * current) / radius - previous
            out += coefficients[:, k, None, None] * following
            previous, current = current, following
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("field evolution exceeds numerical range")
    return out


def _run(prepared, functions, order, state=None):
    Fb, action, interval, encode, decode = prepared
    if state is not None:
        Fb = state
    ncols = int(np.prod(Fb.shape[1:], dtype=np.int64))
    X = Fb.reshape(Fb.shape[0], ncols)
    values = _functions(action, encode(X), interval, functions, order)
    # Decode one time slice at a time: tensor axes are never mistaken for cells.
    out = np.empty((len(functions),) + Fb.shape, dtype=Fb.dtype)
    for i, value in enumerate(values):
        out[i] = decode(value).reshape(Fb.shape)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("field evolution exceeds numerical range")
    return out


def field_heat(rex, F, t, g=None, order=None, M=None, W=None):
    """Numerical exp(-t W^-1 M)F, including indefinite generators.

    Sparse actions, finite real t, no eigendecomposition or dense conjugation.
    Full SPD metrics use sparse factors (which may fill in). Overflow is reported.
    F has a cell leading real/complex tensor shape; edge only inputs lift to C1+C2.
    """
    return field_heat_trajectory(rex, F, [_time(t)], g, order, M, W)[0]


def field_heat_trajectory(rex, F, times, g=None, order=None, M=None, W=None):
    """Heat at arbitrary times, with bounded step propagation for long intervals.

    Short trajectories share one recurrence. Long ones advance monotonically on
    each side of zero using the semigroup law and reuse the prepared metric.
    Bounding each step avoids coefficient cancellation caused by a loose negative
    spectral enclosure, even when the true dynamics barely grow. An explicit
    order applies per step, not to one unbounded time polynomial.
    """
    times = _times(times)
    prepared = _prepare(rex, F, g, M, W)
    degree = _order(times, prepared[2], order, wave=False)
    Fb, action, interval, encode, decode = prepared
    bound = max(abs(interval[0]), abs(interval[1]))
    if np.max(abs(times), initial=0.) * bound <= 4. or interval[0] == interval[1]:
        funcs = [lambda lam, t=t: np.exp(-t * lam) for t in times]
        out = _run(prepared, funcs, degree)
    else:
        ncols = int(np.prod(Fb.shape[1:], dtype=np.int64))
        initial = encode(Fb.reshape(Fb.shape[0], ncols))
        out = np.empty((len(times),) + Fb.shape, dtype=Fb.dtype)
        for sign in (1, -1):
            selected = np.flatnonzero(times * sign > 0)
            selected = selected[np.argsort(abs(times[selected]))]
            current, previous_time = initial, 0.
            for i in selected:
                elapsed = times[i] - previous_time
                steps = max(1, int(np.ceil(abs(elapsed) * bound / 4.)))
                step = elapsed / steps
                step_order = _order(np.array([step]), interval, order, wave=False)
                for _ in range(steps):
                    current = _functions(action, current, interval,
                                         [lambda lam, step=step: np.exp(-step * lam)], step_order)[0]
                out[i] = decode(current).reshape(Fb.shape)
                previous_time = times[i]
    out[times == 0.] = prepared[0]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("field evolution exceeds numerical range")
    return out


def field_wave(rex, F, t, g=None, order=None, M=None, W=None):
    """Zero initial velocity wave: cos on positive modes, cosh on negative modes."""
    return field_wave_trajectory(rex, F, [_time(t)], g, order, M, W)[0]


def field_wave_trajectory(rex, F, times, g=None, order=None, M=None, W=None):
    """Zero initial velocity positions; shared recurrence across arbitrary times."""
    times = _times(times)
    prepared = _prepare(rex, F, g, M, W)
    degree = _order(times, prepared[2], order, wave=True)
    if _wave_needs_steps(times, prepared[2]):
        return _wave_steps(prepared, prepared[0], np.zeros_like(prepared[0]), times, order)[0]
    funcs = [lambda lam, t=t: _wave_values(lam, t, "cos") for t in times]
    out = _run(prepared, funcs, degree)
    out[times == 0.] = prepared[0]
    return out


def _wave_needs_steps(times, interval):
    # A conservative negative enclosure must not make one enormous cosh
    # polynomial swamp bounded true modes through coefficient cancellation.
    return np.max(abs(times), initial=0.) * np.sqrt(max(0., -interval[0])) > 4.


def _wave_steps(prepared, F, V, times, order):
    _, action, interval, encode, decode = prepared
    ncols = int(np.prod(F.shape[1:], dtype=np.int64))
    initial = encode(np.concatenate([F.reshape(F.shape[0], ncols),
                                     V.reshape(V.shape[0], ncols)], axis=1))
    positions = np.empty((len(times),) + F.shape, dtype=F.dtype)
    velocities = np.empty_like(positions)
    growth = np.sqrt(max(0., -interval[0]))
    for sign in (1, -1):
        selected = np.flatnonzero(times * sign > 0)
        selected = selected[np.argsort(abs(times[selected]))]
        current, previous_time = initial, 0.
        for i in selected:
            elapsed = times[i] - previous_time
            steps = max(1, int(np.ceil(abs(elapsed) * growth / 4.)))
            step = elapsed / steps
            degree = _order(np.array([step]), interval, order, wave=True)
            for _ in range(steps):
                funcs = [lambda lam, kind=kind, step=step: _wave_values(lam, step, kind)
                         for kind in ("cos", "sin", "derivative")]
                C, S, D = _functions(action, current, interval, funcs, degree)
                current = np.concatenate([C[:, :ncols] + S[:, ncols:],
                                          D[:, :ncols] + C[:, ncols:]], axis=1)
            decoded = decode(current)
            positions[i] = decoded[:, :ncols].reshape(F.shape)
            velocities[i] = decoded[:, ncols:].reshape(F.shape)
            previous_time = times[i]
    positions[times == 0.] = F
    velocities[times == 0.] = V
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise FloatingPointError("field evolution exceeds numerical range")
    return positions, velocities


def field_wave_full(rex, F, times, g=None, order=None, M=None, W=None, *, velocity=None):
    """Position and velocity for F'' = -W^-1 M F, including arbitrary F'(0).

    C_t(K)=sum (-K)^j t^(2j)/(2j)!, S_t(K)=sum (-K)^j t^(2j+1)/(2j+1)!:
      F(t)  = C_t(K) F(0) + S_t(K) F'(0)
      F'(t) = C'_t(K) F(0) + C_t(K) F'(0).
    S_t(0)=t, so kernel velocities drift rather than disappear. Position and
    velocity functions share one recurrence per nonzero initial state. Long
    growth intervals use bounded steps on the joint position/velocity state.
    """
    times = _times(times)
    prepared = _prepare(rex, F, g, M, W)
    Fb = prepared[0]
    initial_velocity = (np.zeros_like(Fb) if velocity is None
                        else _as_field_block(velocity, int(rex.nE), int(rex.nF_hodge)))
    if initial_velocity.shape != Fb.shape:
        raise ValueError("initial velocity and position must have the same tensor shape")
    dtype = np.result_type(Fb.dtype, initial_velocity.dtype)
    Fb = Fb.astype(dtype, copy=False)
    initial_velocity = initial_velocity.astype(dtype, copy=False)
    degree = _order(times, prepared[2], order, wave=True)
    if _wave_needs_steps(times, prepared[2]):
        return _wave_steps(prepared, Fb, initial_velocity, times, order)
    cosines = [lambda lam, t=t: _wave_values(lam, t, "cos") for t in times]
    derivatives = [lambda lam, t=t: _wave_values(lam, t, "derivative") for t in times]
    result = _run(prepared, cosines + derivatives, degree, state=Fb)
    pos, vel = result[:len(times)], result[len(times):]
    if np.any(initial_velocity):
        sines = [lambda lam, t=t: _wave_values(lam, t, "sin") for t in times]
        extra = _run(prepared, sines + cosines, degree, state=initial_velocity)
        pos += extra[:len(times)]
        vel += extra[len(times):]
    pos[times == 0.] = Fb
    vel[times == 0.] = initial_velocity
    return pos, vel
