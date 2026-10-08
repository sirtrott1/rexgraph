"""rexgraph.sparse_character: scale free character / coherence via a sparse RL.

The dense character path (``build_all_laplacians`` -> ``build_RL`` ->
``compute_chi`` / ``build_character_bundle``) forms dense nE x nE channel
Laplacians, RL, and hats, and was therefore gated to ``nE <= eigen_dense_limit``.
But every RCF channel operator is sparse, so chi / phi / kappa have no intrinsic
size ceiling: the dense ceiling was an implementation choice, not the math.

This module assembles the four channels as scipy CSR, sums the trace normalized
hats into a sparse RL, and computes exactly what the dense path computes:

    chi(e,k)   = hat_k[e,e] / RL[e,e]                          (diagonals only)
    phi(v,k)   = x_v^T hat_k x_v / (b_v^T x_v),  x_v = RL^-1 b_v,  b_v = B1[v,:]^T
    chi*(v,k)  = mean_{e in star(v)} chi(e,k)
    kappa(v)   = 1 - 0.5 * ||phi(v) - chi*(v)||_1

The native phi evaluator uses preconditioned CG on sparse RL actions, batched
over right hand sides. Its numerical residual is checked explicitly; floating
evaluation is not exact rational arithmetic. No dense pseudoinverse or
eigendecomposition is needed.
Verified identical to the dense path on small graphs (``test`` via the graph
properties); this is the path that removes the arbitrary size limit on character.

The four channels (matching the dense builders exactly):
  * L1_down = B1^T B1                                    (_laplacians.build_L1_down)
  * L_O     = raw |B1|^T|B1|, or normalized I - D^-1/2 K D^-1/2   (g_channel)
  * L_SG    = (T-G)_off + diag(sum|(T-G)_off|), using raw G
  * L_C     = D_L - A_L over the selected co participation reading (`rex.c_channel`):
              share (default, conserving) or count (structural). Independent readings,
              not a rescaling; they coincide at arity 2.
"""
from __future__ import annotations

import numpy as np

from rexgraph.compute import sparse_mm

_f64 = np.float64


def _b1_csr(rex):
    """Primary C1 boundary as scipy CSR, at its declared arity."""
    from rexgraph.core._sparse import to_scipy_csr
    return to_scipy_csr(rex._B1_dual).tocsr()


def _channel_metric(rex):
    """Numerical metric, refusing lost nonzero mass instead of reporting zero hats."""
    stored = getattr(rex, 'w_E', None)
    if stored is None:
        return None
    w = np.asarray(stored, dtype=_f64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        squared = w * w
    if (np.any(~np.isfinite(squared))
            or np.any((np.asarray(stored) != 0) & (squared == 0))):
        raise FloatingPointError('channel metric mass is outside float64; use exact_character')
    return w


def _require_distinct_channel_participants(rex):
    """Guard the shared aggregated/slot-incidence channel identity."""
    for support in rex.relation_supports():
        if len(set(support)) != len(support):
            raise ValueError('this channel reading requires distinct participants per relation; repeated-slot conventions differ')


def build_sparse_channels(rex, *, native=False):
    """Assemble the four channels through core sparse products.

    Native storage is available without SciPy. The default remains an explicit
    SciPy compatibility export for existing callers. Factored actions do not
    call this materializer; sparse Gram products may fill on large stars.
    """
    from rexgraph.core import _sparse
    from rexgraph.native_sparse import NativeSparse, native_diagonal
    rex._ensure_clean()
    B = NativeSparse(rex._B1_dual)
    unsigned = B.with_data(abs(B.data))
    weight = _channel_metric(rex)
    weight = np.ones(B.shape[1]) if weight is None else weight
    Bw = B.with_data(B.data * weight[B.dual.col_idx])
    Aw = unsigned.with_data(unsigned.data * weight[unsigned.dual.col_idx])
    T = Bw.T.product(Bw)
    G = Aw.T.product(Aw)
    selected_g = G
    if rex.g_channel != 'raw':
        if np.any(weight < 0):
            raise ValueError('normalized G requires nonnegative relation weights')
        degree = G.apply(np.ones(G.shape[1]))
        if not np.all(np.isfinite(degree)) or np.any(degree < 0):
            raise ValueError('normalized G requires finite nonnegative row mass')
        inverse = np.zeros_like(degree)
        nz = degree > 0
        inverse[nz] = 1 / np.sqrt(degree[nz])
        rows = np.repeat(np.arange(G.shape[0]), np.diff(G.dual.row_ptr))
        scaled = G.with_data(G.data * (inverse[rows] * inverse[G.dual.col_idx]))
        # The isolated diagonal is one. For active rows, use the off mass
        # directly so near cancellation in 1 minus the normalized diagonal
        # does not change the diagonal contract.
        off = G.add(native_diagonal(G.diagonal()), -1)
        diagonal = np.ones_like(degree)
        diagonal[nz] = off.apply(np.ones(G.shape[1]))[nz] / degree[nz]
        scaled_off = scaled.add(native_diagonal(scaled.diagonal()), -1)
        selected_g = native_diagonal(diagonal).add(scaled_off, -1)
    difference = T.add(G, -1)
    difference = difference.add(native_diagonal(difference.diagonal()), -1)
    absolute = difference.with_data(abs(difference.data))
    F = difference.add(native_diagonal(absolute.apply(np.ones(G.shape[1]))))
    Cfactor = unsigned
    if rex.c_channel == 'count':
        support = NativeSparse(_sparse.canonical_dual(unsigned.dual))
        Cfactor = support.with_data(np.ones(support.nnz))
    Kc = Cfactor.T.product(Cfactor)
    off = Kc.add(native_diagonal(Kc.diagonal()), -1)
    C = native_diagonal(off.apply(np.ones(Kc.shape[1]))).add(off, -1)
    channels = [('L1_down', T), ('L_O', selected_g), ('L_SG', F), ('L_C', C)]
    return channels if native else [(name, matrix.as_scipy()) for name, matrix in channels]


def _trace_normalized(L, tr, nE):
    """`L / tr` as CSR, or a zero matrix when the channel carries no mass.

    A channel is kept whatever its trace, so this is the one place that decides what
    a zero trace hat looks like. Normalising by its own zero trace would be 0/0;
    the hat is the zero operator, which is what "this channel measures nothing here"
    means, and it contributes nothing to RL.
    """
    import scipy.sparse as sp
    if not np.isfinite(tr) or tr < 0:
        raise ValueError("channel trace must be finite and nonnegative")
    if tr > 0:
        result = L.astype(_f64, copy=True).tocsr()
        result.data /= tr
        if not np.all(np.isfinite(result.data)):
            raise FloatingPointError("channel normalization is not representable")
        return result
    if L.nnz and np.any(L.data != 0):
        raise ValueError('zero-trace channel must be the zero operator')
    return sp.csr_matrix((nE, nE), dtype=_f64)


def build_sparse_rl(rex):
    """Assemble the sparse RL from trace normalized hats. Returns
    ``(RL_csr, hats, names, traces)``, matching ``_relational.build_RL``.

    Every channel is kept. One carrying no mass contributes a zero hat rather
    than being dropped, so ``hats`` and ``names`` have a fixed width and the
    character built from them is comparable between complexes. ``tr(RL)`` is
    therefore the number of channels carrying mass, which is ``nhats`` whenever
    none of them is degenerate and less when one is.
    """
    import scipy.sparse as sp
    nE = int(rex.nE)
    hats, names, traces = [], [], []
    RL = sp.csr_matrix((nE, nE), dtype=_f64)
    for name, L in build_sparse_channels(rex):
        tr = float(L.diagonal().sum())
        hat = _trace_normalized(L, tr, nE)
        if hat.nnz:
            RL = (RL + hat).tocsr()
        hats.append(hat)
        names.append(name)
        traces.append(tr)
    return RL, hats, names, traces


def _block_cg(apply_A, B, dinv, tol=1e-10, maxit=1000, *, return_info=False):
    """Jacobi preconditioned block conjugate gradient: solve A X = B for all
    columns of B at once (A SPD), where ``apply_A(P)`` returns A @ P. Matrix free -
    no factorization. The supplied action may use assembled sparse channels or
    incidence factors. Returns X only when the recomputed relative residual is
    below tol for every column; otherwise raises ArithmeticError. Trace
    normalization alone is not a condition number guarantee. ``return_info``
    additionally returns per column iterations and recomputed residuals; no
    global mutable solver state is used."""
    X = np.zeros_like(B)
    R = B - apply_A(X)
    Z = dinv[:, None] * R
    P = Z.copy()
    rz = (R * Z).sum(0)
    bnorm = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    # Each column is an independent CG: alpha and beta are per column scalars and the
    # only thing shared is the operator. So a converged column can LEAVE the block,
    # and it must, because the loop below stops on the worst column and every column
    # still in the block pays that column's iteration count. Measured on two ontology
    # slices, 596 columns took 117 iterations and 1000 took 286: the width was driving
    # the depth.
    iterations = np.zeros(B.shape[1], dtype=np.int64)
    active = np.flatnonzero(np.linalg.norm(R, axis=0) / bnorm >= tol)
    R, P, rz = R[:, active], P[:, active], rz[active]
    for _ in range(maxit):
        if not active.size:
            break
        iterations[active] += 1
        AP = apply_A(P)
        # A is PSD, so curv = P^T A P is >= 0 exactly, and curv == 0 means this
        # direction has no curvature: the column is converged or P has landed in
        # ker(A). Clamping the denominator up to 1e-300 instead turns that into
        # rz/1e-300 = +-inf, and X += alpha*P then makes the whole column NaN. On a
        # 66 component ontology slice that was 348 of 385 relations.
        curv = (P * AP).sum(0)
        alpha = np.zeros_like(rz)
        np.divide(rz, curv, out=alpha, where=curv > 0.0)
        X[:, active] += alpha * P
        R -= alpha * AP
        keep = (np.linalg.norm(R, axis=0) / bnorm[active]) >= tol
        if not keep.any():
            break
        if not keep.all():                      # retire the converged columns
            active = active[keep]
            R, P, rz, AP = R[:, keep], P[:, keep], rz[keep], AP[:, keep]
        Z = dinv[:, None] * R
        rz_new = (R * Z).sum(0)
        beta = np.zeros_like(rz)
        np.divide(rz_new, rz, out=beta, where=rz > 0.0)
        P = Z + beta * P
        rz = rz_new
    relative = np.linalg.norm(B - apply_A(X), axis=0) / bnorm
    if not np.all(np.isfinite(relative)) or np.any(relative >= tol):
        worst = float(relative.max(initial=0.0))
        raise ArithmeticError(f"block CG did not meet residual tolerance {tol:g}; worst {worst:g}")
    return (X, {"iterations": iterations.tolist(), "relative_residuals": relative.tolist()}) if return_info else X


def factored_channel_actions(rex, names, traces):
    """Return channel hat actions and their sum through the selected incidence profile.

    G follows the source's raw or normalized selection. F uses raw unsigned G
    in either profile. Repeated slots follow the assembled numerical channel
    reading. No relation pair matrix is formed.
    """
    from rexgraph.channel_operator import _build_numerical_factored_operator
    apply_rl, apply_hat, _ = _build_numerical_factored_operator(rex, names, traces)
    return apply_rl, apply_hat


class _CheapCharacter(dict):
    """The cheap character bundle, with the channel OPERATORS built on first read.

    chi, chi_star and rl_diag are diagonals and cost O(nnz). RL and hats are the
    edge x edge operators, wanted only by the opt in per vertex Green's path, and they
    are the expensive part (sum_v deg(v)^2 nonzeros). Resolving them on access keeps
    the always affordable layer actually affordable.
    """

    _LAZY = ('RL', 'hats')

    def __init__(self, base, rex, names, traces):
        super().__init__(base)
        self._rex = rex
        self._names = list(names)
        self._traces = list(traces)
        self._filled = False

    def _fill(self) -> None:
        if self._filled:
            return
        import scipy.sparse as sp
        nE = int(self._rex.nE)
        chan = dict(build_sparse_channels(self._rex))
        hats = []
        RL = sp.csr_matrix((nE, nE), dtype=_f64)
        for name, tr in zip(self._names, self._traces, strict=True):
            L = chan.get(name)
            if L is None:
                continue
            hat = _trace_normalized(L, tr, nE)
            hats.append(hat)
            if hat.nnz:
                RL = (RL + hat).tocsr()
        dict.__setitem__(self, 'RL', RL)
        dict.__setitem__(self, 'hats', hats)
        self._filled = True

    def __getitem__(self, key):
        if key in self._LAZY and not self._filled:
            self._fill()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in self._LAZY and not self._filled:
            self._fill()
        return dict.get(self, key, default)

    def __iter__(self):
        # also takes dict(...) / {**...} off CPython's dict to dict fast path
        self._fill()
        return dict.__iter__(self)

    def keys(self):
        self._fill()
        return dict.keys(self)

    def values(self):
        self._fill()
        return dict.values(self)

    def items(self):
        self._fill()
        return dict.items(self)

    def copy(self):
        self._fill()
        return dict(self)


def closed_form_applies(rex) -> bool:
    """True when `channel_diagonals` is exact for this complex: every relation binary,
    every entry +-1, and no edge weighting. Structural, O(nnz), no threshold."""
    import numpy as _np
    try:
        B1 = rex._B1_dual
        column_ptr, values = B1.col_ptr, B1.vals_csc
    except AttributeError:
        return False
    if int(rex.nE) == 0:
        return False
    if _np.any(_np.diff(column_ptr) != 2):          # every relation binary
        return False
    if _np.any(B1.row_idx[column_ptr[:-1]] == B1.row_idx[column_ptr[:-1] + 1]):
        return False
    if not _np.all(_np.abs(_np.asarray(values)) == 1.0):
        return False
    w = getattr(rex, 'w_E', None)                    # no edge weighting
    if w is not None and not _np.all(_np.asarray(w) == 1):
        return False
    for attr in ('w_V', 'vertex_weights'):
        wv = getattr(rex, attr, None)
        if wv is not None and not _np.all(_np.asarray(wv) == 1):
            return False
    return True


def _count_c_diagonal(rex):
    """C-count diagonal from support incidence, without forming the line graph."""
    rex._ensure_clean()
    bp, bi = rex._boundary_ptr, rex._boundary_idx
    supports = [set(map(int, bi[bp[e]:bp[e + 1]])) for e in range(int(rex.nE))]
    degrees = {}
    for support in supports:
        for vertex in support:
            degrees[vertex] = degrees.get(vertex, 0) + 1
    return [sum(degrees[v] - 1 for v in support) for support in supports]


def channel_diagonals_integer(rex):
    """The four diagonals as EXACT int64 numerators over one common denominator.

    No float enters at any point. The share 1/(k-1) becomes the integer L/(k-1) for
    L = lcm(k-1), so a pairwise complex works at L = 1 and needs no denominator at all.
    C and F multiply two shares, so the whole tower lands over L^2 and that single
    integer is returned beside it.

    Returns ({name: int64 numerators}, scale) or (None, None) when the scale would
    leave int64, which is when the approximation tower is the honest answer and
    `channel_diagonals` gives it.

    A DECLARED share carries its own denominator, which is not lcm(k-1), so it takes the
    route a weighted complex already takes: the exact reader, scaled into this carrier.
    """
    from collections import defaultdict

    import numpy as _np

    rex._ensure_clean()
    if getattr(rex, "g_channel", "raw") != "raw":
        return None, None
    if rex.edge_metric_exact is not None or getattr(rex, "declares_columns", False):
        # Relation weights, and a declared share, both add denominators beyond the arity
        # only scale. Reuse the exact reader and fit its result into the fixed width
        # carrier.
        from math import lcm

        from rexgraph.rational_trig import exact_channel_diagonals
        diagonals, names = exact_channel_diagonals(rex)
        if diagonals is None:
            return None, None
        scale = 1
        for values in diagonals.values():
            for value in values:
                scale = lcm(scale, value.denominator)
        numerators = {name: [int(value*scale) for value in diagonals[name]] for name in names}
        limits = _np.iinfo(_np.int64)
        if any(value < limits.min or value > limits.max
               for values in numerators.values() for value in values):
            return None, None
        return {name: _np.asarray(values, dtype=_np.int64)
                for name, values in numerators.items()}, scale
    bp = _np.asarray(rex._boundary_ptr)
    bi = _np.asarray(rex._boundary_idx)
    prec = channel_tower_precision(bp, bi)
    if prec["scale"] is None:
        return None, None
    L2 = int(prec["scale"])
    L = int_sqrt_limit(L2)
    nE = int(rex.nE)

    # |c| as an integer over L: 1 becomes L, and the share 1/(k-1) becomes L/(k-1)
    negu = defaultdict(int)
    posu = defaultdict(int)
    cols = []
    for e in range(nE):
        span = [int(v) for v in bi[bp[e]:bp[e + 1]]]
        k = len(span)
        if k == 0:
            cols.append(([], 0)); continue
        if k == 1:                                   # a witness is (+1): positive
            cols.append(([(span[0], L, +1)], k))
            posu[span[0]] += L
            continue
        share = L // (k - 1)
        entries = [(span[0], L, -1)]                 # the head, magnitude 1
        negu[span[0]] += L
        for v in span[1:]:
            entries.append((v, share, +1))
            posu[v] += share
        cols.append((entries, k))

    T = _np.zeros(nE, dtype=_np.int64)
    C = _np.zeros(nE, dtype=_np.int64)
    Fc = _np.zeros(nE, dtype=_np.int64)
    for e, (entries, k) in enumerate(cols):
        if not entries:
            continue
        T[e] = L2 if k == 1 else L2 + L2 // (k - 1)
        c = 0
        f = 0
        for v, mag, sign in entries:
            total = negu[v] + posu[v]
            c += mag * (total - mag)
            f += mag * (posu[v] if sign < 0 else negu[v])
        C[e] = c
        Fc[e] = 2 * f
    if getattr(rex, "c_channel", "share") == "count":
        C = _np.asarray([value * L2 for value in _count_c_diagonal(rex)], dtype=_np.int64)
    return {"L1_down": T, "L_O": T.copy(), "L_SG": Fc, "L_C": C}, L2


def channel_tower_precision(bp, bi, *, int64_max=(1 << 62)):
    """Which numeric tower this complex's channel diagonals actually need.

    The share is 1/(k-1), which is 1 at a witness and at a pairwise relation, so a
    complex carrying only those has NO denominator and its whole tower is exact in
    int64. Branching introduces k-1, and C and F multiply two shares, so the common
    denominator is lcm(k-1)^2. Where that scale still fits an int64 the tower is exact
    in integers anyway; only past it is a float needed, and then the width is chosen
    from the magnitude rather than assumed.

    Returns {tower, scale, dtype, reason}. `scale` is what every diagonal must be
    multiplied by to be integral, so 1 means integral as it stands.
    """
    from math import gcd

    import numpy as _np

    k = _np.diff(_np.asarray(bp, dtype=_np.int64))
    if k.size == 0:
        return {"tower": "integer", "scale": 1, "dtype": "int64", "reason": "empty"}
    dens = _np.unique(_np.maximum(k - 1, 1))          # k=1 and k=2 both give 1
    lcm = 1
    for d in dens:
        d = int(d)
        lcm = lcm // gcd(lcm, d) * d
        if lcm > int_sqrt_limit(int64_max):
            return {"tower": "float", "scale": None, "dtype": "float64",
                    "reason": f"lcm(k-1) = {lcm} squares past int64"}
    scale = lcm * lcm
    if scale == 1:
        return {"tower": "integer", "scale": 1, "dtype": "int64",
                "reason": "every relation is a witness or pairwise, so the share is 1"}
    # the largest diagonal is bounded by the co participation mass, and the incidence
    # count bounds that, so this is a bound and not a sample
    bound = int(scale) * int(_np.bincount(_np.asarray(bi, dtype=_np.int64)).max() or 1) * int(k.sum())
    if bound < int64_max:
        return {"tower": "rational", "scale": int(scale), "dtype": "int64",
                "reason": f"exact over a common denominator of {scale}"}
    return {"tower": "float", "scale": None, "dtype": "float64",
            "reason": f"scaled bound {bound} past int64"}


def int_sqrt_limit(n: int) -> int:
    """The largest L with L*L <= n. Integer only, no float sqrt to round wrongly."""
    lo, hi = 0, 1
    while hi * hi <= n:
        hi <<= 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo


def _hip_reads_declared() -> bool:
    """Whether the built HIP object carries the declared column entry point."""
    try:
        from rexgraph import hip_ternary
        lib = hip_ternary._load()
    except Exception:                                    # noqa: BLE001  # absent is False
        return False
    return lib is not None and hasattr(lib, "tower_launch_coef")


def _any_arity_diagonals(rex):
    """The tower past the pairwise derivation, by accumulating at the vertex.

    Raw T/F/C and the raw G diagonal come from the compiled incidence tower.
    Normalized G needs one further incidence pass, not an assembled Gram matrix.

    A DECLARED head or share travels to the kernel as one coefficient per incidence, so
    this route reads the column the complex declares. It is the route every branching
    complex takes, because the closed form above applies only to a pairwise one.
    """
    import numpy as _np
    for attr in ("w_V", "vertex_weights"):
        wv = getattr(rex, attr, None)
        if wv is not None and not _np.all(_np.asarray(wv) == 1):
            return None
    bp = getattr(rex, "_boundary_ptr", None)
    bi = getattr(rex, "_boundary_idx", None)
    if bp is None or bi is None:
        return None
    from rexgraph import compute
    w = _channel_metric(rex)
    # Use the explicit default backend, or prefer OpenMP. Declared coefficients
    # fall back to OpenMP when the installed HIP lane lacks that capability.
    coefficients = None
    if getattr(rex, "declares_columns", False):
        from rexgraph.column import slot_coefficients
        coefficients = slot_coefficients(bp, bi, rex._declaration)
    prefer = compute.get_default_backend() or "openmp"
    if coefficients is not None and prefer == "hip" and not _hip_reads_declared():
        prefer = "openmp"
    declared = {} if coefficients is None else {"coefficients": coefficients}
    T, G, F, C = compute.dispatch("channel_tower",
                                  _np.asarray(bp), _np.asarray(bi), int(rex.nV),
                                  None if w is None else _np.asarray(w, float),
                                  prefer=prefer, **declared)
    if getattr(rex, "c_channel", "share") == "count":
        C = _np.asarray(_count_c_diagonal(rex), dtype=_f64)
    if getattr(rex, 'g_channel', 'raw') == 'normalized':
        _require_distinct_channel_participants(rex)
        if w is not None and _np.any(w < 0):
            raise ValueError('normalized G requires nonnegative relation weights')
        from rexgraph.native_sparse import NativeSparse
        boundary = NativeSparse(rex._B1_dual)
        unsigned = boundary.with_data(abs(boundary.data))
        if w is not None:
            unsigned = unsigned.with_data(unsigned.data * w[unsigned.dual.col_idx])
        degree = unsigned.transpose_apply(unsigned.apply(_np.ones(int(rex.nE))))
        if _np.any(~_np.isfinite(degree)):
            raise FloatingPointError('normalized G row mass is outside float64')
        G = _np.ones(int(rex.nE))
        nz = degree > 0
        own = unsigned.column_quadrances()
        # Use the same incidence arithmetic for self mass and row mass. Isolated
        # relations have exactly zero off mass, not a roundoff sized false channel.
        off_mass = _np.maximum(0, degree - own)
        G[nz] = off_mass[nz] / degree[nz]
    return {'L1_down': T, 'L_O': G, 'L_SG': F, 'L_C': C}


#### the lanes. Same registry as every other device specialised kernel here, so a new
#### architecture is a register_op call and nothing in this module moves.
def _tower_cpu(bp, bi, nV, w, threads=1, transposed=None, coefficients=None):
    from rexgraph.core._channel_tower import channel_diagonals_any_arity
    return channel_diagonals_any_arity(bp, bi, nV, w, threads, transposed, coefficients)


def _tower_width() -> int:
    """Return the explicit compute thread count, or the available physical core count."""
    from rexgraph import compute as _c
    explicit = _c.get_threads()
    if explicit:
        return int(explicit)
    from rexgraph.hardware import physical_cores
    return int(physical_cores())


def _tower_openmp(bp, bi, nV, w, transposed=None, coefficients=None):
    """Run the channel tower with the configured thread width.

    Transpose incidence into vertex ordered rows and assign each row to one thread.
    Passing transposed reuses an existing incidence transpose.
    """
    return _tower_cpu(bp, bi, nV, w, _tower_width(), transposed, coefficients)


def _register_tower_lanes():
    from rexgraph import compute as _c
    _c.register_op("channel_tower", "cpu", _tower_cpu)
    _c.register_op("channel_tower", "openmp", _tower_openmp)
    # The native HIP lane registers itself on import, so without an import it was
    # registered nowhere and could never be selected: `hip` was not a live backend at
    # all. Importing it costs a torch import (it has to go first, since both link
    # libamdhip64 and the loader binds one SONAME per process), so the import is
    # deferred behind the object's own presence. The filename is repeated rather than
    # read from `hip_ternary.library_path`, because reaching that function is the
    # import this guard exists to avoid.
    import contextlib
    import os
    if (os.environ.get("REXGRAPH_TERNARY_HIP")
            or os.path.exists(os.path.join(os.path.dirname(__file__), "core",
                                           "lib_ternary_hip.so"))):
        # an optional lane never breaks the tower
        with contextlib.suppress(Exception):
            import rexgraph.hip_ternary  # noqa: F401  registers on import


_register_tower_lanes()


def channel_arithmetic_contract(rex):
    """Arithmetic carried by the channel tower and used by its live evaluator.

    The raw channel structure is rational whenever its primary columns and relation
    metric are rational.  The production channel kernel evaluates those coefficients in
    float64.  ``integer_scale`` records when the same result also has an exact int64
    numerator carrier; it is a capability verdict, not a claim that the live evaluator
    executed in integer arithmetic.
    """
    rex._ensure_clean()
    if getattr(rex, "g_channel", "raw") != "raw":
        return {
            "carrier": "mixed-rational-algebraic",
            "evaluator": "approximate-float64",
            "integer_scale": None,
            "reason": ("T/F/C retain rational carriers; normalized G introduces "
                       "inverse square roots and can be algebraic"),
        }
    if rex.edge_metric_exact is not None or getattr(rex, "declares_columns", False):
        return {
            "carrier": "rational",
            "evaluator": "approximate-float64",
            "integer_scale": None,
            "reason": ("declared rational shares or relation metric use the exact "
                       "carrier but the live compiled tower evaluates float64"),
        }
    precision = channel_tower_precision(rex._boundary_ptr, rex._boundary_idx)
    return {
        "carrier": "integer" if precision["scale"] == 1 else "rational",
        "evaluator": "approximate-float64",
        "integer_scale": precision["scale"],
        "reason": precision["reason"],
    }


def channel_diagonals(rex):
    """The four channel diagonals in closed form, O(nnz), forming no edge x edge matrix.

    A declared head or share reaches the compiled tower as one coefficient per incidence,
    so this reads the column the complex declares; `rexgraph.rational_trig` is the same
    reading over the rationals.

    Every quantity the cheap character needs is a diagonal, and each one is a direct
    reading of B1 rather than something to extract from an assembled operator:

        T[e,e] = ||B1[:,e]||^2                  boundary concentration, 1 + 1/(k-1)
        G[e,e] = T[e,e]                         squaring kills the sign, so they agree
        C[e,e] = sum over supp(e) of deg(v)-1   the relation's line-graph degree
        F[e,e] = sum_j |T[e,j] - G[e,j]|        the signed/unsigned mismatch, which off
                                                the diagonal is |s_e s_j - 1|: 0 when
                                                two relations agree at a shared vertex
                                                and 2 when they disagree. So it is
                                                twice the disagreement count, and at a
                                                vertex carrying p positive and m
                                                negative entries every positive
                                                relation disagrees with all m and every
                                                negative one with all p.

    Assembling the operators to read these costs sum_v deg(v)^2 nonzeros, which one hub
    detonates: on a GO shaped complex (max degree 24256) that is 760 million entries
    and 112 s for four arrays of length nE.

    The derivation above is exact for a SIGNED PAIRWISE UNWEIGHTED complex and only
    there, because a branching column carries -1 and 1/(k-1), so an off diagonal T-G
    entry is not |s_e s_j - 1| and the disagreement count is not the F diagonal.
    `closed_form_applies` reports whether THAT derivation holds.

    Past it the disagreement count is standing in for a MAGNITUDE, and the magnitude
    accumulates at any arity. Both off diagonal channels sum over pairs that share a
    vertex and a pair contributes only there, so the sum reorders onto the vertex and
    no pair is formed:

        M[v] = sum over relations f at v of |c_f[v]|
        C[e] = sum over v in supp(e) of |c_e[v]| * (M[v] - |c_e[v]|)
        F[e] = 2 * sum over v in supp(e) of |c_e[v]| * (opposite-sign mass at v)

    Here c is the unweighted boundary coefficient. With relation weights, the F
    mass at v is sum_f |w_f| |c_f[v]| over opposite B1 orientations, and the outer
    factor is |w_e| |c_e[v]|. T/G diagonals scale by w_e^2; C stays unweighted.
    The coefficient 2 comes from 1-(-1), not from arity. This identity assumes
    distinct participants within each relation; it does not resolve the separate
    repeated-slot/self-loop conventions of the assembled readers. Repeated raw
    slots instead use the numerical incidence factors, accumulating signed
    coefficients for T and unsigned magnitudes for G before reading quadrance.
    Normalized character profiles require distinct participants.

    `core._channel_tower` evaluates that identity in float64, still O(nnz), and runs
    when the pairwise derivation does not. Rational arithmetic itself is provided by
    `exact_channel_diagonals`; an eigenfree evaluation need not be rational exact.
    Checked against that reader on witness, pairwise, branching and mixed complexes,
    including signed and zero relation weights.

    VERTEX weighting is the one case still refused: there diag(G) = sum_v w_v B1[v,e]^2
    no longer equals diag(T), so the two channels separate and the caller assembles.
    Normalized G has diagonal 1-K[e,e]/sum_f K[e,f], with no square root;
    it is also read in O(nnz). Non unit vertex weighting returns None.

    Returns {name: diagonal} for the active channels, in the canonical channel order.
    """
    supports = getattr(rex, 'relation_supports', None)
    if supports is not None:
        rex._ensure_clean()
    if supports is not None and any(len(set(support)) != len(support) for support in supports()):
        if getattr(rex, 'g_channel', 'raw') != 'raw':
            _require_distinct_channel_participants(rex)
        for attr in ('w_V', 'vertex_weights'):
            weights = getattr(rex, attr, None)
            if weights is not None and np.any(np.asarray(weights) != 1):
                return None
        from rexgraph.channel_operator import _numerical_channel_diagonals
        return _numerical_channel_diagonals(rex)
    if getattr(rex, "g_channel", "raw") != "raw":
        return _any_arity_diagonals(rex)
    if not closed_form_applies(rex):
        return _any_arity_diagonals(rex)

    nV, nE = int(rex.nV), int(rex.nE)
    B1 = rex._B1_dual
    if nE == 0:
        z = np.zeros(0, dtype=_f64)
        return {'L1_down': z, 'L_O': z.copy(), 'L_SG': z.copy(), 'L_C': z.copy()}
    vrows = np.repeat(np.arange(nV, dtype=np.int64), np.diff(B1.row_ptr))
    vals = np.asarray(B1.vals, dtype=_f64)
    dT = np.bincount(B1.col_idx, weights=vals**2, minlength=nE)
    deg = np.bincount(vrows, weights=abs(vals), minlength=nV)
    dC = np.bincount(B1.col_idx, weights=deg[vrows] - 1.0, minlength=nE)
    pos, neg = vals > 0, vals < 0
    npos = np.bincount(vrows[pos], minlength=nV)
    nneg = np.bincount(vrows[neg], minlength=nV)
    opposite = np.where(pos, nneg[vrows], np.where(neg, npos[vrows], 0))
    dF = 2.0 * np.bincount(B1.col_idx, weights=opposite.astype(_f64), minlength=nE)

    return {'L1_down': dT, 'L_O': dT.copy(), 'L_SG': dF, 'L_C': dC}


def build_sparse_character_cheap(rex):
    """Build sparse channel characters and incident star averages.

    Assemble T, G, F and C and their trace normalized sum. Read per edge chi and
    star average chi_star from operator diagonals without vertex Green solves.
    Cost depends on incidence and operator fill. compute_sparse_phi supplies the
    separate vertex Green reading.

    Return chi, chi_star, nhats, hat_names, trace_values, RL, hats and rl_diag.
    """

    nV, nE = int(rex.nV), int(rex.nE)
    # Diagonals in closed form: assembling the channel operators to read them costs
    # sum_v deg(v)^2 nonzeros (see channel_diagonals). The operators themselves are
    # only wanted by the opt in Green's path, so they are built on demand below.
    diags = channel_diagonals(rex)
    if diags is None:                       # arity or weighting: assemble and read
        diags = {n: L.diagonal()
                 for n, L in dict(build_sparse_channels(rex)).items()}
    # Every channel is a coordinate of the character and is always carried. A
    # channel with no mass reads ZERO, which is a measurement and not an absence:
    # frustration in particular vanishes exactly on a uniformly oriented complex,
    # where every vertex is a pure source or a pure sink so the signed and unsigned
    # overlaps agree at every shared vertex. Dropping it there made the character
    # three wide on those complexes and four elsewhere, so two characters were not
    # comparable, and the F column had no place to be read even as zero.
    names, traces, hat_diags_list = [], [], []
    for name in ('L1_down', 'L_O', 'L_SG', 'L_C'):
        d = diags.get(name)
        if d is None:
            continue
        tr = float(d.sum())
        if not np.isfinite(tr) or tr < 0 or np.any(d < 0):
            raise FloatingPointError("channel diagonal/trace must be finite and nonnegative")
        names.append(name)
        traces.append(tr)
        # tr == 0 means the channel carries nothing; normalising would be 0/0
        hat_diags_list.append(d / tr if tr > 0 else np.zeros_like(d))
    nhats = len(names)
    uniform = 1.0 / nhats if nhats > 0 else 0.0

    # chi(e,k) = hat_k[e,e] / RL[e,e], and RL[e,e] = sum_k hat_k[e,e]
    chi = np.full((nE, nhats), uniform, dtype=_f64)
    rl_diag = np.zeros(nE, dtype=_f64)
    if nhats > 0 and nE > 0:
        hat_diags = np.stack(hat_diags_list, axis=1)
        rl_diag = hat_diags.sum(axis=1)
        good = rl_diag > 0
        chi[good] = hat_diags[good] / rl_diag[good, None]

    # chi_star(v) = mean chi over incident edges (O(nnz))
    chi_star = np.full((nV, nhats), uniform, dtype=_f64)
    if nhats > 0:
        v2e_ptr, v2e_idx = rex._v2e
        v2e_ptr = np.asarray(v2e_ptr)
        v2e_idx = np.asarray(v2e_idx)
        chi_inc = chi[v2e_idx] if v2e_idx.size else chi[:0]
        if chi_inc.shape[0]:
            # segmented mean, not one numpy dispatch per vertex
            counts = np.diff(v2e_ptr).astype(np.int64)
            seg = np.repeat(np.arange(nV, dtype=np.int64), counts)
            tot = np.zeros((nV, nhats), dtype=_f64)
            np.add.at(tot, seg, chi_inc)
            nz = counts > 0
            chi_star[nz] = tot[nz] / counts[nz, None]

    return _CheapCharacter(
        {'chi': chi, 'chi_star': chi_star, 'nhats': nhats, 'hat_names': names,
         'trace_values': np.asarray(traces), 'rl_diag': rl_diag,
         'channel_arithmetic': channel_arithmetic_contract(rex)},
        rex, names, traces)


def _compute_sparse_phi_gpu(rex, cheap, chunk, device=None):
    """GPU resident per vertex Green's character: RL, the channel hats, and the
    Jacobi preconditioner stay on device; each vertex tile's block CG solve, hat
    applications, and the numerator/denominator reductions all run on the GPU, and
    only the (csize x nhats) phi block comes back. Identical to the CPU path."""
    import torch

    from rexgraph import scale_propagator as _spg

    # resolved, not hardcoded: on a multi GPU node a hardcoded "cuda" always lands
    # on device 0, so the other cards can never be addressed.
    from rexgraph.scale_propagator import _torch_device
    dev = _torch_device(device)
    nV, nhats = int(rex.nV), int(cheap['nhats'])
    uniform = 1.0 / nhats if nhats > 0 else 0.0
    phi = np.full((nV, nhats), uniform, dtype=_f64)

    def _to_gpu(A):
        A = A.tocsr()
        from rexgraph.compute import sparse_csr_tensor

        return sparse_csr_tensor(
            torch.as_tensor(A.indptr, dtype=torch.int64),
            torch.as_tensor(A.indices, dtype=torch.int64),
            torch.as_tensor(A.data, dtype=torch.float64), size=A.shape, device=dev)

    RLt = _to_gpu(cheap['RL'])
    hats_t = [_to_gpu(h) for h in cheap['hats']]
    dinv = np.ones_like(cheap['rl_diag'], dtype=_f64)
    np.divide(1.0, cheap['rl_diag'], out=dinv, where=np.abs(cheap['rl_diag']) > 1e-30)
    dinv_t = torch.as_tensor(dinv, dtype=torch.float64, device=dev)
    Bs = _b1_csr(rex)
    from rexgraph.fiedler import solve_block_width
    safe_width = solve_block_width(int(rex.nE), int(rex.nV))
    step = max(1, min(nV, int(chunk), int(safe_width)))
    for start in range(0, nV, step):
        stop = min(start + step, nV)
        Bc = torch.as_tensor(np.ascontiguousarray(Bs[start:stop].toarray().T),
                             dtype=torch.float64, device=dev)     # nE x csize
        Xc = _spg._block_cg_gpu(RLt, Bc, dinv_t)                  # RL^-1 B1^T
        s0 = (Bc * Xc).sum(0)
        ok = torch.abs(s0) > 1e-15
        denom = torch.where(ok, s0, torch.ones_like(s0))
        for k in range(nhats):
            num = (Xc * sparse_mm(hats_t[k], Xc)).sum(0)
            vals = torch.where(ok, num / denom, torch.full_like(num, uniform))
            phi[start:stop, k] = vals.cpu().numpy()
    kappa = 1.0 - 0.5 * np.abs(phi - cheap['chi_star']).sum(axis=1)
    return {'phi': phi, 'kappa': kappa}


def compute_sparse_phi(rex, cheap, chunk=1024, backend=None, device=None):
    """Per vertex Green's character phi and coherence kappa, given the cheap bundle.

    phi(v,k) = [b_v^T RL^-1 hat_k RL^-1 b_v] / [b_v^T RL^-1 b_v], b_v = B1[v,:], via
    Fixed tolerance numerical per vertex block CG solves RL X = B1^T (accuracy
    is scale independent). This is the O(nV·solve) global Green's refinement: the
    sandwiched two inverse numerator resists selected inversion, so it genuinely
    costs the nV solves; callers gate it to a tractable node budget and fall back to
    the O(nnz) cheap character (chi/chi*) + moment character otherwise. Returns
    {phi, kappa}. `chunk` only tiles vertices for peak memory (does not change math)."""
    nV, nE = int(rex.nV), int(rex.nE)
    nhats = int(cheap['nhats'])
    uniform = 1.0 / nhats if nhats > 0 else 0.0
    chi_star = cheap['chi_star']
    phi = np.full((nV, nhats), uniform, dtype=_f64)
    if nhats > 0 and nE > 0 and nV > 0:
        # GPU resident solve when a GPU backend is active and the work (nV*nE) clears
        # the auto gate: the agent's coherence/character hot path runs on device.
        from rexgraph import scale_propagator as _spg
        if nV * nE >= _spg._GPU_MIN_WORK and _spg._resolve_backend(backend) == "gpu":
            try:
                return _compute_sparse_phi_gpu(rex, cheap, chunk, device=device)
            except Exception:
                pass                                    # any GPU issue -> CPU tiling
        from rexgraph import compute as _compute
        # The channels apply THROUGH INCIDENCE; RL is never assembled here. An
        # assembled RL carries sum_v deg(v)^2 nonzeros: the hub blocks a Gram
        # materialises: and those blocks are what the action never forms.
        apply_rl, apply_hat = factored_channel_actions(
            rex, list(cheap['hat_names']), list(np.asarray(cheap['trace_values'], dtype=_f64)))
        Bs = _b1_csr(rex)
        rl_diag = cheap['rl_diag']
        dinv = np.ones_like(rl_diag, dtype=_f64)
        np.divide(1.0, rl_diag, out=dinv, where=np.abs(rl_diag) > 1e-30)
        from rexgraph.fiedler import solve_block_width
        safe_width = solve_block_width(nE, nV)
        step = max(1, min(nV, int(chunk), int(safe_width)))
        starts = list(range(0, nV, step))

        # Each vertex chunk is an INDEPENDENT block CG solve (its own convergence /
        # stopping, its own reductions), exactly as the serial loop below computes it.
        # Fanning the chunks across a thread pool (compute.parallel_map) is therefore a
        # pure dispatch concern: the sparse matvecs / einsums release the GIL, and each
        # chunk's phi block is bit identical to the serial version because nothing is
        # shared across chunks except read only operators. parallel_map preserves order,
        # honors get_threads() (the OMP/setup width), and no ops for a single chunk.
        def _phi_chunk(start):
            stop = min(start + step, nV)
            Bc = np.ascontiguousarray(Bs[start:stop].toarray().T)  # nE x csize
            Xc = _block_cg(apply_rl, Bc, dinv)         # RL^-1 B1^T
            s0 = np.einsum('ev,ev->v', Bc, Xc)         # b_v . x_v
            ok = np.abs(s0) > 1e-15
            denom = np.where(ok, s0, 1.0)
            block = np.full((stop - start, nhats), uniform, dtype=_f64)
            for k, name in enumerate(cheap['hat_names']):
                num = np.einsum('ev,ev->v', Xc, apply_hat(name, Xc))
                block[:, k] = np.where(ok, num / denom, uniform)
            return start, stop, block

        for start, stop, block in _compute.parallel_map(_phi_chunk, starts):
            phi[start:stop] = block
    kappa = 1.0 - 0.5 * np.abs(phi - chi_star).sum(axis=1)
    return {'phi': phi, 'kappa': kappa}


def _rl_resolvent_apply(rex, B, tol=1e-10):
    """Apply RL4⁺ to the columns of B via a single Jacobi preconditioned block CG
    solve. RL4 is full rank SPD (``build_green_cache_spd``), so RL4⁺ = RL4⁻¹ and one
    solve RL4 X = B gives X = RL4⁻¹ B exactly: the matrix free resolvent seam behind
    every ``uᵀRL⁺v`` bilinear (spectral channel score, group scores), no eigendecomposition
    and no dense nE×nE inverse. B is (nE, m); returns X (nE, m)."""
    import numpy as _np

    from rexgraph import scale_propagator as _spg
    RL = rex._rl4_sparse.tocsr()
    rl_diag = RL.diagonal()
    dinv = _np.where(_np.abs(rl_diag) > 1e-30, 1.0 / rl_diag, 1.0)
    B = _np.ascontiguousarray(B, dtype=_f64)
    if B.ndim == 1:
        B = B[:, None]
    return _spg.block_cg_solve(RL, B, dinv, tol=tol)


def pinv_quadratic_form(A, v, atol=1e-13, btol=1e-13, iter_lim=20000):
    """Numerical v.T A^+ v for a real symmetric PSD operator.

    Native LSQR computes the minimum norm solution without a spectrum or an
    explicit kernel projector. atol and btol set normal and relative residual
    thresholds; iter_lim caps iterations. An unconverged solve raises.
    Native, dense and supplied SciPy sparse matrices are accepted.
    """
    from rexgraph.core._hodge import least_squares
    from rexgraph.linear_operator import _numeric_array
    from rexgraph.native_sparse import as_native
    v = _numeric_array(v, operation="pseudoinverse quadratic form").ravel()
    if np.iscomplexobj(v):
        raise TypeError("pseudoinverse quadratic form requires a real vector")
    A = as_native(A)
    if A.shape != (v.size, v.size):
        raise ValueError("pseudoinverse quadratic form requires a matching square operator")
    x = least_squares(A, v, atol=atol, btol=btol, maxiter=iter_lim)
    with np.errstate(over='ignore', invalid='ignore'):
        result = float(v @ x)
    if not np.isfinite(result):
        raise FloatingPointError("pseudoinverse quadratic form is outside float64")
    return result


def primal_signal_character_sparse(rex, psi):
    """Energy of an edge signal across typed channels, ``E_X = psiᵀ hat_X⁺ psi``
    (returned as fractions summing to 1), eigen free via LSQR pseudoinverse quadratic
    forms on the sparse channel hats, with NO per channel eigendecomposition (removes the
    dense ``hat_eigen`` bundle). Equals ``_channels.primal_signal_character`` to ~1e-9."""
    cheap = build_sparse_character_cheap(rex)
    hats = cheap['hats']
    nhats = int(cheap['nhats'])
    psi = np.ascontiguousarray(psi, dtype=_f64).ravel()
    if nhats == 0:
        return np.zeros(0, dtype=_f64)
    e = np.array([pinv_quadratic_form(h, psi) for h in hats], dtype=_f64)
    total = float(e.sum())
    if total > 1e-30:
        return e / total
    return np.full(nhats, 1.0 / nhats, dtype=_f64)


def spectral_channel_score_sparse(rex, source, target, tol=1e-10):
    """Scale free spectral channel score ``sourceᵀ RL4⁺ target`` via one block CG
    solve. Equals the dense eigenmode sum ``Σ_j <v_j,src><v_j,tgt>/λ_j`` (over λ_j>0)
    to ~1e-9 because RL4 is full rank SPD (all λ_j>0, so RL4⁺=RL4⁻¹): no eigendecomposition."""
    src = np.ascontiguousarray(source, dtype=_f64).ravel()
    tgt = np.ascontiguousarray(target, dtype=_f64).ravel()
    x = _rl_resolvent_apply(rex, tgt, tol=tol)[:, 0]        # RL4⁻¹ target
    return float(src @ x)


def _smallest_pos_small_kernel(M, tol=1e-9):
    """Smallest positive mode of a vertex dual PSD operator.

    Connected blocks of at most 512 vertices use their full native spectra.
    Larger blocks use the numerical minimum norm inverse iteration.
    """
    import numpy as _np
    from rexgraph.native_sparse import as_native
    from rexgraph.core import _sparse
    M = as_native(M)
    n = M.shape[0]
    if n == 0 or float(_np.max(_np.abs(M.data), initial=0.0)) < 1e-30:
        return 0.0
    labels, count = _sparse.connected_components(M.dual.row_ptr, M.dual.col_idx, n)
    largest = int(_np.max(_np.bincount(labels, minlength=count), initial=0))
    from rexgraph.evaluator import eigen_dense_limit
    if largest <= 2 or largest <= min(512, eigen_dense_limit()):
        from rexgraph.core._linalg import symmetric_sparse_spectrum
        w = symmetric_sparse_spectrum(M)
    else:
        return _smallest_positive_eig(M, n, tol=tol)
    pos = _np.sort(_np.asarray(w, dtype=_f64))
    pos = pos[pos > tol]
    return float(pos.min()) if pos.size else 0.0


def channel_spectral_gaps(rex):
    """Smallest positive mode of each trace normalized channel, keyed by its name.

    T and raw G use the nonzero spectral correspondence of B.T B and B B.T,
    avoiding the kernel on relations. C, F and normalized G use their channel
    operators directly. Small support components use full native spectra;
    larger operators use an approximate minimum norm inverse iteration.
    """
    chan = dict(build_sparse_channels(rex))
    B1 = _b1_csr(rex)
    nE = int(rex.nE)
    g_raw = getattr(rex, 'g_channel', 'raw') == 'raw'
    gaps = {}
    for name in ('L1_down', 'L_O', 'L_SG', 'L_C'):
        if name not in chan:
            continue
        L = chan[name]
        tr = float(L.diagonal().sum())
        if tr < 1e-15:
            continue
        if name == 'L1_down':
            lam = _smallest_pos_small_kernel((B1 @ B1.T).tocsr())          # vertex dual L0
        elif name == 'L_O' and g_raw:
            aB1 = abs(B1); lam = _smallest_pos_small_kernel((aB1 @ aB1.T).tocsr())
        else:
            lam = _smallest_positive_eig(L.tocsr(), nE)                    # F, C, normalized G
        gaps[name] = lam / tr        # lambda_2 of the trace normalized hat
    return gaps


def per_channel_mixing_times_sparse(rex):
    """mu_X = ln(nE) / lambda_2(hat_X), in hat_names order.

    Gap estimates come from channel_spectral_gaps. For nE > 1 a channel with
    zero mass reports 0; a channel with mass but no positive gap reports inf.
    nE <= 1 reports inf. Gap approximations for large operators carry into these times.
    """
    import numpy as _np
    cheap = build_sparse_character_cheap(rex)
    names = cheap['hat_names']
    nhats = int(cheap['nhats'])
    nE = int(rex.nE)
    times = _np.empty(nhats, dtype=_f64)
    if nE <= 1:
        times[:] = _np.inf
        return times
    log_nE = float(_np.log(nE))
    gaps = channel_spectral_gaps(rex)
    traces = _np.asarray(cheap['trace_values'], dtype=_f64)
    for k, nm in enumerate(names):
        lam2 = gaps.get(nm, 0.0)
        if k < traces.shape[0] and traces[k] <= 1e-15:
            times[k] = 0.0            # no operator, so nothing to equilibrate
        else:
            times[k] = (log_nE / lam2) if lam2 > 1e-15 else _np.inf
    return times


# Full component spectra below this bound; minimum norm inverse iteration above it.
_MIXING_DENSE_MAX = 512


def _smallest_positive_eig(H, nE, tol=1e-9):
    """Smallest positive mode of a sparse symmetric PSD operator.

    Small operators use their full native component spectra. Larger operators
    use inverse iteration with native minimum norm LSQR actions. LSQR checks
    its residual; the outer iteration returns an approximate Rayleigh quotient
    and can converge slowly when the lowest positive modes are close. A zero
    operator returns 0.0. tol sets the cutoff for positive modes.
    """
    import numpy as _np
    from rexgraph.native_sparse import as_native
    from rexgraph.core._hodge import least_squares
    H = as_native(H)
    if H.shape != (nE, nE):
        raise ValueError("spectral gap requires a matching square operator")
    if float(_np.max(_np.abs(H.data), initial=0.0)) < 1e-30:
        return 0.0
    from rexgraph.evaluator import eigen_dense_limit
    if nE <= min(_MIXING_DENSE_MAX, eigen_dense_limit()):
        from rexgraph.evaluator import require_small_dense_eigen
        require_small_dense_eigen(
            "channel spectral gap", nE, max_dimension=_MIXING_DENSE_MAX)
        from rexgraph.core._linalg import symmetric_sparse_spectrum
        w = symmetric_sparse_spectrum(H)
        pos = w[w > tol]
        return float(pos.min()) if pos.size else 0.0

    # inverse power on H^+ (min norm LSQR = pseudoinverse, kernel robust).
    rs = _np.random.RandomState(0)
    x = rs.standard_normal(nE)
    x /= _np.linalg.norm(x)
    lam_prev = 0.0
    for _ in range(80):
        y = least_squares(H, x, tol=1e-9, maxiter=2000)
        ny = float(_np.linalg.norm(y))
        if ny < 1e-300:
            return 0.0
        x = y / ny
        lam = 1.0 / ny
        if abs(lam - lam_prev) < 1e-8 * max(lam, 1e-30):
            break
        lam_prev = lam
    Hx = H.apply(x)
    denom = float(x @ x)
    lam2 = float((x @ Hx) / denom) if denom > 0 else 0.0
    return lam2 if lam2 > tol else 0.0


def void_character_sparse(rex, Bvoid):
    """Per void typed channel character, the eigen free twin of
    ``_void.void_character_all``. Each void basis vector (column of Bvoid) gets its
    channel character = the primal signal character E_X = v^T hat_X^+ v (fractions
    summing to 1) via LSQR pseudoinverse quadratic forms on the sparse channel hats -
    no dense RL / hats, no eigendecomposition. Returns f64[n_voids, nhats]."""
    import numpy as _np
    cheap = build_sparse_character_cheap(rex)
    hats = cheap['hats']
    nhats = int(cheap['nhats'])
    sparse_voids = hasattr(Bvoid, "getcol") and hasattr(Bvoid, "tocsr")
    if sparse_voids:
        Bvoid = Bvoid.tocsc()
        n_voids = int(Bvoid.shape[1])
    else:
        Bvoid = _np.ascontiguousarray(Bvoid, dtype=_f64)
        n_voids = Bvoid.shape[1] if Bvoid.ndim == 2 else 0
    out = _np.zeros((n_voids, nhats), dtype=_f64)
    uniform = 1.0 / nhats if nhats > 0 else 0.0
    for i in range(n_voids):
        # One void column is the signal being classified. There is no reason to
        # materialize all nE x n_voids columns just to consume them one at a time.
        v = (Bvoid.getcol(i).toarray().ravel() if sparse_voids else Bvoid[:, i])
        e = _np.array([pinv_quadratic_form(h, v) for h in hats], dtype=_f64)
        tot = float(e.sum())
        out[i] = (e / tot) if tot > 1e-30 else _np.full(nhats, uniform, dtype=_f64)
    return out


def spectral_propagate_sparse(rex, source, target, tol=1e-10):
    """Scale free spectral propagation, the eigen free twin of
    ``_query.spectral_propagate``. RL4 is full rank SPD, so RL4⁺ = RL4⁻¹ and one
    block CG solve gives ``prop = RL4⁻¹ source``; the score, per channel typed
    scores, and energy are then sparse matvecs / inner products: no rl_eigen, no
    dense RL. Returns {score, typed_scores, energy, coverage}.

        score        = <RL4⁻¹ source, target> / (||source|| ||target||)
        typed_scores = <source, hat_k @ RL4⁻¹ source>  per channel
        energy       = <source, RL4 @ source>
        coverage     = ||P_range source|| / ||source|| = 1.0 for full-rank SPD RL4
                       (all modes active); 0 for a zero source. The dense path's
                       n_covered/n_modes reduces to this exactly when every mode is
                       positive, which it is for RL4.
    """
    import numpy as _np
    cheap = build_sparse_character_cheap(rex)
    hats = cheap['hats']
    nhats = int(cheap['nhats'])
    RL = cheap['RL'].tocsr()
    src = _np.ascontiguousarray(source, dtype=_f64).ravel()
    tgt = _np.ascontiguousarray(target, dtype=_f64).ravel()

    prop = _rl_resolvent_apply(rex, src, tol=tol)[:, 0]      # RL4⁻¹ source
    ns = float(_np.sqrt(src @ src))
    nt = float(_np.sqrt(tgt @ tgt))
    score = float(prop @ tgt) / (ns * nt) if ns > 1e-15 and nt > 1e-15 else 0.0

    typed = _np.zeros(nhats, dtype=_f64)
    for k in range(nhats):
        typed[k] = float(src @ (hats[k] @ prop))

    energy = float(src @ (RL @ src))
    coverage = 1.0 if ns > 1e-15 else 0.0
    return {'score': score, 'typed_scores': typed,
            'energy': energy, 'coverage': coverage}


def compute_sparse_character(rex, chunk=1024):
    """Full {chi, phi, chi_star, kappa, nhats, hat_names, RL, hats}: the cheap
    O(nnz) character plus the per vertex Green's phi/kappa. Kept for callers that
    want the complete bundle in one shot; the pipeline uses the split accessors
    (cheap by default, phi on demand) to stay O(nnz) at scale."""
    cheap = build_sparse_character_cheap(rex)
    ph = compute_sparse_phi(rex, cheap, chunk)
    # Add the solved readings to the cheap bundle rather than copying it into a plain
    # dict. RL and hats stay LAZY: reading them here to hand them back would assemble
    # the nE x nE operator for every caller, including the ones that only wanted chi,
    # phi and kappa.
    dict.__setitem__(cheap, 'phi', ph['phi'])
    dict.__setitem__(cheap, 'kappa', ph['kappa'])
    return cheap
