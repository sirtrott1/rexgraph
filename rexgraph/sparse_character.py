"""rexgraph.sparse_character: scale-free character / coherence via a sparse RL.

The dense character path (``build_all_laplacians`` -> ``build_RL`` ->
``compute_chi`` / ``build_character_bundle``) forms dense nE x nE channel
Laplacians, RL, and hats, and was therefore gated to ``nE <= eigen_dense_limit``.
But every RCF channel operator is sparse, so chi / phi / kappa have no intrinsic
size ceiling: the dense ceiling was an implementation choice, not the math.

This module assembles the four channels as scipy CSR, sums the trace-normalized
hats into a sparse RL, and computes exactly what the dense path computes:

    chi(e,k)   = hat_k[e,e] / RL[e,e]                          (diagonals only)
    phi(v,k)   = x_v^T hat_k x_v / (b_v^T x_v),  x_v = RL^-1 b_v,  b_v = B1[v,:]^T
    chi*(v,k)  = mean_{e in star(v)} chi(e,k)
    kappa(v)   = 1 - 0.5 * ||phi(v) - chi*(v)||_1

RL3/RL4 is full-rank SPD (see ``_relational.build_green_cache_spd``), so phi uses
a single sparse LU factorization of RL reused across chunked per-vertex solves -
no dense pseudoinverse, no eigendecomposition, and no nE x nE materialization.
Verified identical to the dense path on small graphs (``test`` via the graph
properties); this is the path that removes the arbitrary size limit on character.

The four channels (matching the dense builders exactly):
  * L1_down = B1^T B1                                    (_laplacians.build_L1_down)
  * L_O     = raw |B1|^T|B1|, or normalized I - D^-1/2 K D^-1/2   (g_channel)
  * L_SG    = diag(sum|K_off|) - K_off,  K_s = S diag(w) S^T      (_frustration)
  * L_C     = D_L - A_L over the selected co-participation reading (`rex.c_channel`):
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


def build_sparse_channels(rex):
    """The four RCF channel Laplacians as scipy CSR: ``[(name, L_csr), ...]``.

    Each equals the corresponding dense ``build_all_laplacians`` channel operator
    (verified elementwise on small graphs).
    """
    import scipy.sparse as sp

    from rexgraph.core._laplacians import build_L1_down_sparse

    _nV, nE = int(rex.nV), int(rex.nE)

    channels = []

    # The metric enters as a symmetric per-relation scale, T = W B1^T B1 W and G its
    # unsigned twin. Weighting is per relation and not per sqrt, so a rational weight
    # keeps both channels rational; sqrt(w) lives in the NORMALIZED G alone. C is the
    # line-graph degree and is deliberately unweighted: it is pure co-participation.
    _w = rex.edge_metric
    _D = sp.diags(_w) if _w is not None else None

    def _metric(X):
        return X if _D is None else (_D @ X @ _D).tocsr()

    # 1. L1_down = B1^T B1
    channels.append(('L1_down', _metric(build_L1_down_sparse(rex._B1_dual).tocsr())))

    # 2. L_O: the selected G-channel operator (raw Gramian or normalized L_O)
    K = rex.overlap_gramian_sparse.tocsr()               # raw |B1|^T|B1|
    if rex.g_channel == 'raw':
        L_O = K
    else:
        d = np.asarray(K.sum(axis=1)).ravel()            # row sums (incl. diagonal)
        inv_sqrt = np.zeros(nE, dtype=_f64)
        nz = d > 1e-12
        inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        Dis = sp.diags(inv_sqrt)
        S = (Dis @ K @ Dis).tocsr()
        L_O = (sp.identity(nE, format='csr', dtype=_f64) - S).tocsr()
    channels.append(('L_O', L_O))

    # 3. F = T - G frustration (INTEGER, Def 3.3): off-diag T-G (0 same-orientation,
    #    -2 opposite at a shared vertex), diagonal = Σ|off-diag|. Pure integer.
    T = _metric(build_L1_down_sparse(rex._B1_dual).tocsr())   # W B1^T B1 W (signed Gram)
    Foff = (T - K).tocsr()                               # T - G (G = K here)
    Foff.setdiag(0.0); Foff.eliminate_zeros()
    d_f = np.asarray(abs(Foff).sum(axis=1)).ravel()
    channels.append(('L_SG', (Foff + sp.diags(d_f)).tocsr()))

    # 4. C co-participation (INTEGER, Def 3.4): Laplacian of the WEIGHTED line graph
    #    whose adjacency is the shared-vertex counts K_off = off-diagonal of |B1|^T|B1|.
    #    off-diagonal = -(# shared vertices), diagonal = sum of shared-vertex counts (the
    #    MULTIPLICITY row-sum) so it is a proper zero-row-sum PSD Laplacian at ANY arity.
    #    The old binarized diagonal (# distinct adjacent edges) matched the multiplicity
    #    only when every edge pair shares <=1 vertex (simple graphs); for branching
    #    hyperedges / parallel edges / self-loops it broke row-sum=0 and made L_C
    #    indefinite (dragging RL4 and the moment character non-PSD). "Unweighted" in
    #    Def 3.4 means orientation/edge-weight independent, NOT binary: the shared-vertex
    #    count is pure topology and is carried here (identical to the old form on any
    #    complex where no two edges share >1 vertex).
    # C reads the UNWEIGHTED overlap: co-participation is a topological fact about which
    # relations meet, not a geometric one about how far apart they are. Weighting it too
    # shifts every channel (measured 0.286 flat against the canonical 0.351/0.351/
    # 0.172/0.126 on a triangle with one relation at weight 5).
    # the reading the CHARACTER is set to. The two are independent in both directions,
    # so this is a choice of question: share is how much of each relation meets and
    # conserves; count is how many vertices they meet at and is structural. The flow layer
    # pins the share whatever this says, because moving signal is a different job.
    Kc = (rex.overlap_count_sparse if getattr(rex, "c_channel", "share") == "count"
          else rex.overlap_share_sparse).tocsr()
    K_off = (Kc - sp.diags(Kc.diagonal())).tocsr()       # G_off = shared-vertex counts
    deg_L = np.asarray(K_off.sum(axis=1)).ravel()        # weighted line-graph degree = Sum shared counts
    L_C = (sp.diags(deg_L) - K_off).tocsr()              # D_L - G_off, weighted line-graph Laplacian
    channels.append(('L_C', L_C))   # kept even at zero mass; see build_sparse_rl

    return channels


def _trace_normalized(L, tr, nE):
    """`L / tr` as CSR, or a zero matrix when the channel carries no mass.

    A channel is kept whatever its trace, so this is the one place that decides what
    a zero-trace hat looks like. Normalising by its own zero trace would be 0/0;
    the hat is the zero operator, which is what "this channel measures nothing here"
    means, and it contributes nothing to RL.
    """
    import scipy.sparse as sp
    if tr > 1e-15:
        return (L * (1.0 / tr)).tocsr()
    return sp.csr_matrix((nE, nE), dtype=_f64)


def build_sparse_rl(rex):
    """Assemble the sparse RL from trace-normalized hats. Returns
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


def _block_cg(apply_A, B, dinv, tol=1e-10, maxit=1000):
    """Jacobi-preconditioned block conjugate gradient: solve A X = B for all
    columns of B at once (A SPD), where ``apply_A(P)`` returns A @ P. Matrix-free -
    no factorization, no fill-in. Here ``apply_A`` is the *factored* RL operator
    (each channel through B1/|B1|, O(nE) per matvec), so it never touches the
    hub clique blocks. RL is trace-normalized and well-conditioned, so CG converges
    in a few dozen iters. Returns X with ||A X - B|| / ||B|| < tol per column."""
    X = np.zeros_like(B)
    R = B - apply_A(X)
    Z = dinv[:, None] * R
    P = Z.copy()
    rz = (R * Z).sum(0)
    bnorm = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    # Each column is an independent CG: alpha and beta are per-column scalars and the
    # only thing shared is the operator. So a converged column can LEAVE the block,
    # and it must, because the loop below stops on the worst column and every column
    # still in the block pays that column's iteration count. Measured on two ontology
    # slices, 596 columns took 117 iterations and 1000 took 286: the width was driving
    # the depth.
    active = np.arange(B.shape[1])
    for _ in range(maxit):
        AP = apply_A(P)
        # A is PSD, so curv = P^T A P is >= 0 exactly, and curv == 0 means this
        # direction has no curvature: the column is converged or P has landed in
        # ker(A). Clamping the denominator up to 1e-300 instead turns that into
        # rz/1e-300 = +-inf, and X += alpha*P then makes the whole column NaN. On a
        # 66-component ontology slice that was 348 of 385 relations.
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
    return X


# NOTE: the edge-primacy MATRIX-FREE RL operator (`build_factored_operator`) lives
# in rexgraph._experimental: it is bit-identical to the assembled channels but was
# overhead-bound versus a single assembled `RL @ P` matmul at moderate nE, so the
# default path below uses the assembled matvec. See _experimental.py for details.


class _CheapCharacter(dict):
    """The cheap character bundle, with the channel OPERATORS built on first read.

    chi, chi_star and rl_diag are diagonals and cost O(nnz). RL and hats are the
    edge x edge operators, wanted only by the opt-in per-vertex Green's path, and they
    are the expensive part (sum_v deg(v)^2 nonzeros). Resolving them on access keeps
    the always-affordable layer actually affordable.
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
        self._filled = True
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

    def __getitem__(self, key):
        if key in self._LAZY and not self._filled:
            self._fill()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in self._LAZY and not self._filled:
            self._fill()
        return dict.get(self, key, default)

    def __iter__(self):
        # also takes dict(...) / {**...} off CPython's dict-to-dict fast path
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
        from rexgraph.core._sparse import to_scipy_csr
        B1 = to_scipy_csr(rex._B1_dual)
    except Exception:
        return False
    if int(rex.nE) == 0:
        return False
    Bc = B1.tocsc()
    if _np.any(_np.diff(Bc.indptr) != 2):            # every relation binary
        return False
    if not _np.all(_np.abs(_np.asarray(Bc.data)) == 1.0):
        return False
    w = getattr(rex, 'w_E', None)                    # no edge weighting
    if w is not None and not _np.allclose(_np.asarray(w, dtype=float), 1.0):
        return False
    for attr in ('w_V', 'vertex_weights'):
        wv = getattr(rex, attr, None)
        if wv is not None and not _np.allclose(_np.asarray(wv, dtype=float), 1.0):
            return False
    return True


def channel_diagonals_integer(rex):
    """The four diagonals as EXACT int64 numerators over one common denominator.

    No float enters at any point. The share 1/(k-1) becomes the integer L/(k-1) for
    L = lcm(k-1), so a pairwise complex works at L = 1 and needs no denominator at all.
    C and F multiply two shares, so the whole tower lands over L^2 and that single
    integer is returned beside it.

    Returns ({name: int64 numerators}, scale) or (None, None) when the scale would
    leave int64, which is when the approximation tower is the honest answer and
    `channel_diagonals` gives it.
    """
    import numpy as _np
    from collections import defaultdict

    rex._ensure_clean()
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
    import numpy as _np
    from math import gcd

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
    # the largest diagonal is bounded by the co-participation mass, and the incidence
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


def _any_arity_diagonals(rex):
    """The tower past the pairwise derivation, by accumulating at the vertex.

    Refuses only what genuinely separates the channels: a vertex weighting, which makes
    diag(G) differ from diag(T), and the normalized G channel, which takes a square
    root. Both then have to be assembled and read.
    """
    import numpy as _np
    if str(getattr(rex, "g_channel", "raw")) != "raw":
        return None
    for attr in ("w_V", "vertex_weights"):
        wv = getattr(rex, attr, None)
        if wv is not None and not _np.allclose(_np.asarray(wv, dtype=float), 1.0):
            return None
    bp = getattr(rex, "_boundary_ptr", None)
    bi = getattr(rex, "_boundary_idx", None)
    if bp is None or bi is None:
        return None
    from rexgraph import compute
    w = getattr(rex, "w_E", None)
    # Prefer the parallel lane. `_tower_cpu`'s `threads=1` is a signature default rather
    # than a decision, and dispatch was reaching it first because "cpu" registers before
    # "openmp", so the tower ran serial. Bit-identical either way, the serial lane
    # staying as the reference. Measured on this path, serial against the width
    # `_tower_width` picks:
    #
    #     nnz         nnz/nV     serial      now
    #     10.5M            7   379.9 ms   96.1 ms
    #     10.5M           52   318.4 ms   42.8 ms
    #      1.4M           23    23.4 ms    8.3 ms
    #
    # Preferred only when the caller has expressed nothing: an explicit
    # `set_default_backend` still wins, which is how the hip lane is reached. Hip is not
    # the default because it is SLOWER, measured once the transpose went parallel too:
    # 208.0 ms against 104.4 at 10.5M over 1.5M vertices, 199.6 against 43.8 over 200k,
    # 22.3 against 10.2 at 1.7M. It pays the same CPU transpose and then gathers over
    # variable-degree vertices, which suits the device poorly. Against the serial path it
    # won; the CPU overtook it.
    T, G, F, C = compute.dispatch("channel_tower",
                                  _np.asarray(bp), _np.asarray(bi), int(rex.nV),
                                  None if w is None else _np.asarray(w, float),
                                  prefer=compute.get_default_backend() or "openmp")
    return {'L1_down': T, 'L_O': G, 'L_SG': F, 'L_C': C}


#### the lanes. Same registry as every other device-specialised kernel here, so a new
#### architecture is a register_op call and nothing in this module moves.
def _tower_cpu(bp, bi, nV, w, threads=1, transposed=None):
    from rexgraph.core._channel_tower import channel_diagonals_any_arity
    return channel_diagonals_any_arity(bp, bi, nV, w, threads, transposed)


def _tower_width() -> int:
    """The width the tower runs at when the caller has not set one.

    PHYSICAL cores, not logical. Both passes stream the incidence and scatter into
    vertex-indexed arrays, so the kernel is memory-bound, and SMT siblings share L1 and
    the load/store units: they add contention without adding memory parallelism. Swept
    at three densities, median of 5, the best width is 14 on a 16c/32t machine and the
    curve is flat from 10 to 24, while all 32 logical costs 10 to 15%:

        nnz/nV      12      14      16      32
             7    96.7    91.8    95.6   107.4  ms
            52    47.6    44.8    49.4    55.3

    Physical cores lands inside that flat region on every case without anyone picking a
    number, and it moves correctly on a machine with a different SMT ratio or none.
    An explicit `compute.set_threads` still wins, which is how a measured per-host
    optimum gets used where someone has actually measured one.
    """
    from rexgraph import compute as _c
    explicit = _c.get_threads()
    if explicit:
        return int(explicit)
    from rexgraph.hardware import physical_cores
    return int(physical_cores())


def _tower_openmp(bp, bi, nV, w, transposed=None):
    """The accumulation is over VERTICES after transposing the incidence, so each
    thread owns what it writes and no atomic is needed. Measured 9.2x at 12 threads on
    12M nonzeros, bit-identical to the serial path at every width.

    The transpose it depends on is split the same way, so a cold call is parallel
    throughout. Passing `transposed` still skips it entirely, which is worth it when the
    same complex is read more than once."""
    return _tower_cpu(bp, bi, nV, w, _tower_width(), transposed)


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


def channel_diagonals(rex):
    """The four channel diagonals in closed form, O(nnz), forming no edge x edge matrix.

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
    detonates: on a GO-shaped complex (max degree 24256) that is 760 million entries
    and 112 s for four arrays of length nE.

    The derivation above is exact for a SIGNED PAIRWISE UNWEIGHTED complex and only
    there, because a branching column carries -1 and 1/(k-1), so an off-diagonal T-G
    entry is not |s_e s_j - 1| and the disagreement count is not the F diagonal.
    `closed_form_applies` reports whether THAT derivation holds.

    Past it the disagreement count is standing in for a MAGNITUDE, and the magnitude
    accumulates at any arity. Both off-diagonal channels sum over pairs that share a
    vertex and a pair contributes only there, so the sum reorders onto the vertex and
    no pair is formed:

        M[v] = sum over relations f at v of |c_f[v]|
        C[e] = sum over v in supp(e) of |c_e[v]| * (M[v] - |c_e[v]|)
        F[e] = 2 * sum over v in supp(e) of |c_e[v]| * (opposite-sign mass at v)

    `core._channel_tower` carries that, still O(nnz), still exact, and it is what runs
    when the pairwise derivation does not. Checked against the exact rational tower on
    witness, pairwise, branching and mixed complexes.

    VERTEX weighting is the one case still refused: there diag(G) = sum_v w_v B1[v,e]^2
    no longer equals diag(T), so the two channels separate and the caller assembles.
    The normalized G channel is refused for the same reason it has no rational
    character, a square root. Returns None in both.

    Returns {name: diagonal} for the active channels, in the canonical channel order.
    """
    from rexgraph.core._sparse import to_scipy_csr
    if not closed_form_applies(rex):
        return _any_arity_diagonals(rex)

    nV, nE = int(rex.nV), int(rex.nE)
    B1 = to_scipy_csr(rex._B1_dual).astype(_f64)
    if nE == 0:
        z = np.zeros(0, dtype=_f64)
        return {'L1_down': z, 'L_O': z.copy(), 'L_SG': z.copy(), 'L_C': z.copy()}
    Bc = B1.tocsc()
    dT = np.asarray(Bc.multiply(Bc).sum(axis=0)).ravel()          # T, and G with it

    absB = abs(Bc)
    deg = np.asarray(absB.sum(axis=1)).ravel()                    # weighted vertex degree
    rows, cols = absB.nonzero()
    dC = np.bincount(cols, weights=(deg[rows] - 1.0), minlength=nE)

    Br = B1.tocsr()                                               # by vertex
    vrows = np.repeat(np.arange(nV, dtype=np.int64), np.diff(Br.indptr))
    vals = np.asarray(Br.data, dtype=_f64)
    pos, neg = vals > 0, vals < 0
    npos = np.bincount(vrows[pos], minlength=nV)
    nneg = np.bincount(vrows[neg], minlength=nV)
    opposite = np.where(pos, nneg[vrows], np.where(neg, npos[vrows], 0))
    dF = 2.0 * np.bincount(Br.indices, weights=opposite.astype(_f64), minlength=nE)

    return {'L1_down': dT, 'L_O': dT.copy(), 'L_SG': dF, 'L_C': dC}


def build_sparse_character_cheap(rex):
    """The O(nnz) character: assemble the doc-exact channels (T,G,F=T-G,C) and the
    trace-normalized RL once, then the per-edge character chi and star-average chi*
    from DIAGONALS only: no per-vertex solves, no eigendecomposition. This is the
    always-affordable layer; the per-vertex Green's phi/kappa (nV solves) is a
    separate, opt-in refinement in ``compute_sparse_phi``.

    Returns {chi, chi_star, nhats, hat_names, trace_values, RL, hats, rl_diag}."""

    nV, nE = int(rex.nV), int(rex.nE)
    # Diagonals in closed form: assembling the channel operators to read them costs
    # sum_v deg(v)^2 nonzeros (see channel_diagonals). The operators themselves are
    # only wanted by the opt-in Green's path, so they are built on demand below.
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
        names.append(name)
        traces.append(tr)
        # tr == 0 means the channel carries nothing; normalising would be 0/0
        hat_diags_list.append(d / tr if tr > 1e-15 else np.zeros_like(d))
    nhats = len(names)
    uniform = 1.0 / nhats if nhats > 0 else 0.0

    # chi(e,k) = hat_k[e,e] / RL[e,e], and RL[e,e] = sum_k hat_k[e,e]
    chi = np.full((nE, nhats), uniform, dtype=_f64)
    rl_diag = np.zeros(nE, dtype=_f64)
    if nhats > 0 and nE > 0:
        hat_diags = np.stack(hat_diags_list, axis=1)
        rl_diag = hat_diags.sum(axis=1)
        good = rl_diag > 1e-15
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
         'trace_values': np.asarray(traces), 'rl_diag': rl_diag},
        rex, names, traces)


def _compute_sparse_phi_gpu(rex, cheap, chunk, device=None):
    """GPU-resident per-vertex Green's character: RL, the channel hats, and the
    Jacobi preconditioner stay on-device; each vertex tile's block-CG solve, hat
    applications, and the numerator/denominator reductions all run on the GPU, and
    only the (csize x nhats) phi block comes back. Identical to the CPU path."""
    import torch

    from rexgraph import scale_propagator as _spg

    # resolved, not hardcoded: on a multi-GPU node a hardcoded "cuda" always lands
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
    dinv = np.where(np.abs(cheap['rl_diag']) > 1e-30, 1.0 / cheap['rl_diag'], 1.0)
    dinv_t = torch.as_tensor(dinv, dtype=torch.float64, device=dev)
    Bs = _b1_csr(rex)
    step = max(1, min(nV, int(chunk)))
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
    """Per-vertex Green's character phi and coherence kappa, given the cheap bundle.

    phi(v,k) = [b_v^T RL^-1 hat_k RL^-1 b_v] / [b_v^T RL^-1 b_v], b_v = B1[v,:], via
    EXACT per-vertex block-CG solves RL X = B1^T (CG to a fixed tolerance; accuracy
    is scale-independent). This is the O(nV·solve) global-Green's refinement - the
    sandwiched two-inverse numerator resists selected inversion, so it genuinely
    costs the nV solves; callers gate it to a tractable node budget and fall back to
    the O(nnz) cheap character (chi/chi*) + moment character otherwise. Returns
    {phi, kappa}. `chunk` only tiles vertices for peak memory (does not change math)."""
    nV, nE = int(rex.nV), int(rex.nE)
    nhats = int(cheap['nhats'])
    uniform = 1.0 / nhats if nhats > 0 else 0.0
    chi_star = cheap['chi_star']
    phi = np.full((nV, nhats), uniform, dtype=_f64)
    if nhats > 0 and nE > 0 and nV > 0:
        # GPU-resident solve when a GPU backend is active and the work (nV*nE) clears
        # the auto-gate - the agent's coherence/character hot path runs on-device.
        from rexgraph import scale_propagator as _spg
        if nV * nE >= _spg._GPU_MIN_WORK and _spg._resolve_backend(backend) == "gpu":
            try:
                return _compute_sparse_phi_gpu(rex, cheap, chunk, device=device)
            except Exception:
                pass                                    # any GPU issue -> CPU tiling
        from rexgraph import compute as _compute
        RLc = cheap['RL'].tocsr()
        hat_by_name = dict(zip(cheap['hat_names'], cheap['hats'], strict=False))
        apply_rl = lambda P: RLc @ P
        apply_hat = lambda name, P: hat_by_name[name] @ P
        Bs = _b1_csr(rex)
        rl_diag = cheap['rl_diag']
        dinv = np.where(np.abs(rl_diag) > 1e-30, 1.0 / rl_diag, 1.0)  # Jacobi precond
        step = max(1, min(nV, int(chunk)))
        starts = list(range(0, nV, step))

        # Each vertex chunk is an INDEPENDENT block-CG solve (its own convergence /
        # stopping, its own reductions), exactly as the serial loop below computes it.
        # Fanning the chunks across a thread pool (compute.parallel_map) is therefore a
        # pure dispatch concern: the sparse matvecs / einsums release the GIL, and each
        # chunk's phi block is bit-identical to the serial version because nothing is
        # shared across chunks except read-only operators. parallel_map preserves order,
        # honors get_threads() (the OMP/setup width), and no-ops for a single chunk.
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
    """Apply RL4⁺ to the columns of B via a single Jacobi-preconditioned block-CG
    solve. RL4 is full-rank SPD (``build_green_cache_spd``), so RL4⁺ = RL4⁻¹ and one
    solve RL4 X = B gives X = RL4⁻¹ B exactly - the matrix-free resolvent seam behind
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
    """``vᵀ A⁺ v`` for a symmetric PSD sparse ``A`` (possibly SINGULAR), matrix-free
    via LSQR: ``x = A⁺ v`` is the minimum-norm least-squares solution, so LSQR projects
    off ``ker(A)`` exactly (unlike CG/MINRES, which diverge on a kernel component of v)
    and ``vᵀ A⁺ v = vᵀ x``. Equals the dense eigenmode pseudoinverse
    ``Σ_{λ_j>0} <u_j,v>²/λ_j`` to machine precision - no eigendecomposition, no explicit
    kernel projection. This is the reusable seam behind every ``vᵀ hat⁺ v`` energy."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    v = np.ascontiguousarray(v, dtype=_f64).ravel()
    A = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=_f64))
    x = sla.lsqr(A, v, atol=atol, btol=btol, iter_lim=iter_lim)[0]
    return float(v @ x)


def primal_signal_character_sparse(rex, psi):
    """Energy of an edge signal across typed channels, ``E_X = psiᵀ hat_X⁺ psi``
    (returned as fractions summing to 1), eigen-free via LSQR pseudoinverse quadratic
    forms on the sparse channel hats, with NO per-channel eigendecomposition (removes the
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
    """Scale-free spectral channel score ``sourceᵀ RL4⁺ target`` via one block-CG
    solve. Equals the dense eigenmode sum ``Σ_j <v_j,src><v_j,tgt>/λ_j`` (over λ_j>0)
    to ~1e-9 because RL4 is full-rank SPD (all λ_j>0, so RL4⁺=RL4⁻¹) - no eigendecomposition."""
    src = np.ascontiguousarray(source, dtype=_f64).ravel()
    tgt = np.ascontiguousarray(target, dtype=_f64).ravel()
    x = _rl_resolvent_apply(rex, tgt, tol=tol)[:, 0]        # RL4⁻¹ target
    return float(src @ x)


def _smallest_pos_small_kernel(M, tol=1e-9):
    """Smallest strictly-positive eigenvalue of a sparse symmetric PSD M with a SMALL,
    known kernel (the vertex-dual Laplacians here have kernel = beta_0 components).
    Dense eigvalsh when affordable (exact); smallest-algebraic Lanczos otherwise: fast
    and exact precisely because only a few near-zero modes sit below lambda_2."""
    import numpy as _np
    import scipy.sparse.linalg as sla
    M = M.tocsr()
    n = M.shape[0]
    if n == 0 or float(abs(M).sum()) < 1e-30:
        return 0.0
    if n <= 512:
        w = _np.linalg.eigvalsh(np.asarray(M.toarray(), dtype=_f64))
    else:
        try:
            w = sla.eigsh(M, k=min(n - 2, 16), which='SA', return_eigenvectors=False)
        except Exception:
            w = _np.linalg.eigvalsh(np.asarray(M.toarray(), dtype=_f64))
    pos = _np.sort(_np.asarray(w, dtype=_f64))
    pos = pos[pos > tol]
    return float(pos.min()) if pos.size else 0.0


def channel_spectral_gaps(rex):
    """Exact-where-possible per-channel spectral gap lambda_2 (smallest positive
    eigenvalue of each trace-normalized hat) - a METRIC, dict keyed by channel name.

    T (L1_down) and raw G (L_O) use the A^TA<->AA^T transpose duality: lambda_2 of the
    huge edge-space Gram equals lambda_2 of the tiny nV x nV VERTEX-dual Laplacian
    (kernel = beta_0), computed exactly and cheaply - the topological zeros collapse into
    the small vertex space instead of a ~nE-dimensional numerical cluster. C (L_C, the
    line-graph Laplacian) has a small line-graph-component kernel and F (L_SG, a
    difference of Grams with no transpose dual) fall back to the kernel-robust
    _smallest_positive_eig. Normalized G (I - D^-1/2 K D^-1/2) is not a Gram, so it also
    uses the general path. This is the exact spectral-gap metric; it is NOT the
    edge-centric relaxation object (see the moment tower / relaxation accessors)."""
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
        gaps[name] = lam / tr        # lambda_2 of the trace-normalized hat
    return gaps


def per_channel_mixing_times_sparse(rex):
    """Per-channel mixing times mu_X = ln(nE) / lambda_2(hat_X), the spectral-gap METRIC
    per channel. lambda_2 comes from channel_spectral_gaps (T/G exact via the transpose
    duality on the tiny vertex-dual Laplacian; C/F kernel-robust), so the mixing time is
    exact for the topology/overlap channels and no longer the pathological huge-kernel
    inverse-power on those. Returns f64[nhats] in hat_names order; inf where there is no
    gap and nE<=1.

    A channel carrying NO MASS reports 0, not inf. Its operator is the zero matrix,
    so `e^{-tL} = I` and every state is already stationary at t = 0: there is nothing
    to equilibrate, which is a mixing time of zero rather than a process that never
    settles. Frustration does this on any uniformly oriented complex. Reporting inf
    there would also poison anything derived across channels, the anisotropy first.
    A channel that HAS mass but no gap is the other case and still reads inf."""
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


# Dense eigvalsh is exact and cheap up to this hat size; above it the typed hats carry
# a large near-zero kernel (e.g. dim ker(B1^T B1) = nE - rank(B1) ~ nE), which defeats
# both dense (O(nE^3)) and smallest-algebraic Lanczos (cannot get past the kernel), so
# lambda_2 comes from a kernel-robust inverse-power iteration instead.
_MIXING_DENSE_MAX = 512


def _smallest_positive_eig(H, nE, tol=1e-9):
    """Smallest strictly-positive eigenvalue (spectral gap lambda_2) of a sparse
    symmetric PSD matrix H whose kernel may be LARGE and is not known here.

    - nE <= _MIXING_DENSE_MAX: exact dense eigvalsh (covers all realistic test / agent
      graphs; parity with the dense hat_eigen path is exact).
    - larger: INVERSE-POWER iteration on the pseudoinverse. Each step applies H^+ via a
      min-norm LSQR solve, which projects off ker(H) exactly (unlike a sigma=0 shift-
      invert, whose singular factorization is slow and fragile, or smallest-algebraic
      Lanczos, which cannot resolve past a huge kernel). x converges to the smallest
      POSITIVE eigenvector and the closing Rayleigh quotient gives lambda_2. This is an
      APPROXIMATE spectral gap at scale (a few % when lambda_2 / lambda_3 are close) -
      the documented scale-free surrogate for this diagnostic. Returns 0.0 if H is 0."""
    import numpy as _np
    import scipy.sparse.linalg as sla

    if float(abs(H).sum()) < 1e-30:
        return 0.0
    H = H.tocsr()
    if nE <= _MIXING_DENSE_MAX:
        w = _np.linalg.eigvalsh(np.asarray(H.toarray(), dtype=_f64))
        pos = w[w > tol]
        return float(pos.min()) if pos.size else 0.0

    # inverse-power on H^+ (min-norm LSQR = pseudoinverse, kernel-robust).
    rs = _np.random.RandomState(0)
    x = rs.standard_normal(nE)
    x /= _np.linalg.norm(x)
    lam_prev = 0.0
    for _ in range(80):
        y = sla.lsqr(H, x, atol=1e-9, btol=1e-9, iter_lim=2000)[0]
        ny = float(_np.linalg.norm(y))
        if ny < 1e-300:
            return 0.0
        x = y / ny
        lam = 1.0 / ny
        if abs(lam - lam_prev) < 1e-8 * max(lam, 1e-30):
            break
        lam_prev = lam
    Hx = H @ x
    denom = float(x @ x)
    lam2 = float((x @ Hx) / denom) if denom > 0 else 0.0
    return lam2 if lam2 > tol else 0.0


def void_character_sparse(rex, Bvoid):
    """Per-void typed-channel character, the eigen-free twin of
    ``_void.void_character_all``. Each void basis vector (column of Bvoid) gets its
    channel character = the primal signal character E_X = v^T hat_X^+ v (fractions
    summing to 1) via LSQR pseudoinverse quadratic forms on the sparse channel hats -
    no dense RL / hats, no eigendecomposition. Returns f64[n_voids, nhats]."""
    import numpy as _np
    cheap = build_sparse_character_cheap(rex)
    hats = cheap['hats']
    nhats = int(cheap['nhats'])
    if hasattr(Bvoid, 'toarray'):
        Bvoid = Bvoid.toarray()
    Bvoid = _np.ascontiguousarray(Bvoid, dtype=_f64)
    n_voids = Bvoid.shape[1] if Bvoid.ndim == 2 else 0
    out = _np.zeros((n_voids, nhats), dtype=_f64)
    uniform = 1.0 / nhats if nhats > 0 else 0.0
    for i in range(n_voids):
        v = Bvoid[:, i]
        e = _np.array([pinv_quadratic_form(h, v) for h in hats], dtype=_f64)
        tot = float(e.sum())
        out[i] = (e / tot) if tot > 1e-30 else _np.full(nhats, uniform, dtype=_f64)
    return out


def spectral_propagate_sparse(rex, source, target, tol=1e-10):
    """Scale-free spectral propagation, the eigen-free twin of
    ``_query.spectral_propagate``. RL4 is full-rank SPD, so RL4⁺ = RL4⁻¹ and one
    block-CG solve gives ``prop = RL4⁻¹ source``; the score, per-channel typed
    scores, and energy are then sparse matvecs / inner products - no rl_eigen, no
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
    """Full {chi, phi, chi_star, kappa, nhats, hat_names, RL, hats} - the cheap
    O(nnz) character plus the per-vertex Green's phi/kappa. Kept for callers that
    want the complete bundle in one shot; the pipeline uses the split accessors
    (cheap by default, phi on demand) to stay O(nnz) at scale."""
    cheap = build_sparse_character_cheap(rex)
    ph = compute_sparse_phi(rex, cheap, chunk)
    return {
        'chi': cheap['chi'], 'phi': ph['phi'], 'chi_star': cheap['chi_star'],
        'kappa': ph['kappa'], 'nhats': cheap['nhats'],
        'hat_names': cheap['hat_names'], 'trace_values': cheap['trace_values'],
        'RL': cheap['RL'], 'hats': cheap['hats'],
    }
