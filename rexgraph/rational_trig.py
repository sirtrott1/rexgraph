"""
rexgraph.rational_trig: the spread tower, exactly.

Distance and angle are the wrong primitives for a complex built out of integer
boundary operators. Both need a square root, and the square root is the only place
the whole pipeline leaves exact arithmetic: the characters are rational (a star
character reads ``220/969``), the Hodge percentages are rational, the boundary is
integer, and then a cosine takes two norms and everything downstream is a float with
an error bar.

Their squared counterparts have no such problem.

    quadrance   Q(v)      = <v, v>                         replaces length
    spread      s(u, v)   = 1 - <u,v>^2 / (Q(u) Q(v))      replaces angle

Both are rational functions of the inner products, so a complex whose entries are
rational has rational quadrances and rational spreads. Nothing is approximated and
nothing needs a tolerance.

Two identities make this more than a change of units.

**The spread is the Gram block over its own diagonal.** For vectors ``u, v`` with
Gram ``G = [[<u,u>, <u,v>], [<u,v>, <v,v>]]``::

    s(u, v) = det(G) / (G[0,0] * G[1,1])

so the inner ranking (the diagonal, the quadrances) and the outer ranking (the block,
the pairwise inner products) enter as a determinant over a product of diagonal
entries. That generalises to any number of vectors without modification, which is
what makes it usable at a branching relation: ``k`` incident fibers give one ``k x k``
block and one exact rational number for their joint alignment.

**The degeneracy locus is the cycle space.** ``det(Gram) = 0`` exactly when the
vectors are linearly dependent, so for a set of boundary columns it vanishes exactly
when that set carries a cycle, and ``k - rank(Gram_k)`` is the number of independent
cycles the set carries. Betti numbers are reachable without computing Betti numbers,
and the same rank test applied to a candidate face column answers whether attaching
it would fill a hole before it is attached.

Cost. Exact arithmetic over Fractions is cubic in the number of vectors with
coefficient growth on top, so the exact entry points are for small blocks: a face
candidate, a star, a handful of channels. The float entry points carry no square root
either and are the ones to use across a whole complex. `exact_character` is the
exception: it reads diagonals only and is exact at any size.

## Which path, and why

There is one place a square root is genuinely unavoidable, and everything else is
arranged so it does not spread.

**Normalization is not the main path.** The G channel has two forms. `raw` is
``K = |B1|^T W |B1|``, the co-incidence Gramian, whose entries are integers on a pairwise
complex and rationals once a relation branches, because the boundary share is ``1/(k-1)``.
`normalized` is ``I - D^{-1/2} K D^{-1/2}``, which is degree-comparable and therefore
useful when you are comparing cells of very different degree, and which takes a square
root of the degree, so it lands on the float tower and stays there. `raw` is the
constructor default for that reason, and `exact_channel_diagonals` reports
``(None, [])`` on a normalized complex rather than approximating a rational character
that does not exist. Normalization is a choice you make when comparability matters more
than exactness, not a default you inherit.

**Angles never need the square root.** This is what the spread tower buys. An angle costs
an arccosine of a ratio of norms, and each norm costs a square root; the squared
counterparts do not, and carry the same ordering::

    quadrance   Q(v)     = <v, v>                      instead of length
    spread      s(u, v)  = 1 - <u,v>^2 / (Q(u) Q(v))   instead of angle, and s = sin^2

so ``cos^2 = 1 - s`` exactly. `geometry.relation_quadrance` and `geometry.relation_spread`
are the whole-complex readings; `projection` places cells in the plane through the
half-angle parametrisation ``cos = (1-t^2)/(1+t^2)``, ``sin = 2t/(1+t^2)``, which gives a
rational point on the unit circle for every rational ``t``, so a rendering has exact
coordinates and exact angles between them. Nothing in that path calls sqrt, sin, cos or
atan2.

**Exactness needs an exact SOURCE, not an exact reading.** This is the failure mode to
watch, because it looks like rigour. ``Fraction(x)`` on a float is exact for the double
it holds, which is a different number from the one meant whenever the value is not
binary-exact, and ``1/(k-1)`` is not for most arities. Taking an exact Gram over a
densified float ``B1`` returned ``432691404877902290367942354447019 /
324518553658426726783156020576256`` at ``k = 4`` where the answer is ``4/3``. So the exact
entry points rebuild columns from the boundary CSR (`faces._exact_b1_block`) and read
weights through `RexGraph.edge_metric_exact`, never off the assembled operators.

**What the float tower is still for.** Anything spectral, anything needing a degree
normalization, anything at a scale where cubic exact elimination is the wrong trade, and
the propagators, which are Chebyshev or CG by construction. The two towers are meant to
agree: `exact_character` against `structural_character` is a test, not a fallback, and a
disagreement is a defect in one of them by definition.

**Where a tolerance is still wrong.** A predicate is not a magnitude. The chain condition
holds at zero, not near it, so `chain_valid` and `FlowComplex.chain_residual` are
adjudicated over the rationals and the float magnitude is reported separately when the
scale is what you want. A float column of a k-ary relation sums to zero for k = 3..12 by
rounding luck and does not at 483 arities below 4000, so a tolerance there was standing in
for the arithmetic rather than for the mathematics.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

__all__ = [
    "bareiss_determinant",
    "exact_channel_diagonals",
    "quadrance", "spread", "gram", "gram_determinant", "gram_rank",
    "carries_cycle", "independent_cycles", "spread_matrix",
    "cross_spread", "rank_increment", "rational_reconstruct",
    "exact_character", "exact_star_character", "CHANNEL_ORDER",
    "MAX_RECOVERABLE_DENOMINATOR",
]


def _exact(values):
    """A row of values as exact Fractions.

    Integers and Fractions pass through. A float is taken at face value via
    `Fraction(float)`, which is exact for the binary value it holds; callers wanting
    a small denominator should pass integers or Fractions in the first place.
    """
    out = []
    for x in values:
        if isinstance(x, Fraction):
            out.append(x)
        elif isinstance(x, (int, np.integer)):
            out.append(Fraction(int(x)))
        else:
            out.append(Fraction(float(x)))
    return out


def quadrance(v, *, exact: bool = False):
    """``Q(v) = <v, v>``. The squared length, without the square root.

    Rational whenever the entries are, which is what makes it the primitive rather
    than the length.
    """
    if exact:
        e = _exact(np.asarray(v).ravel())
        return sum(x * x for x in e)
    a = np.asarray(v, dtype=np.float64).ravel()
    return float(a @ a)


def spread(u, v, *, exact: bool = False):
    """``s(u, v) = 1 - <u,v>^2 / (Q(u) Q(v))``. The squared sine of the angle.

    0 when the vectors are parallel, 1 when perpendicular, and rational throughout.
    Returns ``None`` when either vector is zero, where no angle is defined; that is
    an absence rather than a value and callers must not read it as 0.
    """
    if exact:
        a, b = _exact(np.asarray(u).ravel()), _exact(np.asarray(v).ravel())
        ip = sum(x * y for x, y in zip(a, b, strict=True))
        qa = sum(x * x for x in a)
        qb = sum(y * y for y in b)
        if qa == 0 or qb == 0:
            return None
        return Fraction(1) - (ip * ip) / (qa * qb)
    a = np.asarray(u, dtype=np.float64).ravel()
    b = np.asarray(v, dtype=np.float64).ravel()
    qa, qb = float(a @ a), float(b @ b)
    if qa == 0.0 or qb == 0.0:
        return None
    ip = float(a @ b)
    return 1.0 - (ip * ip) / (qa * qb)


def gram(vectors, *, exact: bool = False):
    """The Gram block ``G[i,j] = <v_i, v_j>``.

    The diagonal is the inner ranking (quadrances) and the off-diagonal the outer
    ranking (pairwise inner products). Everything else here is a function of this
    block.
    """
    if exact:
        rows = [_exact(np.asarray(v).ravel()) for v in vectors]
        k = len(rows)
        return [[sum(rows[i][t] * rows[j][t] for t in range(len(rows[0])))
                 for j in range(k)] for i in range(k)]
    M = np.asarray([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    return M @ M.T


def bareiss_determinant(A_in):
    """Exact determinant by fraction-free (Bareiss) elimination.

    Every division is exact in the ring the entries came from, so integer input
    stays integer all the way through and never grows a denominator. Ordinary
    elimination divides by the pivot at each step, which turns integers into
    rationals whose denominators compound down the matrix; that is the cost this
    avoids. Intermediates are bounded by minors of the input rather than by the
    product of the pivots.
    """
    k = len(A_in)
    if k == 0:
        return Fraction(1)
    A = [row[:] for row in A_in]
    prev = Fraction(1)
    sign = 1
    for i in range(k - 1):
        if A[i][i] == 0:
            pivot = next((r for r in range(i + 1, k) if A[r][i] != 0), None)
            if pivot is None:
                return Fraction(0)
            A[i], A[pivot] = A[pivot], A[i]
            sign = -sign
        for r in range(i + 1, k):
            for c in range(i + 1, k):
                A[r][c] = (A[r][c] * A[i][i] - A[r][i] * A[i][c]) / prev
        prev = A[i][i]
    return sign * A[k - 1][k - 1]


def gram_determinant(vectors, *, exact: bool = True):
    """``det`` of the Gram block, by fraction-free elimination when exact.

    Zero exactly when the vectors are linearly dependent. For boundary columns that
    is exactly when the set carries a cycle, which is why this is a homology test
    that never forms a Laplacian.
    """
    G = gram(vectors, exact=exact)
    if not exact:
        return float(np.linalg.det(np.asarray(G)))
    return bareiss_determinant(G)


def gram_rank(vectors, *, exact: bool = True) -> int:
    """Rank of the Gram block, which equals the rank of the vectors themselves.

    Exact by elimination over Fractions, so there is no tolerance to choose and no
    singular value to threshold.
    """
    G = gram(vectors, exact=exact)
    if not exact:
        return int(np.linalg.matrix_rank(np.asarray(G)))
    k = len(G)
    A = [row[:] for row in G]
    rank = 0
    for col in range(k):
        pivot = next((r for r in range(rank, k) if A[r][col] != 0), None)
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        for r in range(k):
            if r != rank and A[r][col] != 0:
                f = A[r][col] / A[rank][col]
                for c in range(k):
                    A[r][c] -= f * A[rank][c]
        rank += 1
    return rank


def carries_cycle(columns, *, exact: bool = True) -> bool:
    """Whether a set of boundary columns is dependent, so carries a cycle.

    ``det(Gram) = 0`` and nothing else is computed: no Laplacian, no spectrum, no
    spanning tree.
    """
    return gram_determinant(columns, exact=exact) == 0


def independent_cycles(columns, *, exact: bool = True) -> int:
    """``k - rank(Gram_k)``: how many independent cycles a set of columns carries.

    The rank deficiency of the Gram block IS the cycle dimension of that set, which
    is the local form of what Betti counts globally.
    """
    return len(list(columns)) - gram_rank(columns, exact=exact)


def rank_increment(existing, candidate, *, exact: bool = True) -> int:
    """Whether adding `candidate` raises the rank of `existing`: 1 or 0.

    Applied to face columns this answers whether attaching a face would fill a hole,
    BEFORE attaching it. A column that raises the rank converts one harmonic class to
    curl; a column that does not is redundant, adding a face and killing nothing.
    """
    before = gram_rank(list(existing), exact=exact) if len(list(existing)) else 0
    after = gram_rank([*list(existing), candidate], exact=exact)
    return int(after - before)


def spread_matrix(vectors, *, exact: bool = False):
    """Pairwise spreads. Zero on the diagonal, since a vector is parallel to itself.

    A zero vector has no spread against anything; those entries are NaN in the float
    form and None in the exact form, so an absent angle never reads as a right one.
    """
    vs = list(vectors)
    n = len(vs)
    if exact:
        return [[Fraction(0) if i == j else spread(vs[i], vs[j], exact=True)
                 for j in range(n)] for i in range(n)]
    M = np.asarray([np.asarray(v, dtype=np.float64).ravel() for v in vs])
    G = M @ M.T
    q = np.diag(G).copy()
    denom = np.outer(q, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - (G * G) / denom
    out[denom == 0] = np.nan
    np.fill_diagonal(out, 0.0)
    return out


def cross_spread(T, G):
    """The spread difference of a signed/unsigned Gram pair sharing a diagonal.

    ``T = B^T B`` and ``G = |B|^T |B|`` have identical diagonals at every grade,
    because squaring an entry discards its sign. So the two spreads share a
    denominator and differ only in the determinant::

        s_T - s_G = (det G - det T) / prod(diag)

    which isolates the orientation content as a single rational number. It is not a
    restatement of ``T - G``: that is the off-diagonal mismatch, this is what the
    mismatch does to the block's degeneracy.

    Returns ``(s_T, s_G, difference, shared_denominator)``, or ``None`` for the
    spreads when the shared denominator is zero.
    """
    Te = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row]
          for row in T]
    Ge = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row]
          for row in G]
    diag_T = [Te[i][i] for i in range(len(Te))]
    diag_G = [Ge[i][i] for i in range(len(Ge))]
    if diag_T != diag_G:
        raise ValueError(
            "the signed and unsigned Gram blocks do not share a diagonal, so they "
            "do not share a spread denominator; this pair is not a T/G pair")
    denom = Fraction(1)
    for x in diag_T:
        denom *= x
    if denom == 0:
        return None, None, None, denom
    det_T = _det_of(Te)
    det_G = _det_of(Ge)
    s_T = Fraction(1) - det_T / denom
    s_G = Fraction(1) - det_G / denom
    return s_T, s_G, (det_G - det_T) / denom, denom


def _det_of(A_in):
    """Exact determinant of a square Fraction matrix."""
    return bareiss_determinant(A_in)


#: A rational p/q is uniquely determined by a double approximation only while
#: q < sqrt(1 / (2 * eps)). Past that, some fraction with a large denominator matches
#: any float to machine precision and recovering one proves nothing.
MAX_RECOVERABLE_DENOMINATOR = int((1.0 / (2.0 * np.finfo(np.float64).eps)) ** 0.5)


def rational_reconstruct(values, *, max_denominator: int | None = None):
    """Recover the exact rational a float array approximates, or refuse.

    The characters ARE rational: they come from integer boundary operators through
    rational operations, and on a small complex they read as small fractions:
    `1/4` on a triangle, `37/135` on K4, `220/969` on a five-edge path. Stored as
    float64 the fraction is still there and can be recovered.

    It does not survive size. The denominator grows with the complex, and past roughly
    a few dozen cells it exceeds what a double can pin down: a random 20-vertex
    complex needs a denominator near 1e9 to match its stored float, which is not the
    true value but merely a fraction close to that float. Continued fractions will
    always produce such a thing, so a reconstruction that does not check is a
    reconstruction that always "succeeds".

    The check is the classical bound: a rational is uniquely determined by a double
    only while its denominator is below `sqrt(1/(2 eps))`, about 4.7e7. Above that
    this returns None, and the exact value has to be computed in exact arithmetic from
    the boundary operators rather than read back out of a float.

    Returns a list of Fractions in the input's shape, or None.
    """
    bound = int(max_denominator or MAX_RECOVERABLE_DENOMINATOR)
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.ravel()
    out = []
    tolerance = 8.0 * np.finfo(np.float64).eps
    for x in flat:
        f = Fraction(float(x)).limit_denominator(bound)
        scale = max(abs(float(x)), 1.0)
        if abs(float(f) - float(x)) > tolerance * scale:
            return None                       # not recoverable at this precision
        if f.denominator > MAX_RECOVERABLE_DENOMINATOR:
            return None                       # matched the float, not the value
        out.append(f)
    if arr.ndim <= 1:
        return out
    rows, cols = arr.shape[0], int(np.prod(arr.shape[1:]))
    return [out[r * cols:(r + 1) * cols] for r in range(rows)]


#: the relational Laplacian's channels, in the order the character stacks them
CHANNEL_ORDER = ("L1_down", "L_O", "L_SG", "L_C")


def exact_channel_diagonals(rex):
    """The four channel diagonals as exact Fractions, built from the boundary structure.

    Every channel is a polynomial in the entries of B1, and those entries are `-1` and
    `1/(k-1)` at arity k, so each diagonal is exactly rational::

        T[e,e] = w_e^2 * sum_v c_e[v]^2         = w_e^2 * (1 + 1/(k-1))
        G[e,e] = w_e^2 * sum_v |c_e[v]|^2       = T[e,e], since squaring kills the
                                                  sign. G is T's unsigned twin and
                                                  carries the same metric; C does not,
                                                  co-participation being topological
        F[e,e] = sum_{f != e} |T[e,f] - G[e,f]|   the signed/unsigned mismatch, which
                                                  lives entirely off-diagonal
        C[e,e] = sum_{f != e} G[e,f]              the share-weighted overlap row sum

    Read from the boundary CSR rather than the assembled float64 channels. Recovering
    these from the float would put the whole rational tower on the approximation tower
    for no reason, and at arity k the values are not integers: `1 + 1/(k-1)` is in
    `(1, 3/2]` for every k >= 3, so anything that rounds collapses every branching arity
    onto the same value and zeroes the mismatch that F is made of.

    Returns `(diagonals, names)` with `diagonals` a dict name -> list of Fractions, or
    `(None, [])` when the complex is not exactly representable. The normalized G channel
    is the one case that is not: `I - D^-1/2 K D^-1/2` takes a square root, so a complex
    carrying it has no rational character and says so rather than approximating one.
    """
    if getattr(rex, "g_channel", "raw") != "raw":
        return None, []                     # normalized L_O takes a sqrt: not rational
    rex._ensure_clean()
    nE = int(rex.nE)
    if nE == 0:
        return None, []

    bp = np.asarray(rex._boundary_ptr)
    bi = np.asarray(rex._boundary_idx)
    supports = [[int(v) for v in bi[bp[e]:bp[e + 1]]] for e in range(nE)]

    # the RATIONAL reader, not `edge_metric`: that one is float64 by construction, so
    # taking it here would put the exact tower on the exact value of a double
    metric = getattr(rex, "edge_metric_exact", None)
    w = list(metric) if metric is not None else [Fraction(1)] * nE

    # the exact B1 column: the head is distinguished at -1 and the rest share
    # 1/(k-1), which is what makes the column sum to zero at every arity k >= 2.
    #
    # A WITNESS (k = 1) is the exception and does not follow the head rule: the
    # construction emits (+1), so that L0 u = u, and there is no second vertex for
    # the zero-sum condition to constrain. Reconstructing it as (-1) flipped the
    # sign of every T off-diagonal it took part in, and since F is built from
    # T - G off-diagonal it was the only channel that moved: on a 1-ary/2-ary
    # complex F read [0,2,4,2] against the definition's [2,2,4,4], and on one
    # carrying arities 1..4 it read [0,0,0,0] against [6,2,2,2]. T, G and C were
    # untouched, the diagonal squaring the sign away and C taking absolute values.
    cols = []
    for support in supports:
        k = len(support)
        if k == 0:
            cols.append({})
            continue
        if k == 1:
            cols.append({support[0]: Fraction(1)})
            continue
        share = Fraction(1, k - 1)
        col = {support[0]: Fraction(-1)}
        for v in support[1:]:
            col[v] = col.get(v, Fraction(0)) + share
        cols.append(col)

    incident = {}
    for e, col in enumerate(cols):
        for v in col:
            incident.setdefault(v, []).append(e)

    # G is T's unsigned TWIN and carries the same per-relation metric: `overlap_gramian`
    # is already weighted, so leaving it unweighted here made diag(T) != diag(G) at any
    # w != 1 and broke the identity F is defined by. C stays unweighted on purpose, since
    # co-participation is a topological fact about which relations meet.
    T = [w[e] * w[e] * sum((c * c for c in cols[e].values()), Fraction(0))
         for e in range(nE)]
    G = [w[e] * w[e] * sum((abs(c) * abs(c) for c in cols[e].values()), Fraction(0))
         for e in range(nE)]

    F = [Fraction(0)] * nE
    C = [Fraction(0)] * nE
    for e in range(nE):
        neighbours = {f for v in cols[e] for f in incident[v] if f != e}
        for f in neighbours:
            shared = cols[e].keys() & cols[f].keys()
            t = w[e] * w[f] * sum((cols[e][v] * cols[f][v] for v in shared), Fraction(0))
            g = w[e] * w[f] * sum((abs(cols[e][v]) * abs(cols[f][v]) for v in shared),
                                  Fraction(0))
            F[e] += abs(t - g)
            C[e] += sum((abs(cols[e][v]) * abs(cols[f][v]) for v in shared), Fraction(0))

    # Every channel is reported, including one summing to zero. A channel with no
    # mass is a MEASUREMENT and not an absence: frustration vanishes exactly on a
    # uniformly oriented complex, where every vertex is a pure source or a pure sink
    # so the signed and unsigned overlaps agree at every shared vertex. Dropping it
    # there disagreed with the float bundle, which now keeps it too, and made two
    # characters of different widths not comparable.
    diagonals, names = {}, []
    for name, values in (("L1_down", T), ("L_O", G), ("L_SG", F), ("L_C", C)):
        names.append(name)
        diagonals[name] = values
    return diagonals, names


def exact_character(rex):
    """The per-edge structural character as exact Fractions, computed not recovered.

    The character is a ratio of DIAGONALS::

        hat_k    = L_k / trace(L_k)
        chi[e,k] = hat_k[e,e] / RL[e,e]        where  RL = sum_k hat_k

    and every channel's diagonal is a polynomial in B1's entries, which are `-1` and
    `1/(k-1)`, so the whole thing is a ratio of rationals. No solve, no eigenvalue and
    no square root enters at any point. The diagonals come from `exact_channel_diagonals`,
    built from the boundary structure: they are NOT integers once any relation branches,
    and reading them back off the float channels would lose exactly the arity content the
    character is there to carry.

    That distinction matters against `rational_reconstruct`, which tries to recover a
    rational from a float that has already lost it and refuses past ~4.7e7. This
    cannot lose it, because it never converts to float. The denominators stay whatever
    the complex genuinely produces at any size.

    `vertex_character` (phi) is NOT of this form. It is a Green's function and needs
    solves, so an exact phi is a different and much more expensive problem.

    Returns `(chi, channel_names)` with `chi` a list of rows of Fractions. Every
    channel the complex carries gets a column, including one with no mass, which
    reads exactly zero: the row is still on the simplex and the columns keep fixed
    positions, so two characters are comparable.
    """
    diagonals, names = exact_channel_diagonals(rex)
    if not names:
        return None, []
    names = [n for n in CHANNEL_ORDER if n in diagonals]
    diags = [diagonals[n] for n in names]
    traces = [sum(d, Fraction(0)) for d in diags]

    n_edges = len(diags[0])
    uniform = Fraction(1, len(names))
    chi = []
    for e in range(n_edges):
        # A channel with no mass reads exactly zero rather than being normalised by
        # its own zero trace. Frustration does this on any uniformly oriented complex,
        # where every vertex is a pure source or a pure sink and there is no
        # orientation conflict to measure. The remaining channels still sum to one, so
        # the row stays on the simplex and the column keeps its position.
        hats = [(diags[k][e] / traces[k]) if traces[k] != 0 else Fraction(0)
                for k in range(len(names))]
        rl = sum(hats, Fraction(0))
        chi.append([uniform] * len(names) if rl == 0 else [h / rl for h in hats])
    return chi, names


def exact_star_character(rex):
    """`chi*(v)`, the mean of `chi(e)` over the edges at `v`, as exact Fractions.

    The fiber coordinate of the bundle, and the input to `spread_similarity`'s cosine
    factor. Exact for the same reason `exact_character` is: it is an average of
    rationals.
    """
    chi, names = exact_character(rex)
    if chi is None:
        return None, []
    v2e_ptr, v2e_idx = rex._v2e
    ptr = np.asarray(v2e_ptr)
    idx = np.asarray(v2e_idx)
    uniform = Fraction(1, len(names))
    out = []
    for v in range(int(rex.nV)):
        lo, hi = int(ptr[v]), int(ptr[v + 1])
        if hi <= lo:
            out.append([uniform] * len(names))
            continue
        incident = [chi[int(e)] for e in idx[lo:hi]]
        count = Fraction(len(incident))
        out.append([sum((row[k] for row in incident), Fraction(0)) / count
                    for k in range(len(names))])
    return out, names
