"""The mass tower: one integer sequence that determines every trace and every moment.

`L_k = B_k^T B_k + B_{k+1} B_{k+1}^T` and `tr(X^T X) = ||X||_F^2`, so

    tr(L_k)  =  ||B_k||_F^2 + ||B_{k+1}||_F^2

with `B_0` and `B_{G+1}` absent. The whole trace tower, and therefore every moment
between grades, is fixed by the single sequence `||B_k||_F^2`. Verified exact on the
solid octahedron, the square pyramid and the truncated icosahedron.

The moments fall out of it:

    d tr(0 -> 1)  =  ||B_2||^2
    d tr(1 -> 2)  =  ||B_3||^2 - ||B_1||^2
    d tr(2 -> 3)  =  -||B_2||^2

The k=0 case is why the grade-0-to-1 moment reads exactly zero on a face-free complex:
`||B_2||^2 = 0` when there are no faces, whatever the arity.

**Why the mass and not the normalised character.** `||B_k||_F^2` is a sum over stored
entries, so it is EXTENSIVE: additive over disjoint components, exactly. The
trace-normalised character is not, and that failure of additivity is precisely the global
coupling that makes a character-derived position context-dependent. The mass tower carries
the structural content with none of it.

Each mass is also a geometric reading. `||B_1||_F^2 = sum_e Q(e)`, the total quadrance, so
the gradient mass IS the sum of the relations' squared lengths.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

__all__ = ["boundary_mass", "mass_tower", "trace_tower", "moments",
           "tower_law", "incidence_degrees", "closure_at", "manifold_state",
           "surface_identity", "apd", "semantic_closure",
           "graded_delta", "channel_delta"]


def _boundaries(rex):
    from rexgraph.graded_boundary import graded_boundaries_from_rex
    return [sp.csr_matrix(b) for b in graded_boundaries_from_rex(rex)]


def boundary_mass(rex, grade: int, *, exact: bool = True):
    """`||B_grade||_F^2`, the sum of squared entries. Grades count from 1.

    Grade 1 is computed from the boundary STRUCTURE rather than the assembled float
    `B1`, because the share `1/(k-1)` has no exact double for most arities and reading
    the stored value would give the exact mass of a set of doubles. It equals
    `sum_e Q(e)`, the total quadrance.

    Higher grades read the stored coefficients. A face solved by
    `faces.solve_face_column` has its denominators cleared, so those are integers and a
    double holds them exactly; a hand-supplied non-integer coefficient is taken at face
    value and `exact` should be read as False by the caller in that case.
    """
    grade = int(grade)
    if grade < 1:
        raise ValueError(f"grades count from 1, got {grade}")
    if grade == 1:
        from rexgraph.geometry import relation_quadrance
        total = sum((relation_quadrance(rex, e) for e in range(int(rex.nE))),
                    Fraction(0))
        return total if exact else float(total)
    bounds = _boundaries(rex)
    if grade > len(bounds):
        return Fraction(0) if exact else 0.0
    B = bounds[grade - 1]
    if not exact:
        return float(B.multiply(B).sum())
    return sum((Fraction(float(v)) ** 2 for v in B.data), Fraction(0))


def mass_tower(rex, *, exact: bool = True) -> list:
    """`[||B_1||^2, ||B_2||^2, ...]`, the sequence everything else follows from."""
    return [boundary_mass(rex, k + 1, exact=exact)
            for k in range(len(_boundaries(rex)))]


def trace_tower(rex) -> list:
    """`[tr(L_0), tr(L_1), ...]`, one per grade."""
    from rexgraph.graded_boundary import graded_laplacians
    return [float(sp.csr_matrix(L).diagonal().sum())
            for L in graded_laplacians(_boundaries(rex))]


def moments(rex) -> list:
    """The per-grade increments of the trace tower.

    The moment half of the action/moment pair; `scale_propagator.action_moment` takes
    the tower itself when both are wanted.
    """
    t = trace_tower(rex)
    return [t[i + 1] - t[i] for i in range(len(t) - 1)]


def tower_law(rex) -> dict:
    """Check `tr(L_k) = ||B_k||^2 + ||B_{k+1}||^2` at every grade.

    An identity, so a mismatch is a defect rather than a tolerance. Reported rather than
    asserted, because the caller may be looking at a hand-built boundary.
    """
    masses = [float(m) for m in mass_tower(rex, exact=False)]
    traces = trace_tower(rex)
    predicted = []
    for k in range(len(traces)):
        down = masses[k - 1] if k >= 1 else 0.0
        up = masses[k] if k < len(masses) else 0.0
        predicted.append(down + up)
    worst = max((abs(a - b) for a, b in zip(traces, predicted, strict=True)),
                default=0.0)
    return {"mass": masses, "trace": traces, "predicted": predicted,
            "holds": bool(worst < 1e-9), "residual": float(worst),
            "moments": [traces[i + 1] - traces[i] for i in range(len(traces) - 1)]}


def incidence_degrees(rex, grade: int) -> np.ndarray:
    """How many `grade`-cells each `(grade-1)`-cell bounds.

    At grade 2 this is the number of faces on each edge, which is what the closed
    2-manifold condition is actually about.
    """
    bounds = _boundaries(rex)
    grade = int(grade)
    if grade < 1 or grade > len(bounds):
        return np.zeros(0, dtype=int)
    B = bounds[grade - 1]
    return np.asarray((abs(B) > 0).sum(axis=1)).ravel().astype(int)


def surface_identity(rex) -> dict:
    """`2/d + 2/k = 1 + chi/E` on a closed surface, exactly and rationally.

    With `d = 2E/V` the mean vertex degree and `k = 2E/F` the mean face size, this is
    Euler's relation divided through by `E`: `V = 2E/d`, `F = 2E/k`, and `V - E + F =
    chi` gives it in one line. Every term is rational, so it holds as an equality and not
    to a tolerance.

    **What it separates.** Homology cannot tell a tetrahedron from an octahedron: both
    are `chi = 2`, `betti = [1,0,0,0]`. The pair `(k, d)` can. Tetrahedron `(3,3)`,
    octahedron `(3,4)`, cube `(4,3)`, icosahedron `(3,5)`, dodecahedron `(5,3)`. Same
    topology, different geometry, and the identity is what ties the two together.

    **The continuum limit is the topological ideal.** As the complex refines, `E` grows
    and `chi/E -> 0`, leaving

        2/d + 2/k = 1

    whose integer solutions are exactly `(k,d) = (3,6), (4,4), (6,3)`: the three regular
    tilings of the plane. So the k-gon structure decides WHICH ideal a refinement
    approaches, and the deviation from it at any finite stage is exactly `chi/E`.

    **The rate is `chi/E`.** Not an error term and not asymptotic: at every finite stage
    the deviation IS that rational number. How fast it falls is how fast `E` grows under
    refinement, which is set by the k-gon structure being subdivided. Measured on a
    triangulated sphere, `6 - d = 6chi/V = 12/V` exactly at every level.

    That is the sense in which a discrete manifold is not an approximation of a smooth
    one. Each discrete stage satisfies the identity exactly, carrying its own `chi/E`;
    the Riemannian case is the degenerate one where that term has vanished.

    **Arity-general form.** The two 2s are the arity of a relation and the closure
    degree of an edge. Replacing them by their means gives

        a/d + c/k  =  1 + chi/E

    with `a` the mean arity (vertices per relation) and `c` the mean closure (faces per
    relation). Exact on branching, open and face-free complexes, and it reduces to the
    surface form at `a = c = 2`. Be clear about what it is: `a/d = V/E` and `c/k = F/E`,
    so this is Euler divided by `E`, a rewriting rather than new content. The CONTENT is the
    specialisation, because fixing `a` and `c` turns it into a relation between two
    intensive quantities.

    **The ideals are indexed by (a, c), not unique.** The limit `a/d + c/k = 1` has a
    different finite family for each arity/closure profile:

        a=2 c=2   (3,6) (4,4) (6,3)          <- the three regular tilings
        a=2 c=3   (3,9) (4,6) (5,5) (8,4)
        a=3 c=2   (4,8) (5,5) (6,4) (9,3)
        a=3 c=3   (4,12) (6,6) (12,4)
        a=4 c=2   (5,10) (6,6) (8,4) (12,3)

    So the classical three tilings are the `a = c = 2` row of a family, and a branching
    complex refines toward a different ideal set. The self-dual member is always
    `d = k = a + c`, since `(a+c)/d = 1` at `d = k`.

    **Scope of the SURFACE reading.** `applicable` is True only for pairwise relations
    with every edge in exactly two faces, which is what makes the `(k, d)` pair the whole
    story. The identity itself holds regardless and is reported either way.
    """
    from rexgraph.graded_boundary import graded_boundaries_from_rex

    bounds = [sp.csr_matrix(b) for b in graded_boundaries_from_rex(rex)]
    if len(bounds) < 2:
        return {"applicable": False, "reason": "no faces: not a surface"}
    B1, B2 = bounds[0], bounds[1]
    nV, nE, nF = B1.shape[0], B1.shape[1], B2.shape[1]
    if nV == 0 or nE == 0 or nF == 0:
        return {"applicable": False, "reason": "empty at some grade"}

    arity = np.asarray((abs(B1) > 0).sum(axis=0)).ravel()
    face_deg = np.asarray((abs(B2) > 0).sum(axis=1)).ravel()
    pairwise = bool((arity == 2).all())
    closed = bool((face_deg == 2).all())

    i1 = int((abs(B1) > 0).sum())                # vertex-relation incidences
    i2 = int((abs(B2) > 0).sum())                # relation-face incidences
    chi = nV - nE + nF
    a = Fraction(i1, nE)                         # mean arity
    d = Fraction(i1, nV)                         # mean vertex degree
    c = Fraction(i2, nE)                         # mean closure
    k = Fraction(i2, nF)                         # mean face size
    lhs = a / d + c / k
    rhs = Fraction(1) + Fraction(chi, nE)
    return {
        "applicable": bool(pairwise and closed),
        "pairwise": pairwise, "closed": closed,
        "V": int(nV), "E": int(nE), "F": int(nF), "chi": int(chi),
        "mean_arity": str(a), "mean_degree": str(d),
        "mean_closure": str(c), "mean_face_size": str(k),
        "lhs": str(lhs), "rhs": str(rhs), "holds": bool(lhs == rhs),
        "deviation_from_ideal": str(Fraction(chi, nE)),
        "self_dual_ideal": str(a + c),
        "reading": ("a/d + c/k = 1 + chi/E exactly. The limit a/d + c/k = 1 has a "
                    "finite ideal family per (a, c); a = c = 2 gives the three regular "
                    "tilings, and the self-dual member is always d = k = a + c"),
    }


def manifold_state(rex, grade: int = 2) -> dict:
    """Whether the complex is latent, filled, or closed at `grade`. They are three
    states, not two, and the middle one is why homology alone cannot answer this.

        latent    cycles exist and none of them bound       harmonic = cycles, curl = 0
        filled    every cycle bounds                        harmonic = 0
        closed    filled AND every cell bounds exactly two

    The separating case is a tetrahedron carrying three of its four faces. Three faces
    already span the three cycles, so `harmonic = 0` and there is no hole anywhere. Yet
    every edge does not lie in two faces, so it is not a closed surface. The fourth face
    adds NO rank; it is homologically redundant and geometrically necessary.

    So `beta = 0` does not mean closed, and a reading built only on homology will call
    that tetrahedron a sphere. Closure is a statement about incidence, which is what
    `closure_at` measures and what the mass tower is extensive over.

    Filling never changes how many cycles there are: `ker(B_1)` is fixed by the
    1-skeleton. Filling moves them from harmonic to curl, which is what
    `harmonic_shadow` counts.
    """
    from rexgraph.graded_boundary import _sparse_rank

    bounds = _boundaries(rex)
    if not bounds:
        return {"state": "empty", "cycles": 0, "curl": 0, "harmonic": 0}
    n_cells = bounds[0].shape[1]
    cycles = n_cells - int(_sparse_rank(bounds[0]))
    curl = int(_sparse_rank(bounds[1])) if len(bounds) > 1 else 0
    harmonic = cycles - curl
    closed = closure_at(rex, grade)["every_two"]
    if cycles == 0:
        state = "acyclic"
    elif curl == 0:
        state = "latent"
    elif harmonic > 0:
        state = "partially filled"
    else:
        state = "closed" if closed else "filled"
    return {
        "state": state,
        "cycles": int(cycles), "curl": int(curl), "harmonic": int(harmonic),
        "closed": bool(closed),
        "reading": ("cycles is fixed by the 1-skeleton; filling moves them from "
                    "harmonic to curl. closed is an incidence fact homology cannot see"),
    }


def closure_at(rex, grade: int = 2) -> dict:
    """Whether the complex closes at `grade`, by two readings that are not the same.

    `mass_equal` is `||B_{grade-1}||^2 == ||B_grade||^2`. It is cheap, exact and
    grade-general, and it is a statement about the MEAN incidence degree being 2, so it
    is NECESSARY for closure and not sufficient. A boundary with degrees (1, 2, 2, 3)
    satisfies it while being nothing of the kind.

    `every_two` is the actual condition: every `(grade-1)`-cell bounds exactly two
    `grade`-cells. Costs a pass over the stored pattern.

    Both are returned because the first is the one worth computing on every complex and
    the second is the one worth trusting. On the solid octahedron, square pyramid,
    truncated icosahedron and closed tetrahedron they agree; on a tetrahedron missing a
    face and on a single triangle they agree that it is open.
    """
    grade = int(grade)
    lower = boundary_mass(rex, grade - 1, exact=False) if grade >= 2 else 0.0
    upper = boundary_mass(rex, grade, exact=False)
    deg = incidence_degrees(rex, grade)
    every_two = bool(deg.size and (deg == 2).all())
    return {
        "grade": grade,
        "mass_below": float(lower), "mass_at": float(upper),
        "mass_equal": bool(abs(float(lower) - float(upper)) < 1e-9),
        "every_two": every_two,
        "degrees": {str(int(d)): int(c) for d, c in
                    zip(*np.unique(deg, return_counts=True), strict=True)} if deg.size
                   else {},
        "closed": every_two,
        "reading": ("mass_equal is necessary and cheap; every_two is the condition. "
                    "They can disagree, and where they do the second is right."),
    }


#: why a per-cell sign is not a reading of the complex from grade 2 up
_PARITY_NOTE = (
    "a solved column is determined only up to an overall sign, so per-cell parity and "
    "n_negative describe the REPRESENTATIVE, not the complex: negate one column and both "
    "move without a cell changing. The invariant is the holonomy around a loop of cells, "
    "reported as balanced/n_frustrated in the global view and by "
    "faces.orientation_holonomy, where +1 everywhere IS coherent orientability"
)


def apd(rex, grade: int = 1, *, view: str = "local"):
    """Arity, parity and degree per cell: the three directions of a graded complex.

        A(c)   |boundary of c|            looks DOWN a grade
        P(c)   sign product of that boundary   the ORIENTATION reading
        D(c)   how many cells contain c   looks UP a grade

    They are the same three separable axes the canon names (share, orientation,
    existence), read per cell instead of per channel. Nothing here is derived from
    another: a cell can be wide and lonely, narrow and busy, balanced or frustrated,
    independently.

    **P is never a gauge-free reading.** At grade 1 it is constant; from grade 2 it varies
    but only with the chosen representative, since a solved column is fixed only up to an
    overall sign. The invariant one grade up is the holonomy around a loop of cells, which
    the global view carries as `balanced` / `n_frustrated` and
    `faces.orientation_holonomy` computes: +1 on every independent loop IS coherent
    orientability, and it survives any per-cell flip.

    **P is only representative-dependent from grade 2 up.** A `B_1` column is canonically
    `(-1, +share, ..., +share)`, exactly one negative whatever the arity and whatever
    vertex is distinguished, so its sign product is -1 for every relation and reversing
    an edge does not move it. Measured, not assumed. From grade 2 the coefficients are
    solved and their signs vary: a triangle reads `[1,1,1]` and the same triangle with an
    edge reversed reads `[1,-1,-1]`. The product is the HOLONOMY, +1 balanced and -1
    frustrated; `n_negative` counts how many relations run against their stored
    orientation.

    `view="global"` returns the means instead of the cells. Those means ARE the terms of
    `surface_identity`: `a` is mean A at grade 1, `d` is mean D at grade 0, `c` is mean D
    at grade 1, `k` is mean A at grade 2. So the identity is an APD statement about
    consecutive grades, and local and global here are one operator read at two scopes.
    """
    if view not in ("local", "global"):
        raise ValueError(f"view must be 'local' or 'global', got {view!r}")
    grade = int(grade)
    bounds = _boundaries(rex)
    if grade < 1 or grade > len(bounds):
        return {"grade": grade, "view": view, "cells": [], "n": 0,
                "reason": "no cells at this grade"}

    down = bounds[grade - 1].tocsc()
    up = bounds[grade] if grade < len(bounds) else None
    n_cells = down.shape[1]
    up_deg = (np.asarray((abs(up) > 0).sum(axis=1)).ravel()
              if up is not None else np.zeros(n_cells, dtype=int))
    # `> 0` above and the nonzero filter below are the same rule: a solved face column can
    # carry an explicit zero for a relation it does not use (a nullspace basis vector over
    # a group reads [3,-3,-2,-1,0]), and counting that entry would make arity the number
    # of relations OFFERED rather than the gon. `faces.face_support` counts the same way,
    # and so does `surface_identity`, which is what makes the means agree.

    cells = []
    for c in range(n_cells):
        stored = down.data[down.indptr[c]:down.indptr[c + 1]]
        col = stored[stored != 0]           # a stored zero is not a side
        signs = np.sign(col)
        cells.append({
            "index": c,
            "arity": int(col.shape[0]),
            "parity": int(np.prod(signs)) if col.shape[0] else 0,
            "n_negative": int((signs < 0).sum()),
            "degree": int(up_deg[c]) if c < up_deg.shape[0] else 0,
        })

    if view == "local":
        return {"grade": grade, "view": "local", "n": n_cells, "cells": cells,
                "parity_informative": grade >= 2,
                "parity_is_gauge": grade >= 2,
                "parity_note": _PARITY_NOTE if grade >= 2 else None}

    arities = [c["arity"] for c in cells]
    degrees = [c["degree"] for c in cells]
    holonomy = None
    if grade >= 2:
        from rexgraph.faces import orientation_holonomy
        holonomy = orientation_holonomy(rex, grade=grade)
    return {
        "grade": grade, "view": "global", "n": n_cells,
        "mean_arity": str(Fraction(sum(arities), n_cells)) if n_cells else "0",
        "mean_degree": str(Fraction(sum(degrees), n_cells)) if n_cells else "0",
        "n_frustrated": (holonomy["frustrated"] if holonomy else None),
        "n_loops": (holonomy["n_loops"] if holonomy else None),
        "balanced": (holonomy["orientable"] if holonomy else None),
        "parity_informative": grade >= 2,
        "parity_is_gauge": grade >= 2,
        "parity_note": _PARITY_NOTE if grade >= 2 else None,
        "reading": ("the means at consecutive grades are the terms of "
                    "surface_identity: a = mean arity at 1, d = mean degree at 0, "
                    "c = mean degree at 1, k = mean arity at 2"),
    }


def semantic_closure(rex, seed: int, *, max_depth: int = 8, grade: int = 0) -> dict:
    """How far "tell me about X" has to reach before the answer stops changing.

    The open question in graph engineering, phrased as the analogue of statistical
    significance: given a query about one entity, what is enough? Ask for too little and
    the answer is a fragment; ask for too much and you have returned the database.

    There is an exact stopping rule and it needs no threshold. Expand the seed's
    neighbourhood one hop at a time and read the SHAPE of the subcomplex it induces at
    each step. The first depth whose reading repeats the one before it is where more
    context stops being more answer. Nothing is being fitted and no tolerance is being
    chosen: the reading either changed or it did not.

    The reading is `(nV, nE, betti)`, because betti is what says whether the evidence
    closes. A neighbourhood that is still a tree is a chain of facts hanging off the
    seed; one that has acquired a cycle has facts that corroborate each other through a
    second path, and that is a different kind of answer.

    Measured on real BindingDB binding data, the depth is a property of the ENTITY and not
    a setting::

        P00734   depth 1     its ligand panel is self-contained
        P00918   depth 2     at depth 1 a star, at depth 2 six independent cycles,
                             because its ligands are shared with other targets

    So "how much is enough" is answerable per query rather than configured globally, and
    a promiscuous entity honestly needs more context than a self-contained one.

    Returns the depth, the reading at each step, and the relations the closure contains.
    `converged: False` means it was still growing at `max_depth`, which is a real answer
    about the seed rather than a failure: some entities are not locally closed.
    """
    supports = rex.relation_supports()
    nE = int(rex.nE)

    def neighbourhood(depth):
        vertices = {int(seed)}
        for _ in range(depth):
            touching = [e for e in range(nE) if vertices & set(supports[e])]
            vertices |= {v for e in touching for v in supports[e]}
        return ([e for e in range(nE) if set(supports[e]) <= vertices],
                sorted(vertices))

    def shape(edges, vertices):
        if not edges:
            return (0, 0, ())
        index = {v: i for i, v in enumerate(vertices)}
        ptr, idx = [0], []
        for e in edges:
            idx.extend(index[v] for v in supports[e])
            ptr.append(len(idx))
        from rexgraph.graph import RexGraph
        sub = RexGraph(boundary_ptr=np.asarray(ptr, dtype=np.int32),
                       boundary_idx=np.asarray(idx, dtype=np.int32))
        sub._ensure_clean()
        return (int(sub.nV), int(sub.nE), tuple(int(b) for b in sub.betti))

    steps, previous, depth, edges = [], None, None, []
    for d in range(1, int(max_depth) + 1):
        e, v = neighbourhood(d)
        reading = shape(e, v)
        steps.append({"depth": d, "nV": reading[0], "nE": reading[1],
                      "betti": list(reading[2])})
        if reading == previous:
            depth = d - 1
            break
        previous, edges = reading, e
    return {
        "seed": int(seed), "grade": int(grade),
        "depth": depth, "converged": depth is not None,
        "steps": steps, "relations": edges,
        "reading": ("the first depth whose shape repeats the one before it: more context "
                    "stops being more answer there. Exact, with no threshold, because the "
                    "reading either changed or it did not"),
    }


def graded_delta(rex) -> list:
    """L_gb across every adjacent grade pair: the graded boundary delta.

    Where the mass tower reads each grade on its own, this reads the COUPLING
    between adjacent grades. `L_gb = a a^T/|a|^2 - b b^T/|b|^2` on the two grades'
    normalized coherence spectra, a difference of two rank-1 orthogonal projectors,
    so its whole spectrum follows from one dot product:

        nonzero eigenvalues   +-sqrt(spread(a, b))
        ||L_gb||_F            sqrt(2 * spread(a, b))

    No eigensolver and no L x L matrix for those three. `localization` still reads
    the entrywise |L_gb|, which is not rank-2 and has no closed form.

    Returns one dict per pair with `pair`, `top_eig`, `bot_eig`, `spread`, `frob`
    and `localization`. The tower is a fingerprint: a sphere has a distinctive
    signature across its pairs.
    """
    from rexgraph.core._l_gb import l_gb_tower
    return l_gb_tower([np.asarray(b.todense(), dtype=np.float64)
                       for b in _boundaries(rex)])


def channel_delta(rex):
    """L_gb between the four channel hats at grade 1, as a 4x4 array.

    The within-grade companion to `graded_delta`: entry [i, j] is the Frobenius
    norm of L_gb between channel i and channel j, in the order
    (topology, geometry, frustration, coparticipation). The diagonal is zero by
    construction, a channel matching itself.
    """
    from rexgraph.core._l_gb import l_gb_channel_tensor
    rex._ensure_clean()
    hats = list(rex._rcf_bundle.get("hats", []) or [])
    if not hats:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(l_gb_channel_tensor(hats), dtype=np.float64)
