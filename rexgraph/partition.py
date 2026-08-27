"""Section readings: the leverage is a measure, so it partitions.

`R_eff` is the diagonal of the orthogonal projector onto the row space of `B_k`, so it
refines the rank tower and distributes over ANY sectioning of the cells. The sections may
differ in type, arity and degree without disturbing it, and the totals close exactly.

Per section `S`::

    mass        sum R_eff(c)            its share of rank(B_k)
    own_rank    rank(B_k restricted)    what it spans by itself
    efficiency  mass / own_rank         how much of what it spans it is granted
    own_cycles  |S| - own_rank          the cycles it carries alone
    share       sum (1 - R_eff(c))      its share of the global cycle space
    gap         share - own_cycles      what the REST of the complex closes for it

The gap is the one with no local analogue: a section whose gap is zero carries its own
cycles entirely and depends on nothing outside, and a large gap means the surrounding
structure is doing the closing. Both bounds are theorems rather than conventions, and
`verify=True` asserts them: masses sum to `rank(B_k)`, and no section's own cycles exceed
its share.

One solve serves every section, because the leverage is computed once over all cells and
then regrouped. Sectioning a complex a different way costs a rank per section and nothing
else.

`coupling` is separate and optional, because it needs the field rather than its
quadrance: the spread squares the SIGN of the field Gram away, so it is a coordinate the
spread cannot carry and not a variant of one.

What the sign means is settled for the pairwise case and measured for the rest. By hand
on a triangle whose edges run consistently around the cycle, `<b_i, L0+ b_j> = -1/3`, and
reversing one edge flips it to `+1/3`; on a tree it is exactly 0, because distinct edges
share no current path. So NEGATIVE is co-oriented along a shared cycle.

That does NOT license reading the fraction as "these relations disagree". On a knowledge
complex every relation is built with the same convention (the record at position 0), so
there is no orientation disagreement present to find, and the fraction reads the geometry
of how supports overlap instead. Measured on 1373 Complex Portal complexes against the
file's own GO annotations, with the shared-subunit count HELD FIXED by stratification,
functionally related pairs score HIGHER: pooled +0.0411 +- 0.0045, z = +9.12, 11 of 14
strata agreeing, and only rho +0.21 against the raw overlap. So it is coupling, it is not
a rescaled overlap count, and it is not conflict. The mean Gram does not survive the same
control (its per-stratum sign flips), so the FRACTION is the reading and the mean is not.
"""
from __future__ import annotations

import numpy as np

__all__ = ["section_readings", "section_response", "section_coverage",
           "coupling_fraction",
           "grade_leverage", "section_tensor",
           "candidate_readings", "byte_energy", "energy_tensor", "compose_substrates",
           "hodge_share"]


def section_readings(rex, sections, *, leverage=None, verify=True):
    """Readings for each section of a sectioning of the relations.

    `sections` maps a name to the relation indices it contains. Sections may overlap, and
    group sections DO: a pairwise relation belongs to every group holding both its
    members. The closure identity is asserted only for a genuine partition, meaning the
    sections are disjoint AND cover every relation; an overlapping cover double-counts
    mass and closing on the rank is not something it can be expected to do.

    Returns `{name: {n, mass, own_rank, efficiency, own_cycles, share, gap}}`.
    """
    from rexgraph.graded_boundary import _sparse_rank

    nE = int(rex.nE)
    reff = (np.asarray(leverage) if leverage is not None
            else np.asarray(rex._effective_resistance_batch(np.arange(nE))))
    if reff.shape != (nE,):
        raise ValueError(f"leverage has {reff.shape} for {nE} relations")
    Bint = rex._integer_B1().tocsc()          # rank on the exact integer representative

    out = {}
    for name, ids in sections.items():
        e = np.asarray(sorted({int(i) for i in ids}), dtype=np.int64)
        if e.size == 0:
            continue
        if e.min() < 0 or e.max() >= nE:
            raise IndexError(f"section {name!r} indexes outside 0..{nE - 1}")
        own_rank = int(_sparse_rank(Bint[:, e]))
        mass = float(reff[e].sum())
        share = float((1.0 - reff[e]).sum())
        own_cycles = int(e.size - own_rank)
        out[name] = {
            "n": int(e.size), "mass": mass, "own_rank": own_rank,
            "efficiency": mass / own_rank if own_rank else float("nan"),
            "own_cycles": own_cycles, "share": share,
            "gap": share - own_cycles,
        }

    if verify:
        for name, r in out.items():
            if r["mass"] > r["own_rank"] + 1e-6:
                raise ValueError(
                    f"section {name!r} holds mass {r['mass']:.6f} above its own rank "
                    f"{r['own_rank']}: a projector diagonal cannot exceed the rank it "
                    f"refines")
            if r["own_cycles"] > r["share"] + 1e-6:
                raise ValueError(
                    f"section {name!r} carries {r['own_cycles']} cycles of its own "
                    f"against a share of {r['share']:.6f}: restriction cannot raise rank")
        # a PARTITION closes on the rank; an overlapping COVER does not, and group
        # sections overlap by construction because a pair relation belongs to every group
        # holding both its members. Testing coverage alone made a legitimate cover raise.
        total_cells = sum(len({int(i) for i in ids}) for ids in sections.values())
        covered = sorted({int(i) for ids in sections.values() for i in ids})
        disjoint = total_cells == len(covered)
        if disjoint and len(covered) == nE and covered == list(range(nE)):
            total = sum(r["mass"] for r in out.values())
            rank = int(_sparse_rank(Bint))
            if abs(total - rank) > 1e-6 * max(rank, 1):
                raise ValueError(
                    f"the sections partition the relations but their masses total "
                    f"{total:.6f} against rank(B1) {rank}. The leverage refines the "
                    f"rank, so a partition must close.")
    return out


def coupling_fraction(rex, sections, *, field=None):
    """Fraction of each section's field-Gram off-diagonals that are POSITIVE.

    NEGATIVE is co-oriented along a shared cycle and 0 is no shared current path, both
    verified by hand on a triangle and a tree. On a uniformly oriented complex the
    fraction reads coupling geometry rather than disagreement, and it separates
    functionally related sections from unrelated ones with the overlap held fixed; the
    module docstring carries the numbers. Do not read it as conflict.

    The FRACTION is the reading. The mean Gram does not survive the same control.

    Needs the field, so it costs a solve over the union of the sections.
    """

    from rexgraph.semantic import relation_field

    need = sorted({int(i) for ids in sections.values() for i in ids})
    if not need:
        return {}
    V = field if field is not None else relation_field(rex, need)[0]
    pos = {e: k for k, e in enumerate(need)}
    # rex.B1 is DENSE by design (to_dense_f64 of the dual), so going through it
    # allocates nV x nE. The dual is the sparse object; read it directly.
    from rexgraph.core._sparse import to_scipy_csr
    B = to_scipy_csr(rex._B1_dual).tocsc()[:, need].tocsc()

    out = {}
    for name, ids in sections.items():
        c = np.asarray([pos[int(i)] for i in sorted({int(x) for x in ids})
                        if int(i) in pos], dtype=np.int64)
        if c.size < 3:
            out[name] = float("nan")
            continue
        G = B[:, c].T @ V[:, c]
        G = 0.5 * (G + G.T)
        iu = np.triu_indices(c.size, 1)
        out[name] = float((np.asarray(G)[iu] > 0).mean())
    return out


def _leverage_of(B, *, block=None):
    """diag of the projector onto row(B), matrix-free and in blocks.

    `R_eff(c) = b_c^T (B B^T)^+ b_c` with the kernel deflated, which is the same solve
    `_effective_resistance_batch` runs at grade 1, so both go through the one primitive
    rather than blocking the same solve two different ways.
    """
    from rexgraph.fiedler import leverage_diagonal

    return leverage_diagonal(B, block=block)


def grade_leverage(rex, k, *, verify=True):
    """The leverage at grade `k`: the diagonal of the projector onto row(B_k).

    Nothing about the grade-1 reading was about grade 1. `R_eff(e) = z^T B^T (B B^T)^+ B
    z` is the e-th diagonal of the orthogonal projector onto the row space, so writing it
    for `B_k` is the same expression with a different operator, and every reading built on
    it comes with it: a cell's mass, its share of the cycle space, and a section's gap.

    Foster generalises with it and is the self-test at every grade::

        sum_c R_eff_k(c) = rank(B_k)        so   sum_c (1 - R_eff_k(c)) = dim ker(B_k)

    Returns `(leverage, rank)`. An empty grade returns an empty array and rank 0.
    """
    import scipy.sparse as sp

    Bs = rex.graded_boundaries()
    if k < 1 or k > len(Bs):
        raise IndexError(f"grade {k} outside 1..{len(Bs)} for this complex")
    if k == 1:
        # DELEGATE. _effective_resistance_batch is the canonical reading: it settles
        # bridges by one walk with no solve at all, runs deflated block CG for the rest,
        # and only reaches a dense decomposition behind check_dense_allocation. Writing
        # a bare SVD here threw all of that away.
        nE = int(rex.nE)
        lev = np.asarray(rex._effective_resistance_batch(np.arange(nE)))
        rank = int(rex.rank_tower()["ranks"][0])
    else:
        B = sp.csc_matrix(Bs[k - 1])
        if B.shape[1] == 0:
            return np.zeros(0), 0
        lev, rank = _leverage_of(B)
    if verify and lev.size and abs(float(lev.sum()) - rank) > 1e-6 * max(rank, 1):
        raise ValueError(
            f"Foster fails at grade {k}: sum of leverage {lev.sum():.6f} against "
            f"rank(B_{k}) {rank}. The leverage refines the rank at every grade.")
    return lev, rank


def section_tensor(rex, sections, *, grades=None, leverage=None, verify=True):
    """The embedding as a graded tensor, not a row of scalars.

    A section's reading is indexed by GRADE as well as by what is being read, because the
    same construction repeats at every grade over that grade's boundary operator. So the
    object is `E[section, grade, reading]` and flattening it to one vector per section
    throws away the axis that says which grade the structure lives at.

    The readings per (section, grade) are the ones the leverage supports there: `n`,
    `mass`, `own_rank`, `efficiency`, `own_cycles`, `share`, `gap`, plus the section's OWN
    Hodge blocks as exact integers. The blocks come from the rank tower restricted to the
    section, which is what makes them coordinates rather than a per-grade constant::

        gradient(S, k) = rank(B_k |S)                  what its own boundary spans
        curl(S, k)     = rank(B_{k+1} |S)              what its own faces fill
        harmonic(S, k) = n_k - gradient - curl         its own beta_k, what stays a hole

    and that is also the decomposition of a reading the flat version left whole::

        own_cycles = n_k - gradient = curl + harmonic

    A section carrying cycles says nothing about whether they are filled. Splitting them
    is the difference between "there is a loop here" and "there is a hole here", so the
    block axis is the one that makes `own_cycles` actionable.

    `sections` maps a name to that grade's cell indices. To section several grades, pass
    `{name: {grade: ids}}`; a flat `{name: ids}` is read as grade 1.

    `leverage` is `{grade: array}` and is the whole point of the design: one solve serves
    every sectioning, so a caller reading the same complex several ways passes it in
    rather than paying the solve again. Grade 1 over 11616 real relations costs about
    17s, so recomputing it per call is the difference between one solve and n.

    Returns `(tensor, axes)` with `tensor[i, k, r]` and
    `axes = {"sections": [...], "grades": [...], "readings": [...]}`.
    """
    from rexgraph.graded_boundary import _sparse_rank

    per_grade = {}
    for name, val in sections.items():
        if isinstance(val, dict):
            for k, ids in val.items():
                per_grade.setdefault(int(k), {})[name] = ids
        else:
            per_grade.setdefault(1, {})[name] = val
    ks = sorted(per_grade) if grades is None else sorted(int(g) for g in grades)
    names = list(sections)
    READINGS = ("n", "mass", "own_rank", "efficiency", "own_cycles", "share", "gap",
                "gradient", "curl", "harmonic")
    T = np.full((len(names), len(ks), len(READINGS)), np.nan)

    import scipy.sparse as sp

    Bs = rex.graded_boundaries()

    def _op(g):
        """The exact integer representative of B_g, or None past the top grade."""
        if g < 1 or g > len(Bs):
            return None
        return rex._integer_B1().tocsc() if g == 1 else sp.csc_matrix(Bs[g - 1])

    for a, k in enumerate(ks):
        want = per_grade.get(k, {})
        if not want:
            continue
        if leverage is not None and k in leverage:
            lev = np.asarray(leverage[k])
        else:
            lev, _rank = grade_leverage(rex, k, verify=verify)
        Bk, Bup = _op(k), _op(k + 1)
        for i, name in enumerate(names):
            ids = want.get(name)
            if ids is None:
                continue
            c = np.asarray(sorted({int(x) for x in ids}), dtype=np.int64)
            if c.size == 0:
                continue
            if c.min() < 0 or c.max() >= lev.size:
                raise IndexError(f"section {name!r} indexes outside grade {k}")
            grad = int(_sparse_rank(Bk[:, c]))
            mass = float(lev[c].sum()); share = float((1.0 - lev[c]).sum())
            ownc = int(c.size - grad)
            # the section's own curl: the faces it supports, restricted to ITS cells, so
            # a face reaching outside the section does not fill a cycle inside it
            curl = 0
            up = sections.get(name)
            up_ids = up.get(k + 1) if isinstance(up, dict) else None
            if Bup is not None and up_ids is not None and len(up_ids):
                f = np.asarray(sorted({int(x) for x in up_ids}), dtype=np.int64)
                if f.size and f.max() < Bup.shape[1]:
                    curl = int(_sparse_rank(Bup[c, :][:, f]))
            harm = ownc - curl
            T[i, a, :] = (c.size, mass, grad, mass / grad if grad else np.nan, ownc,
                          share, share - ownc, grad, curl, harm)
            if verify:
                if harm < 0:
                    raise ValueError(
                        f"section {name!r} at grade {k} has curl {curl} above its own "
                        f"cycles {ownc}: faces cannot fill more cycles than there are")
                if mass > grad + 1e-6:
                    raise ValueError(
                        f"section {name!r} at grade {k} holds mass {mass:.6f} above its "
                        f"own rank {grad}")
                if ownc > share + 1e-6:
                    raise ValueError(
                        f"section {name!r} at grade {k} carries {ownc} cycles against a "
                        f"share of {share:.6f}")
    return T, {"sections": names, "grades": ks, "readings": list(READINGS)}


def candidate_readings(rex, candidates, *, shares=True):
    """Read relations that do NOT exist yet, before committing any of them.

    Declaring is not materialising. A candidate relation's effect on the complex is
    decided by one question with an exact answer, and no eigensolver and no trial
    insertion are needed to ask it: its boundary column either lies in `range(B_1)` or it
    does not.

        outside   the column adds a direction nothing else reaches. Materialising it
                  raises `rank(B_1)` by one, joins what it spans, and closes no cycle.
        inside    the column is already spanned. Materialising it leaves the rank alone
                  and adds exactly one cycle, and `quadrance` says how far apart its
                  support already is: near 0 means the complex already ties those
                  vertices tightly and the relation would say almost nothing new.

    That is the generation predicate. `frustration_delta` measures a candidate the same
    way in spore; here the same decision falls out of the rank tower, so it costs a
    projection rather than a rebuild.

    `candidates` is an iterable of vertex supports. With `shares=True` each becomes the
    zero-sum column `(-1, 1/(k-1), ...)` the model uses at any arity; pass explicit
    `(support, values)` pairs to read a column verbatim.

    Returns a list of `{support, k, kind, spans_new, quadrance, closes}` in the order
    given. `kind` names the arity class (`witness` (k=1), `pairwise` (k=2) or
    `branching` (k>2)) because the three behave differently under the boundary and a
    reading that does not say which it read cannot be checked.
    """
    import scipy.sparse as sp

    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.fiedler import deflated_operator
    from rexgraph.graded_boundary import _sparse_rank
    from rexgraph.sparse_character import _block_cg

    nV = int(rex.nV)
    # NO DECOMPOSITION OF B. range(B) = range(B B^T) = ker(L0) orthogonal, so "is this
    # column spanned" is a test against the KERNEL, which has dimension beta_0 and is
    # tiny, rather than against a factorisation of the whole operator. The quadrance is
    # then one deflated CG solve, matrix-free, exactly as _effective_resistance_batch
    # does it. The earlier version densified B and took its SVD, which is 1.05 GB at
    # nV 4000 and nE 33k and is not what this library does anywhere else.
    # L0 is never formed here either: membership is a kernel test and the quadrance is
    # one deflated solve, both of which the boundary supplies directly.
    Bfull = to_scipy_csr(rex._B1_dual).tocsc()
    _apply, dinv, U, _nk = deflated_operator(Bfull)

    Bint = rex._integer_B1().tocsc()
    rank_B = None                    # exact rank is computed only if adjudication needs it
    adjudicated = 0

    out = []
    for cand in candidates:
        if (isinstance(cand, tuple) and len(cand) == 2
                and not np.isscalar(cand[1])):
            sup, vals = np.asarray(cand[0], np.int64), np.asarray(cand[1], float)
        else:
            sup = np.asarray(sorted({int(x) for x in cand}), np.int64)
            k = sup.size
            if k == 0:
                out.append({"support": [], "k": 0, "kind": "empty", "spans_new": False,
                            "quadrance": 0.0, "closes": 0.0})
                continue
            if k == 1 and (sup[0] < 0 or sup[0] >= nV):
                raise IndexError(f"candidate touches a vertex outside 0..{nV - 1}")
            if k == 1:
                # A WITNESS, and the answer is exact without a solve. Its column is
                # `(+1)`, which SUMS TO ONE; every existing boundary column sums to zero,
                # so their span lies inside the zero-sum subspace and `(+1)` is outside
                # it. A witness therefore always adds rank and closes nothing, at any
                # size and against any complex.
                #
                # Reporting `spans_new: False` here said the opposite of the truth. It
                # also erased the class. A vocative ("Take away your mother, Jerry.")
                # IS a witness, a participant that exists and bounds nothing, and calling
                # it a non-answer silently turns that sentence into a different one.
                out.append({"support": sup.tolist(), "k": 1, "kind": "witness",
                            "spans_new": True, "quadrance": 0.0, "closes": 0.0})
                continue
            vals = np.full(k, 1.0 / (k - 1)) if shares else np.ones(k)
            vals[0] = -1.0
        if sup.size and (sup.min() < 0 or sup.max() >= nV):
            raise IndexError(f"candidate touches a vertex outside 0..{nV - 1}")
        b = np.zeros(nV)
        np.add.at(b, sup, vals)
        # "Is this column in range(B1)" has an integer answer, so a float residual must
        # never be the thing that decides it. It is also the case that the projection is
        # right almost always: over 1200 candidates on 300 random complexes it never
        # disagreed with the exact rank. Deciding everything exactly anyway cost 52s
        # against the projection's 0.11s at nE 3268, because fraction-free elimination
        # over the whole operator runs per candidate.
        #
        # So the projection RULES, and the exact rank ADJUDICATES. The residual is
        # compared against the decomposition's own noise floor with a wide margin either
        # way; outside that band the answer is settled by orders of magnitude and no
        # tolerance is doing any work, and inside it the integer rank decides. The band
        # selects the method, never the verdict.
        kind = ("pairwise" if sup.size == 2 else "branching")
        nb = float(np.linalg.norm(b)) or 1.0
        # a column lies in range(B) exactly when it is orthogonal to ker(L0), so the
        # test is |U^T b| and costs nV x beta_0. In-span leaves this at machine level and
        # out-of-span leaves an O(1) fraction of the column, thirteen orders apart, so
        # the band picks the method and the integer rank settles anything between.
        rel = (float(np.linalg.norm(U.T @ b)) / nb) if U.shape[1] else 0.0
        if rel > 1e-6:
            spans_new = True
        elif rel < 1e-10:
            spans_new = False
        else:
            if rank_B is None:
                rank_B = int(_sparse_rank(Bint))
            col = np.zeros(nV)
            np.add.at(col, sup,
                      vals * ((sup.size - 1) if shares and sup.size > 1 else 1))
            aug = sp.hstack([Bint, sp.csc_matrix(col.reshape(-1, 1))]).tocsc()
            spans_new = int(_sparse_rank(aug)) > rank_B
            adjudicated += 1
        if spans_new:
            q = float("inf")
        else:
            y = _block_cg(_apply, b.reshape(-1, 1), dinv, tol=1e-12, maxit=500)
            q = float(b @ y[:, 0])
        out.append({"support": sup.tolist(), "k": int(sup.size), "kind": kind,
                    "spans_new": bool(spans_new), "quadrance": q,
                    "closes": 0.0 if spans_new else 1.0,
                    "adjudicated": bool(1e-10 <= rel <= 1e-6)})
    return out


#### the second substrate #####################################################
#
# Every reading above is taken ON the complex, so all of them read one propagated signal
# and inherit its statistics. The byte energy does not: it reads the ENCODING, before any
# relation exists, and it is the only corpus-free quantity here.
#
# Theorem 27 is the rule for putting them together, and it is a prohibition. Carrying the
# energy as a source and solving `L0 u = B1 E` re-imports the frequency coupling the
# energy was free of, and the propagated readings then INVERT: dissipated power ranks
# function words above content words, exactly reversing the ungated energy. So the
# composition is multiplicative and the substrates stay apart.
#
# That is why this is not another column on the readings axis. Appending it would fuse
# the two, and what is missing from a substrate cannot be recovered by mixing harder. It
# gets its own axis, and `compose_substrates` multiplies.

def byte_energy(label) -> float:
    """`E(w) = sum (byte * position)^2` over the utf-8 encoding of `label`.

    No complex, no corpus, no neighbours: this is a property of the string. Position is
    1-based so the first byte contributes rather than vanishing.
    """
    return float(sum((b * (i + 1)) ** 2
                     for i, b in enumerate(str(label).encode("utf-8"))))


def energy_tensor(rex, sections, labels, *, moments=("total", "mean", "spread")):
    """`E[section, moment]`: the energy substrate, computed WITHOUT the complex.

    A section's energy is read off the labels of the vertices its cells touch. Only the
    incidence is used, to find WHICH labels; no solve, no field and no propagation, which
    is what keeps it corpus-free.

        total    sum of the byte energies of the section's distinct vertices
        mean     total / number of them
        spread   peak / total, so a section dominated by one long label reads high

    PICK THE MOMENT DELIBERATELY. Per VERTEX the energy is corpus-free, and measured on
    prose it sits at rho = +0.175 against the structural reading while the structural
    readings sit at -0.737 with frequency among themselves. Per SECTION that only
    survives for `mean`. Over 135 sentence sections of the same corpus::

        total    +0.61 to +0.70 against n, mass, own_rank, share and gap
        spread   -0.49 to -0.63 against the same
        mean     |rho| <= 0.13 against all of them except efficiency, at +0.33

    `total` is a sum over the section's vertices and `spread` is a share of it, so both
    carry how BIG the section is, which the structural readings already say. `mean` is
    the size-free one and is the moment that earns a separate axis.

    `labels` is indexed by vertex. Returns `(E, moment_names)`.
    """

    lab = list(labels)
    if len(lab) != int(rex.nV):
        raise ValueError(f"labels has {len(lab)} entries for {int(rex.nV)} vertices")
    e_of = np.asarray([byte_energy(x) for x in lab], dtype=np.float64)
    sup = rex._boundary_incidence().T.tocsc()      # |B1|, nV x nE, built by the kernel

    names = list(moments)
    E = np.zeros((len(sections), len(names)))
    for i, (_name, val) in enumerate(sections.items()):
        ids = val.get(1) if isinstance(val, dict) else val
        c = np.asarray(sorted({int(x) for x in (ids or ())}), dtype=np.int64)
        if c.size == 0:
            E[i, :] = np.nan
            continue
        verts = np.unique(sup[:, c].indices)
        w = e_of[verts]
        tot = float(w.sum())
        for j, m in enumerate(names):
            if m == "total":
                E[i, j] = tot
            elif m == "mean":
                E[i, j] = tot / w.size if w.size else np.nan
            elif m == "spread":
                E[i, j] = float(w.max()) / tot if tot else np.nan
            elif m == "peak":
                E[i, j] = float(w.max())
            elif m == "n_vertices":
                E[i, j] = float(w.size)
            else:
                raise ValueError(f"unknown energy moment {m!r}")
    return E, names


def compose_substrates(T, E, *, verify=True):
    """`P[section, grade, reading, moment] = T[...] * E[section, moment]`.

    Theorem 27's composition, written out. The energy enters as a FACTOR and never as a
    source, so the two substrates are multiplied and not mixed, and the result is rank one
    in the (reading, moment) plane for every section: exactly the statement that nothing
    fused. `verify=True` checks that rank, which is the theorem being asserted rather
    than assumed.
    """
    T = np.asarray(T, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    if T.shape[0] != E.shape[0]:
        raise ValueError(f"{T.shape[0]} sections in T against {E.shape[0]} in E")
    P = T[..., None] * E[:, None, None, :]
    if verify:
        for i in range(P.shape[0]):
            for k in range(P.shape[1]):
                M = P[i, k]
                M = M[np.isfinite(M).all(axis=1)]
                if M.shape[0] < 2 or not np.isfinite(M).all() or not M.any():
                    continue
                s = np.linalg.svd(M, compute_uv=False)
                if s.size > 1 and s[1] > 1e-9 * s[0]:
                    raise ValueError(
                        f"section {i} grade {k} composes to rank > 1 "
                        f"(s2/s1 = {s[1]/s[0]:.2e}). The substrates multiply; a higher "
                        f"rank means they were mixed rather than composed.")
    return P


def hodge_share(rex, signal, *, grade=1):
    """The Hodge split of a signal, against the split a structureless signal would give.

    The three shares alone are not a finding. The decomposition is orthogonal, so the
    energy shares sum to one, but how much of it CAN land in each piece is fixed by the
    complex before the signal says anything: the pieces have dimensions
    `r_k`, `r_{k+1}` and `beta_k`, so a signal with no structure lands in each in
    proportion to its dimension. Reporting "89% gradient" without that comparison says
    nothing, because 89% may be less than chance.

    So every share comes back with its dimensional null and the EXCESS over it. A signal
    is gradient-like when its gradient share exceeds `r_k / n_k`, not when it is large.

    Returns `{share, null, excess, dims, n}` with the first three keyed by
    `gradient`, `curl`, `harmonic`.
    """
    f = np.asarray(signal, dtype=np.float64).ravel()
    n = int(rex.nE) if grade == 1 else None
    if grade != 1:
        raise NotImplementedError(
            "the Hodge split is wired for grade 1; grade_leverage covers the tower")
    if f.size != n:
        raise ValueError(f"signal is {f.size} long for {n} relations")

    g, c, h = rex.hodge(f)
    tot = float(f @ f)
    if tot <= 0:
        raise ValueError("the signal is zero, so it has no shares")
    share = {"gradient": float(g @ g) / tot, "curl": float(c @ c) / tot,
             "harmonic": float(h @ h) / tot}

    d = rex.hodge_dimensions(grade=grade)
    dims = {k: int(d[k]) for k in ("gradient", "curl", "harmonic") if k in d}
    tdim = sum(dims.values()) or n
    null = {k: dims.get(k, 0) / tdim for k in share}
    return {"n": n, "share": share, "null": null,
            "excess": {k: share[k] - null[k] for k in share},
            "dims": dims, "residual": abs(sum(share.values()) - 1.0)}


def section_coverage(rex, sections, seeds, *, seed_weight="invdeg",
                     n_sections=None, owner=None):
    """How EVENLY each section's relations are covered by a seed set.

    A companion reading to :func:`section_response`, not a replacement, and the two are
    for different query shapes. Per relation,

        m_e = (|B1|^T x)_e     unsigned: how much seed its support carries      (G side)
        g_e = ( B1^T x)_e      signed:   how unevenly it carries it             (T side)
        coverage_e = m_e - |g_e|

    which is the T/G off-diagonal mismatch on a seeded cochain. It is NOT the spread:
    Corollary 25.2 of the spread tower separates the signed reading from the spread as its
    own quantity, and this is the signed one.

    Why it exists. Every boundary column sums to zero, so `g_e = x . sum(column) = 0`
    exactly when the support is seeded UNIFORMLY: the gate shuts, and a shut gate means
    the query covers that relation evenly. But it equally means the query never touched
    it, which is the opposite fact, so the imbalance has to be read against the mass
    present. That is what the subtraction does.

    WHICH READING TO USE IS A PROPERTY OF THE QUERY, measured at n=149 per regime, top-1
    over the section a query was lifted from:

        query is...          magnitude   coverage
        the whole section       94.6%      38.3%
        half of it              71.8%      53.0%
        a quarter of it         33.6%      51.0%     <- coverage wins

    A query that quotes its section is the easy end of that curve and is not what a real
    question looks like; somewhere between a half and a quarter the two swap. Spread was
    tried as a third reading and carries nothing (0.7% / 1.3% / 2.7%), for an exact
    reason: on a column seeded at one coordinate the angle is `1 - c[j]^2/Q(c)`,
    a function of arity and role with the query divided out.

    Returns `(scores, labels)` aligned to the sectioning's own order, like
    `section_response`.
    """
    import numpy as np

    from rexgraph.core._sparse import to_scipy_csr

    seeds = np.asarray(seeds, dtype=int).ravel()
    nV, nE = int(rex.nV), int(rex.nE)
    labels = list(getattr(sections, "labels", []) or [])
    if owner is None:
        owner = np.asarray(sections.owner_cochain(nE), dtype=np.int64)
        n_sections = len(sections) if n_sections is None else n_sections
    n_sections = int(n_sections if n_sections is not None else (owner.max() + 1))
    out = np.zeros(max(n_sections, 0), dtype=np.float64)
    if seeds.size == 0 or nV == 0 or nE == 0 or n_sections == 0:
        return out, labels

    ok = seeds[(seeds >= 0) & (seeds < nV)]
    if ok.size == 0:
        return out, labels
    ind = np.zeros(nV, dtype=np.float64)
    if str(seed_weight) == "invdeg":
        deg = np.asarray(rex.degree, dtype=np.float64)
        ind[ok] = 1.0 / np.maximum(deg[ok], 1.0)
    else:
        ind[ok] = 1.0

    exact = _coverage_exact(rex, ok, owner, n_sections)
    if exact is not None:
        return exact, labels
    B = to_scipy_csr(rex._B1_dual).tocsr()
    per_cell = (abs(B).T @ ind) - np.abs(B.T @ ind)
    keep = owner >= 0
    np.add.at(out, owner[keep], per_cell[keep])
    return out, labels


def _edge_terms(rex, seeds):
    """The contributions a seed set makes to each relation, as integers.

    Scaling a boundary column by `(k-1)` clears the share, so a seed carries `-(k-1)`
    where it heads the relation and `+1` where it argues. With the vertex degrees and
    the arities as the two denominators, the reading is exact.
    """
    import numpy as np

    ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
    idx = np.asarray(rex._boundary_idx, dtype=np.int64)
    nE = int(rex.nE)
    if ptr.size != nE + 1:
        return None
    arity = np.diff(ptr)
    km1 = np.maximum(arity - 1, 1)
    deg = np.bincount(idx, minlength=int(rex.nV)).astype(np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    rank = np.full(int(rex.nV), -1, dtype=np.int64)
    rank[seeds] = np.arange(seeds.size, dtype=np.int64)
    here = rank[idx]
    take = here >= 0
    if not take.any():
        return None
    cell = np.repeat(np.arange(nE, dtype=np.int64), arity)[take]
    at_head = np.zeros(idx.size, dtype=bool)
    at_head[ptr[:-1][arity > 0]] = True
    signed = np.where(at_head[take], -km1[cell], 1).astype(np.int64)
    return cell, here[take], signed, np.maximum(deg[seeds], 1).astype(np.int64), km1, nE


def _mass_exact(rex, seeds, owner, n_sections):
    """The edge-primary reading, exactly, or None where the kernel is not built.

        mass[e] = SUM over seeds v in e of |B[v,e]|/deg[v]

    A relation answers with what its own boundary column carries from the seeds, which
    is one hop and is read where the data is. The vertex reading `|B(B^T x)|` is two,
    and a vertex is the boundary of the relations rather than the thing they carry.

    The unsigned total is what survives an evenly covered column: `B^T x` is exactly
    zero there, so a signed reading shuts precisely where a section is most
    distinctive.
    """
    import numpy as np

    try:
        from rexgraph.core import _exact_ratio
    except ImportError:
        return None
    got = _edge_terms(rex, seeds)
    if got is None:
        return np.zeros(max(n_sections, 0), dtype=np.float64)
    cell, which, signed, deg, km1, nE = got
    return _exact_ratio.axis_ratio(
        cell, np.abs(signed), which, deg, km1, nE,
        int(_exact_ratio.frac_bits_for(int(km1.max(initial=1)), deg.size, nE)),
        np.asarray(owner, dtype=np.int64), int(max(n_sections, 0)),
        _exact_ratio.SUM)


def _mass_channels(rex, seeds, owner, n_sections, labels):
    """The edge-primary reading resolved into the character's channels.

    A relation's mass is carried to its section through that relation's own profile, so
    the axes survive the accumulation instead of collapsing in it.
    """
    import numpy as np

    try:
        from rexgraph.core import _exact_ratio
    except ImportError:
        return np.zeros((max(n_sections, 0), 4), dtype=np.float64), labels, []
    got = _edge_terms(rex, seeds)
    names = list(getattr(rex, "character_channels", None)
                 or ["topology", "geometry", "frustration", "coparticipation"])
    if got is None:
        return np.zeros((max(n_sections, 0), len(names)), dtype=np.float64), labels, names
    cell, which, signed, deg, km1, nE = got
    per_cell = _exact_ratio.axis_ratio(
        cell, np.abs(signed), which, deg, km1, nE,
        int(_exact_ratio.frac_bits_for(int(km1.max(initial=1)), deg.size, nE)),
        None, 0, _exact_ratio.SUM)
    chi = np.asarray(rex.structural_character, dtype=np.float64)
    if chi.ndim != 2 or chi.shape[0] != nE:
        out = np.zeros(max(n_sections, 0), dtype=np.float64)
        np.add.at(out, owner[owner >= 0], per_cell[owner >= 0])
        return out, labels, []
    prof = np.zeros((max(n_sections, 0), chi.shape[1]), dtype=np.float64)
    keep = owner >= 0
    for k in range(chi.shape[1]):
        np.add.at(prof[:, k], owner[keep], per_cell[keep] * chi[keep, k])
    return prof, labels, names[:chi.shape[1]]


def _coverage_exact(rex, seeds, owner, n_sections):
    """Coverage over the rationals, or None where the kernel is not built.

    Every quantity is one: a boundary entry is -1 at position 0 and `1/(k-1)` after it,
    a seed weight is `1/deg`, and the reading is the unsigned total less the magnitude
    of the signed one. Scaling a column by `(k-1)` clears the share to integers, so the
    contribution is `-(k-1)` where the seed heads the relation and `+1` where it argues.
    """
    import numpy as np

    try:
        from rexgraph.core import _exact_ratio
    except ImportError:
        return None

    ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
    idx = np.asarray(rex._boundary_idx, dtype=np.int64)
    nE = int(rex.nE)
    if ptr.size != nE + 1:
        return None
    arity = np.diff(ptr)
    km1 = np.maximum(arity - 1, 1)
    deg = np.bincount(idx, minlength=int(rex.nV)).astype(np.int64)

    seeds = np.asarray(seeds, dtype=np.int64)
    rank = np.full(int(rex.nV), -1, dtype=np.int64)
    rank[seeds] = np.arange(seeds.size, dtype=np.int64)
    here = rank[idx]
    take = here >= 0
    if not take.any():
        return np.zeros(max(n_sections, 0), dtype=np.float64)

    cell = np.repeat(np.arange(nE, dtype=np.int64), arity)[take]
    at_head = np.zeros(idx.size, dtype=bool)
    at_head[ptr[:-1][arity > 0]] = True
    carried = np.where(at_head[take], -km1[cell], 1).astype(np.int64)
    return _exact_ratio.axis_ratio(
        cell, carried, here[take],
        np.maximum(deg[seeds], 1).astype(np.int64), km1.astype(np.int64), nE,
        int(_exact_ratio.frac_bits_for(int(km1.max(initial=1)), seeds.size, nE)),
        np.asarray(owner, dtype=np.int64), int(max(n_sections, 0)),
        _exact_ratio.COVERAGE)


def section_response(rex, sections, seeds, *, t=1.0, seed_weight="invdeg",
                     n_sections=None, owner=None, propagator="mass",
                     channels=False):
    """How strongly each SECTION answers a seed set, by diffusion on the field.

    This is the lookup the layer design exists for. A query names vertices; heat from
    those vertices spreads through the document's own relations (`propagate_signal`, the
    script-15 scale bridge), and each section's answer is that response restricted to the
    cells it owns. Nothing scans the text, nothing pattern-matches, and no section is
    scored by how many words it happens to share: the field decides, and the partition
    says where the answer lives.

    Cost is one diffusion plus one pass over `nnz(B1)`: the (vertex, relation) incidences
    are walked once and each contributes to its cell's owner. No section is materialised
    and no text is read.

    `t` is the scale, not a threshold. Small t keeps the answer to the star around the
    seeds; larger t lets it reach the document's global role. Both are true readings of
    the same propagator at different scales, which is why the caller picks rather than
    the library. It applies to `propagator="rl4"` only.

    `propagator` picks WHICH operator carries the signal, and the two are different
    readings rather than one made faster:

      "mass"      the EDGE-PRIMARY reading. A relation answers with what its own
                  boundary column carries from the seeds, `SUM over seeds v in e of
                  |B[v,e]|/deg[v]`, and a section sums the relations it owns. One hop,
                  read where the data is, and exact: every quantity is rational and
                  `rexgraph.core._exact_ratio` evaluates it over the integers.
      "boundary"  L0 = B1 B1^T applied MATRIX-FREE, never formed. The short-time moment:
                  the seeded mass lands on exactly the relations that name the query's
                  terms and is read back at the VERTICES they bound.
      "rl4"       S0 = B1 f(RL4) B1^T through `propagate_signal`, the full edge-space
                  relational operator and the script-15 scale bridge. `f` is a matrix
                  exponential, so this reading is transcendental and no arithmetic makes
                  it exact; the other two are ratios of integers.

    The two hops are what separate them. A vertex is the boundary of the relations
    rather than the thing they carry, so `|B(B^T x)|` reads a derived object and
    compounds its denominators through pairs, which is why it is float where the rest of
    this module is exact. Measured on 193 queries lifted from 10 Gutenberg books, one
    query per sampled section, `owner` supplied to both so neither pays to derive it:

        mass        top-1 61.7%   median rank  1    3.9 ms
        boundary    top-1 10.9%   median rank 23    2.1 ms

    `mass` is the default. It costs a little more per call and localises five times as
    often, and it is the reading that is exact.

    The unsigned total is also what survives an evenly covered column, where `B^T x` is
    exactly zero and a signed reading shuts precisely where a section is most
    distinctive.

    The default is "boundary" because it was MEASURED equal and is not close on cost. On
    46 identical queries over 10 Gutenberg documents, both read 97.8% top-1 and 100%
    top-5 with median rank 1, in 115.4 s against 0.1 s: a factor of 1154. RL4 is not
    affordable at document scale for a reason the structure explains: two relations
    co-participate when they share a vertex, and a common word puts most spans in contact
    with most others, so RL4 carries 15 to 58 MILLION nonzeros at nE 7,000 to 17,000.
    "rl4" stays available because it reads the four channels and this task does not
    exercise them.

    A second boundary step was tried and is WORSE, not better: 0.0% top-1 against 97.8%.
    The same thing happens one grade up on the corpus index. One application is the
    reading; further ones smear it.

    `seed_weight="invdeg"` is what makes the reading work, and it is a structural
    statement rather than a tuned one. A vertex's degree is how many relations it
    participates in, so a word appearing in 952 of a book's 1,469 sentences says nothing
    about WHERE an answer is; weighting each seed by `1/deg` lets a diffuse participant
    contribute diffusely. That is inverse document frequency derived from the complex's
    own incidence instead of imported as a corpus statistic, and no vertex is excluded.

    The accumulation is a SUM and there is no averaging option, because the sum is the
    field integrated over the section and that is the exact quantity. A mean would be a
    statistic standing in for a reading the field already gives: charge accumulates over
    a region, it is not averaged over it. Measured, ranking the section a query was
    lifted FROM over 20 queries on one book, the sum is also simply correct:

        flat seeding, sum       median rank  38-42     top-1   0%   top-10   0%
        invdeg,       sum       median rank   2        top-1  50%   top-10 100%

   , and normalising by incidence count returned every three-word line in the book. I
    had read the raw sum's apparent size-bias as the defect and normalised; the bias was
    in the SEEDS, and it left with them. `t` moved none of this (0.05, 0.3 and 1.0 rank
    identically), so scale was never what was wrong either.

    `channels=True` returns the response RESOLVED INTO THE FOUR CHANNELS instead of
    summed into one number: `(n_sections, 4)` over `rex.structural_character`'s
    (topology, geometry, frustration, co-participation), plus their names. A section then
    answers with a PROFILE rather than a scalar, which is the difference between "this
    section responds 0.42" and "this section responds, and it responds by
    co-participation rather than by topology".

    The scalar is the profile summed over its channels, so nothing is added by asking for
    it: what is added is not having thrown the axes away. `chi` is a property of the
    DOCUMENT and not of the query, exact rationally (`rational_trig.exact_character`) and
    O(nnz) as floats, so it costs one cached read per document rather than a solve.

    Returns `(scores, labels)`, or `(profiles, labels, channel_names)` with `channels`.
    """
    import numpy as np

    from rexgraph.core._sparse import to_scipy_csr

    seeds = np.asarray(seeds, dtype=int).ravel()
    nV, nE = int(rex.nV), int(rex.nE)
    labels = list(getattr(sections, "labels", []) or [])
    if owner is None:
        owner = np.asarray(sections.owner_cochain(nE), dtype=np.int64)
        n_sections = len(sections) if n_sections is None else n_sections
    n_sections = int(n_sections if n_sections is not None else (owner.max() + 1))
    out = np.zeros(max(n_sections, 0), dtype=np.float64)
    if seeds.size == 0 or nV == 0 or nE == 0 or n_sections == 0:
        return out, labels

    ind = np.zeros(nV, dtype=np.float64)
    ok = seeds[(seeds >= 0) & (seeds < nV)]
    if ok.size == 0:
        return out, labels
    if str(seed_weight) == "invdeg":
        deg = np.asarray(rex.degree, dtype=np.float64)
        ind[ok] = 1.0 / np.maximum(deg[ok], 1.0)
    else:
        ind[ok] = 1.0
    if str(propagator) == "mass":
        got = _mass_exact(rex, ok, owner, n_sections)
        if got is not None:
            if not channels:
                return got, labels
        # the profile carries the same per relation reading through the character
        return _mass_channels(rex, ok, owner, n_sections, labels)
    if str(propagator) == "rl4":
        resp = np.abs(np.asarray(rex.propagate_signal(ind, mode="heat", t=float(t)),
                                 dtype=np.float64).ravel())
    else:
        Bc = to_scipy_csr(rex._B1_dual).tocsr()
        resp = np.abs(Bc @ (Bc.T @ ind))       # L0 applied, never formed

    # THE MAGNITUDE IS TAKEN AT THE VERTEX, and that is not a detail.
    #
    # Deferring it to the section, so head and argument contributions cancel first, reads
    # better on a corpus sample: 100.0% top-1 against 94.0% over 50 queries on 10
    # documents, and is WRONG, because a zero-sum column passes nothing when its support
    # is seeded uniformly: `B^T x = x . sum(column) = 0`, exactly. The gate shuts.
    #
    # That is worst precisely where a section is most distinctive. A section whose terms
    # appear nowhere else has degree 1 throughout, so `1/deg` seeding IS uniform and its
    # own column cancels to zero; the signed reading then scores it near the floor. The
    # corpus sample hid this because repeated vocabulary makes the degrees uneven and lets
    # residue leak through. `test_section_response_finds_the_section_a_query_was_lifted_from`
    # does not hide it: the source section fell to 0.154 against another section's 0.846.
    #
    # So the cancellation is real and it is informative: it means the query covers that
    # column evenly, but it is not a ranking signal, and the vertex magnitude is what
    # survives it.

    B = to_scipy_csr(rex._B1_dual).tocoo()
    # one pass over the incidences: each (vertex, relation) hands its response to the
    # section that owns the relation. A cell with no owner (-1) contributes nothing.
    cell_owner = owner[B.col]
    keep = cell_owner >= 0
    if not channels:
        np.add.at(out, cell_owner[keep], resp[B.row[keep]])
        return out, labels

    chi = np.asarray(rex.structural_character, dtype=np.float64)
    if chi.ndim != 2 or chi.shape[0] != nE:
        np.add.at(out, cell_owner[keep], resp[B.row[keep]])
        return out, labels, []
    names = list(getattr(rex, "character_channels", None)
                 or ["topology", "geometry", "frustration", "coparticipation"])
    prof = np.zeros((max(n_sections, 0), chi.shape[1]), dtype=np.float64)
    # each incidence hands its response to the owning section THROUGH the relation's
    # channel profile, so the axes survive the accumulation instead of collapsing in it
    mass = resp[B.row[keep]]
    cols = B.col[keep]
    for k in range(chi.shape[1]):
        np.add.at(prof[:, k], cell_owner[keep], mass * chi[cols, k])
    return prof, labels, names[:chi.shape[1]]
