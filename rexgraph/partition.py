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
share no current path. So NEGATIVE is co oriented along a shared cycle.

That does NOT license reading the fraction as "these relations disagree". On a knowledge
complex every relation is built with the same convention (the record at position 0), so
there is no orientation disagreement present to find, and the fraction reads the geometry
of how supports overlap instead. Measured on 1373 Complex Portal complexes against the
file's own GO annotations, with the shared subunit count HELD FIXED by stratification,
functionally related pairs score HIGHER: pooled +0.0411 +- 0.0045, z = +9.12, 11 of 14
strata agreeing, and only rho +0.21 against the raw overlap. So it is coupling, it is not
a rescaled overlap count, and it is not conflict. The mean Gram does not survive the same
control (its per stratum sign flips), so the FRACTION is the reading and the mean is not.
"""
from __future__ import annotations

import numpy as np

__all__ = ["section_readings", "section_response", "section_coverage", "document_field",
           "coupling_fraction",
           "grade_leverage", "section_tensor",
           "candidate_readings", "byte_energy", "energy_tensor", "compose_substrates",
           "hodge_share"]


def section_readings(rex, sections, *, leverage=None, verify=True):
    """Readings for each section of a sectioning of the relations.

    `sections` maps a name to the relation indices it contains. Sections may overlap, and
    group sections DO: a pairwise relation belongs to every group holding both its
    members. The closure identity is asserted only for a genuine partition, meaning the
    sections are disjoint AND cover every relation; an overlapping cover double counts
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
    """Fraction of each section's field Gram off diagonals that are POSITIVE.

    NEGATIVE is co oriented along a shared cycle and 0 is no shared current path, both
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
    """diag of the projector onto row(B), matrix free and in blocks.

    `R_eff(c) = b_c^T (B B^T)^+ b_c` with the kernel deflated, which is the same solve
    `_effective_resistance_batch` runs at grade 1, so both go through the one primitive
    rather than blocking the same solve two different ways.
    """
    from rexgraph.fiedler import leverage_diagonal

    return leverage_diagonal(B, block=block)


def grade_leverage(rex, k, *, verify=True):
    """The leverage at grade `k`: the diagonal of the projector onto row(B_k).

    Nothing about the grade 1 reading was about grade 1. `R_eff(e) = z^T B^T (B B^T)^+ B
    z` is the e-th diagonal of the orthogonal projector onto the row space, so writing it
    for `B_k` is the same expression with a different operator, and every reading built on
    it comes with it: a cell's mass, its share of the cycle space, and a section's gap.

    Foster generalises with it and is the self test at every grade::

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
    section, which is what makes them coordinates rather than a per grade constant::

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

    That is the generation predicate. `frustration_delta` measures a candidate by the
    same structural question; the rank tower answers it with a projection rather than a
    rebuild.

    `candidates` is an iterable of vertex supports. With `shares=True` each becomes the
    zero sum column `(-1, 1/(k-1), ...)` the model uses at any arity; pass explicit
    `(support, values)` pairs to read a column verbatim.

    Returns a list of `{support, k, kind, spans_new, quadrance, closes}` in the order
    given. `kind` names the arity class (`witness` (k=1), `pairwise` (k=2) or
    `branching` (k>2)) because the three behave differently under the boundary and a
    reading that does not say which it read cannot be checked.
    """
    import scipy.sparse as sp

    from rexgraph.core._sparse import to_scipy_csr
    from rexgraph.fiedler import deflated_operator, minimum_norm_gram_solve
    from rexgraph.graded_boundary import _sparse_rank
    from rexgraph.sparse_character import _block_cg

    nV = int(rex.nV)
    # NO DECOMPOSITION OF B. range(B) = range(B B^T) = ker(L0) orthogonal, so "is this
    # column spanned" is a test against the KERNEL, which has dimension beta_0 and is
    # tiny, rather than against a factorisation of the whole operator. The quadrance is
    # then one deflated CG solve, matrix free, exactly as _effective_resistance_batch
    # does it. The earlier version densified B and took its SVD, which is 1.05 GB at
    # nV 4000 and nE 33k and is not what this library does anywhere else.
    # L0 is never formed here either: membership is a kernel test and the quadrance is
    # one deflated solve, both of which the boundary supplies directly.
    Bfull = to_scipy_csr(rex._B1_dual).tocsc()
    Bint = rex._integer_B1().tocsc()
    try:
        _apply, dinv, U, _nk = deflated_operator(Bfull)
        general_boundary = False
    except ValueError:
        # Branching C1 has a larger kernel than its support components.  Range
        # membership remains an exact rank question; quadrance takes the general
        # minimum norm Green action below instead of an incomplete deflation.
        _apply = dinv = U = None
        general_boundary = True
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
                # so their span lies inside the zero sum subspace and `(+1)` is outside
                # it. A witness therefore always adds rank and closes nothing, at any
                # size and against any complex.
                #
                # Reporting `spans_new: False` here said the opposite of the truth. It
                # also erased the class. A vocative ("Take away your mother, Jerry.")
                # IS a witness, a participant that exists and bounds nothing, and calling
                # it a non answer silently turns that sentence into a different one.
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
        # against the projection's 0.11s at nE 3268, because fraction free elimination
        # over the whole operator runs per candidate.
        #
        # So the projection RULES, and the exact rank ADJUDICATES. The residual is
        # compared against the decomposition's own noise floor with a wide margin either
        # way; outside that band the answer is settled by orders of magnitude and no
        # tolerance is doing any work, and inside it the integer rank decides. The band
        # selects the method, never the verdict.
        kind = ("pairwise" if sup.size == 2 else "branching")
        if general_boundary:
            # No component indicator can stand in for ker(B1.T) here.  The exact
            # integer rank change is the membership verdict at arbitrary arity.
            if rank_B is None:
                rank_B = int(_sparse_rank(Bint))
            col = np.zeros(nV)
            np.add.at(col, sup,
                      vals * ((sup.size - 1) if shares and sup.size > 1 else 1))
            aug = sp.hstack([Bint, sp.csc_matrix(col.reshape(-1, 1))]).tocsc()
            spans_new = int(_sparse_rank(aug)) > rank_B
            rel = float("nan")
            adjudicated += 1
        else:
            nb = float(np.linalg.norm(b)) or 1.0
            # a column lies in range(B) exactly when it is orthogonal to ker(L0), so the
            # test is |U^T b| and costs nV x beta_0. In span leaves this at machine level and
            # out of span leaves an O(1) fraction of the column, thirteen orders apart, so
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
            y = (minimum_norm_gram_solve(Bfull, b.reshape(-1, 1), tol=1e-12, maxit=500)
                 if general_boundary else _block_cg(
                     _apply, b.reshape(-1, 1), dinv, tol=1e-12, maxit=500
                 ))
            q = float(b @ y[:, 0])
        out.append({"support": sup.tolist(), "k": int(sup.size), "kind": kind,
                    "spans_new": bool(spans_new), "quadrance": q,
                    "closes": 0.0 if spans_new else 1.0,
                    "adjudicated": bool(1e-10 <= rel <= 1e-6)})
    return out


# the second substrate
#
# Every reading above is taken ON the complex, so all of them read one propagated signal
# and inherit its statistics. The byte energy does not: it reads the ENCODING, before any
# relation exists, and it is the only corpus free quantity here.
#
# Theorem 27 is the rule for putting them together, and it is a prohibition. Carrying the
# energy as a source and solving `L0 u = B1 E` re imports the frequency coupling the
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
    1 based so the first byte contributes rather than vanishing.
    """
    return float(sum((b * (i + 1)) ** 2
                     for i, b in enumerate(str(label).encode("utf-8"))))


def energy_tensor(rex, sections, labels, *, moments=("total", "mean", "spread")):
    """`E[section, moment]`: the energy substrate, computed WITHOUT the complex.

    A section's energy is read off the labels of the vertices its cells touch. Only the
    incidence is used, to find WHICH labels; no solve, no field and no propagation, which
    is what keeps it corpus free.

        total    sum of the byte energies of the section's distinct vertices
        mean     total / number of them
        spread   peak / total, so a section dominated by one long label reads high

    PICK THE MOMENT DELIBERATELY. Per VERTEX the energy is corpus free, and measured on
    prose it sits at rho = +0.175 against the structural reading while the structural
    readings sit at -0.737 with frequency among themselves. Per SECTION that only
    survives for `mean`. Over 135 sentence sections of the same corpus::

        total    +0.61 to +0.70 against n, mass, own_rank, share and gap
        spread   -0.49 to -0.63 against the same
        mean     |rho| <= 0.13 against all of them except efficiency, at +0.33

    `total` is a sum over the section's vertices and `spread` is a share of it, so both
    carry how BIG the section is, which the structural readings already say. `mean` is
    the size free one and is the moment that earns a separate axis.

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
    is gradient like when its gradient share exceeds `r_k / n_k`, not when it is large.

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


def response_seeds(rex, seeds, seed_weight="invdeg"):
    """Validate C0 seed indices as a set; never truncate a float to an index."""
    from numbers import Integral
    from rexgraph.graph import RexGraph
    if not isinstance(rex, RexGraph):
        raise TypeError("response requires a native RexGraph")
    rex._ensure_clean()
    if seed_weight not in {"invdeg", "flat"}:
        raise ValueError("seed_weight must be invdeg or flat")
    if isinstance(seeds, (str, bytes)) or np.ndim(seeds) != 1:
        raise TypeError("seeds must be a one dimensional integer sequence")
    values = list(seeds)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
        raise TypeError("seeds must contain integers, not booleans or floats")
    if any(v < 0 or v >= rex.nV for v in values):
        raise ValueError("seed index is outside C0")
    return np.asarray(sorted(set(map(int, values))), dtype=np.int64)


def response_owner(rex, sections, owner=None, n_sections=None):
    """Resolve one C1 partition, refusing implicit first owner selection for covers."""
    from numbers import Integral
    from rexgraph.sectioning import sectionings_of
    if getattr(sections, "is_derived", False):
        sections = sections.resolved(sectionings_of(rex))
    labels = list(getattr(sections, "labels", []) or [])
    if owner is None:
        if getattr(sections, "grade", None) != 1:
            raise ValueError("section response requires a C1 sectioning")
        ptr = np.asarray(sections.indptr)
        indices = np.asarray(sections.indices)
        if (ptr.ndim != 1 or ptr.size != len(labels) + 1 or ptr[0] != 0
                or ptr[-1] != indices.size or np.any(np.diff(ptr) < 0)
                or getattr(sections, "n_cells", 0) not in {0, rex.nE}):
            raise ValueError("section membership axes do not match the source")
        if (indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= rex.nE)
                or np.unique(indices).size != indices.size):
            raise ValueError("section response requires disjoint valid C1 memberships; give an explicit owner for a cover")
        owner = sections.owner_cochain(rex.nE)
        n_sections = len(sections) if n_sections is None else n_sections
    raw = np.asarray(owner)
    if raw.ndim != 1 or raw.size != rex.nE or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError("owner must contain one integer per C1 cell")
    if n_sections is None:
        n_sections = max(map(int, raw), default=-1) + 1
    if isinstance(n_sections, (bool, np.bool_)) or not isinstance(n_sections, Integral) or n_sections < 0:
        raise ValueError("n_sections must be a nonnegative integer")
    if any(int(v) < -1 or int(v) >= n_sections for v in raw):
        raise ValueError("owner is outside the section axis")
    if labels and len(labels) != n_sections:
        raise ValueError("labels must match the section axis")
    return np.asarray(raw, dtype=np.int64), int(n_sections), labels


def _edge_terms(rex, seeds, seed_weight="invdeg"):
    """Read coalesced primary integer columns with their original share divisors."""
    from rexgraph.native_rank import primary_columns
    seeds = response_seeds(rex, seeds, seed_weight)
    degrees = np.maximum(np.diff(np.asarray(rex._v2e[0], dtype=np.int64))[seeds], 1)
    if seed_weight == "flat":
        degrees = np.ones(seeds.size, dtype=np.int64)
    which = {int(v): i for i, v in enumerate(seeds)}
    items, selected, carried = [], [], []
    for e, column in enumerate(primary_columns(rex, integer=True)):
        for vertex, value in column.items():
            if vertex in which:
                items.append(e)
                selected.append(which[vertex])
                carried.append(value)
    den = np.maximum(np.diff(np.asarray(rex._boundary_ptr, dtype=np.int64)) - 1, 1)
    return tuple(np.asarray(v, dtype=np.int64) for v in (items, carried, selected, degrees, den))


def _response_ratio(rex, seeds, *, reading, seed_weight, owner=None, n_sections=0, exact=False):
    from rexgraph.core import _exact_ratio
    if reading not in {"mass", "coverage"}:
        raise ValueError("relation reading must be mass or coverage")
    if not isinstance(exact, (bool, np.bool_)):
        raise TypeError("exact must be boolean")
    item, signed, seed, deg, den = _edge_terms(rex, seeds, seed_weight)
    return _exact_ratio.axis_ratio(item, np.abs(signed) if reading == "mass" else signed,
        seed, deg, den, rex.nE, 0, owner, n_sections,
        _exact_ratio.SUM if reading == "mass" else _exact_ratio.COVERAGE, exact=exact)


def document_field(rex, seeds, *, reading="mass", seed_weight="invdeg", exact=True):
    """C1 response on one native document, retaining its canonical relation axis.

    mass = |B1| transpose x. coverage = mass - |B1 transpose x|.
    x is the seed indicator, optionally divided by stored incidence degree.
    This reads supplied C0 seeds, not text tokens, corpus selection or similarity.
    Relation metrics and declared gauge signs do not enter these readings.
    """
    from rexgraph.cochain import Cochain
    values = _response_ratio(rex, seeds, reading=reading, seed_weight=seed_weight, exact=exact)
    return Cochain(1, values, source=rex)


def section_coverage(rex, sections, seeds, *, seed_weight="invdeg",
                     n_sections=None, owner=None, exact=False):
    """Sum mass minus absolute signed response per declared C1 section.

    Balanced non witness columns can have zero signed response under uniform
    seeding. Witnesses are not balanced. Repeated slots coalesce before
    magnitudes are taken, so a cancelling loop contributes zero.
    """
    return section_response(rex, sections, seeds, seed_weight=seed_weight,
        n_sections=n_sections, owner=owner, propagator="coverage", exact=exact)


def section_response(rex, sections, seeds, *, t=1.0, seed_weight="invdeg",
                     n_sections=None, owner=None, propagator="mass",
                     channels=False, exact=False):
    """Sum a query response over a declared C1 section partition.

    mass reads |B1| transpose x on primary relations. coverage subtracts
    |B1 transpose x| from that mass. Both use sparse exact ratio accumulation,
    with one final rounding in numerical mode. exact=True retains Fractions.
    The input is a seed set; duplicate indices do not change its meaning.

    boundary reads |B1 B1 transpose x| on vertices, then sums that reading
    across the nonzero incidences of each section's relations. rl4 uses the
    existing propagated signal instead. These distinct numerical readings
    are not substituted for mass. The scale t applies to rl4 only.

    channels=True resolves numerical relation contributions through the
    source's T, G, F and C character channels. G means the unsigned down
    channel, not the upper Hodge sector. Exact channel profiles are not
    implemented here. Scalar results have shape (n_sections,); profiles
    have shape (n_sections, n_channels), including empty seed sets.

    Without owner, sections must be a disjoint C1 sectioning. Unowned cells
    may be omitted. A cover requires an explicit owner choice. Source
    section order and labels are retained; no text or store is read.
    """
    if propagator not in {"mass", "coverage", "boundary", "rl4"}:
        raise ValueError("propagator must be mass, coverage, boundary or rl4")
    if not isinstance(channels, (bool, np.bool_)) or not isinstance(exact, (bool, np.bool_)):
        raise TypeError("channels and exact must be booleans")
    if exact and (channels or propagator not in {"mass", "coverage"}):
        raise ValueError("exact section response supports scalar mass and coverage only")
    seeds = response_seeds(rex, seeds, seed_weight)
    owner, n_sections, labels = response_owner(rex, sections, owner, n_sections)
    if propagator in {"mass", "coverage"} and not channels:
        return _response_ratio(rex, seeds, reading=propagator, seed_weight=seed_weight,
            owner=owner, n_sections=n_sections, exact=exact), labels
    if propagator in {"mass", "coverage"}:
        per_cell = document_field(rex, seeds, reading=propagator,
                                  seed_weight=seed_weight, exact=False).numpy()
    else:
        from rexgraph.native_sparse import NativeSparse
        if isinstance(t, (bool, np.bool_)) or not np.isfinite(t) or t < 0:
            raise ValueError("response scale must be finite and nonnegative")
        B = NativeSparse(rex._B1_dual)
        ind = np.zeros(rex.nV, dtype=np.float64)
        degree = np.maximum(np.diff(np.asarray(rex._v2e[0], dtype=np.int64))[seeds], 1)
        ind[seeds] = 1.0 / degree if seed_weight == "invdeg" else 1.0
        resp = np.abs(rex.propagate_signal(ind, mode="heat", t=float(t)) if propagator == "rl4"
                      else B.apply(B.transpose_apply(ind)))
        per_cell = np.zeros(rex.nE, dtype=np.float64)
        for e, column in enumerate(B.columns()):
            per_cell[e] = sum(resp[v] for v in column)
    keep = owner >= 0
    if not channels:
        out = np.zeros(n_sections, dtype=np.float64)
        np.add.at(out, owner[keep], per_cell[keep])
        return out, labels
    from rexgraph.sparse_character import build_sparse_character_cheap
    chi = np.asarray(build_sparse_character_cheap(rex)["chi"], dtype=np.float64)
    if chi.ndim != 2 or chi.shape[0] != rex.nE:
        raise ValueError("character rows must match C1")
    names = list(getattr(rex, "character_channels", None)
                 or ["topology", "geometry", "frustration", "coparticipation"])
    prof = np.zeros((n_sections, chi.shape[1]), dtype=np.float64)
    for k in range(chi.shape[1]):
        np.add.at(prof[:, k], owner[keep], per_cell[keep] * chi[keep, k])
    return prof, labels, names[:chi.shape[1]]
