"""What differs between two boundary tensors, entry by entry, and what merging costs.

Betti numbers are not a diff. Two complexes can agree on every one of them and disagree in
every entry, because topology is a quotient and the entries are the representative. So the
comparison is taken on the signed tensor itself, where it decomposes exactly into the
trichotomy the model is built on:

    existence     a relation reaches a vertex in one tensor and not in the other, so the
                  entry is 0 against +-1. This is the sparsity pattern differing.
    orientation   both reach it and disagree on sense, so the entry is +1 against -1. On
                  the canonical column that is the distinguished vertex moving.
    share         1/(k-1) from the span width, carrying no information the support does
                  not already carry, so it never appears as a difference of its own.

**This is legitimate at grade 1 and NOT at grade 2.** At grade 1 the column is canonical,
`(-1, +share, ...)` with the distinguished vertex at position 0, so there is no gauge
freedom left and an entry-wise reading means what it appears to mean. A solved face column
is determined only up to overall sign and `solve_face_column` returns the leading-positive
representative because something has to be returned, so comparing face entries measures the
representative rather than the complex: negating one column moves a raw sign product
without changing a single cell. Above grade 1 the invariant is the holonomy, and
`grade2_diff` reads that instead. Asking for an entry-wise diff above grade 1 raises rather
than returning a number that would be about the arithmetic.

The merge preview is the other half, and it is JOINT rather than a sum of parts. Every
novel relation on its own either lies outside `range(B_1)` of the reference, so absorbing
it raises the rank, or lies inside it, so exactly one cycle appears (Theorem 53). Those
marginals are each exact and they DO NOT ADD UP, because absorbing one relation changes
the span the next is judged against: measured on 120 Complex Portal relations the
marginals totalled 120 while the actual merge moved the rank by 108 and opened 12 cycles.
So `rank_delta` and `cycle_delta` are computed on the augmented operator in one rank, the
marginals are reported beside them under their own names, and `marginals_sum_to_joint`
says whether the two happened to agree.
"""
from __future__ import annotations

import numpy as np

__all__ = ["tensor_diff", "grade2_diff", "difference_tensor", "format_diff"]


def _keys(rex, vmap=None):
    """Support and oriented keys per relation.

    The support key is the vertex set, so it answers "is this the same relation". The
    oriented key keeps position 0 apart from the rest, so two relations over the same
    vertices that disagree on the distinguished one differ here and not there. That is
    the existence/orientation split, taken on the keys rather than reconstructed later.
    """
    sup, ori, missing = [], [], []
    # relation_supports() is the arity-general read of the complex and it keeps the
    # distinguished vertex first, which is the orientation. Walking ptr/idx by hand here
    # rebuilt what it already returns.
    for e, span in enumerate(rex.relation_supports()):
        span = [int(v) for v in span]
        if vmap is not None:
            mapped = [vmap.get(v) for v in span]
            if any(m is None for m in mapped):
                sup.append(None); ori.append(None); missing.append(e)
                continue
            span = mapped
        sup.append(frozenset(span))
        ori.append((span[0], frozenset(span[1:])))
    return sup, ori, missing


def tensor_diff(ref, inp, *, ref_labels=None, inp_labels=None, grade=1,
                merge_preview=True):
    """Entry-wise diff of `inp` against `ref` at grade 1, plus what merging would cost.

    Vertices are aligned by label when both label lists are given, and by index otherwise.
    An input relation touching a vertex the reference does not have is `unmapped`: it is
    novel by existence at the vertex level, which is a stronger statement than being novel
    at the relation level, so it is counted separately.

    Returns a dict of exact integers plus, when `merge_preview` is on, the rank and cycle
    movement absorbing the input would cause.
    """
    if int(grade) != 1:
        raise ValueError(
            f"entry-wise diff is defined at grade 1 only, not grade {grade}. Above it the "
            f"column is determined up to sign, so an entry-wise reading measures the "
            f"representative and not the complex. Use grade2_diff, which reads holonomy.")

    vmap = None
    if ref_labels is not None or inp_labels is not None:
        if ref_labels is None or inp_labels is None:
            raise ValueError("give labels for both tensors or for neither")
        if len(ref_labels) != int(ref.nV) or len(inp_labels) != int(inp.nV):
            raise ValueError("each label list must have one entry per vertex")
        pos = {str(x): i for i, x in enumerate(ref_labels)}
        vmap = {i: pos[str(x)] for i, x in enumerate(inp_labels) if str(x) in pos}

    i_spans = [list(map(int, sp_)) for sp_ in inp.relation_supports()]
    rsup, rori, _ = _keys(ref)
    isup, iori, unmapped = _keys(inp, vmap)

    # MULTISET matching, because parallel relations are real cells. Two relations over
    # the same vertices are two columns and are exactly what carries a cycle without
    # raising rank, so matching by support alone and taking the first would drop every
    # extra copy. Counting instead needs no arbitrary pairing: within one support,
    # relations with the same distinguished vertex are interchangeable, so pairing them
    # in any order gives the same answer, and the leftovers pair off as reorientations
    # before anything is called novel.
    from collections import Counter

    ref_ori_count = Counter(rori)
    ref_by_sup = Counter(rsup)

    identical, reoriented, novel = [], [], []
    avail_ori = Counter(ref_ori_count)
    avail_sup = Counter(ref_by_sup)
    for e, (s, o) in enumerate(zip(isup, iori, strict=True)):
        if s is None or avail_sup[s] <= 0:
            novel.append(e)
            continue
        avail_sup[s] -= 1
        if avail_ori[o] > 0:
            avail_ori[o] -= 1
            identical.append(e)
        else:
            reoriented.append(e)

    # the entries themselves. A reoriented relation disagrees in exactly two places: the
    # reference's distinguished vertex, and the input's.
    orientation_entries = 2 * len(reoriented)
    existence_entries = sum(len(i_spans[e]) for e in novel)
    # what the reference holds and the input did not account for, counted the same way
    only_ref = int(sum((ref_by_sup - Counter(s for s in isup if s is not None)).values()))

    nvr, nvi = int(ref.nV), int(inp.nV)
    shared_v = len(vmap) if vmap is not None else min(nvr, nvi)
    out = {
        "grade": 1,
        "n_ref": int(ref.nE), "n_input": int(inp.nE),
        "identical": len(identical), "reoriented": len(reoriented),
        "novel": len(novel), "only_in_reference": only_ref,
        "unmapped_relations": len(unmapped),
        "shared_vertices": int(shared_v),
        "vertices_only_in_input": int(nvi - shared_v),
        "existence_entries": int(existence_entries),
        "orientation_entries": int(orientation_entries),
        "reoriented_ids": reoriented, "novel_ids": novel,
    }
    denom = out["identical"] + out["reoriented"] + out["novel"]
    out["agreement"] = out["identical"] / denom if denom else float("nan")

    if merge_preview and novel:
        from rexgraph.partition import candidate_readings
        cands, keep = [], []
        for e in novel:
            if e in unmapped:
                continue
            span = i_spans[e]
            sup = [vmap[v] for v in span] if vmap is not None else list(span)
            if max(sup) < nvr and len(set(sup)) >= 2:
                cands.append(sorted(set(sup))); keep.append(e)
        read = candidate_readings(ref, cands) if cands else []
        adds = [keep[i] for i, r in enumerate(read) if r["spans_new"]]
        closes = [keep[i] for i, r in enumerate(read) if not r["spans_new"]]

        # THE MARGINALS DO NOT SUM, so the joint move is computed jointly. Each
        # candidate reading is taken against the reference as it stands, but absorbing
        # one relation changes the span the next is judged against, so adding them up
        # over-counts: on 120 Complex Portal relations the marginals totalled 120 while
        # the merge moved the rank by 108 and opened 12 cycles. One rank on the
        # augmented operator is exact and costs no more than the marginals did.
        rank_delta = cycle_delta = 0
        if cands:
            import scipy.sparse as sp

            from rexgraph.graded_boundary import _sparse_rank
            Bint = ref._integer_B1().tocsc()
            cols = []
            for sup in cands:
                k = len(sup)
                col = np.zeros(nvr)
                col[sup] = 1.0
                col[sup[0]] = -(k - 1)
                cols.append(col)
            aug = sp.hstack([Bint, sp.csc_matrix(np.asarray(cols).T)]).tocsc()
            rank_delta = int(_sparse_rank(aug)) - int(_sparse_rank(Bint))
            # nE grows by every absorbed relation and beta_1 = nE - rank
            cycle_delta = len(cands) - rank_delta

        out["merge"] = {
            "evaluated": len(cands),
            "rank_delta": rank_delta, "cycle_delta": cycle_delta,
            "marginal_adds_a_direction": len(adds),
            "marginal_closes_a_cycle": len(closes),
            "not_evaluable": len(novel) - len(cands),
            "marginal_adds_ids": adds, "marginal_closes_ids": closes,
            "marginals_sum_to_joint": len(adds) == rank_delta,
        }
    return out


def grade2_diff(ref, inp):
    """Compare two complexes at grade 2 by HOLONOMY, which is what survives the gauge.

    A face column is fixed only up to sign, so an entry-wise reading is about the
    representative. The holonomy of a closed loop of faces is not: each face appears
    twice in the loop and its sign cancels, so it is a property of the complex.

    Returns each side's loop count, frustrated count and orientability, and whether the
    two agree on being orientable. It does not attempt to match individual faces, because
    matching representatives is the thing this function exists to avoid.
    """
    from rexgraph.faces import orientation_holonomy

    a = orientation_holonomy(ref, grade=2)
    b = orientation_holonomy(inp, grade=2)

    def _pack(h):
        return {"n_cells": h.get("n_cells", 0), "n_loops": h.get("n_loops", 0),
                "frustrated": h.get("frustrated", 0),
                "orientable": h.get("orientable"),
                "rate": float(h["rate"]) if h.get("rate") is not None else None}
    ra, rb = _pack(a), _pack(b)
    return {
        "grade": 2, "reference": ra, "input": rb,
        "same_orientability": ra["orientable"] == rb["orientable"],
        "frustration_delta": (rb["rate"] - ra["rate"])
        if (ra["rate"] is not None and rb["rate"] is not None) else None,
        "reading": ("holonomy, not entries: a face column is fixed up to sign, so only "
                    "the sign product around a closed loop of faces is a property of the "
                    "complex"),
    }


def format_diff(d) -> str:
    """The diff as plain sentences, stating counts and no verdicts."""
    if d.get("grade") == 2:
        r, i = d["reference"], d["input"]
        s = [f"At grade 2 the reference has {r['n_cells']} faces over {r['n_loops']} "
             f"independent loops and the input has {i['n_cells']} over {i['n_loops']}."]
        s.append(f"Reference orientable: {r['orientable']}; input: {i['orientable']}.")
        if d["frustration_delta"] is not None:
            s.append(f"Frustrated fraction moves by {d['frustration_delta']:+.4f}.")
        return " ".join(s)
    L = [f"Of {d['n_input']} input relations, {d['identical']} are already in the "
         f"reference exactly, {d['reoriented']} share their support but disagree on the "
         f"distinguished vertex, and {d['novel']} are new."]
    L.append(f"That is {d['orientation_entries']} entries differing by orientation and "
             f"{d['existence_entries']} by existence.")
    if d["vertices_only_in_input"]:
        L.append(f"{d['vertices_only_in_input']} vertices appear only in the input.")
    if d["only_in_reference"]:
        L.append(f"{d['only_in_reference']} reference relations have no counterpart.")
    m = d.get("merge")
    if m and m.get("evaluated"):
        L.append(f"Absorbing the input would raise rank(B1) by {m['rank_delta']} and add "
                 f"{m['cycle_delta']} cycles.")
        if not m["marginals_sum_to_joint"]:
            L.append(f"Taken one at a time {m['marginal_adds_a_direction']} of them would "
                     f"each add a direction, so the relations overlap and the marginals "
                     f"do not sum to the merge.")
        if m.get("not_evaluable"):
            L.append(f"{m['not_evaluable']} novel relations could not be previewed "
                     f"because they reach vertices the reference does not have.")
    return " ".join(L)


def difference_tensor(ref, inp, *, ref_labels=None, inp_labels=None, verify=True):
    """`D = B_inp - B_ref`, aligned on the union of supports, as an OPERATOR.

    The delta is not a report about two complexes, it is a complex. Zero column sum is
    the condition that makes a column a boundary, and it is preserved under subtraction
    because zero-sum vectors form a linear subspace. So `D` is itself a boundary tensor,
    it has its own `L = D D^T` with the constant vector in the kernel, and every reading
    in the library applies to it.

    Two of its subspaces already mean something without any further work::

        ker by column   a zero column is a relation the two tensors agree on exactly
        support         the columns that moved, which is the disagreement

    so agreement and disagreement are a kernel and a support rather than counters.

    Columns are the union of the two supports: present in both, the difference of the
    columns; present in one, that column signed by which side it came from. Vertices
    align by label when both label lists are given and by index otherwise.

    Parallel relations survive. Pairing is a MULTISET match per support: copies with the
    same distinguished vertex are interchangeable so they cancel in any order, the
    leftovers pair off as reorientations, and only what remains unmatched becomes a column
    of its own. Nothing is dropped for sharing a vertex set.

    Returns `(D, readings)` with `D` a `scipy.sparse` matrix over the reference's vertex
    space. `verify=True` asserts the zero column sum, which is the theorem rather than a
    convention: a nonzero sum means the alignment dropped an entry.
    """
    import scipy.sparse as sp

    from rexgraph.graded_boundary import _sparse_rank

    vmap = None
    if ref_labels is not None or inp_labels is not None:
        if ref_labels is None or inp_labels is None:
            raise ValueError("give labels for both tensors or for neither")
        if len(ref_labels) != int(ref.nV) or len(inp_labels) != int(inp.nV):
            raise ValueError("each label list must have one entry per vertex")
        pos = {str(x): i for i, x in enumerate(ref_labels)}
        vmap = {i: pos[str(x)] for i, x in enumerate(inp_labels) if str(x) in pos}

    nV = int(ref.nV)
    rsup, rori_d, _ = _keys(ref)
    isup, iori_d, unmapped = _keys(inp, vmap)

    # rex.B1 is DENSE by design; the dual is the sparse object
    from rexgraph.core._sparse import to_scipy_csr
    Br = to_scipy_csr(ref._B1_dual).tocsc()
    Bi = to_scipy_csr(inp._B1_dual).tocsc()

    def _col(B, e, mapper):
        """One column as (rows, values), re-indexed into the reference's vertex space.

        Sparse triplets rather than a dense vector: a relation touches k vertices and
        the whole point of the boundary tensor is that k is small, so materialising nV
        zeros per column would make the difference denser than either argument.
        """
        lo, hi = B.indptr[e], B.indptr[e + 1]
        rows, vals = [], []
        for r, x in zip(B.indices[lo:hi], B.data[lo:hi], strict=True):
            t = mapper(int(r)) if mapper else int(r)
            if t is None or t >= nV:
                return None
            rows.append(t); vals.append(float(x))
        return np.asarray(rows, dtype=np.int64), np.asarray(vals, dtype=np.float64)

    # MULTISET pairing per support, so parallel relations survive. Within one support
    # the copies with the same distinguished vertex are interchangeable, so they pair in
    # any order and the answer does not depend on which; the leftovers then pair off as
    # reorientations, and only what is still unmatched becomes a column of its own.
    rsup_ori, isup_ori = {}, {}
    for e, (sk, ok) in enumerate(zip(rsup, rori_d, strict=True)):
        rsup_ori.setdefault(sk, []).append((ok, e))
    for e, (sk, ok) in enumerate(zip(isup, iori_d, strict=True)):
        if sk is not None:
            isup_ori.setdefault(sk, []).append((ok, e))

    order = list(rsup_ori)
    order += [sk for sk in isup_ori if sk not in rsup_ori]

    mapper = (lambda v: vmap.get(v)) if vmap is not None else None
    rows_acc, cols_acc, vals_acc = [], [], []
    ncol = only_ref = only_inp = both = paired_same = 0

    def _emit(parts):
        """Append one column to the triplets. `parts` is a list of (rows, vals, sign)."""
        nonlocal ncol
        for r, v, sg in parts:
            if r.size:
                rows_acc.append(r); vals_acc.append(sg * v)
                cols_acc.append(np.full(r.size, ncol, dtype=np.int64))
        ncol += 1
    for sk in order:
        r_here = list(rsup_ori.get(sk, []))
        i_here = list(isup_ori.get(sk, []))
        by_ori = {}
        for ok, e in r_here:
            by_ori.setdefault(ok, []).append(e)
        left_i = []
        for ok, e in i_here:                  # exact matches first: those cancel
            if by_ori.get(ok):
                by_ori[ok].pop()
                both += 1; paired_same += 1
                _emit([])                     # an exact match cancels: an empty column
            else:
                left_i.append(e)
        left_r = [e for v in by_ori.values() for e in v]
        for e_i in left_i:                    # leftovers pair as reorientations
            if left_r:
                e_r = left_r.pop()
                a = _col(Br, e_r, None); b = _col(Bi, e_i, mapper)
                _emit([] if (a is None or b is None)
                      else [(b[0], b[1], 1.0), (a[0], a[1], -1.0)])
                both += 1
            else:
                b = _col(Bi, e_i, mapper)
                if b is not None:
                    _emit([(b[0], b[1], 1.0)]); only_inp += 1
        for e_r in left_r:                    # reference relations with no counterpart
            a = _col(Br, e_r, None)
            if a is not None:
                _emit([(a[0], a[1], -1.0)]); only_ref += 1
    if rows_acc:
        D = sp.coo_matrix((np.concatenate(vals_acc),
                           (np.concatenate(rows_acc), np.concatenate(cols_acc))),
                          shape=(nV, ncol)).tocsc()
        D.sum_duplicates(); D.eliminate_zeros()
    else:
        D = sp.csc_matrix((nV, ncol))

    csum = np.asarray(D.sum(axis=0)).ravel() if D.shape[1] else np.zeros(0)
    if verify and csum.size and np.abs(csum).max() > 1e-9:
        raise ValueError(
            f"a column of the difference sums to {np.abs(csum).max():.3e} rather than "
            f"zero. Zero-sum is preserved under subtraction, so the alignment dropped an "
            f"entry rather than the data being unusual.")

    nz = np.diff(D.indptr)
    readings = {
        "shape": tuple(D.shape),
        "agree": int((nz == 0).sum()),
        "disagree": int((nz > 0).sum()),
        "in_both": both, "only_in_reference": only_ref, "only_in_input": only_inp,
        "unmapped_relations": len(unmapped),
        "matched_identical": paired_same,
        "max_column_sum": float(np.abs(csum).max()) if csum.size else 0.0,
        "rank": int(_sparse_rank(D)) if D.nnz else 0,
        "frobenius2": float(D.multiply(D).sum()),
    }
    readings["nullity"] = int(D.shape[1] - readings["rank"])
    return D, readings
