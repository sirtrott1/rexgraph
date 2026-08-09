"""Joining two relational complexes, at any arity.

Two complexes are joined through a vertex identification: a correspondence saying which
cell of one is which cell of the other. What is joined is the RELATIONS, since they are
primitive and the vertices are their boundary, so the question a join answers is which
relations both complexes carry, not which rows match.

A relation's identity is its oriented support. `B_1`'s column is `-1` at the
distinguished vertex and `1/(k-1)` on the rest, so two relations are the same relation
exactly when they distinguish the same vertex and reach the same others. That is read
off the boundary structure, which makes it arity-general: a branching relation is
matched as one relation of arity k, not as a set of pairs.

`core._joins` is the pairwise dense oracle, and it is not what this goes through. It
takes dense `B_1`, scans every vertex per edge to find endpoints, and identifies them
with `abs(B1[v, e]) > 0.5`. A branching column carries `1/(k-1)`, which is exactly 0.5
at k=3 and smaller above, so that test sees only the distinguished vertex and every
share is invisible to it: a k-ary relation reads as touching one vertex. Measured at
k = 2, 3, 4, 5.

The chain condition is preserved rather than assumed. A face survives only when every
relation it bounds survives, so its column remaps intact, and `B_1 B_2 = 0` continues to
hold because both operators were restricted together.
"""

from __future__ import annotations

import numpy as np

__all__ = ["vertex_correspondence", "relation_key", "join", "HOW"]

#: the join kinds, by which relations they keep
HOW = ("inner", "left", "outer")


def _columns(rex):
    """Each relation's boundary as `(distinguished_vertex, tuple(other_vertices))`.

    Read off the boundary CSR, so arity is whatever the complex has. The first index of
    a column is the distinguished one, matching how `B_1` is stored.
    """
    rex._ensure_clean()
    bp, bi = rex._boundary_ptr, rex._boundary_idx
    if bp is None:
        src, tgt = rex._ensure_src_tgt()
        return [(int(s), (int(t),)) for s, t in zip(src, tgt, strict=True)]
    bp = np.asarray(bp)
    bi = np.asarray(bi)
    out = []
    for e in range(int(rex.nE)):
        span = [int(v) for v in bi[bp[e]:bp[e + 1]]]
        out.append((span[0], tuple(span[1:])) if span else (-1, ()))
    return out


def vertex_correspondence(labels_r, labels_s) -> dict:
    """Which vertex of R is which vertex of S, by label.

    Returns `{v_r: v_s}` for the labels both carry. A label appearing more than once on
    a side is declined rather than resolved: identifying two distinct cells because they
    share a name is how a join silently merges things that were never the same.
    """
    from collections import Counter

    count_s = Counter(str(x) for x in labels_s)
    count_r = Counter(str(x) for x in labels_r)
    index_s = {str(x): i for i, x in enumerate(labels_s)}
    out = {}
    for i, lab in enumerate(labels_r):
        key = str(lab)
        if count_r[key] == 1 and count_s.get(key, 0) == 1:
            out[i] = index_s[key]
    return out


def relation_key(distinguished, others, remap=None):
    """A relation's identity in a shared namespace, or None if it does not translate.

    Orientation is part of the identity: at arity k the distinguished vertex is a k-way
    choice, and two relations over the same vertices that distinguish different ones are
    different relations. The rest is unordered, since the share is uniform across it.
    """
    if remap is None:
        return (int(distinguished), frozenset(int(v) for v in others))
    if distinguished not in remap:
        return None
    mapped = []
    for v in others:
        if v not in remap:
            return None
        mapped.append(int(remap[v]))
    return (int(remap[distinguished]), frozenset(mapped))


def join(rex_r, rex_s, *, how: str = "inner", labels_r=None, labels_s=None,
         correspondence=None):
    """Join two complexes through a vertex identification.

    `inner` keeps the relations both carry, `left` keeps all of R plus S's relations that
    lie entirely on identified vertices, `outer` keeps everything with identified
    vertices merged.

    Returns `(rex, report)`. The report says how many relations each side contributed and
    how many were shared, because a join that silently produced nothing and a join that
    produced nothing because nothing matched look identical from the result alone.
    """
    from rexgraph.graph import RexGraph

    if how not in HOW:
        raise ValueError(f"how must be one of {HOW}, got {how!r}")

    if correspondence is None:
        if labels_r is None or labels_s is None:
            raise ValueError(
                "give either an explicit correspondence or labels for both complexes")
        correspondence = vertex_correspondence(labels_r, labels_s)

    cols_r, cols_s = _columns(rex_r), _columns(rex_s)

    # One vertex namespace. R keeps its numbering; an S vertex is R's where they are
    # identified and a fresh index otherwise, so an inner join never invents a vertex
    # and an outer one never collapses two.
    nV_r = int(rex_r.nV)
    s_to_joint = dict(correspondence.items())
    s_to_joint = {int(v_s): int(v_r) for v_r, v_s in correspondence.items()}
    next_index = nV_r
    if how == "outer":
        for v_s in range(int(rex_s.nV)):
            if v_s not in s_to_joint:
                s_to_joint[v_s] = next_index
                next_index += 1

    keys_r = {relation_key(d, o): e for e, (d, o) in enumerate(cols_r)}
    keys_s_joint = {}
    for e, (d, o) in enumerate(cols_s):
        k = relation_key(d, o, remap=s_to_joint)
        if k is not None:
            keys_s_joint.setdefault(k, e)

    shared = set(keys_r) & set(keys_s_joint)
    if how == "inner":
        kept = [e for e, (d, o) in enumerate(cols_r) if relation_key(d, o) in shared]
        from_s = []
    else:
        # left and outer keep all of R and add S's relations R does not already carry.
        # They differ in the VERTEX namespace, decided above: outer gives an unidentified
        # S vertex a fresh index, left does not, and `relation_key` returns None for a
        # relation touching a vertex it cannot translate. So `keys_s_joint` is already
        # restricted to the relations lying entirely on identified vertices, which is what
        # `left` is defined to add. It was adding none, making it a synonym for `inner`
        # while the docstring said otherwise.
        kept = list(range(len(cols_r)))
        from_s = [e for k, e in keys_s_joint.items() if k not in keys_r]

    # rebuild the boundary, arity-general
    ptr, idx = [0], []
    for e in kept:
        d, o = cols_r[e]
        idx.extend([d, *o])
        ptr.append(len(idx))
    for e in from_s:
        d, o = cols_s[e]
        idx.append(s_to_joint[d])
        idx.extend(s_to_joint[v] for v in o)
        ptr.append(len(idx))

    n_vertices = max(next_index, nV_r)
    if not idx:
        joined = RexGraph(sources=np.zeros(0, np.int32), targets=np.zeros(0, np.int32))
    else:
        joined = RexGraph.from_hypergraph(np.asarray(ptr, np.int32),
                                          np.asarray(idx, np.int32))
        if n_vertices > joined.nV:
            joined._nV = int(n_vertices)

    faces_kept = _carry_faces(rex_r, kept, joined)
    return joined, {
        "how": how,
        "identified_vertices": len(correspondence),
        "relations_r": len(cols_r), "relations_s": len(cols_s),
        "shared_relations": len(shared),
        "kept_from_r": len(kept), "kept_from_s": len(from_s),
        "faces_carried": faces_kept,
        "nV": int(joined.nV), "nE": int(joined.nE), "nF": int(joined.nF),
    }


def _carry_faces(rex_r, kept, joined) -> int:
    """Bring R's faces across, but only the ones whose whole boundary survived.

    A face over a relation that was not kept has nothing to bound, so carrying it would
    break `B_1 B_2 = 0`. Restricting both operators together is what keeps the result a
    complex rather than a pair of arrays that used to be one.
    """
    rex_r._ensure_clean()
    n_faces = int(getattr(rex_r, "_nF", 0) or 0)
    if n_faces == 0 or not kept:
        return 0
    position = {int(e): i for i, e in enumerate(kept)}
    cp = np.asarray(rex_r._B2_col_ptr)
    ri = np.asarray(rex_r._B2_row_idx)
    vals = np.asarray(rex_r._B2_vals)

    cols, signs = [], []
    for f in range(n_faces):
        span = [int(x) for x in ri[cp[f]:cp[f + 1]]]
        if not span or any(e not in position for e in span):
            continue
        cols.append([position[e] for e in span])
        signs.append([float(v) for v in vals[cp[f]:cp[f + 1]]])
    if not cols:
        return 0
    joined.add_faces(cols, signs)
    return int(joined.nF_hodge)
