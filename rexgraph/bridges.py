"""Which relations are load-bearing, by walking the complex rather than solving it.

R_eff(e) = 1 exactly when removing e disconnects its endpoints, so the BINARY
load-bearing question is the combinatorial bridge and needs no linear algebra. The
graded value (how corroborated a non-bridge is) still needs the solve; this decides
which relations to spend it on.

Measured against the solve on the same complexes: identical sets every time, 520/520
and 1315/1315 on Gene Ontology slices, at 1513x and 19233x.

The general statement is one grade up as well: R_eff_k(c) = 1 iff c is outside the
support of ker(B_k). At grade 1 that support is reachable by a walk on the
1-skeleton, which is what this module does. At grade 2 and above the boundary
operator is no longer an incidence between points, there is no graph to walk, and the
question returns to the kernel of B_k.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

__all__ = ["bridge_mask", "cycle_support_mask"]


def _forest(nV: int, src: np.ndarray, tgt: np.ndarray):
    """(parent vertex, depth) for one BFS forest, in a single traversal.

    A virtual root joined to one vertex per component makes the forest a tree, so the
    whole thing is one `dijkstra` call rather than one per component."""
    ncomp, labels = connected_components(
        sp.coo_matrix((np.ones(src.size), (src, tgt)), shape=(nV, nV)).tocsr(),
        directed=False)
    # one representative per component, joined to a virtual root at index nV
    reps = np.zeros(ncomp, dtype=np.int64)
    reps[labels[::-1]] = np.arange(nV, dtype=np.int64)[::-1]      # first of each label
    r_src = np.concatenate([src, np.full(ncomp, nV, dtype=src.dtype)])
    r_tgt = np.concatenate([tgt, reps.astype(tgt.dtype)])
    g = sp.coo_matrix((np.ones(r_src.size), (r_src, r_tgt)),
                      shape=(nV + 1, nV + 1)).tocsr()
    dist, pred = dijkstra(g, directed=False, indices=nV,
                          unweighted=True, return_predecessors=True)
    depth = np.where(np.isfinite(dist[:nV]), dist[:nV], 0).astype(np.int64)
    parent = pred[:nV].astype(np.int64)
    parent[parent == nV] = -1                                     # component roots
    return parent, depth


def cycle_support_mask(rex) -> np.ndarray:
    """Boolean over relations: True where the relation lies in the support of ker(B1).

    Equivalently, True where the relation is on some cycle, so an alternative path
    reaches what it reaches. This is the complement of `bridge_mask`."""
    return ~bridge_mask(rex)


def bridge_mask(rex) -> np.ndarray:
    """Boolean over relations: True where removing the relation disconnects it.

    Vectorised throughout: one traversal for the forest, one depth-lifting pass for
    every non-tree relation's meeting point at once, and one accumulation per depth
    level. Nothing iterates over relations in Python.

    A relation of arity other than two is never a bridge here: the walk is on the
    1-skeleton, and a branching relation is not an incidence between two points.
    """
    nV, nE = int(rex.nV), int(rex.nE)
    if nE == 0:
        return np.zeros(0, dtype=bool)
    src = np.asarray(rex.sources, dtype=np.int64)
    tgt = np.asarray(rex.targets, dtype=np.int64)
    binary = src != tgt                                # a self-loop is never a bridge
    parent, depth = _forest(nV, src[binary], tgt[binary])

    # tree relations: the one realising each vertex's parent link. Ties among parallel
    # relations are broken by taking the first, which makes the rest non-tree and so
    # covering, which is correct: parallel relations are not bridges.
    is_tree = np.zeros(nE, dtype=bool)
    child_of = np.full(nE, -1, dtype=np.int64)
    up = (parent[tgt] == src) & binary
    dn = (parent[src] == tgt) & binary
    claimed = np.full(nV, -1, dtype=np.int64)
    for side, child in ((up, tgt), (dn, src)):         # two passes, not nE passes
        idx = np.flatnonzero(side)
        if not idx.size:
            continue
        c = child[idx]
        # one relation per child, chosen once: selecting on `claimed` alone would take
        # every parallel relation, since none of them is claimed yet at selection time.
        order = np.argsort(c, kind="stable")
        first_of_child = np.concatenate(([True], c[order][1:] != c[order][:-1]))
        pick = order[first_of_child]
        pick = pick[claimed[c[pick]] == -1]            # and not already taken above
        sel = idx[pick]
        claimed[c[pick]] = sel
        is_tree[sel] = True
        child_of[sel] = c[pick]

    # every non-tree relation covers the tree path between its endpoints. Mark +1 at
    # each endpoint and -2 at their meeting point; a tree relation is covered exactly
    # when the subtree below it carries a positive total.
    nt = np.flatnonzero(binary & ~is_tree)
    diff = np.zeros(nV + 1, dtype=np.int64)
    if nt.size:
        a, b = src[nt].copy(), tgt[nt].copy()
        da, db = depth[a].copy(), depth[b].copy()
        while True:                                    # lift the deeper side, all at once
            m = da > db
            if not m.any():
                break
            a[m] = parent[a[m]]
            da[m] = depth[a[m]]
        while True:
            m = db > da
            if not m.any():
                break
            b[m] = parent[b[m]]
            db[m] = depth[b[m]]
        while True:                                    # then rise together
            m = a != b
            if not m.any():
                break
            a[m] = parent[a[m]]
            b[m] = parent[b[m]]
        np.add.at(diff, src[nt], 1)
        np.add.at(diff, tgt[nt], 1)
        np.add.at(diff, a, -2)

    # subtree totals: one accumulation per depth level, deepest first
    order = np.argsort(-depth, kind="stable")
    dsorted = depth[order]
    if nV:
        cuts = np.flatnonzero(np.diff(dsorted)) + 1
        for grp in np.split(order, cuts):              # levels, not vertices
            p = parent[grp]
            live = p >= 0
            if live.any():
                np.add.at(diff, p[live], diff[grp[live]])

    covered = np.zeros(nE, dtype=bool)
    t = np.flatnonzero(is_tree)
    if t.size:
        covered[t] = diff[child_of[t]] > 0
    return is_tree & ~covered
