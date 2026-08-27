"""Rings: the cycles a structure actually has, without choosing a basis.

Everything here reads the CYCLE SPACE Z1 = ker(B1), the 1-skeleton alone. Faces
are not consulted at any point, so on a complex that has them these counts are
dim Z1 = nE - rank(B1) and NOT beta_1: filling K4 with one face leaves a minimum
cycle basis of three while beta_1 is two, and a bigon with a face on it still has
one ring while beta_1 is zero. The two coincide exactly when there are no faces,
which is the case every example below is stated in.

A cycle basis holds exactly dim Z1 cycles, and for most structures that is fewer
than the rings the structure has. Cubane has six square faces and dim Z1 five;
C60 has thirty-two faces and dim Z1 thirty-one. Every basis drops one, and
nothing in the notion of a basis says which. That is the standing complaint
against "the smallest set of smallest rings": the answer depends on an arbitrary
choice made inside the algorithm.

The cycle space is a lattice, so ask a lattice question instead. Two are useful
and both are basis-free:

    shortest    the lattice's minimal vectors, the cycles of least weight
    relevant    the cycles that are not a sum of strictly shorter cycles

`relevant` is the one chemistry wants. It is the union of every minimum cycle
basis, so it contains each ring some basis would pick and never has to break a
tie. On C60 it returns all thirty-two faces where `shortest` returns the twelve
pentagons, since a hexagon is not a minimal vector.

Everything here is GF(2) linear algebra on integer bitmasks: exact, no tolerance
to choose, no float anywhere. Weight is the edge count, which is ring size.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = [
    "cycle_candidates",
    "minimum_cycle_basis",
    "relevant_cycles",
    "ring_sizes",
    "shortest_cycles",
]


def _edge_ends(rex):
    """(src, tgt) per cell, and a check that every cell is 2-ary.

    A ring is a closed walk through relations that each join two vertices. A
    branching relation joins k of them at once and a walk through it is not
    defined without saying which participant it leaves by, so this refuses
    rather than guessing.
    """
    rex._ensure_clean()
    ptr = np.asarray(rex._boundary_ptr)
    idx = np.asarray(rex._boundary_idx)
    arity = np.diff(ptr)
    if arity.size and int(arity.max()) != 2 or (arity.size and int(arity.min()) != 2):
        bad = int(np.argmax(arity != 2))
        raise ValueError(
            f"rings are defined on 2-ary relations; cell {bad} has arity "
            f"{int(arity[bad])}. Reduce or project the branching relations first.")
    return idx[ptr[:-1]].astype(np.int64), idx[ptr[:-1] + 1].astype(np.int64)


def _adjacency(nV, src, tgt):
    adj = [[] for _ in range(nV)]
    for e, (a, b) in enumerate(zip(src, tgt, strict=True)):
        adj[int(a)].append((int(b), e))
        adj[int(b)].append((int(a), e))
    return adj


def _bfs(adj, nV, root):
    """Shortest-path tree from `root`: distance and the edge reaching each vertex."""
    dist = [-1] * nV
    via = [-1] * nV
    prev = [-1] * nV
    dist[root] = 0
    q = deque([root])
    while q:
        v = q.popleft()
        for w, e in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                via[w] = e
                prev[w] = v
                q.append(w)
    return dist, via, prev


def _path_mask(via, prev, v, root):
    """The tree path root..v as a bitmask over edges."""
    m = 0
    while v != root and v >= 0:
        m ^= 1 << via[v]
        v = prev[v]
    return m


def _is_cycle(mask, src, tgt):
    """Every vertex touched has degree 2, and the support is one closed walk."""
    if mask == 0:
        return False
    deg = {}
    edges = []
    m = mask
    while m:
        e = (m & -m).bit_length() - 1
        m ^= 1 << e
        edges.append(e)
        for v in (int(src[e]), int(tgt[e])):
            deg[v] = deg.get(v, 0) + 1
    if any(d != 2 for d in deg.values()):
        return False
    # connected: walk it
    seen = {edges[0]}
    frontier = [edges[0]]
    ends = {e: (int(src[e]), int(tgt[e])) for e in edges}
    while frontier:
        e = frontier.pop()
        for f in edges:
            if f in seen:
                continue
            if set(ends[e]) & set(ends[f]):
                seen.add(f)
                frontier.append(f)
    return len(seen) == len(edges)


def cycle_candidates(rex):
    """Horton's candidate set: every cycle that any minimum basis could contain.

    For each vertex v and each relation (x, y), the closed walk
    `path(v,x) + (x,y) + path(y,v)` taken in a shortest-path tree rooted at v.
    Returned as `(weight, bitmask)` pairs, deduplicated and sorted by weight.
    """
    src, tgt = _edge_ends(rex)
    nV, nE = int(rex.nV), int(rex.nE)
    adj = _adjacency(nV, src, tgt)
    seen = set()
    for root in range(nV):
        dist, via, prev = _bfs(adj, nV, root)
        for e in range(nE):
            x, y = int(src[e]), int(tgt[e])
            if dist[x] < 0 or dist[y] < 0:
                continue
            m = _path_mask(via, prev, x, root) ^ _path_mask(via, prev, y, root)
            m ^= 1 << e
            if m and m not in seen and _is_cycle(m, src, tgt):
                seen.add(m)
    return sorted((int(bin(m).count("1")), m) for m in seen)


def _reduce(mask, basis):
    """Reduce against a GF(2) row-echelon basis keyed by leading bit."""
    while mask:
        lead = mask.bit_length() - 1
        row = basis.get(lead)
        if row is None:
            return mask, lead
        mask ^= row
    return 0, -1


def minimum_cycle_basis(rex, candidates=None):
    """One minimum-weight cycle basis, as `(weight, bitmask)` pairs.

    Horton's algorithm: take candidates in increasing weight, keep the ones that
    are independent over GF(2). The total weight is minimal and is an invariant;
    *which* cycles get picked is not, which is exactly why `relevant_cycles`
    exists.
    """
    cands = cycle_candidates(rex) if candidates is None else candidates
    basis, out = {}, []
    for w, m in cands:
        r, lead = _reduce(m, basis)
        if r:
            basis[lead] = r
            out.append((w, m))
    return out


def shortest_cycles(rex, candidates=None):
    """The lattice's minimal vectors: every cycle of least weight.

    Basis-free. On cubane this is all six faces; on C60 it is the twelve
    pentagons only, the hexagons being one longer.
    """
    cands = cycle_candidates(rex) if candidates is None else candidates
    if not cands:
        return []
    least = cands[0][0]
    return [(w, m) for w, m in cands if w == least]


def relevant_cycles(rex, candidates=None):
    """The cycles that are not a sum of strictly shorter cycles.

    Basis-free, and the union of every minimum cycle basis, so it holds each ring
    that some basis would pick without having to break the tie. This is the
    canonical answer to "what rings does this structure have".
    """
    cands = cycle_candidates(rex) if candidates is None else candidates
    basis, out, i = {}, [], 0
    while i < len(cands):
        w = cands[i][0]
        j = i
        cls = []
        while j < len(cands) and cands[j][0] == w:
            cls.append(cands[j])
            j += 1
        # relevance is against STRICTLY shorter cycles, so test the whole class
        # before any of it enters the basis
        for weight, m in cls:
            r, _ = _reduce(m, basis)
            if r:
                out.append((weight, m))
        for _, m in cls:
            r, lead = _reduce(m, basis)
            if r:
                basis[lead] = r
        i = j
    return out


def cycle_vector(rex, cycle):
    """A rings bitmask as a signed chain: f64[nE], +1 or -1 on the support, 0 off it.

    Everything in this module returns UNSIGNED masks, which say which relations a
    ring uses and nothing about how it closes. A mask is not a chain: the signs are
    the whole content of the boundary, and the unsigned support is exactly the
    set-theoretic encoding that fails to land in ker(B1). This orients the walk, so
    the result is a genuine 1-cycle and `B1 @ v == 0` holds exactly.

    Accepts either an int bitmask or a `(weight, mask)` pair as `cycle_candidates`,
    `shortest_cycles`, `relevant_cycles` and `minimum_cycle_basis` return.
    """
    import numpy as _np

    mask = int(cycle[1]) if isinstance(cycle, tuple) else int(cycle)
    src, tgt = _edge_ends(rex)
    edges = []
    m = mask
    while m:
        e = (m & -m).bit_length() - 1
        m ^= 1 << e
        edges.append(e)
    if not edges:
        return _np.zeros(int(rex.nE), dtype=_np.float64)
    if not _is_cycle(mask, src, tgt):
        raise ValueError("mask is not a simple cycle, so it cannot be oriented")

    at = {}
    for e in edges:
        at.setdefault(int(src[e]), []).append(e)
        at.setdefault(int(tgt[e]), []).append(e)

    out = _np.zeros(int(rex.nE), dtype=_np.float64)
    start = int(src[edges[0]])
    v, used = start, set()
    while len(used) < len(edges):
        nxt = next((e for e in at[v] if e not in used), None)
        if nxt is None:                       # _is_cycle already excludes this
            raise ValueError("the walk did not close")
        used.add(nxt)
        if int(src[nxt]) == v:
            out[nxt] = 1.0
            v = int(tgt[nxt])
        else:
            out[nxt] = -1.0
            v = int(src[nxt])
    if v != start:
        raise ValueError("the walk did not return to its start")
    return out


def cycle_vectors(rex, cycles):
    """`cycle_vector` over an iterable, as a sparse nE x k chain matrix."""
    import numpy as _np
    import scipy.sparse as _sp

    cols = [cycle_vector(rex, c) for c in cycles]
    if not cols:
        return _sp.csc_matrix((int(rex.nE), 0), dtype=_np.float64)
    return _sp.csc_matrix(_np.stack(cols, 1))


def ring_sizes(rex, cycles=None):
    """How many rings of each size, as `{size: count}`.

    Reads `relevant_cycles` by default, so the counts do not depend on a basis.
    """
    cyc = relevant_cycles(rex) if cycles is None else cycles
    out = {}
    for w, _ in cyc:
        out[w] = out.get(w, 0) + 1
    return dict(sorted(out.items()))
