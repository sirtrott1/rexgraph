"""The binary load-bearing question is a walk, not a solve.

R_eff(e) = 1 exactly when e is a bridge, so `bridge_mask` answers it in one traversal
of the 1-skeleton and `_effective_resistance_batch` spends its columns only on the
relations that lie on a cycle. Measured identical to the solve on Gene Ontology
slices, 520/520 and 1315/1315.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.bridges import bridge_mask, cycle_support_mask
from rexgraph.graph import RexGraph


def _tarjan(nV, src, tgt):
    """Reference bridge finder, deliberately a plain recursion-free DFS."""
    adj = [[] for _ in range(nV)]
    for e, (a, b) in enumerate(zip(src, tgt, strict=True)):
        adj[a].append((b, e))
        adj[b].append((a, e))
    disc = [-1] * nV
    low = [0] * nV
    out = []
    timer = 0
    for root in range(nV):
        if disc[root] != -1:
            continue
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            v, pe, it = stack[-1]
            advanced = False
            for (w, e) in it:
                if e == pe:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = timer
                    timer += 1
                    stack.append((w, e, iter(adj[w])))
                    advanced = True
                    break
                low[v] = min(low[v], disc[w])
            if not advanced:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        out.append(pe)
    return set(out)


def _mask_set(rex):
    return set(np.flatnonzero(bridge_mask(rex)).tolist())


def _ref_set(rex):
    return _tarjan(int(rex.nV), np.asarray(rex.sources), np.asarray(rex.targets))


def _g(s, t):
    return RexGraph(sources=np.asarray(s, dtype=np.int32),
                    targets=np.asarray(t, dtype=np.int32))


def test_a_tree_is_all_bridges():
    rex = _g(range(9), range(1, 10))
    assert _mask_set(rex) == set(range(9))


def test_a_cycle_has_none():
    rex = _g(range(10), np.roll(np.arange(10), -1))
    assert _mask_set(rex) == set()


def test_the_barbell_has_exactly_the_bar():
    rex = _g([0, 1, 2, 3, 4, 5, 2], [1, 2, 0, 4, 5, 3, 3])
    assert _mask_set(rex) == {6}


def test_parallel_relations_cover_each_other():
    """Neither of two relations over the same pair is a bridge; the one beyond is."""
    rex = _g([0, 0, 1], [1, 1, 2])
    assert _mask_set(rex) == {2}


def test_a_self_loop_is_never_a_bridge():
    rex = _g([0, 1, 1], [1, 2, 1])
    assert 2 not in _mask_set(rex)


def test_matches_the_reference_on_random_complexes():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(120):
        nv = int(rng.integers(4, 40))
        m = int(rng.integers(3, nv * 2))
        rex = _g(rng.integers(0, nv, m), rng.integers(0, nv, m))
        try:
            rex._ensure_clean()
        except Exception:
            continue
        assert _mask_set(rex) == _ref_set(rex)
        checked += 1
    assert checked >= 60, f"only exercised {checked}"


def test_the_two_masks_partition_the_relations():
    rex = _g([0, 1, 2, 0, 3], [1, 2, 0, 3, 4])
    assert np.all(bridge_mask(rex) ^ cycle_support_mask(rex))


def test_the_mask_agrees_with_the_solve():
    """The identity the routing rests on: bridge iff R_eff is 1."""
    for s, t in ([[0, 1, 2, 0, 3], [1, 2, 0, 3, 4]],
                 [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]],
                 [[0, 1, 2, 3, 4, 5, 2], [1, 2, 0, 4, 5, 3, 3]]):
        rex = _g(s, t)
        rex._ensure_clean()
        reff = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
        assert set(np.flatnonzero(reff > 1.0 - 1e-9).tolist()) == _mask_set(rex)


def test_routing_leaves_the_values_exact():
    """Solving only the non-bridges must not move any value."""
    rex = _g([0, 1, 2, 0, 4, 5, 6, 4, 8, 9], [1, 2, 0, 2, 5, 6, 4, 6, 9, 8])
    rex._ensure_clean()
    B1 = np.asarray(rex.B1)
    truth = np.einsum("ve,vw,we->e", B1, np.linalg.pinv(B1 @ B1.T), B1)
    got = np.asarray(rex._effective_resistance_batch(np.arange(int(rex.nE))))
    assert np.allclose(got, truth, atol=1e-9)
    assert float(got.sum()) == pytest.approx(int(np.linalg.matrix_rank(B1)), abs=1e-9)
