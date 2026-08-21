"""Manifold gluing conditions."""
from itertools import combinations

import numpy as np
import pytest

from rexgraph.graded_boundary import betti_numbers, build_graded_boundaries
from rexgraph.graph import RexGraph
from rexgraph.sheaf import Sheaf


def graded_cells(simplices):
    top = max(len(s) for s in simplices) - 1
    by_grade = [sorted({f for s in simplices for f in combinations(sorted(s), d + 1)})
                for d in range(top + 1)]
    index = [{f: i for i, f in enumerate(fs)} for fs in by_grade]
    cells = [len(by_grade[0])]
    for d in range(1, top + 1):
        cells.append([[(index[d - 1][f[:i] + f[i + 1:]], (-1) ** i) for i in range(len(f))]
                      for f in by_grade[d]])
    return cells, by_grade


def sphere(k):
    return [tuple(c) for c in combinations(range(k + 2), k + 1)]


def link(simplices, v):
    return [tuple(x for x in s if x != v) for s in simplices if v in s]


def betti_of(simplices):
    cells, _ = graded_cells(simplices)
    return betti_numbers(build_graded_boundaries(cells))


#### homology and the chain condition, at every grade #######################
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_the_simplex_boundary_is_a_k_sphere(k):
    assert betti_of(sphere(k)) == [1] + [0] * (k - 1) + [1]


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_the_chain_condition_holds_exactly_at_every_consecutive_pair(k):
    cells, _ = graded_cells(sphere(k))
    b = build_graded_boundaries(cells)
    for d in range(len(b) - 1):
        assert np.abs((b[d] @ b[d + 1]).toarray()).max() == 0.0


#### the two independent halves #############################################
def chart_glue(simplices, k):
    cells, by_grade = graded_cells(simplices)
    verts = [v for (v,) in by_grade[0]]
    pos = {v: i for i, v in enumerate(verts)}
    edges = by_grade[1]
    rex = RexGraph(sources=np.array([pos[e[0]] for e in edges], np.int32),
                   targets=np.array([pos[e[1]] for e in edges], np.int32))
    rex._ensure_clean()
    d = k + 1
    sh = Sheaf(rex, stalk_dim=d, grade=0)
    for v in verts:
        lb = betti_of(link(simplices, v))
        sh.assign(pos[v], (list(lb) + [0] * d)[:d])
    return sh.glue(), verts


@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_sphere_atlas_has_a_global_section(k):
    g, _v = chart_glue(sphere(k), k)
    assert g["ratio"] == 1.0 and g["H1"] == 0


def test_gluing_catches_the_wedge_and_localises_it():
    a = sphere(3)
    b = [tuple(0 if x == 0 else x + 4 for x in s) for s in sphere(3)]
    g, verts = chart_glue(a + b, 3)
    assert g["ratio"] < 1.0 and g["H1"] > 0
    broken = {verts[i] for pair in g["failed"] for i in pair}
    assert 0 in broken                                  # the wedge vertex
    assert betti_of(link(a + b, 0)) == [2, 0, 2]        # two spheres, not one


def test_gluing_alone_is_NOT_sufficient_and_the_link_test_is_the_other_half():
    skel = [tuple(c) for c in combinations(range(5), 3)]
    g, _v = chart_glue(skel, 2)
    assert g["ratio"] == 1.0 and g["H1"] == 0           # glues: charts all agree
    for v in range(5):
        assert betti_of(link(skel, v)) == [1, 3]        # and none of them is S^1


def test_where_the_recursion_bottoms_out():
    s3 = betti_of(sphere(3))
    assert s3 == [1, 0, 0, 1]
    # any complex sharing this Betti vector is indistinguishable to a homology reading,
    # which is precisely the situation the conjecture is about.
    assert s3 == [1, 0, 0, 1]
