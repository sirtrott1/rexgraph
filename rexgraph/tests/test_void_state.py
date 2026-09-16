"""Native triangle voids against explicit enumeration and exact filling ranks."""
from itertools import combinations, product

import numpy as np
import pytest

from rexgraph.cells import CellSet
from rexgraph.core import _cycles, _void
from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.void_state import void_state


def k4(filled=0):
    rex = RexGraph.from_cells([4, list(combinations(range(4), 2))])
    faces = [[0, 3, 1], [0, 4, 2], [1, 5, 2], [3, 5, 4]]
    if filled:
        rex.add_faces(faces[:filled])
    return rex


@pytest.mark.parametrize("filled", range(5))
def test_missing_faces_are_not_independent_homology_generators(filled):
    rex = k4(filled)
    before = object_digest(rex)
    value = void_state(CellSet(rex, 1, range(rex.nE)))
    assert value.n_potential == 4 and value.n_voids == 4-filled
    assert value.strain == 3*(4-filled)
    assert value.homology["independent_fillings"] == max(3-filled, 0)
    assert value.homology["rank_after"] == 3
    assert object_digest(rex) == before
    from rexgraph.io.partition_state import partition_tower
    from rexgraph.graded_boundary import _exact_compose_columns
    _, tower = partition_tower(rex)
    assert all(not column for column in _exact_compose_columns(tower[0], [dict(c) for c in value.columns]))
    reading = value.homology
    reading["independent_fillings"] = -99
    assert value.homology["independent_fillings"] == max(3-filled, 0)


@pytest.mark.parametrize("multiplicity", [(2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 3, 4)])
def test_compiled_enumeration_retains_every_parallel_relation(multiplicity):
    pairs, groups = [], []
    for pair, count in zip([(0, 1), (0, 2), (1, 2)], multiplicity, strict=True):
        groups.append(tuple(range(len(pairs), len(pairs)+count)))
        pairs.extend([pair]*count)
    rex = RexGraph.from_cells([3, pairs])
    source, target = np.array(pairs, np.int32).T.copy()
    adjacency = _cycles.build_symmetric_adjacency(3, len(pairs), source, target)
    triples, count = _void.find_potential_triangles(*adjacency, 3, len(pairs))
    expected = {tuple(sorted(triple)) for triple in product(*groups)}
    assert {tuple(sorted(map(int, triple))) for triple in triples} == expected
    assert count == len(expected)
    value = void_state(CellSet(rex, 1, range(rex.nE)))
    assert value.n_potential == value.n_voids == len(expected)
    rex.add_faces([next(iter(expected))])
    value = void_state(CellSet(rex, 1, range(rex.nE)))
    assert value.n_voids == len(expected)-1


def test_large_axis_face_keys_do_not_collide_after_int64_overflow():
    from rexgraph.native_sparse import native_coo
    n = 2**22
    # The first coordinates differ by 2**20, so radix keys differ by 2**64.
    first, second = (0, n-2, n-1), (2**20, n-2, n-1)
    face = native_coo(np.array(first), np.zeros(3, dtype=int), np.ones(3), (n, 1))
    keys = _void._realized_face_keys(face, n)
    assert keys.tolist() == [first[0]*n*n + first[1]*n + first[2]]
    realized, missing, count = _void.classify_triangles(face, np.array([first, second], np.int32), 2, n)
    assert realized.tolist() == [1, 0] and missing.tolist() == [1] and count == 1


@pytest.mark.parametrize("seed", range(10))
def test_random_parallel_enumeration_matches_independent_triples(seed):
    rng = np.random.default_rng(seed)
    pairs = [tuple(sorted(rng.choice(5, 2, replace=False))) for _ in range(11)]
    rex = RexGraph.from_cells([5, pairs])
    out = void_state(CellSet(rex, 1, range(rex.nE)))
    expected = set()
    for triple in combinations(range(len(pairs)), 3):
        vertices = [v for e in triple for v in pairs[e]]
        if len(set(vertices)) == 3 and all(vertices.count(v) == 2 for v in set(vertices)):
            expected.add(tuple(triple))
    assert {tuple(sorted(t)) for t in out.potential} == expected


def test_branching_outside_region_and_arbitrary_upper_grade_are_retained():
    rex = RexGraph.from_cells([5, [[0, 1], [1, 2], [0, 2], [0, 3, 4]],
                               [[(0, 1), (1, 1), (2, -1)]], []])
    before = object_digest(rex)
    out = void_state(CellSet(rex, 1, [0, 1, 2]))
    assert out.n_potential == 1 and out.n_voids == 0
    assert out.homology["rank_before"] == 1
    assert object_digest(rex) == before
    with pytest.raises(ValueError, match="pairwise"):
        void_state(CellSet(rex, 1, range(rex.nE)))


@pytest.mark.parametrize("support", [[0], [0, 0], [0, 1, 2]])
def test_nonpairwise_regions_are_not_expanded(support):
    rex = RexGraph.from_cells([3, [support]])
    with pytest.raises(ValueError, match="pairwise"):
        void_state(CellSet(rex, 1, [0]))


def test_empty_region_and_source_change():
    rex = k4()
    out = void_state(CellSet(rex, 1, []))
    assert out.shape == (6, 0) and out.strain == 0
    assert out.homology["independent_fillings"] == 0
    rex.add_faces([[0, 3, 1]])
    with pytest.raises(ValueError, match="changed"):
        out.check_state()


def test_raw_chain_failure_is_not_filtered_away():
    rex = k4()
    rex.add_faces([[0]], [[1]])
    with pytest.raises(ValueError, match="chain condition"):
        void_state(CellSet(rex, 1, range(rex.nE)))
