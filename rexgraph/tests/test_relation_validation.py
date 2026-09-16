"""Candidate validation against original shares and independent Q products."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.relation_validation import validate_relations


def branching(arity=4):
    return RexGraph.from_cells([arity, [list(range(arity)), *[[0, v] for v in range(1, arity)]]])


@pytest.mark.parametrize("arity", [3, 4, 5, 17, 129])
def test_original_shares_and_fill_parity(arity):
    rex = branching(arity)
    cycle = [(0, arity-1), *[(v, -1) for v in range(1, arity)]]
    wrong = [(i, 1 if i == 0 else -1) for i in range(arity)]
    before = object_digest(rex)
    result = validate_relations(rex, [cycle, wrong, cycle, []])
    assert result["valid"] == (True, False, True, False)
    assert result["accepted"] == (0, 2)
    assert result["closed"] == (True, False, True, True)
    assert result["residuals"][1] == ((0, Q(arity-2)), *[(v, Q(1, arity-1)-1) for v in range(1, arity)])
    values = np.array([v for _, v in cycle], object)
    assert rex.fill_cycle(values).nF == 1
    assert object_digest(rex) == before


def test_closure_integrality_storage_and_nonzero_are_distinct():
    rex = branching(3)
    base = [(0, 2), (1, -1), (2, -1)]
    scales = [Q(1, 2), 2**53+1, 2**2000, 2**100]
    candidates = [[(i, v*s) for i, v in base] for s in scales]
    candidates.extend([[(0, 1), (0, -1)], [(0, 1), (0, 1), (1, -1), (2, -1)]])
    result = validate_relations(rex, candidates)
    assert all(result["closed"])
    assert result["integral"] == (False, True, True, True, True, True)
    assert result["storable"] == (False, False, False, True, True, True)
    assert result["valid"] == (False, False, False, True, False, True)


def test_witness_loop_parallel_and_empty_candidates():
    rex = RexGraph.from_cells([2, [[0], [1, 1], [0, 1], [0, 1]]])
    result = validate_relations(rex, [[(0, 1)], [(1, 1)], [(2, 1), (3, -1)]])
    assert result["valid"] == (False, True, True)
    assert result["residuals"] == (((0, Q(1)),), (), ())
    assert validate_relations(rex, [])["valid"] == ()


@pytest.mark.parametrize("grade", [2, 3, 4, 5])
def test_full_tower_matches_independent_rational_products(grade):
    simplices = [list(combinations(range(5), k+1)) for k in range(5)]
    cells = [5, simplices[1]]
    maps = []
    for k in range(1, 5):
        index = {s: i for i, s in enumerate(simplices[k-1])}
        entries = [[(index[s[:i]+s[i+1:]], (-1)**i) for i in range(len(s))] for s in simplices[k]]
        if k >= 2:
            cells.append(entries)
        matrix = np.full((len(simplices[k-1]), len(simplices[k])), Q(0), object)
        for j, column in enumerate(entries):
            for i, v in column:
                matrix[i, j] = Q(v)
        maps.append(matrix)
    rex = RexGraph.from_cells(cells)
    lower = maps[grade-2]
    rng = np.random.default_rng(grade)
    proposed = rng.integers(-3, 4, size=(lower.shape[1], 12)).astype(object)
    expected = lower @ proposed
    candidates = [[(i, int(v)) for i, v in enumerate(proposed[:, j]) if v] for j in range(12)]
    out = validate_relations(rex, candidates, grade=grade)
    assert out["residuals"] == tuple(tuple((i, Q(v)) for i, v in enumerate(expected[:, j]) if v) for j in range(12))
    assert out["valid"] == tuple(bool(np.any(proposed[:, j])) and not np.any(expected[:, j]) for j in range(12))
    if grade < 5:
        existing = cells[grade]
        assert all(validate_relations(rex, existing, grade=grade)["valid"])


@pytest.mark.parametrize("grade", [True, 1.0, "2", -1, 0, 1, 3])
def test_bad_grades(grade):
    with pytest.raises((TypeError, ValueError)):
        validate_relations(branching(), [], grade=grade)


@pytest.mark.parametrize("candidates", [None, {}, [None], [[1]], [[(0,)]], [[(0, 1, 2)]],
    [[(True, 1)]], [[(1.0, 1)]], [[(-1, 1)]], [[(4, 1)]], [[(0, True)]],
    [[(0, 1.0)]], [[(0, 0.1)]], [[(0, float("nan"))]], [[(0, float("inf"))]], [[(0, "1/2")]]])
def test_malformed_or_inexact_declarations(candidates):
    with pytest.raises((TypeError, ValueError)):
        validate_relations(branching(), candidates)


def test_rejects_invalid_raw_source_instead_of_filtering_faces():
    rex = branching()
    rex.add_faces([[0]], [[1]])
    with pytest.raises(ValueError, match="chain condition"):
        validate_relations(rex, [])
