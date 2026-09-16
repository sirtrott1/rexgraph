"""Exact full tower homotopy equations retain explicit endpoint complexes."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.chain_map import ChainHomotopy, CoordinateComplex, GradedMap
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph


def interval():
    rex = RexGraph.from_cells([2, [[0, 1]]])
    c = CoordinateComplex.from_rex(rex)
    left = GradedMap(c, c, ((), ()))
    a, b = Q(1, 3), Q(2, 5)
    right = GradedMap(c, c, (((0, 0, -a), (1, 0, a), (0, 1, -b), (1, 1, b)), ((0, 0, b-a),)))
    witness = (((0, 0, a), (0, 1, b)), ())
    return rex, left, right, witness


def test_nonzero_rational_witness_and_reversal():
    _, f, g, h = interval()
    proof = ChainHomotopy(f, g.verify(), h)
    assert proof.residuals == (0, 0) and proof.shapes == ((1, 2), (0, 1))
    reverse = tuple(tuple((i, j, -v) for i, j, v in component) for component in h)
    assert ChainHomotopy(g, f, reverse).residuals == (0, 0)
    assert proof.coefficient_digest != ChainHomotopy(g, f, reverse).coefficient_digest
    proof.check_state()


@pytest.mark.parametrize("defect", ["sign", "top", "missing", "float", "different-domain", "nonmap", "stale"])
def test_invalid_witnesses_and_endpoint_states_fail(defect):
    rex, f, g, h = interval()
    if defect == "sign":
        h = (((0, 0, -Q(1, 3)), (0, 1, Q(2, 5))), ())
    elif defect == "top":
        h = (h[0], ((0, 0, 1),))
    elif defect == "missing":
        h = h[:1]
    elif defect == "float":
        h = (((0, 0, 1/3),), ())
    elif defect == "different-domain":
        g = replace(g, domain=CoordinateComplex.from_rex(rex))
    elif defect == "nonmap":
        g = replace(g, components=(g.components[0], ()))
    else:
        rex.add_edges(np.array([1]), np.array([0]))
    with pytest.raises((ValueError, TypeError)):
        ChainHomotopy(f, g, h)


@pytest.mark.parametrize("cells", [[4, [[0, 1, 2, 3]]], [1, [[0]]], [1, [[0, 0]]], [1, []], solid_octahedron_3rex()])
def test_zero_homotopy_covers_branching_witness_loop_empty_and_full_tower(cells):
    rex = RexGraph.from_cells(cells)
    c = CoordinateComplex.from_rex(rex)
    identity = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    proof = ChainHomotopy(identity, identity, tuple(() for _ in c.sizes))
    assert proof.residuals == (0,) * len(c.sizes)
    assert len(proof.witness) == len(c.sizes) and proof.shapes[-1][0] == 0
