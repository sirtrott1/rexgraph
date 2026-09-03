"""Direct contracts for RCQL's current relational-complex math adapters.

These tests intentionally call the operator functions rather than going through
the parser/executor.  They fix the behaviour the Phase 1 signature catalogue
must describe: source-bound variance, exact rational geometry, total upper
co-boundary, and unapplied Green actions.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from rexgraph import (
    Cell,
    CellBoundary,
    CellCoboundary,
    CellSet,
    Chain,
    Cochain,
    CompositeBinary,
    Field,
    GradedCellPattern,
)
from rexgraph.graph import RexGraph
from rexgraph.green import GreenOperator
from rexgraph.metric_field import MetricCurvature

from rcql import Executor, call, query, source
from rcql.operators import (
    apply,
    arity,
    betti,
    boundary,
    cell_at,
    cells_at,
    character,
    closure,
    co_relations,
    coboundary,
    composite,
    describe,
    existence,
    grade,
    graded_enclosure,
    graded_star,
    green,
    harmonic,
    head,
    hodge,
    hodge_coordinates,
    hodge_op,
    metric_curvature,
    nullity,
    quadrance,
    rank,
    rex_source,
    share,
    share_support,
    significance,
    spread,
    winding,
    zero,
)
from rcql.types import Exactness


@pytest.fixture
def triangle():
    return RexGraph.from_simplicial(
        np.array([0, 1, 0], dtype=np.int32),
        np.array([1, 2, 2], dtype=np.int32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )


def test_boundary_is_a_chain_map_and_coboundary_is_a_cochain_map(triangle):
    face = Chain(2, np.ones(triangle.nF, dtype=np.int64), source=triangle)
    boundary_value = boundary(triangle, 2, face)
    assert isinstance(boundary_value, Chain)
    assert boundary_value.grade == 1
    assert boundary_value.n_cells == triangle.nE

    top = coboundary(
        triangle,
        2,
        Cochain(2, np.ones(triangle.nF, dtype=np.int64), source=triangle),
    )
    assert isinstance(top, Cochain)
    assert (top.grade, top.values.shape) == (3, (0,))

    with pytest.raises(TypeError, match="chain"):
        boundary(triangle, 2, Cochain(2, np.ones(triangle.nF), source=triangle))
    with pytest.raises(TypeError, match="cochain"):
        coboundary(triangle, 1, Chain(1, np.ones(triangle.nE), source=triangle))


def test_exact_c1_actions_read_declared_relation_columns_not_float_b1():
    branching = RexGraph.from_hypergraph(
        np.asarray([0, 3], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
    )

    boundary_value = boundary(
        branching, 1, Chain(1, np.asarray([2], dtype=np.int64), source=branching),
    )
    coboundary_value = coboundary(
        branching, 0,
        Cochain(0, np.asarray([2, 4, 8], dtype=np.int64), source=branching),
    )

    assert boundary_value.values.tolist() == [Fraction(-2), Fraction(1), Fraction(1)]
    assert coboundary_value.values.tolist() == [Fraction(4)]

    supplied = Chain(
        1, np.asarray([Fraction(2)], dtype=object), source=branching,
    )
    planned = Executor(sources={"branching": branching}).execute(query(
        source("branching"), call("QUADRANCE", call("BOUNDARY", 1, supplied), True),
    ))
    assert planned.values == (Fraction(6),)
    assert planned.exactness == (Exactness.RATIONAL,)


def test_geometry_retains_exact_rational_results_and_zero_spread_is_absent(triangle):
    left = Cochain(1, np.array([1, 2, 3], dtype=np.int64), source=triangle)
    perpendicular = Cochain(1, np.array([2, -1, 0], dtype=np.int64), source=triangle)
    zero_value = Cochain(1, np.zeros(triangle.nE, dtype=np.int64), source=triangle)

    assert quadrance(triangle, left, exact=True) == Fraction(14)
    assert spread(triangle, left, perpendicular, exact=True) == Fraction(1)
    assert spread(triangle, zero_value, perpendicular, exact=True) is None
    assert spread(triangle, zero_value, perpendicular) is None

    with pytest.raises(TypeError, match="matching chain/cochain variance"):
        spread(
            triangle,
            Chain(1, left.values, source=triangle),
            perpendicular,
            exact=True,
        )


def test_math_refuses_cross_source_and_cross_basis_values(triangle):
    values = np.array([1, 2, 3], dtype=np.int64)
    left = Cochain(1, values, cell_keys=("a", "b", "c"), source=triangle)
    reordered = Cochain(1, values, cell_keys=("c", "b", "a"), source=triangle)
    foreign = Cochain(1, values, source=object())

    with pytest.raises(ValueError, match="ordered basis"):
        spread(triangle, left, reordered, exact=True)
    with pytest.raises(ValueError, match="bound to its source Rex"):
        quadrance(triangle, foreign, exact=True)


def test_green_is_an_unapplied_action_with_compatibility_application(triangle):
    input_value = Cochain(0, np.ones(triangle.nV), source=triangle)
    action = green(triangle)
    assert isinstance(action, GreenOperator)

    field = apply(triangle, action, input_value)
    assert isinstance(field, Field)
    assert field.grade == 0
    assert isinstance(green(triangle, input_value), Field)

    with pytest.raises(ValueError, match="grade 0"):
        apply(triangle, action, Cochain(1, np.ones(triangle.nE), source=triangle))


def test_executor_composes_green_action_and_application_without_a_dense_inverse(triangle):
    input_value = Cochain(0, np.ones(triangle.nV), source=triangle)
    result = Executor(sources={"complex": triangle}).execute(
        query(source("complex"), call("APPLY", call("GREEN"), input_value))
    )
    assert isinstance(result.values[0], Field)
    assert result.values[0].operator.kind == "pseudoinverse"


def test_executor_applies_a_declared_hodge_operator_to_an_explicit_c1_seed(triangle):
    result = Executor(sources={"complex": triangle}).execute(query(
        source("complex"),
        call("APPLY", call("HODGE_OPERATOR", 1), call("INDICATOR", call("CELL", 1, 0))),
    ))

    field = result.values[0]
    assert isinstance(field, Field)
    assert field.grade == 1
    assert field.kind == "L1"
    assert result.exactness == (Exactness.APPROXIMATE,)

    with pytest.raises(TypeError, match="HODGE_OPERATOR"):
        Executor(sources={"complex": triangle}).execute(query(
            source("complex"),
            call("APPLY", call("BOUNDARY", 1), call("INDICATOR", call("CELL", 1, 0))),
        ))


def test_hodge_and_closure_refuse_unsupported_semantics(triangle):
    with pytest.raises(ValueError, match="grade 1"):
        hodge(triangle, Cochain(0, np.ones(triangle.nV), source=triangle))
    with pytest.raises(NotImplementedError, match="only grade-0"):
        closure(triangle, 0, grade=1)
    with pytest.raises(ValueError, match="not present"):
        closure(triangle, triangle.nV)


def test_grade_one_hodge_readings_require_a_bound_cochain(triangle):
    flow = Cochain(1, np.arange(triangle.nE, dtype=float), source=triangle)
    split = hodge(triangle, flow)
    assert set(split) == {"gradient", "curl", "harmonic"}
    assert all(value.grade == 1 and value.source is triangle for value in split.values())
    assert hodge_coordinates(triangle, flow).harmonic.ndim == 1
    assert winding(triangle, flow).ndim == 1

    with pytest.raises(TypeError, match="cochain"):
        hodge(triangle, np.arange(triangle.nE, dtype=float))
    with pytest.raises(TypeError, match="cochain"):
        hodge_coordinates(triangle, np.arange(triangle.nE, dtype=float))
    with pytest.raises(TypeError, match="cochain"):
        winding(triangle, np.arange(triangle.nE, dtype=float))


def test_metric_curvature_is_a_source_bound_c1_reading_at_actual_relation_arity():
    branching = RexGraph.from_hypergraph(
        np.asarray([0, 3, 7], dtype=np.int32),
        np.asarray([0, 1, 2, 0, 2, 3, 4], dtype=np.int32),
    )
    metric = Cochain(1, np.asarray([2, 5], dtype=np.int64), source=branching)

    reading = metric_curvature(branching, metric)
    assert isinstance(reading, MetricCurvature)
    assert reading.total == Fraction(21, 5)
    assert reading.curvature.values.tolist() == [
        Fraction(3), Fraction(0), Fraction(6, 5), Fraction(0), Fraction(0),
    ]
    assert reading.relation_contribution.values.tolist() == [Fraction(21, 10)] * 2

    result = Executor(sources={"branching": branching}).execute(query(
        source("branching"), call("METRIC_CURVATURE", metric),
    ))
    assert result.values[0].total == Fraction(21, 5)
    assert result.exactness == (Exactness.RATIONAL,)


def test_exact_character_and_typed_zero_are_explicit(triangle):
    exact = character(triangle, exact=True)
    assert exact["exactness"] == "rational"
    assert exact["channels"] == ("L1_down", "L_O", "L_SG", "L_C")
    assert all(isinstance(value, Fraction) for value in exact["values"].flat)

    assert isinstance(zero(triangle, 1), Cochain)
    assert isinstance(zero(triangle, 1, "chain"), Chain)
    with pytest.raises(ValueError, match="kind"):
        zero(triangle, 1, "field")


def test_structural_and_boundary_rank_readings_keep_their_source_contract(triangle):
    flow = Cochain(1, np.arange(triangle.nE, dtype=float), source=triangle)
    operator = hodge_op(triangle, 1)

    assert rex_source(triangle, "triangle") == "triangle"
    assert grade(triangle) == 2
    assert grade(triangle, flow) == 1
    assert describe(triangle)["kind"] == "Rex"
    assert (operator.domain_grade, operator.codomain_grade) == (1, 1)
    assert rank(triangle, 2) == 1
    assert nullity(triangle, 2) == 0
    assert betti(triangle, 0) == 1
    assert isinstance(significance(triangle, 0), float)
    assert harmonic(triangle, flow).grade == 1

    foreign = RexGraph.from_graph(
        np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32)
    )
    with pytest.raises(ValueError, match="bound to its source Rex"):
        rank(triangle, hodge_op(foreign, 0))
    with pytest.raises(ValueError, match="bound to its source Rex"):
        nullity(triangle, hodge_op(foreign, 0))


def test_betti_uses_boundary_rank_not_support_projection_components():
    branching = RexGraph.from_hypergraph(
        np.array([0, 4], dtype=np.int32), np.array([0, 1, 2, 3], dtype=np.int32)
    )
    # One relation touches every vertex, so its support is one connected piece.
    assert (branching.nV, branching.nE) == (4, 1)
    assert betti(branching, 0) == 3


def test_cell_readings_preserve_primary_c1_and_exact_composite_binary_data():
    branching = RexGraph.from_hypergraph(
        np.array([0, 3], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32)
    )
    relation = cell_at(branching, 1, 0)
    assert isinstance(relation, Cell)
    assert (relation.grade, relation.index, relation.source) == (1, 0, branching)
    assert cells_at(branching, 1).indices == (0,)

    binary = composite(branching, relation)
    assert isinstance(binary, CompositeBinary)
    assert binary.arity == 3 and binary.witness is False
    np.testing.assert_array_equal(binary.existence.values, [1, 1, 1])
    np.testing.assert_array_equal(binary.head.values, [1, 0, 0])
    np.testing.assert_array_equal(binary.share_support.values, [0, 1, 1])
    assert binary.boundary.values.tolist() == [Fraction(-1), Fraction(1, 2), Fraction(1, 2)]
    assert binary.share.values.tolist() == [Fraction(0), Fraction(1, 2), Fraction(1, 2)]
    assert binary.integer_boundary.values.tolist() == [Fraction(-2), Fraction(1), Fraction(1)]

    np.testing.assert_array_equal(existence(branching, relation).values, [1, 1, 1])
    np.testing.assert_array_equal(head(branching, relation).values, [1, 0, 0])
    np.testing.assert_array_equal(share_support(branching, relation).values, [0, 1, 1])
    assert share(branching, relation).values.tolist() == [
        Fraction(0), Fraction(1, 2), Fraction(1, 2)
    ]
    assert arity(branching, relation) == 3

    direct = boundary(branching, relation)
    assert isinstance(direct, CellBoundary)
    assert direct.cells.indices == (0, 1, 2)
    assert direct.chain.values.tolist() == binary.boundary.values.tolist()
    assert isinstance(direct.composite, CompositeBinary)
    assert direct.composite.arity == binary.arity
    assert quadrance(branching, direct, exact=True) == Fraction(3, 2)
    assert grade(branching, direct) == 0

    witness = RexGraph.from_hypergraph(
        np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int32)
    )
    witness_binary = composite(witness, cell_at(witness, 1, 0))
    assert witness_binary.witness is True
    np.testing.assert_array_equal(witness_binary.head.values, [0])
    np.testing.assert_array_equal(witness_binary.share_support.values, [0])
    assert witness_binary.boundary.values.tolist() == [Fraction(1)]

    queried = Executor(sources={"complex": branching}).execute(
        query(source("complex"), call("COMPOSITE", call("CELL", 1, 0)))
    )
    assert isinstance(queried.values[0], CompositeBinary)


def test_cell_boundary_coboundary_corelations_and_enclosure_are_graded_not_graph_projection(triangle):
    relation = cell_at(triangle, 1, 0)
    face = cell_at(triangle, 2, 0)

    face_boundary = boundary(triangle, face)
    assert isinstance(face_boundary, CellBoundary)
    assert isinstance(face_boundary.cells, CellSet)
    assert face_boundary.cells.grade == 1
    assert face_boundary.cells.indices == (0, 1, 2)
    assert isinstance(face_boundary.chain, Chain)

    direct_up = coboundary(triangle, relation)
    assert isinstance(direct_up, CellCoboundary)
    assert direct_up.cells.grade == 2
    assert direct_up.cells.indices == (0,)
    assert co_relations(triangle, relation).indices == (0,)

    upward = graded_star(triangle, relation)
    enclosed = graded_enclosure(triangle, relation)
    assert isinstance(upward, GradedCellPattern)
    assert upward.grades == (1, 2)
    assert enclosed.grades == upward.grades
    assert upward.at(1).indices == (0,)
    assert upward.at(2).indices == (0,)


def test_c0_is_derived_even_when_a_relation_boundary_cancels_to_zero():
    self_loop = RexGraph.from_hypergraph(
        np.array([0, 2], dtype=np.int32), np.array([0, 0], dtype=np.int32)
    )
    vertex = cell_at(self_loop, 0, 0)
    relation = cell_at(self_loop, 1, 0)

    # A repeated relation occurrence cancels in the coefficient boundary, but it
    # remains a primary C1 relation containing the C0 boundary cell.
    direct = boundary(self_loop, relation)
    assert direct.cells.indices == (0,)
    assert direct.chain.values.tolist() == [Fraction(0)]
    assert co_relations(self_loop, vertex).indices == (0,)
    up = coboundary(self_loop, vertex)
    assert isinstance(up, CellCoboundary)
    assert up.cells.indices == (0,)
    assert up.cochain.values.tolist() == [Fraction(0)]
    assert grade(self_loop, up) == 1
    # The composite is first class for a self loop rather than a refusal. [v, v] is the
    # one repeated C1 incidence the model admits, and it carries no head and no share, so
    # every coefficient reading is exactly zero while the relation itself still exists.
    binary = composite(self_loop, relation)
    assert isinstance(binary, CompositeBinary)
    assert binary.arity == 2
    assert binary.self_loop is True
    assert binary.witness is False
    assert binary.existence.values.tolist() == [1], "the participant is declared"
    assert binary.head.values.tolist() == [0], "a self loop distinguishes no head"
    assert binary.share_support.values.tolist() == [0]
    assert binary.share.values.tolist() == [Fraction(0)]
    assert binary.boundary.values.tolist() == [Fraction(0)]
    assert binary.integer_boundary.values.tolist() == [0]


def test_cell_sets_aggregate_exactly_and_reject_foreign_source_values():
    branching = RexGraph.from_hypergraph(
        np.array([0, 3, 5], dtype=np.int32),
        np.array([0, 1, 2, 2, 3], dtype=np.int32),
    )
    selected = CellSet(branching, 1, (0, 1))
    aggregate = boundary(branching, selected)
    assert isinstance(aggregate, Chain)
    assert aggregate.values.tolist() == [Fraction(-1), Fraction(1, 2), Fraction(-1, 2), Fraction(1)]

    vertices = CellSet(branching, 0, (0, 1))
    dual = coboundary(branching, vertices)
    assert isinstance(dual, Cochain)
    assert dual.values.tolist() == [Fraction(-1, 2), Fraction(0)]

    foreign = RexGraph.from_hypergraph(
        np.array([0, 2], dtype=np.int32), np.array([0, 1], dtype=np.int32)
    )
    with pytest.raises(ValueError, match="bound to its source Rex"):
        boundary(branching, cell_at(foreign, 1, 0))


def test_every_rex_math_operator_has_a_direct_contract_here():
    """The native math registry is exhaustive in this direct adapter suite."""
    from rcql.operators import _REGISTRY

    math = {
        "REX", "GRADE", "CELL", "CELLS", "COMPOSITE", "EXISTENCE", "HEAD", "SHARE",
        "SHARE_SUPPORT", "ARITY", "BOUNDARY", "COBOUNDARY", "CORELATIONS", "STAR",
        "ENCLOSURE", "DESCRIBE", "HODGE_OPERATOR", "RANK",
        "NULLITY", "BETTI", "HODGE", "HARMONIC", "GREEN", "APPLY", "QUADRANCE",
        "SPREAD", "HODGE_COORDS", "WINDING", "CLOSURE", "SIGNIFICANCE", "CHARACTER",
        "ZERO",
    }
    assert math <= set(_REGISTRY)
