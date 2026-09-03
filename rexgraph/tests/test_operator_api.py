"""Typed cochain, graded operator, and matrix-free Green contracts."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from rexgraph import (
    Chain,
    Cochain,
    Field,
    GradedState,
    GreenOperator,
    RexOperator,
    boundary_operator,
    coboundary_operator,
    down_laplacian,
    hodge_operator,
    up_laplacian,
    vertex_green,
)
from rexgraph import graded_boundary as gb
from rexgraph.graph import RexGraph


def _cycle() -> RexGraph:
    return RexGraph.from_graph(
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 2, 0], dtype=np.int32),
    )


def _filled_triangle() -> RexGraph:
    return RexGraph.from_simplicial(
        np.array([0, 1, 0], dtype=np.int32),
        np.array([1, 2, 2], dtype=np.int32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )


def test_cochain_keeps_grade_basis_and_source():
    values = np.arange(6, dtype=float).reshape(3, 2)
    cochain = Cochain(1, values, cell_keys=("a", "b", "c"), source="fixture")
    assert cochain.grade == 1
    assert cochain.n_cells == 3
    assert cochain.cell_keys == ("a", "b", "c")
    assert cochain.source == "fixture"
    assert np.array_equal(cochain.numpy(copy=True), values)
    assert cochain.with_values(values + 1).cell_keys == cochain.cell_keys


def test_chain_keeps_the_same_basis_identity_without_becoming_a_cochain():
    values = np.arange(6, dtype=float).reshape(3, 2)
    chain = Chain(1, values, cell_keys=("a", "b", "c"), source="fixture")
    assert chain.grade == 1
    assert chain.n_cells == 3
    assert chain.cell_keys == ("a", "b", "c")
    assert np.array_equal(chain.numpy(copy=True), values)
    assert isinstance(chain.with_values(values + 1), Chain)
    assert not isinstance(chain, Cochain)


@pytest.mark.parametrize(
    "args, error",
    [
        ((-1, np.ones(2)), "grade"),
        ((0, 3.0), "cell axis"),
        ((1, np.ones(3), ("a", "b")), "cell_keys"),
    ],
)
def test_cochain_rejects_invalid_identity(args, error):
    with pytest.raises(ValueError, match=error):
        Cochain(*args)


def test_field_and_graded_state_keep_typed_values():
    c0 = Cochain(0, np.ones(3))
    c2 = Cochain(2, np.ones(1))
    state = GradedState({2: c2, 0: c0})
    assert state.grades == (0, 2)
    assert [cochain.grade for cochain in state] == [0, 2]
    assert state[2] is c2 and 1 not in state
    field = Field(c0, operator="fixture", kind="green")
    assert field.grade == 0 and field.values is c0.values
    with pytest.raises(TypeError, match="Cochain"):
        Field(np.ones(3), operator="fixture")
    with pytest.raises(ValueError, match="key"):
        GradedState({1: c0})


def test_rex_operator_validates_shape_grade_and_psd_contract():
    with pytest.raises(ValueError, match="nonnegative"):
        RexOperator("bad", (-1, 2), 0, 0, lambda values: values)
    with pytest.raises(ValueError, match="grades"):
        RexOperator("bad", (2, 2), -1, 0, lambda values: values)
    with pytest.raises(ValueError, match="symmetric"):
        RexOperator("bad", (2, 3), 0, 0, lambda values: values, symmetric=True)
    with pytest.raises(ValueError, match="semidefinite"):
        RexOperator("bad", (2, 2), 0, 0, lambda values: values, psd=True)


def test_boundary_and_coboundary_preserve_grade_and_chain():
    rex = _filled_triangle()
    b1 = boundary_operator(rex, 1)
    b2 = boundary_operator(rex, 2)
    d0 = coboundary_operator(rex, 0)
    assert (b1.domain_grade, b1.codomain_grade) == (1, 0)
    assert (b2.domain_grade, b2.codomain_grade) == (2, 1)
    assert (d0.domain_grade, d0.codomain_grade) == (0, 1)
    assert np.allclose(b1.apply(b2.as_scipy().toarray()), 0.0)
    assert np.array_equal(d0.as_scipy().toarray(), b1.as_scipy().T.toarray())
    with pytest.raises(ValueError, match="first axis"):
        b1.apply(np.zeros(rex.nE + 1))


def test_top_coboundary_is_the_rectangular_zero_upper_sector():
    """The final cochain grade has no upper cells, so its coboundary is zero.

    This is deliberately rectangular: it maps the carried top cochain space into
    the empty next grade, without pretending the complex contains another cell
    grade.  RCQL relies on this for a total typed ``COBOUNDARY`` signature.
    """
    rex = _filled_triangle()
    top = coboundary_operator(rex, 2)
    assert (top.domain_grade, top.codomain_grade) == (2, 3)
    assert top.shape == (0, rex.nF)
    assert top.arithmetic == "structural"
    assert top.as_scipy().nnz == 0
    assert top.apply(np.ones(rex.nF)).shape == (0,)
    assert top.apply(np.ones((rex.nF, 2))).shape == (0, 2)


def test_boundary_operators_reach_grade_three():
    rex = RexGraph.from_cells(gb.solid_octahedron_3rex())
    b2 = boundary_operator(rex, 2)
    b3 = boundary_operator(rex, 3)
    assert b3.shape == (8, 1)
    assert (b3.domain_grade, b3.codomain_grade) == (3, 2)
    assert np.allclose(b2.apply(b3.as_scipy().toarray()), 0.0)
    with pytest.raises(ValueError, match="not present"):
        boundary_operator(rex, 4)


def test_laplacians_match_current_compiled_sparse_core():
    cycle = _cycle()
    filled = _filled_triangle()
    np.testing.assert_allclose(
        hodge_operator(cycle, 0).as_scipy().toarray(),
        cycle.L0_sparse.toarray(),
    )
    np.testing.assert_allclose(
        down_laplacian(cycle, 1).as_scipy().toarray(),
        cycle.L1_sparse.toarray(),
    )
    np.testing.assert_allclose(
        hodge_operator(filled, 1).as_scipy().toarray(),
        filled.L1_sparse.toarray(),
    )
    np.testing.assert_allclose(
        hodge_operator(filled, 2).as_scipy().toarray(),
        filled.L2_sparse.toarray(),
    )


@pytest.mark.parametrize("grade", [1, 2])
def test_down_and_up_laplacians_mutually_annihilate(grade):
    rex = RexGraph.from_cells(gb.solid_octahedron_3rex())
    down = down_laplacian(rex, grade)
    up = up_laplacian(rex, grade)
    block = np.arange(down.shape[0] * 2, dtype=float).reshape(down.shape[0], 2)
    assert np.allclose(down.apply(up.apply(block)), 0.0, atol=1e-12)
    assert np.allclose(up.apply(down.apply(block)), 0.0, atol=1e-12)


def test_missing_laplacian_parts_are_exact_zero():
    rex = RexGraph.from_cells(gb.solid_octahedron_3rex())
    grade0 = down_laplacian(rex, 0)
    top = up_laplacian(rex, 3)
    assert grade0.as_scipy().nnz == 0
    assert top.as_scipy().nnz == 0
    assert np.array_equal(grade0.apply(np.ones((rex.nV, 2))), np.zeros((rex.nV, 2)))
    assert np.array_equal(top.apply(np.ones((1, 2))), np.zeros((1, 2)))


def test_laplacian_apply_stays_factored_until_materialized():
    rex = _filled_triangle()
    operator = hodge_operator(rex, 1)
    values = np.arange(rex.nE, dtype=float)
    assert operator.matrix is None and operator.matrix_factory is not None
    expected = rex.L1_sparse @ values
    np.testing.assert_allclose(operator.apply(values), expected)
    assert operator.matrix is None


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf])
def test_hodge_rejects_coefficients_that_do_not_preserve_psd(alpha):
    with pytest.raises(ValueError, match="finite and >= 0"):
        hodge_operator(_cycle(), 0, alpha=alpha)


def test_hodge_is_symmetric_psd_at_every_grade():
    rex = RexGraph.from_cells(gb.solid_octahedron_3rex())
    for grade in range(4):
        operator = hodge_operator(rex, grade, alpha=0.25)
        matrix = operator.as_scipy().toarray()
        assert operator.symmetric and operator.psd
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_resolvent_matches_direct_sparse_solve_without_inverse():
    operator = hodge_operator(_cycle(), 0)
    green = GreenOperator.resolvent(operator, alpha=0.7)
    sources = np.arange(6, dtype=float).reshape(3, 2)
    expected = spla.spsolve(
        sp.eye(3, format="csr") + 0.7 * operator.as_scipy(), sources
    )
    np.testing.assert_allclose(green.solve(sources), expected, atol=1e-10)
    assert operator.matrix is None
    with pytest.raises(ValueError, match="finite and >= 0"):
        GreenOperator.resolvent(operator, alpha=-1)


def test_vertex_green_matches_effective_resistance_on_pairwise_rex():
    rex = _cycle()
    sources = boundary_operator(rex, 1).as_scipy().toarray()
    green = vertex_green(rex)
    field = green.solve(sources)
    expected = rex._effective_resistance_batch(np.arange(rex.nE))
    np.testing.assert_allclose(green.quadrance(sources, field), expected, atol=1e-9)
    np.testing.assert_allclose(
        field, np.linalg.pinv(rex.L0_sparse.toarray()) @ sources, atol=1e-12
    )


def test_vertex_green_uses_complete_kernel_on_branching_rex():
    rex = RexGraph.from_hypergraph(
        np.array([0, 3, 7], dtype=np.int32),
        np.array([0, 1, 2, 2, 3, 4, 5], dtype=np.int32),
    )
    sources = np.column_stack(
        [np.asarray(rex.B1)[:, 0], np.arange(rex.nV, dtype=float)]
    )
    expected = np.linalg.pinv(rex.L0_sparse.toarray()) @ sources
    np.testing.assert_allclose(vertex_green(rex).solve(sources), expected, atol=1e-9)


def test_green_gram_quadrance_and_spread_are_consistent():
    rex = _cycle()
    sources = boundary_operator(rex, 1).as_scipy().toarray()
    green = vertex_green(rex)
    gram = green.gram(sources)
    quadrance = green.quadrance(sources)
    spread = green.spread(sources)
    np.testing.assert_allclose(np.diag(gram), quadrance)
    np.testing.assert_allclose(gram, gram.T)
    np.testing.assert_allclose(spread, spread.T)
    assert np.array_equal(np.diag(spread), np.zeros(rex.nE))
    assert np.all((spread >= 0.0) & (spread <= 1.0))
