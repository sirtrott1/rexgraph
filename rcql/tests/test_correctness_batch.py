"""Refusal, arithmetic and rewrite regressions from the live math review."""
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph import Chain, Cochain
from rexgraph.graph import RexGraph

from rcql import Executor, call, parse, query, source
from rcql.operators import quadrance, spread
from rcql.types import Exactness


@pytest.fixture
def r():
    return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("text", [
    'BOUNDARY(1, BOUNDARY(2, "not a chain"))',
    'BOUNDARY(1, BOUNDARY(2, $missing))',
    'BOUNDARY(1.1, BOUNDARY(2.1, "not a chain"))',
])
def test_invalid_original_expression_is_not_erased(r, explain, text):
    with pytest.raises((TypeError, ValueError, KeyError)):
        Executor(sources={'r': r}).execute(parse(('EXPLAIN ' if explain else '') + 'FROM $r RETURN ' + text))


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_spread_checks_both_operands_before_adapter(r, reverse, monkeypatch):
    a = Cochain(1, np.array([1, 2, 3]), source=r)
    b = Cochain(1, np.array([.1, .2, .4]), source=r)
    if reverse:
        a, b = b, a
    with pytest.raises(TypeError, match="integer or rational"):
        spread(r, a, b, exact=True)
    import rcql.executor
    monkeypatch.setattr(rcql.executor, 'get_operator', lambda *a: pytest.fail('adapter resolved'))
    with pytest.raises(TypeError, match="integer or rational"):
        Executor(sources={'r': r}).execute(query(source('r'), call('SPREAD', a, b, True)))


def test_complex_vector_and_block_quadrance_agree(r):
    v = np.array([1j, 0, 0])
    assert quadrance(r, Cochain(1, v, source=r)) == 1
    assert quadrance(r, Cochain(1, v[:, None], source=r)).tolist() == [1]


def test_exact_character_has_one_consistent_contract():
    r = RexGraph.from_graph([0, 1, 2], [1, 2, 0], g_channel='normalized')
    result = Executor(sources={'r': r}).execute(query(source('r'), call('CHARACTER', True)))
    assert result.values[0]['values'].shape == (3, 4)
    assert result.values[0]['exactness'] == 'rational'
    assert result.exactness == (Exactness.RATIONAL,)


def test_rank_plan_matches_exact_result(r):
    executor = Executor(sources={'r': r})
    got = executor.execute(query(source('r'), call('RANK', 1)))
    explained = executor.execute(parse('EXPLAIN FROM $r RETURN RANK(1)'))
    assert got.values == (2,)
    assert got.exactness == (Exactness.INTEGER,)
    assert explained.values[0]['returns'][0]['result']['exactness'] == 'integer'


def test_higher_boundary_exact_input_stays_exact(r):
    face = Chain(2, np.array([Q(2**60 + 1, 3)], dtype=object), source=r)
    result = Executor(sources={'r': r}).execute(query(source('r'), call('BOUNDARY', 2, face)))
    assert result.exactness == (Exactness.RATIONAL,)
    assert all(isinstance(x, Q) for x in result.values[0].values)
    assert set(abs(x) for x in result.values[0].values) == {Q(2**60 + 1, 3)}


@pytest.mark.parametrize('explain', [False, True])
@pytest.mark.parametrize('case', ['shape', 'grade', 'foreign', 'basis'])
def test_boundary_rewrite_refuses_invalid_carrier(r, explain, case, monkeypatch):
    face = Chain(2, np.array([1, 2]) if case == 'shape' else np.array([1]),
                 cell_keys=('not-canonical',) if case == 'basis' else None,
                 source=object() if case == 'foreign' else r)
    if case == 'grade':
        face = Chain(1, np.ones(3, dtype=int), source=r)
    import rcql.executor
    monkeypatch.setattr(rcql.executor, 'get_operator', lambda *a: pytest.fail('adapter resolved'))
    q = parse(('EXPLAIN ' if explain else '') + 'FROM $r RETURN BOUNDARY(1, BOUNDARY(2, $face))')
    with pytest.raises((ValueError, TypeError)):
        Executor(sources={'r': r}, params={'face': face}).execute(q)


def test_boundary_block_preserves_shape_without_vector_zero_rewrite(r):
    face = Chain(2, np.array([[Q(2**60 + 1, 3), Q(-7, 5)]], object), source=r)
    result = Executor(sources={'r': r}).execute(query(source('r'),
        call('BOUNDARY', 1, call('BOUNDARY', 2, face))))
    assert result.values[0].values.shape == (3, 2)
    assert all(x == Q(0) for x in result.values[0].values.flat)
    assert result.rewrites == ()
    assert result.exactness == (Exactness.RATIONAL,)


def test_unproven_chain_is_not_rewritten(r, monkeypatch):
    import rexgraph.graded_boundary
    monkeypatch.setattr(rexgraph.graded_boundary, '_exact_chain_residual', lambda *a: None)
    face = Chain(2, np.array([1]), source=r)
    result = Executor(sources={'r': r}).execute(query(source('r'),
        call('BOUNDARY', 1, call('BOUNDARY', 2, face))))
    assert result.rewrites == ()
    assert result.values[0].values.tolist() == [Q(0)] * 3


def test_empty_exact_character_has_four_columns():
    empty = RexGraph.from_graph([], [])
    result = Executor(sources={'r': empty}).execute(query(source('r'), call('CHARACTER', True)))
    assert result.values[0]['values'].shape == (0, 4)
    assert result.exactness == (Exactness.RATIONAL,)


def test_declared_type_is_not_an_executable_chain(r):
    from rcql.binding import bind
    from rcql.capabilities import SourcePolicy
    from rcql.types import BasisRef, Domain, RCType, ShapeRef, ValueKind, Variance
    binding = bind('r', r, SourcePolicy.allow('*'))
    declared = RCType('Chain', grade=2, kind=ValueKind.CHAIN, variance=Variance.CHAIN,
        domain=Domain.INTEGER, exactness=Exactness.INTEGER, shape=ShapeRef((1,)),
        source=binding.ref, basis=BasisRef('r', 2))
    executor = Executor(sources={'r': r}, params={'face': declared})
    with pytest.raises(TypeError, match='chain'):
        executor.execute(parse('FROM $r RETURN BOUNDARY(1, BOUNDARY(2, $face))'))
