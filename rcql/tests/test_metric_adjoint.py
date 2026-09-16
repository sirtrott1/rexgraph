"""ADJOINT/APPLY carry endpoint metrics, variance and exactness through RCQL."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator, boundary_operator, metric_adjoint

from rcql import Executor, call, param, parse, query, source


def make():
    return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])


def run(rex, text, **params):
    return Executor(sources={'r': rex}, params=params).execute(parse(text))


def test_named_endpoint_metrics_exact_application_and_methods():
    rex = make()
    result = run(rex, '''FROM $r
        LET md = METRIC(1, $w1) LET mc = METRIC(0, $w0)
        LET a = ADJOINT(operator=BOUNDARY(1), codomain_metric=mc, domain_metric=md)
        RETURN APPLY(a, $y, exact=true) AS raised''',
        w1=Cochain(1, np.array([2, 3]), source=rex),
        w0=Cochain(0, np.array([1, 2, 3, 4]), source=rex),
        y=Chain(0, np.array([1, 2, 3, 4]), source=rex))
    value = result.named_values['raised']
    assert isinstance(value, Chain) and value.grade == 1
    assert list(value.values) == [Q(-1), Q(8, 3)]
    assert result.exactness[0].value == 'rational'
    nodes = {n.get('operator'): n for n in result.native_plan['nodes'] if n.get('operator')}
    desc = nodes['ADJOINT']['result']['operator']
    assert desc['domain']['grade'] == 0 and desc['codomain']['grade'] == 1
    assert desc['adjoint_domain_metric']['basis']['grade'] == 1
    assert desc['adjoint_codomain_metric']['basis']['grade'] == 0
    assert desc['exact_action'] and desc['action_variance'] == 'chain'
    assert not desc['symmetric'] and not desc['psd']
    assert nodes['APPLY']['physical']['method'] == 'rational-metric-adjoint-action'
    event = next(e for e in result.execution if e['operator'] == 'APPLY')
    assert event['methods'][0]['method'] == nodes['APPLY']['physical']['method']
    json.dumps(result.native_plan, allow_nan=False)


@pytest.mark.parametrize('columns', [None, 0, 2])
def test_exact_adjoint_identity_composes_with_native_boundary_and_moment(columns):
    rex = make()
    def values(n):
        shape = (n,) if columns is None else (n, columns)
        return np.array([Q(i+1, i+2) for i in range(int(np.prod(shape)))], object).reshape(shape)
    result = run(rex, '''FROM $r LET md=METRIC(1,$w1) LET mc=METRIC(0,$w0)
        RETURN MOMENT(BOUNDARY(1,$x),$y,mc,true),
               MOMENT($x,APPLY(ADJOINT(BOUNDARY(1),md,mc),$y,true),md,true)''',
        x=Chain(1, values(2), source=rex), y=Chain(0, values(4), source=rex),
        w1=Cochain(1, np.array([2, 3]), source=rex), w0=Cochain(0, np.array([1, 2, 3, 4]), source=rex))
    assert result.values[0] == result.values[1]
    assert isinstance(result.values[0], Q)


def test_external_native_handle_and_builder_match_text():
    rex = make()
    a = metric_adjoint(boundary_operator(rex, 1), DiagonalMetric(rex, 1, (2, 3)))
    y = Chain(0, np.array([1, 2, 3, 4]), source=rex)
    text = 'FROM $r RETURN APPLY(action=$a, values=$y, exact=true)'
    built = query(source('r'), call('APPLY', action=param('a'), values=param('y'), exact=True))
    assert parse(text) == built
    result = run(rex, text, a=a, y=y)
    assert list(result.values[0].values) == [Q(-1, 3), Q(2, 3)]


def test_coboundary_adjoint_preserves_dual_variance():
    rex = make()
    x = Cochain(1, np.array([3, 2]), source=rex)
    result = run(rex, 'FROM $r RETURN APPLY(ADJOINT(COBOUNDARY(0)), $x, true)', x=x)
    assert isinstance(result.values[0], Cochain)
    assert result.values[0].grade == 0
    assert list(result.values[0].values) == [Q(-1), Q(1), Q(-1), Q(1)]


def test_complex_adjoint_result_is_not_coerced_to_real():
    rex = make()
    values = np.array([1, 2j, 3-1j, 4])
    y = Chain(0, values, source=rex)
    result = run(rex, 'FROM $r RETURN APPLY(ADJOINT(BOUNDARY(1)), $y)', y=y)
    np.testing.assert_allclose(result.values[0].values, [-4/3+5j/3, 2-1j])
    assert result.provenance[0]['result_type']['domain'] == 'complex'
    assert result.exactness[0].value == 'approximate'


@pytest.mark.parametrize('bad', ['variance', 'grade', 'basis', 'source', 'float-exact', 'metric-exact',
                                  'metric-source', 'metric-grade', 'metric-basis', 'shape'])
@pytest.mark.parametrize('explain', [False, True])
def test_bad_spaces_and_domains_fail_before_adapters(bad, explain, monkeypatch):
    rex = make()
    y = Chain(0, np.array([1, 2, 3, 4]), source=rex)
    md = DiagonalMetric(rex, 1, (2, 3))
    if bad == 'variance':
        y = Cochain(0, y.values, source=rex)
    elif bad == 'grade':
        y = Chain(1, np.ones(2, int), source=rex)
    elif bad == 'basis':
        y = Chain(0, y.values, ('d', 'c', 'b', 'a'), rex)
    elif bad == 'source':
        y = Chain(0, y.values, source=make())
    elif bad == 'float-exact':
        y = y.with_values(y.values.astype(float))
    elif bad == 'metric-exact':
        md = DiagonalMetric(rex, 1, (2., 3.))
    elif bad == 'metric-source':
        md = DiagonalMetric(make(), 1, (2, 3))
    elif bad == 'metric-grade':
        md = DiagonalMetric(rex, 0, (1, 2, 3, 4))
    elif bad == 'metric-basis':
        md = DiagonalMetric(rex, 1, (2, 3), ('b', 'a'))
    else:
        y = Chain(0, np.array([1, 2, 3]), source=rex)
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran'))
    with pytest.raises((TypeError, ValueError)):
        run(rex, ('EXPLAIN ' if explain else '') + 'FROM $r RETURN APPLY(ADJOINT(BOUNDARY(1),$md),$y,true)', y=y, md=md)


@pytest.mark.parametrize('expression', [
    'RESOLVENT(ADJOINT(HODGE_OPERATOR(1)))',
    'APPLY(ADJOINT(HODGE_OPERATOR(1)), ZERO(1), true)',
    'APPLY(HODGE_OPERATOR(1), ZERO(1), true)',
    'ADJOINT(GREEN())',
    'ADJOINT(BOUNDARY(1), METRIC(0))',
    'APPLY(ADJOINT(BOUNDARY(1)), ZERO(0))',
    'APPLY(ADJOINT(BOUNDARY(1)), ZERO(0,"chain"), 1)',
])
def test_no_false_psd_exact_or_variance_contract(expression, monkeypatch):
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran'))
    with pytest.raises((TypeError, ValueError)):
        run(make(), 'EXPLAIN FROM $r RETURN '+expression)


def test_opaque_operator_needs_declared_transpose_even_for_explain(monkeypatch):
    rex = make()
    a = RexOperator('opaque', (2, 2), 1, 1, lambda v: v, source=rex,
                    matrix_factory=lambda: pytest.fail('materialized'))
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran'))
    with pytest.raises(TypeError, match='transpose'):
        run(rex, 'EXPLAIN FROM $r RETURN ADJOINT($a)', a=a)


def test_absent_upper_grade_is_not_a_new_nonempty_population():
    rex = make()
    # This primary hypergraph carries C0/C1; d1 lands in the absent C2.
    y = Cochain(2, np.empty((0, 2), object), source=rex)
    result = run(rex, 'FROM $r RETURN APPLY(ADJOINT(COBOUNDARY(1)), $y, true)', y=y)
    assert result.values[0].grade == 1 and result.values[0].values.shape == (2, 2)
    assert not any(result.values[0].values.flat)
    with pytest.raises(ValueError, match='present'):
        run(rex, 'FROM $r RETURN METRIC(4)')


def test_adjoint_does_not_assemble_sparse_products_or_diagonalize(monkeypatch):
    rex = make()
    monkeypatch.setattr(RexOperator, 'as_scipy', lambda *a: pytest.fail('materialized'))
    monkeypatch.setattr(np.linalg, 'eigh', lambda *a: pytest.fail('diagonalized'))
    result = run(rex, 'FROM $r RETURN APPLY(ADJOINT(BOUNDARY(1)), $y, true)',
                 y=Chain(0, np.array([1, 2, 3, 4]), source=rex))
    assert result.values[0].values.tolist() == [Q(-2, 3), Q(2)]


def test_computed_metric_positivity_remains_deferred():
    rex = make()
    text = 'FROM $r RETURN ADJOINT(BOUNDARY(1), METRIC(1, ZERO(1)))'
    result = run(rex, 'EXPLAIN '+text)
    assert 'adjoint_metric_positivity' in json.dumps(result.values)
    assert 'deferred' in json.dumps(result.values)
    with pytest.raises(ValueError, match='positive'):
        run(rex, text)
