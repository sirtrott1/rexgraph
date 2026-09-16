"""Weighted Hodge sectors keep Chain variance and explicit metric certificates."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graded_metric import DiagonalMetric
from rexgraph.graph import RexGraph
from rexgraph.linear_operator import RexOperator, metric_adjoint
from rexgraph.weighted_hodge import weighted_hodge

from rcql import Executor, call, param, parse, query, source


def make():
    return RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])


def run(rex, text, **params):
    return Executor(sources={'r': rex}, params=params).execute(parse(text))


def test_named_metrics_exact_hand_energy_and_provenance():
    rex = make()
    result = run(rex, '''FROM $r LET m=METRIC(1,$w) LET lower=METRIC(0,$v)
        LET h=HODGE_SUM(grade=1,lower_metric=lower,metric=m)
        LET y=APPLY(h,$x,exact=true)
        RETURN y AS action, MOMENT($x,y,m,true) AS energy''',
        w=Cochain(1, np.array([2, 3]), source=rex),
        v=Cochain(0, np.array([1, 2, 3, 4]), source=rex),
        x=Chain(1, np.array([3, 2]), source=rex))
    assert result.named_values['action'].values.tolist() == [Q(7, 3), Q(-2, 3)]
    assert result.named_values['energy'] == Q(10)
    assert [e.value for e in result.exactness] == ['rational', 'rational']
    node = next(n for n in result.native_plan['nodes'] if n.get('operator') == 'HODGE_SUM')
    desc = node['result']['operator']
    assert desc['action_variance'] == 'chain'
    assert desc['metric_self_adjoint'] and desc['metric_psd']
    assert not desc['symmetric'] and not desc['psd']
    assert [m['basis']['grade'] for m in desc['grade_metrics']] == [1, 0, 2]
    assert len(desc['grade_metrics'][0]['coefficient_digest']) == 64
    assert node['physical']['method'] == 'factored-weighted-hodge-handle'
    assert any(e['methods'][0]['method'] == 'rational-weighted-hodge-action'
               for e in result.execution if e['operator'] == 'APPLY')
    json.dumps(result.native_plan, allow_nan=False)


@pytest.mark.parametrize('name', ['HODGE_DOWN', 'HODGE_UP', 'HODGE_SUM', 'HODGE_DIFFERENCE'])
@pytest.mark.parametrize('columns', [None, 0, 2])
def test_text_builder_core_and_external_handles_agree(name, columns):
    rex = make()
    shape = (2,) if columns is None else (2, columns)
    x = Chain(1, np.arange(int(np.prod(shape))).reshape(shape), source=rex)
    m = DiagonalMetric(rex, 1, (2, 3))
    h = weighted_hodge(rex, 1, sector=name.removeprefix('HODGE_').lower(), metric=m)
    text = f'FROM $r RETURN APPLY({name}(grade=1,metric=$m),$x,true)'
    built = query(source('r'), call('APPLY', call(name, grade=1, metric=param('m')), param('x'), True))
    assert built == parse(text)
    out = run(rex, text, x=x, m=m)
    external = run(rex, 'FROM $r RETURN APPLY($h,$x,true)', x=x, h=h)
    np.testing.assert_array_equal(out.values[0].values, h.apply(x.values, exact=True))
    np.testing.assert_array_equal(out.values[0].values, external.values[0].values)
    assert out.values[0].values.shape == shape
    assert out.provenance[0]['result_type']['variance'] == 'chain'


def test_weighted_sum_adjoint_retains_all_original_metric_descriptors():
    rex = make()
    m = DiagonalMetric(rex, 1, (2, 3))
    lower = DiagonalMetric(rex, 0, (1, 2, 3, 4))
    h = weighted_hodge(rex, 1, metric=m, lower_metric=lower)
    a = metric_adjoint(h, m, m)
    result = run(rex, 'FROM $r RETURN APPLY($a,$x,true)', a=a, x=Chain(1, np.array([3, 2]), source=rex))
    assert result.values[0].values.tolist() == [Q(7, 3), Q(-2, 3)]
    encoded = json.dumps(result.native_plan)
    assert lower.coefficient_digest in encoded
    assert 'primal_operator' in encoded and 'grade_metrics' in encoded


def test_exact_sector_cancellation_in_composed_rcql():
    rex = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    result = run(rex, '''FROM $r LET m=METRIC(1,$w)
        LET d=HODGE_DOWN(1,m) LET u=HODGE_UP(1,m)
        RETURN APPLY(d,APPLY(u,$x,true),true), APPLY(u,APPLY(d,$x,true),true),
               HODGE_DIFFERENCE(1,m)''',
        w=Cochain(1, np.array([2, 3, 5]), source=rex), x=Chain(1, np.array([3, 2, 1]), source=rex))
    assert not any(result.values[0].values) and not any(result.values[1].values)
    desc = result.provenance[2]['result_type']['operator']
    assert desc['metric_self_adjoint'] and desc['metric_psd'] is None


@pytest.mark.parametrize('bad', ['variance', 'basis', 'shape', 'source', 'metric-source', 'metric-grade',
                                  'metric-basis', 'metric-exact', 'carrier-exact'])
@pytest.mark.parametrize('explain', [False, True])
def test_bad_spaces_fail_before_adapters(bad, explain, monkeypatch):
    rex = make()
    x = Chain(1, np.array([1, 2]), source=rex)
    m = DiagonalMetric(rex, 1, (2, 3))
    if bad == 'variance':
        x = Cochain(1, x.values, source=rex)
    elif bad == 'basis':
        x = Chain(1, x.values, ('b', 'a'), rex)
    elif bad == 'shape':
        x = Chain(1, np.ones(3, int), source=rex)
    elif bad == 'source':
        x = Chain(1, x.values, source=make())
    elif bad == 'metric-source':
        m = DiagonalMetric(make(), 1, (2, 3))
    elif bad == 'metric-grade':
        m = DiagonalMetric(rex, 0, (1,)*4)
    elif bad == 'metric-basis':
        m = DiagonalMetric(rex, 1, (2, 3), ('b', 'a'))
    elif bad == 'metric-exact':
        m = DiagonalMetric(rex, 1, (2., 3.))
    else:
        x = x.with_values(x.values.astype(float))
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran'))
    with pytest.raises((TypeError, ValueError)):
        run(rex, ('EXPLAIN ' if explain else '') + 'FROM $r RETURN APPLY(HODGE_SUM(1,$m),$x,true)', m=m, x=x)


@pytest.mark.parametrize('expression', [
    'HODGE_DOWN(0,lower_metric=METRIC(0))', 'HODGE_SUM(-1)', 'HODGE_SUM(true)',
    'HODGE_UP(2)', 'HODGE_SUM(1,upper_metric=METRIC(0))',
    'RESOLVENT(HODGE_DIFFERENCE(1))',
    'HODGE_DOWN(1,upper_metric=METRIC(2))',
])
def test_wrong_options_and_euclidean_solver_do_not_run(expression, monkeypatch):
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran'))
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        run(make(), 'EXPLAIN FROM $r RETURN '+expression)


def test_zero_sector_exactness_does_not_depend_on_float_metric():
    rex = make()
    result = run(rex, 'FROM $r RETURN APPLY(HODGE_UP(1,$m),$x,true)',
                 m=DiagonalMetric(rex, 1, (2., 3.)), x=Chain(1, np.array([1, 2]), source=rex))
    assert result.values[0].values.tolist() == [Q(0), Q(0)]
    desc = next(n['result']['operator'] for n in result.native_plan['nodes'] if n.get('operator') == 'HODGE_UP')
    assert desc['exact_action'] and dict(desc['parameters'])['active_sectors'] == []


def test_computed_metric_positivity_is_not_a_static_proof():
    text = 'FROM $r RETURN HODGE_SUM(1, METRIC(1, ZERO(1)))'
    explained = run(make(), 'EXPLAIN '+text)
    assert 'deferred' in json.dumps(explained.values)
    with pytest.raises(ValueError, match='positive'):
        run(make(), text)


def test_complex_input_and_no_materialization(monkeypatch):
    rex = make()
    x = Chain(1, np.array([1j, 2]), source=rex)
    expected = weighted_hodge(rex, 1).apply(x.values)
    monkeypatch.setattr(RexOperator, 'as_scipy', lambda *a: pytest.fail('assembled'))
    result = run(rex, 'FROM $r RETURN APPLY(HODGE_SUM(1),$x)', x=x)
    np.testing.assert_allclose(result.values[0].values, expected)
    assert result.provenance[0]['result_type']['domain'] == 'complex'


def test_unsupported_higher_exact_domain_is_refused_before_adapters(monkeypatch):
    from rexgraph.native_sparse import as_native
    rex = RexGraph.from_cells(solid_octahedron_3rex())
    upper = as_native(rex._graded_duals[0])
    rex._graded_duals = [upper.with_data(upper.data * .5).dual]
    x = Chain(3, np.ones(1, int), source=rex)
    result = run(rex, 'FROM $r RETURN APPLY(HODGE_SUM(3),$x)', x=x)
    assert result.values[0].values.tolist() == [2.]
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran'))
    with pytest.raises(TypeError, match='certified exact'):
        run(rex, 'FROM $r RETURN APPLY(HODGE_SUM(3),$x,true)', x=x)
