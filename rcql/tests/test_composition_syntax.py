"""Composition syntax lowers to native dataflow with the existing math contracts."""
import inspect
import json
from fractions import Fraction

import numpy as np
import pytest
from rexgraph import Chain, Cochain
from rexgraph.graph import RexGraph

from rcql import (
    Alias,
    BoundSource,
    Executor,
    ListExpr,
    SourcePolicy,
    alias,
    call,
    let,
    member,
    param,
    parse,
    query,
    ref,
    source,
    source_call,
)
from rcql.arguments import EXPRESSION_ARGUMENTS
from rcql.operators import _REGISTRY


@pytest.fixture
def rex():
    return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])


def execute(rex, text, **params):
    return Executor(sources={'r': rex}, params=params).execute(parse(text))


@pytest.mark.parametrize('name', sorted(EXPRESSION_ARGUMENTS))
def test_named_contract_matches_actual_adapter(name):
    actual = tuple(inspect.signature(_REGISTRY[name].fn).parameters.values())[1:]
    names, defaults = EXPRESSION_ARGUMENTS[name]
    assert names == tuple(p.name for p in actual)
    assert defaults == tuple(p.default for p in actual if p.default is not inspect.Parameter.empty)


def test_every_registered_adapter_has_a_named_contract():
    assert set(_REGISTRY) == set(EXPRESSION_ARGUMENTS)


def test_source_and_return_aliases_named_args_and_lists_match_builder(rex):
    text = ('FROM $r AS r LET selected = r.CELLS(indices=[0, 2], grade=1) '
            'RETURN INDICATOR(selected) AS signal, [17/20, -2, [true, NONE]] AS constants')
    built = query(source('r'), alias(call('INDICATOR', ref('selected')), 'signal'),
                  alias([Fraction(17, 20), -2, [True, None]], 'constants'), source_alias='r',
                  bindings=(let('selected', call('CELLS', grade=1, indices=[0, 2])),))
    assert parse(text) == built
    result = Executor(sources={'r': rex}).execute(built)
    np.testing.assert_array_equal(result.named_values['signal'].values, [1, 0, 1])
    assert result.named_values['constants'] == [Fraction(17, 20), -2, [True, None]]
    assert result.native_plan['return_aliases'] == ['signal', 'constants']
    assert result.native_plan['source_alias'] == 'r'
    assert result.provenance[0]['alias'] == 'signal'
    assert result.provenance[0]['logical_operator'] == 'INDICATOR'
    json.dumps(result.native_plan, allow_nan=False)


def test_named_argument_interior_defaults_reuse_same_native_call(rex):
    x = Cochain(1, np.array([1, 2, 3]), source=rex)
    text = ('FROM $r RETURN MOMENT(right=$x, exact=true, left=$x), '
            'MOMENT($x, $x, NONE, true)')
    result = execute(rex, text, x=x)
    assert result.values == (Fraction(14), Fraction(14))
    assert result.native_plan['outputs'][0] == result.native_plan['outputs'][1]


def test_source_form_named_arguments_have_separate_contracts():
    assert parse('FROM AT(version=2, source=$r) RETURN 1').source == source_call('AT', source=param('r'), version=2)
    assert parse('FROM RCDB_GET(record_id="x", source=$store) RETURN 1').source == source_call(
        'RCDB_GET', param('store'), 'x')
    assert parse('FROM $store RETURN RCDB_GET(record_id="x")').returns[0] == call('RCDB_GET', 'x')


@pytest.mark.parametrize('text', [
    'FROM $r RETURN CELL(index=0)', 'FROM $r RETURN CELL(1, grade=1, index=0)',
    'FROM $r RETURN CELL(grade=1, grade=1, index=0)',
    'FROM $r RETURN CELL(grade=1, 0)', 'FROM $r RETURN CELL(grdae=1, index=0)',
    'FROM $r RETURN [1,]', 'FROM $r RETURN [1,2', 'FROM $r RETURN (1',
    'FROM $r RETURN 1 AS x, 2 AS x', 'FROM $r RETURN 1 AS $x',
    'FROM $r AS a LET a = 1 RETURN 1', 'FROM $r AS TRUE RETURN 1',
    'FROM $r RETURN $x.__class__', 'FROM $r RETURN $x[0]',
])
def test_malformed_composition_is_refused(text):
    with pytest.raises(SyntaxError):
        parse(text)


def test_parentheses_and_record_projection_preserve_typed_hodge_carrier(rex):
    x = Cochain(1, np.array([1., 2., 4.]), source=rex)
    result = execute(rex, 'FROM $r LET split = HODGE($x) RETURN (split).harmonic AS h, '
                           'QUADRANCE(split.harmonic)', x=x)
    assert isinstance(result.named_values['h'], Cochain)
    assert result.named_values['h'].source is rex
    assert result.provenance[0]['result_type']['grade'] == 1
    assert result.provenance[0]['result_type']['basis']['source_id'] == 'r'
    assert sum(n.get('operator') == 'HODGE' for n in result.native_plan['nodes']) == 1
    assert any(n['kind'] == 'member' for n in result.native_plan['nodes'])
    assert result.values[1] == pytest.approx(np.dot(result.values[0].values, result.values[0].values))


def test_record_keys_not_object_attributes(rex):
    assert execute(rex, 'FROM $r RETURN $record.measure', record={'measure': Fraction(2, 3)}).values == (Fraction(2, 3),)
    class SideEffect:
        @property
        def measure(self):
            pytest.fail('attribute was evaluated')
    for text in ('FROM $r RETURN $record.measure', 'EXPLAIN FROM $r RETURN $record.measure'):
        with pytest.raises(TypeError, match='record'):
            execute(rex, text, record=SideEffect())
    with pytest.raises(TypeError, match='no member'):
        execute(rex, 'FROM $r RETURN DESCRIBE().missing')


def test_member_and_list_typing_precedes_every_adapter(rex, monkeypatch):
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter executed'))
    with pytest.raises(TypeError):
        execute(rex, 'FROM $r RETURN [BETTI(0), DESCRIBE().missing]')
    denied = BoundSource(rex, SourcePolicy.allow('identity'))
    with pytest.raises(PermissionError):
        execute(denied, 'FROM $r RETURN DESCRIBE().nE')
    explained = execute(rex, 'EXPLAIN FROM $r RETURN DESCRIBE().nE AS count, [BETTI(0)]')
    assert explained.values[0]['native_plan']['return_aliases'] == ['count', None]


def test_optimizer_traverses_aliases_and_lists(rex):
    c = Chain(2, np.array([1], dtype=np.int64), source=rex)
    result = execute(rex, 'FROM $r RETURN [BOUNDARY(grade=1, values=BOUNDARY(2,$c))] AS z', c=c)
    assert result.rewrites
    np.testing.assert_array_equal(result.named_values['z'][0].values, np.zeros(3))


def test_duplicate_or_nested_builder_aliases_are_refused(rex):
    for built in (query(source('r'), alias(1, 'x'), alias(2, 'x')),
                  query(source('r'), [alias(1, 'x')])):
        with pytest.raises((TypeError, ValueError)):
            Executor(sources={'r': rex}).execute(built)


def test_list_nodes_keep_single_let_evaluation(rex):
    result = execute(rex, 'FROM $r LET x = [BETTI(0)] RETURN x, x')
    assert result.values[0] is result.values[1]
    assert len([n for n in result.native_plan['nodes'] if n['kind'] == 'list']) == 1
    assert isinstance(parse('FROM $r RETURN []').returns[0], ListExpr)
    assert isinstance(alias(1, 'x'), Alias)
    assert member(param('record'), 'x') == parse('FROM $r RETURN $record.x').returns[0]


def test_named_arguments_evaluate_in_declared_parameter_order(rex, monkeypatch):
    from rcql.operators import get_operator
    observed = []
    def observe(name):
        observed.append(name)
        return get_operator(name)
    monkeypatch.setattr('rcql.executor.get_operator', observe)
    flow = Cochain(1, np.array([1., 2., 3.]), source=rex)
    execute(rex, 'FROM $r RETURN QUADRANCE(metric=METRIC(1), values=HARMONIC($flow))', flow=flow)
    assert observed == ['HARMONIC', 'METRIC', 'QUADRANCE']


def test_mutation_source_alias_builder_and_parser_agree():
    from rcql import mutation
    built = mutation(source('db'), 'copy', call('RCDB_GET', 'original'), source_alias='s')
    assert built == parse('FROM $db AS s MUTATE "copy" SET state=s.RCDB_GET("original") COMMIT')
    with pytest.raises(ValueError, match='identifier'):
        mutation(source('db'), 'copy', param('state'), source_alias='not an alias')


@pytest.mark.parametrize('make', [lambda: ListExpr((1,)), lambda: ListExpr([]),
                                 lambda: Alias(1, 'x'), lambda: member(1, '_private')])
def test_builder_nodes_refuse_malformed_expression_children(make):
    with pytest.raises((TypeError, ValueError)):
        make()
