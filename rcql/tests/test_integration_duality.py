"""INTEGRATE transports native duality and exact Stokes through typed plans."""
import json
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph

from rcql import BoundSource, Executor, SourcePolicy, alias, call, param, parse, query, source


def make():
    return RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])


def run(rex, text, **params):
    return Executor(sources={'r': rex}, params=params).execute(parse(text))


@pytest.mark.parametrize('case', ['branching', 'witness', 'parallel', 'face', 'grade3'])
@pytest.mark.parametrize('columns', [None, 2])
def test_exact_stokes_uses_canonical_boundary_and_dual_coboundary(case, columns):
    if case == 'branching':
        rex, grade = RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2]), 1
    elif case == 'witness':
        rex, grade = RexGraph.from_hypergraph([0, 1, 3], [0, 0, 1]), 1
    elif case == 'parallel':
        rex, grade = RexGraph.from_graph([0, 0, 1], [1, 1, 2]), 1
    elif case == 'face':
        rex, grade = make(), 2
    else:
        rex, grade = RexGraph.from_cells(solid_octahedron_3rex()), 3
    matrix = rex.graded_boundaries()[grade-1]
    def coefficients(n):
        shape = (n,) if columns is None else (n, columns)
        return np.asarray([Q(i-2, i+1) for i in range(int(np.prod(shape)))], dtype=object).reshape(shape)
    omega = Cochain(grade-1, coefficients(matrix.shape[0]), source=rex)
    chain = Chain(grade, coefficients(matrix.shape[1]), source=rex)
    result = run(rex, f'FROM $r RETURN INTEGRATE(COBOUNDARY({grade-1}, $omega), $chain, true) AS left, '
                 f'INTEGRATE($omega, BOUNDARY({grade}, $chain), true) AS right', omega=omega, chain=chain)
    assert isinstance(result.values[0], Q)
    assert result.values[0] == result.values[1]
    assert [x.value for x in result.exactness] == ['rational', 'rational']
    assert all(n['physical']['method'] == 'rational-dual-pairing'
               for n in result.native_plan['nodes'] if n.get('operator') == 'INTEGRATE')
    json.dumps(result.native_plan, allow_nan=False)


def test_text_builder_named_outputs_and_method_contract_agree():
    rex = make()
    omega, chain = Cochain(1, np.array([1, 2, 3]), source=rex), Chain(1, np.array([-1, 3, 2]), source=rex)
    text = 'FROM $r RETURN INTEGRATE(chain=$c, cochain=$w, exact=true) AS integral'
    built = query(source('r'), alias(call('INTEGRATE', cochain=param('w'), chain=param('c'), exact=True), 'integral'))
    assert parse(text) == built
    result = run(rex, text, w=omega, c=chain)
    assert result.named_values == {'integral': Q(11)}
    method = result.execution[0]['methods'][0]
    assert method['method'] == 'rational-dual-pairing'
    assert method['pairing'] == 'bilinear-cochain-chain'


@pytest.mark.parametrize('mismatch', ['variance', 'grade', 'basis', 'source', 'shape', 'raw', 'float-exact'])
@pytest.mark.parametrize('explain', [False, True])
def test_invalid_dual_contract_fails_before_every_adapter(mismatch, explain, monkeypatch):
    rex = make()
    w, c = Cochain(1, np.array([1, 2, 3]), source=rex), Chain(1, np.array([1, 2, 3]), source=rex)
    if mismatch == 'variance':
        c = w
    elif mismatch == 'grade':
        c = Chain(0, c.values, source=rex)
    elif mismatch == 'basis':
        c = Chain(1, c.values, ('b', 'a', 'c'), rex)
    elif mismatch == 'source':
        c = Chain(1, c.values, source=make())
    elif mismatch == 'shape':
        c = c.with_values(c.values[:, None])
    elif mismatch == 'raw':
        c = c.values
    elif mismatch == 'float-exact':
        c = c.with_values(c.values.astype(float))
    monkeypatch.setattr('rcql.executor.get_operator', lambda *a: pytest.fail('adapter ran before validation'))
    with pytest.raises((TypeError, ValueError)):
        run(rex, ('EXPLAIN ' if explain else '') + 'FROM $r RETURN INTEGRATE($w, $c, true)', w=w, c=c)


def test_complex_pairing_remains_distinct_from_moment():
    rex = make()
    w = Cochain(1, np.array([1j, 1, 0]), source=rex)
    c = Chain(1, np.array([1, 2j, 0]), source=rex)
    result = run(rex, 'FROM $r RETURN INTEGRATE($w, $c), MOMENT($w, $v)',
                 w=w, c=c, v=Cochain(1, c.values, source=rex))
    assert result.values == (3j, 1j)
    assert result.provenance[0]['result_type']['domain'] == 'complex'


def test_read_policy_is_checked_even_for_zero_pairing():
    rex = make()
    denied = BoundSource(rex, SourcePolicy.allow('identity'))
    with pytest.raises(PermissionError):
        run(denied, 'FROM $r RETURN INTEGRATE(ZERO(1), ZERO(1,"chain"), true)')


def test_branching_stokes_has_an_independent_hand_value():
    rex = RexGraph.from_hypergraph([0, 4, 6], [2, 0, 1, 3, 0, 2])
    omega = Cochain(0, np.array([1, 2, 3, 4]), source=rex)
    chain = Chain(1, np.array([3, 2]), source=rex)
    # d omega = (-2/3, 2); B chain = (-1, 1, -1, 1).
    # Both pairings are exactly 2, retaining the primary four ary share 1/3.
    result = run(rex, 'FROM $r RETURN INTEGRATE(COBOUNDARY(0,$w),$c,true), '
                 'INTEGRATE($w,BOUNDARY(1,$c),true)', w=omega, c=chain)
    assert result.values == (Q(2), Q(2))
