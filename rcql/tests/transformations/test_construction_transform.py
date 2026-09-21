from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rcql import (ProgramTransformation, NameRelation, Executor, parse, bind, SourcePolicy,
                  Program, ProgramInput, ProgramStep, OutputRef, QueryCache)
from rcql.ast import Call, ListExpr, Parameter, Literal
from rcql.program_codec import dumps, loads
from rexgraph.graph import RexGraph


def add(name='add', inputs=('x', 'y'), defaults=()):
    return NameRelation(name, inputs, Call('SUM', (ListExpr(tuple(Parameter(k) for k in inputs)),)), defaults)


@pytest.fixture
def rex():
    return RexGraph.from_cells([3, [[0, 1, 2]]])


@pytest.fixture
def binding(rex):
    return bind('r', rex, SourcePolicy.allow('read'))


@pytest.mark.parametrize('fixed', [{'y': Q(2**140+1, 7)}, {'x': Q(1, 3), 'y': Q(2, 7)}])
def test_exact_specialization(binding, fixed):
    n = add()
    original = n.to_bytes()
    t = ProgramTransformation.specialize(n, fixed)
    args = {k: Q(1, 11) for k in n.inputs if k not in fixed}
    candidate = t.compile(binding, n, args)
    assert candidate.apply(binding, args) == n.apply(binding, {**args, **fixed})
    assert n.to_bytes() == original
    proof = t.verify(n)
    assert all(not any(row) for row in proof['tower_closure_residuals'])
    assert 'cross tower chain map' in proof['excluded_claims']
    assert proof['fixed_inputs']


def test_repeated_parameter_changes_topology(binding):
    n = NameRelation('twice', ('x',), Call('SUM', (ListExpr((Parameter('x'), Parameter('x'))),)))
    t = ProgramTransformation.specialize(n, {'x': 7})
    candidate = t.compile(binding, n, [])
    assert candidate.apply(binding, []) == 14
    assert n.topology().boundary_tower().sizes != candidate.topology().boundary_tower().sizes
    assert 'components' not in t.verify()


def test_defaults_and_binding_order(binding):
    n = add(defaults=(('x', 4), ('y', 5)))
    a = ProgramTransformation.specialize(n, {'y': 2, 'x': 3})
    b = ProgramTransformation.specialize(n, {'x': 3, 'y': 2})
    assert a.to_bytes() == b.to_bytes()
    assert a.compile(binding, n, {}).apply(binding, {}) == 5
    partial = ProgramTransformation.specialize(n, {'x': 11}).compile(binding, n, {})
    assert partial.defaults == (('y', 5),) and partial.apply(binding, {}) == 16


@pytest.mark.parametrize('fixed', [{}, {'z': 1}, {'x': object()}, {'x': float('nan')}, [1], {1: 2}])
def test_invalid_specialization(fixed):
    with pytest.raises((ValueError, TypeError)):
        ProgramTransformation.specialize(add(), fixed)


def test_specialization_does_not_rebind_capture():
    with pytest.raises(ValueError):
        ProgramTransformation.specialize(add().bind('x', 1), {'x': 2})


@pytest.mark.parametrize('rule', ['specialize', 'compose'])
def test_verifier_independent_of_construction(binding, monkeypatch, rule):
    n = add()
    if rule == 'specialize':
        t = ProgramTransformation.specialize(n, {'y': 3})
    else:
        t = ProgramTransformation.compose(n, add('next', ('a', 'b')), ('a', 'b'))
    def fail(*a, **kw):
        raise AssertionError('verifier used candidate constructor')
    monkeypatch.setattr(NameRelation, 'bind', fail)
    monkeypatch.setattr(NameRelation, 'then', fail)
    restored = ProgramTransformation.from_bytes(t.to_bytes())
    assert restored.verify(n) == t.verify(n)
    assert restored.compile(binding, n, [2] if rule == 'specialize' else [2, 3])


@pytest.mark.parametrize('ports', ['a', ['a', 'b']])
def test_composition_is_declared_value_chain(binding, ports):
    n, g = add(), add('following', ('a', 'b'), (('b', Q(2, 7)),))
    t = ProgramTransformation.compose(n, g, ports)
    connected = (ports,) if isinstance(ports, str) else ports
    args = {'x': Q(1, 3), 'y': Q(1, 5)}
    candidate = t.compile(binding, n, args)
    value = n.apply(binding, args)
    expected = g.apply(binding, {k: value for k in connected})
    assert candidate.apply(binding, args) == expected
    assert candidate.apply(binding, args) != value
    assert t.verify()['following_digest'] == g.coefficient_digest


def test_capture_and_stage_collisions(binding):
    left = add('left', ('x', 'amount')).bind('amount', 7)
    right = add('right', ('input', 'amount')).bind('amount', 11)
    left = left.then(add('double', ('arg', 'other')).bind('other', 2), 'arg')
    right = right.then(add('last', ('arg', 'other')).bind('other', 3), 'arg')
    t = ProgramTransformation.compose(left, right, 'input')
    candidate = t.compile(binding, left, [5])
    assert candidate.apply(binding, [5]) == 28
    proof = t.verify()
    assert len(proof['capture_mapping']['left']) == len(left.bound_values)
    assert len(proof['capture_mapping']['right']) == len(right.bound_values)
    assert len(proof['stage_mapping']) == len(left.stages)+len(right.stages)


@pytest.mark.parametrize('ports', [[], ['a', 'a'], ['missing'], [1], object()])
def test_invalid_composition_ports(ports):
    with pytest.raises((ValueError, TypeError)):
        ProgramTransformation.compose(add(), add('next', ('a', 'b')), ports)


def test_unconnected_input_collision():
    with pytest.raises(ValueError, match='collide'):
        ProgramTransformation.compose(add(), add('next', ('a', 'x')), 'a')


def test_construction_does_not_execute_and_intermediate_runs_once(rex, monkeypatch):
    from rcql.operators import _REGISTRY
    op = _REGISTRY['SUM']
    calls = []
    def counted(*args):
        calls.append(1)
        return op.fn(*args)
    monkeypatch.setitem(_REGISTRY, 'SUM', replace(op, fn=counted))
    n, g = add(), add('double', ('a', 'b'))
    e = Executor(sources={'r': rex}, params={'n':n, 'g':g})
    query = '''FROM $r LET t=TRANSFORM_COMPOSE($n,$g,["a","b"])
        RETURN TRANSFORM_COMPILE(t,$n,[2,3])'''
    assert e.execute(parse('EXPLAIN '+query)).execution == () and not calls
    candidate = e.execute(parse(query)).values[0]
    assert not calls
    assert candidate.apply(bind('r', rex, SourcePolicy.allow('read')), [2, 3]) == 10
    assert len(calls) == 2


def test_result_type_can_change(binding):
    left = NameRelation('list', ('x',), ListExpr((Parameter('x'), Parameter('x'))))
    right = NameRelation.operator('SUM')
    t = ProgramTransformation.compose(left, right, 'values')
    assert t.compile(binding, left, [Q(1, 7)]).apply(binding, [Q(1, 7)]) == Q(2, 7)


@pytest.mark.parametrize('change', ['stage', 'body', 'capture', 'derivation', 'ports', 'following', 'proof', 'tower'])
def test_tampering_refused(change):
    n = add('source', ('x', 'amount')).bind('amount', 2)
    g = add('following', ('a', 'amount')).bind('amount', 3)
    t = ProgramTransformation.compose(n, g, 'a')
    data = loads(t.to_bytes())
    candidate = NameRelation.from_bytes(data['target'])
    if change == 'stage':
        key, expr = candidate.stages[0]
        candidate = replace(candidate, stages=((key, Call('SUM', (ListExpr((expr, Literal(1))),))),))
    elif change == 'body':
        candidate = replace(candidate, body=Call('SUM', (ListExpr((candidate.body, Literal(1))),)))
    elif change == 'capture':
        candidate = candidate.rebind(candidate.bound_values[0][0], 999)
    elif change == 'derivation':
        candidate = replace(candidate, derivation=())
    elif change == 'ports':
        data['arguments'] = (g.to_bytes(), ('amount',))
    elif change == 'following':
        data['arguments'] = (g.rebind('amount', 999).to_bytes(), ('a',))
    elif change == 'proof':
        data['certificate']['scope'] = 'unconditional equivalence'
    else:
        data['after']['tower']['digest'] = '0'*64
    if change in {'stage','body','capture','derivation'}:
        from rcql.program_transformation import _check_composition
        with pytest.raises(ValueError):
            _check_composition(n, candidate, data['arguments'])
        data['target'] = candidate.to_bytes()
    with pytest.raises(ValueError):
        ProgramTransformation.from_bytes(dumps(data))


@pytest.mark.parametrize('rule', ['specialize','compose'])
@pytest.mark.parametrize('format', ['bytes','rcbd','safetensors','rcdb'])
def test_roundtrip(binding, tmp_path, rule, format):
    n = add()
    t = (ProgramTransformation.specialize(n, {'y':3}) if rule=='specialize' else
         ProgramTransformation.compose(n, add('double', ('a','b')), ('a','b')))
    args = [2] if rule=='specialize' else [2,3]
    if format == 'bytes':
        restored = ProgramTransformation.from_bytes(t.to_bytes())
    else:
        record = t.to_record()
        if format == 'rcbd':
            from rexgraph.io.bundle import save_rcbd, load_rcbd
            path = tmp_path/'t.rcbd'; save_rcbd(path,record); record = load_rcbd(path)
        elif format == 'safetensors':
            from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
            path = tmp_path/'t.safetensors'; rex_to_safetensors(record,path); record = safetensors_to_rex(path)
        else:
            from rcdb import open_store
            uri = 'file://'+str(tmp_path/'store')
            db = open_store(uri); db.commit_mutation('t',record,expected_version=0,analytics=False); db.close()
            db = open_store(uri); record = db.read_record('t',version=1).value
            assert db.verify_commits('t'); db.close()
        restored = ProgramTransformation.from_record(record)
    assert restored.to_bytes() == t.to_bytes()
    assert restored.compile(binding,n,args).apply(binding,args) == (5 if rule=='specialize' else 10)
    assert restored.topology().corelations() == t.topology().corelations()


def test_meta_program_cache_and_original_mismatch(rex,binding):
    from rcdb import open_store
    n,g=add(),add('double',('a','b'))
    query=parse('FROM $r RETURN TRANSFORM_COMPOSE($n,$g,["a","b"])')
    program=Program('compose',(ProgramStep('candidate',query),),
        (ProgramInput('n','NameRelation'),ProgramInput('g','NameRelation')),
        (('transform',OutputRef('candidate')),))
    e=Executor(sources={'r':rex},params={'n':n,'g':g})
    t=e.execute_program(Program.from_record(program.to_record())).values[0]
    assert e.execute_program(program,explain=True)['evaluation'].startswith('none')
    with pytest.raises(ValueError,match='source declaration'):
        t.compile(binding,n.named('other'),[2,3])
    db=open_store('memory://')
    try:
        cache=QueryCache(db)
        e.execute_cached(query,cache)
        assert e.execute_cached(query,cache).native_plan['cache']['hit']
        e.params['g']=g.named('different')
        assert not e.execute_cached(query,cache).native_plan['cache']['hit']
    finally:
        db.close()


def case(name,rex):
    return {'TRANSFORM_SPECIALIZE':(add(),{'y':Q(2,7)}),
            'TRANSFORM_COMPOSE':(add(),add('double',('a','b')),('a','b'))}[name]


@pytest.mark.parametrize('name',['TRANSFORM_SPECIALIZE','TRANSFORM_COMPOSE'])
def test_operator_contract(name,rex):
    from rcql import call,query,source
    from rcql.operators import get_operator
    args=case(name,rex)
    direct=get_operator(name).fn(rex,*args)
    result=Executor(sources={'r':rex}).execute(query(source('r'),call(name,*args)))
    assert result.values[0].coefficient_digest==direct.coefficient_digest


def test_specialization_verifies_every_retained_field():
    from rcql.program_transformation import _check_specialization
    n=add()
    target=n.bind('y',3)
    for changed in (target.rebind('y',4),replace(target,defaults=(('x',2),)),
                    replace(target,body=Call('SUM',(ListExpr((target.body,Literal(1))),)))):
        with pytest.raises(ValueError):
            _check_specialization(n,changed,(('y',3),))


@pytest.mark.parametrize('rule',['specialize','compose'])
def test_permissions_and_native_source_remain_live(rex,rule):
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.tensor_field import TensorField,FieldSource
    from rcql import BoundSource
    n=NameRelation.operator('NATIVE_RESPONSE')
    # Capture only a finite optional input, never the native field.
    if rule=='specialize':
        from rcql.arguments import EXPRESSION_ARGUMENTS
        labels,defaults=EXPRESSION_ARGUMENTS['NATIVE_RESPONSE']
        assert defaults
        t=ProgramTransformation.specialize(n,{labels[-1]:defaults[-1]})
    else:
        identity=NameRelation('identity',('value',),Parameter('value'))
        t=ProgramTransformation.compose(n,identity,'value')
    space=NativeFieldCalculus.from_rex(rex).complex.spaces[1]
    field=TensorField(space,[1],source=FieldSource(rex),grade=1,variance='chain')
    b=bind('r',rex,SourcePolicy.allow('read'))
    candidate=t.compile(b,n,[field])
    other=RexGraph.from_cells([3,[[0,1,2]]])
    with pytest.raises(ValueError):
        candidate.apply(bind('r',other,SourcePolicy.allow('read')),[field])
    denied=Executor(sources={'r':BoundSource(rex,SourcePolicy.allow())},params={'n':candidate,'f':field})
    with pytest.raises(PermissionError):
        denied.execute(parse('FROM $r RETURN NAME_APPLY($n,[$f])'))


def test_nested_compositions_retain_shared_stages(binding):
    left=add('initial',('x','amount')).bind('amount',1)
    double=add('double',('a','b'))
    for _ in range(8):
        t=ProgramTransformation.compose(left,double,('a','b'))
        left=t.compile(binding,left,[2])
    assert left.apply(binding,[2])==3*2**8
    assert len(left.stages)==8


@pytest.mark.parametrize('rule',['specialize','compose'])
def test_recursive_transformation_values(rex,binding,rule):
    from rcql import RecursiveDefinition,RecursiveProgram,RecursionResult,ValueKind,recur
    from rcql.ast import Comparison
    remaining,value=Parameter('remaining'),Parameter('transform')
    program=RecursiveProgram('carry',(RecursiveDefinition('carry',
        (ProgramInput('remaining',ValueKind.EXACT_INTEGER.value),
         ProgramInput('transform',ValueKind.PROGRAM_TRANSFORMATION.value)),
        Comparison('==',remaining,Literal(0)),value,
        recur('carry',Call('SUM',(ListExpr((remaining,Literal(-1))),)),value),
        ProgramInput('result',ValueKind.PROGRAM_TRANSFORMATION.value),decreases='remaining'),))
    t=(ProgramTransformation.specialize(add(),{'y':3}) if rule=='specialize' else
       ProgramTransformation.compose(add(),add('double',('a','b')),('a','b')))
    result=program.execute(binding,'carry',[3,t])
    restored=RecursionResult.from_record(result.to_record(),result.dependencies)
    assert restored.value.to_bytes()==t.to_bytes()


def test_recursive_program_constructs_successive_names(binding):
    from rcql import RecursiveDefinition,RecursiveProgram,RecursionResult,ValueKind,recur
    from rcql.ast import Comparison
    count,operation,following=Parameter('count'),Parameter('operation'),Parameter('following')
    proposal=Call('TRANSFORM_COMPOSE',(operation,following,ListExpr((Literal('a'),Literal('b')))))
    candidate=Call('TRANSFORM_COMPILE',(proposal,operation,ListExpr((Literal(2),))))
    program=RecursiveProgram('grow',(RecursiveDefinition('grow',
        (ProgramInput('count',ValueKind.EXACT_INTEGER.value),ProgramInput('operation','NameRelation'),
         ProgramInput('following','NameRelation')),
        Comparison('==',count,Literal(0)),operation,
        recur('grow',Call('SUM',(ListExpr((count,Literal(-1))),)),candidate,following),
        ProgramInput('result','NameRelation'),decreases='count'),))
    original=add('seed',('x','amount')).bind('amount',1)
    result=program.execute(binding,'grow',[4,original,add('double',('a','b'))])
    restored=RecursionResult.from_record(result.to_record(),result.dependencies)
    assert restored.value.apply(binding,[3])==64
    assert len(restored.value.stages)==4
    assert original.apply(binding,[3])==4
