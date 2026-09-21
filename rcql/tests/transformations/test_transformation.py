from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest

from rcql import (NameRelation, ProgramTransformation, Executor, parse, bind,
                  SourcePolicy, BoundSource, Program, ProgramInput, ProgramStep,
                  OutputRef, QueryCache, SnapshotContext, SourceSelection)
from rcql.ast import Call, ListExpr, Parameter, Literal
from rcql.program_codec import dumps, loads
from rexgraph.graph import RexGraph


def base():
    return NameRelation('add', ('x', 'amount'), Call('SUM', (
        ListExpr((Parameter('x'), Parameter('x'), Parameter('amount'))),))).bind('amount', Q(2**140+1, 7))


@pytest.fixture
def rex():
    return RexGraph.from_cells([3, [[0, 1, 2]]])


@pytest.fixture
def binding(rex):
    return bind('input', rex, SourcePolicy.allow('*'))


@pytest.mark.parametrize('rule,args', [('port', ('x', 'value')), ('alias', ('renamed',))])
def test_immutable_exact_mapping(binding, rule, args):
    source = base()
    original = source.to_bytes()
    t = ProgramTransformation.name(source, rule, args)
    certificate = t.verify(source)
    assert not any(certificate['chain_residuals'])
    assert certificate['captures'][0][:2] == ('amount', 'amount')
    assert len(certificate['grade_mapping']) == 3
    assert certificate['scope'] == 'declaration under explicit interface renaming'
    candidate = t.compile(binding, source, [Q(3, 11)])
    assert candidate.apply(binding, [Q(3, 11)]) == source.apply(binding, [Q(3, 11)]) == Q(6, 11)+Q(2**140+1, 7)
    assert source.to_bytes() == original
    assert t.coefficient_digest == ProgramTransformation.name(source, rule, args).coefficient_digest
    with pytest.raises(FrozenInstanceError):
        t.data = b''


def test_verifier_does_not_call_construction_method(monkeypatch):
    source = base()
    t = ProgramTransformation.name(source, 'port', ['x', 'z'])
    def forbidden(*args, **kw):
        raise AssertionError('verification called construction')
    monkeypatch.setattr(NameRelation, 'rename', forbidden)
    assert t.verify(source)['ports'] == (('x', 'z'),)


def test_shared_stages_and_defaults(binding):
    n = NameRelation('twice', ('x',), Call('SUM', (ListExpr((Parameter('x'), Parameter('x'))),)),
                     defaults=(('x', Q(1, 7)),))
    for _ in range(6):
        n = n.then(
            NameRelation('double', ('y',), Call('SUM', (ListExpr((Parameter('y'), Parameter('y'))),))), 'y')
    t = ProgramTransformation.name(n, 'port', ['x', 'input'])
    renamed = t.compile(binding, n, {})
    assert renamed.apply(binding, {}) == Q(128, 7)
    assert len(renamed.stages) == 6
    assert t.topology().port_realization().entries == n.topology().port_realization().entries


@pytest.mark.parametrize('rule,args', [('missing', []), ('port', ['x']), ('port', ['x', 'x']),
    ('port', ['missing', 'new']), ('port', ['x', 'amount']), ('port', ['x', 'bad name']),
    ('port', 'x'), ('port', ['x', 3]), ('alias', []), ('alias', ['bad name'])])
def test_invalid_transform_refused(rule, args):
    with pytest.raises((TypeError, ValueError)):
        ProgramTransformation.name(base(), rule, args)


@pytest.mark.parametrize('change', ['target', 'source', 'ports', 'captures', 'scope', 'chain', 'tower', 'rule', 'version', 'extra'])
def test_tamper_refused(change):
    t = ProgramTransformation.name(base(), 'port', ['x', 'z'])
    data = loads(t.to_bytes())
    if change == 'target':
        target = NameRelation.from_bytes(data['target']).rebind('amount', 9)
        data['target'] = target.to_bytes()
    elif change == 'source':
        data['source'] = base().rebind('amount', 9).to_bytes()
    elif change == 'ports':
        data['certificate']['ports'] = (('x', 'wrong'),)
    elif change == 'captures':
        data['certificate']['captures'] = ()
    elif change == 'scope':
        data['certificate']['scope'] = 'arbitrary program equivalence'
    elif change == 'chain':
        data['certificate']['components'] = ((), (), ())
    elif change == 'tower':
        boundaries = list(data['after']['tower']['boundaries'])
        boundaries[1] = (*boundaries[1], (0, 0, 1))
        data['after']['tower']['boundaries'] = tuple(boundaries)
    elif change == 'rule':
        data['rule'] = 'inline'
    elif change == 'version':
        data['version'] = True
    else:
        data['authority'] = '*'
    with pytest.raises((TypeError, ValueError)):
        ProgramTransformation.from_bytes(dumps(data))


def test_source_mismatch(binding):
    n = base()
    t = ProgramTransformation.name(n, 'port', ['x', 'z'])
    with pytest.raises(ValueError, match='source declaration'):
        t.compile(binding, n.rebind('amount', 1), [2])


def test_no_target_execution_and_explicit_application(rex, monkeypatch):
    from rcql.operators import _REGISTRY
    n = base()
    original = _REGISTRY['SUM']
    calls = []
    def counted(*args):
        calls.append(1)
        return original.fn(*args)
    monkeypatch.setitem(_REGISTRY, 'SUM', replace(original, fn=counted))
    e = Executor(sources={'r': rex}, params={'n': n})
    text = 'FROM $r LET t=TRANSFORM_NAME($n,"port",["x","z"]) RETURN TRANSFORM_COMPILE(t,$n,[3])'
    explained = e.execute(parse('EXPLAIN '+text))
    assert explained.execution == () and not calls
    candidate = e.execute(parse(text)).values[0]
    assert not calls
    assert Executor(sources={'r':rex}, params={'p':candidate}).execute(parse(
        'FROM $r RETURN NAME_APPLY($p,[3])')).values[0] == 6+Q(2**140+1, 7)
    assert len(calls) == 1


@pytest.mark.parametrize('format', ['bytes', 'rcbd', 'safetensors', 'rcdb'])
def test_portable(tmp_path, format):
    n = base()
    t = ProgramTransformation.name(n, 'port', ['x', 'z'])
    if format == 'bytes':
        restored = ProgramTransformation.from_bytes(t.to_bytes())
    else:
        record = t.to_record()
        if format == 'rcbd':
            from rexgraph.io.bundle import save_rcbd, load_rcbd
            path = tmp_path/'t.rcbd'
            save_rcbd(path, record)
            record = load_rcbd(path)
        elif format == 'safetensors':
            from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
            path = tmp_path/'t.safetensors'
            rex_to_safetensors(record, path)
            record = safetensors_to_rex(path)
        else:
            from rcdb import open_store
            uri = 'file://'+str(tmp_path/'store')
            db = open_store(uri)
            db.commit_mutation('t', record, expected_version=0, analytics=False)
            db.close()
            db = open_store(uri)
            try:
                record = db.read_record('t', version=1).value
                assert db.verify_commits('t')
            finally:
                db.close()
        restored = ProgramTransformation.from_record(record)
    assert restored.coefficient_digest == t.coefficient_digest
    assert restored.topology().corelations() == t.topology().corelations()


def test_record_carrier_tamper():
    t = ProgramTransformation.name(base(), 'port', ['x', 'z'])
    record = t.to_record()
    record.attach_metadata(1, 0, 'unexpected', 'changed')
    with pytest.raises(ValueError, match='carrier'):
        ProgramTransformation.from_record(record)


def test_permissions_not_restored(rex):
    n = base()
    t = ProgramTransformation.from_record(ProgramTransformation.name(n, 'port', ['x','z']).to_record())
    e = Executor(sources={'r': BoundSource(rex, SourcePolicy.allow())}, params={'n':n,'t':t})
    with pytest.raises(PermissionError):
        e.execute(parse('FROM $r RETURN TRANSFORM_COMPILE($t,$n,[2])'))


def test_source_and_evidence_validation(rex):
    from rcdb import open_store
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.tensor_field import FieldSource, TensorField
    n = NameRelation.operator('NATIVE_RESPONSE')
    t = ProgramTransformation.name(n, 'port', ['field','input'])
    db = open_store('memory://')
    try:
        db.commit_mutation('data', rex, expected_version=0, tx_time=10, analytics=False)
        context = SnapshotContext.select(db, (SourceSelection('r','data'),), cutoff=20)
        selected = context.sources['r']
        space = NativeFieldCalculus.from_rex(selected.value).complex.spaces[1]
        field = TensorField(space,[1],source=FieldSource(selected.value,'data',1),grade=1,variance='chain')
        e = Executor(sources=context.sources,params={'n':n,'t':t,'x':field},evidence=context)
        candidate = e.execute(parse('FROM $r RETURN TRANSFORM_COMPILE($t,$n,[$x])')).values[0]
        assert candidate.inputs[0] == 'input'
        db.commit_mutation('data', rex, expected_version=1, tx_time=30, analytics=False)
        future = db.read_record('data',version=2)
        e.params['x'] = replace(field,source=FieldSource(future.value,'data',2))
        with pytest.raises(ValueError):
            e.execute(parse('FROM $r RETURN TRANSFORM_COMPILE($t,$n,[$x])'))
        other = RexGraph.from_cells([3, [[0,1,2]]])
        with pytest.raises(ValueError):
            t.compile(bind('other',other,SourcePolicy.allow('*')),n,[field])
    finally:
        db.close()


def test_meta_program_persists_without_running_target(rex):
    p = Program('rename', (ProgramStep('candidate',parse(
        'FROM $r RETURN TRANSFORM_NAME($operation,"port",["x","z"])')),),
        (ProgramInput('operation','NameRelation'),), (('transformation',OutputRef('candidate')),))
    p = Program.from_record(p.to_record())
    n = base()
    e = Executor(sources={'r':rex},params={'operation':n})
    t = e.execute_program(p).values[0]
    assert isinstance(t,ProgramTransformation)
    assert e.execute_program(p,explain=True)['evaluation'].startswith('none')
    assert ProgramTransformation.from_record(t.to_record()).verify(n)['ports'] == (('x','z'),)


def test_cache_roundtrip_and_identity(rex):
    from rcdb import open_store
    db = open_store('memory://')
    try:
        cache = QueryCache(db)
        e = Executor(sources={'r':rex},params={'n':base()})
        q = parse('FROM $r RETURN TRANSFORM_NAME($n,"port",["x","z"])')
        first = e.execute_cached(q,cache)
        second = e.execute_cached(q,cache)
        assert second.native_plan['cache']['hit']
        assert second.values[0].coefficient_digest == first.values[0].coefficient_digest
        e.params['n'] = base().rebind('amount',1)
        third = e.execute_cached(q,cache)
        assert not third.native_plan['cache']['hit']
        assert third.values[0].coefficient_digest != first.values[0].coefficient_digest
    finally:
        db.close()


def test_recursive_transformation_result_restores(rex, binding):
    from rcql import RecursiveDefinition, RecursiveProgram, RecursionResult, ValueKind, recur
    from rcql.ast import Comparison
    n, t = Parameter('remaining'), Parameter('transform')
    program = RecursiveProgram('carry', (RecursiveDefinition('carry',
        (ProgramInput('remaining',ValueKind.EXACT_INTEGER.value),
         ProgramInput('transform',ValueKind.PROGRAM_TRANSFORMATION.value)),
        Comparison('==',n,Literal(0)), t,
        recur('carry',Call('SUM',(ListExpr((n,Literal(-1))),)),t),
        ProgramInput('result',ValueKind.PROGRAM_TRANSFORMATION.value), decreases='remaining'),))
    transform = ProgramTransformation.name(base(),'port',['x','z'])
    result = program.execute(binding,'carry',[3,transform])
    restored = RecursionResult.from_record(result.to_record(),result.dependencies)
    assert restored.value.coefficient_digest == transform.coefficient_digest
    assert restored.coefficient_digest == result.coefficient_digest


def test_stage_capture_collision(binding):
    n = base().then(NameRelation('twice',('p',),Call('SUM',(ListExpr((Parameter('p'),Parameter('p'))),))),'p')
    with pytest.raises(ValueError):
        ProgramTransformation.name(n,'port',['x',n.stages[0][0]])


@pytest.mark.parametrize('operator', ['MODEL_TRAIN','RCDB_GET','PROGRAM_RUN'])
def test_effectful_or_controlled_names_refused(operator):
    with pytest.raises((TypeError,ValueError)):
        ProgramTransformation.name(NameRelation.operator(operator),'alias',['unsafe'])


def test_malformed_arguments_refused():
    t = ProgramTransformation.name(base(),'port',['x','z'])
    data = loads(t.to_bytes())
    data['arguments'] = 'xz'
    with pytest.raises(TypeError):
        ProgramTransformation.from_bytes(dumps(data))


def test_name_can_propose_its_own_next_declaration(rex):
    meta = NameRelation('reflect',('operation',),Call('TRANSFORM_NAME',(
        Parameter('operation'),Literal('port'),ListExpr((Literal('operation'),Literal('subject'))))))
    original = meta.to_bytes()
    e = Executor(sources={'r':rex},params={'meta':meta})
    result = e.execute(parse('''FROM $r
        LET proposal=NAME_APPLY($meta,[$meta])
        LET candidate=TRANSFORM_COMPILE(proposal,$meta,[$meta])
        RETURN proposal,candidate'''))
    proposal,candidate = result.values
    assert candidate.inputs == ('subject',)
    assert meta.to_bytes() == original and meta.inputs == ('operation',)
    assert candidate.coefficient_digest != meta.coefficient_digest
    assert proposal.verify(meta)['ports'] == (('operation','subject'),)
    restored = ProgramTransformation.from_record(proposal.to_record())
    assert restored.coefficient_digest == proposal.coefficient_digest
    followup = Executor(sources={'r':rex},params={'candidate':candidate,'meta':meta}).execute(parse(
        'FROM $r RETURN NAME_APPLY($candidate,[$meta])')).values[0]
    assert followup.coefficient_digest == proposal.coefficient_digest


def case(name, rex):
    n = base()
    t = ProgramTransformation.name(n,'port',['x','z'])
    return {'TRANSFORM_NAME':(n,'port',['x','z']), 'TRANSFORM_VERIFY':(t,n),
            'TRANSFORM_COMPILE':(t,n,[3]), 'TRANSFORM_RECORD':(t,),
            'TRANSFORM_READ':(t.to_record(),), 'TRANSFORM_TOPOLOGY':(t,)}[name]


@pytest.mark.parametrize('name', ['TRANSFORM_NAME','TRANSFORM_VERIFY','TRANSFORM_COMPILE',
                                  'TRANSFORM_RECORD','TRANSFORM_READ','TRANSFORM_TOPOLOGY'])
def test_direct_and_typed_contract(rex,name):
    from rcql import call,query,source
    from rcql.operators import get_operator
    from rcql.executor import value_exactness
    values = case(name,rex)
    direct = get_operator(name).fn(rex,*values)
    result = Executor(sources={'r':rex}).execute(query(source('r'),call(name,*values)))
    assert result.exactness[0] == value_exactness(direct)
    if hasattr(direct,'coefficient_digest'):
        assert result.values[0].coefficient_digest == direct.coefficient_digest
    elif name == 'TRANSFORM_VERIFY':
        assert result.values[0] == direct
    elif name == 'TRANSFORM_READ':
        assert isinstance(result.values[0], ProgramTransformation)
    else:
        from rexgraph.io.catalog import object_digest
        assert object_digest(result.values[0]) == object_digest(direct)
