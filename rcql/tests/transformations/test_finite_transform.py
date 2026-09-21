from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rcql import (Program, ProgramInput, ProgramStep, OutputRef, ProgramTransformation,
    Executor, parse, bind, BoundSource, SourcePolicy, QueryCache)
from rcql.program_codec import dumps, loads
from rexgraph.graph import RexGraph


def fixture():
    r = RexGraph.from_cells([2, [[0,1]]])
    policy = SourcePolicy.allow('read')
    return r, {'r':bind('r',r,policy)}, {'r':BoundSource(r,policy)}


def program(kind='ExactInteger'):
    return Program('linked', (
        ProgramStep('first',parse('FROM $r RETURN SUM([$x,1]), $x')),
        ProgramStep('second',parse('FROM $r RETURN SUM([$v,$v])'),(('v',OutputRef('first',0)),))),
        (ProgramInput('x',kind),), (('answer',OutputRef('second')),('original',OutputRef('first',1))))


def setup(rule):
    return {'input':(['x','renamed'],{'renamed':Q(2**140+1,7)}),
            'step':(['first','start'],{'x':Q(2**140+1,7)}),
            'specialize':({'x':Q(2**140+1,7)}, {})}[rule]


@pytest.mark.parametrize('rule',['input','step','specialize'])
def test_linked_program_exactness(rule):
    r,sources,bound=fixture(); p=program('ExactRational'); args,params=setup(rule)
    t=ProgramTransformation.program(p,rule,args)
    candidate=t.compile_program(p,sources,params)
    result=candidate.execute(Executor(sources=bound,params=params))
    assert result.values == (2*(Q(2**140+1,7)+1),Q(2**140+1,7))
    assert result.aliases==('answer','original')
    assert t.original().to_bytes()==p.to_bytes()
    proof=t.verify(p)
    assert proof['source_selectors']
    assert proof['topology_status'].startswith('bound static plan required')
    assert 'components' not in proof


@pytest.mark.parametrize('rule,args', [('input',['x','r']),('input',['x','v']),('input',['x','x']),
    ('input',['missing','z']),('input',['x','bad name']),('step',['first','second']),
    ('step',['missing','a']),('specialize',{}),('specialize',{'missing':2}),
    ('specialize',{'x':object()}),('missing',[])])
def test_invalid_transform(rule,args):
    with pytest.raises((ValueError,TypeError)):
        ProgramTransformation.program(program(),rule,args)


@pytest.mark.parametrize('rule,args', [('input',['cutoff','clock']),('specialize',{'cutoff':1})])
def test_source_selector_cannot_change(rule,args):
    p=Program('temporal',(ProgramStep('a',parse('FROM AT($r,$cutoff) RETURN 1')),),
              (ProgramInput('cutoff','ExactInteger'),))
    with pytest.raises(ValueError,match='source selection'):
        ProgramTransformation.program(p,rule,args)


def test_specialization_retains_type_contract_and_forbids_override():
    r,sources,bound=fixture(); p=program()
    t=ProgramTransformation.program(p,'specialize',{'x':Q(1,2)})
    with pytest.raises(TypeError,match='declared type'):
        t.compile_program(p,sources,{})
    candidate=Program.from_bytes(loads(t.to_bytes())['target'])
    assert candidate.captures[0][0] == ProgramInput('x','ExactInteger')
    with pytest.raises(TypeError,match='declared type'):
        candidate.execute(Executor(sources=bound,params={}))
    with pytest.raises(ValueError,match='exactly'):
        candidate.execute(Executor(sources=bound,params={'x':2}))


def test_unused_capture_still_has_type_contract():
    r,sources,bound=fixture()
    p=Program('unused',(ProgramStep('a',parse('FROM $r RETURN 1')),),(ProgramInput('x','ExactInteger'),))
    candidate=p.specialize({'x':'wrong'})
    with pytest.raises(TypeError,match='declared type'):
        candidate.execute(Executor(sources=bound,params={}))


@pytest.mark.parametrize('rule',['input','step','specialize'])
def test_independent_verification(rule,monkeypatch):
    p=program('ExactRational'); args,_=setup(rule)
    t=ProgramTransformation.program(p,rule,args)
    def fail(*a,**kw):
        raise AssertionError('verification invoked construction')
    monkeypatch.setattr(Program,'rename_input',fail)
    monkeypatch.setattr(Program,'rename_step',fail)
    monkeypatch.setattr(Program,'specialize',fail)
    assert ProgramTransformation.from_bytes(t.to_bytes()).verify(p)==t.verify(p)


@pytest.mark.parametrize('change',['link','export','source','contract','capture','certificate','version','extra'])
def test_tamper_refused(change):
    p=program(); t=ProgramTransformation.program(p,'specialize',{'x':2})
    data=loads(t.to_bytes()); candidate=Program.from_bytes(data['target'])
    if change=='link':
        candidate=replace(candidate,steps=(candidate.steps[0],replace(candidate.steps[1],inputs=(('v',OutputRef('first',1)),))))
    elif change=='export':
        candidate=replace(candidate,outputs=(('answer',OutputRef('first',1)),('original',OutputRef('first',1))))
    elif change=='source':
        candidate=replace(candidate,steps=(replace(candidate.steps[0],query=parse('FROM $other RETURN SUM([$x,1]),$x')),candidate.steps[1]))
    elif change=='contract':
        candidate=replace(candidate,captures=((ProgramInput('x'),2),))
    elif change=='capture':
        candidate=replace(candidate,captures=((ProgramInput('x','ExactInteger'),3),))
    elif change=='certificate':
        data['certificate']['scope']='arbitrary program equivalence'
    elif change=='version':
        data['version']=True
    else:
        data['authority']='*'
    data['target']=candidate.to_bytes()
    with pytest.raises((ValueError,TypeError)):
        ProgramTransformation.from_bytes(dumps(data))


def test_sealed_capture_is_copied_and_mutation_refused():
    r,sources,bound=fixture()
    p=Program('p',(ProgramStep('a',parse('FROM $r RETURN SUM($x)')),),(ProgramInput('x'),))
    values=[1,2]; candidate=p.specialize({'x':values}); values.append(3)
    assert candidate.execute(Executor(sources=bound,params={})).values==(3,)
    candidate.captures[0][1].append(4)
    with pytest.raises(ValueError,match='outside an explicit'):
        candidate.execute(Executor(sources=bound,params={}))


def test_program_schema_one_unchanged_and_two_validated():
    p=program(); raw=p.to_bytes()
    assert loads(raw)['version']==1 and 'captures' not in loads(raw)
    assert Program.from_bytes(raw).to_bytes()==raw
    captured=p.specialize({'x':2}); raw=captured.to_bytes()
    assert loads(raw)['version']==2 and Program.from_bytes(raw).to_bytes()==raw
    data=loads(raw);data['captures']=()
    with pytest.raises(ValueError):
        Program.from_bytes(dumps(data))


def test_rcql_stages_do_not_execute_program(monkeypatch):
    from rcql.operators import _REGISTRY
    r,sources,bound=fixture();p=program();calls=[];op=_REGISTRY['SUM']
    def counted(*a):
        calls.append(1);return op.fn(*a)
    monkeypatch.setitem(_REGISTRY,'SUM',replace(op,fn=counted))
    e=Executor(sources=bound,params={'p':p,'sources':sources,'params':{'z':2}})
    text='''FROM $r LET t=TRANSFORM_PROGRAM($p,"input",["x","z"])
        RETURN TRANSFORM_PROGRAM_COMPILE(t,TRANSFORM_SOURCE(t),$sources,$params)'''
    explained=e.execute(parse('EXPLAIN '+text))
    assert explained.execution==() and not calls
    candidate=e.execute(parse(text)).values[0]
    assert not calls
    result=candidate.execute(Executor(sources=bound,params={'z':2}))
    assert result.values==(6,2) and len(calls)==2


@pytest.mark.parametrize('rule',['input','step','specialize'])
@pytest.mark.parametrize('format',['rcbd','safetensors','rcdb'])
def test_portable(rule,format,tmp_path):
    r,sources,bound=fixture();p=program('ExactRational');args,params=setup(rule)
    t=ProgramTransformation.program(p,rule,args);record=t.to_record()
    if format=='rcbd':
        from rexgraph.io.bundle import save_rcbd,load_rcbd
        path=tmp_path/'t.rcbd';save_rcbd(path,record);record=load_rcbd(path)
    elif format=='safetensors':
        from rexgraph.io.safetensors_bridge import rex_to_safetensors,safetensors_to_rex
        path=tmp_path/'t.safetensors';rex_to_safetensors(record,path);record=safetensors_to_rex(path)
    else:
        from rcdb import open_store
        uri='file://'+str(tmp_path/'db');db=open_store(uri)
        db.commit_mutation('t',record,expected_version=0,analytics=False);db.close()
        db=open_store(uri);record=db.read_record('t',version=1).value
        assert db.verify_commits('t');db.close()
    restored=ProgramTransformation.from_record(record)
    assert restored.to_bytes()==t.to_bytes()
    candidate=restored.compile_program(p,sources,params)
    assert Program.from_record(candidate.to_record()).execute(Executor(sources=bound,params=params)).values[1]==Q(2**140+1,7)


def test_bound_topology_and_no_unbound_claim():
    from rcql.plan_topology import PlanTopology
    r,sources,bound=fixture();p=program();t=ProgramTransformation.program(p,'step',['first','start'])
    with pytest.raises(ValueError,match='TRANSFORM_PROGRAM_TOPOLOGY'):
        t.topology()
    e=Executor(sources=bound,params={'t':t,'p':p,'sources':sources,'params':{'x':2}})
    record=e.execute(parse('FROM $r RETURN TRANSFORM_PROGRAM_TOPOLOGY($t,$p,$sources,$params)')).values[0]
    topology=PlanTopology.from_record(record)
    assert [s['name'] for s in topology.declaration['stages']]==['start','second']
    assert topology.port_realization().entries


def test_dynamic_match_preserved_but_not_claimed_static_topology():
    r,sources,bound=fixture()
    p=Program('matched',(ProgramStep('a',parse('FROM $r MATCH e IN CELLS(1) RETURN $x')),),(ProgramInput('x','ExactInteger'),))
    t=ProgramTransformation.program(p,'input',['x','z'])
    c=t.compile_program(p,sources,{'z':2})
    assert c.steps[0].query.matches==p.steps[0].query.matches
    with pytest.raises(ValueError,match='MATCH'):
        c.topology(Executor(sources=bound,params={'z':2}))


def test_persisted_meta_program_returns_executable_program():
    r,sources,bound=fixture();p=program()
    meta=Program('meta',(ProgramStep('candidate',parse('''FROM $r
        LET t=TRANSFORM_PROGRAM($p,"input",["x","z"])
        RETURN TRANSFORM_PROGRAM_COMPILE(t,$p,$sources,$params)''')),),
        (ProgramInput('p','Program'),ProgramInput('sources'),ProgramInput('params')),
        (('candidate',OutputRef('candidate')),))
    e=Executor(sources=bound,params={'p':p,'sources':sources,'params':{'z':2}})
    result=e.execute_program(Program.from_record(meta.to_record()))
    assert result.values[0].execute(Executor(sources=bound,params={'z':2})).values==(6,2)
    assert e.execute_program(meta,explain=True)['output_types'][0].kind.value=='Program'


def test_program_values_cache_and_recursive_restore():
    from rcdb import open_store
    from rcql import RecursiveDefinition,RecursiveProgram,RecursionResult,ValueKind,recur
    from rcql.ast import Parameter,Comparison,Literal,Call,ListExpr
    r,sources,bound=fixture();p=program()
    db=open_store('memory://')
    try:
        e=Executor(sources=bound,params={'p':p})
        cache=QueryCache(db);q=parse('FROM $r RETURN $p')
        e.execute_cached(q,cache)
        cached=e.execute_cached(q,cache)
        assert cached.native_plan['cache']['hit'] and cached.values[0].to_bytes()==p.to_bytes()
    finally:
        db.close()
    count,value=Parameter('count'),Parameter('value')
    recursion=RecursiveProgram('carry',(RecursiveDefinition('carry',
        (ProgramInput('count',ValueKind.EXACT_INTEGER.value),ProgramInput('value','Program')),
        Comparison('==',count,Literal(0)),value,
        recur('carry',Call('SUM',(ListExpr((count,Literal(-1))),)),value),
        ProgramInput('result','Program'),decreases='count'),))
    result=recursion.execute(sources['r'],'carry',[3,p])
    restored=RecursionResult.from_record(result.to_record(),result.dependencies)
    assert restored.value.to_bytes()==p.to_bytes()


def test_policy_intersection_not_bypassed():
    r,sources,bound=fixture();p=program();t=ProgramTransformation.program(p,'input',['x','z'])
    e=Executor(sources={'r':r},params={'t':t,'p':p,'sources':sources,'params':{'z':2}})
    with pytest.raises(PermissionError,match='intersection'):
        e.execute(parse('FROM $r RETURN TRANSFORM_PROGRAM_COMPILE($t,$p,$sources,$params)'))


def test_multiple_sources_keep_their_identities():
    r,sources,bound=fixture()
    s=RexGraph.from_cells([3,[[0,1],[1,2]]])
    policy=SourcePolicy.allow('read')
    sources['s']=bind('s',s,policy);bound['s']=BoundSource(s,policy)
    p=Program('sources',(
        ProgramStep('a',parse('FROM $r RETURN $x')),
        ProgramStep('b',parse('FROM $s RETURN SUM([$v,$x])'),(('v',OutputRef('a')),))),
        (ProgramInput('x','ExactInteger'),))
    before=p.execute(Executor(sources=bound,params={'x':3}))
    t=ProgramTransformation.program(p,'input',['x','y'])
    candidate=t.compile_program(p,sources,{'y':3})
    after=candidate.execute(Executor(sources=bound,params={'y':3}))
    assert after.values==before.values==(6,)
    assert after.sources==before.sources and len(after.sources)==2
    assert p.inputs[0].name=='x'


def test_successive_typed_captures_after_renaming():
    r,sources,bound=fixture()
    p=Program('p',(ProgramStep('a',parse('FROM $r RETURN SUM([$x,$y])')),),
        (ProgramInput('x','ExactInteger'),ProgramInput('y','ExactRational')))
    first=ProgramTransformation.program(p,'specialize',{'x':2})
    a=first.compile_program(p,sources,{'y':Q(1,3)})
    second=ProgramTransformation.program(a,'input',['y','z'])
    b=second.compile_program(a,sources,{'z':Q(1,3)})
    third=ProgramTransformation.program(b,'specialize',{'z':Q(1,3)})
    c=ProgramTransformation.from_record(third.to_record()).compile_program(b,sources,{})
    assert c.execute(Executor(sources=bound,params={})).values==(Q(7,3),)
    assert [(spec.name,spec.kind) for spec,_ in c.captures]==[('x','ExactInteger'),('z','ExactRational')]


def test_program_explicitly_transforms_its_own_declaration():
    r,sources,bound=fixture()
    p=Program('self',(ProgramStep('make',parse(
        'FROM $r RETURN TRANSFORM_PROGRAM($self,"step",["make","next"])')),),
        (ProgramInput('self','Program'),))
    original=p.to_bytes()
    t=p.execute(Executor(sources=bound,params={'self':p})).values[0]
    assert p.to_bytes()==original and t.original().to_bytes()==original
    candidate=t.compile_program(p,sources,{'self':p})
    assert candidate.steps[0].name=='next'
    assert candidate.execute(Executor(sources=bound,params={'self':p})).values[0].to_bytes()==t.to_bytes()


def test_historical_evidence_remains_bound_after_transformation():
    from rcdb import open_store
    from rcql import SnapshotContext,SourceSelection
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.tensor_field import TensorField,FieldSource
    r,_,_=fixture();db=open_store('memory://')
    try:
        db.commit_mutation('data',r,expected_version=0,tx_time=10,analytics=False)
        context=SnapshotContext.select(db,(SourceSelection('r','data'),),cutoff=20)
        selected=context.sources['r']
        space=NativeFieldCalculus.from_rex(selected.value).complex.spaces[1]
        field=TensorField(space,[1],source=FieldSource(selected.value,'data',1),grade=1,variance='chain')
        p=Program('field',(ProgramStep('response',parse('FROM $r RETURN NATIVE_RESPONSE($field)')),),
                  (ProgramInput('field'),))
        t=ProgramTransformation.program(p,'input',['field','input'])
        binding=Executor(sources=context.sources)._planning_binding(parse('FROM $r RETURN 1').source,selected)
        parameters={'t':t,'p':p,'sources':{'r':binding},'params':{'input':field}}
        e=Executor(sources=context.sources,params=parameters,evidence=context)
        query=parse('FROM $r RETURN TRANSFORM_PROGRAM_COMPILE($t,$p,$sources,$params)')
        candidate=e.execute(query).values[0]
        assert candidate.inputs[0].name=='input'
        db.commit_mutation('data',r,expected_version=1,tx_time=30,analytics=False)
        future=db.read_record('data',version=2)
        e.params['params']={'input':replace(field,source=FieldSource(future.value,'data',2))}
        with pytest.raises(ValueError):
            e.execute(query)
    finally:
        db.close()


def case(name,rex):
    p=program();t=ProgramTransformation.program(p,'input',['x','z'])
    sources={'r':bind('r',rex,SourcePolicy.allow('*'))}
    return {'TRANSFORM_PROGRAM':(p,'input',['x','z']),
        'TRANSFORM_PROGRAM_COMPILE':(t,p,sources,{'z':2}),
        'TRANSFORM_PROGRAM_TOPOLOGY':(t,p,sources,{'z':2})}[name]


@pytest.mark.parametrize('name',['TRANSFORM_PROGRAM','TRANSFORM_PROGRAM_COMPILE','TRANSFORM_PROGRAM_TOPOLOGY'])
def test_operator_contract(name):
    from rcql import call,query,source
    from rcql.operators import get_operator
    from rexgraph.io.catalog import object_digest
    r,_,_=fixture();args=case(name,r)
    direct=get_operator(name).fn(r,*args)
    result=Executor(sources={'r':r}).execute(query(source('r'),call(name,*args))).values[0]
    if hasattr(direct,'coefficient_digest'):
        assert result.coefficient_digest==direct.coefficient_digest
    else:
        assert object_digest(result)==object_digest(direct)
