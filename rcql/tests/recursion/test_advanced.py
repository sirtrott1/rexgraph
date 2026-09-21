from dataclasses import replace
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from rcql import *
from rcql.ast import Parameter,Literal,Comparison,Call,ListExpr
from rexgraph.tensor_field import FieldSource,TensorField
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.type_accession import CoordinateSpace
from .helpers import source,binding,counter,integer
from .advanced_helpers import field,field_program
from .query_cases import add


def double():
    x=Parameter('x')
    return NameRelation('double',('x',),Call('SUM',(ListExpr((x,x)),)))


def test_shared_modifier_does_not_expand_paths():
    op=add(0)
    for _ in range(20):op=op.then(double(),'x',modifier=True)
    assert len(op.stages)==20
    assert len(op.to_bytes())<30000
    assert op.apply(binding(source()),[3])==3*2**20
    topology=op.topology()
    assert len(topology.declaration['nodes'])<80
    from rexgraph.chain_map import _chain_residual
    assert _chain_residual(topology.boundary_tower())==0
    assert topology.declaration['origin']['stages']==op.stages


def test_repeated_argument_evaluated_once(monkeypatch):
    from rcql.operators import _REGISTRY
    r=source();base=add(3);op=base.then(double(),'x');record=[]
    entry=_REGISTRY['SUM']
    def observed(*args):record.append(args[1]);return entry.fn(*args)
    monkeypatch.setitem(_REGISTRY,'SUM',replace(entry,fn=observed))
    assert op.apply(binding(r),[2])==10 and len(record)==2


def test_separate_names_do_not_change_registry():
    from rcql.operators import _REGISTRY
    original=dict(_REGISTRY)
    r=source();a=add(2);b=double()
    e=Executor(sources={'r':r},params={'a':a,'b':b})
    out=e.execute(parse('FROM $r LET left=NAME_APPLY($a,[3]) LET right=NAME_APPLY($b,[3]) RETURN left,right'))
    assert out.values==(5,6) and original==_REGISTRY
    assert a.apply(binding(r),[3])==5


def test_iteration_topology_uses_prior_state_as_input():
    r=source();out=Executor(sources={'r':r},params={'op':add(1)}).execute(parse('FROM $r RETURN NAME_ITERATE($op,0,3,"x")')).values[0]
    nodes={v['id']:v for v in out.topology().declaration['nodes']}
    assert nodes['call/0']['inputs']==()
    assert nodes['call/2']['inputs']==('call/1',)
    assert out.topology().declaration['outputs']==('call/3',)


def test_tensor_recursion_and_moment_axes():
    from rexgraph.tensor_moment import CoordinatePairing
    from rexgraph.coordinate_map import CoordinateMetric
    r=source();x=field(r);p=field_program()
    result=p.execute(binding(r),'resolve',[3,x])
    assert result.value.axes==x.axes and result.value.values.shape==(3,2)
    channels=result.fields(); assert len(channels.fields)==4
    pairing=CoordinatePairing.metric(CoordinateMetric.identity(x.space))
    e=Executor(sources={'r':r},params={'p':p,'x':x,'m':pairing})
    out=e.execute(parse('FROM $r LET result=RECURSIVE_RUN($p,"resolve",[3,$x]) LET channels=RECURSIVE_FIELDS(result) LET final=RECURSIVE_VALUE(result) RETURN channels,MOMENT_CONTRACT(FIELD_PAIR(final,final,$m)),RECURSIVE_TRACE(result)'))
    assert out.values[1].values.shape==(1,2,2)
    assert len(out.values[1].axes)==2
    assert out.values[0].field('resolve/0').coefficient_digest==result.value.coefficient_digest


def test_iteration_and_recursive_evaluation_agree_as_requested():
    r=source();x=field(r);p=field_program();name=NameRelation.operator('NATIVE_RESPONSE')
    a=Executor(sources={'r':r},params={'n':name,'x':x}).execute(parse('FROM $r RETURN NAME_ITERATE($n,$x,3,"field")')).values[0]
    b=p.execute(binding(r),'resolve',[3,x])
    assert a.value.coefficient_digest==b.value.coefficient_digest
    assert a.fields().names[0].endswith('/0')


def test_recursive_group_source_policy():
    r=source();p=counter()
    with pytest.raises(PermissionError):
        Executor(sources={'r':BoundSource(r,SourcePolicy.allow())},params={'p':p}).execute(parse('FROM $r RETURN RECURSIVE_RUN($p,"count",[1])'))


def test_recursive_native_state_version():
    from rcdb import open_store
    with closing(open_store('memory://')) as store:
        store.commit_mutation('r',source(),expected_version=0,tx_time=10)
        one=store.read_record('r',version=1)
        store.commit_mutation('r',one.value,expected_version=1,tx_time=30)
        x=field(one.value,'r',1)
        p=field_program()
        with pytest.raises(ValueError):Executor(sources={'db':store},params={'p':p,'x':x}).execute(parse('FROM RCDB_VERSION($db,"r",2) RETURN RECURSIVE_RUN($p,"resolve",[1,$x])'))
        from rcql.types import SourceRef
        selected=BoundSource(one.value,SourcePolicy.allow('*'),ref=SourceRef('selected',state_digest=one.state_digest,record_id='r',record_version=1))
        answer=Executor(sources={'r':selected},params={'p':p,'x':x}).execute(parse('FROM $r RETURN RECURSIVE_VALUE(RECURSIVE_RUN($p,"resolve",[1,$x]))')).values[0]
        assert answer.source.version==1


def test_recursive_evidence_context_propagates():
    from rcdb import open_store
    from rcql.source_context import SnapshotContext,SourceSelection
    with closing(open_store('memory://')) as store:
        store.commit_mutation('r',source(),expected_version=0,tx_time=10)
        early=SnapshotContext.select(store,(SourceSelection('r','r',version=1),),cutoff=20)
        x=field(early.sources['r'].value,'r',1)
        result=Executor(sources=early.sources,params={'p':field_program(),'x':x},evidence=early).execute(parse('FROM $r RETURN RECURSIVE_RUN($p,"resolve",[2,$x])')).values[0]
        assert result.dependencies[0].version==1
        later=source();later.attach_metadata(1,0,'later','observed')
        store.commit_mutation('later',later,expected_version=0,tx_time=30)
        snapshot=store.read_record('later');bad=replace(x,dependencies=(FieldSource(snapshot.value,'later',1),))
        with pytest.raises(ValueError):Executor(sources=early.sources,params={'p':field_program(),'x':bad},evidence=early).execute(parse('FROM $r RETURN RECURSIVE_RUN($p,"resolve",[2,$x])'))


def test_finite_program_calls_recursive_definition():
    r=source();p=counter()
    step=ProgramStep('calculate',parse('FROM $r RETURN RECURSIVE_VALUE(RECURSIVE_RUN($p,"count",[$n]))'))
    program=Program('outer',(step,),(ProgramInput('p',ValueKind.RECURSIVE_PROGRAM.value),integer('n')), (('answer',OutputRef('calculate',0)),))
    e=Executor(sources={'r':r},params={'p':p,'n':4})
    assert e.execute_program(program).values==(4,)


def test_agent_recursive_convenience_uses_query_path():
    from agent.rcql_runtime import RCQLRuntime
    runtime=RCQLRuntime();runtime.register('r',source())
    out=runtime.execute_recursive(counter(),'count',[7],source='r')
    assert out.values[0].value==7
    assert runtime.execute_recursive(counter(),'count',[7],source='r',explain=True).values[0]['result_type']


def test_concurrent_recursion_scopes_are_independent():
    p=counter()
    def run(n):return p.execute(binding(source()),'count',[n]).value
    with ThreadPoolExecutor(max_workers=4) as pool: assert list(pool.map(run,range(8)))==list(range(8))


def test_runtime_source_change_is_detected(monkeypatch):
    from rcql.operators import _REGISTRY
    r=source();old=_REGISTRY['SUM'];used=[]
    def changed(*args):
        out=old.fn(*args)
        if not used:r.attach_metadata(1,0,'modified','yes');used.append(True)
        return out
    monkeypatch.setitem(_REGISTRY,'SUM',replace(old,fn=changed))
    with pytest.raises(ValueError,match='changed'):counter().execute(binding(r),'count',[2])


def test_feedback_free_interiors_have_fixed_difference():
    from rexgraph.affine_feedback import AffineFeedback,FeedbackEquation
    r=source();ref=FieldSource(r);s=CoordinateSpace('scalar',('value',));a=CoordinateMap.identity(s)
    z=TensorField(s,[0],source=ref)
    system=AffineFeedback('mutual',(('x',s),('y',s)),(FeedbackEquation('x_eq','x',(('y',a),),z),FeedbackEquation('y_eq','y',(('x',a),),z)),ref)
    family=system.complete(); assert family.dimension==1
    out=CoordinateSpace('difference',('x_minus_y',))
    read=CoordinateMap(system.space,out,((0,0,1),(0,1,-1)))
    assert family.observe(read).value().values.item()==0
    assert family.directions.shape==(2,1)


def test_feedback_tensor_contradiction_is_checked_per_column():
    from rexgraph.affine_feedback import AffineFeedback,FeedbackEquation
    from rexgraph.section_calculus import InconsistentSectionError
    r=source();ref=FieldSource(r);s=CoordinateSpace('scalar',('x',));axis=CoordinateSpace('cases',('valid','invalid'))
    z=TensorField(s,[[0,1]],(axis,),ref);a=CoordinateMap.identity(s)
    f=AffineFeedback('mixed',(('x',s),),(FeedbackEquation('eq','x',(('x',a),),z),),ref)
    with pytest.raises(InconsistentSectionError):f.complete()


def test_non_native_callables_are_not_certified_names():
    with pytest.raises((ValueError,TypeError)):NameRelation('bad',(),Literal(lambda:1))
    with pytest.raises((TypeError,ValueError)):NameRelation.operator('SUM').bind('values',[object()])


@pytest.mark.parametrize('fieldname',['calls','evaluations','definition_digest','entry','history','dependencies'])
def test_completed_record_tamper_detected(fieldname):
    from rcql.program_codec import dumps,loads
    r=source();result=counter().execute(binding(r),'count',[2]);record=result.to_record()
    data=loads(record.get_metadata(1,0,'rcql_recursive_result'))
    replacement={'calls':999,'evaluations':999,'definition_digest':'0'*64,'entry':'other','history':(),'dependencies':()}
    data[fieldname]=replacement[fieldname]
    record.attach_metadata(1,0,'rcql_recursive_result',dumps(data).decode())
    with pytest.raises(ValueError):RecursionResult.from_record(record,result.dependencies)


def test_declared_name_detects_mutated_capture():
    values=[1,2];name=NameRelation.operator('SUM').bind('values',values)
    values.append(3)
    with pytest.raises(ValueError,match='changed'):name.apply(binding(source()),[])


def test_named_native_boundary_accepts_existing_chain():
    from rexgraph.cochain import Chain
    r=source();x=Chain(1,np.array([1,0,0],dtype=object),source=r)
    operation=NameRelation.operator('BOUNDARY').bind('grade',1)
    value=operation.apply(binding(r),{'values':x})
    assert value.values.tolist()==[-1,1,0,0]


def test_existing_native_operator_can_be_argument():
    from rexgraph.linear_operator import boundary_operator
    r=source();operator=boundary_operator(r,1)
    from rcql.relation_runtime import fingerprint
    assert fingerprint(operator)[0]=='native_operator'


def test_no_lower_grade_collapse_in_named_native_response():
    r=source();x=field(r)
    out=NameRelation.operator('NATIVE_RESPONSE').apply(binding(r),{'field':x})
    assert out.grade==1 and out.variance=='chain' and out.axes==x.axes


def test_tail_recursion_retains_input_fields_not_only_final_output():
    r=source();x=field(r);p=field_program();out=p.execute(binding(r),'resolve',[3,x])
    states=out.fields('resolve','invocation','x')
    assert len(states.fields)==4
    assert states.fields[0].coefficient_digest==x.coefficient_digest
    assert len({v.coefficient_digest for v in states.fields})==4
    assert len({v.coefficient_digest for v in out.fields().fields})==1
    q=parse('FROM $r RETURN RECURSIVE_FIELDS($result,"resolve","invocation","x")')
    selected=Executor(sources={'r':r},params={'result':out}).execute(q).values[0]
    assert [v.coefficient_digest for v in selected.fields]==[v.coefficient_digest for v in states.fields]
    restored=RecursionResult.from_record(out.to_record(),out.dependencies)
    assert restored.coefficient_digest==out.coefficient_digest
    assert restored.fields('resolve','invocation','x').fields[1].axes==x.axes


def test_recursion_accumulates_modifiers_without_replacing_base_name():
    from rexgraph.chain_map import _chain_residual
    x=Parameter('x')
    base=NameRelation('base',('x',),x)
    modifier=NameRelation('plus_one',('x',),Call('SUM',(ListExpr((x,Literal(1))),)))
    n,operation,mod=Parameter('remaining'),Parameter('operation'),Parameter('modifier')
    ports=(ProgramInput('remaining',ValueKind.EXACT_INTEGER.value),
           ProgramInput('operation',ValueKind.NAME_RELATION.value),
           ProgramInput('modifier',ValueKind.NAME_RELATION.value))
    definition=RecursiveDefinition('grow',ports,Comparison('==',n,Literal(0)),operation,
        recur('grow',Call('SUM',(ListExpr((n,Literal(-1))),)),
              Call('NAME_MODIFY',(operation,mod,Literal('x'))),mod),
        ProgramInput('result',ValueKind.NAME_RELATION.value),decreases='remaining')
    program=RecursiveProgram('named_recursion',(definition,))
    r=source();executor=Executor(sources={'r':r},params={'program':program,'base':base,'modifier':modifier})
    text='FROM $r LET result=RECURSIVE_RUN($program,"grow",[5,$base,$modifier]) LET operation=RECURSIVE_VALUE(result) RETURN NAME_APPLY(operation,[3]),NAME_APPLY($base,[3]),result'
    executor.execute(parse('EXPLAIN '+text))
    changed,original,result=executor.execute(parse(text)).values
    assert (changed,original)==(8,3)
    assert result.value.name==base.name and len(result.value.stages)==5
    assert _chain_residual(result.value.topology().boundary_tower())==0
    restored=RecursionResult.from_record(result.to_record(),result.dependencies)
    assert restored.coefficient_digest==result.coefficient_digest
    assert restored.value.apply(binding(r),[3])==8
    assert base.apply(binding(r),[3])==3
