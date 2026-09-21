from dataclasses import replace
from contextlib import closing
from fractions import Fraction as Q
import pytest

from rcql import Executor,parse,Program,ProgramStep,ProgramInput,OutputRef,BoundSource,SourcePolicy,bind
from rcql.source_context import SnapshotContext,SourceSelection
from rcql.types import SourceRef
from rcdb import open_store
from rexgraph.tensor_field import TensorField,FieldSource,TensorChannels
from rexgraph.span import SpanBlock,SpanAttachment
from rexgraph.attachment_field import AttachmentField
from .helpers import native,family


def versioned(r,version=2):
    from rexgraph.io.catalog import object_digest
    ref=SourceRef('r',object_digest(r),record_id='record',record_version=version)
    return BoundSource(r,SourcePolicy.allow('*'),ref=ref)

@pytest.mark.parametrize('derived',(False,True))
def test_source_version_validation_direct_and_derived(derived):
    r,x,_=native();wrong=x.with_values([1,0,0]);wrong=replace(wrong,source=FieldSource(r,'record',1))
    p={'x':wrong,'fields':TensorChannels(('x',),(wrong,))}
    expression='CHANNEL_FIELD($fields,"x")' if derived else '$x'
    with pytest.raises(ValueError,match='version'):
        Executor(sources={'r':versioned(r)},params=p).execute(parse('FROM $r RETURN NATIVE_RESPONSE('+expression+')'))

@pytest.mark.parametrize('explain',(False,True))
def test_span_selected_source_validation(explain):
    r,_,_=native();a=SpanAttachment('A','event','time','document',time=SpanBlock('t','calendar','years',(('one',1,2),)))
    r.attach_metadata(1,0,'span',a)
    wrong=AttachmentField.from_source(r,record_id='record',version=1)
    q=parse(('EXPLAIN ' if explain else '')+'FROM $r RETURN SPAN_FIELD($a,"time")')
    with pytest.raises(ValueError,match='version'):Executor(sources={'r':versioned(r)},params={'a':wrong}).execute(q)

@pytest.mark.parametrize('tail',('',' LIMIT 0'))
def test_unused_return_is_not_evaluated_for_empty_match(tail):
    r,_,_=native();f,a,c=family(r)
    from rexgraph.coordinate_map import CoordinateMap
    readout=CoordinateMap(a.domain,a.codomain,((0,1,1),))
    q=parse('FROM $r MATCH e IN CELLS(1) WHERE e.index < 0 RETURN SECTION_VALUE(SECTION_OBSERVE($f,$o))'+tail)
    result=Executor(sources={'r':r},params={'f':f,'o':readout}).execute(q)
    assert result.values==((),)
    assert not any(e['operator']=='SECTION_VALUE' for e in result.execution)


def test_let_still_evaluates_before_empty_match():
    r,_,_=native();f,a,c=family(r)
    from rexgraph.coordinate_map import CoordinateMap
    readout=CoordinateMap(a.domain,a.codomain,((0,1,1),))
    q=parse('FROM $r LET v=SECTION_VALUE(SECTION_OBSERVE($f,$o)) MATCH e IN CELLS(1) WHERE e.index < 0 RETURN v')
    with pytest.raises(ValueError):Executor(sources={'r':r},params={'f':f,'o':readout}).execute(q)


def test_pure_invariant_evaluates_once_for_nonempty_match(monkeypatch):
    from rcql import operators
    r,_,_=native();original=operators._REGISTRY['COUNT'];calls=[]
    def counted(*args):calls.append(1);return original.fn(*args)
    monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=counted))
    result=Executor(sources={'r':r}).execute(parse('FROM $r MATCH e IN CELLS(1) RETURN COUNT(CELLS(0)),e.index'))
    assert result.values==(((4,0),(4,1),(4,2)),)
    assert len(calls)==1


def test_program_pins_source_before_each_stage(monkeypatch):
    from rcql import operators
    with closing(open_store('memory://')) as store:
        r,_,_=native();store.commit_mutation('data',r,expected_version=0,tx_time=10,analytics=False)
        p=Program('snapshot',(ProgramStep('one',parse('FROM RCDB_GET($db,"data") RETURN COUNT(CELLS(1))')),
             ProgramStep('two',parse('FROM RCDB_GET($db,"data") RETURN COUNT(CELLS(1))'))),
             outputs=(('first',OutputRef('one')),('second',OutputRef('two'))))
        original=operators._REGISTRY['COUNT'];calls=[]
        def append_after_first(*args):
            v=original.fn(*args);calls.append(1)
            if len(calls)==1:
                from rexgraph.graph import RexGraph
                newer=RexGraph.from_graph([0,1,2,3],[1,2,3,4]);store.commit_mutation('data',newer,expected_version=1,tx_time=20,analytics=False)
            return v
        monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=append_after_first))
        result=Executor(sources={'db':store}).execute_program(p)
        assert result.values==(3,3)
        assert {ref.version for ref in result.dependencies}=={1}
        assert store.read_record('data').record.version==2


def test_nested_program_cannot_bypass_evidence_cutoff(monkeypatch):
    from rcql import operators
    with closing(open_store('memory://')) as store:
        r,_,_=native();store.commit_mutation('past',r,expected_version=0,tx_time=10,analytics=False)
        store.commit_mutation('future',r,expected_version=0,tx_time=30,analytics=False)
        context=SnapshotContext.select(store,(SourceSelection('r','past'),),cutoff=20)
        program=Program('future',(ProgramStep('read',parse('FROM RCDB_VERSION($db,"future",1) RETURN COUNT(CELLS(1))')),))
        original=operators._REGISTRY['COUNT'];calls=[]
        def counted(*args):calls.append(1);return original.fn(*args)
        monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=counted))
        e=Executor(sources=context.sources,evidence=context,params={'p':program,'s':{'db':bind('db',store,SourcePolicy.allow('*'))},'v':{}})
        with pytest.raises((ValueError,TypeError)):e.execute(parse('FROM $r RETURN PROGRAM_RUN($p,$s,$v)'))
        assert not calls


def test_program_permissions_are_checked_on_restore():
    r,_,_=native()
    program=Program('read',(ProgramStep('count',parse('FROM $r RETURN COUNT(CELLS(1))')),))
    restored=Program.from_record(program.to_record())
    with pytest.raises(PermissionError):Executor(sources={'r':BoundSource(r,SourcePolicy.allow())}).execute_program(restored)


def test_complete_temporal_axes_inside_a_program():
    from rexgraph.process_field import sampled_field
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.coordinate_map import CoordinateMetric
    from rexgraph.tensor_moment import CoordinatePairing
    r,_,_=native();space=CoordinateSpace('geometric_reading',('volume',))
    field=TensorField(space,[0],source=FieldSource(r));values=sampled_field((field,field.with_values([2]),field),(0,Q(1,3),1),axis='elapsed',unit='s')
    pairing=CoordinatePairing.metric(CoordinateMetric.identity(space))
    p=Program('retained_time',(
        ProgramStep('moment',parse('FROM $r RETURN MOMENT_SUPPORT(FIELD_PAIR($field,$field,$metric))')),
        ProgramStep('diagonal',parse('FROM $r RETURN TENSOR_DIAGONAL($moment,"left/sample/elapsed/s","right/sample/elapsed/s","sample/elapsed/s")'),(('moment',OutputRef('moment')),)),
        ProgramStep('rate',parse('FROM $r RETURN SAMPLE_RATES($field)'))),
        (ProgramInput('field'),ProgramInput('metric')),
        (('support',OutputRef('diagonal')),('rate',OutputRef('rate'))))
    result=Executor(sources={'r':r},params={'field':values,'metric':pairing}).execute_program(p)
    assert result.values[0].values.tolist()==[[0,4,0]]
    assert result.values[1].values.tolist()==[[6,-3]]
    assert result.values[0].dependencies


def test_plan_type_render_keeps_nested_axes():
    from .helpers import tensor
    r,_,_=native();e=Executor(sources={'r':r},params={'x':tensor(r)})
    plan=e.topology(parse('FROM $r RETURN TENSOR_DIAGONAL($x,"left/sample","right/sample","sample")'))
    node=next(n for n in plan.declaration['nodes'] if n.get('operator')=='TENSOR_DIAGONAL')
    assert node['result']['tensor_axes']==[{'name':'sample','keys':['0','1']}]


def test_plan_record_declares_source_evidence():
    with closing(open_store('memory://')) as store:
        r,_,_=native();store.commit_mutation('data',r,expected_version=0,tx_time=10,analytics=False)
        t=Executor(sources={'db':store}).topology(parse('FROM RCDB_VERSION($db,"data",1) RETURN COUNT(CELLS(1))'))
        store.commit_mutation('plan',t.to_record(),expected_version=0,tx_time=20,analytics=False)
        ctx=SnapshotContext.select(store,(SourceSelection('p','plan'),),cutoff=25)
        assert {key[1] for key in ctx.entries}=={'plan','data'}


def test_program_declaration_tamper_is_not_executable():
    p=Program('one',(ProgramStep('read',parse('FROM $r RETURN 1')),))
    from rcql.program_codec import loads,dumps
    data=loads(p.to_bytes());data['version']=True
    with pytest.raises(ValueError):Program.from_bytes(dumps(data))
