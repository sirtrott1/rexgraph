from fractions import Fraction as Q
import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.tensor_field import TensorField,FieldSource
from rexgraph.tensor_moment import TensorMomentKernel,MomentSpan,CoordinatePairing
from rexgraph.temporal_field import TensorEvolution
from rexgraph.span import SpanBlock,SpanAttachment
from rexgraph.attachment_field import AttachmentField,common_attachment_observations
from rexgraph.type_accession import CoordinateSpace
from rcql import parse
from rcql.builder import query,call,param
from rcql.executor import Executor


def setup():
    rex=RexGraph.from_cells([4,[[0,1],[1,2],[2,3]]])
    calculus=NativeFieldCalculus.from_rex(rex)
    s=calculus.complex.spaces[1]
    x=TensorField(s,[1,0,0],source=FieldSource(rex),grade=1,variance='chain')
    return rex,calculus,x


def test_native_response_and_builder_preserve_exact_field():
    rex,calculus,x=setup()
    executor=Executor(sources={'g':rex},params={'x':x,'calculus':calculus})
    text=executor.execute(parse('FROM $g RETURN NATIVE_RESPONSE($x,1,$calculus)'))
    built=executor.execute(query(param('g'),call('NATIVE_RESPONSE',param('x'),1,param('calculus'))))
    assert text.values[0].values.tolist()==[Q(8,21),Q(1,7),Q(1,21)]
    assert text.values[0].coefficient_digest==built.values[0].coefficient_digest
    assert text.exactness[0].value=='rational'
    assert text.execution[0]['methods'][0]['method']=='rational-retained-tensor-action'


def test_moment_query_retains_field_axes_and_support():
    rex,calc,_=setup();s=calc.complex.spaces[1]
    f=CoordinateSpace('probes',('a','b'))
    x=TensorField(s,[[1,3],[2,4],[0,0]],(f,),FieldSource(rex),1,'chain')
    kernel=TensorMomentKernel(('field',),(CoordinateMap.identity(s),),common_metric=calc.metrics[1])
    scope={'kernel':kernel,'x':x}
    expression='MOMENT_PAIR(TENSOR_MOMENTS($kernel,$x),"field","field")'
    result=Executor(sources={'g':rex},params=scope).execute(parse('FROM $g RETURN MOMENT_SUPPORT('+expression+'), MOMENT_CONTRACT('+expression+')'))
    assert result.values[0].values.shape==(3,2,2)
    assert result.values[1].values.tolist()==[[[5,11],[11,25]]]
    with pytest.raises(Exception):
        Executor(sources={'g':rex},params=scope).execute(parse('FROM $g RETURN TENSOR_SCALAR(MOMENT_CONTRACT('+expression+'))'))


def test_explain_does_not_apply_a_native_action(monkeypatch):
    from rexgraph.native_field import FieldAction
    rex,calc,x=setup()
    def forbidden(*a,**k):raise AssertionError('EXPLAIN applied a native action')
    monkeypatch.setattr(FieldAction,'apply',forbidden)
    result=Executor(sources={'g':rex},params={'calc':calc,'x':x}).execute(parse('EXPLAIN FROM $g RETURN NATIVE_RESPONSE($x,1,$calc)'))
    assert 'TensorField' in str(result.values[0])


def test_named_axis_and_foreign_source_rejected():
    rex,calc,x=setup()
    wrong=TensorField(CoordinateSpace('other',('0','1','2')),[1,0,0],source=FieldSource(rex),grade=1,variance='chain')
    other,_,foreign=setup()
    for field in (wrong,foreign):
        with pytest.raises(Exception):
            Executor(sources={'g':rex},params={'x':field,'a':calc.hodge(1)}).execute(parse('FROM $g RETURN TENSOR_APPLY($a,$x)'))


def fixture(owner='A',start=2024):
    g=RexGraph.from_cells([2,[[0,1]]])
    t=SpanBlock('time','event_time','abstract_year',(('first',2017,2019),('second',start,2026)))
    g.attach_metadata(1,0,'time',SpanAttachment('ann1',owner,'time','doc',time=t))
    g.attach_metadata(1,0,'modifier',SpanAttachment('ann2',owner,'modifier','doc',time=t))
    return g


def test_queries_derive_attachments_and_observations_from_selected_state():
    g=fixture()
    result=Executor(sources={'g':g}).execute(parse('FROM $g RETURN ATTACHMENTS(), SPAN_FIELD(ATTACHMENTS(),"time",false)'))
    assert len(result.values[0].attachments)==2
    assert result.values[1].values.tolist()==[2,0,2]
    assert result.values[1].source.state_digest==result.values[0].source.state_digest


def test_span_delta_uses_both_selected_endpoint_sources():
    old,new=fixture(),fixture(owner='B')
    new_field=AttachmentField.from_source(new,record_id='doc',version=2)
    scope={'new':new_field}
    result=Executor(sources={'g':old},params=scope).execute(parse('FROM $g RETURN CHANNEL_TOTAL(SPAN_DELTA(ATTACHMENTS(),$new))'))
    value=result.values[0]
    assert sum(v*v for v in value.values)==8
    assert len(value.dependencies)==2
    assert value.source.version==2
    assert value.source.source is new


def test_full_rcdb_roundtrip_versions_and_retained_moment():
    from rcdb import open_store
    store=open_store('memory://').configure_security(require_commits=True)
    try:
        one=store.commit_mutation('doc',fixture(),expected_version=0)
        two=store.commit_mutation('doc',fixture(start=2025),expected_version=one.version)
        new=Executor(sources={'db':store}).execute(parse(f'FROM RCDB_VERSION($db,"doc",{two.version}) RETURN ATTACHMENTS()')).values[0]
        delta=Executor(sources={'db':store},params={'new':new}).execute(parse(f'FROM RCDB_VERSION($db,"doc",{one.version}) RETURN CHANNEL_TOTAL(SPAN_DELTA(ATTACHMENTS(),$new))')).values[0]
        old=Executor(sources={'db':store}).execute(parse(f'FROM RCDB_VERSION($db,"doc",{one.version}) RETURN ATTACHMENTS()')).values[0]
        _,observation=common_attachment_observations(old,new)
        moment=MomentSpan(delta,delta,CoordinatePairing.metric(observation.metric))
        assert moment.scalar()==2
        carrier=RexGraph.from_cells([1,[[0]]]);carrier.attach_metadata(1,0,'moment',moment)
        committed=store.commit_mutation('result',carrier,expected_version=0)
        selected=store.read_record('result',version=committed.version)
        restored=selected.value.get_metadata(1,0,'moment')
        assert restored.scalar()==2
        assert len(restored.support().dependencies)==2
        assert {d.version for d in restored.support().dependencies}=={one.version,two.version}
        assert store.verify_commits('doc') and store.verify_commits('result')
    finally:store.close()


def test_independent_endpoint_reference_refuses_wrong_new_version():
    old,new=fixture(),fixture(start=2025)
    a,b=AttachmentField.from_source(old,record_id='doc',version=1),AttachmentField.from_source(new,record_id='doc',version=2)
    left,right=common_attachment_observations(a,b)
    evolution=TensorEvolution.from_observations(left,right)
    wrong=TensorField(b.space,[1,1],source=FieldSource(new,'doc',3))
    with pytest.raises(ValueError):evolution.delta(a.amplitudes(),wrong)


def test_native_query_does_not_use_ranking_or_legacy_moment(monkeypatch):
    import rexgraph.ranking_response as ranking
    from rexgraph.coordinate_map import CoordinateMetric
    rex,calc,x=setup()
    def forbidden(*a,**k):raise AssertionError('native field used a collapsed observation')
    monkeypatch.setattr(ranking,'pagerank_solve',forbidden)
    monkeypatch.setattr(CoordinateMetric,'moment',forbidden)
    monkeypatch.setattr(np.linalg,'eigh',forbidden)
    result=Executor(sources={'g':rex},params={'x':x}).execute(parse('FROM $g RETURN NATIVE_RESPONSE($x)'))
    assert result.values[0].values[0]==Q(8,21)
