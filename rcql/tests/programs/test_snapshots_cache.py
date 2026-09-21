from dataclasses import replace
from fractions import Fraction as Q
import pytest

from rcql import Executor,parse,SourcePolicy,BoundSource
from rcql.source_context import SnapshotContext,SourceSelection
from rcql.query_cache import QueryCache
from rexgraph.tensor_field import TensorField,FieldSource
from rcdb import open_store as _open_store
from contextlib import closing
def open_store(*args,**kwargs):
    return closing(_open_store(*args,**kwargs))
from .helpers import native


def store_record(store,name='input',time=20,version=0):
    r,x,m=native()
    store.commit_mutation(name,r,expected_version=version,tx_time=time,analytics=False)
    return store.read_record(name)


def execution(snapshot,cache=None,evidence=None):
    from rcql.types import SourceRef
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.coordinate_map import CoordinateMetric
    from rexgraph.tensor_moment import CoordinatePairing
    r=snapshot.value
    space=NativeFieldCalculus.from_rex(r).complex.spaces[1]
    source=FieldSource(r,snapshot.record.id,snapshot.record.version,snapshot.state_digest)
    x=TensorField(space,[1,0,0],source=source,grade=1,variance='chain')
    m=CoordinatePairing.metric(CoordinateMetric.identity(space))
    ref=SourceRef('input',snapshot.state_digest,record_id=snapshot.record.id,record_version=snapshot.record.version)
    return Executor(sources={'r':BoundSource(r,SourcePolicy.allow('*'),ref=ref)},params={'x':x,'m':m},evidence=evidence)


QUERY=parse('FROM $r LET g=NATIVE_RESPONSE($x) RETURN g,MOMENT_SUPPORT(FIELD_PAIR(g,g,$m))')


def test_pinned_history_survives_future_change():
    with open_store('memory://') as store:
        store_record(store)
        context=SnapshotContext.select(store,(SourceSelection('r','input'),),cutoff=25)
        second=native()[0];second.attach_metadata(1,0,'revision','later')
        store.commit_mutation('input',second,expected_version=1,tx_time=30,analytics=False)
        again=SnapshotContext.select(store,(SourceSelection('r','input'),),cutoff=25)
        assert again.digest==context.digest
        assert context.sources['r'].ref.record_version==1
        assert SnapshotContext.select(store,(SourceSelection('r','input'),),cutoff=35).sources['r'].ref.record_version==2
        assert context.bindings()[0].ref.record_version==1
        context.check()
        e=Executor(sources=context.sources,evidence=context)
        assert e.execute(parse('FROM $r RETURN COUNT(CELLS(1))')).values==(3,)


def test_declared_evidence_closure_and_cutoff():
    with open_store('memory://') as store:
        first=store_record(store,'facts',10)
        result=native()[0]
        result.attach_metadata(1,0,'field',TensorField(native()[1].space,[2,3,4],source=FieldSource(None,'facts',1,first.state_digest)))
        store.commit_mutation('result',result,expected_version=0,tx_time=20,analytics=False)
        ctx=SnapshotContext.select(store,(SourceSelection('r','result'),),cutoff=25)
        assert len(ctx.entries)==2
        assert {k[1] for k in ctx.entries}=={'facts','result'}
        ctx.check()
        late=store_record(store,'later',40)
        retro=native()[0]
        retro._agent_meta={'rcql_evidence':[{'record_id':'later','version':1,'state_digest':late.state_digest}]}
        store.commit_mutation('retro',retro,expected_version=0,tx_time=22,analytics=False)
        with pytest.raises(ValueError,match='after'):SnapshotContext.select(store,(SourceSelection('r','retro'),),cutoff=25)

@pytest.mark.parametrize('case',('missing','digest','unversioned','malformed'))
def test_invalid_evidence_refused(case):
    with open_store('memory://') as store:
        first=store_record(store,'facts',10);r=native()[0]
        row={'record_id':'facts','version':1,'state_digest':first.state_digest}
        if case=='missing':row['record_id']='absent'
        if case=='digest':row['state_digest']='0'*64
        if case=='unversioned':row['version']=None
        if case=='malformed':row['extra']=1
        r._agent_meta={'rcql_evidence':[row]}
        store.commit_mutation('result',r,expected_version=0,tx_time=20,analytics=False)
        with pytest.raises((ValueError,TypeError)):SnapshotContext.select(store,(SourceSelection('r','result'),),cutoff=25)


def test_cutoff_checks_all_query_parameters():
    with open_store('memory://') as store:
        store_record(store,'facts',10);late=store_record(store,'future',30)
        ctx=SnapshotContext.select(store,(SourceSelection('r','facts'),),cutoff=20)
        data=native()[1]
        bad=replace(data,source=FieldSource(late.value,'future',1,late.state_digest))
        with pytest.raises(ValueError,match='outside'):
            Executor(sources=ctx.sources,params={'future':bad},evidence=ctx).execute(parse('FROM $r RETURN $future'))


def test_snapshot_source_mutation_detected():
    with open_store('memory://') as store:
        store_record(store)
        ctx=SnapshotContext.select(store,(SourceSelection('r','input'),),cutoff=25)
        ctx.sources['r'].value.attach_metadata(1,0,'changed',True)
        with pytest.raises(ValueError):ctx.check()


def test_snapshot_policy_not_recovered_from_data():
    with open_store('memory://') as store:
        store_record(store)
        ctx=SnapshotContext.select(BoundSource(store,SourcePolicy.allow('read','identity')),(SourceSelection('r','input'),),cutoff=25)
        src=ctx.sources['r']
        bad=BoundSource(src.value,SourcePolicy.allow('*'),ref=src.ref)
        with pytest.raises(PermissionError):Executor(sources={'r':bad},evidence=ctx).execute(parse('FROM $r RETURN COUNT(CELLS(1))'))

@pytest.mark.parametrize('clock',(float('nan'),True,'25'))
def test_invalid_cutoff(clock):
    with open_store('memory://') as store:
        store_record(store)
        with pytest.raises((TypeError,ValueError)):SnapshotContext.select(store,(SourceSelection('r','input'),),cutoff=clock)


def test_exact_clock_selection_is_recorded_clock_not_event_time():
    with open_store('memory://') as store:
        store_record(store,time=20)
        ctx=SnapshotContext.select(store,(SourceSelection('r','input',version=1),),cutoff=Q(41,2))
        assert ctx.cutoff==Q(41,2)
        with pytest.raises(ValueError):SnapshotContext.select(store,(SourceSelection('r','input',version=1),),cutoff=19)


def test_cache_reuses_fields_and_retains_sources(tmp_path,monkeypatch):
    import rcql.operators as ops
    original=ops._REGISTRY['NATIVE_RESPONSE'];calls=[]
    def counted(*args):calls.append(1);return original.fn(*args)
    monkeypatch.setitem(ops._REGISTRY,'NATIVE_RESPONSE',replace(original,fn=counted))
    with open_store('memory://') as source:
        first=store_record(source)
        cachepath='file://'+str(tmp_path/'results')
        with open_store(cachepath) as target:
            cache=QueryCache(target)
            e=execution(first)
            a=e.execute_cached(QUERY,cache)
            b=e.execute_cached(QUERY,cache)
            assert len(calls)==1
            assert a.native_plan['cache']['hit'] is False
            assert b.native_plan['cache']['hit'] is True
            assert b.execution[0]['operator']=='RESULT_REUSE'
            assert b.values[0].values.tolist()==[Q(8,21),Q(1,7),Q(1,21)]
            assert b.values[0].source.source is first.value
            assert b.values[0].source.record_id=='input'
            assert b.values[1].dependencies
        with open_store(cachepath) as reopened:
            c=execution(first).execute_cached(QUERY,QueryCache(reopened))
            assert c.native_plan['cache']['hit']
            assert len(calls)==1
            assert c.values[0].coefficient_digest==a.values[0].coefficient_digest

@pytest.mark.parametrize('change',('field','metric','version','policy','code'))
def test_cache_invalidation(tmp_path,monkeypatch,change):
    import rcql.query_cache as qc
    from rexgraph.coordinate_map import CoordinateMetric
    from rexgraph.tensor_moment import CoordinatePairing
    with open_store('memory://') as source,open_store('memory://') as target:
        first=store_record(source);cache=QueryCache(target);e=execution(first)
        old=e.execute_cached(QUERY,cache)
        if change=='field':e.params['x']=e.params['x'].with_values([2,0,0])
        if change=='metric':e.params['m']=CoordinatePairing.metric(CoordinateMetric(e.params['x'].space,((0,0,2),(1,1,1),(2,2,1))))
        if change=='version':
            source.commit_mutation('input',first.value,expected_version=1,tx_time=30,analytics=False)
            e=execution(source.read_record('input'))
        if change=='policy':
            value=e.sources['r'];e.sources['r']=BoundSource(value.value,SourcePolicy.allow('read','identity'),ref=value.ref)
        if change=='code':monkeypatch.setattr(qc,'implementation_identity',lambda:'changed_implementation')
        new=e.execute_cached(QUERY,cache)
        assert not new.native_plan['cache']['hit']
        assert old.native_plan['cache']['key']!=new.native_plan['cache']['key']


def test_cache_output_moment_roundtrip(tmp_path):
    with open_store('memory://') as source,open_store('memory://') as target:
        first=store_record(source);e=execution(first);cache=QueryCache(target)
        q=parse('FROM $r RETURN FIELD_PAIR($x,$x,$m)')
        a=e.execute_cached(q,cache);b=e.execute_cached(q,cache)
        assert b.values[0].support().values.tolist()==a.values[0].support().values.tolist()

@pytest.mark.parametrize('query',('FROM $r RETURN MODEL_TRAIN($m,$x,1)', 'FROM $r MATCH e IN CELLS(1) RETURN e'))
def test_cache_refuses_nonpure_queries(query):
    with open_store('memory://') as source,open_store('memory://') as target:
        first=store_record(source);e=execution(first)
        with pytest.raises((TypeError,ValueError)):e.execute_cached(parse(query),QueryCache(target))


def test_cache_no_result_on_failure():
    with open_store('memory://') as source,open_store('memory://') as target:
        first=store_record(source);e=execution(first)
        with pytest.raises((ValueError,TypeError)):e.execute_cached(parse('FROM $r RETURN TENSOR_SELECT($x,"missing","x")'),QueryCache(target))
        assert list(target.list())==[]


def test_evidence_context_in_cache_key():
    with open_store('memory://') as source,open_store('memory://') as target:
        store_record(source)
        one=SnapshotContext.select(source,(SourceSelection('r','input'),),cutoff=25)
        two=SnapshotContext.select(source,(SourceSelection('r','input'),),cutoff=26)
        q=parse('FROM $r RETURN COUNT(CELLS(1))');cache=QueryCache(target)
        a=Executor(sources=one.sources,evidence=one).execute_cached(q,cache)
        b=Executor(sources=two.sources,evidence=two).execute_cached(q,cache)
        assert a.values==b.values==(3,)
        assert a.native_plan['cache']['key']!=b.native_plan['cache']['key']


def test_cache_record_has_explicit_evidence_dependencies():
    with open_store('memory://') as source,open_store('memory://') as target:
        first=store_record(source)
        result=execution(first).execute_cached(QUERY,QueryCache(target))
        key='rcql_result/'+result.native_plan['cache']['key']
        record=target.read_record(key).value
        refs=record._agent_meta['rcql_evidence']
        assert any(r['record_id']=='input' and r['version']==1 for r in refs)


def test_cache_record_modified_under_same_key_is_not_trusted():
    with open_store('memory://') as source,open_store('memory://') as target:
        first=store_record(source);e=execution(first);cache=QueryCache(target)
        result=e.execute_cached(QUERY,cache)
        name='rcql_result/'+result.native_plan['cache']['key']
        record=target.read_record(name).value
        from rcql.program_codec import loads,dumps
        manifest=loads(record.get_metadata(1,0,'rcql_cache_manifest'));manifest['key']='wrong'
        record.attach_metadata(1,0,'rcql_cache_manifest',dumps(manifest).decode())
        target.commit_mutation(name,record,expected_version=1,analytics=False)
        with pytest.raises(ValueError,match='identity'):e.execute_cached(QUERY,cache)


def test_evidence_result_reports_cutoff_and_closure():
    with open_store('memory://') as source:
        store_record(source)
        context=SnapshotContext.select(source,(SourceSelection('r','input'),),cutoff=Q(51,2))
        result=Executor(sources=context.sources,evidence=context).execute(parse('FROM $r RETURN COUNT(CELLS(1))'))
        assert result.native_plan['evidence']['cutoff']=='51/2'
        assert result.provenance[0]['evidence']['closure'][0]['version']==1
