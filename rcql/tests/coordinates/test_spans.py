from dataclasses import replace
from fractions import Fraction as Q
import numpy as np
import pytest
from rexgraph.span import SpanBlock,SpanRefinement,SpanAttachment
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric
from rexgraph.temporal_calculus import TemporalOperation
from rexgraph.io.rex_state import to_state,from_state
from rexgraph.io.catalog import object_digest
from rexgraph.graph import RexGraph


def block(components=((2017,2019),(2024,2026)),name='dates',axis='event',unit='year'):
    return SpanBlock(name,axis,unit,tuple((str(i),a,b) for i,(a,b) in enumerate(components)))


def attachment():
    text=block(((0,3),(8,13)),name='T1',axis='doc:sha256',unit='character')
    time=block()
    gamma=CoordinateMap(text.coordinates,time.coordinates,((0,0,Q(1,3)),(1,1,2)))
    return SpanAttachment('A1','E1','time','corpus/doc/version1',text,time,gamma,(('status','asserted'),))


@pytest.mark.parametrize('components,values,q', [(((2017,2019),),[1],2),(((2017,2019),(2024,2026)),[1,0,1],4), (((2019,2025),(2020,2026)),[1,2,1],22)])
def test_exact_support_observations(components,values,q):
    b=block(components); r=SpanRefinement.from_blocks([b]); f=r.coverage(b,mode='sum')
    assert f.tolist()==values
    assert r.metric.moment(f,f)==q
    expected=[-values[0]]+[values[i-1]-values[i] for i in range(1,len(values))]+[values[-1]]
    assert r.boundary.apply(f).tolist()==expected


def test_overlapping_components_union_and_gram():
    b=block(((2019,2025),(2020,2026))); r=SpanRefinement.from_blocks([b])
    f=r.realization(b)
    x=f.apply([1,0]);y=f.apply([0,1])
    assert [[r.metric.moment(a,z) for z in [x,y]] for a in [x,y]]==[[6,5],[5,6]]
    union=r.coverage(b,mode='union')
    assert union.tolist()==[1,1,1]
    assert r.metric.moment(union,union)==7
    assert r.boundary.apply(r.coverage(b,mode='sum')).tolist()==[-1,-1,1,1]
    assert r.boundary.apply(union).tolist()==[-1,0,0,1]


def test_refinement_preserves_boundary_and_metric_exactly():
    b=block(((Q(1,3),Q(7,3)),)); coarse=SpanRefinement.from_blocks([b])
    fine=SpanRefinement(b.axis,b.unit,(Q(1,3),Q(2,3),1,Q(7,3)))
    j=coarse.refinement_map(fine);j0=coarse.endpoint_map(fine)
    assert fine.boundary.apply(j.apply([Q(5,7)])).tolist()==j0.apply(coarse.boundary.apply([Q(5,7)])).tolist()
    assert fine.metric.moment(j.apply([3]),j.apply([3]))==coarse.metric.moment([3],[3])==18
    with pytest.raises(ValueError): fine.refinement_map(coarse)
    with pytest.raises(ValueError): coarse.refinement_map(replace(fine,unit='seconds'))


def test_boundary_shift_delta_and_moment_difference():
    b=block();bp=block(((2017,2019),(2025,2026)))
    r=SpanRefinement.from_blocks([b,bp]);x=r.coverage(b,mode='union');y=r.coverage(bp,mode='union');d=y-x
    assert r.points==(2017,2019,2024,2025,2026)
    assert d.tolist()==[0,0,-1,0]
    assert r.metric.moment(d,d)==1
    assert r.metric.moment(y,y)-r.metric.moment(x,x)==-1
    assert r.boundary.apply(d).tolist()==[0,0,1,-1,0]
    op=TemporalOperation(r.realization(b),r.realization(bp),CoordinateMap.identity(b.coordinates),CoordinateMap.identity(r.space))
    assert op.field_delta([1,1],[1,1]).moments(r.metric).contract()==1


def test_duplicate_event_support_shared_and_local():
    r=RexGraph.from_cells([2,[[0,1],[0,1]]]);b=block();f=SpanRefinement.from_blocks([b])
    a=f.accession(r,1,'shared',((0,b),(1,b)),mode='union')
    local=f.accession(r,1,'local',((0,b),(1,b)),mode='union',local=True)
    from rexgraph.cochain import Chain
    def val(acc,values): return acc.apply(Chain(1,np.array(values,object),source=r),exact=True).values
    left=val(a,[1,0]);right=val(a,[0,1])
    assert f.metric.moment(left,right)==4
    weights=tuple(z-x for x,z in f.intervals)*2
    metric=CoordinateMetric.diagonal(local.coordinates,weights)
    l=val(local,[1,0]);rr=val(local,[0,1])
    assert metric.moment(l,rr)==0
    assert metric.moment(rr-l,rr-l)==8
    assert f.metric.moment(right-left,right-left)==0


def test_exact_huge_endpoints():
    n=2**200+1;b=block(((Q(n,7),Q(n,7)+Q(2,9)),))
    r=SpanRefinement.from_blocks([b]); assert r.metric.moment([1],[1])==Q(2,9)
    assert r.points[0]==Q(n,7)


@pytest.mark.parametrize('bad', [(((1,1),)),(((2,1),)),(((1.0,2),)),(((True,2),))])
def test_invalid_component_refusal(bad):
    with pytest.raises((ValueError,TypeError)): block(bad)


def test_alternative_is_not_automatically_union():
    b=replace(block(),interpretation='alternative');r=SpanRefinement.from_blocks([b])
    for mode in ('sum','union'):
        with pytest.raises(ValueError): r.coverage(b,mode=mode)
    assert r.realization(b).apply([1,0]).tolist()==[1,0,0]


def test_empty_support_has_empty_realization():
    b=block(());r=SpanRefinement.from_blocks([b])
    assert r.coverage(b,mode='union').tolist()==[]
    assert r.metric.moment([],[])==0
    assert r.realization(b).shape==(0,0)


def test_grounding_and_role_identity():
    a=attachment()
    g=CoordinateMap(a.text.coordinates,a.time.coordinates,((1,0,Q(1,3)),(0,1,2)))
    for other in (replace(a,grounding=g),replace(a,owner_id='E2'),replace(a,role='onset')):
        assert other.coefficient_digest != a.coefficient_digest
    with pytest.raises(ValueError): replace(a,grounding=CoordinateMap.identity(a.time.coordinates))


@pytest.mark.parametrize('nested',[False,True])
def test_canonical_annotation_roundtrip(nested):
    a=attachment();child=RexGraph.from_cells([2,[[0,1]]]);child.attach_metadata(1,0,'annotation',a)
    if nested:
        g=RexGraph.from_cells([1,[[0]]]);g.attach_metadata(1,0,'child',child)
    else:g=child
    state=to_state(g)
    assert state.header['format_version']==4
    restored=from_state(state)
    target=restored.get_metadata(1,0,'child') if nested else restored
    assert target.get_metadata(1,0,'annotation')==a
    assert object_digest(restored)==object_digest(g)
    assert to_state(restored).header==state.header


def test_legacy_numeric_and_exact_state_versions_unchanged():
    numeric=RexGraph.from_cells([2,[[0,1]]]); assert to_state(numeric).header['format_version']==2
    exact=RexGraph.from_hypergraph([0,2],[0,1],w_E=np.array([Q(1,3)],object))
    assert to_state(exact).header['format_version']==3
    assert from_state(to_state(exact)).w_E[0]==Q(1,3)


def test_changed_annotation_changes_native_digest():
    g=RexGraph.from_cells([2,[[0,1]]]);a=attachment();g.attach_metadata(1,0,'annotation',a)
    before=object_digest(g);g.attach_metadata(1,0,'annotation',replace(a,owner_id='E2'))
    assert before != object_digest(g)


def test_mixed_metadata_refuses_lossy_fallback():
    g=RexGraph.from_cells([2,[[0,1],[0,1]]]);g.attach_metadata(1,0,'annotation',attachment());g.attach_metadata(1,1,'annotation','wrong')
    with pytest.raises(TypeError): to_state(g)


def test_corrupt_schema_is_refused():
    import json
    from rexgraph.io.span_state import pack_attachment,unpack_attachment
    ts=pack_attachment(attachment()); spec=json.loads(ts['spec'].tobytes()); spec['version']=99
    ts['spec']=np.frombuffer(json.dumps(spec).encode(),np.uint8)
    with pytest.raises(ValueError): unpack_attachment(ts)


@pytest.mark.parametrize('backend',['memory','file','rex','sqlite'])
def test_rcdb_versions_and_query_roundtrip(tmp_path,backend):
    from rcdb import open_store, RexStore
    from rcql import Executor,parse
    if backend=='rex':store=RexStore(str(tmp_path/'native'))
    else:
        uri={'memory':'memory://','file':'file://'+str(tmp_path/'files'),'sqlite':'sqlite:///'+str(tmp_path/'sql.db')}[backend]
        store=open_store(uri)
    try:
        g=RexGraph.from_cells([2,[[0,1]]]);a=attachment();g.attach_metadata(1,0,'annotation',a)
        c1=store.commit_mutation('doc',g,expected_version=0)
        g2=from_state(to_state(g));b=replace(a,time=replace(a.time,components=(('0',2017,2019),('1',2025,2026))))
        g2.attach_metadata(1,0,'annotation',b)
        c2=store.commit_mutation('doc',g2,expected_version=c1.version)
        assert store.read_record('doc',version=c1.version).value.get_metadata(1,0,'annotation')==a
        assert store.read_record('doc',version=c2.version).value.get_metadata(1,0,'annotation')==b
        assert store.verify_commits('doc')
        out=Executor(sources={'db':store}).execute(parse('FROM $db RETURN RCDB_GET("doc")')).values[0]
        assert out.get_metadata(1,0,'annotation')==b
    finally:store.close()


@pytest.mark.parametrize('format_name',['bundle','safetensors','hdf5'])
def test_annotation_file_formats(tmp_path,format_name):
    g=RexGraph.from_cells([2,[[0,1]]]);a=attachment();g.attach_metadata(1,0,'annotation/role%@',a)
    if format_name=='bundle':
        from rexgraph.io.bundle import save_rcbd,load_rcbd
        save_rcbd(str(tmp_path/'state.rcbd'),g);out=load_rcbd(str(tmp_path/'state.rcbd'))
    elif format_name=='safetensors':
        from rexgraph.io.safetensors_bridge import rex_to_safetensors,safetensors_to_rex
        rex_to_safetensors(g,str(tmp_path/'state.safetensors'));out=safetensors_to_rex(str(tmp_path/'state.safetensors'))
    else:
        pytest.importorskip('h5py')
        from rexgraph.io.hdf5_format import save_hdf5,load_hdf5
        save_hdf5(str(tmp_path/'state.h5'),g);out=load_hdf5(str(tmp_path/'state.h5'))
    assert out.get_metadata(1,0,'annotation/role%@')==a
    assert object_digest(out)==object_digest(g)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_store_reopen_with_unrelated_working_directory(tmp_path,backend,monkeypatch):
    from rcdb import open_store
    uri={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'sql.db')}[backend]
    g=RexGraph.from_cells([2,[[0,1]]]);a=attachment();g.attach_metadata(1,0,'annotation',a)
    store=open_store(uri)
    try:store.put('document',g)
    finally:store.close()
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();monkeypatch.chdir(elsewhere)
    store=open_store(uri)
    try:
        out=store.read_record('document').value
        assert out.get_metadata(1,0,'annotation')==a
        assert object_digest(out)==object_digest(g)
    finally:store.close()


def test_complete_mutation_replay_retains_annotation():
    from rexgraph.io.mutation import prepare_mutation,mutation_to_bytes,mutation_from_bytes,apply_mutation
    old=RexGraph.from_cells([2,[[0,1]]]);a=attachment();old.attach_metadata(1,0,'annotation',a)
    new=from_state(to_state(old));new.attach_metadata(1,0,'annotation',replace(a,owner_id='E2'))
    package=mutation_from_bytes(mutation_to_bytes(prepare_mutation(old,new,tx_time=1)))
    out=apply_mutation(package,previous=old)
    assert out.get_metadata(1,0,'annotation').owner_id=='E2'
    assert object_digest(out)==object_digest(new)


def test_spec_payload_is_sealed():
    from rexgraph.io.rex_state import verify_state
    g=RexGraph.from_cells([2,[[0,1]]]);g.attach_metadata(1,0,'annotation',attachment());state=to_state(g)
    name=next(k for k in state.tensors if k.startswith('annotation/') and k.endswith('/spec'))
    state.tensors[name][0]^=1
    assert not verify_state(state)
    with pytest.raises(ValueError):from_state(state)


def test_legacy_reader_explicitly_refuses_new_schema():
    import rexgraph.io.rex_state as rs
    g=RexGraph.from_cells([2,[[0,1]]]);g.attach_metadata(1,0,'annotation',attachment());state=to_state(g)
    # The old reader guard has the same entry condition and only its old version set.
    versions=rs.READABLE_VERSIONS
    try:
        rs.READABLE_VERSIONS=(1,2,3)
        with pytest.raises(ValueError):rs.from_state(state)
    finally:rs.READABLE_VERSIONS=versions
