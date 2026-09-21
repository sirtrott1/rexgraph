from dataclasses import replace
from fractions import Fraction as Q
import numpy as np
import pytest

pytest.importorskip("torch")
from rexgraph.nn.lifecycle import create_checkpoint,train_checkpoint
from rexgraph.model_runtime import transport_model,train_model
from rexgraph.model_state import ModelBatch
from rexgraph.tensor_field import FieldSource
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.type_accession import CoordinateSpace
from rexgraph.graph import RexGraph,TemporalRex
from rexgraph.io.model_record import model_record,read_model_record,model_history,read_model_history,save_model,load_model
from rexgraph.io.model_state import pack_model,unpack_model
from .helpers import rex,state,batch,equal_tree


def permuted(g):
    return RexGraph.from_hypergraph([0,2,4,6],[2,3,0,1,1,2],relation_ids=np.array([13,11,12]))


def test_parameter_and_optimizer_permutation():
    g=rex();m=state(g);a=train_checkpoint(m,g,batch(m),steps=3);gp=permuted(g)
    mapping=CoordinateMap(a.space,a.space,[(0,2,1),(1,0,1),(2,1,1)])
    b=transport_model(a,g,FieldSource(gp),mapping,optimizer='carry')
    equal_tree(b.payload['weights']['Z'],a.payload['weights']['Z'][[2,0,1]])
    for k in ('m','v'):
        equal_tree(b.payload['optimizer']['state'][0][k],a.payload['optimizer']['state'][0][k][[2,0,1]])
    y=batch(a);yp=ModelBatch(b.source,b.space,y.targets[[2,0,1]],y.observed[[2,0,1]])
    next_a=train_checkpoint(a,g,y);next_b=train_checkpoint(b,gp,yp)
    np.testing.assert_allclose(next_b.payload['weights']['Z'],next_a.payload['weights']['Z'][[2,0,1]],rtol=0,atol=2e-15)
    assert b.source.state_digest!=a.source.state_digest


def test_rectangular_transport_resets_optimizer_explicitly():
    g=rex();m=state(g);a=train_checkpoint(m,g,batch(m),steps=2)
    dest=RexGraph.from_hypergraph([0,2,4,6,8],[0,1,1,2,2,3,3,4],relation_ids=np.array([11,12,13,14]))
    from rexgraph.model_runtime import model_coordinates
    J=CoordinateMap(a.space,model_coordinates(dest),[(0,0,1),(1,1,1),(2,2,1)])
    with pytest.raises(ValueError,match='permutation'):transport_model(a,g,FieldSource(dest),J,optimizer='carry')
    b=transport_model(a,g,FieldSource(dest),J,optimizer='reset')
    np.testing.assert_array_equal(b.payload['weights']['Z'][:3],a.payload['weights']['Z'])
    np.testing.assert_array_equal(b.payload['weights']['Z'][3],np.zeros(2))
    assert b.payload['optimizer']['state']=={} and b.payload['observations'] is None
    assert b.parent==a.coefficient_digest and b.payload['transport']['map_digest']==J.coefficient_digest


def test_nonpermutation_cannot_carry_adam_moments():
    g=rex();a=state(g);J=CoordinateMap(a.space,a.space,[(0,0,1),(0,1,Q(1,2)),(1,1,1),(2,2,1)])
    with pytest.raises(ValueError,match='permutation'):transport_model(a,g,FieldSource(g),J,optimizer='carry')


def test_general_model_transport_needs_static_contract():
    from .test_lifecycle import _dropout_adapter
    _dropout_adapter();g=rex();m=create_checkpoint(g,adapter='test_dropout',axes=(CoordinateSpace('outputs',('a','b')),))
    with pytest.raises(ValueError,match='independent'):transport_model(m,g,FieldSource(g),CoordinateMap.identity(m.space))


def test_online_lineage_preserves_parallel_identities():
    from rexgraph.flow.model import online_model,infer_online,correct_online
    g=RexGraph.from_hypergraph([0,2,4],[0,1,0,1],relation_ids=np.array([71,72]))
    m=online_model(g,values=[1,7]);assert list(m.payload['relation_ids'])==[71,72]
    b=ModelBatch(m.source,m.space,np.array([0.,20.]),np.array([True,False]))
    o=correct_online(m,g,b)
    assert o.payload['values'][0]!=o.payload['values'][1]
    assert o.payload['observed'].tolist()==[True,False]
    np.testing.assert_array_equal(o.payload['prediction'],infer_online(m,g).values)
    assert m.payload['values'].tolist()==[1,7]


def test_online_without_explicit_primary_ids_refused():
    from rexgraph.flow.model import online_model
    g=RexGraph.from_hypergraph([0,2],[0,1])
    with pytest.raises(ValueError,match='identities'):online_model(g)


def test_online_store_and_continue(tmp_path):
    from rexgraph.flow.model import online_model
    g=rex();m=online_model(g);b=ModelBatch(m.source,m.space,np.array([1.,0.,0.]),np.array([True,True,True]))
    a=train_model(m,g,b)
    save_model(tmp_path/'field.safetensors',model_record(g,a));saved,s=read_model_record(load_model(tmp_path/'field.safetensors'))
    assert a.coefficient_digest==saved.coefficient_digest
    b2=ModelBatch(saved.source,saved.space,np.array([2.,3.,4.]),np.array([True,False,True]))
    actual=train_model(saved,s,b2);expected=train_model(a,g,b2)
    equal_tree(actual.payload,expected.payload)


@pytest.mark.parametrize('extension',['.rcbd','.safetensors'])
def test_history_retains_exact_clock_full_source_and_training(extension,tmp_path):
    g=rex();g.attach_metadata(1,0,'note','first source annotation')
    m=state(g);n=train_checkpoint(m,g,batch(m))
    eps=Q(1,10**40);times=(Q(1,3),Q(1,3)+eps)
    holder=model_history([m,n],[g,g],times)
    timeline,_=read_model_history(holder)
    assert timeline.at(Q(1,3))==0 and timeline.at(times[1])==1
    assert timeline.at(Q(1,3)+eps/2)==0
    assert timeline.temporal_state().header['times']==[0.,1.]
    save_model(tmp_path/('history'+extension),holder)
    restored=load_model(tmp_path/('history'+extension));old,data=read_model_history(restored,times[0]);new,_=read_model_history(restored,times[1])
    assert old.coefficient_digest==m.coefficient_digest and new.coefficient_digest==n.coefficient_digest
    assert data.get_metadata(1,0,'note')=='first source annotation'


def test_history_rejects_unrelated_temporal_index():
    from rexgraph.io.temporal_state import to_temporal_state
    g=rex();m=state(g);holder=model_history([m],[g],[Q(1,3)])
    timeline,_=read_model_history(holder)
    other=TemporalRex([]);other.append_snapshot(RexGraph.from_hypergraph([0,2],[0,1],relation_ids=np.array([9])),at=0)
    different=to_temporal_state(other)
    bad=replace(timeline,temporal_header=different.header,temporal_tensors=different.tensors)
    holder.attach_metadata(1,0,'model_timeline',bad)
    with pytest.raises(ValueError,match='index differs'):read_model_history(holder)


@pytest.mark.parametrize('times',[(Q(1,2),Q(1,2)),(2,1),(0.1,0.2)])
def test_history_rejects_ambiguous_or_inexact_times(times):
    g=rex();m=state(g);n=train_checkpoint(m,g,batch(m))
    with pytest.raises((TypeError,ValueError)):model_history([m,n],[g,g],times)


def test_history_requires_parent_chain():
    g=rex();m=state(g);n=create_checkpoint(g,configuration={'n_classes':2},seed=40)
    with pytest.raises(ValueError,match='parent'):model_history([m,n],[g,g],[1,2])


@pytest.mark.parametrize('value',[np.array([2**200+11,Q(-7,13)],dtype=object),np.array([True,False]),np.array(2.5),np.array([0.,-0.],dtype=np.float32),b'\x00\xff'])
def test_model_payload_typed_values_survive_formats(value,tmp_path):
    g=rex();m=replace(state(g),payload={'test':value})
    for ext in ('.rcbd','.safetensors'):
        p=tmp_path/('payload'+ext);save_model(p,model_record(g,m));back,_=read_model_record(load_model(p))
        assert back.coefficient_digest==m.coefficient_digest
        equal_tree(back.payload['test'],value)


def test_model_codec_detects_tampering_and_unused_arrays():
    g=rex();m=state(g);packed=pack_model(m)
    extra={**packed,'unclaimed':np.zeros(3)}
    with pytest.raises(ValueError):unpack_model(extra)
    broken={k:v.copy() for k,v in packed.items()};key=next(k for k,v in broken.items() if v.dtype.kind=='f');broken[key].flat[0]+=1
    with pytest.raises(ValueError,match='identity'):unpack_model(broken)


