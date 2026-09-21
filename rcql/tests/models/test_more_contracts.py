from dataclasses import replace
from fractions import Fraction as Q
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from rexgraph.model_state import ModelInput,ModelBatch,model_input,thaw_tree
from rexgraph.model_runtime import native_model,infer_model,train_model,transport_model,model_coordinates
from rexgraph.nn.lifecycle import create_checkpoint,train_checkpoint,infer_checkpoint,ModelAdapter,register_model_adapter
from rexgraph.tensor_field import TensorField,FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.io.model_record import save_model,load_model,model_record,read_model_record,model_history,read_model_history
from .helpers import rex,state,batch,equal_tree


def test_real_relational_attention_model_resumes(tmp_path):
    from rexgraph.nn.relational_attention import CausalPropagatorAttention
    class Relational(torch.nn.Module):
        def __init__(self,g,config):
            super().__init__();self._rex=g
            self.attention=CausalPropagatorAttention(4,2,hops=2,window=2).double()
            self.readout=torch.nn.Linear(4,2,dtype=torch.float64)
        def forward(self,x):
            field,_=self.attention(x.unsqueeze(0))
            return self.readout(field.squeeze(0))
    register_model_adapter('test_relational_attention',ModelAdapter('1',Relational,lambda m,x:m(x),
        lambda p,y:torch.nn.functional.cross_entropy(p,y,reduction='none'),lambda m:{},transport_static=True))
    g=rex();m=create_checkpoint(g,adapter='test_relational_attention',axes=(CoordinateSpace('labels',('a','b')),),
                               optimizer={'name':'adam','lr':0.01},seed=73)
    inputs=TensorField(m.space,np.arange(12).reshape(3,4),(CoordinateSpace('features',('0','1','2','3')),),FieldSource(g),1,'chain')
    converted=model_input(inputs)
    b=ModelBatch(m.source,m.space,np.array([0,1,0]),np.array([True,True,True]),converted)
    mid=train_checkpoint(m,g,b,steps=2);save_model(tmp_path/'attention.safetensors',model_record(g,mid))
    saved,data=read_model_record(load_model(tmp_path/'attention.safetensors'))
    actual=train_checkpoint(saved,data,b,steps=2);wanted=train_checkpoint(m,g,b,steps=4)
    equal_tree(actual.payload['weights'],wanted.payload['weights'])
    equal_tree(actual.payload['optimizer'],wanted.payload['optimizer'])
    out=infer_checkpoint(actual,data,converted);assert out.values.shape==(3,2)


def test_exact_calculated_target_is_retained_for_learning(tmp_path):
    class Local(torch.nn.Module):
        def __init__(self,g,c):super().__init__();self._rex=g;self.z=torch.nn.Parameter(torch.zeros(g.nE,dtype=torch.float64))
    register_model_adapter('test_exact_targets',ModelAdapter('1',Local,lambda m,x:m.z,
        lambda p,y:(p-y)**2,lambda m:{'z':0},transport_static=True))
    g=rex();exact=native_model(g);x=TensorField(exact.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    truth=infer_model(exact,g,x);target=model_input(truth)
    m=create_checkpoint(g,adapter='test_exact_targets',axes=(),optimizer={'name':'adam'},seed=22)
    batch_=ModelBatch(m.source,m.space,target,np.ones(3,dtype=bool))
    learned=train_checkpoint(m,g,batch_,steps=2)
    save_model(tmp_path/'targets.rcbd',model_record(g,learned));saved,data=read_model_record(load_model(tmp_path/'targets.rcbd'))
    origin=saved.payload['observations']['target_input']
    assert tuple(origin.original)==(Q(8,21),Q(1,7),Q(1,21))
    assert origin.origin_digest==truth.coefficient_digest
    assert saved.arithmetic=='approximate' and truth.arithmetic=='rational'
    equal_tree(infer_checkpoint(saved,data).values,infer_checkpoint(learned,g).values)


@pytest.mark.parametrize('steps',[True,False,0,-1,1.5])
def test_all_learning_routes_refuse_invalid_step_count(steps):
    from rexgraph.flow.model import online_model
    g=rex();m=online_model(g)
    b=ModelBatch(m.source,m.space,np.ones(3),np.ones(3,dtype=bool))
    with pytest.raises((ValueError,TypeError)):train_model(m,g,b,steps=steps)


def test_restore_refuses_silent_parameter_dtype_cast():
    g=rex();m=state(g);p=thaw_tree(m.payload);p['weights']['Z']=p['weights']['Z'].astype(np.float32)
    bad=replace(m,payload=p)
    with pytest.raises(ValueError,match='dtype'):infer_checkpoint(bad,g)


def test_record_requires_actual_model_source():
    g=rex();m=state(g)
    changed=rex();changed.attach_metadata(1,0,'annotation_owner','other')
    with pytest.raises(ValueError):model_record(changed,m)


def test_transported_history_with_growing_source_roundtrips(tmp_path):
    from rexgraph.graph import RexGraph
    g=rex();m=train_checkpoint(state(g),g,batch(state(g)),steps=2)
    dest=RexGraph.from_hypergraph([0,2,4,6,8],[0,1,1,2,2,3,3,4],relation_ids=np.array([11,12,13,14]))
    J=CoordinateMap(m.space,model_coordinates(dest),[(0,0,1),(1,1,1),(2,2,1)])
    n=transport_model(m,g,FieldSource(dest),J)
    holder=model_history((m,n),(g,dest),(Q(1,3),Q(5,7)))
    save_model(tmp_path/'growing.safetensors',holder)
    timeline,snapshots=read_model_history(load_model(tmp_path/'growing.safetensors'))
    assert snapshots[1][1].nE==4 and timeline.at(Q(1,2))==0
    assert snapshots[1][0].payload['transport']['map_entries']==J.entries


def test_public_model_exports_are_resolvable():
    from rexgraph import model_input as public_input,ModelInput as PublicInput
    from rexgraph.nn import create_checkpoint as public_checkpoint
    from rexgraph.flow import online_model
    from rexgraph.io import model_record as public_record
    assert public_input is model_input and PublicInput is ModelInput
    assert callable(public_checkpoint) and callable(online_model) and public_record is model_record


def test_nested_nlp_attachments_remain_in_model_source(tmp_path):
    from agent.auto import build_rex_from_edges
    from agent.adapters.formats import read
    text='A rose in 2017 and 2024.'
    (tmp_path/'doc.txt').write_text(text)
    (tmp_path/'doc.ann').write_text('T1\tEvent 2 6\trose\nT2\tTime 10 14;19 23\t2017 2024\nE1\tEvent:T1 Time:T2\n')
    construction=read(tmp_path/'doc.ann',document_id='model_document')
    g=build_rex_from_edges(construction)
    from rexgraph.attachment_field import AttachmentField
    original=AttachmentField.from_source(g)
    assert len(original.attachments)>0
    m=create_checkpoint(g,configuration={'n_classes':2})
    save_model(tmp_path/'annotations.safetensors',model_record(g,m))
    saved,source=read_model_record(load_model(tmp_path/'annotations.safetensors'))
    restored=AttachmentField.from_source(source)
    assert restored.coefficient_digest==original.coefficient_digest
    assert any(len(a.text.components)==2 for a in restored.attachments if a.text is not None)


def test_numerical_objective_reduction_is_declared():
    class Local(torch.nn.Module):
        def __init__(self,g,c):super().__init__();self.z=torch.nn.Parameter(torch.zeros(g.nE,dtype=torch.float64))
    g=rex();values={}
    for reduction in ('mean','sum'):
        name='test_reduction_'+reduction
        register_model_adapter(name,ModelAdapter('1',Local,lambda m,x:m.z,lambda p,y:(p-y)**2,reduction=reduction))
        m=create_checkpoint(g,adapter=name,axes=(),optimizer={'name':'sgd','lr':0.1})
        b=ModelBatch(m.source,m.space,np.ones(3),np.ones(3,dtype=bool))
        n=train_checkpoint(m,g,b)
        values[reduction]=n.payload['weights']['z']
        assert n.payload['observations']['objective_reduction']==reduction
        np.testing.assert_array_equal(n.payload['observations']['loss_values'],np.ones(3))
    np.testing.assert_allclose(values['sum'],3*values['mean'],rtol=1e-15)


def test_exact_native_route_imports_no_torch():
    import subprocess,sys
    code="""import sys
from rexgraph.graph import RexGraph
from rexgraph.model_runtime import native_model,infer_model
from rexgraph.tensor_field import TensorField,FieldSource
from fractions import Fraction as Q
r=RexGraph.from_hypergraph([0,2],[0,1])
m=native_model(r)
x=TensorField(m.space,[1],source=FieldSource(r),grade=1,variance='chain')
assert infer_model(m,r,x).values[0]==Q(1,3)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)


def test_input_conversion_preserves_signed_zero():
    from rexgraph.model_state import ModelOutput
    g=rex();m=state(g)
    output=ModelOutput(np.array([-0.,0.,1.]),m.space,(),m.source,'a'*64,'approximate','recorded values')
    converted=model_input(output)
    assert np.signbit(converted.values[0])
    with pytest.raises(ValueError,match='conversion'):
        replace(converted,values=np.array([0.,0.,1.]))


def test_history_axis_and_unit_are_retained(tmp_path):
    g=rex();m=state(g)
    holder=model_history([m],[g],[Q(2017)],time_axis='annotation_revision',time_unit='abstract_year')
    save_model(tmp_path/'axis.rcbd',holder)
    timeline,_=read_model_history(load_model(tmp_path/'axis.rcbd'))
    assert timeline.time_axis=='annotation_revision' and timeline.time_unit=='abstract_year'


def test_registered_implementation_fingerprint_is_checked():
    g=rex();m=state(g);data=thaw_tree(m.payload)
    data['implementation']['callbacks']['forward']='0'*64
    changed=replace(m,payload=data)
    with pytest.raises(ValueError,match='implementation'):infer_checkpoint(changed,g)
