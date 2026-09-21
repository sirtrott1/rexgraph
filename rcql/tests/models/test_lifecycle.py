from dataclasses import replace
from fractions import Fraction as Q
import random
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from rexgraph.nn.lifecycle import (create_checkpoint,capture_checkpoint,infer_checkpoint,train_checkpoint,
    restore_checkpoint,ModelAdapter,register_model_adapter)
from rexgraph.model_runtime import native_model,infer_model,train_model
from rexgraph.model_state import ModelBatch,freeze_tree,thaw_tree
from rexgraph.tensor_field import FieldSource,TensorField
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMetric
from rexgraph.io.model_record import model_record,read_model_record,save_model,load_model
from rexgraph.io.rex_state import to_state,from_state
from .helpers import rex,state,batch,equal_tree


def test_exact_native_model_known_response():
    g=rex();m=native_model(g)
    x=TensorField(m.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    o=infer_model(m,g,x)
    assert tuple(o.values)==(Q(8,21),Q(1,7),Q(1,21))
    assert o.arithmetic=='rational' and o.tensor().variance=='chain' and o.tensor().grade==1


@pytest.mark.parametrize('operation',['green','hodge','gradient','curl','harmonic'])
def test_exact_native_action_matches_owner(operation):
    from rexgraph.native_field import NativeFieldCalculus
    g=rex();cal=NativeFieldCalculus.from_rex(g)
    m=native_model(g,operation=operation)
    x=TensorField(m.space,[[Q(1,3),2],[0,1],[3,4]],(CoordinateSpace('requests',('a','b')),),FieldSource(g),1,'chain')
    expected=(cal.green(1,Q(1)) if operation=='green' else cal.hodge(1) if operation=='hodge' else cal.sector(1,operation)).apply(x.values)
    out=infer_model(m,g,x)
    equal_tree(out.values,np.asarray(expected,dtype=object));assert out.axes==x.axes


def test_native_action_metric_change_has_its_own_identity():
    from rexgraph.native_field import NativeFieldCalculus
    g=rex();cal=NativeFieldCalculus.from_rex(g)
    metrics=list(cal.metrics);metrics[1]=CoordinateMetric(cal.complex.spaces[1],[(0,0,2),(1,1,3),(2,2,5)])
    m=native_model(g,metrics=metrics);plain=native_model(g)
    assert m.coefficient_digest!=plain.coefficient_digest
    x=TensorField(m.space,[1,2,3],source=FieldSource(g),grade=1,variance='chain')
    assert not np.array_equal(infer_model(m,g,x).values,infer_model(plain,g,x).values)


def test_exact_native_has_no_training_stage():
    g=rex();m=native_model(g)
    with pytest.raises(TypeError,match='training'):train_model(m,g,batch(m))


def test_numeric_output_cannot_become_exact_implicitly():
    g=rex();m=state(g);o=infer_checkpoint(m,g)
    with pytest.raises(TypeError,match='not an exact'):o.tensor()
    f=o.recorded_binary();assert f.grade==1 and f.variance=='cochain'
    assert all(isinstance(v,Q) for v in f.values.flat)
    assert o.arithmetic=='approximate'


@pytest.mark.parametrize('steps',[1,2,4])
def test_training_continuation_matches_uninterrupted(steps):
    g=rex();m=state(g);b=batch(m)
    a=train_checkpoint(m,g,b,steps=steps)
    saved,data=read_model_record(from_state(to_state(model_record(g,a))))
    b2=batch(saved)
    end=train_checkpoint(saved,data,b2,steps=3)
    full=train_checkpoint(m,g,b,steps=steps+3)
    equal_tree(end.payload['weights'],full.payload['weights'])
    equal_tree(end.payload['optimizer'],full.payload['optimizer'])
    assert end.step==steps+3 and end.parent==a.coefficient_digest
    assert np.count_nonzero(m.payload['weights']['Z'])==0


@pytest.mark.parametrize('extension',['.rcbd','.safetensors'])
def test_file_resume(extension,tmp_path):
    g=rex();m=state(g);a=train_checkpoint(m,g,batch(m),steps=2)
    save_model(tmp_path/('model'+extension),model_record(g,a))
    saved,data=read_model_record(load_model(tmp_path/('model'+extension)))
    assert saved.coefficient_digest==a.coefficient_digest
    continued=train_checkpoint(saved,data,batch(saved),steps=2)
    expected=train_checkpoint(m,g,batch(m),steps=4)
    equal_tree(continued.payload['weights'],expected.payload['weights'])
    equal_tree(continued.payload['optimizer'],expected.payload['optimizer'])


def test_observed_zero_differs_from_unobserved():
    g=rex();m=state(g)
    a=train_checkpoint(m,g,batch(m,(0,-10000,1)))
    b=train_checkpoint(m,g,batch(m,(0,10000,1)))
    equal_tree(a.payload['weights'],b.payload['weights'])
    assert a.payload['observations']['observed'].tolist()==[True,False,True]
    assert a.coefficient_digest!=b.coefficient_digest


@pytest.mark.parametrize('mask',[np.array([0,1,0]),np.array([True]),np.array([[True,False,True]])])
def test_mask_requires_declared_boolean_rows(mask):
    g=rex();m=state(g)
    with pytest.raises(ValueError):ModelBatch(m.source,m.space,np.array([0,1,0]),mask)


def test_empty_observations_refused_without_update():
    g=rex();m=state(g);digest=m.coefficient_digest
    b=ModelBatch(m.source,m.space,np.array([0,1,0]),np.zeros(3,dtype=bool))
    with pytest.raises(ValueError):train_checkpoint(m,g,b)
    assert m.coefficient_digest==digest


def test_rng_is_local_to_checkpoint():
    g=rex();m=state(g);random.seed(901);np.random.seed(912);torch.manual_seed(903)
    from rexgraph.nn.lifecycle import _rng
    before=_rng();train_checkpoint(m,g,batch(m),steps=2);infer_checkpoint(m,g)
    equal_tree(_rng(),before)


def test_source_mismatch_and_model_tamper():
    g=rex();m=state(g)
    with pytest.raises(ValueError):infer_checkpoint(m,rex((13,12,11)))
    weights=m.payload['weights']['Z'];weights.flags.writeable=True;weights[0,0]=1
    with pytest.raises(ValueError,match='changed'):m.check_state()


def test_inference_checkpoint_is_not_resumable():
    g=rex();m=state(g);model,_=restore_checkpoint(m,g)
    captured=capture_checkpoint(model,g,configuration={'n_classes':2})
    equal_tree(infer_checkpoint(m,g).values,infer_checkpoint(captured,g).values)
    with pytest.raises(ValueError,match='continuation'):train_checkpoint(captured,g,batch(captured))


def test_adapter_version_is_explicit():
    g=rex();m=state(g);changed=replace(m,adapter_version='does_not_match')
    with pytest.raises(ValueError,match='version'):infer_checkpoint(changed,g)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_numerical_dtype_is_preserved(dtype):
    g=rex();m=create_checkpoint(g,configuration={'n_classes':2,'dtype':dtype})
    a=train_checkpoint(m,g,batch(m),steps=2)
    assert infer_checkpoint(a,g).values.dtype==np.dtype(dtype)


def _dropout_adapter():
    class Local(torch.nn.Module):
        def __init__(self,g,c):
            super().__init__();self._rex=g
            self.drop=torch.nn.Dropout(0.3);self.linear=torch.nn.Linear(2,2,dtype=torch.float64)
        def forward(self,x):return self.linear(self.drop(x))
    register_model_adapter('test_dropout',ModelAdapter('1',Local,lambda m,x:m(x),
        lambda p,y:torch.nn.functional.cross_entropy(p,y,reduction='none')))


@pytest.mark.parametrize('extension',['.rcbd','.safetensors'])
def test_custom_model_dropout_and_adam_resume(extension,tmp_path):
    _dropout_adapter();g=rex();axes=(CoordinateSpace('labels',('no','yes')),)
    m=create_checkpoint(g,adapter='test_dropout',axes=axes,optimizer={'name':'adam','lr':0.01},seed=42)
    b=ModelBatch(m.source,m.space,np.array([0,1,0]),np.array([True,True,True]),np.array([[1.,2.],[3.,4.],[5.,6.]]))
    mid=train_checkpoint(m,g,b,steps=2)
    save_model(tmp_path/('dropout'+extension),model_record(g,mid))
    mid2,g2=read_model_record(load_model(tmp_path/('dropout'+extension)))
    random.random();np.random.randn(5);torch.rand(2)
    final=train_checkpoint(mid2,g2,ModelBatch(mid2.source,mid2.space,b.targets,b.observed,b.inputs),steps=3)
    uninterrupted=train_checkpoint(m,g,b,steps=5)
    equal_tree(final.payload['weights'],uninterrupted.payload['weights'])
    equal_tree(final.payload['optimizer'],uninterrupted.payload['optimizer'])
    equal_tree(final.payload['rng'],uninterrupted.payload['rng'])


def test_custom_model_named_noncell_coordinates():
    class Local(torch.nn.Module):
        def __init__(self,g,c):super().__init__();self.P=torch.nn.Parameter(torch.zeros(2,1,dtype=torch.float64))
    coords=lambda g,c:CoordinateSpace('local_regions',('fragment-A','fragment-B'))
    register_model_adapter('named_regions',ModelAdapter('1',Local,lambda m,x:m.P,
        lambda p,y:(p[:,0]-y)**2,coordinates=coords,grade=None,variance='coordinate'))
    g=rex();m=create_checkpoint(g,adapter='named_regions',axes=(CoordinateSpace('response',('y',)),),optimizer={'name':'adam'})
    b=ModelBatch(m.source,m.space,np.array([0.,1.]),np.array([True,True]))
    n=train_checkpoint(m,g,b);out=infer_checkpoint(n,g)
    assert out.space==coords(g,{}) and out.grade is None and out.variance=='coordinate'


def test_model_identity_includes_configuration_and_mask():
    g=rex();m=state(g)
    assert replace(m,configuration={**m.configuration,'description':'new'}).coefficient_digest!=m.coefficient_digest
    a=batch(m);b=replace(a,observed=np.array([True,True,True]))
    assert a.coefficient_digest!=b.coefficient_digest


def test_resume_environment_mismatch_refused():
    g=rex();m=state(g);payload=thaw_tree(m.payload);payload['environment']['torch']='wrong'
    changed=replace(m,payload=payload)
    with pytest.raises(ValueError,match='environment'):train_checkpoint(changed,g,batch(changed))
    assert infer_checkpoint(changed,g).values.shape==(3,2)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),lambda:None,object(),np.array([complex(1,2)])])
def test_model_codec_rejects_unsafe_payloads(bad):
    with pytest.raises((TypeError,ValueError)):freeze_tree({'v':bad})
