from dataclasses import replace
from fractions import Fraction as Q
import json
import numpy as np
import pytest

pytest.importorskip("torch")
from rcql import Executor,parse,query,source,call,mutation,BoundSource,SourcePolicy
from rcql.binding import bind
from rcql.types import Exactness
from rexgraph.model_runtime import native_model,infer_model
from rexgraph.nn.lifecycle import create_checkpoint,train_checkpoint,infer_checkpoint
from rexgraph.model_state import ModelBatch,ModelInput,model_input
from rexgraph.tensor_field import TensorField,FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.io.model_record import model_record,read_model_record,model_history
from .helpers import rex,state,batch,equal_tree


def prepared(name):
    g=rex();m=state(g);b=batch(m)
    native=native_model(g);x=TensorField(native.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    out=infer_model(native,g,x)
    args={
        'MODEL_INIT':('coparticipation',{'n_classes':2}),
        'MODEL_BATCH':(m,b.targets,b.observed), 'MODEL_INFER':(m,), 'MODEL_TRAIN':(m,b),
        'MODEL_TRANSPORT':(m,bind('destination',g,SourcePolicy.allow('*')),CoordinateMap.identity(m.space)),
        'MODEL_RECORD':(m,), 'MODEL_FIELD':(out,), 'MODEL_VALUES':(out,), 'MODEL_INFO':(m,),
        'MODEL_INPUT':(x,), 'MODEL_CERTIFY':(native,x,out),
        'MODEL_HISTORY':((m,), (bind('data',g,SourcePolicy.allow('*')),), (Q(1,3),)),
    }
    if name=='MODEL_STATE':return model_record(g,m),(0,)
    if name=='MODEL_AT':return model_history([m],[g],[Q(1,3)]),(Q(1,3),)
    return g,args[name]


@pytest.mark.parametrize('name',[
    'MODEL_INIT','MODEL_STATE','MODEL_AT','MODEL_BATCH','MODEL_INFER','MODEL_TRAIN','MODEL_TRANSPORT',
    'MODEL_RECORD','MODEL_HISTORY','MODEL_FIELD','MODEL_VALUES','MODEL_INPUT','MODEL_INFO','MODEL_CERTIFY'])
def test_direct_planned_contracts(name):
    from rcql.operators import get_operator
    from rcql.executor import value_exactness
    from rcql.signatures import lookup
    g,args=prepared(name)
    direct=get_operator(name).fn(g,*args)
    ex=Executor(sources={'g':g});q=query(source('g'),call(name,*args))
    result=ex.execute(q);plan=ex.execute(replace(q,explain=True)).values[0]
    assert value_exactness(direct)==result.exactness[0]
    assert plan['returns'][0]['result']['exactness']==result.exactness[0].value
    assert plan['returns'][0]['requires']==sorted(lookup(name).requires)
    assert plan['returns'][0]['memoizable']==lookup(name).memoizable


def test_nested_model_query_keeps_training_functional():
    g=rex();before=FieldSource(g).state_digest
    text='FROM $g LET m = MODEL_INIT("coparticipation", $config) LET b = MODEL_BATCH(m, $y, $mask) LET n = MODEL_TRAIN(m,b,2) RETURN m, n, MODEL_INFER(n), MODEL_RECORD(n)'
    out=Executor(sources={'g':g},params={'config':{'n_classes':2},'y':np.array([0,99,1]),'mask':np.array([True,False,True])}).execute(parse(text))
    old,new,pred,record=out.values
    assert old.step==0 and new.step==2 and new.parent==old.coefficient_digest
    assert out.exactness==(Exactness.STRUCTURAL,Exactness.STRUCTURAL,Exactness.APPROXIMATE,Exactness.STRUCTURAL)
    assert FieldSource(g).state_digest==before
    saved,_=read_model_record(record);assert saved.coefficient_digest==new.coefficient_digest


def test_native_model_query_stays_exact_without_learning():
    g=rex();m=native_model(g);x=TensorField(m.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    result=Executor(sources={'g':g},params={'x':x}).execute(parse('FROM $g RETURN MODEL_FIELD(MODEL_INFER(MODEL_INIT(),$x))'))
    assert tuple(result.values[0].values)==(Q(8,21),Q(1,7),Q(1,21))
    assert result.exactness==(Exactness.RATIONAL,)


def test_numerical_input_retains_original_rational_values(tmp_path):
    from .test_lifecycle import _dropout_adapter
    from rexgraph.io.model_record import save_model,load_model
    _dropout_adapter();g=rex();m=create_checkpoint(g,adapter='test_dropout',axes=(CoordinateSpace('classes',('n','y')),))
    field=TensorField(m.space,[[Q(1,3),1],[2,Q(1,7)],[3,4]],(CoordinateSpace('features',('a','b')),),FieldSource(g),1,'chain')
    data=model_input(field,'float64')
    assert data.original[0,0]==Q(1,3) and data.arithmetic=='rounded' and data.axes==field.axes
    b=ModelBatch(m.source,m.space,np.array([0,1,0]),np.ones(3,dtype=bool),data)
    n=train_checkpoint(m,g,b)
    save_model(tmp_path/'input.safetensors',model_record(g,n));saved,s=read_model_record(load_model(tmp_path/'input.safetensors'))
    inp=saved.payload['observations']['inputs']
    assert isinstance(inp,ModelInput) and inp.coefficient_digest==data.coefficient_digest
    assert inp.original[1,1]==Q(1,7)
    a=train_checkpoint(saved,s,ModelBatch(saved.source,saved.space,b.targets,b.observed,inp))
    b2=train_checkpoint(n,g,b)
    equal_tree(a.payload['weights'],b2.payload['weights'])


def test_model_input_wrong_source_rejected():
    g=rex();m=state(g);other=rex((91,92,93))
    f=TensorField(m.space,[1,2,3],source=FieldSource(other),grade=1,variance='chain')
    with pytest.raises(ValueError,match='source'):ModelBatch(m.source,m.space,np.array([0,1,0]),np.ones(3,dtype=bool),model_input(f))


def test_model_input_source_and_named_axes_are_query_results():
    g=rex();m=native_model(g);x=TensorField(m.space,[[1,Q(1,3)],[0,2],[3,4]],(CoordinateSpace('inputs',('a','b')),),FieldSource(g),1,'chain')
    out=Executor(sources={'g':g},params={'x':x}).execute(parse('FROM $g RETURN MODEL_INPUT($x,"float32")'))
    inp=out.values[0];assert inp.values.dtype==np.float32 and inp.original[0,1]==Q(1,3)
    assert inp.source.state_digest==FieldSource(g).state_digest and inp.axes==x.axes


def test_read_only_policy_refuses_training_but_permits_inference():
    g=rex();m=state(g)
    ex=Executor(sources={'g':BoundSource(g,SourcePolicy.allow('read'))})
    result=ex.execute(query(source('g'),call('MODEL_INFER',m)))
    assert result.values[0].values.shape==(3,2)
    q=query(source('g'),call('MODEL_TRAIN',m,batch(m)))
    for explain in (False,True):
        with pytest.raises(PermissionError):ex.execute(replace(q,explain=explain))


def test_explain_does_not_build_or_train(monkeypatch):
    import rexgraph.nn.lifecycle as mod
    g=rex();m=state(g)
    def forbidden(*args,**kwargs):raise AssertionError('model execution during explanation')
    monkeypatch.setattr(mod,'_restore',forbidden)
    ex=Executor(sources={'g':g})
    q=query(source('g'),call('MODEL_TRAIN',m,batch(m)),call('MODEL_INFER',m),explain=True)
    ex.execute(q)


def test_repeated_functional_training_expression_is_memoized(monkeypatch):
    import rexgraph.nn.lifecycle as mod
    g=rex();m=state(g);calls=[];real=mod.train_checkpoint
    def counted(*args,**kw):calls.append(1);return real(*args,**kw)
    monkeypatch.setattr(mod,'train_checkpoint',counted)
    expr=call('MODEL_TRAIN',m,batch(m))
    r=Executor(sources={'g':g}).execute(query(source('g'),expr,expr))
    assert len(calls)==1 and r.values[0].coefficient_digest==r.values[1].coefficient_digest


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_rcdb_model_versions_commit_reopen_and_train(backend,tmp_path):
    from rcdb import open_store
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'native'),'sqlite':'sqlite:///'+str(tmp_path/'model.sqlite')}[backend]
    db=open_store(url).configure_security(require_commits=True)
    g=rex();db.commit_mutation('data',g,expected_version=0)
    ex=Executor(sources={'db':db},params={'config':{'n_classes':2}})
    m=ex.execute(parse('FROM RCDB_VERSION($db,"data",1) RETURN MODEL_INIT("coparticipation",$config)')).values[0]
    assert m.source.record_id=='data' and m.source.version==1
    record=model_record(g,m)
    # Publish through the existing mutation language, not a hidden training side effect.
    ex.execute(mutation(source('db'),'model',record,expected_version=0))
    before=ex.execute(parse('FROM RCDB_VERSION($db,"model",1) RETURN MODEL_VALUES(MODEL_INFER(MODEL_STATE()))')).values[0]
    n=ex.execute(query(__import__('rcql').source_call('RCDB_VERSION',source('db'),'model',1),call('MODEL_TRAIN',m,batch(m),2))).values[0]
    ex.execute(mutation(source('db'),'model',model_record(g,n),expected_version=1))
    assert len(db.history('model'))==2 and db.verify_commits('model')
    with pytest.raises(Exception):ex.execute(mutation(source('db'),'model',record,expected_version=1))
    db.close();db=open_store(url)
    old=Executor(sources={'db':db}).execute(parse('FROM RCDB_VERSION($db,"model",1) RETURN MODEL_VALUES(MODEL_INFER(MODEL_STATE()))')).values[0]
    np.testing.assert_array_equal(old,before)
    snap=db.read_record('model',version=2);saved,data=read_model_record(snap.value)
    resumed=train_checkpoint(saved,data,batch(saved),steps=2)
    direct=train_checkpoint(m,g,batch(m),steps=4)
    equal_tree(resumed.payload['weights'],direct.payload['weights'])
    db.close()


def test_model_bound_to_selected_source_version():
    g=rex();m=replace(state(g),source=FieldSource(g,'data',2))
    with pytest.raises(ValueError,match='record'):Executor(sources={'g':g}).execute(query(source('g'),call('MODEL_INFER',m)))


def test_agent_runtime_uses_same_native_training_contract():
    from agent.rcql_runtime import RCQLRuntime
    runtime=RCQLRuntime();g=rex();runtime.register('g',g,policy=SourcePolicy.allow('read','train'))
    result=runtime.execute(parse('FROM $g LET m = MODEL_INIT("coparticipation",$cfg) LET b = MODEL_BATCH(m,$y,$mask) RETURN MODEL_TRAIN(m,b,2)'),params={'cfg':{'n_classes':2},'y':np.array([0,1,0]),'mask':np.array([True,True,True])})
    assert result.values[0].step==2
    runtime.register('g',g,policy=SourcePolicy.allow('read'))
    with pytest.raises(PermissionError):runtime.execute(query(source('g'),call('MODEL_TRAIN',result.values[0],batch(result.values[0]))))


def test_system_model_preview_does_not_dump_training_state():
    from system.serialize import json_value
    g=rex();m=state(g);v=json_value(m,max_values=1)
    assert v['kind']=='ModelState' and v['parameter_count']==1 and 'rng' not in json.dumps(v)
    assert 'weights' not in v and 'payload' not in v and v['resumable']
    out=json_value(infer_checkpoint(m,g),max_values=2)
    assert out['values']['shape']==[3,2] and len(out['values']['sample'])==2


def test_system_query_endpoint_runs_same_model_routes():
    from fastapi.testclient import TestClient
    from system.server.app import app
    from system.state import sources
    g=rex();sources.register('model_test',g,policy=SourcePolicy.allow('read','train'))
    try:
        with TestClient(app) as client:
            response=client.post('/api/query',json={'query':'FROM $model_test LET m = MODEL_INIT("coparticipation",$cfg) LET b = MODEL_BATCH(m,$labels,$mask) RETURN MODEL_TRAIN(m,b,2)','params':{'cfg':{'n_classes':2},'labels':[0,1,0],'mask':[True,True,True]}})
            assert response.status_code==200,response.text
            val=response.json()['values'][0];assert val['kind']=='ModelState' and val['step']==2
    finally:sources.remove('model_test')
