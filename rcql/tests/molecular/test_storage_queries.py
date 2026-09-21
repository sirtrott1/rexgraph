from fractions import Fraction as Q
from dataclasses import replace
import numpy as np
import pytest

from agent.adapters.molecular_store import ingest_molecules, verify_molecular_sources
from rcdb import open_store
from rcdb.core import VersionConflictError
from rcql import Executor, parse, call, query, source, bind, SourcePolicy
from rcql.types import SourceRef
from rexgraph.molecular_field import MolecularView, conformation_field
from rexgraph.tensor_field import FieldSource
from rexgraph.tensor_moment import MomentSpan, CoordinatePairing
from rexgraph.coordinate_map import CoordinateMetric
from rexgraph.io.rex_state import to_state, from_state
from .helpers import make, declaration, SMILES

ATOMS=('map/1','map/2','map/3','map/4')


def selected_binding(snapshot):
    return bind('selected',snapshot.value,SourcePolicy.allow("*"),source_ref=SourceRef(
        name='selected',record_id=snapshot.record.id,record_version=snapshot.record.version,state_digest=snapshot.state_digest))


def prepared(name, tmp_path):
    p,b,g=make(tmp_path)
    v=MolecularView.from_source(g,'chain');f=v.conformation('plus');direction=f.with_values(np.ones((4,3),dtype=object))
    y=conformation_field(f,'oriented_volume',(ATOMS,))
    destination=bind('new',g,SourcePolicy.allow("*"))
    values={
        'MOLECULAR_INFO':('chain',),
        'MOLECULAR_FIELD':('chain','bond_presence',True),
        'MOLECULAR_CONFORMER':('chain','plus'),
        'CONFORMATION_FIELD':(f,'oriented_volume',(ATOMS,)),
        'CONFORMATION_DIRECTION':(f,direction,'oriented_volume',(ATOMS,),'translation','angstrom'),
        'MOLECULAR_DELTA':('chain',destination,'chain'),
        'PROCESS_COMPARE':([y,y],[y,y],[0,1],'elapsed','s'),
        'FACTOR_CONTRAST':(y,y.with_values([2]),1,'declared_factor','1'),
    }
    return g,values[name]


@pytest.mark.parametrize('name', ['MOLECULAR_INFO','MOLECULAR_FIELD','MOLECULAR_CONFORMER','CONFORMATION_FIELD',
    'CONFORMATION_DIRECTION','MOLECULAR_DELTA','PROCESS_COMPARE','FACTOR_CONTRAST'])
def test_direct_parser_builder_and_plan(name,tmp_path):
    from rcql.operators import get_operator
    from rcql.executor import value_exactness
    from rcql.signatures import lookup
    g,args=prepared(name,tmp_path)
    direct=get_operator(name).fn(g,*args)
    ex=Executor(sources={'g':g},params={f'a{i}':v for i,v in enumerate(args)})
    ast=parse('FROM $g RETURN '+name+'('+','.join('$a'+str(i) for i in range(len(args)))+')')
    parsed=ex.execute(ast)
    built=ex.execute(query(source('g'),call(name,*args)))
    assert parsed.exactness==built.exactness==(value_exactness(direct),)
    if hasattr(direct,'coefficient_digest'):
        assert parsed.values[0].coefficient_digest==direct.coefficient_digest==built.values[0].coefficient_digest
    elif hasattr(direct,'declaration_digest'):
        assert parsed.values[0].declaration_digest==direct.declaration_digest==built.values[0].declaration_digest
    else:assert parsed.values[0]==direct==built.values[0]
    plan=ex.execute(replace(ast,explain=True)).values[0]
    assert plan['returns'][0]['requires']==sorted(lookup(name).requires)
    assert plan['returns'][0]['result']['exactness']==parsed.exactness[0].value


def test_explain_does_not_evaluate_geometry(tmp_path,monkeypatch):
    import rexgraph.molecular_field as mod
    g,args=prepared('CONFORMATION_FIELD',tmp_path)
    def forbidden(*a,**kw):raise AssertionError('geometry executed during explanation')
    monkeypatch.setattr(mod,'conformation_field',forbidden)
    Executor(sources={'g':g}).execute(query(source('g'),call('CONFORMATION_FIELD',*args),explain=True))


def test_bad_named_input_and_denied_destination(tmp_path):
    p,b,g=make(tmp_path);other=from_state(to_state(g))
    f=MolecularView.from_source(other,'chain').conformation('plus')
    with pytest.raises(ValueError,match='contributor'):
        Executor(sources={'g':g}).execute(query(source('g'),call('CONFORMATION_FIELD',f,'oriented_volume',(ATOMS,))))
    denied=bind('denied',other,SourcePolicy.allow())
    ex=Executor(sources={'g':g})
    for explain in (True,False):
        with pytest.raises(PermissionError):ex.execute(query(source('g'),call('MOLECULAR_DELTA','chain',denied,'chain'),explain=explain))


@pytest.mark.parametrize('backend',['memory://','file://','rex://','sql://'])
def test_store_versions_conditions_and_source_history(tmp_path,backend):
    pytest.importorskip("rdkit")
    if backend=='sql://':
        # The SQLStore reader uses a SQLAlchemy URL after its scheme.
        url='sqlite:///'+str(tmp_path/'molecules.sqlite')
    else:url=backend+(str(tmp_path/'db') if backend!='memory://' else '')
    db=open_store(url)
    p=tmp_path/'molecule.smi';p.write_text(SMILES+' chain\n')
    try:
        conditions={'chain':{'temperature':{'value':'298.15','unit':'K','origin':'synthetic','source':'fixture'},
                              'solvent':{'label':'declared solvent A','origin':'synthetic','source':'fixture'}}}
        conf={'chain':{'frame':declaration(0)}}
        rid,_=ingest_molecules(db,p,document_id='molecules',expected_version=0,tx_time=20,conformers=conf,conditions=conditions)
        old=db.read_record(rid,as_of=25)
        _,same=ingest_molecules(db,p,document_id='molecules',expected_version=1,tx_time=25,conformers=conf,conditions=conditions)
        assert same is None
        conditions['chain']['temperature']['value']='300.15'
        conf['chain']['frame']=declaration(1)
        _,meta=ingest_molecules(db,p,document_id='molecules',expected_version=1,tx_time=30,conformers=conf,conditions=conditions)
        assert meta['version']==2
        assert db.read_record(rid,as_of=25).state_digest==old.state_digest
        with pytest.raises(VersionConflictError):ingest_molecules(db,p,document_id='molecules',expected_version=1,conditions=conditions,conformers=conf)
        ex=Executor(sources={'db':db})
        f=ex.execute(parse('FROM RCDB_VERSION($db,"molecules",1) RETURN MOLECULAR_CONFORMER("chain","frame")')).values[0]
        info=ex.execute(parse('FROM RCDB_VERSION($db,"molecules",2) RETURN MOLECULAR_INFO("chain")')).values[0]
        assert f.source.record_id=='molecules' and f.source.version==1
        assert Q(*map(int,info['conditions']['temperature']['value']))==Q(6003,20)
        assert db.verify_commits(rid)
    finally:db.close()
    if backend!='memory://':
        db=open_store(url)
        try:
            f=Executor(sources={'db':db}).execute(parse('FROM RCDB_VERSION($db,"molecules",2) RETURN MOLECULAR_CONFORMER("chain","frame")')).values[0]
            assert f.values[3,2]==1
        finally:db.close()


@pytest.mark.parametrize('extension',['.rcbd','.safetensors'])
def test_formats_retain_conformation_and_located_moment(tmp_path,extension):
    from rexgraph.io.bundle import save_rcbd,load_rcbd
    from rexgraph.io.safetensors_bridge import rex_to_safetensors,safetensors_to_rex
    from rexgraph.graph import RexGraph
    _,_,g=make(tmp_path);v=MolecularView.from_source(g,'chain');f=conformation_field(v.conformation('plus'),'oriented_volume',(ATOMS,))
    moment=MomentSpan(f,f,CoordinatePairing.metric(CoordinateMetric.identity(f.space)))
    record=RexGraph.from_hypergraph([0,1],[0]);record.attach_metadata(1,0,'molecular_source',g);record.attach_metadata(1,0,'moment',moment)
    path=tmp_path/('result'+extension)
    if extension=='.rcbd':save_rcbd(path,record);restored=load_rcbd(path)
    else:rex_to_safetensors(record,path);restored=safetensors_to_rex(path)
    source2=restored.get_metadata(1,0,'molecular_source');m2=restored.get_metadata(1,0,'moment')
    assert MolecularView.from_source(source2,'chain').conformation('plus').values[3,2]==1
    assert m2.support().values.tolist()==[Q(1)]


def test_external_source_integrity(tmp_path):
    p,b,g=make(tmp_path);assert verify_molecular_sources(g,tmp_path)==(str(p),)
    p.write_text('CCO changed\n')
    with pytest.raises(ValueError,match='changed'):verify_molecular_sources(g,tmp_path)


def test_native_model_retains_molecular_source(tmp_path):
    from rexgraph.io.model_record import save_model,load_model,read_model_record
    g,args=prepared('MOLECULAR_FIELD',tmp_path)
    ex=Executor(sources={'g':g})
    result=ex.execute(parse('FROM $g LET m=MODEL_INIT() LET x=MOLECULAR_FIELD("chain","bond_presence",true) RETURN m, MODEL_INFER(m,x), MODEL_RECORD(m)'))
    state,prediction,record=result.values
    assert prediction.arithmetic=='rational'
    path=tmp_path/'molecule_model.safetensors';save_model(path,record)
    model,restored=read_model_record(load_model(path))
    assert model.coefficient_digest==state.coefficient_digest
    assert MolecularView.from_source(restored,'chain').conformation('minus').values[3,2]==-1


def test_parser_not_imported_during_native_inference(tmp_path,monkeypatch):
    _,_,g=make(tmp_path)
    import sys
    monkeypatch.setitem(sys.modules,'rdkit',None)
    info=Executor(sources={'g':g}).execute(parse('FROM $g RETURN MOLECULAR_INFO("chain")')).values[0]
    assert info['element_counts']==((1,0,10),(6,0,4))


def test_molecular_cochain_training_preserves_exact_source(tmp_path):
    pytest.importorskip("torch")
    from rexgraph.io.model_record import save_model, load_model, read_model_record
    _,_,g=make(tmp_path)
    before=FieldSource(g).state_digest
    params={'targets':np.array([0,9,1]), 'mask':np.array([True,False,True]), 'config':{'n_classes':2}}
    result=Executor(sources={'g':g},params=params).execute(parse('''FROM $g
        LET old=MODEL_INIT("coparticipation",$config)
        LET batch=MODEL_BATCH(old,$targets,$mask)
        LET new=MODEL_TRAIN(old,batch,2)
        RETURN old,new,MODEL_INFER(new),MODEL_RECORD(new)'''))
    old,new,prediction,record=result.values
    assert old.step==0 and new.step==2 and prediction.arithmetic=='approximate'
    assert FieldSource(g).state_digest==before
    path=tmp_path/'learned_molecular_model.rcbd';save_model(path,record)
    saved,restored=read_model_record(load_model(path))
    assert saved.coefficient_digest==new.coefficient_digest
    assert MolecularView.from_source(restored,'chain').conformation('plus').values[3,2]==1
    assert new.space==old.space


