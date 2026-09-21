from dataclasses import replace
from fractions import Fraction as Q
import pytest

from rcql import (Program, ProgramStep, ProgramInput, ProgramAssembly, ProgramFamily,
                  ProgramTransformation, Executor, SourcePolicy, bind, parse)
from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSheaf
from rexgraph.section_calculus import SectionSystem, UnderdeterminedSectionError, InconsistentSectionError
from rcql.program_codec import loads,dumps


def fixture(rex=None):
    r=rex or RexGraph.from_cells([3,[[0,1],[1,2]]])
    system=SectionSystem.from_sheaf(ExactSheaf(r,grade=1))
    cells=system.recipe.cells
    first=Program('first',(ProgramStep('step',parse('FROM $r RETURN SUM([$x,$a])')),),
                  (ProgramInput('x','ExactRational'),ProgramInput('a','ExactRational')))
    second=Program('second',(ProgramStep('step',parse('FROM $r RETURN SUM([$value,$b])')),),
                   (ProgramInput('value','ExactRational'),ProgramInput('b','ExactRational')))
    assembly=ProgramAssembly.create('glued',((cells[0],first),(cells[1],second)),
        ((cells[0],'a','0'),(cells[1],'b','0')), ((cells[0],'x','x'),),
        ((cells[1],'value',cells[0],'0'),))
    sources={'r':bind('r',r,SourcePolicy.allow('*'))}
    return r,system,assembly,sources


def completed():
    r,s,a,b=fixture()
    family=a.complete(s,*s.pins({(s.recipe.cells[0],'0'):Q(2,7)}))
    return r,s,a,b,family


def test_unique_gluing_executes_exactly():
    r,s,a,b,f=completed()
    assert f.family.dimension==0 and f.observation().determined
    p=f.compile(b,{'x':Q(3,7)})
    assert p.execute(Executor(sources={'r':r},params={'x':Q(3,7)})).values==(Q(1),)
    assert len(p.dependencies)==1
    assert a.template().inputs[0].name=='x'
    assert a.template().coefficient_digest!=p.coefficient_digest


def test_free_family_not_selected():
    r,s,a,b=fixture();f=a.complete(s)
    assert f.family.dimension==1 and not f.observation().determined
    with pytest.raises(UnderdeterminedSectionError):
        f.compile(b,{'x':Q(1)})


def test_free_latent_stalks_with_determined_executable_observation():
    r,_,a,b=fixture()
    sheaf=ExactSheaf(r,grade=1,stalk_dims=(2,2),mediator_dims=(1,1,1))
    for cell in range(2):
        sheaf.restrict(cell,[[1,0]])
    system=SectionSystem.from_sheaf(sheaf)
    f=a.complete(system,*system.pins({(system.recipe.cells[0],'0'):Q(2,7)}))
    assert f.family.dimension==2 and f.observation().determined
    p=f.compile(b,{'x':Q(1)})
    assert p.execute(Executor(sources={'r':r},params={'x':Q(1)})).values==(Q(11,7),)


def test_shared_external_ports_and_capture_collision():
    r,s,a,b=fixture();data=a.declaration()
    data['links']=()
    data['inputs']=(*data['inputs'],(s.recipe.cells[1],'value','x'))
    shared=ProgramAssembly(dumps(data))
    assert sum(v.name=='x' for v in shared.template().inputs)==1
    data['inputs']=((s.recipe.cells[0],'x','p1_b'),(s.recipe.cells[1],'value','p1_b'))
    with pytest.raises(ValueError,match='colli'):
        ProgramAssembly(dumps(data))


def test_family_tamper_and_foreign_recipe_rejected():
    r,s,a,b,f=completed()
    other=replace(s.recipe,name='foreign')
    with pytest.raises(ValueError):
        replace(f,recipe=other)
    record=f.to_record();record.attach_metadata(1,0,'rcql_program_family','wrong')
    with pytest.raises(ValueError,match='certificate'):
        ProgramFamily.from_record(record)


def test_dependency_permission_intersection():
    from rcql import BoundSource
    r,s,a,b,f=completed();p=f.compile(b,{'x':Q(1)})
    other=RexGraph.from_cells([2,[[0,1]]])
    # The dependency may be supplied separately from the calculation source.
    e=Executor(sources={'r':other,'proof':BoundSource(r,SourcePolicy.allow('read'))},params={'x':Q(1)})
    with pytest.raises(PermissionError,match='intersection'):
        p.execute(e)


def test_exact_contradiction_is_retained():
    r,s,a,b=fixture()
    with pytest.raises(InconsistentSectionError) as error:
        a.complete(s,*s.pins({(s.recipe.cells[0],'0'):1,(s.recipe.cells[1],'0'):2}))
    assert any(error.value.contradiction.flat)
    assert error.value.witness.values.size


def test_no_target_execution_during_compile(monkeypatch):
    from rcql.operators import _REGISTRY
    r,s,a,b,f=completed();op=_REGISTRY['SUM'];calls=[]
    def count(*args):
        calls.append(1);return op.fn(*args)
    monkeypatch.setitem(_REGISTRY,'SUM',replace(op,fn=count))
    p=f.compile(b,{'x':Q(1)})
    assert not calls
    p.execute(Executor(sources={'r':r},params={'x':Q(1)}))
    assert len(calls)==2


def test_link_contract_not_erased():
    r,s,a,b=fixture();data=a.declaration()
    cell,raw=data['fragments'][1];p=Program.from_bytes(raw)
    p=replace(p,inputs=(replace(p.inputs[0],kind='ExactInteger'),p.inputs[1]))
    data['fragments']=(data['fragments'][0],(cell,p.to_bytes()))
    a=ProgramAssembly(dumps(data))
    f=a.complete(s,*s.pins({(s.recipe.cells[0],'0'):Q(2,7)}))
    with pytest.raises(TypeError,match='declared type'):
        f.compile(b,{'x':Q(1,7)})


def test_compiled_dependency_not_erased():
    r,s,a,b,f=completed();p=f.compile(b,{'x':Q(1)})
    other=RexGraph.from_cells([3,[[0,2],[2,1]]])
    with pytest.raises(ValueError,match='dependency'):
        p.execute(Executor(sources={'r':other},params={'x':Q(1)}))
    restored=Program.from_record(p.to_record())
    assert all(a.matches(b) for a,b in zip(restored.dependencies,p.dependencies,strict=True))


@pytest.mark.parametrize('rule,args',[('input',['x','y']),('step',['f0_step','start']),('specialize',{'x':Q(1)})])
def test_schema_three_transforms(rule,args):
    r,s,a,b,f=completed();p=f.compile(b,{'x':Q(1)})
    assert loads(p.to_bytes())['version']==3
    t=ProgramTransformation.program(p,rule,args)
    parameters={'y':Q(1)} if rule=='input' else {} if rule=='specialize' else {'x':Q(1)}
    c=t.compile_program(p,b,parameters)
    assert c.execute(Executor(sources={'r':r},params=parameters)).values==(Q(11,7),)


@pytest.mark.parametrize('format',['rcbd','safetensors','rcdb'])
def test_portable_family_and_compilation(format,tmp_path):
    r,s,a,b,f=completed();record=f.to_record()
    if format=='rcbd':
        from rexgraph.io.bundle import save_rcbd,load_rcbd
        path=tmp_path/'family.rcbd';save_rcbd(path,record);record=load_rcbd(path)
    elif format=='safetensors':
        from rexgraph.io.safetensors_bridge import rex_to_safetensors,safetensors_to_rex
        path=tmp_path/'family.safetensors';rex_to_safetensors(record,path);record=safetensors_to_rex(path)
    else:
        from rcdb import open_store
        uri='file://'+str(tmp_path/'store');db=open_store(uri)
        db.commit_mutation('family',record,expected_version=0,analytics=False);db.close()
        db=open_store(uri);record=db.read_record('family',version=1).value;assert db.verify_commits('family');db.close()
    detached=ProgramFamily.from_record(record)
    assert detached.coefficient_digest==f.coefficient_digest
    e=Executor(sources={'r':r},params={'record':record,'sources':b,'params':{'x':Q(1)}})
    p=e.execute(parse('FROM $r LET f=PROGRAM_FAMILY_READ($record) RETURN PROGRAM_FAMILY_COMPILE(f,$sources,$params)')).values[0]
    assert p.execute(Executor(sources={'r':r},params={'x':Q(1)})).values==(Q(11,7),)


@pytest.mark.parametrize('field,value',[('links',()),('coefficients',()),('inputs',( ('0','x','r'),))])
def test_incomplete_or_colliding_ports_refused(field,value):
    r,s,a,b=fixture();data=a.declaration();data[field]=value
    with pytest.raises((ValueError,TypeError)):
        ProgramAssembly(dumps(data))


def case(name,rex):
    r,s,a,b=fixture(rex)
    f=a.complete(s,*s.pins({(c,'0'):Q(2,7) for c in s.recipe.cells}))
    data=a.declaration()
    return {'PROGRAM_ASSEMBLY':('glued',tuple((c,Program.from_bytes(p)) for c,p in data['fragments']),
                data['coefficients'],data['inputs'],data['links'],data['outputs']),
        'PROGRAM_GLUE':(a,s,*s.pins({(c,'0'):Q(2,7) for c in s.recipe.cells})),
        'PROGRAM_FAMILY_OBSERVE':(f,), 'PROGRAM_FAMILY_COMPILE':(f,b,{'x':Q(1)}),
        'PROGRAM_FAMILY_RECORD':(f,), 'PROGRAM_FAMILY_READ':(f.to_record(),),
        'PROGRAM_ASSEMBLY_RECORD':(a,), 'PROGRAM_ASSEMBLY_READ':(a.to_record(),)}[name]


@pytest.mark.parametrize('name',['PROGRAM_ASSEMBLY','PROGRAM_GLUE','PROGRAM_FAMILY_OBSERVE',
    'PROGRAM_FAMILY_COMPILE','PROGRAM_FAMILY_RECORD','PROGRAM_FAMILY_READ',
    'PROGRAM_ASSEMBLY_RECORD','PROGRAM_ASSEMBLY_READ'])
def test_rcql_assembly_operators(name):
    from rcql import call,query,source
    from rcql.operators import get_operator
    r,_,_,_=fixture();args=case(name,r)
    direct=get_operator(name).fn(r,*args)
    value=Executor(sources={'r':r}).execute(query(source('r'),call(name,*args))).values[0]
    if hasattr(direct,'coefficient_digest'):
        assert direct.coefficient_digest==value.coefficient_digest
    else:
        from rexgraph.io.catalog import object_digest
        assert object_digest(direct)==object_digest(value)


def test_historical_compilation_dependency_stays_pinned():
    from rcdb import open_store
    from rcql import SnapshotContext,SourceSelection
    from rexgraph.tensor_field import FieldSource
    r,s,a,b=fixture();db=open_store('memory://')
    try:
        db.commit_mutation('proof',r,expected_version=0,tx_time=10,analytics=False)
        context=SnapshotContext.select(db,(SourceSelection('r','proof'),),cutoff=20)
        selected=context.sources['r']
        system=SectionSystem.from_sheaf(ExactSheaf(selected.value,grade=1),
            source=FieldSource(selected.value,'proof',1))
        f=a.complete(system,*system.pins({(system.recipe.cells[0],'0'):Q(2,7)}))
        e=Executor(sources=context.sources,evidence=context)
        binding=e._planning_binding(parse('FROM $r RETURN 1').source,selected)
        e.params={'f':f,'sources':{'r':binding},'params':{'x':Q(1)}}
        p=e.execute(parse('FROM $r RETURN PROGRAM_FAMILY_COMPILE($f,$sources,$params)')).values[0]
        db.commit_mutation('proof',r,expected_version=1,tx_time=30,analytics=False)
        later=SnapshotContext.select(db,(SourceSelection('r','proof',version=2),))
        with pytest.raises(ValueError,match='dependency'):
            p.execute(Executor(sources=later.sources,params={'x':Q(1)}))
        assert p.execute(Executor(sources=context.sources,evidence=context,params={'x':Q(1)})).values==(Q(11,7),)
    finally:
        db.close()
