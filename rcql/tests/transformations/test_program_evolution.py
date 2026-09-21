from fractions import Fraction as Q
import pytest
import numpy as np
from rcql import NameRelation, ProgramEvolution, Program, ProgramStep, ProgramInput, Executor, parse
from rcql.program_evolution import _tower
from rcql.program_codec import dumps,loads
from rexgraph.graph import RexGraph


def identity_pairs(old,new):
    a,b=_tower(old),_tower(new)
    return tuple(tuple(zip(x.keys,y.keys,strict=True)) for x,y in zip(a.spaces,b.spaces,strict=True))


def names():
    n=NameRelation.operator('SUM')
    old,new=n.topology(),n.rename('values','items').topology()
    return old,new


def test_explicit_rename_zero_defect_not_semantic_equivalence():
    old,new=names();e=ProgramEvolution.compare(old,new,identity_pairs(old,new))
    for k in range(1,len(_tower(old).spaces)):
        defect=e.boundary_change(k)
        assert not any(defect.apply([Q(1)]*defect.shape[1]))
    assert e.explain()['operations']
    assert 'program equivalence' in e.explain()['excluded']


def test_equal_shape_not_identity():
    old,new=names()
    matches=tuple(() for _ in _tower(old).spaces)
    report=ProgramEvolution.compare(old,new,matches).explain()
    for grade,space in zip(report['grades'],_tower(old).spaces,strict=True):
        assert not grade['matched'] and grade['removed']==space.keys


def test_capture_change_visible_despite_same_boundary():
    n=NameRelation.operator('SUM')
    old,new=n.bind('values',[Q(1)]).topology(),n.bind('values',[Q(2)]).topology()
    e=ProgramEvolution.compare(old,new,identity_pairs(old,new))
    assert any(row[0]=='origin' for row in e.explain()['declarations'])
    for k in range(1,len(_tower(old).spaces)):
        defect=e.boundary_change(k)
        assert not any(defect.apply([Q(1)]*defect.shape[1]))


def test_boundary_change_matches_independent_exact_products():
    old,new=names();pairs=list(identity_pairs(old,new));pairs[0]=pairs[0][1:]
    e=ProgramEvolution.compare(old,new,pairs)
    a,b=_tower(old),_tower(new)
    def dense(entries,shape):
        v=np.full(shape,Q(0),dtype=object)
        for i,j,c in entries:v[i,j]=c
        return v
    for k in range(1,len(a.spaces)):
        p=e.correspondence(k);q=e.correspondence(k-1)
        expected=dense(b.boundaries[k-1],(b.sizes[k-1],b.sizes[k]))@dense(p.entries,p.shape)-dense(q.entries,q.shape)@dense(a.boundaries[k-1],(a.sizes[k-1],a.sizes[k]))
        for column in range(a.sizes[k]):
            x=[Q(int(i==column)) for i in range(a.sizes[k])]
            assert tuple(e.boundary_change(k).apply(x))==tuple(expected[:,column])


@pytest.mark.parametrize('bad',['missing_grade','duplicate','unknown','reverse','version','tower'])
def test_invalid_correspondence_and_persistence_refused(bad):
    old,new=names();e=ProgramEvolution.compare(old,new,identity_pairs(old,new));data=loads(e.data)
    rows=list(data['matches'])
    if bad=='missing_grade':rows.pop()
    elif bad=='duplicate':rows[0]=(*rows[0],rows[0][0])
    elif bad=='unknown':rows[0]=(('unknown',rows[0][0][1]),)
    elif bad=='reverse':rows[1]=tuple((b,a) for a,b in rows[1]);rows[0]=(('bad','bad'),)
    elif bad=='version':data['version']=True
    else:data['new']['tower']['digest']='wrong'
    data['matches']=tuple(rows)
    with pytest.raises((ValueError,TypeError)):
        ProgramEvolution(dumps(data))


def test_bound_program_plan_versions():
    r=RexGraph.from_cells([2,[[0,1]]]);e=Executor(sources={'r':r},params={'x':Q(1)})
    p=Program('p',(ProgramStep('sum',parse('FROM $r RETURN SUM([$x,1])')),),(ProgramInput('x'),))
    old,new=p.topology(e),p.rename_step('sum','addition').topology(e)
    evolution=ProgramEvolution.compare(old,new,identity_pairs(old,new))
    assert evolution.dependencies and evolution.explain()['operations']
    assert ProgramEvolution.from_record(evolution.to_record()).data==evolution.data


def test_data_change_and_program_change_are_separate():
    r=RexGraph.from_cells([2,[[0,1]]]);s=RexGraph.from_cells([3,[[0,1,2]]])
    p=Program('p',(ProgramStep('read',parse('FROM $r RETURN SUM([ARITY(CELL(1,0)),$x])')),),
              (ProgramInput('x','ExactInteger'),))
    old,new=p.specialize({'x':1}),p.specialize({'x':2})
    outputs=tuple(program.execute(Executor(sources={'r':source})).values[0]
                  for source in (r,s) for program in (old,new))
    assert outputs==(3,4,4,5)
    e=Executor(sources={'r':r})
    a,b=old.topology(e),new.topology(e)
    comparison=ProgramEvolution.compare(a,b,identity_pairs(a,b))
    assert any(key=='program' for key,_,_ in comparison.explain()['declarations'])


@pytest.mark.parametrize('format',['rcbd','safetensors','rcdb'])
def test_evolution_portable(format,tmp_path):
    old,new=names();e=ProgramEvolution.compare(old,new,identity_pairs(old,new));record=e.to_record()
    if format=='rcbd':
        from rexgraph.io.bundle import save_rcbd,load_rcbd
        path=tmp_path/'change.rcbd';save_rcbd(path,record);record=load_rcbd(path)
    elif format=='safetensors':
        from rexgraph.io.safetensors_bridge import rex_to_safetensors,safetensors_to_rex
        path=tmp_path/'change.safetensors';rex_to_safetensors(record,path);record=safetensors_to_rex(path)
    else:
        from rcdb import open_store
        uri='file://'+str(tmp_path/'db');db=open_store(uri)
        db.commit_mutation('change',record,expected_version=0,analytics=False);db.close()
        db=open_store(uri);record=db.read_record('change',version=1).value;assert db.verify_commits('change');db.close()
    assert ProgramEvolution.from_record(record).data==e.data


def case(name,rex):
    old,new=names();matches=identity_pairs(old,new);e=ProgramEvolution.compare(old,new,matches)
    return {'PROGRAM_COMPARE':(old.to_record(),new.to_record(),matches),
        'PROGRAM_EVOLUTION_INFO':(e,), 'PROGRAM_BOUNDARY_CHANGE':(e,1),
        'PROGRAM_EVOLUTION_RECORD':(e,), 'PROGRAM_EVOLUTION_READ':(e.to_record(),)}[name]


@pytest.mark.parametrize('name',['PROGRAM_COMPARE','PROGRAM_EVOLUTION_INFO','PROGRAM_BOUNDARY_CHANGE',
                               'PROGRAM_EVOLUTION_RECORD','PROGRAM_EVOLUTION_READ'])
def test_rcql_names(name):
    from rcql import call,query,source
    r=RexGraph.from_cells([2,[[0,1]]])
    value=Executor(sources={'r':r}).execute(query(source('r'),call(name,*case(name,r)))).values[0]
    assert value is not None
