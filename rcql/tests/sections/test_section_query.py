from fractions import Fraction as F
from dataclasses import replace
import pytest
from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSheaf
from rexgraph.section_calculus import SectionFamily
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.type_accession import CoordinateSpace
from rcql import Executor, parse, query, source, call, PhraseSheaf, PhraseStalk, PhraseCorrespondence, bind, SourcePolicy, SourceRef


def local():
    r=RexGraph.from_hypergraph([0,3],[0,1,2]);s=ExactSheaf(r,grade=0)
    s.restrict(0,[[2]])
    c=s.section_system();o,b=c.pins({('0','0'):3})
    return r,s,c,o,b


def phrase():
    states=[RexGraph.from_hypergraph([0,2],[0,1]) for _ in range(2)]
    bindings=tuple(bind(n,v,SourcePolicy.allow('read'),source_ref=SourceRef(n,record_id='document',record_version=i+1))
                   for i,(n,v) in enumerate(zip(('old','new'),states,strict=False)))
    sh=PhraseSheaf(tuple(PhraseStalk(n,b) for n,b in zip(('old','new'),bindings,strict=False)),
                  (PhraseCorrespondence('revisions',('old','new')),),
                  stalk_dims={'old':2,'new':2},correspondence_dims={'revisions':1})
    sh.restrict('old','revisions',[[1,0]]);sh.restrict('new','revisions',[[1,0]])
    return sh,bindings


def test_query_completion_and_members():
    r,s,c,o,b=local()
    e=Executor(sources={'g':r},params={'s':s,'c':c,'o':o,'b':b,'out':c.selection('1')})
    result=e.execute(parse('FROM $g LET f=SECTION_COMPLETE($c,$o,$b) LET i=SECTION_OBSERVE(f,$out) RETURN f.dimension, i.determined, SECTION_VALUE(i)'))
    assert result.values[:2]==(0,True)
    assert result.values[2].values.tolist()==[6]
    assert result.exactness[2].value=='rational'


def test_builder_matches_parsed():
    r,s,c,o,b=local()
    text=Executor(sources={'g':r},params={'c':c,'o':o,'b':b}).execute(parse('FROM $g RETURN SECTION_COMPLETE($c,$o,$b)')).values[0]
    built=Executor(sources={'g':r}).execute(query(source('g'),call('SECTION_COMPLETE',c,o,b))).values[0]
    assert text.coefficient_digest==built.coefficient_digest


def test_phrase_native_sources_and_invariant_prediction():
    sh,bindings=phrase();c=sh.section_system()
    o,b=c.pins({('old','0'):F(2,7)})
    target=CoordinateSpace('event_output',('quantity',))
    out=CoordinateMap(c.space,target,((0,2,1),))
    e=Executor(params={'s':sh,'c':c,'o':o,'b':b,'out':out})
    result=e.execute(parse('FROM PHRASE($s) LET f=SECTION_COMPLETE($c,$o,$b) RETURN f.dimension, SECTION_VALUE(SECTION_OBSERVE(f,$out))'))
    assert result.values[0]==2
    assert result.values[1].values.tolist()==[F(2,7)]
    assert [d.version for d in result.values[1].dependencies]==[1,2]
    assert c.recipe.stalk_sources[0].version==1
    assert c.recipe.stalk_sources[1].version==2
    assert sh.rex.nE==1


def test_phrase_does_not_infer_agreement_maps():
    sh,_=phrase();sh._R.clear()
    with pytest.raises(ValueError):Executor(params={'s':sh}).execute(parse('FROM PHRASE($s) RETURN SECTION_SYSTEM($s)'))


def test_explain_does_not_solve_or_glue(monkeypatch):
    r,s,c,o,b=local()
    monkeypatch.setattr(SectionFamily,'solve',lambda *a,**kw:pytest.fail('solve in explain'))
    monkeypatch.setattr(s,'glue',lambda:pytest.fail('glue in explain'))
    e=Executor(sources={'g':r},params={'c':c,'o':o,'b':b})
    result=e.execute(parse('EXPLAIN FROM $g RETURN SECTION_COMPLETE($c,$o,$b)'))
    assert result.values[0]['returns'][0]['result']['kind']=='SectionFamily'


def test_foreign_system_refused_before_solve(monkeypatch):
    r,s,c,o,b=local();other=RexGraph.from_hypergraph([0,3],[0,1,2])
    monkeypatch.setattr(SectionFamily,'solve',lambda *a,**kw:pytest.fail('foreign solve'))
    with pytest.raises(ValueError):Executor(sources={'g':other},params={'c':c,'o':o,'b':b}).execute(parse('FROM $g RETURN SECTION_COMPLETE($c,$o,$b)'))


def test_changed_maps_are_not_cached_between_queries():
    r,s,c,o,b=local();e=Executor(sources={'g':r},params={'s':s})
    text=parse('FROM $g RETURN SECTION_RECIPE(SECTION_SYSTEM($s))')
    before=e.execute(text).values[0].coefficient_digest
    s.restrict(0,[[3]])
    after=e.execute(text).values[0].coefficient_digest
    assert before!=after


def test_stored_recipe_rebinds_explicit_contributors():
    sh,bindings=phrase();c=sh.section_system();recipe=c.recipe.detached()
    e=Executor(params={'s':sh,'recipe':recipe,'refs':bindings})
    restored=e.execute(parse('FROM PHRASE($s) RETURN SECTION_RESTORE($recipe,$refs)')).values[0]
    assert restored.coefficient_digest==c.coefficient_digest
    o,b=restored.pins({('old','0'):9})
    f=Executor(params={'s':sh,'c':restored,'o':o,'b':b}).execute(parse('FROM PHRASE($s) RETURN SECTION_COMPLETE($c,$o,$b)')).values[0]
    assert f.particular.values.tolist()==[9,0,9,0]


@pytest.mark.parametrize('alteration',['missing','wrong_version','denied'])
def test_restoration_refuses_invalid_contributors(alteration):
    sh,bindings=phrase();recipe=sh.section_system().recipe.detached();refs=list(bindings)
    if alteration=='missing':refs=refs[:1]
    if alteration=='wrong_version':refs[1]=replace(refs[1],ref=replace(refs[1].ref,record_version=99))
    if alteration=='denied':refs[1]=bind('denied',refs[1].value,SourcePolicy.allow(),source_ref=refs[1].ref)
    with pytest.raises((ValueError,PermissionError)):
        Executor(params={'s':sh,'recipe':recipe,'refs':refs}).execute(parse('FROM PHRASE($s) RETURN SECTION_RESTORE($recipe,$refs)'))


def test_broader_phrase_binding_does_not_escalate_policy():
    sh,bindings=phrase();recipe=sh.section_system().recipe.detached()
    from rcql import BoundSource
    e=Executor(sources={'g':BoundSource(sh.rex,SourcePolicy.allow('*'))},params={'recipe':recipe,'refs':bindings})
    with pytest.raises(PermissionError):e.execute(parse('FROM $g RETURN SECTION_RESTORE($recipe,$refs)'))


def test_restored_family_query():
    sh,bindings=phrase();c=sh.section_system();o,b=c.pins({('old','0'):4});family=c.complete(o,b)
    from rexgraph.io.section_state import pack_section,unpack_section
    saved=unpack_section(pack_section(family))
    e=Executor(params={'s':sh,'family':saved,'refs':bindings})
    restored=e.execute(parse('FROM PHRASE($s) RETURN SECTION_RESTORE($family,$refs)')).values[0]
    assert restored.coefficient_digest==family.coefficient_digest


def test_temporal_residual_keeps_contributor_versions():
    sh,bindings=phrase();old=sh.section_system();x=old.field([1,0,1,0])
    sh2,bindings2=phrase();sh2.restrict('old','revisions',[[2,0]])
    new=sh2.section_system();y=new.field([1,0,2,0])
    J=CoordinateMap(old.space,new.space,tuple((i,i,1) for i in range(4)))
    K=CoordinateMap(old.residual_space,new.residual_space,((0,0,1),))
    result=Executor(params={'s':sh,'a':old,'b':new,'j':J,'k':K,'x':x,'y':y}).execute(parse('FROM PHRASE($s) RETURN SECTION_DELTA($a,$b,$j,$k,$x,$y)')).values[0]
    assert result.total().values.tolist()==[0]
    for f in result.fields:
        assert all(any(ref.matches(d) for d in f.dependencies) for ref in (*old.recipe.dependencies,*new.recipe.dependencies))


@pytest.mark.parametrize('member',['__class__','particular','directions','_query_contributors'])
def test_no_arbitrary_member_traversal(member):
    r,s,c,o,b=local();e=Executor(sources={'g':r},params={'c':c})
    with pytest.raises((TypeError,ValueError,SyntaxError)):
        e.execute(parse('FROM $g RETURN SECTION_COMPLETE($c).'+member))


@pytest.mark.parametrize('explain', [False, True])
def test_temporal_query_rejects_mismatched_correspondence(explain):
    r, s, c, o, b = local()
    x = c.field()
    wrong = CoordinateMap.identity(c.residual_space)
    output = CoordinateMap.identity(c.residual_space)
    e = Executor(sources={'g': r}, params={'a': c, 'b': c, 'j': wrong, 'k': output, 'x': x})
    prefix = 'EXPLAIN ' if explain else ''
    with pytest.raises(ValueError, match='correspondence'):
        e.execute(parse(prefix + 'FROM $g RETURN SECTION_DELTA($a,$b,$j,$k,$x,$x)'))
