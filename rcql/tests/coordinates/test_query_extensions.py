from fractions import Fraction as Q
from dataclasses import replace
import numpy as np
import pytest
from rcql import Executor,parse
from rexgraph.coordinate_map import CoordinateMap,CoordinateMetric
from rexgraph.temporal_calculus import TemporalWord,TemporalOperation,TemporalMetrics
from rexgraph.type_accession import CoordinateSpace,CoordinateField,TypeAccession
from rexgraph.chain_map import CoordinateComplex,GradedMap
from rexgraph.cochain import Chain,Cochain
from rexgraph.graph import RexGraph
from rexgraph.ranking_response import pagerank_solve
from rexgraph.markov import ParticipationWalk


def setup():
    r=RexGraph.from_cells([2,[[0,1]]]);s=CoordinateSpace('local',('a',))
    a=CoordinateMap(s,s,((0,0,1),));b=CoordinateMap(s,s,((0,0,2),))
    w=TemporalWord((a,b),(b,a),(a,a,a))
    op=TemporalOperation(a,b,a,a)
    x=CoordinateField(r,1,s,np.array([Q(1)],object),'chain');y=CoordinateField(r,1,s,np.array([Q(2)],object),'chain')
    return r,dict(w=w,op=op,a=a,x=x,y=y,k=w.moment_kernel(),m=CoordinateMetric.diagonal(s,[3]))


def test_query_temporal_fields_kernel_and_members():
    r,p=setup();e=Executor(sources={'r':r},params=p)
    out=e.execute(parse('FROM $r RETURN WORD_DELTA($w,$x).moments.total, KERNEL_MOMENTS($k,$x).total, OPERATION_DELTA($op,$x,$y,$m).moments.total, INJECTION_DELTA($op,$x,$y).names'))
    assert out.values==(0,0,27,('injection','amplitude'))
    assert all(x.value=='rational' for x in out.exactness[:3])


def test_coordinate_apply_preserves_grade_variance_and_changes_only_named_coords():
    r,p=setup();v=Executor(sources={'r':r},params=p).execute(parse('FROM $r RETURN COORDINATE_APPLY($a,$x)')).values[0]
    assert v.grade==1 and v.variance=='chain' and v.source is r and v.space==p['x'].space
    assert list(v.values)==[1]


def test_explain_no_temporal_evaluation_and_plan_serializable(monkeypatch):
    r,p=setup();e=Executor(sources={'r':r},params=p)
    monkeypatch.setattr(TemporalWord,'delta',lambda *a,**k:pytest.fail('evaluated during EXPLAIN'))
    out=e.execute(parse('EXPLAIN FROM $r RETURN WORD_DELTA($w,$x), KERNEL_MOMENTS($k,$x)'))
    assert out.execution==()


@pytest.mark.parametrize('change',['space','source','float','variance','block'])
def test_coordinate_mismatches_refuse_before_dispatch(change,monkeypatch):
    import rcql.executor
    r,p=setup()
    if change=='space':p['x']=replace(p['x'],space=CoordinateSpace('different',('a',)))
    elif change=='source':p['x']=replace(p['x'],source=RexGraph.from_cells([2,[[0,1]]]))
    elif change=='float':p['x']=replace(p['x'],values=np.array([0.0]))
    elif change=='variance':p['y']=replace(p['y'],variance='cochain')
    else:p['y']=replace(p['y'],values=np.array([[Q(2)]],object))
    monkeypatch.setattr(rcql.executor,'get_operator',lambda *a,**k:pytest.fail('dispatch before invalid input refusal'))
    with pytest.raises((ValueError,TypeError)):
        Executor(sources={'r':r},params=p).execute(parse('FROM $r RETURN OPERATION_DELTA($op,$x,$y)'))


def test_query_optional_k_matches_core():
    r,p=setup();c=CoordinateComplex.from_rex(r);j=GradedMap(c,c,tuple(tuple((i,i,1) for i in range(n)) for n in c.sizes))
    a=TypeAccession(r,1,'old',((0,0,1),),coordinates=CoordinateSpace('old',('x',)))
    b=TypeAccession(r,1,'new',((0,0,1),(1,0,1)),coordinates=CoordinateSpace('new',('x','y')))
    k=CoordinateMap(a.coordinates,b.coordinates,((0,0,1),(1,0,1)))
    out=Executor(sources={'r':r},params={'a':a,'b':b,'j':j,'k':k}).execute(parse('FROM $r RETURN ACCESSION_DELTA($a,$b,$j,$k).entries'))
    assert out.values==((),)


def test_query_declared_temporal_metrics():
    r=RexGraph.from_simplicial([0,1,0],[1,2,2],[[0,1,2]]);c=CoordinateComplex.from_rex(r)
    j=GradedMap(c,c,tuple(tuple((i,i,1) for i in range(n)) for n in c.sizes))
    old=tuple(CoordinateMetric.identity(s) for s in c.spaces)
    new=list(old);new[1]=CoordinateMetric.diagonal(c.spaces[1],[2,2,2]);m=TemporalMetrics(c,c,old,tuple(new))
    f=Chain(1,np.array([1,1,-1],object),source=r)
    e=Executor(sources={'r':r},params={'f':f,'j':j,'m':m})
    out=e.execute(parse('FROM $r RETURN FIELD_DELTA($f,$j,$m).moment, FIELD_DELTA_MOMENT($f,$j,$m)'))
    assert out.values==(9,9)


def test_exact_pagerank_query_dispatch_and_old_numeric_policy():
    r=RexGraph.from_cells([3,[[0,1],[1,2]]]);f=Cochain(0,np.array([1,0,0],object),source=r)
    e=Executor(sources={'r':r},params={'s':f,'a':Q(1,2)})
    out=e.execute(parse('FROM $r LET t=PARTICIPATION_WALK() RETURN PAGERANK_SOLVE(t,$a,$s), PAGERANK_ITERATION(t,0.5,$s)'))
    exact,numeric=out.values
    assert exact.values.tolist()==pagerank_solve(ParticipationWalk(r),Q(1,2),[1,0,0]).tolist()
    assert np.allclose(np.array(exact.values,float),numeric.values)
    assert out.exactness[0].value=='rational' and out.exactness[1].value=='approximate'


def test_exact_pagerank_explain_does_not_solve(monkeypatch):
    import rexgraph.ranking_response as rr
    r,_=setup();monkeypatch.setattr(rr,'pagerank_solve',lambda *a,**k:pytest.fail('solve during explain'))
    assert Executor(sources={'r':r}).execute(parse('EXPLAIN FROM $r RETURN PAGERANK_SOLVE(PARTICIPATION_WALK())')).execution==()


def test_float_damping_refused_by_exact_query():
    r,_=setup()
    with pytest.raises((ValueError,TypeError)):
        Executor(sources={'r':r}).execute(parse('FROM $r RETURN PAGERANK_SOLVE(PARTICIPATION_WALK(),0.5)'))


def test_parser_builder_and_repeated_kernel_memoization(monkeypatch):
    from rcql import query,source,call
    from rexgraph.temporal_calculus import MomentKernel
    r,p=setup();calls=[];original=MomentKernel.evaluate
    def evaluate(self,x):
        calls.append(1);return original(self,x)
    monkeypatch.setattr(MomentKernel,'evaluate',evaluate)
    e=Executor(sources={'r':r},params=p)
    result=e.execute(parse('FROM $r RETURN KERNEL_MOMENTS($k,$x),KERNEL_MOMENTS($k,$x)'))
    assert calls==[1]
    assert result.values[0]==result.values[1]
    built=Executor(sources={'r':r}).execute(query(source('r'),call('KERNEL_MOMENTS',p['k'],p['x'])))
    assert built.values[0]==result.values[0]


def test_coordinate_binding_digest_changes_when_coefficients_change():
    r,p=setup();e=Executor(sources={'r':r},params=p)
    first=e.execute(parse('FROM $r RETURN COORDINATE_APPLY($a,$x)')).values[0]
    p['a']=replace(p['a'],entries=((0,0,3),))
    second=Executor(sources={'r':r},params=p).execute(parse('FROM $r RETURN COORDINATE_APPLY($a,$x)')).values[0]
    assert first.values.tolist()==[1] and second.values.tolist()==[3]
