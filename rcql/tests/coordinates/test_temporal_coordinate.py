from fractions import Fraction as Q
from dataclasses import replace
import numpy as np
import pytest
import sympy as sp
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric, CoordinateWord
from rexgraph.temporal_calculus import TemporalOperation, TemporalWord, MomentKernel, TemporalDelta, TemporalMetrics, injection_delta
from rexgraph.type_accession import CoordinateSpace, TypeAccession
from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.cochain import Chain
from rexgraph.graph import RexGraph
from rexgraph.field_delta import field_delta
from rexgraph.accession_delta import accession_delta


def space(name,n):
    return CoordinateSpace(name,tuple(str(i) for i in range(n)))


def matrix(domain,codomain,rows):
    return CoordinateMap(domain,codomain,tuple((i,j,Q(v)) for i,row in enumerate(rows) for j,v in enumerate(row) if v))


def dense(a):
    x=sp.zeros(*a.shape)
    for i,j,v in a.entries: x[i,j]=sp.Rational(v.numerator,v.denominator)
    return x


def test_named_composition_and_transpose():
    a,b,c=space('a',2),space('b',3),space('c',1)
    f=matrix(a,b,[[1,2],[3,4],[5,6]])
    g=matrix(b,c,[[2,1,-1]])
    word=f.then(g)
    assert word.apply([2,3]).tolist()==[6]
    assert word.T.apply([5]).tolist()==[0,10]
    assert dense(f.compose(g))==dense(g)*dense(f)
    with pytest.raises(ValueError): f.then(CoordinateMap.identity(a))
    changed=replace(a,keys=('1','0'))
    with pytest.raises(ValueError): CoordinateWord((CoordinateMap.identity(a),CoordinateMap.identity(changed)))


@pytest.mark.parametrize('bad',[0.0,1.0,True,np.float64(0)])
def test_exact_ingress_even_cancelled_or_zero_actions(bad):
    s=space('a',1)
    with pytest.raises((TypeError,ValueError)): CoordinateMap(s,s,((0,0,bad),))
    zero=CoordinateMap(s,s,())
    with pytest.raises((TypeError,ValueError)): zero.apply([bad])
    with pytest.raises((TypeError,ValueError)): CoordinateMetric.diagonal(s,[bad])


@pytest.mark.parametrize('n',[0,1,2,4,7])
def test_positive_metric_solve_against_independent_rational_oracle(n):
    s=space('metric',n)
    u=sp.Matrix(n,n,lambda i,j:sp.Rational((i+2)*(j+3)%7-2,3))
    m=u.T*u+sp.eye(n)
    metric=CoordinateMetric(s,tuple((i,j,Q(m[i,j])) for i in range(n) for j in range(n)))
    x=np.array([Q(2*i+1,7) for i in range(n)],object)
    rhs=np.array([Q(v) for v in m*sp.Matrix(n,1,list(x))],object).reshape(n)
    assert metric.solve(rhs).tolist()==x.tolist()
    block=np.column_stack([x,2*x])
    assert np.array_equal(metric.solve(metric.apply(block)),block)


@pytest.mark.parametrize('rows', [[[0]], [[-1]], [[1,2],[2,1]], [[1,1],[0,1]], [[1,1],[1,1]]])
def test_invalid_metric_refusal(rows):
    s=space('m',len(rows))
    with pytest.raises(ValueError): CoordinateMetric(s,matrix(s,s,rows).entries)


def test_large_rationals_and_digest():
    s=space('large',1); v=Q(2**220+1,2**107+3)
    a=CoordinateMap(s,s,((0,0,v),))
    assert a.apply([2**60+1])[0]==v*(2**60+1)
    assert a.coefficient_digest != replace(a,entries=((0,0,v+1),)).coefficient_digest
    assert a.coefficient_digest != CoordinateMap(replace(s,name='other'),s,a.entries).coefficient_digest


def test_temporal_word_interactions_against_scalar_example():
    s=space('scalar',1)
    old=(matrix(s,s,[[1]]),matrix(s,s,[[2]]))
    new=(matrix(s,s,[[2]]),matrix(s,s,[[1]]))
    ids=(CoordinateMap.identity(s),)*3
    w=TemporalWord(old,new,ids,('first','second'))
    d=w.delta([1]); assert [x.tolist() for x in d.fields]==[[1],[-1]]
    assert d.values.tolist()==[0]
    assert d.moments().values.tolist()==[[1,-1],[-1,1]]
    assert w.moment_kernel().evaluate([1]).contract()==0
    assert d.moments().contract([1,-1])==4


@pytest.mark.parametrize('seed',range(8))
def test_rectangular_word_telescoping_with_independent_dense_oracle(seed):
    rng=np.random.default_rng(seed)
    oldspaces=[space(f'o{i}',n) for i,n in enumerate([2,3,2,4])]
    newspaces=[space(f'n{i}',n) for i,n in enumerate([3,2,4,1])]
    def rand(a,b): return matrix(a,b,[[Q(int(x),3) for x in row] for row in rng.integers(-3,4,(len(b.keys),len(a.keys)))])
    old=tuple(rand(a,b) for a,b in zip(oldspaces[:-1],oldspaces[1:],strict=False))
    new=tuple(rand(a,b) for a,b in zip(newspaces[:-1],newspaces[1:],strict=False))
    js=tuple(rand(a,b) for a,b in zip(oldspaces,newspaces,strict=False))
    w=TemporalWord(old,new,js)
    x=[Q(2,3),Q(-3,7)]
    expected=(dense(new[2])*dense(new[1])*dense(new[0])*dense(js[0])-dense(js[3])*dense(old[2])*dense(old[1])*dense(old[0]))*sp.Matrix(x)
    assert list(w.delta(x).values)==list(expected)
    block=np.column_stack([x,[2*v for v in x]])
    assert w.delta(block).moments().contract()==5*sum(v*v for v in expected)


def test_square_injection_and_independent_realization():
    u=space('request',1); e=space('edges',4)
    p=matrix(u,e,[[1],[1],[0],[0]])
    q=matrix(u,e,[[0],[0],[-1],[-1]])
    d=injection_delta(p,q,CoordinateMap.identity(u),CoordinateMap.identity(e),[2],[3])
    assert d.fields[0].tolist()==[-2,-2,-2,-2]
    assert d.fields[1].tolist()==[0,0,-1,-1]
    assert d.moments().values.tolist()==[[16,4],[4,2]]
    assert d.moments().contract()==26


def test_kernel_retains_inputs_rejects_wrong_target():
    a,b=space('a',2),space('b',1)
    f=matrix(a,b,[[1,2]]); g=matrix(a,b,[[3,4]])
    k=MomentKernel(('f','g'),(f,g),CoordinateMetric.diagonal(b,[Q(3,7)]))
    assert k.evaluate([1,2]).values.tolist()==[[Q(75,7),Q(165,7)],[Q(165,7),Q(363,7)]]
    with pytest.raises(ValueError): MomentKernel(('f','f'),(f,g),CoordinateMetric.identity(b))
    with pytest.raises(ValueError): MomentKernel(('f','g'),(f,g),CoordinateMetric.identity(space('wrong',1)))


def test_temporal_operation_mismatched_block_refusal():
    s=space('a',2); i=CoordinateMap.identity(s); op=TemporalOperation(i,i,i,i)
    with pytest.raises(ValueError): op.field_delta([1,2],[[1],[2]])
    with pytest.raises(ValueError): TemporalDelta(s,('x','x'),([1,2],[3,4]))


def native_fixture():
    r=RexGraph.from_simplicial([0,1,0],[1,2,2],[[0,1,2]])
    c=CoordinateComplex.from_rex(r)
    j=GradedMap(c,c,tuple(tuple((i,i,1) for i in range(n)) for n in c.sizes))
    return r,c,j


def test_metric_temporal_delta_exact_native_oracle():
    r,c,j=native_fixture()
    old=tuple(CoordinateMetric.identity(s) for s in c.spaces)
    new=list(old); new[1]=CoordinateMetric(c.spaces[1],matrix(c.spaces[1],c.spaces[1],[[2,1,0],[1,3,0],[0,0,4]]).entries)
    metrics=TemporalMetrics(c,c,old,tuple(new))
    x=Chain(1,np.array([Q(1,3),Q(2,5),Q(3,7)],object),source=r)
    result=field_delta(x,j,metrics=metrics)
    b2=sp.Matrix(3,1,[1,1,-1]); m=sp.Matrix([[2,1,0],[1,3,0],[0,0,4]])
    expected=(b2.T*(m-sp.eye(3))*sp.Matrix(x.values))[0]
    assert result['up']['values']==(expected,)
    assert result['down']['values']==(0,0,0)
    assert result['up_quadrance']==expected**2
    assert field_delta(x,j)['moment']==0
    assert result['metrics']=='declared'
    bad=replace(metrics,domain=CoordinateComplex(c.spaces,c.boundaries))
    with pytest.raises(ValueError): field_delta(x,j,metrics=bad)


def test_independent_output_correspondence_refinement():
    r,c,j=native_fixture()
    v=space('oldview',1); w=space('newview',2)
    a=TypeAccession(r,1,'old',((0,0,1),),coordinates=v)
    b=TypeAccession(r,1,'new',((0,0,1),(1,0,1)),coordinates=w)
    k=matrix(v,w,[[1],[1]])
    assert accession_delta(a,b,j,k)['entries']==()
    with pytest.raises(ValueError): accession_delta(a,b,j)
    wrong=matrix(space('wrong',1),w,[[1],[1]])
    with pytest.raises(ValueError): accession_delta(a,b,j,wrong)


def test_chronological_contact_order_and_retained_interactions():
    s=space('state',3);i=CoordinateMap.identity(s)
    ab=matrix(s,s,[[Q(1,2),0,0],[Q(1,2),1,0],[0,0,1]])
    bc=matrix(s,s,[[1,0,0],[0,Q(1,2),0],[0,Q(1,2),1]])
    w=TemporalWord((bc,ab),(ab,bc),(i,i,i),('first','second'))
    x=[1,0,0]
    assert CoordinateWord(w.old).apply(x).tolist()==[Q(1,2),Q(1,2),0]
    assert CoordinateWord(w.new).apply(x).tolist()==[Q(1,2),Q(1,4),Q(1,4)]
    gram=w.delta(x).moments().values
    assert gram.tolist()==[[Q(3,8),Q(-3,8)],[Q(-3,8),Q(1,2)]]
    assert w.delta(x).moments().contract()==Q(1,8)
    assert sum(gram[j,j] for j in range(2))==Q(7,8)


def test_closed_route_difference_with_cost_only_selection_change():
    vertices=space('vertices',3);edges=space('edges',3)
    b=matrix(edges,vertices,[[-1,0,-1],[1,-1,0],[0,1,1]])
    old=np.array([1,1,0],object);new=np.array([0,0,1],object)
    assert b.apply(old).tolist()==b.apply(new).tolist()==[-1,0,1]
    assert b.apply(new-old).tolist()==[0,0,0]
    assert sum(v*v for v in new-old)==3
