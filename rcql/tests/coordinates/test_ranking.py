from fractions import Fraction as Q
from dataclasses import replace
import numpy as np
import pytest
import sympy as sp
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.type_accession import CoordinateSpace
from rexgraph.ranking_response import exact_pagerank,pagerank_delta
from rexgraph.graph import RexGraph
from rexgraph.markov import MarkovView
from rexgraph.markov_oracle import PairwiseMarkovOracle


def transitions():
    s=CoordinateSpace('states',('a','b','c'))
    t=CoordinateMap(s,s,((1,0,1),(0,1,Q(1,2)),(2,1,Q(1,2)),(1,2,1)))
    tp=CoordinateMap(s,s,tuple((i,j,Q(1,2)) for i in range(3) for j in range(3) if i!=j))
    return t,tp


def test_paper_ranking_delta_with_fixed_exact_values():
    t,tp=transitions();j=CoordinateMap.identity(t.domain)
    d=pagerank_delta(t,tp,j,damping=Q(1,2),seed=[1,0,0],new_seed=[0,0,1])
    assert d['old_rank']==(Q(7,12),Q(1,3),Q(1,12))
    assert d['new_rank']==(Q(1,5),Q(1,5),Q(3,5))
    assert d['fields'][0]==(Q(1,60),Q(-2,15),Q(7,60))
    assert d['fields'][1]==(Q(-2,5),0,Q(2,5))
    assert d['fields'][2]==(0,0,0)
    assert d['moments']['values']==((Q(19,600),Q(1,25),0),(Q(1,25),Q(8,25),0),(0,0,0))
    assert d['moments']['total']==Q(259,600)


@pytest.mark.parametrize('a,b',[(0,Q(1,2)),(Q(1,3),Q(3,4)),(Q(17,20),0),(Q(999,1000),Q(7,9))])
def test_damping_changes_match_direct_exact_solve(a,b):
    t,tp=transitions();j=CoordinateMap.identity(t.domain)
    d=pagerank_delta(t,tp,j,damping=a,new_damping=b,seed=[1,0,0],new_seed=[0,0,1])
    assert d['new_rank']==tuple(exact_pagerank(tp,b,[0,0,1]))
    pi=exact_pagerank(t,a,[1,0,0])
    assert d['old_rank']==tuple(pi)
    with pytest.raises(ValueError): pagerank_delta(t,tp,j,damping=a,seed=[1,0,0],old_rank=[1,0,1])


def test_rectangular_new_states():
    old=CoordinateSpace('old',('a','b'));new=CoordinateSpace('new',('a','b','c'))
    t=CoordinateMap(old,old,((1,0,1),(0,1,1)))
    tp=CoordinateMap(new,new,((1,0,1),(2,1,1),(0,2,1)))
    j=CoordinateMap(old,new,((0,0,1),(1,1,1)))
    d=pagerank_delta(t,tp,j,damping=Q(2,3),seed=[1,0],new_seed=[0,0,1])
    assert d['new_rank']==tuple(exact_pagerank(tp,Q(2,3),[0,0,1]))


@pytest.mark.parametrize('fixture',['pair','branch','isolated','witness','parallel','empty_edges'])
def test_native_participation_exact_against_independent_dense_system(fixture):
    cells={'pair':[3,[[0,1],[1,2]]],'branch':[4,[[0,1,2],[1,3]]],
           'isolated':[4,[[0,1]]],'witness':[3,[[0],[1,2]]],
           'parallel':[2,[[0,1],[0,1]]],'empty_edges':[3,[]]}[fixture]
    r=RexGraph.from_cells(cells);v=MarkovView(r)
    n=r.nV;cols=[v.apply(np.array([Q(int(i==j)) for i in range(n)],object),exact=True) for j in range(n)]
    t=sp.Matrix(np.column_stack(cols).tolist());source=sp.Matrix([1]+[0]*(n-1))
    expected=(sp.eye(n)-sp.Rational(3,4)*t).inv()*source/4
    result,report=exact_pagerank(v,Q(3,4),list(source),report=True)
    assert list(result)==list(expected)
    assert report['residual']==0
    assert report['system_size']==r.nV+r.nE+1


def test_endpoint_policy_differs_from_native_lazy_participation():
    r=RexGraph.from_cells([3,[[0,1],[1,2]]]);seed=[1,0,0]
    native=exact_pagerank(MarkovView(r),Q(1,2),seed)
    endpoint=exact_pagerank(PairwiseMarkovOracle(r),Q(1,2),seed)
    assert tuple(endpoint)==(Q(7,12),Q(1,3),Q(1,12))
    assert not np.array_equal(native,endpoint)


@pytest.mark.parametrize('bad',[-1,1,Q(4,3),0.5,True])
def test_invalid_damping_refused(bad):
    t,_=transitions()
    with pytest.raises((ValueError,TypeError)): exact_pagerank(t,bad)


@pytest.mark.parametrize('seed',[[0,0,0],[-1,1,1],[1.0,0,0],[True,0,0],[[1],[0],[0]]])
def test_invalid_seeds_refused(seed):
    t,_=transitions()
    with pytest.raises((ValueError,TypeError)): exact_pagerank(t,Q(1,2),seed)


def test_nonstochastic_or_negative_maps_refused():
    t,_=transitions()
    with pytest.raises(ValueError): exact_pagerank(replace(t,entries=()))
    with pytest.raises(ValueError): exact_pagerank(replace(t,entries=((0,0,-1),(1,0,2),(1,1,1),(2,2,1))))


def test_stale_native_view_refused():
    r=RexGraph.from_cells([3,[[0,1],[1,2]]]);v=MarkovView(r);r.add_edges([0],[2])
    with pytest.raises(ValueError): exact_pagerank(v,Q(1,2),[1,0,0,0])
