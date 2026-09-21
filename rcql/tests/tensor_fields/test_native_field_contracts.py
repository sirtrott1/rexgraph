from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.chain_map import CoordinateComplex, GradedMap
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric
from rexgraph.native_field import NativeFieldCalculus, bridge_action
from rexgraph.tensor_field import FieldSource, TensorField, apply_tensor
from rexgraph.tensor_moment import TensorMomentKernel, CoordinatePairing, MomentSpan
from rexgraph.temporal_field import TensorEvolution, ResolvedEvolution, SectorTransport
from rexgraph.type_accession import CoordinateSpace
from rexgraph.span import SpanAttachment, SpanBlock
from rexgraph.attachment_field import AttachmentField, common_attachment_observations
from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import to_state, from_state


def tower():
    spaces=tuple(CoordinateSpace('grade'+str(k),tuple(str(i) for i in range(n))) for k,n in enumerate((2,3,2,1)))
    b1=((0,0,-1),(1,0,1),(0,1,-1),(1,1,1),(0,2,-1),(1,2,1))
    b2=((0,0,1),(1,0,-1),(0,1,1),(1,1,-1))
    b3=((0,0,1),(1,0,-1))
    c=CoordinateComplex(spaces,(b1,b2,b3))
    return NativeFieldCalculus(c,tuple(CoordinateMetric.diagonal(s,tuple(Q(i+2,k+1) for i in range(len(s.keys))))
                                      for k,s in enumerate(spaces)))


@pytest.mark.parametrize('grade',[0,1,2,3])
@pytest.mark.parametrize('parameter',[Q(0),Q(2,7)])
def test_all_grade_exact_green_and_sectors(grade,parameter):
    import sympy as sp
    c=tower(); b,u,ml,m,mu=c.adjacent(grade)
    def matrix(a):return sp.Matrix(a.nrows,a.ncols,lambda i,j:a.entries.get((i,j),0))
    B,U,ML,M,MU=map(matrix,(b,u,ml,m,mu))
    L=M.inv()*B.T*ML*B+U*MU.inv()*U.T*M
    x=[Q(i+3,5) for i in range(M.rows)]
    actual=c.green(grade,parameter).apply(x)
    assert actual.tolist()==list((sp.eye(M.rows)+parameter*L).inv()*sp.Matrix(x))
    assert sum(c.split(grade,x)).tolist()==x
    assert c.green(grade,parameter).T.apply(x).tolist()==list((sp.eye(M.rows)+parameter*L).inv().T*sp.Matrix(x))


def test_family_metric_owner_and_cross_form_bridge():
    from rexgraph.type_accession import TypeAccession,TypeRealization,FamilyMetric,CrossMetric
    from rexgraph.graded_metric import DiagonalMetric
    r=RexGraph.from_cells([3,[[0,1],[1,2]]]);c=NativeFieldCalculus.from_rex(r);ref=FieldSource(r)
    av=CoordinateSpace('a',('a0','a1'));bv=CoordinateSpace('b',('b0',))
    a=TypeAccession(r,1,'a',((0,0,1),(1,1,1)),coordinates=av)
    b=TypeAccession(r,1,'b',((0,1,1),),coordinates=bv)
    fa=TypeRealization(a,((0,0,1),(1,1,1)))
    fb=TypeRealization(b,((1,0,1),))
    family=FamilyMetric((fa,fb),DiagonalMetric(r,1,(2,3)))
    k=TensorMomentKernel.from_family_metric(family,c.complex,source=ref)
    x=TensorField(c.complex.spaces[1],[5,7],source=ref,grade=1,variance='chain')
    assert k.evaluate(x).pair('a','b').scalar()==147
    assert k.evaluate(x).pair('a','a').scalar()==197
    cross=CoordinatePairing.from_cross_metric(CrossMetric(a,b,((0,0,2),(1,0,-1))),c.complex)
    left=apply_tensor(bridge_action(a,c.complex),x);right=apply_tensor(bridge_action(b,c.complex),x)
    assert MomentSpan(left,right,cross).scalar()==21


def test_selected_pair_does_not_evaluate_unselected_action(monkeypatch):
    from rexgraph.native_field import FieldAction
    c=tower();s=c.complex.spaces[1];identity=CoordinateMap.identity(s);green=c.green(1)
    k=TensorMomentKernel(('identity','green'),(identity,green),common_metric=c.metrics[1])
    x=TensorField(s,[1,2,3],grade=1,variance='chain')
    def forbidden(*a,**k):raise AssertionError('unselected field evaluated')
    monkeypatch.setattr(FieldAction,'apply',forbidden)
    assert k.evaluate_pair('identity','identity',x).scalar()==25
    assert k.select(('identity',)).evaluate(x).pair('identity','identity').scalar()==25
    # Direct contraction evaluates declared terms, not every possible channel pair.
    single=k.select(('identity',)); assert single.contract(x,[2]).scalar()==100


def test_typed_sector_reconstruction():
    from rexgraph.reconstruction import ReconstructionFamily
    c=tower();s=c.complex.spaces[1];i=CoordinateMap.identity(s)
    half=CoordinateMap(s,s,tuple((j,j,Q(1,2)) for j in range(len(s.keys))))
    family=ReconstructionFamily(('a','b'),(i,i),(half,half))
    transport=SectorTransport(c,c,1,i)
    x=np.array([Q(2),Q(3),Q(5)],object);result=np.zeros(3,dtype=object)
    for source_type in family.names:
        for target_type in family.names:
            for origin in ('gradient','curl','harmonic'):
                for destination in ('gradient','curl','harmonic'):
                    block=transport.typed_channel(destination,origin,family,family,source_type,target_type)
                    a=family.observations[family.names.index(source_type)]
                    r=family.reconstructions[family.names.index(target_type)]
                    result+=r.apply(block.apply(a.apply(x)))
    assert result.tolist()==x.tolist()


def test_observed_resolved_evolution_has_four_exact_channels():
    from .native_field_fixtures import square
    old=square(False);new=square(True);s=old.complex.spaces[1]
    u=CoordinateSpace('request',('q',));o=CoordinateMap(u,s,((0,0,1),(1,0,1)))
    n=CoordinateMap(u,s,((2,0,-1),(3,0,-1)))
    v=CoordinateSpace('reading',('a','b'))
    a=CoordinateMap(s,v,((0,0,1),(1,1,1)))
    ap=CoordinateMap(s,v,((0,0,2),(1,1,1)))
    evo=ResolvedEvolution(old,new,1,o,n,CoordinateMap.identity(u),CoordinateMap.identity(s),Q(1),
                          old_observation=a,new_observation=ap,observation_map=CoordinateMap.identity(v))
    x=TensorField(u,[2]);xp=TensorField(u,[3]);delta=evo.delta(x,xp)
    direct=ap.apply(new.green(1).apply(n.apply([3])))-a.apply(old.green(1).apply(o.apply([2])))
    assert len(delta.names)==4
    assert delta.total().values.tolist()==direct.tolist()
    pair=evo.kernel(CoordinateMetric.identity(v)).evaluate(evo.diagnostic(x,xp))
    total=sum(pair.pair(a,b).scalar() for a in delta.names for b in delta.names)
    assert total==sum(z*z for z in direct)


def test_empty_attachment_endpoint_is_not_an_empty_time_axis():
    from .native_field_fixtures import attachment as ann
    old=AttachmentField((ann(),));new=AttachmentField(())
    a,b=common_attachment_observations(old,new,local=True)
    delta=TensorEvolution.from_observations(a,b).delta(old.amplitudes(),new.amplitudes())
    assert delta.total().values.tolist()==[-1,0,-1]
    assert b.codomain==a.codomain


@pytest.mark.parametrize('denominator',[3,7,29])
def test_large_exact_span_support(denominator):
    start=Q(2**240+1,denominator)
    a=SpanAttachment('a','e','time','doc',time=SpanBlock('t','event','unit',(('p',start,start+Q(2,9)),)))
    o=AttachmentField((a,)).observe(local=False)
    field=o.evaluate();moment=MomentSpan(field,field,CoordinatePairing.metric(o.metric))
    assert moment.support().values.tolist()==[Q(2,9)]
    r=RexGraph.from_cells([2,[[0,1]]]);r.attach_metadata(1,0,'moment',moment)
    restored=from_state(to_state(r)).get_metadata(1,0,'moment')
    assert restored.scalar()==Q(2,9)


def test_nested_attachment_source_and_role_preservation():
    from .native_field_fixtures import attachment as ann
    child=RexGraph.from_cells([2,[[0,1]]]);child.attach_metadata(1,0,'onset',ann(name='onset',role='onset'))
    child.attach_metadata(1,0,'offset',ann(name='offset',role='offset'))
    parent=RexGraph.from_cells([1,[[0]]]);parent.attach_metadata(1,0,'child',child)
    out=from_state(to_state(parent));f=AttachmentField.from_source(out)
    assert len(f.attachments)==2
    assert all(len(address)==2 for address in f.addresses)
    assert f.observe(local=False).evaluate().values.tolist()==[2,0,2]


def test_schema_refuses_live_kernel_string_fallback():
    r=RexGraph.from_cells([2,[[0,1]]]);c=NativeFieldCalculus.from_rex(r)
    k=TensorMomentKernel(('h',),(c.hodge(1),),common_metric=c.metrics[1])
    r.attach_metadata(1,0,'kernel',k)
    with pytest.raises(TypeError,match='live field declarations'):to_state(r)


def test_partial_moment_schema_requires_explicit_realization():
    from rexgraph.tensor_moment import RealizedPairing
    s=CoordinateSpace('s',('s',));v=CoordinateSpace('v',('v',));a=CoordinateMap(v,s,((0,0,2),))
    m=MomentSpan(TensorField(v,[3]),TensorField(v,[5]),RealizedPairing(a,a,CoordinateMetric.identity(s)))
    r=RexGraph.from_cells([1,[[0]]]);r.attach_metadata(1,0,'m',m)
    with pytest.raises(TypeError):to_state(r)
    r.attach_metadata(1,0,'m',m.realized())
    assert from_state(to_state(r)).get_metadata(1,0,'m').scalar()==60


def test_new_state_integrity_and_bad_deterministic_source():
    from rexgraph.io.rex_state import verify_state
    r=RexGraph.from_cells([1,[[0]]]);f=TensorField(CoordinateSpace('x',('x',)),[Q(1,7)])
    r.attach_metadata(1,0,'field',f);state=to_state(r)
    assert state.header['format_version']==5
    key=next(k for k in state.tensors if k.startswith('field/') and k.endswith('/spec'))
    state.tensors[key][0]^=1
    assert not verify_state(state)
    with pytest.raises(ValueError):from_state(state)
    with pytest.raises(ValueError):FieldSource(None,state_digest='bad')


def test_heterogeneous_fields_through_native_query():
    from rcql import Executor,parse
    r=RexGraph.from_cells([1,[[0]]]);source=FieldSource(r)
    a=CoordinateSpace('a',('a0','a1'));b=CoordinateSpace('b',('b0',))
    x=TensorField(a,[1,2],source=source);y=TensorField(b,[5],source=source)
    k=TensorMomentKernel(('a','b'),(CoordinateMap.identity(a),CoordinateMap.identity(b)),
                         (('a','b',CoordinatePairing(a,b,((0,0,2),(1,0,-3)))),))
    result=Executor(sources={'g':r},params={'kernel':k,'fields':(x,y)}).execute(parse(
        'FROM $g RETURN MOMENT_SUPPORT(MOMENT_PAIR(TENSOR_MOMENTS($kernel,$fields),"a","b"))')).values[0]
    assert result.values.tolist()==[10,-30]


def test_span_rate_is_not_the_moment_of_a_field_rate():
    s=CoordinateSpace('span',('a','b'));x=TensorField(s,[1,-2]);m=MomentSpan(x,x,CoordinatePairing(s,s,((0,0,2),(1,1,3))))
    assert m.scalar()==14
    assert m.rate(Q(2)).support().values.tolist()==[1,6]
    xr=x.rate(Q(2));assert MomentSpan(xr,xr,m.pairing).scalar()==Q(7,2)
    for value in (x,m):
        with pytest.raises(ValueError):value.rate(0)
        with pytest.raises(TypeError):value.rate(0.5)


def test_cross_version_moment_requires_declared_sources():
    from rcql import Executor,parse
    old=RexGraph.from_cells([1,[[0]]]);new=RexGraph.from_cells([1,[[0]]])
    old_ref=FieldSource(old,'doc',1);new_ref=FieldSource(new,'doc',2)
    s=CoordinateSpace('reading',('value',));i=CoordinateMap.identity(s)
    k=TensorMomentKernel(('value',),(i,),common_metric=CoordinateMetric.identity(s),endpoint_sources=(old_ref,new_ref))
    x=TensorField(s,[2],source=old_ref);y=TensorField(s,[3],source=new_ref)
    query=parse('FROM $old RETURN MOMENT_SUPPORT(MOMENT_PAIR(TENSOR_MOMENTS($k,$x,$y),"value","value"))')
    out=Executor(sources={'old':old},params={'k':k,'x':x,'y':y}).execute(query).values[0]
    assert out.values.tolist()==[6]
    assert {ref.version for ref in out.dependencies}=={1,2}
    wrong=TensorField(s,[3],source=FieldSource(new,'doc',3))
    with pytest.raises(Exception):Executor(sources={'old':old},params={'k':k,'x':x,'y':wrong}).execute(query)


def test_native_adjacent_delta_through_rcql():
    from rcql import Executor,parse
    from rexgraph.temporal_field import NativeFieldEvolution
    edges=[[0,1],[1,2],[2,3],[3,0]]
    old=RexGraph.from_cells([4,edges,[]])
    new=RexGraph.from_cells([4,edges,[[(i,1) for i in range(4)]]])
    a,b=NativeFieldCalculus.from_rex(old,top_grade=2),NativeFieldCalculus.from_rex(new)
    assert len(a.complex.spaces)==len(b.complex.spaces)==3
    maps=tuple(tuple((i,i,1) for i in range(min(m,n)))
               for m,n in zip(a.complex.sizes,b.complex.sizes,strict=True))
    ar,br=FieldSource(old,'document',1),FieldSource(new,'document',2)
    evo=NativeFieldEvolution(a,b,1,GradedMap(a.complex,b.complex,maps),ar,br)
    x=TensorField(a.complex.spaces[1],[2,2,0,0],source=ar,grade=1,variance='chain')
    y=TensorField(b.complex.spaces[1],[0,0,-3,-3],source=br,grade=1,variance='chain')
    out=Executor(sources={'old':old},params={'e':evo,'x':x,'y':y}).execute(parse(
        'FROM $old RETURN TENSOR_DELTA($e,$x,$y)')).values[0]
    form=CoordinatePairing.metric(evo.metric)
    assert [[MomentSpan(p,q,form).scalar() for q in out.fields] for p in out.fields]==[[16,-40],[-40,102]]
    assert MomentSpan(out.total(),out.total(),form).scalar()==38
    assert {r.version for r in out.endpoint_sources}=={1,2}
    params={'e':evo,'x':replace(x,variance='cochain'),'y':replace(y,variance='cochain')}
    with pytest.raises(Exception):
        Executor(sources={'old':old},params=params).execute(parse('FROM $old RETURN TENSOR_DELTA($e,$x,$y)'))
    with pytest.raises(ValueError):evo.delta(params['x'],params['y'])
