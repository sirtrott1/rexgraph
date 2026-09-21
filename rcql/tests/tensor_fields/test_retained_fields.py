from fractions import Fraction as Q
import numpy as np
import pytest
import sympy as sp

from rexgraph.chain_map import CoordinateComplex
from rexgraph.coordinate_map import CoordinateMap,CoordinateMetric
from rexgraph.native_field import NativeFieldCalculus,bridge_action
from rexgraph.tensor_field import TensorField,FieldSource,apply_tensor
from rexgraph.tensor_moment import CoordinatePairing,RealizedPairing,MomentSpan,TensorMomentKernel
from rexgraph.temporal_calculus import TemporalOperation
from rexgraph.temporal_field import TensorEvolution,ResolvedEvolution,SectorTransport,moment_change,factor_change_channels
from rexgraph.reconstruction import ReconstructionFamily
from rexgraph.span import SpanBlock,SpanAttachment
from rexgraph.attachment_field import AttachmentField,common_attachment_observations
from rexgraph.type_accession import CoordinateSpace


def space(name,n):return CoordinateSpace(name,tuple(str(i) for i in range(n)))
def mapping(domain,codomain,m):
    return CoordinateMap(domain,codomain,tuple((i,j,Q(v)) for i,row in enumerate(m) for j,v in enumerate(row) if v))
def square(filled=False,metric=None):
    spaces=(space('vertices',4),space('edges',4),space('faces',int(filled)))
    b=[[-1,0,0,1],[1,-1,0,0],[0,1,-1,0],[0,0,1,-1]]
    b1=mapping(spaces[1],spaces[0],b)
    b2=CoordinateMap(spaces[2],spaces[1],tuple((i,0,1) for i in range(4)) if filled else ())
    tower=CoordinateComplex(spaces,(b1.entries,b2.entries))
    metrics=tuple(CoordinateMetric.identity(s) for s in spaces)
    if metric is not None:metrics=(metrics[0],CoordinateMetric(spaces[1],tuple((i,j,Q(v)) for i,row in enumerate(metric) for j,v in enumerate(row) if v)),metrics[2])
    return NativeFieldCalculus(tower,metrics)


def test_retains_column_and_support_axes():
    s=space('support',2);f=space('field',2)
    x=TensorField(s,[[1,3],[2,4]],(f,))
    kernel=TensorMomentKernel(('identity',),(CoordinateMap.identity(s),),common_metric=CoordinateMetric.identity(s))
    pair=kernel.evaluate(x).pair('identity','identity')
    assert pair.contract_support().tolist()==[[5,11],[11,25]]
    assert pair.support().values.tolist()==[[[1,3],[3,9]],[[4,8],[8,16]]]
    with pytest.raises(ValueError):pair.scalar()


def test_arbitrary_tensor_axes_and_bilinear_inputs():
    s=space('primary',2);axes=tuple(space(name,2) for name in ('event','fragment','sample'))
    values=np.arange(16,dtype=object).reshape((2,2,2,2))
    x=TensorField(s,values,axes);y=TensorField(s,values+1,axes)
    pair=MomentSpan(x,y,CoordinatePairing.metric(CoordinateMetric.identity(s)))
    expected=np.multiply.outer(values[0],values[0]+1)+np.multiply.outer(values[1],values[1]+1)
    assert np.array_equal(pair.contract_support(),expected)
    assert pair.support().values.shape==(2,2,2,2,2,2,2)
    assert np.array_equal(apply_tensor(CoordinateMap.identity(s),x).values,values)


def test_heterogeneous_form_and_singular_realization():
    a,b,c=space('a',2),space('b',1),space('ambient',2)
    pair=MomentSpan(TensorField(a,[1,2]),TensorField(b,[5]),CoordinatePairing(a,b,((0,0,2),(1,0,-3))))
    assert pair.support().values.tolist()==[10,-30]
    assert pair.scalar()==-20
    f=mapping(b,c,[[1],[1]])
    p=MomentSpan(TensorField(b,[3]),TensorField(b,[4]),RealizedPairing(f,f,CoordinateMetric.identity(c)))
    assert p.scalar()==24
    assert p.support().values.tolist()==[12,12]


@pytest.mark.parametrize('filled',[False,True])
def test_exact_sectors_reconstruct_without_eigensolve(filled,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('forbidden numerical spectral path')
    for name in ('eig','eigh','inv'):monkeypatch.setattr(np.linalg,name,forbidden)
    calculus=square(filled)
    x=np.asarray([1,Q(2,3),3,4],dtype=object)
    g,c,h=calculus.split(1,x)
    assert np.array_equal(g+c+h,x)
    assert sum(h)==0 if filled else sum(c)==0
    y=calculus.green(1).apply(x)
    assert np.array_equal(y+calculus.hodge(1).apply(y),x)


def test_weighted_transposes_against_independent_oracle():
    m=sp.Matrix([[2,1,0,0],[1,3,0,0],[0,0,4,0],[0,0,0,5]])
    calc=square(True,m.tolist())
    b=sp.Matrix([[-1,0,0,1],[1,-1,0,0],[0,1,-1,0],[0,0,1,-1]])
    c=sp.ones(4,1);l=m.inv()*b.T*b+c*c.T*m
    x=np.asarray([1,2,Q(3,7),4],dtype=object)
    for action,oracle in ((calc.hodge(1),l),(calc.green(1,Q(2,3)),(sp.eye(4)+sp.Rational(2,3)*l).inv())):
        assert list(action.apply(x))==list(oracle*sp.Matrix(x))
        assert list(action.T.apply(x))==list(oracle.T*sp.Matrix(x))
    h=calc.sector(1,'curl')
    matrix=c*(c.T*m*c).inv()*c.T*m
    assert list(h.T.apply(x))==list(matrix.T*sp.Matrix(x))


def test_native_primary_lift_remains_one_relation():
    from rexgraph.graph import RexGraph
    from rexgraph.column_expansion import ColumnExpansion
    rex=RexGraph.from_cells([4,[[0,1,2,3]]])
    calc=NativeFieldCalculus.from_rex(rex)
    from rexgraph.linear_operator import boundary_operator
    expansion=ColumnExpansion(boundary_operator(rex,1))
    lift,legs=bridge_action(expansion.lift,calc.complex),bridge_action(expansion.legs,calc.complex)
    result=lift.then(legs).apply([6])
    assert result.tolist()==[-6,2,2,2]
    assert lift.then(legs).T.apply([1,2,3,4]).tolist()==[2]


def block(name,components):return SpanBlock(name,'event_time','abstract_year',tuple((str(i),Q(a),Q(b)) for i,(a,b) in enumerate(components)))
def attachment(name='a',owner='event_a',role='time',support=None):
    return SpanAttachment(name,owner,role,'doc',time=support or block('span',[(2017,2019),(2024,2026)]))


def test_interval_ranges_and_overlap():
    field=AttachmentField((attachment(support=block('overlap',[(2019,2025),(2020,2026)])),))
    observation=field.observe(local=False)
    values=observation.evaluate()
    assert values.values.tolist()==[1,2,1]
    assert observation.fragment_action.T.apply([1,2,3]).tolist()==[3,5]
    assert MomentSpan(values,values,CoordinatePairing.metric(observation.metric)).scalar()==22
    union=field.observe(local=False,mode='union')
    assert MomentSpan(union.evaluate(),union.evaluate(),CoordinatePairing.metric(union.metric)).scalar()==7


def test_local_attachment_identity_and_shared_calendar():
    old,new=AttachmentField((attachment(),)),AttachmentField((attachment(owner='event_b'),))
    for local,expected in ((True,8),(False,0)):
        a,b=common_attachment_observations(old,new,local=local)
        evolution=TensorEvolution.from_observations(a,b)
        delta=evolution.delta(old.amplitudes(),new.amplitudes()).total()
        assert MomentSpan(delta,delta,CoordinatePairing.metric(b.metric)).scalar()==expected


def test_support_delta_retains_changed_interval():
    old=AttachmentField((attachment(),))
    new=AttachmentField((attachment(support=block('span',[(2017,2019),(2025,2026)])),))
    a,b=common_attachment_observations(old,new,local=False)
    evolution=TensorEvolution.from_observations(a,b)
    delta=evolution.delta(old.amplitudes(),new.amplitudes()).total()
    assert delta.values.tolist()==[0,0,-1,0]
    moment=MomentSpan(delta,delta,CoordinatePairing.metric(b.metric))
    assert moment.support().values.tolist()==[0,0,1,0]
    change=moment_change(a.evaluate(),b.evaluate(),CoordinateMap.identity(a.codomain),a.evaluate(),b.evaluate(),
                        CoordinateMap.identity(a.codomain),CoordinatePairing.metric(a.metric),CoordinatePairing.metric(b.metric))
    assert change.contract_support().item()==-1


def test_grounding_remains_a_factor():
    text=SpanBlock('mention','text','character',(('a',0,2),('b',4,6)))
    time=block('dates',[(2017,2019),(2024,2026)])
    gamma=CoordinateMap(text.coordinates,time.coordinates,((0,0,1),(1,1,1)))
    a=SpanAttachment('a','event','time','doc',text=text,time=time,grounding=gamma)
    obs=AttachmentField((a,)).observe(support='grounded_time',local=False)
    assert obs.fragment_action.apply([2,3]).tolist()==[2,0,3]
    assert len(obs.factors)==3


def test_multiple_attachments_from_native_state_and_stale_refusal():
    from rexgraph.graph import RexGraph
    rex=RexGraph.from_cells([2,[[0,1]]])
    rex.attach_metadata(1,0,'first',attachment('a'))
    rex.attach_metadata(1,0,'second',attachment('b',role='modifier'))
    f=AttachmentField.from_source(rex,record_id='document',version=1)
    assert len(f.attachments)==2
    assert len(f.observe().evaluate().values)==6
    rex.attach_metadata(1,0,'second',attachment('b',owner='changed'))
    with pytest.raises(ValueError):f.observe().evaluate()


def test_paper_injection_sector_and_moment_values():
    old,new=square(False),square(True)
    u=space('request',1);p=mapping(u,old.complex.spaces[1],[[1],[1],[0],[0]])
    q=mapping(u,new.complex.spaces[1],[[0],[0],[-1],[-1]])
    j=CoordinateMap.identity(old.complex.spaces[1]);k=CoordinateMap.identity(u)
    evolution=TensorEvolution(TemporalOperation(p,q,k,j),names=('injection','amplitude'))
    a,b=TensorField(u,[2]),TensorField(u,[3])
    delta=evolution.delta(a,b)
    kernel=evolution.kernel(new.metrics[1]);moments=kernel.evaluate(evolution.diagnostic(a,b))
    assert [[moments.pair(i,j).scalar() for j in delta.names] for i in delta.names]==[[16,4],[4,2]]
    assert MomentSpan(delta.total(),delta.total(),CoordinatePairing.metric(new.metrics[1])).scalar()==26
    x,y=TensorField(j.domain,[2,2,0,0],grade=1),TensorField(j.codomain,[0,0,-3,-3],grade=1)
    sectors=SectorTransport(old,new,1,j)
    assert sectors.transport(x).field('curl/harmonic').values.tolist()==[1,1,1,1]
    assert np.array_equal(sectors.reconstruct(x,y).total().values,y.values)
    assert sum(x.values*old.hodge(1).apply(x.values))==8
    assert sum(y.values*new.hodge(1).apply(y.values))==54
    resolved=ResolvedEvolution(old,new,1,p,q,k,j)
    response=resolved.delta(a,b).total()
    expected=new.green(1).apply(q.apply([3]))-old.green(1).apply(p.apply([2]))
    assert np.array_equal(response.values,expected)


def test_subset_attribution_is_not_position_attribution():
    s=space('one',1);one=CoordinateMap.identity(s);two=mapping(s,s,[[2]])
    channels=factor_change_channels((one,one),(two,two),((0,),(1,),(0,1)))
    assert [a.apply([1]).tolist() for a in channels]==[[1],[1],[1]]


def test_reconstruction_and_redundant_coordinates():
    c=space('primary',1);v=space('view_a',1);w=space('view_b',1)
    a=mapping(c,v,[[1]]);b=mapping(c,w,[[1]])
    with pytest.raises(ValueError):ReconstructionFamily(('a','b'),(a,b),(a.T,b.T))
    family=ReconstructionFamily(('a','b'),(a,b),(mapping(v,c,[[Q(1,2)]]),mapping(w,c,[[Q(1,2)]])))
    assert family.reconstruct.apply(family.observe.apply([7])).tolist()==[7]
    assert family.consistency.apply([1,-1]).tolist()==[0,0]
    pair=family.representation_pairing(CoordinateMetric.identity(c))
    assert MomentSpan(TensorField(family.view_space,[1,-1]),TensorField(family.view_space,[1,-1]),pair).scalar()==0


def test_retained_codec_native_roundtrip():
    from rexgraph.graph import RexGraph
    from rexgraph.io.rex_state import to_state,from_state
    g=RexGraph.from_cells([2,[[0,1]]]);source=FieldSource(g,'old',1)
    s=space('support',2);axes=(space('events',2),)
    x=TensorField(s,[[Q(1,3),2**100+1],[2,Q(3,7)]],axes,source)
    pair=MomentSpan(x,x,CoordinatePairing.metric(CoordinateMetric.identity(s)))
    result=RexGraph.from_cells([1,[[0]]]);result.attach_metadata(1,0,'moment',pair)
    state=to_state(result)
    assert state.header['format_version']==5
    restored=from_state(state).get_metadata(1,0,"moment")
    assert np.array_equal(restored.support().values,pair.support().values)
    assert restored.left.source.state_digest==source.state_digest
    assert restored.left.source.source is None
    assert restored.left.bind(g).source.source is g


@pytest.mark.parametrize('bad',[True,1.0,float('nan')])
def test_exact_ingress_rejects_uncertified_values(bad):
    with pytest.raises((TypeError,ValueError)):TensorField(space('x',1),[bad])


def test_complete_native_adjacent_delta_and_hodge_identity():
    from rexgraph.chain_map import GradedMap
    from rexgraph.temporal_field import NativeFieldEvolution
    old,new=square(False),square(True)
    components=tuple(CoordinateMap(a,b,tuple((i,i,1) for i in range(min(len(a.keys),len(b.keys))))).entries
                     for a,b in zip(old.complex.spaces,new.complex.spaces,strict=True))
    mapping_=GradedMap(old.complex,new.complex,components)
    evolution=NativeFieldEvolution(old,new,1,mapping_)
    x,y=TensorField(old.complex.spaces[1],[2,2,0,0],grade=1,variance='chain'),TensorField(new.complex.spaces[1],[0,0,-3,-3],grade=1,variance='chain')
    delta=evolution.delta(x,y)
    moment=evolution.kernel().evaluate(evolution.diagnostic(x,y))
    assert [[moment.pair(a,b).scalar() for b in delta.names] for a in delta.names]==[[16,-40],[-40,102]]
    assert MomentSpan(delta.total(),delta.total(),CoordinatePairing.metric(evolution.metric)).scalar()==38
    assert evolution.hodge_delta(x).total().values.tolist()==[4,4,4,4]
    assert evolution.compatibility(2,TensorField(old.complex.spaces[2],[],grade=2,variance='chain')).values.tolist()==[0,0,0,0]


def test_form_action_returns_dual_variance_without_losing_axes():
    s=space('source',2)
    form=CoordinatePairing(s,s,((0,0,2),(1,1,3)))
    x=TensorField(s,[1,2],variance='chain')
    y=apply_tensor(form.form_action(),x)
    assert y.variance=='cochain'
    assert y.values.tolist()==[2,6]
    assert sum(x.values*y.values)==MomentSpan(x,x,form).scalar()
    with pytest.raises(ValueError):apply_tensor(form.form_action(),TensorField(s,[1,2]))
