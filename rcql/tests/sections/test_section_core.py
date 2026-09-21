from fractions import Fraction as F
from dataclasses import replace
import numpy as np
import pytest
from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSheaf, UndeclaredRestrictionError
from rexgraph.section_calculus import (SectionSystem, InconsistentSectionError, UnderdeterminedSectionError)
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.tensor_field import TensorField
from rexgraph.tensor_moment import CoordinatePairing, MomentSpan
from rexgraph.type_accession import CoordinateSpace


def system(n=3, width=1, name='section'):
    r=RexGraph.from_hypergraph([0,n],list(range(n)))
    s=ExactSheaf(r,grade=0,stalk_dim=width)
    return s, SectionSystem.from_sheaf(s,name=name)


def test_primary_correspondence_not_expanded():
    s,c=system(7)
    assert s.rex.nE==1
    assert c.action.shape==(6,7)
    assert len(c.action.entries)==12
    assert len(c.comparisons)==6


@pytest.mark.parametrize('n',[2,3,7])
@pytest.mark.parametrize('width',[0,1,3])
def test_residual_matches_existing_anchor_checks(n,width):
    r=RexGraph.from_hypergraph([0,n],list(range(n)))
    s=ExactSheaf(r,grade=0,stalk_dims=[width]*n,mediator_dims=[width])
    for i in range(n):s.assign(i,[F(i,j+1) for j in range(width)])
    c=SectionSystem.from_sheaf(s)
    y=c.residual(c.field()).values
    legacy=s.check_section()
    for k,(a,b,m) in enumerate(c.comparisons):
        part=tuple(y[k*width:(k+1)*width])
        found=[v for v in legacy.obstructions if (v.left_cell,v.right_cell,v.mediator)==(a,b,m)]
        assert part == (found[0].residual if found else (F(0),)*width)
    assert legacy.compatible==(not any(y))


def test_missing_restriction_not_assumed_for_strict():
    r=RexGraph.from_hypergraph([0,2],[0,1]);s=ExactSheaf(r,grade=0,require_declared_restrictions=True)
    with pytest.raises(UndeclaredRestrictionError):SectionSystem.from_sheaf(s)


def test_assigned_zero_is_not_observation_by_default():
    s,c=system()
    family=c.complete()
    assert family.dimension==1
    assert list(family.particular.values)==[0,0,0]
    with pytest.raises(UnderdeterminedSectionError):family.observe(c.selection('1')).value()
    o,b=c.pins({('0','0'):0})
    fixed=c.complete(o,b)
    assert fixed.dimension==0
    assert list(fixed.observe(c.selection('1')).value().values)==[0]


def test_exact_scaled_prediction():
    s,_=system()
    s.restrict(0,[[2]])
    c=SectionSystem.from_sheaf(s)
    o,b=c.pins({('0','0'):F(7,3)})
    f=c.complete(o,b)
    assert list(f.particular.values)==[F(7,3),F(14,3),F(14,3)]
    assert f.observe(c.selection('2')).determined
    assert s.rex.nE==1


def test_underdetermined_state_has_determined_observation():
    r=RexGraph.from_hypergraph([0,2],[0,1]);s=ExactSheaf(r,grade=0,stalk_dims=(2,2),mediator_dims=(1,))
    s.restrict(0,[[1,0]]);s.restrict(1,[[1,0]])
    c=SectionSystem.from_sheaf(s)
    o,b=c.pins({('0','0'):F(1,3)})
    family=c.complete(o,b)
    assert family.dimension==2
    target=CoordinateSpace('known',('first',))
    image=family.observe(CoordinateMap(c.space,target,((0,2,1),)))
    assert image.determined
    assert image.value().values[0]==F(1,3)
    assert not family.observe(c.selection('1')).determined


def test_contradiction_is_exact_witness():
    s,c=system(2)
    o,b=c.pins({('0','0'):1,('1','0'):2})
    with pytest.raises(InconsistentSectionError) as error:c.complete(o,b)
    witness=error.value.witness
    D=CoordinateMap(c.space,witness.space, c.action.entries+tuple((1+i,j,v) for i,j,v in o.entries))
    assert not any(D.T.apply(witness.values))
    assert error.value.contradiction!=0


def test_arbitrary_precision_and_retained_axes():
    s,c=system(2)
    axis=CoordinateSpace('trials',('a','b'))
    o,_=c.pins({('0','0'):1})
    values=TensorField(o.codomain,[[2**100+1,F(2,7)]],(axis,),c.source)
    f=c.complete(o,values)
    assert f.particular.axes==(axis,)
    assert f.particular.values.tolist()==[[2**100+1,F(2,7)]]*2


@pytest.mark.parametrize('bad',[0.5,True,np.float64(1)])
def test_inexact_evidence_refused(bad):
    s,c=system(2)
    with pytest.raises(TypeError):c.pins({('0','0'):bad})


def test_changing_restrictions_requires_fresh_system():
    s,c=system(2);s.restrict(0,[[2]])
    with pytest.raises(ValueError,match='restrictions changed'):c.field()


def test_assigning_field_does_not_change_equations():
    s,c=system(2);s.assign(0,[3]);s.assign(1,[5])
    assert c.field().values.tolist()==[3,5]
    assert c.residual(c.field()).values.tolist()==[-2]


def test_exact_boundary_share_restrictions():
    s,_=system(4);s.bind_boundary();c=SectionSystem.from_sheaf(s)
    o,b=c.pins({('0','0'):1})
    assert c.complete(o,b).particular.values.tolist()==[1,-3,-3,-3]


def test_multiple_mediators_same_pair():
    r=RexGraph.from_hypergraph([0,2,4],[0,1,0,1]);s=ExactSheaf(r)
    s.restrict(1,[[2]],mediator=1);c=SectionSystem.from_sheaf(s)
    assert len(c.comparisons)==2
    f=c.complete()
    assert f.dimension==0


def test_no_incidence_has_free_coordinates():
    r=RexGraph.from_hypergraph([0,1,2],[0,1])
    s=ExactSheaf(r,grade=1)
    c=SectionSystem.from_sheaf(s)
    assert c.complete().dimension==2


def test_family_rejects_incomplete_kernel():
    s,c=system(2);f=c.complete()
    empty=CoordinateMap(CoordinateSpace('empty',()),c.space,())
    with pytest.raises(ValueError,match='complete kernel'):replace(f,directions=empty)


def test_free_parameter_evaluation():
    s,c=system(2);f=c.complete()
    field=f.evaluate(TensorField(f.directions.domain,[5]))
    assert field.values.tolist()==[5,5]
    assert not any(c.residual(field).values)


def test_temporal_restriction_and_section_channels():
    s,old=system(2);t,new0=system(2);t.restrict(0,[[2]]);new=SectionSystem.from_sheaf(t)
    J=CoordinateMap(old.space,new.space,((0,0,1),(1,1,1)))
    K=CoordinateMap(old.residual_space,new.residual_space,((0,0,1),))
    channels=old.delta(new,J,K,old.field([1,1]),new.field([1,2]))
    assert channels.names==('restrictions','section')
    assert channels.fields[0].values.tolist()==[1]
    assert channels.fields[1].values.tolist()==[-1]
    assert channels.total().values.tolist()==[0]
    form=CoordinatePairing(new.residual_space,new.residual_space,((0,0,1),))
    assert MomentSpan(*channels.fields,form).scalar()==-1


def test_recipe_binding_requires_live_sources():
    s,c=system(2);recipe=c.recipe.detached()
    with pytest.raises(ValueError):recipe.bind(recipe.source)
    rebound=recipe.bind(c.source)
    assert rebound.coefficient_digest==c.coefficient_digest
    with pytest.raises(ValueError):rebound.field()
    assert rebound.residual(rebound.field([3,4])).values.tolist()==[-1]


def test_recipe_rejects_missing_map():
    s,c=system(2)
    with pytest.raises(ValueError,match='every section incidence'):replace(c.recipe,restrictions=())


def test_user_named_support_axes_remain_located():
    s,_=system(2,width=2)
    supports=CoordinateSpace('intervals',('[2017,2019)','[2024,2026)'))
    c=SectionSystem.from_sheaf(s,stalk_spaces=(supports,supports),mediator_spaces=(supports,))
    residual=c.residual(c.field([1,1,1,0]))
    assert residual.values.tolist()==[0,1]
    assert '[2024,2026)' in residual.space.keys[1]
    metric=CoordinatePairing(residual.space,residual.space,((0,0,2),(1,1,2)))
    assert MomentSpan(residual,residual,metric).support().values.tolist()==[0,2]


def test_direct_recipe_cannot_replace_native_incidence():
    from dataclasses import replace
    from rexgraph.sheaf import ExactSheaf
    from rexgraph.graph import RexGraph
    from rexgraph.section_calculus import SectionSystem
    r = RexGraph.from_hypergraph([0, 2], [0, 1])
    c = ExactSheaf(r, grade=0).section_system()
    bad = replace(c.recipe, incidences=((0,), ()), restrictions=(c.recipe.restrictions[0],))
    with pytest.raises(ValueError, match='incidence'):
        SectionSystem(bad)


def test_rectangular_temporal_section_innovation():
    from rexgraph.sheaf import ExactSheaf
    from rexgraph.graph import RexGraph
    from rexgraph.coordinate_map import CoordinateMap
    a = ExactSheaf(RexGraph.from_hypergraph([0, 2], [0, 1]), grade=0).section_system()
    b = ExactSheaf(RexGraph.from_hypergraph([0, 3], [0, 1, 2]), grade=0).section_system()
    J = CoordinateMap(a.space, b.space, ((0, 0, 1), (1, 1, 1), (2, 0, 1)))
    K = CoordinateMap(a.residual_space, b.residual_space, ((0, 0, 1),))
    result = a.delta(b, J, K, a.field([1, 1]), b.field([1, 1, 2]))
    assert result.field('restrictions').values.tolist() == [0, 0]
    assert result.field('section').values.tolist() == [0, -1]
    assert result.total().values.tolist() == [0, -1]
