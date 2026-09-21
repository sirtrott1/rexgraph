from fractions import Fraction as Q
from itertools import combinations
import numpy as np
import pytest
import sympy as sp
from rexgraph.molecular_field import MolecularView, conformation_field, conformation_direction
from rexgraph.process_field import (sampled_field, trajectory_comparison, sample_rates, diagonal_axes,
    factor_contrast, factorial_contrast, response_direction)
from rexgraph.tensor_field import TensorField, apply_tensor
from rexgraph.coordinate_map import CoordinateMap, CoordinateMetric
from rexgraph.tensor_moment import MomentSpan, CoordinatePairing
from rexgraph.type_accession import CoordinateSpace
from rexgraph.native_field import NativeFieldCalculus
from .helpers import make, declaration

ATOMS=('map/1','map/2','map/3','map/4')
PAIR=tuple(combinations(ATOMS,2))


def test_fixed_bonds_and_moving_conformation(tmp_path):
    _,_,g=make(tmp_path)
    view=MolecularView.from_source(g,'chain')
    a,b=view.conformation('plus'),view.conformation('minus')
    assert np.array_equal(conformation_field(a,'pair_quadrance',PAIR).values,conformation_field(b,'pair_quadrance',PAIR).values)
    assert conformation_field(a,'oriented_volume',(ATOMS,)).values[0]==1
    assert conformation_field(b,'oriented_volume',(ATOMS,)).values[0]==-1
    for name in ('torsion_dot','torsion_spread'):
        assert np.array_equal(conformation_field(a,name,(ATOMS,)).values,conformation_field(b,name,(ATOMS,)).values)
    assert tuple(conformation_field(a,'pair_quadrance',tuple(zip(ATOMS[:-1],ATOMS[1:],strict=True))).values)==(1,1,1)


@pytest.mark.parametrize('reading,selections', [
    ('pair_quadrance',PAIR),('angle_dot',(ATOMS[:3],ATOMS[1:])),('angle_spread',(ATOMS[:3],ATOMS[1:])),
    ('torsion_dot',(ATOMS,)),('torsion_spread',(ATOMS,)),('oriented_volume',(ATOMS,))])
def test_rigid_motion_invariance(tmp_path,reading,selections):
    _,_,g=make(tmp_path,conformers={'chain':{'q':declaration(Q(3,4))}})
    f=MolecularView.from_source(g,'chain').conformation('q')
    rot=np.array([[Q(3,5),Q(-4,5),0],[Q(4,5),Q(3,5),0],[0,0,1]],dtype=object)
    moved=f.with_values(f.values@rot.T+np.array([17,Q(2,7),-9],dtype=object))
    assert np.array_equal(conformation_field(f,reading,selections).values,conformation_field(moved,reading,selections).values)


def test_reflection_retains_handedness(tmp_path):
    _,_,g=make(tmp_path)
    f=MolecularView.from_source(g,'chain').conformation('plus')
    reflected=f.with_values(f.values*np.array([1,1,-1],dtype=object))
    assert conformation_field(f,'oriented_volume',(ATOMS,)).values[0]==-conformation_field(reflected,'oriented_volume',(ATOMS,)).values[0]


@pytest.mark.parametrize('reading,arity', [('pair_quadrance',2),('angle_dot',3),('angle_spread',3),('torsion_dot',4),('torsion_spread',4),('oriented_volume',4)])
def test_exact_derivative_against_symbolic_oracle(tmp_path,reading,arity):
    _,_,g=make(tmp_path,conformers={'chain':{'q':declaration(Q(3,4))}})
    f=MolecularView.from_source(g,'chain').conformation('q')
    d=f.with_values([[Q(1,3),0,1],[0,1,0],[Q(-2,7),0,0],[1,Q(2,3),-2]])
    selected=(ATOMS[:arity],)
    result=conformation_direction(f,d,reading,selected,parameter='displacement',parameter_unit='1')
    t=sp.Symbol('t');p=[sp.Matrix([sp.Rational(x.numerator,x.denominator)+t*sp.Rational(dx.numerator,dx.denominator) for x,dx in zip(row,drow,strict=True)]) for row,drow in zip(f.values,d.values,strict=True)]
    if arity==2:
        a=p[1]-p[0];expr=a.dot(a)
    elif arity==3:
        a,b=p[0]-p[1],p[2]-p[1]
        expr=a.dot(b) if reading=='angle_dot' else 1-a.dot(b)**2/(a.dot(a)*b.dot(b))
    else:
        a,b,c=p[1]-p[0],p[2]-p[1],p[3]-p[2];n,m=a.cross(b),b.cross(c)
        expr=n.dot(c) if reading=='oriented_volume' else n.dot(m) if reading=='torsion_dot' else 1-n.dot(m)**2/(n.dot(n)*m.dot(m))
    expected=sp.diff(expr,t).subs(t,0)
    assert result.values[0]==Q(int(expected.p),int(expected.q))


def test_degenerate_spread_is_not_a_zero_angle(tmp_path):
    vals={k:[i,0,0] for i,k in enumerate(ATOMS)}
    _,_,g=make(tmp_path,conformers={'chain':{'line':declaration(values=vals)}})
    f=MolecularView.from_source(g,'chain').conformation('line')
    with pytest.raises(ValueError,match='undefined'):conformation_field(f,'torsion_spread',(ATOMS,))
    assert conformation_field(f,'oriented_volume',(ATOMS,)).values[0]==0


def test_sampled_divergence_and_reconvergence(tmp_path):
    _,_,g=make(tmp_path)
    view=MolecularView.from_source(g,'chain')
    fields={n:conformation_field(view.conformation(n),'oriented_volume',(ATOMS,)) for n in ('zero','plus','minus')}
    series=trajectory_comparison([fields['zero'],fields['minus'],fields['zero']],
                                 [fields['zero'],fields['plus'],fields['zero']],(0,Q(1,3),1),axis='elapsed',unit='s')
    delta=series.field('difference')
    assert delta.values.tolist()==[[0,2,0]]
    moments=MomentSpan(delta,delta,CoordinatePairing.metric(CoordinateMetric.identity(delta.space)))
    full=moments.support();assert full.values.shape==(1,3,3)
    located=diagonal_axes(full,full.axes[0].name,full.axes[1].name,'sample/elapsed/s')
    assert located.values.tolist()==[[0,4,0]]
    rates=sample_rates(delta);assert rates.values.tolist()==[[6,-3]]
    assert len(delta.dependencies)==1
    with pytest.raises(ValueError):moments.scalar()


def test_divergence_requires_correspondence(tmp_path):
    a=TensorField(CoordinateSpace('left',('a','b')),[1,2])
    b=TensorField(CoordinateSpace('right',('b','a')),[2,1])
    with pytest.raises(ValueError):trajectory_comparison([a],[b],[0],axis='time',unit='s')
    mapping=CoordinateMap(b.space,a.space,((0,1,1),(1,0,1)))
    result=trajectory_comparison([a],[b],[0],axis='time',unit='s',right_maps=[mapping])
    assert not any(result.field('difference').values.flat)


def test_factor_secants_and_interactions():
    space=CoordinateSpace('measured',('response',));f=lambda a,b:TensorField(space,[a+2*b+3*a*b])
    c=factor_contrast(f(0,0),f(2,0),2,parameter='a',unit='1')
    assert c.field('per_unit').values[0]==1
    channels=factorial_contrast(f(0,0),f(2,0),f(0,3),f(2,3),first_step=2,second_step=3,first_parameter=('a','1'),second_parameter=('b','1'))
    assert [c.values[0] for c in channels.fields]==[2,6,18]
    assert channels.total().values[0]==26
    with pytest.raises(ValueError):factor_contrast(f(0,0),f(1,0),0,parameter='a',unit='1')


@pytest.mark.parametrize('scale,dl,expected', [(1,0,Q(-2,9)),(0,-2,Q(2,9))])
def test_native_response_direction_without_eigensolve(tmp_path,scale,dl,expected):
    _,_,g=make(tmp_path,'[CH3:1][OH:2] m\n',conformers={})
    v=MolecularView.from_source(g,'m');f=v.field('bond_presence',native=True)
    calc=NativeFieldCalculus.from_rex(g);action=calc.green(1,1);response=apply_tensor(action,f)
    direction=CoordinateMap(action.domain,action.domain,((0,0,dl),))
    result=response_direction(action,response,f,direction,f.with_values([0]),parameter='declared_scale',unit='1',scale_direction=scale)
    assert result.values[0]==expected
    with pytest.raises(ValueError,match='does not solve'):
        response_direction(action,f,f,direction,f.with_values([0]),parameter='p',unit='1')


@pytest.mark.parametrize('times', [(0,0),(1,0),(0,), (0,0.5)])
def test_invalid_time_sequences(times):
    f=TensorField(CoordinateSpace('x',('x',)),[1])
    with pytest.raises((ValueError,TypeError)):sampled_field([f,f],times,axis='time',unit='s')


def test_old_conformer_refuses_mutated_source(tmp_path):
    _,_,g=make(tmp_path);v=MolecularView.from_source(g,'chain')
    g.attach_metadata(0,0,'new_fact','changed')
    with pytest.raises(ValueError,match='changed'):v.conformation('zero')


@pytest.mark.parametrize('axis',['cartesian/angstrom','cartesian//frame','cartesian/angstrom/'])
def test_cartesian_unit_and_frame_are_required(axis):
    f=TensorField(CoordinateSpace('atoms',('a','b')),[[0,0,0],[1,0,0]],
                  (CoordinateSpace(axis,('x','y','z')),))
    with pytest.raises(ValueError,match='unit and frame'):
        conformation_field(f,'pair_quadrance',(('a','b'),))
