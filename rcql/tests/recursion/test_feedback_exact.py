from fractions import Fraction as Q
import pytest
import numpy as np
from rcql import Executor,parse
from rexgraph.affine_feedback import FeedbackEquation,AffineFeedback
from rexgraph.section_calculus import InconsistentSectionError,UnderdeterminedSectionError
from rexgraph.tensor_field import TensorField,FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMap
from .helpers import source


def fixture(coefficient,forcing=1):
    r=source();ref=FieldSource(r);s=CoordinateSpace('local',('x',))
    a=CoordinateMap(s,s,((0,0,Q(coefficient)),))
    f=AffineFeedback('feedback',(('x',s),),(FeedbackEquation('equation','x',(('x',a),),TensorField(s,[forcing],source=ref)),),ref)
    return r,f


@pytest.mark.parametrize('a,b,expected',[(Q(1,2),1,2),(-1,1,Q(1,2)),(2,1,-1),(0,Q(1,3),Q(1,3))])
def test_exact_feedback(a,b,expected):
    r,f=fixture(a,b);family=f.complete()
    assert family.dimension==0 and family.particular.values.tolist()==[expected]
    result=Executor(sources={'r':r},params={'f':f}).execute(parse('FROM $r LET family=FEEDBACK_COMPLETE($f) RETURN SECTION_VALUE(SECTION_OBSERVE(family,FEEDBACK_SELECT($f,"x")))'))
    assert result.values[0].values.tolist()==[expected]


def test_free_feedback_is_family():
    _,f=fixture(1,0);family=f.complete()
    assert family.dimension==1
    with pytest.raises(UnderdeterminedSectionError):family.observe(f.selection('x')).value()


def test_contradiction_witness():
    _,f=fixture(1,1);a,b=f.assemble()
    with pytest.raises(InconsistentSectionError) as caught:f.complete()
    witness=caught.value.witness
    assert all(v==0 for v in a.T.apply(witness.values))
    assert caught.value.contradiction.item()!=0


def test_recurrence_is_not_fixed_point():
    r,f=fixture(-1)
    initial=TensorField(f.space,[0],source=f.source)
    values=f.iterate(initial,4)
    assert [v.values.item() for v in values]==[0,1,0,1,0]
    assert f.complete().particular.values.item()==Q(1,2)
    out=Executor(sources={'r':r},params={'f':f,'x':initial}).execute(parse('FROM $r RETURN FEEDBACK_ITERATE($f,$x,4)')).values[0]
    assert len(out.fields)==5


def test_feedback_retains_tensor_axes():
    r=source();ref=FieldSource(r);s=CoordinateSpace('local',('x',));axis=CoordinateSpace('observations',('first','second'))
    a=CoordinateMap(s,s,((0,0,Q(1,2)),));b=TensorField(s,[[1,2]],(axis,),ref)
    f=AffineFeedback('batch',(('x',s),),(FeedbackEquation('equation','x',(('x',a),),b),),ref)
    result=f.complete()
    assert result.particular.axes==(axis,) and result.particular.values.tolist()==[[2,4]]


@pytest.mark.parametrize('n',[2**53+1,2**63+3,2**130+7])
def test_exact_legacy_ingress(n):
    from rexgraph.hodge_coords import _exact_ints,_solve_gram,harmonic_structure_constants,harmonic_closure,harmonic_gram_det
    assert _exact_ints([n],'input')==[Q(n)]
    assert _solve_gram(np.asarray([[n]],dtype=object),[1],True)==[Q(1,n)]
    H=np.asarray([[n],[1]],dtype=object)
    assert harmonic_structure_constants(None,0,0,frame=H,exact=True)==[Q(n**3+1,n*n+1)]
    assert harmonic_gram_det(None,frame=H)==n*n+1
    assert harmonic_closure(None,frame=H,exact=True)==[[Q((n**3+1)**2,(n*n+1)*(n**4+1))]]


@pytest.mark.parametrize('value',[True,Q(1,2),0.5,float('inf'),float('nan'),float(2**54)])
def test_nonintegral_or_uncertified_frame_refused(value):
    from rexgraph.hodge_coords import _exact_ints
    with pytest.raises((TypeError,ValueError)):_exact_ints([value],'input')


def test_empty_and_column_exact_gram_contracts():
    from rexgraph.hodge_coords import _solve_gram,harmonic_gram_det,harmonic_closure
    assert _solve_gram(np.zeros((0,0),dtype=int),[],True)==[]
    assert _solve_gram(np.asarray([[2,0],[0,3]]),np.asarray([[1],[1]]),True)==[Q(1,2),Q(1,3)]
    assert harmonic_gram_det(None,frame=np.zeros((0,0),dtype=int))==1
    assert harmonic_closure(None,frame=np.zeros((0,0),dtype=int),exact=True).shape==(0,0)


def test_sparse_duplicate_large_integer_frame_is_exact():
    from scipy.sparse import coo_matrix
    from rexgraph.hodge_coords import harmonic_gram_det
    n=2**53+1
    f=coo_matrix((np.asarray([n,2,3],dtype=np.int64),(np.asarray([0,0,1]),np.asarray([0,0,0]))),shape=(2,1))
    assert harmonic_gram_det(None,frame=f)==(n+2)**2+9
