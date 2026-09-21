from fractions import Fraction as Q
from dataclasses import replace
import numpy as np
import pytest
from rexgraph.model_runtime import native_model,infer_model,certify_native_response
from rexgraph.model_state import ModelOutput
from rexgraph.tensor_field import TensorField,FieldSource
from rexgraph.coordinate_map import CoordinateMetric
from rexgraph.type_accession import CoordinateSpace
from .helpers import rex


@pytest.mark.parametrize('parameter',[Q(0),Q(1,3),Q(2)])
def test_residual_bound_exact_without_recomputing_solution(parameter):
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.tensor_moment import MomentSpan,CoordinatePairing
    g=rex();cal=NativeFieldCalculus.from_rex(g)
    metrics=list(cal.metrics);metrics[1]=CoordinateMetric(cal.complex.spaces[1],[(0,0,2),(1,1,3),(2,2,5)])
    m=native_model(g,parameter=parameter,metrics=metrics)
    x=TensorField(m.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    guess=TensorField(m.space,[Q(1,3),Q(1,7),Q(1,11)],source=FieldSource(g),grade=1,variance='chain')
    result=certify_native_response(m,g,x,guess)
    truth=infer_model(m,g,x).tensor()
    error=replace(guess,values=guess.values-truth.values)
    true_q=MomentSpan(error,error,CoordinatePairing.metric(metrics[1])).scalar()
    assert true_q<=result['quadrance_bound'].scalar()
    assert result['residual'].axes==()
    assert not result['exact_solution']
    perfect=certify_native_response(m,g,x,truth)
    assert perfect['exact_solution'] and perfect['quadrance_bound'].scalar()==0


def test_certificate_retains_field_axes_and_does_not_sum_them():
    g=rex();m=native_model(g)
    x=TensorField(m.space,[[1,0],[0,1],[0,0]],(CoordinateSpace('requests',('a','b')),),FieldSource(g),1,'chain')
    guess=replace(x,values=np.zeros((3,2),dtype=object))
    c=certify_native_response(m,g,x,guess)
    assert c['quadrance_bound'].contract_support().shape==(2,2)
    with pytest.raises(ValueError):c['quadrance_bound'].scalar()


def test_numeric_candidate_requires_declared_binary_interpretation():
    g=rex();m=native_model(g);x=TensorField(m.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    candidate=ModelOutput(np.array([0.3,0.2,0.1]),m.space,(),m.source,'a'*64,'approximate','recorded estimate',(),1,'chain')
    with pytest.raises(TypeError,match='explicit'):certify_native_response(m,g,x,candidate)
    c=certify_native_response(m,g,x,candidate,recorded_binary=True)
    assert c['candidate_arithmetic']=='approximate' and candidate.arithmetic=='approximate'
    assert c['interpretation']=='recorded binary candidate'
    assert all(isinstance(v,Q) for v in c['residual'].values)


def test_certificate_does_not_run_a_green_solve(monkeypatch):
    from rexgraph.native_field import FieldAction
    g=rex();m=native_model(g);x=TensorField(m.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    real=FieldAction.apply
    def checked(self,values):
        assert self.operation!='green'
        return real(self,values)
    monkeypatch.setattr(FieldAction,'apply',checked)
    certify_native_response(m,g,x,x)


def test_certificate_refuses_wrong_variance():
    g=rex();m=native_model(g);x=TensorField(m.space,[1,0,0],source=FieldSource(g),grade=1,variance='chain')
    with pytest.raises(ValueError,match='variance'):certify_native_response(m,g,x,replace(x,variance='cochain'))
