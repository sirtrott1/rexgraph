from fractions import Fraction as Q
import pytest
import numpy as np
from rcql import Executor, parse, query, source as src, NameRelation, catalogued
from rcql.ast import Call, ListExpr, Literal, Parameter
from rexgraph.tensor_field import TensorField, FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.chain_map import _chain_residual
from .helpers import source, binding


def add(n):
    return NameRelation('add',('x',),Call('SUM',(ListExpr((Parameter('x'),Literal(n))),)))


def double():
    return NameRelation('double',('x',),Call('SUM',(ListExpr((Parameter('x'),Parameter('x'))),)))


@pytest.mark.parametrize('arguments',[[[1,2,3]],{'values':[1,2,3]}])
def test_named_argument_equivalence(arguments):
    r=source(); before=catalogued()
    assert NameRelation.operator('SUM').apply(binding(r),arguments)==6
    assert before==catalogued()


def test_no_default_chaining_between_names():
    r=source(); ex=Executor(sources={'r':r})
    result=ex.execute(parse('FROM $r LET first=NAME("SUM") LET second=NAME("COUNT") RETURN NAME_APPLY(first,[[1,2,3]]),NAME_APPLY(second,[[1,2,3]])'))
    assert result.values==(6,3)


def test_noncommuting_modifiers():
    r=source(); a,b=add(3),double()
    assert a.then(b,'x',modifier=True,name='add_then_double').apply(binding(r),[1])==8
    assert b.then(a,'x',modifier=True,name='double_then_add').apply(binding(r),[1])==5
    assert a.apply(binding(r),[1])==4
    assert b.apply(binding(r),[1])==2


def test_explicit_capture_replacement():
    r=source(); op=NameRelation.operator('SUM').bind('values',[1,2])
    assert op.apply(binding(r),[])==3
    with pytest.raises(TypeError):op.apply(binding(r),{'values':[9]})
    revised=op.rebind('values',[9])
    assert revised.apply(binding(r),[])==9 and op.apply(binding(r),[])==3
    assert revised.coefficient_digest != op.coefficient_digest


def test_port_rename_and_alias():
    r=source(); op=NameRelation.operator('SUM').rename('values','terms').named('accumulate')
    assert op.apply(binding(r),{'terms':[Q(1,3),Q(2,3)]})==1
    with pytest.raises(TypeError):op.apply(binding(r),{'values':[1,2]})
    assert op.name=='accumulate'


@pytest.mark.parametrize('port',['missing',(),('x','x')])
def test_no_guessed_composition_port(port):
    with pytest.raises(ValueError):add(1).then(double(),port)


def test_repeated_value_ports_remain_occurrences():
    op=double(); topology=op.topology(); record=topology.to_record()
    nodes=topology.declaration['nodes']
    sequence=next(n for n in nodes if n['kind']=='tuple')
    assert sequence['inputs'][0]==sequence['inputs'][1]
    assert len(sequence['roles'])==2
    assert _chain_residual(topology.boundary_tower())==0
    assert topology.port_realization().shape[1] > len(nodes)
    assert type(topology).from_record(record).coefficient_digest==topology.coefficient_digest


def test_primary_composite_shares_and_filling():
    op=add(1).then(double(),'x')
    tower=op.topology().boundary_tower()
    assert tower.sizes[2]==1 and _chain_residual(tower)==0
    assert any(v.denominator>1 for _,_,v in tower.boundaries[0])
    with pytest.raises(ValueError,match='chain condition'):
        op.topology().declare_filling('bad',((tower.spaces[1].keys[0],1),))


def test_impossible_string_rejected_before_response(monkeypatch):
    from rcql.operators import _REGISTRY
    from dataclasses import replace
    r=source(); f=TensorField(CoordinateSpace('C1',tuple(str(i) for i in range(r.nE))),[1,0,0],source=FieldSource(r),grade=1,variance='chain')
    op=NameRelation.operator('NATIVE_RESPONSE').then(NameRelation.operator('BOUNDARY').bind('grade',2),'values')
    original=_REGISTRY['NATIVE_RESPONSE']
    monkeypatch.setitem(_REGISTRY,'NATIVE_RESPONSE',replace(original,fn=lambda *a:pytest.fail('must not evaluate')))
    with pytest.raises((TypeError,ValueError)):
        op.apply(binding(r),[f])


def test_exact_tensor_axes_survive_named_response():
    from rexgraph.native_field import NativeFieldCalculus
    r=source(); calc=NativeFieldCalculus.from_rex(r)
    space=calc.complex.spaces[1]
    axis=CoordinateSpace('sources',('left','right'))
    f=TensorField(space,[[1,0],[0,0],[0,1]],(axis,),FieldSource(r),1,'chain')
    op=NameRelation.operator('NATIVE_RESPONSE').bind('parameter',Q(1,2))
    out=op.apply(binding(r),{'field':f})
    assert out.axes==(axis,) and out.values.shape==(3,2)
    direct=Executor(sources={'r':r},params={'f':f}).execute(parse('FROM $r RETURN NATIVE_RESPONSE($f,parameter=1/2)')).values[0]
    assert np.array_equal(out.values,direct.values)


def test_serialized_name_and_captured_origin():
    op=NameRelation.operator('NATIVE_RESPONSE').bind('parameter',Q(1,2))
    lifted=op.then(NameRelation.operator('TENSOR_APPLY'),'field',name='observed')
    assert 'left__parameter' in dict(lifted.bound_values)
    restored=NameRelation.from_bytes(lifted.to_bytes())
    assert restored.coefficient_digest==lifted.coefficient_digest
    changed=restored.rebind('left__parameter',Q(1,3))
    assert changed.coefficient_digest!=restored.coefficient_digest


def test_named_composition_parser_builder():
    r=source(); e=Executor(sources={'r':r},params={'a':add(3),'b':double()})
    text='FROM $r LET op=NAME_MODIFY($a,$b,"x","combined") RETURN NAME_APPLY(op,[1])'
    assert e.execute(parse(text)).values==(8,)
    relation=add(3).then(double(),'x',modifier=True,name='combined')
    assert Executor(sources={'r':r}).execute(query(src('r'),relation.call(x=1))).values==(8,)


def test_explanation_does_not_evaluate(monkeypatch):
    from dataclasses import replace
    from rcql.operators import _REGISTRY
    r=source(); op=NameRelation.operator('SUM'); entry=_REGISTRY['SUM']
    monkeypatch.setitem(_REGISTRY,'SUM',replace(entry,fn=lambda *a:pytest.fail('EXPLAIN evaluated')))
    assert op.explain(binding(r),[[1,2]]).returns


@pytest.mark.parametrize('op',['MODEL_TRAIN','PROGRAM_RUN','RCDB_GET'])
def test_effectful_or_observable_body_refused(op):
    with pytest.raises(ValueError):NameRelation.operator(op)
