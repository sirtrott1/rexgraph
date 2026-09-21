from dataclasses import replace
from fractions import Fraction as Q
from threading import Event
import pytest
from rcql import *
from rcql.ast import Parameter,Literal,Comparison,Call
from .helpers import source,binding,counter,fibonacci,integer,minus


@pytest.mark.parametrize('n',[0,1,2,10,32])
def test_counter_exact(n):
    result=counter().execute(binding(source()),'count',[n])
    assert result.value==n and result.calls==n+1 and result.completed
    assert result.history[-1][2]==n


def test_parser_recursive_result():
    r=source(); e=Executor(sources={'r':r},params={'p':counter()})
    q=parse('FROM $r LET result=RECURSIVE_RUN($p,"count",[8]) RETURN RECURSIVE_VALUE(result)')
    assert e.execute(q).values==(8,)


def test_mutual_recursion():
    n=Parameter('n'); defs=[]
    for name,target,base in [('even','odd',True),('odd','even',False)]:
        defs.append(RecursiveDefinition(name,(integer('n'),),Comparison('==',n,Literal(0)),Literal(base),
                    recur(target,minus(n)),ProgramInput('out',ValueKind.BOOLEAN.value),decreases='n'))
    p=RecursiveProgram('parity',tuple(defs)); b=binding(source())
    assert p.execute(b,'even',[11]).value is False
    assert p.execute(b,'odd',[11]).value is True


@pytest.mark.parametrize('n,expected',[(0,0),(1,1),(2,1),(7,13),(15,610)])
def test_completed_pure_calls_memoized(n,expected):
    result=fibonacci().execute(binding(source()),'fib',[n])
    assert result.value==expected
    if n>=3:assert any(v['branch']=='reuse' for v in result.invocations)
    assert result.calls<=2*n+1


def test_active_cycle_not_a_cached_answer():
    n=Parameter('n')
    definition=RecursiveDefinition('loop',(integer('n'),),Literal(False),Literal(0),recur('loop',n),integer('out'))
    with pytest.raises(RecursiveCycleError):RecursiveProgram('looping',(definition,)).execute(binding(source()),'loop',[1])


@pytest.mark.parametrize('limits',[{'calls':2},{'depth':2},{'evaluations':2}])
def test_incomplete_budget_is_not_result(limits):
    with pytest.raises(RecursionLimitError):counter().execute(binding(source()),'count',[4],limits=limits)


def test_declared_progress_refuses_growth():
    n=Parameter('n')
    p=RecursiveProgram('wrong',(RecursiveDefinition('grow',(integer('n'),),Comparison('==',n,Literal(0)),Literal(0),
        recur('grow',minus(n,-1)),integer('out'),decreases='n'),))
    with pytest.raises(ValueError,match='decrease'):p.execute(binding(source()),'grow',[2])


def test_cancellation():
    from rcql.scheduling import QueryCancelledError
    cancellation=Event();cancellation.set()
    with pytest.raises(QueryCancelledError):counter().execute(binding(source()),'count',[2],cancellation=cancellation)


def test_invalid_argument_types():
    with pytest.raises(TypeError):counter().execute(binding(source()),'count',[Q(1,2)])
    with pytest.raises(ValueError):counter().execute(binding(source()),'count',[-1])


def test_invalid_branch_rejected_before_base(monkeypatch):
    wrong=Call('COUNT',(Literal(3),))
    definition=RecursiveDefinition('bad',(integer('n'),),Literal(True),Literal(0),wrong,integer('out'))
    with pytest.raises(TypeError):RecursiveProgram('bad',(definition,)).execute(binding(source()),'bad',[0])


def test_bad_return_contract():
    p=counter();d=replace(p.definitions[0],base=Literal('wrong'))
    with pytest.raises(TypeError):RecursiveProgram('wrong',(d,)).explain(binding(source()),'count',[0])


def test_roundtrip_mutual_references():
    p=fibonacci(); restored=RecursiveProgram.from_bytes(p.to_bytes())
    assert restored.coefficient_digest==p.coefficient_digest
    assert restored.execute(binding(source()),'fib',[8]).value==21


def test_result_roundtrip_and_topology():
    r=source();result=counter().execute(binding(r),'count',[4]);record=result.to_record()
    loaded=RecursionResult.from_record(record,result.dependencies)
    assert loaded.coefficient_digest==result.coefficient_digest and loaded.value==4
    topology=loaded.topology(); assert len(topology.declaration['nodes'])==5
    assert RelationTopology.from_record(topology.to_record()).coefficient_digest==topology.coefficient_digest


def test_recursive_definition_topology_is_finite():
    p=fibonacci(); topology=p.topology()
    assert len(topology.declaration['nodes'])<30
    assert any(n['kind']=='recursive_reference' for n in topology.declaration['nodes'])
    from rexgraph.chain_map import _chain_residual
    assert _chain_residual(topology.boundary_tower())==0


def test_explain_does_not_evaluate(monkeypatch):
    from rcql.operators import _REGISTRY
    old=_REGISTRY['SUM'];monkeypatch.setitem(_REGISTRY,'SUM',replace(old,fn=lambda *a:pytest.fail('executed')))
    assert counter().explain(binding(source()),'count',[5])['evaluation']=='none'


def test_deep_calls_use_explicit_stack():
    result=counter().execute(binding(source()),'count',[1050],limits={'depth':1200,'calls':1100,'evaluations':10000},history=False)
    assert result.value==1050 and result.history==()
