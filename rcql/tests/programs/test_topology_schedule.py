from dataclasses import replace
from fractions import Fraction as Q
from threading import Barrier, Lock
import pytest

from rcql import Executor,parse,PlanTopology,PlanScheduler,ReadoutEquivalence
from rcql.scheduling import QueryCancelledError
from rexgraph.coordinator import Coordinator,LanePools
from rexgraph.coordinate_map import CoordinateMap
from .helpers import native,family


@pytest.fixture
def scheduler():
    pools=LanePools(cores_budget=4)
    scheduler=PlanScheduler(Coordinator(pools=pools,detect=False),width=4)
    yield scheduler
    pools.shutdown()


def test_parallel_matches_serial(scheduler):
    r,x,m=native();params={'x':x,'m':m}
    q=parse('FROM $r LET g=NATIVE_RESPONSE($x) RETURN g, MOMENT_SUPPORT(FIELD_PAIR(g,g,$m)), COUNT(CELLS(0)), COUNT(CELLS(1))')
    serial=Executor(sources={'r':r},params=params).execute(q)
    parallel=Executor(sources={'r':r},params=params,scheduler=scheduler).execute(q)
    assert serial.values[0].values.tolist()==parallel.values[0].values.tolist()
    assert serial.values[1].values.tolist()==parallel.values[1].values.tolist()
    assert serial.values[2:]==parallel.values[2:]==(4,3)
    assert [e['operator'] for e in serial.execution]==[e['operator'] for e in parallel.execution]
    assert any(len(w)>1 for w in scheduler.last_waves)


def test_independent_branches_overlap_and_traces_are_ordered(monkeypatch,scheduler):
    from rcql import operators
    original=operators._REGISTRY['COUNT'];barrier=Barrier(2);seen=[];lock=Lock()
    def both(*args):
        barrier.wait(timeout=5)
        value=original.fn(*args)
        with lock:seen.append(value)
        return value
    monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=both))
    r,_,_=native()
    answer=Executor(sources={'r':r},scheduler=scheduler).execute(parse('FROM $r RETURN COUNT(CELLS(0)),COUNT(CELLS(1))'))
    assert answer.values==(4,3) and sorted(seen)==[3,4]
    assert [e['operator'] for e in answer.execution]==['CELLS','COUNT','CELLS','COUNT']


def test_worker_error_reaches_caller_and_blocks_descendant(monkeypatch,scheduler):
    from rcql import operators
    original=operators._REGISTRY['COUNT'];down=operators._REGISTRY['SUM'];called=[]
    def bad(*args):raise ArithmeticError('test failure')
    def later(*args):called.append(1);return down.fn(*args)
    monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=bad))
    monkeypatch.setitem(operators._REGISTRY,'SUM',replace(down,fn=later))
    r,_,_=native()
    with pytest.raises(ArithmeticError,match='test failure'):
        Executor(sources={'r':r},scheduler=scheduler).execute(parse('FROM $r LET n=COUNT(CELLS(1)) RETURN SUM([n,1])'))
    assert not called


def test_cancellation_before_work(scheduler,monkeypatch):
    from rcql import operators
    original=operators._REGISTRY['COUNT'];called=[]
    def op(*args):called.append(1);return original.fn(*args)
    monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=op))
    scheduler.cancel.set();r,_,_=native()
    with pytest.raises(QueryCancelledError):Executor(sources={'r':r},scheduler=scheduler).execute(parse('FROM $r RETURN COUNT(CELLS(1))'))
    assert not called


def test_cancellation_after_wave(scheduler,monkeypatch):
    from rcql import operators
    original=operators._REGISTRY['COUNT']
    def op(*args):scheduler.cancel.set();return original.fn(*args)
    monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=op))
    r,_,_=native()
    with pytest.raises(QueryCancelledError):Executor(sources={'r':r},scheduler=scheduler).execute(parse('FROM $r RETURN COUNT(CELLS(1))'))


def test_parallel_source_change_refused(scheduler,monkeypatch):
    from rcql import operators
    original=operators._REGISTRY['COUNT'];r,_,_=native()
    def changed(*args):
        value=original.fn(*args);r.attach_metadata(1,0,'changed','yes');return value
    monkeypatch.setitem(operators._REGISTRY,'COUNT',replace(original,fn=changed))
    with pytest.raises(ValueError):Executor(sources={'r':r},scheduler=scheduler).execute(parse('FROM $r RETURN COUNT(CELLS(1))'))


def test_scheduler_explanation_executes_no_tasks(scheduler,monkeypatch):
    def bad(*args,**kwargs):raise AssertionError('unexpected evaluation')
    monkeypatch.setattr(scheduler.coordinator.pools,'run',bad)
    r,_,_=native()
    assert Executor(sources={'r':r},scheduler=scheduler).execute(parse('EXPLAIN FROM $r RETURN COUNT(CELLS(1))')).execution==()


def test_parallel_rejects_match_and_observable_reads(scheduler):
    r,_,_=native()
    with pytest.raises((ValueError,TypeError)):
        Executor(sources={'r':r},scheduler=scheduler).execute(parse('FROM $r MATCH e IN CELLS(1) RETURN e'))
    from rcql import Program,ProgramStep
    program=Program('simple',(ProgramStep('read',parse('FROM $r RETURN 1')),))
    with pytest.raises(ValueError):Executor(sources={'r':program.to_record()},scheduler=scheduler).execute(parse('FROM $r RETURN PROGRAM_READ()'))


def test_topology_preserves_repeated_ports_and_sharing():
    r,x,m=native();e=Executor(sources={'r':r},params={'x':x,'m':m})
    t=e.topology(parse('FROM $r LET g=NATIVE_RESPONSE($x) RETURN FIELD_PAIR(g,g,$m),NATIVE_RESPONSE($x)'))
    nodes=t.declaration['nodes']
    pair=next(n for n in nodes if n.get('operator')=='FIELD_PAIR')
    assert pair['inputs'][0]==pair['inputs'][1]
    assert len([n for n in nodes if n.get('operator')=='NATIVE_RESPONSE'])==1
    rplan=t.to_record()
    count=Executor(sources={'p':rplan}).execute(parse('FROM $p RETURN COUNT(CELLS(1))'))
    assert count.values==(len(nodes),)
    assert rplan.nF==0
    assert PlanTopology.from_record(rplan).declaration==t.declaration
    lift=t.port_realization()
    column_sums={}
    for row,col,value in lift.entries:column_sums[col]=column_sums.get(col,0)+value
    assert set(column_sums.values())=={Q(1)}
    ports=rplan._agent_meta['rcql_plan_ports']
    repeated=[p for p in ports if p[0]==pair['id'] and p[1]=='argument']
    assert repeated[0][3]==repeated[1][3] and repeated[0][2]!=repeated[1][2]

@pytest.mark.parametrize('suffix',('rcbd','safetensors'))
def test_topology_portability(tmp_path,suffix):
    from rexgraph.io.bundle import save_rcbd,load_rcbd
    from rexgraph.io.safetensors_bridge import rex_to_safetensors,load_safetensors
    r,_,_=native();t=Executor(sources={'r':r}).topology(parse('FROM $r RETURN COUNT(CELLS(1))'))
    path=tmp_path/('plan.'+suffix)
    if suffix=='rcbd':save_rcbd(path,t.to_record());record=load_rcbd(path)
    else:rex_to_safetensors(t.to_record(),path);record=load_safetensors(path)['object']
    assert PlanTopology.from_record(record).coefficient_digest==t.coefficient_digest
    record._agent_meta['rcql_plan_ports'][0][1]='argument'
    with pytest.raises(ValueError):PlanTopology.from_record(record)

@pytest.mark.parametrize('change',('future','duplicate','output'))
def test_invalid_topology_refused(change):
    r,_,_=native();data=Executor(sources={'r':r}).topology(parse('FROM $r RETURN COUNT(CELLS(1))')).declaration
    if change=='future':data['nodes'][0]['inputs']=['not_yet_defined']
    if change=='duplicate':data['nodes'][1]['id']=data['nodes'][0]['id']
    if change=='output':data['outputs']=['absent']
    with pytest.raises(ValueError):PlanTopology(data)


def test_readout_certificate_true_false_and_tamper():
    r,_,_=native();f,a,c=family(r)
    cert=ReadoutEquivalence.check(f,a,c)
    assert cert.equivalent and cert.verify(f,a,c)
    assert cert.replacement(f,a,c).value().values.tolist()==[Q(2,7)]
    b=CoordinateMap(a.domain,a.codomain,((0,1,1),))
    no=ReadoutEquivalence.check(f,a,b)
    assert not no.equivalent
    with pytest.raises(ValueError):no.replacement(f,a,b)
    with pytest.raises(ValueError):replace(cert,equivalent=False).verify(f,a,c)
    answer=Executor(sources={'r':r},params={'f':f,'a':a,'c':c}).execute(parse('FROM $r LET proof=SECTION_EQUIVALENT($f,$a,$c) RETURN proof.equivalent,proof.family_dimension'))
    assert answer.values==(True,2)

@pytest.mark.parametrize('seed',range(10))
def test_readout_equivalence_matches_independent_rational_oracle(seed):
    import random,sympy as sp
    from rexgraph.section_calculus import SectionFamily
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.tensor_field import TensorField,FieldSource
    rng=random.Random(seed);r,_,_=native();D=sp.Matrix([[rng.randint(-3,3) for _ in range(4)] for _ in range(2)])
    solution=sp.Matrix([rng.randint(-2,2) for _ in range(4)]);rhs=D*solution
    space=CoordinateSpace('inputs',tuple('abcd'));rows=CoordinateSpace('constraints',('one','two'))
    entries=tuple((i,j,Q(int(D[i,j]))) for i in range(2) for j in range(4) if D[i,j])
    f=SectionFamily.solve(CoordinateMap(space,rows,entries),TensorField(rows,[int(v) for v in rhs],source=FieldSource(r)),'random')
    out=CoordinateSpace('readout',('v',));a=sp.Matrix([[rng.randint(-3,3) for _ in range(4)]])
    b=a+sp.Matrix([[1,0]])*D if seed%2 else sp.Matrix([[rng.randint(-3,3) for _ in range(4)]])
    left=CoordinateMap(space,out,tuple((0,j,int(a[j])) for j in range(4) if a[j]))
    right=CoordinateMap(space,out,tuple((0,j,int(b[j])) for j in range(4) if b[j]))
    expected=(a-b)*solution==sp.zeros(1,1) and all((a-b)*n==sp.zeros(1,1) for n in D.nullspace())
    assert ReadoutEquivalence.check(f,left,right).equivalent==expected


@pytest.mark.parametrize("mode", ("program", "scheduler"))
def test_original_exception_preserved_without_note_api(monkeypatch,scheduler,mode):
    from rcql import operators,Program,ProgramStep
    class LegacyError(Exception):
        add_note = None
    original=operators._REGISTRY["COUNT"]
    def fail(*args):raise LegacyError("original failure")
    monkeypatch.setitem(operators._REGISTRY,"COUNT",replace(original,fn=fail))
    r,_,_=native();query=parse("FROM $r RETURN COUNT(CELLS(1))")
    with pytest.raises(LegacyError,match="original failure"):
        if mode=="program":Executor(sources={"r":r}).execute_program(Program("fails",(ProgramStep("read",query),)))
        else:Executor(sources={"r":r},scheduler=scheduler).execute(query)
