from dataclasses import replace
from fractions import Fraction as Q
import pytest

from rcql import Executor, Program, ProgramStep, ProgramInput, OutputRef, parse, call, param
from rcql.program_codec import dumps,loads
from .helpers import native,tensor,count_program,query_case

@pytest.mark.parametrize('name',('TENSOR_SELECT','TENSOR_CONTRACT','TENSOR_DIAGONAL','SAMPLE_RATES','SECTION_EQUIVALENT',
                                 'PROGRAM_READ','PROGRAM_RECORD','PROGRAM_RUN','PROGRAM_EXPLAIN','PROGRAM_OUTPUT','PROGRAM_TOPOLOGY'))
def test_new_queries_and_explain(name):
    from rcql import query
    r,_,_=native();args=query_case(name,r)
    q=query(param('r'),call(name,*(param('p'+str(i)) for i in range(len(args)))))
    parameters={'p'+str(i):value for i,value in enumerate(args)}
    execution=Executor(sources={'r':r},params=parameters)
    answer=execution.execute(q)
    explained=execution.execute(replace(q,explain=True))
    assert not explained.execution
    assert answer.values


def test_axis_expected_values():
    r,_,_=native();x=tensor(r);e=Executor(sources={'r':r},params={'x':x})
    selected=e.execute(parse('FROM $r RETURN TENSOR_SELECT($x,"left/sample","0")')).values[0]
    assert selected.values.tolist()==[[1,2],[5,6]]
    contracted=Executor(sources={'r':r},params={'x':x,'w':(1,2)}).execute(parse('FROM $r RETURN TENSOR_CONTRACT($x,"left/sample",$w)')).values[0]
    assert contracted.values.tolist()==[[7,10],[19,22]]
    diagonal=e.execute(parse('FROM $r RETURN TENSOR_DIAGONAL($x,"left/sample","right/sample","sample")')).values[0]
    assert diagonal.values.tolist()==[[1,4],[5,8]]
    assert diagonal.axes[0].keys==('0','1')
    assert diagonal.dependencies
    with pytest.raises(ValueError):e.execute(parse('FROM $r RETURN TENSOR_SCALAR(TENSOR_DIAGONAL($x,"left/sample","right/sample","sample"))'))

@pytest.mark.parametrize('tail',('TENSOR_SELECT($x,"missing","0")','TENSOR_SELECT($x,"left/sample","9")',
 'TENSOR_CONTRACT($x,"left/sample",[1])','TENSOR_DIAGONAL($x,"left/sample","left/sample","a")'))
@pytest.mark.parametrize('prefix',('', 'EXPLAIN '))
def test_axis_refusals(tail,prefix):
    r,_,_=native()
    with pytest.raises((ValueError,KeyError,TypeError)):
        Executor(sources={'r':r},params={'x':tensor(r)}).execute(parse(prefix+'FROM $r RETURN '+tail))


def test_unknown_axes_explain_does_not_crash():
    from rcql import RCType,ValueKind,Exactness
    r,_,_=native();x=RCType('TensorField',kind=ValueKind.TENSOR_FIELD,exactness=Exactness.RATIONAL)
    Executor(sources={'r':r},params={'x':x}).execute(parse('EXPLAIN FROM $r RETURN TENSOR_SELECT($x,"unknown","key")'))


def test_program_chain_values_and_identity():
    r,x,m=native()
    p=Program('response',(ProgramStep('solve',parse('FROM $r RETURN NATIVE_RESPONSE($x)')),
       ProgramStep('observe',parse('FROM $r RETURN MOMENT_SUPPORT(FIELD_PAIR($g,$g,$m))'),(('g',OutputRef('solve')),))),
       (ProgramInput('x','TensorField',1,'chain'),ProgramInput('m')))
    e=Executor(sources={'r':r},params={'x':x,'m':m})
    result=e.execute_program(p)
    assert result.values[0].values.tolist()==[Q(64,441),Q(1,49),Q(1,441)]
    assert len(result.steps)==2
    assert result.program_digest==Program.from_bytes(p.to_bytes()).coefficient_digest
    assert not e.execute_program(p,explain=True).get('execution')
    with pytest.raises(ValueError):Executor(sources={'r':r},params={'x':x,'m':m,'extra':1}).execute_program(p)

@pytest.mark.parametrize('case',('future','duplicate','missing','effect','live_literal','wrong_input'))
def test_invalid_program_declarations(case):
    from rcql.ast import Query,Parameter,Literal
    with pytest.raises((ValueError,TypeError)):
        if case=='future':Program('bad',(ProgramStep('a',parse('FROM $r RETURN $v'),(('v',OutputRef('later')),)),))
        if case=='duplicate':Program('bad',(ProgramStep('a',parse('FROM $r RETURN 1')),ProgramStep('a',parse('FROM $r RETURN 2'))))
        if case=='missing':Program('bad',(ProgramStep('a',parse('FROM $r RETURN $v')),))
        if case=='effect':Program('bad',(ProgramStep('a',parse('FROM $r RETURN MODEL_TRAIN($m,$b,1)')),),(ProgramInput('m'),ProgramInput('b')))
        if case=='live_literal':ProgramStep('a',Query(Parameter('r'),(Literal(native()[0]),)))
        if case=='wrong_input':ProgramInput('value','NotAType')

@pytest.mark.parametrize('value',(Q(2,7),2**230,True,None,'nonascii Ω',b'abc',{'a':(Q(1,3),-2)},[1,2],0.25))
def test_finite_literals(value):
    assert loads(dumps(value))==value

@pytest.mark.parametrize('raw',(b'{}',b'{"literal":"int","value":"0x1","value":"0x2"}',
 b'{"node":"Exec","fields":{}}',b'{"literal":"float","value":"nan"}'))
def test_invalid_program_codec(raw):
    with pytest.raises((ValueError,TypeError)):loads(raw)

@pytest.mark.parametrize('suffix',('rcbd','safetensors'))
def test_program_portable_record(tmp_path,suffix):
    from rexgraph.io.bundle import save_rcbd,load_rcbd
    from rexgraph.io.safetensors_bridge import rex_to_safetensors,load_safetensors
    p=count_program();record=p.to_record();path=tmp_path/('program.'+suffix)
    if suffix=='rcbd':save_rcbd(path,record);restored=load_rcbd(path)
    else:rex_to_safetensors(record,path);restored=load_safetensors(path)['object']
    q=Program.from_record(restored)
    r,_,_=native()
    assert Executor(sources={'r':r}).execute_program(q).values==(3,)
    assert p.to_bytes()==q.to_bytes()


def test_program_native_query_route():
    from rcql import bind,SourcePolicy
    r,_,_=native();p=count_program();record=p.to_record()
    e=Executor(sources={'definition':record},params={'sources':{'r':bind('r',r,SourcePolicy.allow('*'))},'inputs':{}})
    result=e.execute(parse('FROM $definition LET p=PROGRAM_READ() LET run=PROGRAM_RUN(p,$sources,$inputs) RETURN PROGRAM_OUTPUT(run,0)'))
    assert result.values==(3,)


def test_program_agent_route():
    from agent.rcql_runtime import RCQLRuntime
    r,_,_=native();runtime=RCQLRuntime();runtime.register('r',r)
    assert runtime.execute_program(count_program()).values==(3,)


def test_program_result_feeds_typed_operation():
    from rcql import bind,SourcePolicy
    r,x,m=native()
    p=Program('respond',(ProgramStep('solve',parse('FROM $r RETURN NATIVE_RESPONSE($x)')),),(ProgramInput('x'),))
    execution=Executor(sources={'r':r},params={'p':p,'sources':{'r':bind('r',r,SourcePolicy.allow('*'))},'inputs':{'x':x},'m':m})
    q=parse('FROM $r LET out=PROGRAM_RUN($p,$sources,$inputs) LET g=PROGRAM_OUTPUT(out,0) RETURN MOMENT_SUPPORT(FIELD_PAIR(g,g,$m))')
    assert execution.execute(q).values[0].values.tolist()==[Q(64,441),Q(1,49),Q(1,441)]
    execution.execute(replace(q,explain=True))


def test_program_topology_retains_interstage_maps():
    from rcql import PlanTopology
    r,_,_=native();p=count_program();e=Executor(sources={'r':r})
    t=p.topology(e);d=t.declaration
    node=next(n for n in d['nodes'] if n.get('parameter')=='n')
    assert node['inputs'][0].startswith('count/')
    assert d['program_digest']==p.coefficient_digest
    assert PlanTopology.from_record(t.to_record()).declaration==d


def test_program_matching_retains_one_table_output():
    p=Program('iterate',(ProgramStep('choose',parse('FROM $r MATCH e IN CELLS(1) RETURN e.index,e.grade')),))
    r,_,_=native();e=Executor(sources={'r':r})
    assert e.execute_program(p).values==(((0,1),(1,1),(2,1)),)
    assert e.execute_program(p,explain=True)['output_types'][0].kind.value=='QueryTable'
    with pytest.raises(ValueError):p.topology(e)


def test_program_accepts_exact_scalar_contract():
    p=Program('scalar',(ProgramStep('read',parse('FROM $r RETURN $q')),),(ProgramInput('q','ExactRational'),))
    r,_,_=native()
    assert Executor(sources={'r':r},params={'q':Q(2,7)}).execute_program(p).values==(Q(2,7),)


@pytest.mark.parametrize("operation", ("PROGRAM_RUN", "PROGRAM_EXPLAIN", "PROGRAM_TOPOLOGY"))
def test_recursive_program_calls_are_not_implicitly_enabled(operation):
    with pytest.raises(ValueError,match='recursive'):
        Program('recursive',(ProgramStep('self',parse('FROM $r RETURN '+operation+'($p,$s,$v)')),),
                tuple(ProgramInput(n) for n in ('p','s','v')))


def test_literal_program_output_has_a_type():
    from rcql import bind,SourcePolicy
    r,_,_=native();p=Program('constant',(ProgramStep('read',parse('FROM $r RETURN 1')),))
    e=Executor(sources={'r':r},params={'p':p,'sources':{'r':bind('r',r,SourcePolicy.allow('*'))},'args':{}})
    q=parse('FROM $r RETURN PROGRAM_OUTPUT(PROGRAM_RUN($p,$sources,$args),0)')
    assert e.execute(q).values==(1,)
    e.execute(replace(q,explain=True))


def test_program_size_limit_applies_on_construction():
    from rcql.ast import Query,Parameter,Literal
    with pytest.raises(ValueError,match="size limit"):
        Program("huge",(ProgramStep("read",Query(Parameter("r"),(Literal("x"*(4*1024*1024)),))),))
