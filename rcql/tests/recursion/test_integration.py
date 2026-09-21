from fractions import Fraction as Q
from dataclasses import replace
from contextlib import closing
import pytest
from rcql import Executor,parse,query,call,param,NameRelation,RecursiveProgram,RecursionResult,QueryCache
from rcql.recursion_contracts import ARGUMENTS
from .query_cases import query_case,feedback
from .helpers import source,binding,counter,integer,minus


@pytest.mark.parametrize('name',tuple(ARGUMENTS))
def test_query_and_explanation(name):
    r=source();args=query_case(name,r)
    e=Executor(sources={'r':r},params={'p'+str(i):v for i,v in enumerate(args)})
    expr=call(name,*(param('p'+str(i)) for i in range(len(args))))
    q=query(param('r'),expr)
    answer=e.execute(q)
    explain=e.execute(replace(q,explain=True))
    assert answer.values and not explain.execution
    from rcql.operators import get_operator
    direct=get_operator(name).fn(r,*args)
    from rcql.executor import value_exactness
    assert value_exactness(direct)==answer.exactness[0]
    assert explain.values[0]['returns'][0]['result']['exactness']==answer.exactness[0].value


def store_format(record,path,kind):
    if kind=='rcbd':
        from rexgraph.io.bundle import save_rcbd,load_rcbd
        save_rcbd(path,record)
        return load_rcbd(path)
    from rexgraph.io.safetensors_bridge import rex_to_safetensors,load_safetensors
    rex_to_safetensors(record,path)
    return load_safetensors(path)['object']


@pytest.mark.parametrize('kind',['rcbd','safetensors'])
@pytest.mark.parametrize('value',['name','program','result','feedback'])
def test_portable_roundtrip(tmp_path,kind,value):
    r=source(); b=binding(r)
    if value=='name':
        original=NameRelation.operator('NATIVE_RESPONSE').bind('parameter',Q(2,7))
        record=original.to_record(); restore=NameRelation.from_record
    elif value=='program':
        original=counter();record=original.to_record();restore=RecursiveProgram.from_record
    elif value=='result':
        original=counter().execute(b,'count',[5]);record=original.to_record()
        restore=lambda rec:RecursionResult.from_record(rec,original.dependencies)
    else:
        from rcql.recursive_state import feedback_record,restore_feedback
        original=feedback(r);record=feedback_record(original)
        restore=lambda rec:restore_feedback(rec,(original.source,))
    loaded=store_format(record,tmp_path/('object.'+kind),kind)
    restored=restore(loaded)
    assert restored.coefficient_digest==original.coefficient_digest
    if value=='feedback':assert restored.complete().particular.values.item()==2


def test_program_as_stored_rcdb_record(tmp_path):
    from rcdb import open_store
    p=counter()
    with closing(open_store('rex://'+str(tmp_path/'db'))) as db:
        db.commit_mutation('source',source(),expected_version=0,tx_time=1)
        db.commit_mutation('definition',p.to_record(),expected_version=0,tx_time=2)
        definition=db.read_record('definition').value
        loaded=RecursiveProgram.from_record(definition)
        q=parse('FROM RCDB_VERSION($db,"source",1) LET result=RECURSIVE_RUN($p,"count",[6]) RETURN RECURSIVE_VALUE(result),RECURSIVE_RESULT_RECORD(result)')
        out=Executor(sources={'db':db},params={'p':loaded}).execute(q)
        assert out.values[0]==6
        db.commit_mutation('answer',out.values[1],expected_version=0,tx_time=3)
        assert db.verify_commits('answer')
    with closing(open_store('rex://'+str(tmp_path/'db'))) as db:
        record=db.read_record('answer').value
        got=Executor(sources={'db':db},params={'record':record}).execute(parse('FROM RCDB_VERSION($db,"source",1) RETURN RECURSIVE_VALUE(RECURSIVE_RESULT_READ($record))'))
        assert got.values==(6,)


def test_cached_completed_recursion(tmp_path):
    from rcdb import open_store
    r=source();p=counter();e=Executor(sources={'r':r},params={'p':p})
    q=parse('FROM $r RETURN RECURSIVE_RUN($p,"count",[5])')
    with closing(open_store('rex://'+str(tmp_path/'cache'))) as store:
        cache=QueryCache(store)
        first=e.execute_cached(q,cache);second=e.execute_cached(q,cache)
        assert first.values[0].value==second.values[0].value==5
        assert second.execution[0]['operator']=='RESULT_REUSE'
        assert first.values[0].coefficient_digest==second.values[0].coefficient_digest


def test_incomplete_recursion_not_published(tmp_path):
    from rcdb import open_store
    from rcql import RecursionLimitError
    r=source();e=Executor(sources={'r':r},params={'p':counter(),'limits':{'calls':2}})
    with closing(open_store('rex://'+str(tmp_path/'cache'))) as store:
        cache=QueryCache(store)
        with pytest.raises(RecursionLimitError):e.execute_cached(parse('FROM $r RETURN RECURSIVE_RUN($p,"count",[5],$limits)'),cache)
        assert not store.list()


def test_result_digest_includes_history_and_counts():
    result=counter().execute(binding(source()),'count',[3])
    changed=replace(result,history=tuple((i,n,999 if i==1 else v) for i,n,v in result.history))
    assert changed.coefficient_digest!=result.coefficient_digest
    assert replace(result,evaluations=result.evaluations+1).coefficient_digest!=result.coefficient_digest


def test_tampered_result_rejected():
    from rcql.program_codec import loads,dumps
    result=counter().execute(binding(source()),'count',[3]);record=result.to_record()
    data=loads(record.get_metadata(1,0,'rcql_recursive_result'))
    data['evaluations']+=1
    record.attach_metadata(1,0,'rcql_recursive_result',dumps(data).decode())
    with pytest.raises(ValueError,match='identity'):RecursionResult.from_record(record,result.dependencies)


def test_feedback_source_tamper_rejected():
    from rcql.recursive_state import feedback_record,restore_feedback
    from rcql.program_codec import loads,dumps
    f=feedback(source());record=feedback_record(f)
    data=loads(record._agent_meta['rcql_feedback']);data['ports']=(('fake','role','x'),)
    record._agent_meta['rcql_feedback']=dumps(data).decode()
    with pytest.raises(ValueError):restore_feedback(record,(f.source,))


def test_wrong_result_source_rejected():
    r=source(); result=counter().execute(binding(r),'count',[1])
    from rexgraph.graph import RexGraph
    other=RexGraph.from_graph(sources=[0],targets=[1])
    with pytest.raises(ValueError):Executor(sources={'r':other},params={'v':result}).execute(parse('FROM $r RETURN RECURSIVE_VALUE($v)'))


def test_named_recursive_arguments():
    from rcql import RecursiveDefinition,recur
    from rcql.ast import Parameter,Literal,Comparison,Call,ListExpr
    n=Parameter('n');x=Parameter('x')
    p=RecursiveProgram('named',(RecursiveDefinition('f',(integer('n'),integer('x')),Comparison('==',n,Literal(0)),x,
       recur('f',x=Call('SUM',(ListExpr((x,Literal(2))),)),n=minus(n)),integer('out'),decreases='n'),))
    assert p.execute(binding(source()),'f',{'x':3,'n':4}).value==11
    assert p.topology().declaration['origin']==p.declaration()


def test_duplicate_recursive_arguments_refused():
    from rcql import RecursiveDefinition,recur
    from rcql.ast import Parameter,Literal
    n=Parameter('n')
    d=RecursiveDefinition('f',(integer('n'),),Literal(True),n,recur('f',n,n=n),integer('out'))
    with pytest.raises(TypeError):RecursiveProgram('bad',(d,))


def test_dynamic_name_modification_in_recursion():
    from rcql import RecursiveDefinition,ProgramInput,ValueKind,recur
    from rcql.ast import Parameter,Literal,Comparison,Call,ListExpr
    n=Parameter('n');op=Parameter('operation')
    # Each step explicitly replaces one captured port. The original name remains unchanged.
    base=Call('NAME_APPLY',(op,ListExpr(())))
    new=Call('NAME_REBIND',(op,Literal('values'),ListExpr((n,))))
    d=RecursiveDefinition('f',(integer('n'),ProgramInput('operation',ValueKind.NAME_RELATION.value)),
        Comparison('==',n,Literal(0)),base,recur('f',minus(n),new),integer('out'),decreases='n')
    p=RecursiveProgram('modify',(d,));name=NameRelation.operator('SUM').bind('values',[99]);before=name.coefficient_digest
    assert p.execute(binding(source()),'f',[3,name]).value==1
    assert name.coefficient_digest==before and dict(name.bound_values)['values']==[99]
