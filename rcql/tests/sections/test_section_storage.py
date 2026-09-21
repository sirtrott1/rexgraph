import json
import numpy as np
import pytest
from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSheaf
from rexgraph.io.section_state import pack_section, unpack_section
from rexgraph.io.rex_state import to_state, from_state, encode_tensors
from rcdb import open_store
from rcql import Executor,parse


def native():
    r=RexGraph.from_hypergraph([0,2],[0,1]);s=ExactSheaf(r,grade=0,stalk_dim=2)
    c=s.section_system();o,b=c.pins({('0','0'):2**100+1})
    return c,c.complete(o,b)


@pytest.mark.parametrize('kind',['recipe','family'])
def test_exact_codec_roundtrip(kind):
    c,f=native();value=c.recipe if kind=='recipe' else f
    raw=pack_section(value)
    restored=unpack_section(raw)
    assert restored.coefficient_digest==value.coefficient_digest
    encoded={k:v.copy() for k,v in raw.items()};spec=encode_tensors(encoded)
    # The canonical coefficient codec is exercised by native storage below.
    assert any(s['c']=='exact' for s in spec.values())


@pytest.mark.parametrize('kind',['recipe','family'])
def test_native_record_roundtrip(kind):
    c,f=native();value=c.recipe if kind=='recipe' else f
    record=RexGraph.from_hypergraph([0,1],[0])
    record.attach_metadata(1,0,'section',value)
    state=to_state(record)
    assert state.header['format_version']==6
    restored=from_state(state).get_metadata(1,0,'section')
    assert restored.coefficient_digest==value.coefficient_digest
    if kind=='recipe':
        bound=restored.bind(c.source)
        assert bound.complete(*bound.pins({('0','0'):2**100+1})).coefficient_digest==f.coefficient_digest
    else:
        assert restored.dimension==1
        assert restored.particular.values.tolist()==[2**100+1,0,2**100+1,0]


def test_nested_recipe_selects_version6():
    c,_=native();child=RexGraph.from_hypergraph([0,1],[0]);child.attach_metadata(1,0,'equations',c.recipe)
    parent=RexGraph.from_hypergraph([0,1],[0]);parent.attach_metadata(1,0,'child',child)
    state=to_state(parent)
    assert state.header['format_version']==6
    restored=from_state(state).get_metadata(1,0,'child').get_metadata(1,0,'equations')
    assert restored.coefficient_digest==c.coefficient_digest


@pytest.mark.parametrize('backend',['memory','rex','file','sql'])
def test_store_reopen_and_query(backend,tmp_path):
    c,f=native()
    record=RexGraph.from_hypergraph([0,1],[0]);record.attach_metadata(1,0,'recipe',c.recipe);record.attach_metadata(1,0,'family',f)
    uri={'memory':'memory://','rex':str(tmp_path/'history.rexdb'),'file':'file://'+str(tmp_path/'files'),
         'sql':'sqlite:///'+str(tmp_path/'history.db')}[backend]
    store=open_store(uri)
    try:
        saved=store.commit_mutation('section_result',record,expected_version=0)
        version=saved.version
        if backend!='memory':
            store.close();store=open_store(uri)
        restored=store.read_record('section_result',version=version).value
        recipe=restored.get_metadata(1,0,'recipe')
        out=Executor(sources={'g':c.source.source},params={'r':recipe}).execute(parse('FROM $g RETURN SECTION_RESTORE($r)')).values[0]
        assert out.coefficient_digest==c.coefficient_digest
        family=restored.get_metadata(1,0,'family')
        bound=Executor(sources={'g':c.source.source},params={'f':family}).execute(parse('FROM $g RETURN SECTION_RESTORE($f)')).values[0]
        assert bound.coefficient_digest==f.coefficient_digest
        assert store.verify_commits('section_result')
    finally:store.close()


def test_mutation_carries_complete_recipe():
    c,_=native();a=RexGraph.from_hypergraph([0,1],[0]);b=RexGraph.from_hypergraph([0,1],[0]);b.attach_metadata(1,0,'recipe',c.recipe)
    from rexgraph.io.mutation import prepare_mutation,apply_mutation,mutation_to_bytes,mutation_from_bytes
    package=mutation_from_bytes(mutation_to_bytes(prepare_mutation(a,b,tx_time=1)))
    output=apply_mutation(package,previous=a)
    assert output.get_metadata(1,0,'recipe').coefficient_digest==c.coefficient_digest


@pytest.mark.parametrize('which',['spec','map','family'])
def test_corrupt_payload_refused(which):
    c,f=native();raw=pack_section(f if which=='family' else c.recipe)
    if which=='spec':
        spec=json.loads(raw['spec'].tobytes());spec['cells'][0]='forged';raw['spec']=np.frombuffer(json.dumps(spec).encode(),dtype=np.uint8)
    elif which=='map':
        key=next(k for k in raw if k.endswith('/values'));raw[key]=np.array(raw[key],copy=True);raw[key][0]+=1
    else:
        key=next(k for k in raw if k.startswith('particular/') and k.endswith('values'));raw[key]=np.array(raw[key],copy=True);raw[key][0]+=1
    with pytest.raises(ValueError):unpack_section(raw)


def test_live_actions_are_not_generic_metadata():
    c,f=native();record=RexGraph.from_hypergraph([0,1],[0])
    record.attach_metadata(1,0,'action',c)
    with pytest.raises(TypeError):to_state(record)


def test_source_change_rejected_on_restore():
    c,_=native();recipe=c.recipe.detached();c.source.source.attach_metadata(1,0,'changed','yes')
    with pytest.raises(ValueError):recipe.bind(c.source)
