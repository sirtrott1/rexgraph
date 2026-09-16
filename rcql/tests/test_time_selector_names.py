"""Named clocks reuse the RCDB source reader, including its policy and identity."""
from dataclasses import replace

import pytest

from rexgraph.graph import RexGraph
from rcql import BoundSource, Executor, SourcePolicy, parse, query, source, source_call, call


@pytest.fixture
def store(tmp_path):
    import rcdb
    value = rcdb.open_store(f"rex://{tmp_path / 'db'}")
    value.put("r", RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]]]), valid_from=10, valid_to=20)
    value.put("r", RexGraph.from_cells([3, [[0, 1]]]), valid_from=20)
    yield value
    value.close()


@pytest.mark.parametrize("name,original", [("VALID_AT", "RCDB_VALID_AT"), ("TRANSACTION_AT", "RCDB_AS_OF")])
@pytest.mark.parametrize("explain", [False, True])
def test_same_selected_state_policy_and_provenance(store, name, original, explain):
    when = 12 if name == "VALID_AT" else store.history("r")[0].tx_from
    policy = SourcePolicy.allow("identity", "read")
    engine = Executor(sources={"db": BoundSource(store, policy)})
    forms = [query(source_call(n, source=source("db"), record_id="r", time=when),
                   call("BETTI", 1), call("STATE_HASH")) for n in (name, original)]
    actual, expected = [engine.execute(replace(form, explain=explain)) for form in forms]
    assert actual.values == expected.values
    if not explain:
        assert actual.values[0] == 1
    text = f'FROM {name}(time={when},record_id="r",source=$db) RETURN BETTI(1)'
    assert engine.execute(parse(text)).values == (1,)


@pytest.mark.parametrize("name", ["VALID_AT", "TRANSACTION_AT"])
@pytest.mark.parametrize("when", [True, "12", float("nan"), float("inf")])
def test_bad_clocks_refused_before_reader(store, name, when, monkeypatch):
    monkeypatch.setattr(type(store), "read_record", lambda *a, **kw: pytest.fail("reader reached"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"db": store}).execute(query(source_call(name, source("db"), "r", when), call("GRADE")))


@pytest.mark.parametrize("name", ["VALID_AT", "TRANSACTION_AT"])
def test_no_clock_or_lookup_capability_is_invented(store, name, monkeypatch):
    monkeypatch.setattr(type(store), "read_record", lambda *a, **kw: pytest.fail("reader reached"))
    engine = Executor(sources={"db": BoundSource(store, SourcePolicy.allow("read"))})
    with pytest.raises(PermissionError):
        engine.execute(parse(f'FROM {name}($db,"r",12) RETURN GRADE()'))
    with pytest.raises((TypeError, KeyError)):
        Executor(sources={"r": RexGraph.from_cells([2, [[0, 1]]])}).execute(
            parse(f'FROM {name}($r,"r",12) RETURN GRADE()'))
    with pytest.raises(KeyError):
        Executor(sources={"db": store}).execute(parse(f'FROM $db RETURN {name}($db,"r",12)'))


def test_valid_intervals_and_missing_record(store):
    e = Executor(sources={"db": store})
    assert e.execute(parse('FROM VALID_AT($db,"r",20) RETURN BETTI(1)')).values == (0,)
    for text in ('VALID_AT($db,"r",9)', 'TRANSACTION_AT($db,"absent",0)'):
        with pytest.raises(KeyError):
            e.execute(parse(f'FROM {text} RETURN GRADE()'))
