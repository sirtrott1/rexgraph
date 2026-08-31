from rcql import Call, Literal, Parameter, call, query, source


def test_typed_builder_keeps_source_and_calls():
    q = query(source("main"), call("BETTI", 1), explain=True)
    assert q.source == Parameter("main")
    assert q.returns == (Call("BETTI", (Literal(1),)),)
    assert q.explain is True
