from rcql import Call, Literal, Parameter, parse


def test_parse_basic_query():
    query = parse("FROM $r RETURN BETTI(1), RANK(1)")
    assert query.source == Parameter("r")
    assert query.returns[0] == Call("BETTI", (Literal(1),))
    assert query.returns[1].name == "RANK"


def test_parse_explain():
    query = parse("EXPLAIN FROM REX(\"main\") RETURN BETTI(1)")
    assert query.explain is True
