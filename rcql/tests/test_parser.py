from rcql import Call, Literal, Parameter, parse


def test_parse_basic_query():
    query = parse("FROM $r RETURN BETTI(1), RANK(1)")
    assert query.source == Parameter("r")
    assert query.returns[0] == Call("BETTI", (Literal(1),))
    assert query.returns[1].name == "RANK"


def test_parse_explain():
    query = parse("EXPLAIN FROM REX(\"main\") RETURN BETTI(1)")
    assert query.explain is True


def test_true_and_false_parse_as_literals_not_calls():
    """A bare name becomes a nullary Call, which made TRUE a missing operator.

    Several signatures take an `exact: bool`. Without a boolean literal the flag could
    only be supplied by binding a parameter, so the exact path was reachable but not
    writable, and `QUADRANCE(..., TRUE)` failed as an unknown operator rather than
    running exactly.
    """
    from rcql.ast import Literal

    for text, want in (("TRUE", True), ("true", True), ("True", True),
                       ("FALSE", False), ("false", False)):
        arg = parse(f"FROM $r RETURN QUADRANCE(BOUNDARY(CELL(1,0)), {text})").returns[0].args[1]
        assert isinstance(arg, Literal), f"{text} parsed as {type(arg).__name__}"
        assert arg.value is want, f"{text} carried {arg.value!r}"


def test_a_boolean_name_followed_by_a_parenthesis_is_still_a_call():
    """The literal must not shadow an operator that might one day carry either name."""
    from rcql.ast import Call

    expr = parse("FROM $r RETURN TRUE()").returns[0]
    assert isinstance(expr, Call) and expr.name == "TRUE"
