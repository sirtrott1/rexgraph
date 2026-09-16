from rcql.ast import Call, Literal
from rcql.optimizer import optimize_expr


def test_unbound_syntax_cannot_certify_a_chain_rewrite():
    expr = Call("BOUNDARY", (
        Literal(1),
        Call("BOUNDARY", (Literal(2), Literal("face"))),
    ))
    got, rewrites = optimize_expr(expr)
    assert got == expr
    assert rewrites == []
