from rcql.ast import Call, Literal
from rcql.optimizer import optimize_expr


def test_chain_rewrite():
    expr = Call("BOUNDARY", (
        Literal(1),
        Call("BOUNDARY", (Literal(2), Literal("face"))),
    ))
    got, rewrites = optimize_expr(expr)
    assert got == Call("ZERO", (Literal(0),))
    assert len(rewrites) == 1
