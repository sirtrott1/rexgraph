"""Algebraic rewrites for RCQL expressions."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .ast import Alias, Call, Expr, ListExpr, Literal, Member, Parameter, Query, Reference


@dataclass(frozen=True)
class Rewrite:
    before: Expr
    after: Expr
    reason: str
    predicates: tuple = ()


def optimize_expr(expr: Expr, *, plan=None, chain_proof=None) -> tuple[Expr, list[Rewrite]]:
    """Rewrite only a validated, source bound plan with an exact chain proof.

    An unbound syntax tree cannot prove even that an operand is a chain. Blocks
    and unknown shapes remain actions, since ZERO currently constructs a vector.
    """
    if isinstance(expr, (Alias, Member)):
        child, rewrites = optimize_expr(expr.value, plan=None if plan is None else plan.children[0],
                                        chain_proof=chain_proof)
        return replace(expr, value=child), rewrites
    if isinstance(expr, ListExpr):
        items, rewrites = [], []
        for i, item in enumerate(expr.items):
            child, used = optimize_expr(item, plan=None if plan is None else plan.children[i],
                                        chain_proof=chain_proof)
            items.append(child)
            rewrites.extend(used)
        return replace(expr, items=tuple(items)), rewrites
    if not isinstance(expr, Call):
        return expr, []
    args = []
    rewrites = []
    for index, arg in enumerate(expr.args):
        child = plan.children[index] if plan is not None else None
        got, used = optimize_expr(arg, plan=child, chain_proof=chain_proof)
        args.append(got)
        rewrites.extend(used)
    current = Call(expr.name, tuple(args))
    if current.name == "BOUNDARY" and len(current.args) == 2:
        inner = current.args[1]
        if isinstance(inner, Call) and inner.name == "BOUNDARY" and len(inner.args) == 2:
            outer_grade = current.args[0]
            inner_grade = inner.args[0]
            from .types import Exactness, RCType
            operand = (plan.children[1].children[1].result
                       if plan is not None and len(plan.children[1].children) == 2 else None)
            if (isinstance(outer_grade, Literal) and type(outer_grade.value) is int
                    and isinstance(inner_grade, Literal) and type(inner_grade.value) is int
                    and outer_grade.value >= 1
                    and inner_grade.value == outer_grade.value + 1
                    and isinstance(operand, RCType)
                    and operand.exactness in {Exactness.INTEGER, Exactness.RATIONAL}
                    and operand.shape is not None and len(operand.shape.dims) == 1
                    and chain_proof is not None and chain_proof(plan.children[1].children[1])):
                after = Call("ZERO", (Literal(int(outer_grade.value) - 1), Literal("chain")))
                rewrites.append(Rewrite(current, after, "consecutive boundaries compose to zero", plan.predicates))
                return after, rewrites
    return current, rewrites


def optimize(query: Query, *, plan=None, parameters=None) -> tuple[Query, list[Rewrite]]:
    if plan is not None and plan.query is not query:
        raise ValueError('optimizer plan must describe the original query')
    def chain_proof(operand):
        from rexgraph.cochain import Chain
        while isinstance(operand.expr, Reference):
            operand = operand.children[0]
        expression = operand.expr
        value = (expression.value if isinstance(expression, Literal) else
                 (parameters or {}).get(expression.name) if isinstance(expression, Parameter) else None)
        # A declared RCType is enough to EXPLAIN, but is not an executable chain.
        # Never erase that distinction by optimizing a placeholder to a real zero.
        return isinstance(value, Chain) and value.source is plan.binding.value

    rewrites = []
    def optimize_fragment(expr, fragment):
        def proven(operand, root=fragment):
            pending = [root] if root is not None else []
            visited = set()
            while pending:
                item = pending.pop()
                if id(item) in visited:
                    continue
                visited.add(id(item))
                if any(p.name == 'chain_condition' and p.status == 'verified' for p in item.predicates):
                    return chain_proof(operand)
                pending.extend(item.children)
            return False
        got, used = optimize_expr(expr, plan=fragment,
                                 chain_proof=None if plan is None else proven)
        rewrites.extend(used)
        return got
    bindings = tuple(replace(item, value=optimize_fragment(
        item.value, None if plan is None else plan.bindings[index][1]))
        for index, item in enumerate(query.bindings))
    returns = tuple(optimize_fragment(expr, None if plan is None else plan.returns[index])
                    for index, expr in enumerate(query.returns))
    # Keep the ordered bindings and their single evaluation contract. Rewriting
    # through a reference is deliberately not substitution of its definition.
    return replace(query, returns=returns, bindings=bindings), rewrites
