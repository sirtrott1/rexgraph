"""Exact core certificates, without an RCQL matrix implementation."""
from .execution_trace import record_method


def adjugate(source, operator):
    from rexgraph.adjugate_operator import AdjugateOperator
    from .calculus_operators import _action
    result = AdjugateOperator(_action(source, operator))
    record_method("exact-adjugate-handle", coefficient_method="streamed-power-traces",
                  coefficient_actions=dict(result.parameters)["coefficient_actions"])
    return result


def homotopy(source, left, right, witness):
    from rexgraph.chain_map import ChainHomotopy, ChainMap, GradedMap
    for value in (left, right):
        declaration = value.declaration if isinstance(value, ChainMap) else value
        if not isinstance(declaration, GradedMap) or declaration.domain.source is not source:
            raise TypeError("HOMOTOPY requires maps bound to its source")
    result = ChainHomotopy(left, right, witness)
    record_method("exact-sparse-chain-homotopy", coefficient_digest=result.coefficient_digest,
                  shapes=result.shapes, residuals=tuple(map(str, result.residuals)), verified=True)
    return result


ADAPTERS = {"ADJUGATE": adjugate, "HOMOTOPY": homotopy}


def install(register):
    for name, fn in ADAPTERS.items():
        register(name)(fn)
