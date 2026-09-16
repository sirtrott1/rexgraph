"""Thin query adapters for Core rational operator actions."""
from .execution_trace import record_method


def cayley(source, generator, parameter, tol=1e-10, maxiter=1000):
    from rexgraph.rational_operator import cayley as build
    if generator.source is not source:
        raise ValueError("Cayley requires its bound source")
    result = build(generator, parameter, tol=float(tol), maxiter=maxiter)
    record_method("native-factored-cayley", parameter=str(parameter))
    return result


def complex_structure(source, generator, scale=None):
    from rexgraph.rational_operator import complex_structure as build
    if generator.source is not source:
        raise ValueError("complex structure requires its bound source")
    result = build(generator, scale)
    record_method("exact-streamed-complex-certificate", scale=str(dict(result.parameters)["scale"]))
    return result


def rational_rotation(source, structure, a, b, c):
    from rexgraph.rational_operator import rational_rotation as build
    if structure.source is not source:
        raise ValueError("rotation requires its bound source")
    result = build(structure, a, b, c)
    record_method("native-rational-rotation", kernel_policy="identity")
    return result


def resolvent_group(source, operators, parameters, word=(), tol=1e-10, maxiter=1000):
    from rexgraph.rational_operator import ResolventGroup
    if any(getattr(op, "source", None) is not source for op in operators):
        raise ValueError("resolvent generators require the bound source")
    result = ResolventGroup(operators, parameters, word, tol=tol, maxiter=maxiter)
    record_method("native-resolvent-group", generators=len(operators), word=result.word)
    return result


def install(register):
    for name, adapter in (("CAYLEY", cayley), ("COMPLEX_STRUCTURE", complex_structure),
                          ("RATIONAL_ROTATION", rational_rotation), ("RESOLVENT_GROUP", resolvent_group)):
        register(name)(adapter)
