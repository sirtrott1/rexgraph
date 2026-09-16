"""Core attachment and rank adapters."""
from .execution_trace import record_method


def fill(source, cycle):
    from .operators import _typed_value
    cycle = _typed_value(source, cycle, operator="FILL", variance="chain", grade=1)
    result = source.fill_cycle(cycle.numpy())
    record_method("core-exact-cycle-attachment", chain_condition="exact-zero")
    return result


def harmonic_shadow(source):
    result = dict(source.harmonic_shadow)
    record_method("core-exact-boundary-shadow-ranks")
    return result


def void(source, region):
    from rexgraph.void_state import void_state
    from .operators import _typed_cells
    _typed_cells(source, region, operator="VOID")
    result = void_state(region)
    record_method("core-exact-triangle-void", potential=result.n_potential, missing=result.n_voids)
    return result


def validate_relations(source, candidates, grade=2):
    from rexgraph.relation_validation import validate_relations as core_validate
    result = core_validate(source, candidates, grade=grade)
    record_method("core-exact-candidate-boundaries", grade=grade, attachment=False)
    return result


def install(register):
    register("FILL")(fill)
    register("HARMONIC_SHADOW")(harmonic_shadow)
    register("VOID")(void)
    register("VALIDATE_RELATIONS")(validate_relations)
