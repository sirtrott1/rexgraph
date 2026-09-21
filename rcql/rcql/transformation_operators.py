"""Explicit transformation stages through existing name execution."""
from .execution_trace import record_method, current_binding, current_evidence


def validate_readout(binding, family, left, right, certificate, side):
    from .section_operators import authorize_family
    from .readout_equivalence import ReadoutEquivalence
    if side not in ("left", "right"):
        raise ValueError("readout side must be left or right")
    authorize_family(family, binding)
    evidence = current_evidence()
    if evidence is not None:
        evidence.validate_binding(binding)
        evidence.validate_values((family, left, right))
    return ReadoutEquivalence.verify_bytes(certificate, family, left, right)


def section_certified_observe(source, family, left, right, certificate, side="right"):
    from .section_operators import _selected
    _selected(source, family.source)
    proof = validate_readout(current_binding(), family, left, right, certificate, side)
    record_method("certified-section-readout", certificate=proof.digest, side=side)
    return family.observe(left if side == "left" else right)


def transform_section(source, family, left, right):
    from .section_operators import authorize_family, _selected
    from .program_transformation import ProgramTransformation
    _selected(source, family.source)
    authorize_family(family, current_binding())
    return ProgramTransformation.section(family, left, right)


def transform_source(source, transformation):
    return transformation.original()


def transform_specialize(source, operation, bindings):
    from .program_transformation import ProgramTransformation
    return ProgramTransformation.specialize(operation, bindings)


def transform_compose(source, operation, following, ports):
    from .program_transformation import ProgramTransformation
    return ProgramTransformation.compose(operation, following, ports)


def transform_program(source, program, rule, arguments):
    from .program_transformation import ProgramTransformation
    return ProgramTransformation.program(program, rule, arguments)


def transform_program_compile(source, transformation, program, sources, parameters):
    candidate = transformation.compile_program(program, sources, parameters)
    record_method("validated-finite-program-transformation", declaration=transformation.coefficient_digest,
                  target=candidate.coefficient_digest, target_evaluated=False)
    return candidate


def transform_program_topology(source, transformation, program, sources, parameters):
    from .program_operators import _program_executor
    candidate = transformation.compile_program(program, sources, parameters)
    return candidate.topology(_program_executor(sources, parameters)).to_record()


def transform_name(source, operation, rule, arguments):
    from .program_transformation import ProgramTransformation
    return ProgramTransformation.name(operation, rule, arguments)


def transform_verify(source, transformation, operation):
    result = transformation.verify(operation)
    record_method("verified-operation-transformation", declaration=transformation.coefficient_digest,
                  rule=result["rule_version"], scope=result["scope"])
    return result


def transform_compile(source, transformation, operation, arguments):
    from .recursion_operators import _binding
    result = transformation.compile(_binding(source), operation, arguments)
    record_method("validated-name-transformation", declaration=transformation.coefficient_digest,
                  target=result.coefficient_digest, target_evaluated=False)
    return result


def transform_record(source, transformation):
    return transformation.to_record()


def transform_read(source, record=None):
    from .program_transformation import ProgramTransformation
    return ProgramTransformation.from_record(source if record is None else record)


def transform_topology(source, transformation):
    return transformation.topology().to_record()


def install(register):
    from .transformation_contracts import ARGUMENTS
    for name in ARGUMENTS:
        register(name)(globals()[name.lower()])
