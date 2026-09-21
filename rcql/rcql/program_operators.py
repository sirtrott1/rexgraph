"""Explicit tensor observations and exact section equivalence."""
from .execution_trace import current_binding, record_method


def tensor_select(source, field, axis, key):
    field.check_state()
    result = field.select(axis, key)
    record_method("exact-retained-axis-selection", axis=axis, key=key)
    return result


def tensor_contract(source, field, axis, weights):
    field.check_state()
    result = field.contract_axis(axis, weights)
    record_method("exact-explicit-axis-contraction", axis=axis)
    return result


def tensor_diagonal(source, field, first, second, name):
    from rexgraph.process_field import diagonal_axes
    result = diagonal_axes(field, first, second, name)
    record_method("exact-matching-axis-selection", first=first, second=second, name=name)
    return result


def sample_rates(source, field):
    from rexgraph.process_field import sample_rates as rates
    result = rates(field)
    record_method("exact-interval-secants")
    return result


def section_equivalent(source, family, left, right):
    from .section_operators import authorize_family
    from .readout_equivalence import ReadoutEquivalence
    authorize_family(family, current_binding())
    certificate = ReadoutEquivalence.check(family, left, right)
    record_method("exact-section-readout-equivalence", certificate=certificate.digest,
                  equivalent=certificate.equivalent)
    return certificate.as_record()


def install(register):
    from .program_contracts import ARGUMENTS
    for name in ARGUMENTS:
        register(name)(globals()[name.lower()])


def program_read(source, record=None):
    from .program import Program
    result = Program.from_record(source if record is None else record)
    record_method("finite-native-program-read", program_digest=result.coefficient_digest)
    return result


def program_record(source, program):
    record_method("finite-native-program-record", program_digest=program.coefficient_digest)
    return program.to_record()


def _program_executor(sources, parameters):
    from .binding import Binding
    from .capabilities import BoundSource, SourcePolicy
    from .executor import Executor
    from collections.abc import Mapping
    if not isinstance(sources, Mapping) or not isinstance(parameters, Mapping):
        raise TypeError("program sources and parameters must be explicit mappings")
    bound = {}
    policies = []
    for name, value in sources.items():
        if not isinstance(name, str) or not isinstance(value, Binding):
            raise TypeError("each program source needs a named policy bearing Binding")
        value.source.require("read")
        bound[name] = BoundSource(value.value, value.source.policy, ref=value.ref, temporal=value.temporal)
        policies.append(value.source.policy)
    binding = current_binding()
    if binding is not None and SourcePolicy.intersection(binding.source.policy, *policies).digest != binding.source.policy.digest:
        raise PermissionError("bind the program call under the contributor intersection policy")
    from .execution_trace import current_evidence
    return Executor(sources=bound, params=dict(parameters), evidence=current_evidence())


def program_run(source, program, sources, parameters):
    result = _program_executor(sources, parameters).execute_program(program)
    record_method("finite-native-program-execution", program_digest=program.coefficient_digest,
                  steps=tuple((name, value.execution) for name, value in result.steps))
    return result


def program_explain(source, program, sources, parameters):
    result = _program_executor(sources, parameters).execute_program(program, explain=True)
    record_method("finite-native-program-explanation", program_digest=program.coefficient_digest)
    from .native_plan import plain
    from .planning import _plain_type
    result["output_types"] = tuple(_plain_type(value) for value in result["output_types"])
    return plain(result)


def program_output(source, result, index):
    if type(index) is not int or not 0 <= index < len(result.values):
        raise ValueError("program output index is outside the exported results")
    record_method("finite-program-result-selection", index=index, program_digest=result.program_digest)
    return result.values[index]


def program_topology(source, program, sources, parameters):
    topology = program.topology(_program_executor(sources, parameters))
    record_method("native-program-topology", program_digest=program.coefficient_digest,
                  topology_digest=topology.coefficient_digest)
    return topology.to_record()


def program_assembly(source, name, fragments, coefficients, inputs=(), links=(), outputs=()):
    from .program_family import ProgramAssembly
    return ProgramAssembly.create(name, fragments, coefficients, inputs, links, outputs)


def program_glue(source, assembly, system, observation=None, observed=None):
    from .section_operators import section_complete
    from .program_family import ProgramFamily
    assembly.coefficient_map(system)
    family = section_complete(source, system, observation, observed)
    return ProgramFamily(assembly, system.recipe, family)


def program_family_observe(source, family):
    from .section_operators import authorize_family
    authorize_family(family.family, current_binding())
    return family.observation()


def program_family_compile(source, family, sources, parameters):
    from .section_operators import authorize_family
    authorize_family(family.family, current_binding())
    candidate = family.compile(sources, parameters)
    record_method("exact-determined-program-assembly", family=family.coefficient_digest,
                  target=candidate.coefficient_digest, target_evaluated=False)
    return candidate


def program_family_record(source, family):
    from .section_operators import authorize_family
    authorize_family(family.family, current_binding())
    return family.to_record()


def program_family_read(source, record, contributors=()):
    from .program_family import ProgramFamily
    from .section_operators import validate_restore, authorize_family
    result = ProgramFamily.from_record(record)
    binding = current_binding()
    if binding is None:
        from .binding import bind
        from .capabilities import SourcePolicy
        binding = bind("program_family_source", source, SourcePolicy.allow("read"))
    reference, dependencies = validate_restore(result.recipe, contributors, binding)
    system = result.recipe.bind(reference, dependencies)
    family = result.family.bind(reference, dependencies)
    object.__setattr__(family, "_query_contributors", tuple(contributors))
    authorize_family(family, binding)
    return ProgramFamily(result.assembly, system.recipe, family)


def program_assembly_record(source, assembly):
    return assembly.to_record()


def program_assembly_read(source, record):
    from .program_family import ProgramAssembly
    return ProgramAssembly.from_record(record)


def program_compare(source, old, new, matches):
    from .program_evolution import ProgramEvolution
    from .plan_topology import PlanTopology
    from .relation_topology import RelationTopology
    def topology(value):
        if isinstance(value, (PlanTopology, RelationTopology)):
            return value
        cls = PlanTopology if "rcql_plan_schema" in getattr(value, "_agent_meta", {}) else RelationTopology
        return cls.from_record(value)
    return ProgramEvolution.compare(topology(old), topology(new), matches)


def program_evolution_info(source, evolution):
    return evolution.explain()


def program_boundary_change(source, evolution, grade):
    return evolution.boundary_change(grade)


def program_evolution_record(source, evolution):
    return evolution.to_record()


def program_evolution_read(source, record):
    from .program_evolution import ProgramEvolution
    return ProgramEvolution.from_record(record)
