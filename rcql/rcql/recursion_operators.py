"""Named and recursive relations through the existing typed execution path."""
from .execution_trace import current_binding, record_method


def _binding(source):
    from .binding import bind
    from .capabilities import SourcePolicy
    binding = current_binding() or bind("relation_source", source, SourcePolicy.allow("*"))
    if binding.value is not source:
        raise ValueError("the relation must use its actual selected source")
    binding.source.require("read")
    return binding


def name(source, operator, alias=None):
    from .name_relation import NameRelation
    return NameRelation.operator(operator, name=alias)


def name_bind(source, operation, argument, value):
    return operation.bind(argument, value)


def name_rebind(source, operation, argument, value):
    return operation.rebind(argument, value)


def name_alias(source, operation, name):
    return operation.named(name)


def name_port(source, operation, old, new):
    return operation.rename(old, new)


def name_chain(source, left, right, argument, name=None):
    return left.then(right, argument, name=name)


def name_modify(source, base, modifier, argument, name=None):
    return base.then(modifier, argument, name=name, modifier=True)


def name_apply(source, operation, arguments):
    return operation.apply(_binding(source), arguments)


def name_explain(source, operation, arguments):
    result = operation.explain(_binding(source), arguments).explain()
    return {"name": operation.name, "derivation": operation.declaration(), "plan": result}


def name_record(source, operation):
    return operation.to_record()


def name_read(source, record=None):
    from .name_relation import NameRelation
    return NameRelation.from_record(source if record is None else record)


def name_topology(source, operation):
    return operation.topology().to_record()


def name_iterate(source, operation, initial, steps, argument, parameters=(), limits=None):
    from .recursive_program import RecursionResult
    from .relation_runtime import runtime_scope, fingerprint
    from .source_context import field_references
    from .program_codec import digest
    from rexgraph.tensor_field import FieldSource
    if type(steps) is not int or steps < 0:
        raise ValueError("iteration requires a nonnegative integer horizon")
    parameters = dict(parameters)
    if argument not in operation.inputs or argument in parameters:
        raise ValueError("iteration requires one explicitly unbound state port")
    binding = _binding(source)
    current = initial
    history = [(0, operation.name, initial)]
    events = [{"id": 0, "name": operation.name, "parent": None, "branch": "initial", "arguments": digest(fingerprint(initial))}]
    with runtime_scope(binding, limits) as runtime:
        operation.explain(binding, {**parameters, argument: current})
        for i in range(steps):
            runtime.call(1)
            current = operation.apply(binding, {**parameters, argument: current})
            history.append((i+1, operation.name, current))
            events.append({"id": i+1, "name": operation.name, "parent": i, "branch": "iterate", "arguments": digest(fingerprint(current))})
        own = FieldSource(source, binding.ref.record_id, binding.ref.record_version)
        refs = (own, *field_references(tuple(v for _, _, v in history)))
        result = RecursionResult(current, operation.coefficient_digest, operation.name, runtime.calls,
            runtime.evaluations, tuple(events), tuple(history), tuple({v.coefficient_digest: v for v in refs}.values()))
    record_method("finite-relation-iteration", definition=operation.coefficient_digest, steps=steps)
    return result


def recursive_run(source, program, entry, arguments, limits=None, history=True, memoize=True):
    if type(history) is not bool or type(memoize) is not bool:
        raise TypeError("history and memoization selections must be boolean")
    result = program.execute(_binding(source), entry, arguments, limits=limits, history=history, memoize=memoize)
    record_method("typed-recursive-relations", definition=program.coefficient_digest,
                  calls=result.calls, evaluations=result.evaluations, completed=True)
    return result


def recursive_explain(source, program, entry, arguments):
    from .planning import _plain_type
    from .native_plan import plain
    result = program.explain(_binding(source), entry, arguments)
    result["result_type"] = _plain_type(result["result_type"])
    return plain(result)


def _result_source(source, result):
    from rexgraph.tensor_field import FieldSource
    binding = _binding(source)
    own = FieldSource(source, binding.ref.record_id, binding.ref.record_version)
    if not any(own.matches(ref) for ref in result.dependencies):
        raise ValueError("recursive result belongs to another selected source version")
    for ref in result.dependencies:
        ref.check()


def recursive_value(source, result):
    _result_source(source, result)
    return result.value


def recursive_fields(source, result, definition=None, order="completion", port=None):
    _result_source(source, result)
    return result.fields(definition, order, port)


def recursive_trace(source, result):
    _result_source(source, result)
    return result.topology().to_record()


def recursive_history(source, result):
    _result_source(source, result)
    return tuple({"invocation": i, "name": name, "value": value} for i, name, value in result.history)


def recursive_record(source, program):
    return program.to_record()


def recursive_read(source, record=None):
    from .recursive_program import RecursiveProgram
    return RecursiveProgram.from_record(source if record is None else record)


def recursive_topology(source, program):
    return program.topology().to_record()


def recursive_result_record(source, result):
    _result_source(source, result)
    return result.to_record()


def recursive_result_read(source, record, contributors=()):
    from rexgraph.tensor_field import FieldSource
    from .recursive_program import RecursionResult
    from .binding import Binding
    from .capabilities import SourcePolicy
    binding = _binding(source)
    refs = [FieldSource(source, binding.ref.record_id, binding.ref.record_version)]
    policies = [binding.source.policy]
    for item in contributors:
        if not isinstance(item, Binding):
            raise TypeError("recursive result contributors require explicit Bindings")
        item.source.require("read")
        refs.append(FieldSource(item.value, item.ref.record_id, item.ref.record_version, item.ref.state_digest))
        policies.append(item.source.policy)
    if SourcePolicy.intersection(*policies).digest != binding.source.policy.digest:
        raise PermissionError("use the contributor intersection policy for the result")
    return RecursionResult.from_record(record, refs)


def feedback_record(source, system):
    from .recursive_state import feedback_record as store
    _feedback_source(source, system)
    return store(system)


def feedback_read(source, record, contributors=()):
    from .recursive_state import restore_feedback
    from .binding import Binding
    from .capabilities import SourcePolicy
    from rexgraph.tensor_field import FieldSource
    binding = _binding(source)
    refs = [FieldSource(source, binding.ref.record_id, binding.ref.record_version)]
    policies = [binding.source.policy]
    for item in contributors:
        if not isinstance(item, Binding):
            raise TypeError("feedback contributors require explicit Bindings")
        item.source.require("read")
        refs.append(FieldSource(item.value, item.ref.record_id, item.ref.record_version, item.ref.state_digest))
        policies.append(item.source.policy)
    if SourcePolicy.intersection(*policies).digest != binding.source.policy.digest:
        raise PermissionError("feedback restoration requires the contributor intersection policy")
    return restore_feedback(record, refs)


def feedback_complete(source, system):
    _feedback_source(source, system)
    result = system.complete()
    record_method("exact-affine-feedback", declaration=system.coefficient_digest, dimension=result.dimension)
    return result


def feedback_select(source, system, variable):
    _feedback_source(source, system)
    return system.selection(variable)


def feedback_iterate(source, system, initial, steps):
    from rexgraph.tensor_field import TensorChannels
    _feedback_source(source, system)
    values = system.iterate(initial, steps)
    return TensorChannels(tuple("step/"+str(i) for i in range(len(values))), values,
                          system.coefficient_digest, (system.source,))


def _feedback_source(source, system):
    from .section_operators import _reference
    binding = _binding(source)
    if system.source.source is not source:
        raise ValueError("feedback requires its actual selected native source")
    _reference(binding, system.source)
    system.check_state()


def install(register):
    from .recursion_contracts import ARGUMENTS
    for key in ARGUMENTS:
        register(key)(globals()[key.lower()])
