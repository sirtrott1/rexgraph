"""Native molecular observations and declared process comparisons."""
from .binding import Binding
from .capabilities import SourcePolicy
from .execution_trace import current_binding, record_method


def _reference(source, binding=None):
    from rexgraph.tensor_field import FieldSource
    binding = current_binding() if binding is None else binding
    if binding is None:
        return FieldSource(source)
    if binding.value is not source:
        raise ValueError("molecular operation requires its selected native source")
    binding.source.require("read")
    return FieldSource(source, binding.ref.record_id, binding.ref.record_version, binding.ref.state_digest)


def _destination(binding):
    if not isinstance(binding, Binding):
        raise TypeError("destination requires a policy bearing native Binding")
    binding.source.require("read")
    return _reference(binding.value, binding)


def authorize_fields(source, fields, contributors=()):
    from rexgraph.tensor_field import TensorField
    binding = current_binding()
    refs = [_reference(source)]
    for contributor in contributors:
        refs.append(_destination(contributor))
    if binding is not None:
        policy = SourcePolicy.intersection(binding.source.policy, *(c.source.policy for c in contributors))
        if policy.digest != binding.source.policy.digest:
            raise PermissionError("bind process queries under the contributor intersection policy")
    for field in fields:
        if not isinstance(field, TensorField):
            raise TypeError("process query inputs must be tensor fields")
        field.check_state()
        dependencies = (*field.dependencies, *((field.source,) if field.source else ()))
        for expected in dependencies:
            if not any(expected.matches(actual) and expected.source is actual.source for actual in refs):
                raise ValueError("supply every process field contributor as its selected Binding")


def _view(source, selection):
    from rexgraph.molecular_field import MolecularView
    return MolecularView.from_source(_reference(source), selection)


def molecular_info(source, selection):
    result = _view(source, selection).info()
    record_method("native-molecular-declaration", source_digest=result["source"]["state_digest"])
    return result


def molecular_field(source, selection, reading="bond_order", native=False):
    result = _view(source, selection).field(reading, native=native)
    record_method("exact-molecular-field", reading=reading, declaration_digest=result.coefficient_digest)
    return result


def molecular_conformer(source, selection, name):
    result = _view(source, selection).conformation(name)
    record_method("exact-supplied-conformer", declaration_digest=result.coefficient_digest)
    return result


def conformation_field(source, field, reading, selections):
    from rexgraph.molecular_field import conformation_field as evaluate
    authorize_fields(source, (field,))
    result = evaluate(field, reading, selections)
    record_method("rational-conformation-observation", reading=reading, declaration_digest=result.coefficient_digest)
    return result


def conformation_direction(source, field, direction, reading, selections, parameter, unit, contributors=()):
    from rexgraph.molecular_field import conformation_direction as evaluate
    authorize_fields(source, (field, direction), contributors)
    result = evaluate(field, direction, reading, selections, parameter=parameter, parameter_unit=unit)
    record_method("rational-conformation-derivative", declaration_digest=result.coefficient_digest)
    return result


def molecular_delta(source, selection, destination, new_selection, alignment=None):
    from rexgraph.molecular_field import MolecularView, molecular_changes
    old = _view(source, selection)
    new = MolecularView.from_source(_destination(destination), new_selection)
    binding = current_binding()
    if binding is not None and SourcePolicy.intersection(binding.source.policy, destination.source.policy).digest != binding.source.policy.digest:
        raise PermissionError("bind molecular comparison under both source policies")
    result = molecular_changes(old, new, alignment)
    record_method("exact-molecular-change", declaration_digest=result.declaration_digest,
                  endpoint_sources=tuple(s.as_record() for s in result.endpoint_sources))
    return result


def process_compare(source, left, right, times, axis, unit, contributors=()):
    from rexgraph.process_field import trajectory_comparison
    left, right = tuple(left), tuple(right)
    authorize_fields(source, (*left, *right), contributors)
    result = trajectory_comparison(left, right, times, axis=axis, unit=unit)
    record_method("exact-sampled-process-comparison", declaration_digest=result.declaration_digest)
    return result


def factor_contrast(source, old, new, step, parameter, unit, contributors=()):
    from rexgraph.process_field import factor_contrast as evaluate
    authorize_fields(source, (old, new), contributors)
    result = evaluate(old, new, step, parameter=parameter, unit=unit)
    record_method("exact-declared-factor-secant", declaration_digest=result.declaration_digest)
    return result


def install(register):
    for name, fn in (("MOLECULAR_INFO", molecular_info), ("MOLECULAR_FIELD", molecular_field),
                     ("MOLECULAR_CONFORMER", molecular_conformer), ("CONFORMATION_FIELD", conformation_field),
                     ("CONFORMATION_DIRECTION", conformation_direction), ("MOLECULAR_DELTA", molecular_delta),
                     ("PROCESS_COMPARE", process_compare), ("FACTOR_CONTRAST", factor_contrast)):
        register(name)(fn)
