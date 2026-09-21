"""Native adapters for coordinate actions and temporal channel measurements."""
from .execution_trace import record_method


def coordinate_apply(source, action, field):
    from rexgraph.type_accession import CoordinateField
    if field.source is not source or field.space != action.domain:
        raise ValueError("coordinate field must match the bound source and action domain")
    values = action.apply(field.values)
    record_method("core-exact-coordinate-action", declaration=action.coefficient_digest)
    return CoordinateField(source, field.grade, action.codomain, values, field.variance)


def _operation_delta(source, operation, old, new, metric, names):
    if old.source is not source or old.space != operation.old.domain or new.space != operation.new.domain:
        raise ValueError("temporal input fields must match the declared source and coordinates")
    if old.grade != new.grade or old.variance != new.variance:
        raise ValueError("temporal fields require matching grade and variance")
    result = operation.field_delta(old.values, new.values, names=names)
    record_method("core-exact-temporal-operation", declaration=operation.coefficient_digest,
                  channels=result.names)
    return result.as_record(metric)


def operation_delta(source, operation, old, new, metric=None):
    return _operation_delta(source, operation, old, new, metric, ("operation", "field"))


def injection_delta(source, operation, old, new, metric=None):
    return _operation_delta(source, operation, old, new, metric, ("injection", "amplitude"))


def word_delta(source, word, field, metric=None):
    if field.source is not source or field.space != word.domain:
        raise ValueError("temporal word field must match its declared input")
    result = word.delta(field.values)
    record_method("core-exact-temporal-word", declaration=word.coefficient_digest, channels=result.names)
    return result.as_record(metric)


def kernel_moments(source, kernel, field):
    if field.source is not source or field.space != kernel.domain:
        raise ValueError("moment field must match the kernel input")
    result = kernel.evaluate(field.values)
    record_method("core-exact-moment-kernel", declaration=kernel.coefficient_digest, channels=result.names)
    return result.as_record()


def install(register):
    for name, function in (("COORDINATE_APPLY", coordinate_apply), ("OPERATION_DELTA", operation_delta),
                           ("INJECTION_DELTA", injection_delta), ("WORD_DELTA", word_delta),
                           ("KERNEL_MOMENTS", kernel_moments)):
        register(name)(function)
