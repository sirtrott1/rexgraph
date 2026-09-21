"""Declared types for molecular and sampled process observations."""
from .types import RCType, ValueKind, Domain, Exactness, PredicateResult

ARGUMENTS = {
    "MOLECULAR_INFO": (("selection",), ()),
    "MOLECULAR_FIELD": (("selection", "reading", "native"), ("bond_order", False)),
    "MOLECULAR_CONFORMER": (("selection", "name"), ()),
    "CONFORMATION_FIELD": (("field", "reading", "selections"), ()),
    "CONFORMATION_DIRECTION": (("field", "direction", "reading", "selections", "parameter", "unit", "contributors"), ((),)),
    "MOLECULAR_DELTA": (("selection", "destination", "new_selection", "alignment"), (None,)),
    "PROCESS_COMPARE": (("left", "right", "times", "axis", "unit", "contributors"), ((),)),
    "FACTOR_CONTRAST": (("old", "new", "step", "parameter", "unit", "contributors"), ((),)),
}


def refine(typed, children, context):
    if not context.native:
        raise TypeError("molecular operations require a selected native source")
    name = typed.operator
    values = tuple(context.known_value(c) for c in children)
    labels, defaults = ARGUMENTS[name]
    missing = len(labels)-len(values)
    if missing:
        values += defaults[-missing:]
    if name in {"MOLECULAR_INFO", "MOLECULAR_FIELD", "MOLECULAR_CONFORMER", "MOLECULAR_DELTA"} and isinstance(values[0], str):
        from rexgraph.molecular_field import MolecularView
        view = MolecularView.from_source(context.binding.value, values[0],
                                         record_id=context.binding.ref.record_id, version=context.binding.ref.record_version)
        if name == "MOLECULAR_CONFORMER" and isinstance(values[1], str) and values[1] not in view.record["conformers"]:
            raise ValueError("selected conformer is not recorded")
    if name == "MOLECULAR_DELTA":
        from .binding import Binding
        if isinstance(values[1], Binding):
            values[1].source.require("read")
            from .capabilities import SourcePolicy
            if SourcePolicy.intersection(context.binding.source.policy, values[1].source.policy).digest != context.binding.source.policy.digest:
                raise PermissionError("comparison requires the source policy intersection")
    if name == "CONFORMATION_FIELD":
        from rexgraph.tensor_field import TensorField
        from rexgraph.molecular_field import _geometry_contract
        if isinstance(values[0], TensorField) and isinstance(values[1], str) and isinstance(values[2], (tuple, list)):
            _geometry_contract(*values)
    return typed.result.with_(source=context.binding.ref), [
        PredicateResult("molecular_source", "verified", "selected native source and declared coordinates"),
        PredicateResult("molecular_observation", "deferred", "exact supplied geometry or explicit molecular mapping")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    v = ValueKind
    kinds = {
        "MOLECULAR_INFO": (str,), "MOLECULAR_FIELD": (str, str, bool), "MOLECULAR_CONFORMER": (str, str),
        "CONFORMATION_FIELD": (v.TENSOR_FIELD, str, (tuple, list)),
        "CONFORMATION_DIRECTION": (v.TENSOR_FIELD, v.TENSOR_FIELD, str, (tuple, list), str, str, None),
        "MOLECULAR_DELTA": (str, None, str, None),
        "PROCESS_COMPARE": (None, None, None, str, str, None),
        "FACTOR_CONTRAST": (v.TENSOR_FIELD, v.TENSOR_FIELD, None, str, str, None),
    }
    channels = {"MOLECULAR_DELTA", "PROCESS_COMPARE", "FACTOR_CONTRAST"}
    for name, (labels, defaults) in ARGUMENTS.items():
        patterns = []
        for i, (label, kind) in enumerate(zip(labels, kinds[name], strict=True)):
            optional = i >= len(labels)-len(defaults)
            if isinstance(kind, ValueKind):
                patterns.append(TypePattern(label, kind=kind, optional=optional))
            else:
                patterns.append(TypePattern(label, literal=kind, optional=optional))
        kind = v.RECORD if name == "MOLECULAR_INFO" else v.TENSOR_CHANNELS if name in channels else v.TENSOR_FIELD
        result = RCType(kind.value, kind=kind, domain=Domain.METADATA if kind == v.RECORD else Domain.RATIONAL,
                        exactness=Exactness.STRUCTURAL if kind == v.RECORD else Exactness.RATIONAL)
        register(OperatorSignature(name=name, source_kind=v.REX, inputs=tuple(patterns), result=result,
            implementation_key="rexgraph.molecular_field."+name.lower(), memoizable=True,
            preconditions=("explicit source identities and molecular coordinates", "supplied geometry and declared factor steps")))
