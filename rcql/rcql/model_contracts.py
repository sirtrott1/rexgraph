"""Type and capability declarations for persistent local models."""
from .types import RCType, ValueKind, Domain, Exactness, PredicateResult

ARGUMENTS = {
    "MODEL_INIT": (("adapter", "configuration", "axes", "optimizer", "seed"), ("native_field", None, None, None, 0)),
    "MODEL_STATE": (("index",), (0,)),
    "MODEL_AT": (("time",), ()),
    "MODEL_BATCH": (("state", "targets", "observed", "inputs", "contributors"), (None, ())),
    "MODEL_INFER": (("state", "inputs"), (None,)),
    "MODEL_TRAIN": (("state", "batch", "steps"), (1,)),
    "MODEL_TRANSPORT": (("state", "destination", "mapping", "optimizer"), ("reset",)),
    "MODEL_RECORD": (("state",), ()),
    "MODEL_HISTORY": (("states", "sources", "times", "time_axis", "time_unit"), ("model_observation", "step")),
    "MODEL_FIELD": (("output",), ()),
    "MODEL_VALUES": (("output",), ()),
    "MODEL_INPUT": (("value", "dtype"), ("float64",)),
    "MODEL_INFO": (("state",), ()),
    "MODEL_CERTIFY": (("state", "inputs", "candidate", "recorded_binary"), (False,)),
}


def _type(kind, arithmetic=None):
    return RCType(kind.value, kind=kind,
                  domain=Domain.RATIONAL if arithmetic == "rational" else Domain.REAL if arithmetic == "approximate" else Domain.METADATA,
                  exactness=Exactness.RATIONAL if arithmetic == "rational" else Exactness.APPROXIMATE if arithmetic == "approximate" else Exactness.STRUCTURAL)


def model_literal(binding, value):
    from rexgraph.model_state import ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput
    for cls, kind in ((ModelState, ValueKind.MODEL_STATE), (ModelOutput, ValueKind.MODEL_OUTPUT),
                      (ModelInput, ValueKind.MODEL_INPUT), (ModelBatch, ValueKind.MODEL_BATCH), (ModelTimeline, ValueKind.MODEL_TIMELINE)):
        if isinstance(value, cls):
            value.check_state()
            arithmetic = getattr(value, "arithmetic", None) if isinstance(value, ModelOutput) else None
            return _type(kind, arithmetic).with_(source=binding.ref, declaration_digest=value.coefficient_digest)
    return None


def refine(typed, children, context):
    if not context.native: raise TypeError("model operations require a selected native record")
    from rexgraph.model_state import ModelState, ModelBatch, ModelOutput
    from .model_operators import authorize
    name = typed.operator
    labels, defaults = ARGUMENTS[name]
    values = tuple(context.known_value(c) for c in children)
    missing = len(labels) - len(values)
    if missing: values += defaults[-missing:]
    first = values[0] if values else None
    arithmetic = None
    if isinstance(first, ModelState):
        authorize(context.binding.value, first, context.binding)
        arithmetic = first.arithmetic
    if name == "MODEL_INIT" and isinstance(first, str):
        if first not in {"native_field", "online_flow"}:
            from rexgraph.nn.lifecycle import _adapter
            _adapter(first)
        arithmetic = "rational" if first == "native_field" else "approximate"
    if name == "MODEL_TRAIN":
        if not context.binding.source.policy.permits("train"):
            raise PermissionError("model training requires train permission")
        if isinstance(first, ModelState) and first.adapter == "native_field":
            raise TypeError("the native exact action is evaluated rather than trained")
        if isinstance(first, ModelState) and isinstance(values[1], ModelBatch):
            batch = values[1]; batch.check_state()
            if batch.space != first.space or not batch.source.matches(first.source):
                raise ValueError("training batch source or coordinates disagree")
        if isinstance(values[2], int) and (isinstance(values[2], bool) or values[2] < 1):
            raise ValueError("training steps must be positive")
    if name == "MODEL_FIELD" and isinstance(first, ModelOutput) and first.arithmetic != "rational":
        raise TypeError("numerical output cannot be labeled an exact field")
    if name == "MODEL_FIELD": arithmetic = "rational"
    if name == "MODEL_VALUES" and isinstance(first, ModelOutput): arithmetic = first.arithmetic
    result = typed.result
    if arithmetic is None and name in {"MODEL_INFER", "MODEL_VALUES"}:
        result = result.with_(exactness=None)
    if arithmetic is not None and name in {"MODEL_INFER", "MODEL_FIELD", "MODEL_VALUES"}:
        result = _type(result.kind, arithmetic)
    return result.with_(source=context.binding.ref), [
        PredicateResult("model_identity", "verified" if isinstance(first, ModelState) else "deferred", "native source and complete model digest"),
        PredicateResult("model_execution", "deferred", "declared exact action or registered numerical model")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    v = ValueKind
    inputs = {
        "MODEL_INIT": (str, None, None, None, int), "MODEL_STATE": (int,), "MODEL_AT": (None,),
        "MODEL_BATCH": (v.MODEL_STATE, None, None, None, None),
        "MODEL_INFER": (v.MODEL_STATE, None), "MODEL_TRAIN": (v.MODEL_STATE, v.MODEL_BATCH, int),
        "MODEL_TRANSPORT": (v.MODEL_STATE, None, v.COORDINATE_MAP, str),
        "MODEL_HISTORY": (None, None, None, str, str), "MODEL_RECORD": (v.MODEL_STATE,), "MODEL_FIELD": (v.MODEL_OUTPUT,),
        "MODEL_CERTIFY": (v.MODEL_STATE, v.TENSOR_FIELD, None, bool), "MODEL_INPUT": (None,str), "MODEL_VALUES": (v.MODEL_OUTPUT,), "MODEL_INFO": (v.MODEL_STATE,)}
    outputs = {"MODEL_INIT": v.MODEL_STATE, "MODEL_STATE": v.MODEL_STATE, "MODEL_AT": v.MODEL_STATE,
        "MODEL_BATCH": v.MODEL_BATCH, "MODEL_INFER": v.MODEL_OUTPUT, "MODEL_TRAIN": v.MODEL_STATE,
        "MODEL_TRANSPORT": v.MODEL_STATE, "MODEL_RECORD": v.REX, "MODEL_FIELD": v.TENSOR_FIELD,
        "MODEL_VALUES": v.MODEL_VALUES, "MODEL_INFO": v.RECORD, "MODEL_HISTORY": v.REX, "MODEL_INPUT": v.MODEL_INPUT, "MODEL_CERTIFY": v.RECORD}
    for name, (labels, defaults) in ARGUMENTS.items():
        patterns = []
        for i, (label, kind) in enumerate(zip(labels, inputs[name], strict=True)):
            optional = i >= len(labels) - len(defaults)
            patterns.append(TypePattern(label, kind=kind, optional=optional,
                            literal=type(None) if optional else None) if isinstance(kind, ValueKind)
                            else TypePattern(label, literal=kind, optional=optional))
        requires = frozenset({"read", "train"}) if name == "MODEL_TRAIN" else frozenset({"read"})
        register(OperatorSignature(name=name, source_kind=v.REX, inputs=tuple(patterns), result=_type(outputs[name]),
            implementation_key="rexgraph.model_runtime." + name.lower(), requires=requires,
            memoizable=True,
            preconditions=("native model and observations name their complete source and coordinates",
                           "publication remains an explicit RCDB mutation")))
