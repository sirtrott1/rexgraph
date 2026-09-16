"""Read a conversation history or preview the existing numerical path gate."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {"TURN_FIELD": (("start", "stop"), (0, None)),
             "PATH_CHANGE": (("text",), ())}


def refine(typed, children, context):
    if typed.operator == "TURN_FIELD":
        context.binding.value.validate_interval(*typed.args)
    return [PredicateResult("conversation_read", "verified",
        "bound Core TurnField; capture and candidate preview do not append turns"),
        PredicateResult("path_gate", "deferred" if typed.operator == "PATH_CHANGE" else "not-applicable",
        "existing numerical Malaugh entropy fence, not exact semantic path membership")]


def turn_field(source, start=0, stop=None):
    from .execution_trace import record_method
    result = source.snapshot(start, stop)
    record_method("core-conversation-snapshot", snapshots=result.T)
    return result


def path_change(source, text):
    from .execution_trace import record_method
    result = source.preview(text)
    record_method("core-conversation-preview", baseline_turns=result["baseline_turns"],
                  status=result["status"], gate="malaugh-numerical-fence")
    return result


def install(register):
    from .signatures import OperatorSignature, TypePattern
    required = frozenset({"read", "identity", "history", "agent_read"})
    register(OperatorSignature(name="TURN_FIELD", source_kind=ValueKind.TURN_FIELD,
        inputs=(TypePattern("start", literal=int, optional=True),
                TypePattern("stop", literal=(int, type(None)), optional=True)),
        result=RCType("TemporalRex", kind=ValueKind.TEMPORAL_REX,
                      domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
        implementation_key="rexgraph.flow.TurnField.snapshot", requires=required,
        preconditions=("half open interval of accepted turns; snapshots retain full prefixes",
                       "stable turn IDs and original term order; no Agent import or external lookup")))
    register(OperatorSignature(name="PATH_CHANGE", source_kind=ValueKind.TURN_FIELD,
        inputs=(TypePattern("text", literal=str),),
        result=RCType("PathChange", kind=ValueKind.RECORD, domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL),
        implementation_key="rexgraph.flow.TurnField.preview", requires=required,
        preconditions=("candidate text tokenized by the existing Core TEXT profile",
                       "copied numerical gate baseline; no source mutation or exact semantic certificate")))
