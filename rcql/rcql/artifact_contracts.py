"""Explicit byte artifacts and Core lineage, without implicit serialization."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {
    "HASH": (("value", "kind"), (None, "state")),
    "MANIFEST": (("artifact",), ()), "LINEAGE": (("value",), ()),
    "TRANSPORT": (("payload", "object_type", "metadata"), (None,)),
    "SHOW_CAPABILITIES": ((), ()),
    "ENCRYPT": (("payload", "key_id", "object_type"), ("bytes",)),
    "DECRYPT": (("artifact",), ()), "SIGN": (("payload", "signer_id"), ()),
    "VERIFY_SIGNATURE": (("payload", "signature", "signer_id"), ()),
    "PSEUDONYMIZE": (("value", "scope", "key_id"), ()),
    "EXPORT_PARQUET": (("columns", "partition"), ()),
}
OBSERVABLE = frozenset({"ENCRYPT", "DECRYPT", "SIGN", "VERIFY_SIGNATURE", "PSEUDONYMIZE"})


def json_mapping(value):
    """Validate JSON metadata without coercing integer keys into text identities."""
    from collections.abc import Mapping
    from rexgraph.io.manifest import canonical_json
    if not isinstance(value, Mapping):
        raise TypeError("artifact metadata requires a canonical JSON mapping")
    value = dict(value)
    pending, seen = [value], set()
    while pending:
        node = pending.pop()
        if not isinstance(node, (dict, list, tuple)) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            if any(not isinstance(key, str) for key in node):
                raise TypeError("artifact JSON mapping keys must be strings")
            pending.extend(node.values())
        else:
            pending.extend(node)
    canonical_json(value)
    return value


def _nonempty(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} requires a nonempty string")


def refine(typed, children, context):
    from collections.abc import Mapping
    from rexgraph.graph import RexGraph, TemporalRex
    from rexgraph.io.commit import CommitLink
    from rexgraph.io.export import ExportManifest
    from rexgraph.io.partition_state import PartitionState, RexPartition
    from rexgraph.io.transition import TransitionCommit
    lineage_types = (PartitionState, RexPartition, CommitLink, TransitionCommit)
    args, name = typed.args, typed.operator
    if name == "EXPORT_PARQUET":
        data, partition = args
        if not (isinstance(data, Mapping) or isinstance(data, RCType) and data.kind == ValueKind.RECORD):
            raise TypeError("EXPORT_PARQUET requires explicit column arrays")
        if isinstance(data, Mapping):
            from rexgraph.io.export import _arrays
            _arrays(data)
        if not (isinstance(partition, RexPartition) or
                isinstance(partition, RCType) and partition.kind == ValueKind.REX_PARTITION):
            raise TypeError("EXPORT_PARQUET requires a declared RexPartition lineage")
    elif name == "HASH":
        value = args[0] if args else None
        kind = args[1] if len(args) > 1 else "state"
        if kind not in {"state", "bytes", "manifest", "lineage"}:
            raise ValueError("HASH kind must be state, bytes, manifest or lineage")
        if value is None and kind == "state":
            value = context.binding.value
        expected = {"state": (RexGraph, TemporalRex), "bytes": (bytes,),
                    "manifest": (Mapping,), "lineage": lineage_types}[kind]
        kinds = {"state": {ValueKind.REX, ValueKind.TEMPORAL_REX}, "bytes": {ValueKind.ARTIFACT_BYTES},
                 "manifest": {ValueKind.RECORD}, "lineage": {ValueKind.REX_PARTITION}}[kind]
        if not (isinstance(value, expected) or isinstance(value, RCType) and value.kind in kinds):
            raise TypeError(f"HASH value does not have the declared {kind} contract")
        if kind == "manifest" and isinstance(value, Mapping):
            json_mapping(value)
    elif name in {"LINEAGE", "MANIFEST"}:
        value = args[0]
        expected = lineage_types + ((bytes, ExportManifest) if name == "MANIFEST" else ())
        kinds = {ValueKind.REX_PARTITION} | ({ValueKind.ARTIFACT_BYTES} if name == "MANIFEST" else set())
        if not (isinstance(value, expected) or isinstance(value, RCType) and value.kind in kinds):
            raise TypeError(f"{name} requires a declared Core artifact or lineage value")
    elif name == "TRANSPORT":
        _nonempty(args[1], "object_type")
        if len(args) > 2 and args[2] is not None:
            metadata = args[2]
            if not (isinstance(metadata, Mapping) or isinstance(metadata, RCType) and metadata.kind == ValueKind.RECORD):
                raise TypeError("TRANSPORT metadata requires a canonical JSON mapping")
            if isinstance(metadata, Mapping):
                json_mapping(metadata)
    elif name in {"SIGN", "VERIFY_SIGNATURE"}:
        _nonempty(args[-1], "signer_id")
    elif name == "ENCRYPT":
        _nonempty(args[1], "key_id")
        if len(args) > 2:
            _nonempty(args[2], "object_type")
    elif name == "PSEUDONYMIZE":
        _nonempty(args[1], "scope")
        _nonempty(args[2], "key_id")
    return [PredicateResult("explicit_artifact_contract", "verified",
        "bytes are not implicitly encoded; manifests are public descriptions, not authentication proofs"),
        PredicateResult("artifact_providers", "deferred" if name in OBSERVABLE else "not-required",
        "configured Core providers resolve only at execution; no file or network transfer is implied")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    byte = lambda name: TypePattern(name, kind=ValueKind.ARTIFACT_BYTES, literal=bytes)
    text = lambda name, **kw: TypePattern(name, literal=str, **kw)
    patterns = {
        "HASH": (TypePattern("value", optional=True), text("kind", optional=True)),
        "MANIFEST": (TypePattern("artifact"),), "LINEAGE": (TypePattern("value"),),
        "TRANSPORT": (byte("payload"), text("object_type"), TypePattern("metadata", optional=True)),
        "SHOW_CAPABILITIES": (), "ENCRYPT": (byte("payload"), text("key_id"), text("object_type", optional=True)),
        "DECRYPT": (byte("artifact"),), "SIGN": (byte("payload"), text("signer_id")),
        "VERIFY_SIGNATURE": (byte("payload"), byte("signature"), text("signer_id")),
        "PSEUDONYMIZE": (text("value"), text("scope"), text("key_id")),
        "EXPORT_PARQUET": (TypePattern("columns"), TypePattern("partition", kind=ValueKind.REX_PARTITION)),
    }
    for name, inputs in patterns.items():
        kind = (ValueKind.ARTIFACT_BYTES if name in {"TRANSPORT", "ENCRYPT", "DECRYPT", "SIGN"}
                else ValueKind.BOOLEAN if name == "VERIFY_SIGNATURE"
                else ValueKind.RECORD if name in {"MANIFEST", "LINEAGE", "SHOW_CAPABILITIES", "EXPORT_PARQUET"}
                else ValueKind.TEXT)
        required = ({"read", "identity"} if name in {"HASH", "MANIFEST", "LINEAGE", "EXPORT_PARQUET"}
                    else {"read", "security"} if name in {"ENCRYPT", "DECRYPT", "SIGN", "VERIFY_SIGNATURE"}
                    else {"read", "identity", "security"} if name == "PSEUDONYMIZE"
                    else set() if name == "SHOW_CAPABILITIES" else {"read"})
        register(OperatorSignature(name=name, source_kind=ValueKind.UNKNOWN, inputs=inputs,
            result=RCType("ParquetExport" if name == "EXPORT_PARQUET" else kind.value,
                          kind=kind, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL),
            implementation_key="rcql.artifact_operators." + ("hash_value" if name == "HASH" else name.lower()),
            requires=frozenset(required), memoizable=name not in OBSERVABLE,
            preconditions=("explicit Core artifact semantics, no implicit file writes or remote operations",
                           "configured providers are not query values; provider operations are not memoized")))
