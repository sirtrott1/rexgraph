"""Core artifact readers and byte operations, with no file or network writes."""
from dataclasses import asdict
import hashlib

from .execution_trace import current_artifact_services, current_policy, record_method


def _bytes(value):
    if not isinstance(value, bytes):
        raise TypeError("artifact payloads and signatures require explicit bytes")
    return value


def _lineage(value):
    from rexgraph.io.commit import CommitLink
    from rexgraph.io.partition_state import PartitionState, RexPartition
    from rexgraph.io.transition import TransitionCommit
    if isinstance(value, RexPartition):
        value.check_state()
        value = value.state
    if not isinstance(value, (PartitionState, CommitLink, TransitionCommit)):
        raise TypeError("lineage requires a Core partition, commit link or transition")
    return value


def lineage(source, value):
    value = _lineage(value)
    record_method("core-lineage-manifest", object_type=type(value).__name__)
    return {"object_type": type(value).__name__, "digest": value.digest,
            "manifest": value.manifest(), "signature_verified": False}


def manifest(source, artifact):
    from rexgraph.io.export import ExportManifest
    from rexgraph.io.security import ENVELOPE_MAGIC, envelope_info
    from rexgraph.io.transport import MAGIC, inspect
    value = artifact
    if isinstance(value, bytes):
        if value.startswith(MAGIC):
            info = inspect(value)
        elif value.startswith(ENVELOPE_MAGIC):
            info = envelope_info(value)
        else:
            raise ValueError("MANIFEST requires a native transport package or encrypted envelope")
        result = {**asdict(info), "verification": "header-only"}
    elif isinstance(value, ExportManifest):
        result = {**asdict(value), "digest": value.digest, "object_type": "ExportManifest"}
    else:
        result = lineage(source, value)
    record_method("core-artifact-public-manifest")
    return result


def hash_value(source, value=None, kind="state"):
    from rexgraph.io.catalog import object_digest
    from rexgraph.io.manifest import manifest_digest
    if kind == "state":
        from rexgraph.graph import RexGraph, TemporalRex
        value = source if value is None else value
        if not isinstance(value, (RexGraph, TemporalRex)):
            raise TypeError("state HASH requires a native RexGraph or TemporalRex")
        result = object_digest(value)
    elif kind == "bytes":
        if not isinstance(value, bytes):
            raise TypeError("bytes HASH requires bytes")
        result = hashlib.sha256(value).hexdigest()
    elif kind == "manifest":
        from .artifact_contracts import json_mapping
        result = manifest_digest(json_mapping(value))
    elif kind == "lineage":
        result = _lineage(value).digest
    else:
        raise ValueError("HASH kind must be state, bytes, manifest or lineage")
    record_method("core-artifact-digest", kind=kind)
    return result


def transport(source, payload, object_type, metadata=None):
    from rexgraph.io.transport import pack
    from .artifact_contracts import json_mapping
    if not isinstance(payload, bytes):
        raise TypeError("TRANSPORT requires explicit payload bytes")
    result = pack(payload, object_type=object_type, metadata=None if metadata is None else json_mapping(metadata))
    record_method("core-framed-transport", object_type=object_type, payload_bytes=len(payload))
    return result


def show_capabilities(source):
    policy = current_policy()
    return {"permissions": tuple(sorted(policy.permissions)),
            "record_fields": None if policy.record_fields is None else tuple(sorted(policy.record_fields)),
            "policy_digest": policy.digest,
            "scope": "bound-source", "provider_availability": "not-disclosed"}


def encrypt(source, payload, key_id, object_type="bytes"):
    from rexgraph.io.security import encrypt_bytes
    result = encrypt_bytes(_bytes(payload), key_id=key_id, object_type=object_type,
                           keys=current_artifact_services().provider("keys"))
    record_method("core-authenticated-envelope", object_type=object_type)
    return result


def decrypt(source, artifact):
    from rexgraph.io.security import decrypt_bytes
    result = decrypt_bytes(_bytes(artifact), keys=current_artifact_services().provider("keys"))
    record_method("core-authenticated-envelope-open")
    return result


def export_parquet(source, columns, partition):
    from dataclasses import asdict
    from rexgraph.io.export import export_parquet as core_export
    from rexgraph.io.partition_state import RexPartition
    if not isinstance(partition, RexPartition):
        raise TypeError("EXPORT_PARQUET requires a RexPartition")
    partition.check_state()
    payload, manifest = core_export(columns, partition_digest=partition.digest)
    record_method("core-parquet-export", partition_digest=partition.digest)
    return {"payload": payload, "manifest": asdict(manifest), "digest": manifest.digest}


def sign(source, payload, signer_id):
    result = current_artifact_services().provider("signers", signer_id).sign(_bytes(payload))
    if not isinstance(result, bytes):
        raise TypeError("artifact signers must return signature bytes")
    record_method("core-provider-signature")
    return result


def verify_signature(source, payload, signature, signer_id):
    result = current_artifact_services().provider("verifiers", signer_id).verify(_bytes(payload), _bytes(signature))
    if not isinstance(result, bool):
        raise TypeError("artifact verifiers must return a boolean")
    record_method("core-provider-signature-verification")
    return result


def pseudonymize(source, value, scope, key_id):
    from rexgraph.io.privacy import scoped_pseudonym
    result = scoped_pseudonym(value, scope=scope, key_id=key_id,
                              keys=current_artifact_services().provider("identity_keys"))
    record_method("core-scoped-pseudonym")
    return result


ADAPTERS = {"HASH": hash_value, "MANIFEST": manifest, "LINEAGE": lineage,
            "EXPORT_PARQUET": export_parquet,
            "TRANSPORT": transport, "SHOW_CAPABILITIES": show_capabilities,
            "ENCRYPT": encrypt, "DECRYPT": decrypt, "SIGN": sign,
            "VERIFY_SIGNATURE": verify_signature, "PSEUDONYMIZE": pseudonymize}


def install(register):
    for name, fn in ADAPTERS.items():
        register(name)(fn)
