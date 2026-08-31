"""Canonical, signed ``TemporalRex`` mutation packages."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .commit import COMMIT_VERSION, CommitLink
from .manifest import manifest_digest
from .rex_state import RexState, from_state, to_state, verify_state
from .temporal_state import FORMAT_VERSION as TEMPORAL_STATE_VERSION
from .temporal_state import (
    TemporalState,
    from_temporal_state,
    to_temporal_state,
    verify_temporal_state,
)
from .transition import TRANSITION_VERSION, TransitionCommit
from .transport import pack as pack_transport
from .transport import unpack as unpack_transport

MUTATION_VERSION = 2
READABLE_VERSIONS = (1, MUTATION_VERSION)
POLICY_VERSION = 1
_PARENT_UNSET = object()

__all__ = [
    "MUTATION_VERSION",
    "POLICY_VERSION",
    "READABLE_VERSIONS",
    "MutationPackage",
    "MutationPolicy",
    "mutation_from_bytes",
    "mutation_to_bytes",
    "prepare_mutation",
    "verify_mutation",
]


@dataclass(frozen=True)
class MutationPolicy:
    """Signature requirements applied to one committed relational mutation."""

    require_transition_signature: bool = False
    require_lineage_signature: bool = False
    allowed_signers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        """Return the canonical policy fields."""
        return {
            "allowed_signers": sorted(str(value) for value in self.allowed_signers),
            "require_lineage_signature": bool(self.require_lineage_signature),
            "require_transition_signature": bool(self.require_transition_signature),
            "version": POLICY_VERSION,
        }

    @property
    def digest(self) -> str:
        """Return the stable policy identity."""
        return manifest_digest({"object_type": "MutationPolicy", **self.manifest()})


@dataclass(frozen=True)
class MutationPackage:
    """One Rex state transition, its temporal state, and its lineage link."""

    transition: TransitionCommit
    link: CommitLink
    temporal_state: TemporalState
    resulting_state: RexState | None = None
    version: int = MUTATION_VERSION

    @property
    def digest(self) -> str:
        """Return the package inventory identity."""
        identity = {
            "link": self.link.digest,
            "object_type": "MutationPackage",
            "temporal_state": self.temporal_state.header.get("digest", ""),
            "transition": self.transition.digest,
            "version": int(self.version),
        }
        if self.version >= 2:
            identity["resulting_state"] = (
                ""
                if self.resulting_state is None
                else _resulting_state_digest(self.resulting_state)
            )
        return manifest_digest(identity)


def _legacy_delta_digest(state: TemporalState) -> str:
    """Reproduce the reference v1 tensor-only delta identity for migration."""
    from .rex_state import state_digest

    names = sorted(
        name
        for name in state.tensors
        if name.startswith("delta/") or name.startswith("face_delta/")
    )
    if not names:
        names = sorted(name for name in state.tensors if name.startswith("checkpoint/"))
    return state_digest(state.tensors, names)


def _is_general(rex: Any) -> bool:
    import numpy as np

    return bool(np.any(np.diff(np.asarray(rex._boundary_ptr, dtype=np.int64)) != 2))


def _check_signer(policy: MutationPolicy, signer: Any, role: str) -> None:
    if signer is None:
        return
    signer_id = getattr(signer, "signer_id", None)
    if not isinstance(signer_id, str) or not signer_id:
        raise ValueError(f"{role} signer must expose a nonempty signer_id")
    allowed = set(policy.allowed_signers)
    if allowed and signer_id not in allowed:
        raise PermissionError(f"{role} signer {signer_id!r} is not allowed by policy")


def prepare_mutation(
    previous,
    resulting,
    *,
    tx_time: float,
    actor: str = "",
    policy: MutationPolicy | None = None,
    parent_digest: str | None = None,
    transition_signer=None,
    lineage_signer=None,
) -> MutationPackage:
    """Build a v2 temporal mutation whose signed transition binds all state semantics."""
    from rexgraph.graph import RexGraph, TemporalRex

    from .catalog import object_digest, state_object_digest

    if not isinstance(resulting, RexGraph):
        raise TypeError("resulting must be a RexGraph")
    if isinstance(previous, TemporalRex) or isinstance(resulting, TemporalRex):
        raise TypeError("mutation packages expect RexGraph states, not TemporalRex histories")
    if previous is not None and not isinstance(previous, RexGraph):
        raise TypeError("previous must be a RexGraph or None")
    when = float(tx_time)
    if not math.isfinite(when):
        raise ValueError("tx_time must be finite")
    policy = policy or MutationPolicy()
    if not isinstance(policy, MutationPolicy):
        raise TypeError("policy must be a MutationPolicy")
    if policy.require_transition_signature and transition_signer is None:
        raise ValueError("policy requires a transition signer")
    if policy.require_lineage_signature and lineage_signer is None:
        raise ValueError("policy requires a lineage signer")
    _check_signer(policy, transition_signer, "transition")
    _check_signer(policy, lineage_signer, "lineage")

    directed = bool(resulting._directed)
    if previous is not None and bool(previous._directed) != directed:
        raise ValueError("one TemporalState cannot represent endpoints with different direction modes")
    general = _is_general(resulting) or (previous is not None and _is_general(previous))
    history = TemporalRex([], directed=directed, general=general)
    if previous is not None:
        history.append_snapshot(previous)
    history.append_snapshot(resulting)
    state = to_temporal_state(history)
    if not verify_temporal_state(state):  # pragma: no cover - writer invariant
        raise ValueError("could not produce a verified TemporalState")

    resulting_state = to_state(resulting)
    transition = TransitionCommit(
        "" if previous is None else object_digest(previous),
        str(state.header["digest"]),
        state_object_digest(resulting_state),
        when,
        str(actor),
        policy.digest,
    )
    if transition_signer is not None:
        transition = transition.signed(transition_signer)
    link = CommitLink(transition.digest, parent_digest)
    if lineage_signer is not None:
        link = link.signed(lineage_signer)
    return MutationPackage(transition, link, state, resulting_state)


def _signature_valid(
    value: TransitionCommit | CommitLink,
    *,
    required: bool,
    allowed: set[str],
    verifiers: dict[str, Any],
) -> bool:
    signature = value.signature
    signer_id = value.signer_id
    if (signature is None) != (signer_id is None):
        return False
    if signature is None:
        return not required
    if not isinstance(signature, (bytes, bytearray, memoryview)):
        return False
    if not isinstance(signer_id, str) or not signer_id:
        return False
    verifier = verifiers.get(signer_id)
    if verifier is None:
        return False
    # Verification is a bool API. Branch on it explicitly so a false result can never
    # be discarded as a successful statement call.
    try:
        if not value.verify(verifier):
            return False
    except Exception:  # noqa: BLE001 - malformed signatures fail verification
        return False
    return not allowed or signer_id in allowed


def _resulting_state_digest(state: RexState) -> str:
    from .catalog import state_object_digest

    return state_object_digest(state)


def _resulting_state_valid(state: Any) -> bool:
    if not isinstance(state, RexState) or not verify_state(state):
        return False
    names = state.header.get("digest_names")
    return isinstance(names, list) and set(names) == set(state.tensors)


def _array_or_default(rex: Any, field: str, length: int, default: float):
    import numpy as np

    value = getattr(rex, field, None)
    if value is None:
        return np.full(length, default)
    return np.asarray(value)


def _structural_projection_matches(left: Any, right: Any) -> bool:
    """Compare exactly the structural fields represented by ``TemporalState``."""
    import numpy as np

    try:
        left._ensure_clean()
        right._ensure_clean()
        if (
            bool(left._directed) != bool(right._directed)
            or left.g_channel != right.g_channel
            or left.c_channel != right.c_channel
            or int(left.nE) != int(right.nE)
            or int(left.nF) != int(right.nF)
        ):
            return False
        for field in ("_boundary_ptr", "_boundary_idx", "_B2_col_ptr", "_B2_row_idx"):
            if not np.array_equal(np.asarray(getattr(left, field)), np.asarray(getattr(right, field))):
                return False
        if not np.array_equal(
            np.asarray(left._B2_vals, dtype=np.float64),
            np.asarray(right._B2_vals, dtype=np.float64),
        ):
            return False
        if not np.array_equal(
            _array_or_default(left, "_w_E", int(left.nE), 0.0),
            _array_or_default(right, "_w_E", int(right.nE), 0.0),
        ):
            return False
        if not np.array_equal(
            _array_or_default(left, "_signs", int(left.nE), 1.0),
            _array_or_default(right, "_signs", int(right.nE), 1.0),
        ):
            return False
    except Exception:  # noqa: BLE001 - malformed projections do not match
        return False
    return True


def _endpoints_match(package: MutationPackage, previous: Any) -> bool:
    from rexgraph.graph import RexGraph

    from .catalog import object_digest

    try:
        if not _resulting_state_valid(package.resulting_state):
            return False
        resulting = from_state(package.resulting_state)
        if _resulting_state_digest(package.resulting_state) != package.transition.resulting_state:
            return False
        history = from_temporal_state(package.temporal_state)
        expected_steps = 1 if package.transition.previous_state == "" else 2
        if expected_steps != history.T:
            return False
        if not _structural_projection_matches(
            history.reconstruct_at(history.T - 1), resulting
        ):
            return False
        if expected_steps == 1:
            if previous is not None:
                return False
        else:
            if not isinstance(previous, RexGraph):
                return False
            if object_digest(previous) != package.transition.previous_state:
                return False
            if not _structural_projection_matches(history.reconstruct_at(0), previous):
                return False
    except Exception:  # noqa: BLE001 - malformed packages fail verification
        return False
    return True


def verify_mutation(
    package: MutationPackage,
    *,
    previous,
    policy: MutationPolicy | None = None,
    verifiers: dict[str, Any] | None = None,
    parent_digest: str | None | object = _PARENT_UNSET,
) -> bool:
    """Verify v2 state, endpoints, lineage, policy, and every signature requirement."""
    if not isinstance(package, MutationPackage) or package.version != MUTATION_VERSION:
        return False
    if package.temporal_state.header.get("temporal_state_version") != TEMPORAL_STATE_VERSION:
        return False
    if not verify_temporal_state(package.temporal_state):
        return False
    if package.transition.delta_state != package.temporal_state.header.get("digest"):
        return False
    try:
        transition_digest = package.transition.digest
    except Exception:  # noqa: BLE001 - malformed transition fields fail verification
        return False
    if package.link.transition_digest != transition_digest:
        return False
    if parent_digest is not _PARENT_UNSET and package.link.parent_digest != parent_digest:
        return False
    policy = policy or MutationPolicy()
    if not isinstance(policy, MutationPolicy) or package.transition.policy != policy.digest:
        return False
    allowed = set(policy.allowed_signers)
    verifier_map = dict(verifiers or {})
    if not _signature_valid(
        package.transition,
        required=policy.require_transition_signature,
        allowed=allowed,
        verifiers=verifier_map,
    ):
        return False
    if not _signature_valid(
        package.link,
        required=policy.require_lineage_signature,
        allowed=allowed,
        verifiers=verifier_map,
    ):
        return False
    return _endpoints_match(package, previous)


def _signature_hex(value: bytes | None) -> str | None:
    return None if value is None else bytes(value).hex()


def mutation_to_bytes(package: MutationPackage) -> bytes:
    """Encode one v2 mutation package without executable or pickle metadata."""
    from safetensors.numpy import save

    if package.version != MUTATION_VERSION:
        raise ValueError("only MutationPackage v2 may be written")
    if not verify_temporal_state(package.temporal_state):
        raise ValueError("mutation TemporalState is not verified v2 state")
    if package.transition.delta_state != package.temporal_state.header.get("digest"):
        raise ValueError("transition does not bind the TemporalState v2 digest")
    if package.link.transition_digest != package.transition.digest:
        raise ValueError("mutation lineage does not match transition")
    if not _resulting_state_valid(package.resulting_state):
        raise ValueError("mutation has no verified canonical resulting state")
    if _resulting_state_digest(package.resulting_state) != package.transition.resulting_state:
        raise ValueError("transition does not bind the carried resulting state")
    header = dict(package.temporal_state.header)
    result_header = dict(package.resulting_state.header)
    tensors = {
        **{
            f"temporal/{name}": value
            for name, value in package.temporal_state.tensors.items()
        },
        **{
            f"resulting/{name}": value
            for name, value in package.resulting_state.tensors.items()
        },
    }
    payload = save(
        tensors,
        metadata={
            "rex_meta": json.dumps(
                {"resulting_header": result_header, "temporal_header": header},
                separators=(",", ":"),
                sort_keys=True,
            )
        },
    )
    metadata = {
        "temporal_header": header,
        "resulting_header": result_header,
        "link": {
            **package.link.manifest(),
            "signer_id": package.link.signer_id,
            "signature": _signature_hex(package.link.signature),
        },
        "transition": {
            **package.transition.manifest(),
            "signer_id": package.transition.signer_id,
            "signature": _signature_hex(package.transition.signature),
        },
        "version": MUTATION_VERSION,
    }
    return pack_transport(payload, object_type="MutationPackage", metadata=metadata)


def _decode_signature(value: Any, field: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} signature must be hexadecimal text")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} signature is not valid hexadecimal text") from exc


def mutation_from_bytes(blob: bytes, *, allow_legacy: bool = False) -> MutationPackage:
    """Decode a mutation package, requiring explicit migration for unverified v1 state."""
    from safetensors.numpy import load

    payload, outer = unpack_transport(blob, verify=True)
    if outer.get("object_type") != "MutationPackage":
        raise TypeError("transport payload is not a MutationPackage")
    metadata = outer.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("mutation package metadata is invalid")
    version = metadata.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version not in READABLE_VERSIONS:
        raise ValueError("unsupported mutation package version")
    if version == 1 and not allow_legacy:
        raise ValueError(
            "MutationPackage v1 has unsigned reconstruction metadata; "
            "set allow_legacy=True only for migration"
        )
    temporal_header = metadata.get("temporal_header")
    if not isinstance(temporal_header, dict):
        raise ValueError("mutation package has no temporal header")
    arrays = dict(load(payload))
    if version == MUTATION_VERSION:
        if any(
            not name.startswith(("temporal/", "resulting/")) for name in arrays
        ):
            raise ValueError("mutation v2 contains an unscoped tensor")
        temporal_tensors = {
            name[len("temporal/"):]: value
            for name, value in arrays.items()
            if name.startswith("temporal/")
        }
        resulting_tensors = {
            name[len("resulting/"):]: value
            for name, value in arrays.items()
            if name.startswith("resulting/")
        }
    else:
        temporal_tensors = arrays
        resulting_tensors = {}
    state = TemporalState(temporal_tensors, dict(temporal_header))
    state_version = state.header.get("temporal_state_version")
    if version == MUTATION_VERSION:
        if state_version != TEMPORAL_STATE_VERSION or not verify_temporal_state(state):
            raise ValueError("mutation TemporalState v2 semantic digest mismatch")
        resulting_header = metadata.get("resulting_header")
        if not isinstance(resulting_header, dict):
            raise ValueError("mutation package has no resulting state header")
        resulting_state = RexState(resulting_tensors, dict(resulting_header))
        if not _resulting_state_valid(resulting_state):
            raise ValueError("mutation resulting RexState tensor digest mismatch")
    else:
        if state_version != 1:
            raise ValueError("legacy mutation must contain TemporalState v1")
        from .temporal_state import _legacy_tensor_payload_matches

        if not _legacy_tensor_payload_matches(state):
            raise ValueError("legacy mutation TemporalState tensor digest mismatch")
        resulting_state = None

    transition_data = metadata.get("transition")
    link_data = metadata.get("link")
    if not isinstance(transition_data, dict) or not isinstance(link_data, dict):
        raise ValueError("mutation transition or lineage metadata is invalid")
    if transition_data.get("version") != TRANSITION_VERSION:
        raise ValueError("unsupported transition commit version")
    if link_data.get("version") != COMMIT_VERSION:
        raise ValueError("unsupported commit link version")
    try:
        transition = TransitionCommit(
            transition_data["previous_state"],
            transition_data["delta_state"],
            transition_data["resulting_state"],
            float(transition_data["tx_time"]),
            transition_data.get("actor", ""),
            transition_data.get("policy", ""),
            transition_data.get("signer_id"),
            _decode_signature(transition_data.get("signature"), "transition"),
        )
        link = CommitLink(
            link_data["transition_digest"],
            link_data.get("parent_digest"),
            link_data.get("signer_id"),
            _decode_signature(link_data.get("signature"), "lineage"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mutation transition or lineage fields are invalid") from exc
    if not math.isfinite(transition.tx_time):
        raise ValueError("mutation transition time must be finite")
    if link.transition_digest != transition.digest:
        raise ValueError("mutation lineage does not match transition")
    if version == MUTATION_VERSION:
        if transition.delta_state != state.header.get("digest"):
            raise ValueError("mutation transition does not bind TemporalState v2")
        if _resulting_state_digest(resulting_state) != transition.resulting_state:
            raise ValueError("mutation transition does not bind carried resulting state")
    elif transition.delta_state != _legacy_delta_digest(state):
        raise ValueError("legacy mutation delta tensor digest mismatch")
    return MutationPackage(transition, link, state, resulting_state, version)
