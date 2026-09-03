"""Canonical, semantic tensor state for ``TemporalRex`` histories.

Version 2 binds reconstruction metadata and tensor bytes into one state identity. Version
1 remains readable for reference artifacts, but its digest covered only tensors and must
not be used as the identity of a newly signed mutation.
"""
from __future__ import annotations

import hmac
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .manifest import manifest_digest
from .rex_state import DIGEST_ALGO, state_digest

FORMAT_VERSION = 2
READABLE_VERSIONS = (1, FORMAT_VERSION)

__all__ = [
    "FORMAT_VERSION",
    "READABLE_VERSIONS",
    "TemporalState",
    "from_temporal_state",
    "to_temporal_state",
    "verify_temporal_state",
]


@dataclass
class TemporalState:
    """Named history tensors and the metadata required to reconstruct them."""

    tensors: dict[str, np.ndarray]
    header: dict[str, Any] = field(default_factory=dict)


def _put(tensors: dict[str, np.ndarray], name: str, value: Any) -> None:
    if value is not None:
        tensors[name] = np.ascontiguousarray(np.asarray(value))


def _semantic_digest(header: dict[str, Any]) -> str:
    unsigned = dict(header)
    unsigned.pop("digest", None)
    return manifest_digest({"header": unsigned, "object_type": "TemporalState"})


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _tensor_payload_digest(state: TemporalState) -> str | None:
    names = state.header.get("digest_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(state.tensors)
    ):
        return None
    algorithm = state.header.get("digest_algo", 1)
    if not isinstance(algorithm, int) or isinstance(algorithm, bool):
        return None
    if algorithm not in (1, DIGEST_ALGO):
        return None
    try:
        return state_digest(state.tensors, names, algo=algorithm)
    except (KeyError, TypeError, ValueError):
        return None


def _legacy_tensor_payload_matches(state: TemporalState) -> bool:
    """Check v1 bytes for migration without claiming semantic verification."""
    declared = state.header.get("digest")
    tensor_digest = _tensor_payload_digest(state)
    return bool(
        _is_sha256(declared)
        and tensor_digest is not None
        and hmac.compare_digest(declared, tensor_digest)
    )


def to_temporal_state(trex) -> TemporalState:
    """Return the canonical checkpoint and delta tensor state of a ``TemporalRex``."""
    trex._ensure_index()
    tensors: dict[str, np.ndarray] = {}
    checkpoints = [int(value) for value in trex._index_cp_times.tolist()]
    checkpoint_optional: dict[str, dict[str, bool]] = {}
    for time in checkpoints:
        _, bp, bi, weights, signs, b2_ptr, b2_rows, b2_values, *identity = trex._index_checkpoints[time]
        relation_ids = identity[0] if identity else None
        _put(tensors, f"checkpoint/{time}/boundary_ptr", bp)
        _put(tensors, f"checkpoint/{time}/boundary_idx", bi)
        _put(tensors, f"checkpoint/{time}/w_E", weights)
        _put(tensors, f"checkpoint/{time}/signs", signs)
        _put(tensors, f"checkpoint/{time}/relation_ids", relation_ids)
        has_faces = b2_ptr is not None and len(b2_ptr) > 1
        if has_faces:
            _put(tensors, f"checkpoint/{time}/B2_col_ptr", b2_ptr)
            _put(tensors, f"checkpoint/{time}/B2_row_idx", b2_rows)
            _put(tensors, f"checkpoint/{time}/B2_vals", b2_values)
        checkpoint_optional[str(time)] = {
            "w_E": weights is not None,
            "signs": signs is not None,
            "relation_ids": relation_ids is not None,
            "faces": bool(has_faces),
        }

    for time, delta in enumerate(trex._index_deltas):
        if delta is None:
            continue
        for name in (
            "born_cols",
            "born_offsets",
            "born_wE",
            "born_signs",
            "died_keys",
            "mod_keys",
            "mod_wE",
            "mod_signs",
            "mod_heads",
            "born_ids",
            "died_ids",
            "mod_ids",
            "mod_cols",
            "mod_offsets",
        ):
            _put(tensors, f"delta/{time}/{name}", getattr(delta, name))

    has_faces = any(value["faces"] for value in checkpoint_optional.values())
    for time, delta in enumerate(trex._index_face_deltas):
        if delta is None:
            continue
        has_faces = True
        for name in ("born_edge_keys", "born_offsets", "born_signs", "died_face_keys"):
            _put(tensors, f"face_delta/{time}/{name}", getattr(delta, name))

    total = int(trex.T)
    g_channels = list(getattr(trex, "_g_channels", ()))
    c_channels = list(getattr(trex, "_c_channels", ()))
    g_channels = [str(g_channels[index]) if index < len(g_channels) else "raw" for index in range(total)]
    c_channels = [
        str(c_channels[index]) if index < len(c_channels) else "share" for index in range(total)
    ]
    header: dict[str, Any] = {
        "object_type": "TemporalRex",
        "encoding": "delta",
        "temporal_state_version": FORMAT_VERSION,
        "T": total,
        "directed": bool(trex._directed),
        "general": bool(trex._general),
        "has_faces": bool(has_faces),
        "checkpoint_threshold": float(trex._checkpoint_threshold),
        "checkpoint_times": checkpoints,
        "checkpoint_optional": checkpoint_optional,
        "times": [float(value) for value in trex._times],
        "g_channels": g_channels,
        "c_channels": c_channels,
        "digest_names": sorted(tensors),
        "digest_algo": DIGEST_ALGO,
    }
    header["tensor_digest"] = state_digest(
        tensors, header["digest_names"], algo=DIGEST_ALGO
    )
    header["digest"] = _semantic_digest(header)
    return TemporalState(tensors, header)


def verify_temporal_state(state: TemporalState) -> bool:
    """Return whether tensors and semantic reconstruction metadata match the state seal."""
    if not isinstance(state, TemporalState) or not isinstance(state.header, dict):
        return False
    version = state.header.get("temporal_state_version")
    if not isinstance(version, int) or isinstance(version, bool):
        return False
    # Version 1 binds tensors only. It is readable through an explicit migration path,
    # but it can never answer that reconstruction semantics are intact.
    if version != FORMAT_VERSION:
        return False
    declared = state.header.get("digest")
    if not _is_sha256(declared):
        return False
    if state.header.get("object_type") != "TemporalRex" or state.header.get("encoding") != "delta":
        return False
    if state.header.get("digest_algo") != DIGEST_ALGO:
        return False
    tensor_digest = _tensor_payload_digest(state)
    if tensor_digest is None:
        return False
    recorded_tensor_digest = state.header.get("tensor_digest")
    if not _is_sha256(recorded_tensor_digest) or not hmac.compare_digest(
        recorded_tensor_digest, tensor_digest
    ):
        return False
    try:
        semantic_digest = _semantic_digest(state.header)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(declared, semantic_digest)


def _read_channels(
    header: dict[str, Any],
    field: str,
    total: int,
    default: str,
    allowed: set[str],
) -> list[str]:
    raw = header.get(field)
    if raw is None:
        return [default] * total
    if (
        not isinstance(raw, list)
        or len(raw) != total
        or any(not isinstance(value, str) or value not in allowed for value in raw)
    ):
        raise ValueError(f"TemporalState {field} are invalid")
    return list(raw)


def _read_layout(
    header: dict[str, Any],
) -> tuple[int, bool, bool, list[int], list[float], float, list[str], list[str]]:
    total = header.get("T")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("TemporalState has an invalid history length")
    if total < 0:
        raise ValueError("TemporalState history length may not be negative")
    directed = header.get("directed", False)
    general = header.get("general", False)
    if not isinstance(directed, bool) or not isinstance(general, bool):
        raise ValueError("TemporalState directed and general flags must be boolean")
    raw_checkpoints = header.get("checkpoint_times", [])
    if not isinstance(raw_checkpoints, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_checkpoints
    ):
        raise ValueError("TemporalState checkpoint_times must be integer indices")
    checkpoints = [int(value) for value in raw_checkpoints]
    if checkpoints != sorted(set(checkpoints)) or any(
        value < 0 or value >= total for value in checkpoints
    ):
        raise ValueError("TemporalState checkpoint indices are invalid")
    if total and (not checkpoints or checkpoints[0] != 0):
        raise ValueError("TemporalState must begin with checkpoint zero")

    raw_times = header.get("times")
    if raw_times is None:
        times = [float(value) for value in range(total)]
    elif not isinstance(raw_times, list) or len(raw_times) != total:
        raise ValueError("TemporalState times must contain one value per step")
    else:
        if any(isinstance(value, bool) for value in raw_times):
            raise ValueError("TemporalState times must be numeric")
        try:
            times = [float(value) for value in raw_times]
        except (TypeError, ValueError) as exc:
            raise ValueError("TemporalState times must be numeric") from exc
        if any(not math.isfinite(value) for value in times) or any(
            right < left for left, right in zip(times, times[1:], strict=False)
        ):
            raise ValueError("TemporalState times must be finite and nondecreasing")
    raw_threshold = header.get("checkpoint_threshold", 0.5)
    if isinstance(raw_threshold, bool):
        raise ValueError("TemporalState checkpoint threshold must be numeric")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("TemporalState checkpoint threshold must be numeric") from exc
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("TemporalState checkpoint threshold must be finite and nonnegative")
    g_channels = _read_channels(
        header, "g_channels", total, "raw", {"raw", "normalized"}
    )
    c_channels = _read_channels(header, "c_channels", total, "share", {"share", "count"})
    return total, directed, general, checkpoints, times, threshold, g_channels, c_channels


def _required(tensors: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in tensors]
    if missing:
        raise ValueError(f"TemporalState is missing required tensors: {missing!r}")


def from_temporal_state(
    state: TemporalState,
    *,
    verify: bool = True,
    allow_legacy: bool = False,
):
    """Reconstruct a delta-backed ``TemporalRex`` from canonical tensor state."""
    header = state.header
    version = header.get("temporal_state_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("invalid TemporalState version")
    if version not in READABLE_VERSIONS:
        raise ValueError(f"unsupported temporal_state_version {version}")
    if version == 1:
        if not allow_legacy:
            raise ValueError(
                "TemporalState v1 has unsigned reconstruction metadata; "
                "set allow_legacy=True only for migration"
            )
        if not _legacy_tensor_payload_matches(state):
            raise ValueError("legacy TemporalState tensor digest mismatch")
    elif verify and not verify_temporal_state(state):
        raise ValueError("TemporalState digest does not match its semantic payload")

    from rexgraph.graph import FaceDelta, TemporalDelta, TemporalRex

    tensors = state.tensors
    (
        total,
        directed,
        general,
        checkpoints,
        times,
        threshold,
        g_channels,
        c_channels,
    ) = _read_layout(header)
    index_checkpoints = {}
    for time in checkpoints:
        prefix = f"checkpoint/{time}/"
        _required(tensors, (prefix + "boundary_ptr", prefix + "boundary_idx"))
        face_names = tuple(
            prefix + name for name in ("B2_col_ptr", "B2_row_idx", "B2_vals")
        )
        if any(name in tensors for name in face_names):
            _required(tensors, face_names)
        index_checkpoints[time] = (
            time,
            tensors[prefix + "boundary_ptr"],
            tensors[prefix + "boundary_idx"],
            tensors.get(prefix + "w_E"),
            tensors.get(prefix + "signs"),
            tensors.get(prefix + "B2_col_ptr"),
            tensors.get(prefix + "B2_row_idx"),
            tensors.get(prefix + "B2_vals"),
            tensors.get(prefix + "relation_ids"),
        )

    edge_deltas = [None] * total
    face_deltas = [None] * total
    edge_names = (
        "born_cols",
        "born_offsets",
        "born_wE",
        "born_signs",
        "died_keys",
        "mod_keys",
        "mod_wE",
        "mod_signs",
    )
    identity_edge_names = ("born_ids", "died_ids", "mod_ids", "mod_cols", "mod_offsets")
    face_names = ("born_edge_keys", "born_offsets", "born_signs", "died_face_keys")
    for time in range(total):
        prefix = f"delta/{time}/"
        if prefix + "born_offsets" in tensors:
            _required(tensors, tuple(prefix + name for name in edge_names))
            present_identity_names = [prefix + name for name in identity_edge_names if prefix + name in tensors]
            if present_identity_names:
                _required(tensors, tuple(prefix + name for name in identity_edge_names))
            edge_deltas[time] = TemporalDelta(
                born_cols=tensors[prefix + "born_cols"],
                born_offsets=tensors[prefix + "born_offsets"],
                born_wE=tensors[prefix + "born_wE"],
                born_signs=tensors[prefix + "born_signs"],
                died_keys=tensors[prefix + "died_keys"],
                mod_keys=tensors[prefix + "mod_keys"],
                mod_wE=tensors[prefix + "mod_wE"],
                mod_signs=tensors[prefix + "mod_signs"],
                mod_heads=tensors.get(prefix + "mod_heads"),
                born_ids=tensors.get(prefix + "born_ids"),
                died_ids=tensors.get(prefix + "died_ids"),
                mod_ids=tensors.get(prefix + "mod_ids"),
                mod_cols=tensors.get(prefix + "mod_cols"),
                mod_offsets=tensors.get(prefix + "mod_offsets"),
                directed=directed,
            )
        prefix = f"face_delta/{time}/"
        if prefix + "born_offsets" in tensors:
            _required(tensors, tuple(prefix + name for name in face_names))
            face_deltas[time] = FaceDelta(
                born_edge_keys=tensors[prefix + "born_edge_keys"],
                born_offsets=tensors[prefix + "born_offsets"],
                born_signs=tensors[prefix + "born_signs"],
                died_face_keys=tensors[prefix + "died_face_keys"],
                directed=directed,
            )

    trex = TemporalRex([], directed=directed, general=general)
    trex._index_checkpoints = index_checkpoints
    trex._index_deltas = edge_deltas
    trex._index_face_deltas = face_deltas
    trex._index_cp_times = np.asarray(checkpoints, dtype=np.int64)
    trex._snapshots_materialized = False
    trex._snapshots = []
    trex._T = total
    trex._checkpoint_threshold = threshold
    trex._times = times
    trex._g_channels = g_channels
    trex._c_channels = c_channels
    return trex
