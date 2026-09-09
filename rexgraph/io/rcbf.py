"""Validated import of Relational Complex Binary Files (``.rcbf``).

RCBF and RCBD are sibling containers for relational complex data.  RCBF is a
sequential binary stream; RCBD is RexGraph's addressable
directory container.  This module maps only carriers that RexGraph can
represent without changing their mathematical meaning:

* C0 labels and primary C1 boundary supports;
* C1 weights, signs, and named metadata;
* C2 boundaries whose coefficients and chain condition are exact.

It deliberately does not import derived numeric products or temporal
state records.  They are not a shared canonical carrier, and silently
reconstructing them with a different engine would be less rigorous than an
explicit bridge.  A stream with temporal records therefore refuses by default;
an explicit current-snapshot opt-in is available when that loss is intended.
"""
from __future__ import annotations

import struct
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np

__all__ = ["RCBFFormatError", "is_rcbf_file", "load_rcbf"]


MAGIC = b"RCBF\0\0\0\0"
LEGACY_MAGIC = b"REXFILE\0"
_HEADER = struct.Struct("<7IQ")
_SUPPORTED_VERSIONS = frozenset({4, 5})
_MAX_NAME_BYTES = 1024 * 1024
_MAX_CELLS = 10_000_000
_MAX_SIGNALS = 100_000
_MAX_HARMONIC_DIM = 100_000
_MAX_EXACT_F64_INT = 2**53
_MAX_ATTRIBUTE_ROWS = 10_000_000
_FIXED_NAME_BYTES = 64

_FLAG_ATTRS = 0x1
_FLAG_STATE_FULL = 0x2
_FLAG_ATTR_STR = 0x4
_FLAG_SLOT_ATTR = 0x8
_FLAG_ATTR_KIND = 0x10


class RCBFFormatError(ValueError):
    """The stream is malformed or cannot be represented as a RexGraph exactly."""


def _read_exact(handle: BinaryIO, size: int, *, section: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise RCBFFormatError(f"truncated RCBF {section}")
    return data


def _require_remaining(handle: BinaryIO, size: int, *, section: str) -> None:
    position = handle.tell()
    handle.seek(0, 2)
    remaining = handle.tell() - position
    handle.seek(position)
    if size > remaining:
        raise RCBFFormatError(f"truncated RCBF {section}")


def _read_array(handle: BinaryIO, dtype: str, count: int, *, section: str) -> np.ndarray:
    if count < 0:
        raise RCBFFormatError(f"invalid negative RCBF {section} count")
    itemsize = np.dtype(dtype).itemsize
    _require_remaining(handle, count * itemsize, section=section)
    return np.frombuffer(
        _read_exact(handle, count * itemsize, section=section), dtype=dtype
    ).copy()


def _read_cstring(handle: BinaryIO, *, section: str) -> str:
    data = bytearray()
    while len(data) <= _MAX_NAME_BYTES:
        byte = _read_exact(handle, 1, section=section)
        if byte == b"\0":
            try:
                return bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RCBFFormatError(f"invalid UTF-8 in RCBF {section}") from exc
        data.extend(byte)
    raise RCBFFormatError(f"RCBF {section} exceeds {_MAX_NAME_BYTES} bytes")


def _read_fixed_string(handle: BinaryIO, size: int, *, section: str) -> str:
    raw = _read_exact(handle, size, section=section).split(b"\0", 1)[0]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RCBFFormatError(f"invalid UTF-8 in RCBF {section}") from exc


def _read_text(handle: BinaryIO, size: int, *, section: str) -> str:
    if size > _MAX_NAME_BYTES:
        raise RCBFFormatError(f"RCBF {section} exceeds {_MAX_NAME_BYTES} bytes")
    try:
        return _read_exact(handle, size, section=section).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RCBFFormatError(f"invalid UTF-8 in RCBF {section}") from exc


def _require_integral(values: np.ndarray, *, section: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise RCBFFormatError(
            f"RCBF {section} must use finite integral coefficients for an exact import"
        )
    if np.any(np.abs(values) > _MAX_EXACT_F64_INT):
        raise RCBFFormatError(
            f"RCBF {section} exceeds the exact integer range of its float64 carrier"
        )
    return np.ascontiguousarray(values.astype(np.int64))


def _skip(handle: BinaryIO, size: int, *, section: str) -> None:
    _require_remaining(handle, size, section=section)
    handle.seek(size, 1)


def _read_attribute_tail(
    handle: BinaryIO,
    *,
    flags: int,
    n_edges: int,
) -> tuple[list[list[dict[str, object]]], list[list[dict[str, object]]]]:
    """Read the current C1 and C1-slot metadata tail without retyping it."""
    attributes: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
    slots: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
    if not (flags & _FLAG_ATTRS):
        if flags & (_FLAG_ATTR_KIND | _FLAG_ATTR_STR):
            raise RCBFFormatError("RCBF attribute text/kind section lacks attribute counts")
        return attributes, slots

    counts = _read_array(handle, "<u2", n_edges, section="edge attribute counts")
    total = int(np.asarray(counts, dtype=np.uint64).sum())
    if total > _MAX_ATTRIBUTE_ROWS:
        raise RCBFFormatError("RCBF edge attribute count exceeds the import limit")
    for edge, count in enumerate(counts):
        for attribute in range(int(count)):
            key = _read_fixed_string(
                handle,
                _FIXED_NAME_BYTES,
                section=f"edge {edge} attribute {attribute} key",
            )
            numeric = struct.unpack(
                "<d",
                _read_exact(
                    handle,
                    8,
                    section=f"edge {edge} attribute {attribute} numeric value",
                ),
            )[0]
            if not np.isfinite(numeric):
                raise RCBFFormatError("RCBF edge attribute numeric values must be finite")
            attributes[edge].append({"key": key, "kind": "number", "value": numeric})

    if flags & _FLAG_ATTR_KIND:
        kind_names = {0: "number", 1: "text", 2: "reference"}
        for edge, rows in enumerate(attributes):
            for attribute, row in enumerate(rows):
                kind, grade, size = struct.unpack(
                    "<BBH",
                    _read_exact(
                        handle,
                        4,
                        section=f"edge {edge} attribute {attribute} kind",
                    ),
                )
                if kind not in kind_names:
                    raise RCBFFormatError("RCBF edge attribute has an unknown kind")
                row["kind"] = kind_names[kind]
                if kind == 0:
                    continue
                text = _read_text(
                    handle,
                    size,
                    section=f"edge {edge} attribute {attribute} text",
                )
                row["value"] = text
                if kind == 2:
                    row["reference_grade"] = int(grade)
    elif flags & _FLAG_ATTR_STR:
        for edge, rows in enumerate(attributes):
            for attribute, row in enumerate(rows):
                size = struct.unpack(
                    "<H",
                    _read_exact(
                        handle,
                        2,
                        section=f"edge {edge} attribute {attribute} text length",
                    ),
                )[0]
                if size:
                    row["kind"] = "text"
                    row["value"] = _read_text(
                        handle,
                        size,
                        section=f"edge {edge} attribute {attribute} text",
                    )

    if flags & _FLAG_SLOT_ATTR:
        for edge in range(n_edges):
            count = struct.unpack(
                "<H", _read_exact(handle, 2, section=f"edge {edge} slot attribute count")
            )[0]
            if count > _MAX_ATTRIBUTE_ROWS:
                raise RCBFFormatError("RCBF slot attribute count exceeds the import limit")
            for attribute in range(count):
                slot = struct.unpack(
                    "<i",
                    _read_exact(
                        handle,
                        4,
                        section=f"edge {edge} slot attribute {attribute} slot",
                    ),
                )[0]
                key = _read_fixed_string(
                    handle,
                    _FIXED_NAME_BYTES,
                    section=f"edge {edge} slot attribute {attribute} key",
                )
                numeric = struct.unpack(
                    "<d",
                    _read_exact(
                        handle,
                        8,
                        section=f"edge {edge} slot attribute {attribute} numeric value",
                    ),
                )[0]
                if not np.isfinite(numeric):
                    raise RCBFFormatError("RCBF slot attribute numeric values must be finite")
                size = struct.unpack(
                    "<H",
                    _read_exact(
                        handle,
                        2,
                        section=f"edge {edge} slot attribute {attribute} text length",
                    ),
                )[0]
                row: dict[str, object] = {"slot": int(slot), "key": key, "value": numeric}
                if size:
                    row["kind"] = "text"
                    row["value"] = _read_text(
                        handle,
                        size,
                        section=f"edge {edge} slot attribute {attribute} text",
                    )
                else:
                    row["kind"] = "number"
                slots[edge].append(row)
    return attributes, slots


def _chain_is_exact(
    boundary_ptr: np.ndarray,
    boundary_idx: np.ndarray,
    face_ptr: np.ndarray,
    face_rows: np.ndarray,
    face_values: np.ndarray,
) -> bool:
    """Check ``B1 B2 == 0`` over rational coefficients, without float tolerance."""
    for face in range(face_ptr.size - 1):
        accumulated: dict[int, Fraction] = {}
        for offset in range(int(face_ptr[face]), int(face_ptr[face + 1])):
            edge = int(face_rows[offset])
            coefficient = int(face_values[offset])
            lo, hi = int(boundary_ptr[edge]), int(boundary_ptr[edge + 1])
            support = boundary_idx[lo:hi]
            arity = int(support.size)
            if arity == 1:
                vertex = int(support[0])
                accumulated[vertex] = accumulated.get(vertex, Fraction()) + coefficient
                continue
            if arity == 2 and int(support[0]) == int(support[1]):
                continue
            head = int(support[0])
            accumulated[head] = accumulated.get(head, Fraction()) - coefficient
            share = Fraction(coefficient, arity - 1)
            for vertex in support[1:]:
                item = int(vertex)
                accumulated[item] = accumulated.get(item, Fraction()) + share
        if any(value for value in accumulated.values()):
            return False
    return True


def is_rcbf_file(path: str | Path) -> bool:
    """Return whether *path* has an RCBF or legacy REXFILE magic prefix."""
    try:
        with Path(path).open("rb") as handle:
            return _read_exact(handle, 8, section="magic") in {MAGIC, LEGACY_MAGIC}
    except (OSError, RCBFFormatError):
        return False


def load_rcbf(path: str | Path, *, allow_current_snapshot: bool = False):
    """Import an exact C0--C2 RCBF carrier as a :class:`~rexgraph.graph.RexGraph`.

    The stream must use version 4 or 5, homogeneous C1 directedness, finite C1
    weights, and integral C2 coefficients that satisfy the chain condition over
    the rationals. Signals are preserved as metadata and harmonic data is not
    treated as a shared canonical carrier. A stream with stored temporal states
    raises unless ``allow_current_snapshot=True`` expressly requests only its
    current C0--C2 state.
    """
    source = Path(path)
    with source.open("rb") as handle:
        magic = _read_exact(handle, 8, section="magic")
        if magic not in {MAGIC, LEGACY_MAGIC}:
            raise RCBFFormatError("not an RCBF stream (unknown magic)")
        version, flags, n_vertices, n_edges, n_faces, n_signals, harmonic_dim, _reserved = (
            _HEADER.unpack(_read_exact(handle, _HEADER.size, section="header"))
        )
        if version not in _SUPPORTED_VERSIONS:
            raise RCBFFormatError(
                f"unsupported RCBF version {version}; supported versions are 4 and 5"
            )
        if max(n_vertices, n_edges, n_faces) > _MAX_CELLS:
            raise RCBFFormatError(
                f"RCBF C0--C2 cell count exceeds the {_MAX_CELLS:,} import limit"
            )
        if n_signals > _MAX_SIGNALS or harmonic_dim > _MAX_HARMONIC_DIM:
            raise RCBFFormatError("RCBF signal or harmonic dimension exceeds the import limit")

        labels = [
            _read_cstring(handle, section=f"vertex label {vertex}")
            for vertex in range(n_vertices)
        ]

        # ``src`` and ``tgt`` are legacy pairwise convenience fields.  The following
        # branching boundary section is the authoritative C1 carrier.
        _read_array(handle, "<i4", n_edges, section="edge sources")
        _read_array(handle, "<i4", n_edges, section="edge targets")
        weights = _read_array(handle, "<f8", n_edges, section="edge weights")
        signs = _require_integral(
            _read_array(handle, "<f8", n_edges, section="edge signs"),
            section="edge signs",
        )
        if not np.all(np.isin(signs, (-1, 0, 1))):
            raise RCBFFormatError("RCBF edge signs must be exact -1, 0, or 1 orientations")
        if not np.all(np.isfinite(weights)):
            raise RCBFFormatError("RCBF edge weights must be finite")
        directed_values = _read_array(handle, "u1", n_edges, section="edge directedness")
        if not np.all(np.isin(directed_values, (0, 1))):
            raise RCBFFormatError("RCBF edge directedness must be 0 or 1")
        if directed_values.size and not np.all(directed_values == directed_values[0]):
            raise RCBFFormatError(
                "mixed per-relation directedness has no exact RexGraph carrier"
            )
        edge_type_codes = _read_array(handle, "u1", n_edges, section="edge type codes")
        edge_names = [
            _read_cstring(handle, section=f"edge name {edge}") for edge in range(n_edges)
        ]
        edge_types = [
            _read_cstring(handle, section=f"edge type {edge}") for edge in range(n_edges)
        ]
        edge_chi = _read_array(handle, "<f8", n_edges * 4, section="edge chi").reshape(
            n_edges, 4
        )

        arities = _read_array(handle, "<u2", n_edges, section="edge boundary arities")
        if np.any(arities == 0):
            raise RCBFFormatError("RCBF C1 relations must have nonempty boundary support")
        boundary_count = int(np.asarray(arities, dtype=np.uint64).sum())
        if boundary_count > np.iinfo(np.int32).max:
            raise RCBFFormatError("RCBF C1 boundary is too large for the RexGraph carrier")
        boundary_idx = _read_array(
            handle, "<i4", boundary_count, section="edge boundary participants"
        )
        if np.any(boundary_idx < 0) or np.any(boundary_idx >= n_vertices):
            raise RCBFFormatError("RCBF C1 boundary refers to a vertex outside C0")
        boundary_ptr = np.empty(n_edges + 1, dtype=np.int32)
        boundary_ptr[0] = 0
        boundary_ptr[1:] = np.cumsum(arities, dtype=np.int64).astype(np.int32)

        face_ptr = np.zeros(n_faces + 1, dtype=np.int32)
        face_rows = np.zeros(0, dtype=np.int32)
        face_values = np.zeros(0, dtype=np.int64)
        if n_faces:
            face_arities = _read_array(handle, "<u2", n_faces, section="face arities")
            if np.any(face_arities == 0):
                raise RCBFFormatError("RCBF C2 cells must have nonempty relation support")
            face_count = int(np.asarray(face_arities, dtype=np.uint64).sum())
            if face_count > np.iinfo(np.int32).max:
                raise RCBFFormatError("RCBF C2 boundary is too large for the RexGraph carrier")
            face_rows = _read_array(handle, "<i4", face_count, section="face relations")
            if np.any(face_rows < 0) or np.any(face_rows >= n_edges):
                raise RCBFFormatError("RCBF C2 boundary refers to a relation outside C1")
            face_values = _require_integral(
                _read_array(handle, "<f8", face_count, section="face coefficients"),
                section="face coefficients",
            )
            face_ptr[1:] = np.cumsum(face_arities, dtype=np.int64).astype(np.int32)
            for face in range(n_faces):
                lo, hi = int(face_ptr[face]), int(face_ptr[face + 1])
                if np.unique(face_rows[lo:hi]).size != hi - lo:
                    raise RCBFFormatError("RCBF C2 boundary repeats a C1 relation")
            for face in range(n_faces):
                _read_cstring(handle, section=f"face name {face}")

        signal_values: dict[str, np.ndarray] = {}
        for signal in range(n_signals):
            name = _read_cstring(handle, section=f"signal name {signal}")
            values = _read_array(
                handle, "<f8", n_edges, section=f"signal values {signal}"
            )
            if not np.all(np.isfinite(values)):
                raise RCBFFormatError("RCBF signal values must be finite")
            signal_values[name] = values
        _skip(
            handle,
            int(harmonic_dim) * int(n_edges) * np.dtype("<f8").itemsize,
            section="harmonic basis",
        )
        temporal_states = struct.unpack("<I", _read_exact(handle, 4, section="state count"))[0]
        if temporal_states and not allow_current_snapshot:
            raise RCBFFormatError(
                "RCBF temporal states require a temporal converter; pass "
                "allow_current_snapshot=True to import only the current C0--C2 state"
            )
        # State-full payloads only exist after temporal records. A current-snapshot
        # import deliberately stops before those records, while a timeless RCBF can
        # carry its complete C1 and per-boundary metadata into the RexGraph sidecar.
        edge_attributes: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
        slot_attributes: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
        if temporal_states == 0:
            edge_attributes, slot_attributes = _read_attribute_tail(
                handle,
                flags=int(flags),
                n_edges=int(n_edges),
            )

    if not _chain_is_exact(
        boundary_ptr, boundary_idx, face_ptr, face_rows, face_values
    ):
        raise RCBFFormatError("RCBF C2 boundary violates the exact chain condition")

    from rexgraph.graph import RexGraph

    kwargs = {
        "boundary_ptr": boundary_ptr,
        "boundary_idx": boundary_idx,
        "directed": bool(directed_values[0]) if directed_values.size else False,
        "w_E": weights,
        "signs": signs,
    }
    if n_faces:
        kwargs.update(
            B2_col_ptr=face_ptr,
            B2_row_idx=face_rows,
            B2_vals=np.asarray(face_values, dtype=np.float64),
        )
    rex = RexGraph(**kwargs)
    # An empty or isolated C0 basis is meaningful and cannot be recovered solely
    # from C1 supports, so retain the explicitly stored C0 cardinality.
    rex._nV = int(n_vertices)
    rex._agent_meta = {
        "vertex_labels": labels,
        "rcbf": {
            "version": int(version),
            "flags": int(flags),
            "signals": {name: values.tolist() for name, values in signal_values.items()},
            "temporal_states_unimported": int(temporal_states),
            "trailing_metadata_unimported": bool(
                temporal_states and flags & (_FLAG_ATTRS | _FLAG_SLOT_ATTR)
            ),
            "edge_attributes": edge_attributes,
            "slot_attributes": slot_attributes,
        },
    }
    for edge, (name, edge_type, code, chi) in enumerate(
        zip(edge_names, edge_types, edge_type_codes, edge_chi, strict=True)
    ):
        if name:
            rex.attach_metadata(1, edge, "rcbf_name", name)
        if edge_type:
            rex.attach_metadata(1, edge, "rcbf_type", edge_type)
        rex.attach_metadata(1, edge, "rcbf_type_code", int(code))
        if np.all(np.isfinite(chi)):
            rex.attach_metadata(1, edge, "rcbf_chi", chi.tolist())
    return rex
