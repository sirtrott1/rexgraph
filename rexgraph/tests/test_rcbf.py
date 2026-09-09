"""RCBF stream import tests using a small independently written wire fixture."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from rexgraph.io import FileCatalog, load, load_rcbf
from rexgraph.io.rcbf import LEGACY_MAGIC, MAGIC, RCBFFormatError


def _cstring(value: str) -> bytes:
    return value.encode("utf-8") + b"\0"


def _fixed_name(value: str) -> bytes:
    encoded = value.encode("utf-8")
    assert len(encoded) < 64
    return encoded + b"\0" * (64 - len(encoded))


def _write_rcbf(
    path,
    *,
    magic: bytes = MAGIC,
    supports: tuple[tuple[int, ...], ...] = ((0, 1, 2), (2, 3)),
    faces: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = (),
    directed: tuple[int, ...] | None = None,
    signals: dict[str, tuple[float, ...]] | None = None,
    temporal_states: int = 0,
    attributes: tuple[tuple[tuple[str, object], ...], ...] | None = None,
    slot_attributes: tuple[tuple[tuple[int, str, object], ...], ...] | None = None,
):
    n_vertices = max(vertex for support in supports for vertex in support) + 1
    n_edges = len(supports)
    n_faces = len(faces)
    directed = directed or (1,) * n_edges
    signals = signals or {}
    if attributes is not None:
        assert len(attributes) == n_edges
    if slot_attributes is not None:
        assert len(slot_attributes) == n_edges
    flags = 0
    if attributes is not None:
        flags |= 0x1 | 0x10
    if slot_attributes is not None:
        flags |= 0x8
    parts = [
        struct.pack(
            "<8s7IQ",
            magic,
            5,
            flags,
            n_vertices,
            n_edges,
            n_faces,
            len(signals),
            0,
            0,
        ),
        b"".join(_cstring(f"v{vertex}") for vertex in range(n_vertices)),
    ]
    sources = [support[0] for support in supports]
    targets = [support[1] if len(support) > 1 else support[0] for support in supports]
    parts.extend(
        [
            np.asarray(sources, dtype="<i4").tobytes(),
            np.asarray(targets, dtype="<i4").tobytes(),
            np.arange(1, n_edges + 1, dtype="<f8").tobytes(),
            np.ones(n_edges, dtype="<f8").tobytes(),
            np.asarray(directed, dtype="u1").tobytes(),
            np.zeros(n_edges, dtype="u1").tobytes(),
            b"".join(_cstring(f"r{edge}") for edge in range(n_edges)),
            b"".join(_cstring("gene_relation") for _ in range(n_edges)),
            np.zeros((n_edges, 4), dtype="<f8").tobytes(),
            np.asarray([len(support) for support in supports], dtype="<u2").tobytes(),
            np.asarray([item for support in supports for item in support], dtype="<i4").tobytes(),
        ]
    )
    if faces:
        parts.extend(
            [
                np.asarray([len(rows) for rows, _values in faces], dtype="<u2").tobytes(),
                np.asarray(
                    [row for rows, _values in faces for row in rows], dtype="<i4"
                ).tobytes(),
                np.asarray(
                    [value for _rows, values in faces for value in values], dtype="<f8"
                ).tobytes(),
                b"".join(_cstring(f"f{face}") for face in range(n_faces)),
            ]
        )
    for name, values in signals.items():
        assert len(values) == n_edges
        parts.extend([_cstring(name), np.asarray(values, dtype="<f8").tobytes()])
    parts.append(struct.pack("<I", temporal_states))
    if attributes is not None:
        parts.append(np.asarray([len(rows) for rows in attributes], dtype="<u2").tobytes())
        for rows in attributes:
            for key, value in rows:
                parts.extend([_fixed_name(key), struct.pack("<d", float(value) if not isinstance(value, str) else 0.0)])
        for rows in attributes:
            for _key, value in rows:
                if isinstance(value, str):
                    encoded = value.encode("utf-8")
                    parts.append(struct.pack("<BBH", 1, 0, len(encoded)))
                    parts.append(encoded)
                else:
                    parts.append(struct.pack("<BBH", 0, 0, 0))
    if slot_attributes is not None:
        for rows in slot_attributes:
            parts.append(struct.pack("<H", len(rows)))
            for slot, key, value in rows:
                parts.extend(
                    [
                        struct.pack("<i", slot),
                        _fixed_name(key),
                        struct.pack("<d", float(value) if not isinstance(value, str) else 0.0),
                    ]
                )
                encoded = value.encode("utf-8") if isinstance(value, str) else b""
                parts.append(struct.pack("<H", len(encoded)))
                parts.append(encoded)
    path.write_bytes(b"".join(parts))


def test_rcbf_preserves_a_branching_primary_relation_and_metadata(tmp_path):
    path = tmp_path / "branching.rcbf"
    _write_rcbf(path, signals={"perturbation": (0.5, -1.0)})

    rex = load_rcbf(path)

    assert rex.relation_supports() == [[0, 1, 2], [2, 3]]
    assert rex._directed is True
    assert rex._w_E.tolist() == [1.0, 2.0]
    assert rex._agent_meta["vertex_labels"] == ["v0", "v1", "v2", "v3"]
    assert rex._agent_meta["rcbf"]["signals"] == {"perturbation": [0.5, -1.0]}
    assert rex.get_metadata(1, 0, "rcbf_type") == "gene_relation"


def test_rcbf_exact_face_and_generic_io_catalog_paths(tmp_path):
    path = tmp_path / "triangle.rcbf"
    _write_rcbf(
        path,
        supports=((0, 1), (1, 2), (0, 2)),
        faces=(((0, 1, 2), (1.0, 1.0, -1.0)),),
    )

    rex = load(str(path))
    assert rex.nF == 1
    assert rex.chain_valid

    catalog = FileCatalog([tmp_path])
    entry = catalog.info("root0/triangle.rcbf")
    assert entry.kind == "rcbf"
    assert catalog.load(entry.name).relation_supports() == [[0, 1], [1, 2], [0, 2]]


def test_rcbf_preserves_edge_and_boundary_slot_metadata(tmp_path):
    path = tmp_path / "annotated.rcbf"
    _write_rcbf(
        path,
        attributes=((('confidence', 0.75), ('source', 'BindingDB')), ()),
        slot_attributes=(((2, 'concentration', 5.0),), ((0, 'state', 'active'),)),
    )

    imported = load_rcbf(path)._agent_meta["rcbf"]

    assert imported["edge_attributes"] == [
        [
            {"key": "confidence", "kind": "number", "value": 0.75},
            {"key": "source", "kind": "text", "value": "BindingDB"},
        ],
        [],
    ]
    assert imported["slot_attributes"] == [
        [{"slot": 2, "key": "concentration", "value": 5.0, "kind": "number"}],
        [{"slot": 0, "key": "state", "value": "active", "kind": "text"}],
    ]


def test_legacy_rexfile_stream_is_detected_without_colliding_with_legacy_rcbd(tmp_path):
    path = tmp_path / "legacy.rex"
    _write_rcbf(path, magic=LEGACY_MAGIC)

    assert load(path).relation_supports() == [[0, 1, 2], [2, 3]]
    assert FileCatalog([tmp_path]).info("root0/legacy.rex").kind == "rcbf"


def test_rcbf_rejects_nonexact_face_coefficients(tmp_path):
    path = tmp_path / "nonexact.rcbf"
    _write_rcbf(
        path,
        supports=((0, 1), (1, 2), (0, 2)),
        faces=(((0, 1, 2), (1.0, 1.0, -0.5)),),
    )

    with pytest.raises(RCBFFormatError, match="integral coefficients"):
        load_rcbf(path)


def test_rcbf_rejects_mixed_directedness_without_flattening_it(tmp_path):
    path = tmp_path / "mixed.rcbf"
    _write_rcbf(path, directed=(0, 1))

    with pytest.raises(RCBFFormatError, match="mixed per-relation directedness"):
        load_rcbf(path)


def test_rcbf_temporal_tail_requires_an_explicit_current_snapshot_opt_in(tmp_path):
    path = tmp_path / "temporal.rcbf"
    _write_rcbf(path, temporal_states=1)

    with pytest.raises(RCBFFormatError, match="temporal converter"):
        load_rcbf(path)
    assert load_rcbf(path, allow_current_snapshot=True).relation_supports() == [
        [0, 1, 2],
        [2, 3],
    ]
