"""Authenticated indexed encryption for the safetensors bridge."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rexgraph import graded_boundary as gb
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io import (
    ContainerEncryptionConfig,
    SafetensorQuerySession,
    load,
    read_safetensor_tensor,
    save,
)
from rexgraph.io._container_crypto import (
    ContainerEncryptionError,
    open_encrypted_manifest,
    protect_tensors,
    read_protected_tensor,
    validate_storage_inventory,
)
from rexgraph.io.rex_state import to_state
from rexgraph.io.safetensors_bridge import (
    fingerprints_to_safetensors,
    load_extra,
    load_safetensors,
    rex_to_safetensors,
    safetensors_to_fingerprints,
    safetensors_to_rex,
    safetensors_to_temporal_rex,
    save_safetensors,
    temporal_rex_to_safetensors,
)


class _AeadProperties:
    """Test-only KMS/property object; core receives only this opaque action."""

    def __init__(self, configuration, keys):
        self.configuration = configuration
        self._keys = dict(keys)
        self.open_calls: list[str] = []
        self.seal_calls: list[str] = []

    @staticmethod
    def authenticated_encryption():
        return True

    def seal(self, key_id, plaintext, aad):
        AESGCM = pytest.importorskip(
            "cryptography.hazmat.primitives.ciphers.aead"
        ).AESGCM
        nonce = os.urandom(12)
        encoded = key_id.encode("utf-8")
        self.seal_calls.append(key_id)
        return (
            len(encoded).to_bytes(2, "little")
            + encoded
            + nonce
            + AESGCM(self._keys[key_id]).encrypt(nonce, plaintext, aad)
        )

    def open(self, envelope, aad):
        AESGCM = pytest.importorskip(
            "cryptography.hazmat.primitives.ciphers.aead"
        ).AESGCM
        size = int.from_bytes(envelope[:2], "little")
        key_id = envelope[2:2 + size].decode("utf-8")
        nonce = envelope[2 + size:14 + size]
        ciphertext = envelope[14 + size:]
        self.open_calls.append(key_id)
        return AESGCM(self._keys[key_id]).decrypt(nonce, ciphertext, aad)

    def open_with(self, key_id, envelope, aad):
        size = int.from_bytes(envelope[:2], "little")
        encoded = envelope[2:2 + size].decode("utf-8")
        if encoded != key_id:
            raise PermissionError("authenticated manifest chose the wrong key")
        return self.open(envelope, aad)


def _keys(*names):
    AESGCM = pytest.importorskip(
        "cryptography.hazmat.primitives.ciphers.aead"
    ).AESGCM
    return {name: AESGCM.generate_key(bit_length=256) for name in names}


def _properties(
    *,
    tensor_keys=None,
    plaintext_tensors=(),
    plaintext_manifest=False,
    chunk_size=4096,
    keys=None,
):
    tensor_keys = tensor_keys or {}
    key_names = {"footer", *tensor_keys}
    keys = _keys(*key_names) if keys is None else keys
    config = ContainerEncryptionConfig(
        footer_key="footer",
        tensor_keys=tensor_keys,
        plaintext_tensors=plaintext_tensors,
        plaintext_manifest=plaintext_manifest,
        chunk_size=chunk_size,
    )
    return _AeadProperties(config, keys), keys


def _grade3():
    return RexGraph.from_cells(gb.solid_octahedron_3rex())


def _assert_same_state(left, right):
    a = to_state(left)
    b = to_state(right)
    assert a.header["digest"] == b.header["digest"]
    assert set(a.tensors) == set(b.tensors)
    for name in a.tensors:
        np.testing.assert_array_equal(a.tensors[name], b.tensors[name])


def _raw_file(path):
    safe = pytest.importorskip("safetensors")
    numpy_api = pytest.importorskip("safetensors.numpy")
    with safe.safe_open(str(path), framework="numpy") as opened:
        metadata = dict(opened.metadata() or {})
    return dict(numpy_api.load_file(str(path))), metadata


def _write_raw(path, tensors, metadata):
    pytest.importorskip("safetensors.numpy").save_file(
        tensors,
        str(path),
        metadata=metadata,
    )


def test_encrypted_rex_roundtrip_hides_logical_header_and_requires_keys(tmp_path):
    rex = _grade3()
    properties, keys = _properties(
        tensor_keys={
            "grade1": ["boundary_ptr", "boundary_idx"],
            "grade2": ["B2_col_ptr", "B2_row_idx", "B2_vals"],
            "grade3": ["gd0_data", "gd0_indices"],
        }
    )
    path = tmp_path / "grade3.safetensors"
    rex_to_safetensors(rex, path, encryption_properties=properties)

    tensors, metadata = _raw_file(path)
    assert set(metadata) == {"rex_encrypted", "rex_encryption"}
    assert "boundary_idx" not in metadata["rex_encryption"]
    assert "RexGraph" in metadata["rex_encryption"]  # public container kind only
    assert tensors and all(name.startswith("__rex_encrypted_storage__/")
                           for name in tensors)
    assert all(array.dtype == np.uint8 for array in tensors.values())

    with pytest.raises(PermissionError, match="decryption properties"):
        safetensors_to_rex(path)
    wrong, _ = _properties(
        tensor_keys=properties.configuration.tensor_keys,
    )
    with pytest.raises(PermissionError, match="authentication"):
        safetensors_to_rex(path, decryption_properties=wrong)

    loaded = safetensors_to_rex(path, decryption_properties=properties)
    _assert_same_state(rex, loaded)
    assert properties.open_calls[0] == "footer"
    assert {"footer", "grade1", "grade2", "grade3"} <= set(properties.open_calls)
    assert set(keys) == {"footer", "grade1", "grade2", "grade3"}


def test_signed_plaintext_manifest_is_visible_but_still_requires_auth(tmp_path):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    properties, _ = _properties(plaintext_manifest=True)
    path = tmp_path / "visible.safetensors"
    rex_to_safetensors(rex, path, encryption_properties=properties)
    _, metadata = _raw_file(path)
    descriptor = json.loads(metadata["rex_encryption"])
    assert descriptor["manifest_mode"] == "signed_plaintext"
    assert descriptor["manifest"]["metadata"]["rex_state_header"]
    assert any(member["logical_name"] == "boundary_idx"
               for member in descriptor["manifest"]["members"])
    with pytest.raises(PermissionError, match="decryption properties"):
        safetensors_to_rex(path)
    _assert_same_state(
        rex,
        safetensors_to_rex(path, decryption_properties=properties),
    )


def test_selective_read_opens_only_manifest_and_touched_chunk(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    big = np.arange(32_768, dtype=np.float64).reshape(8192, 4)
    properties, _ = _properties(tensor_keys={"field": ["field/Z"]})
    path = tmp_path / "selective.safetensors"
    rex_to_safetensors(
        rex,
        path,
        extra_tensors={"field/Z": big},
        encryption_properties=properties,
    )
    properties.open_calls.clear()
    got = read_safetensor_tensor(
        path,
        "field/Z",
        index=slice(130, 150),
        decryption_properties=properties,
    )
    np.testing.assert_array_equal(got, big[130:150])
    assert properties.open_calls == ["footer", "field"]


def test_query_session_prunes_chunks_gathers_rows_and_reuses_plaintext(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    row_id = np.arange(4096, dtype=np.int64)
    values = np.column_stack((row_id * 2, row_id * 3))
    cold = row_id + 100_000
    properties, _ = _properties(
        tensor_keys={
            "row": ["query/row_id"],
            "value": ["query/value"],
            "cold": ["query/cold"],
        }
    )
    path = tmp_path / "query.safetensors"
    rex_to_safetensors(
        rex,
        path,
        extra_tensors={
            "query/row_id": row_id,
            "query/value": values,
            "query/cold": cold,
        },
        encryption_properties=properties,
    )

    properties.open_calls.clear()
    with SafetensorQuerySession(
        path,
        decryption_properties=properties,
    ) as query:
        assert "query/cold" in query.names
        selected = query.select(
            "query/value",
            where=("query/row_id", ">=", 3584),
        )
        np.testing.assert_array_equal(selected["query/value"], values[3584:])
        first_calls = list(properties.open_calls)
        again = query.select(
            "query/value",
            where=("query/row_id", ">=", 3584),
        )
        np.testing.assert_array_equal(again["query/value"], values[3584:])
        assert properties.open_calls == first_calls
    assert first_calls == ["footer", "row", "row", "value", "value"]
    assert "cold" not in first_calls
    with pytest.raises(ValueError, match="closed"):
        query.read("query/value")


def test_sealed_statistics_prune_null_chunks_and_hide_member_facts(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    predicate = np.arange(1536, dtype=np.float64)
    predicate[[10, 1200]] = np.nan
    public = np.arange(1536, dtype=np.int32)
    properties, _ = _properties(
        tensor_keys={"predicate": ["query/predicate"]},
        plaintext_tensors=["query/public"],
        plaintext_manifest=True,
    )
    path = tmp_path / "statistics.safetensors"
    rex_to_safetensors(
        rex,
        path,
        extra_tensors={
            "query/predicate": predicate,
            "query/public": public,
        },
        encryption_properties=properties,
    )
    _, metadata = _raw_file(path)
    descriptor = json.loads(metadata["rex_encryption"])
    members = {
        member["logical_name"]: member
        for member in descriptor["manifest"]["members"]
    }
    protected = members["query/predicate"]
    assert protected["statistics_version"] == 1
    assert "statistics_envelope" in protected
    assert "statistics" not in protected
    public_member = members["query/public"]
    assert "statistics" in public_member
    assert "statistics_envelope" not in public_member

    second_path = tmp_path / "different-statistics.safetensors"
    rex_to_safetensors(
        rex,
        second_path,
        extra_tensors={
            "query/predicate": np.full(1536, np.nan),
            "query/public": public,
        },
        encryption_properties=properties,
    )
    _, second_metadata = _raw_file(second_path)
    second_descriptor = json.loads(second_metadata["rex_encryption"])
    second_protected = next(
        member
        for member in second_descriptor["manifest"]["members"]
        if member["logical_name"] == "query/predicate"
    )
    assert len(second_protected["statistics_envelope"]) == len(
        protected["statistics_envelope"]
    )

    properties.open_calls.clear()
    with SafetensorQuerySession(
        path,
        decryption_properties=properties,
    ) as query:
        positions = query.where("query/predicate", "isnull")
    np.testing.assert_array_equal(positions, np.array([10, 1200]))
    # Footer authenticator, one member statistics envelope, and only the two
    # predicate chunks whose authenticated null_count is nonzero.
    assert properties.open_calls == ["footer", "predicate", "predicate", "predicate"]


def test_plain_safetensor_query_session_matches_encrypted_api(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    row_id = np.arange(12, dtype=np.int32)
    values = np.arange(24, dtype=np.float32).reshape(12, 2)
    path = tmp_path / "plain-query.safetensors"
    rex_to_safetensors(
        rex,
        path,
        extra_tensors={"row_id": row_id, "values": values},
    )
    with SafetensorQuerySession(path) as query:
        selected = query.select("values", where=("row_id", "<", 3))
    np.testing.assert_array_equal(selected["values"], values[:3])


def test_optional_workspace_keyring_resolves_only_query_member_keys(
    tmp_path,
    monkeypatch,
):
    kms = pytest.importorskip("agent.kms")
    scope = pytest.importorskip("agent.server.scope")
    auth = pytest.importorskip("agent.server.auth")
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "agent-config"))
    monkeypatch.setenv(
        "REXGRAPH_SECRETS_URI",
        f"file://{tmp_path / 'workspace-secrets.json'}",
    )
    scope.reset_secret_store()
    auth.reset_auth_manager()
    token = scope.set_workspace("alpha")
    try:
        store = scope.secret_store()
        for key_id in ("footer", "row", "value", "cold"):
            store.put(key_id, f"alpha-secret-{key_id}", "key")
        writer = kms.WorkspaceKeyring(
            configuration=ContainerEncryptionConfig(
                footer_key="footer",
                tensor_keys={
                    "row": ["query/row_id"],
                    "value": ["query/value"],
                    "cold": ["query/cold"],
                },
                chunk_size=4096,
            )
        )
        row_id = np.arange(1024, dtype=np.int64)
        path = tmp_path / "workspace-query.safetensors"
        rex_to_safetensors(
            RexGraph.from_graph([0], [1]),
            path,
            extra_tensors={
                "query/row_id": row_id,
                "query/value": row_id * 2,
                "query/cold": row_id + 10_000,
            },
            encryption_properties=writer,
        )

        reader = kms.WorkspaceKeyring().load("footer")
        with SafetensorQuerySession(
            path,
            decryption_properties=reader,
        ) as query:
            selected = query.select(
                "query/value",
                where=("query/row_id", ">=", 1000),
            )
        np.testing.assert_array_equal(
            selected["query/value"], row_id[1000:] * 2
        )
        assert reader.holds("footer")
        assert reader.holds("row")
        assert reader.holds("value")
        assert not reader.holds("cold")
    finally:
        scope.reset_workspace(token)
        scope.reset_secret_store()
        auth.reset_auth_manager()


def test_statistics_predicates_match_numpy_exactly_across_dtypes(tmp_path):
    rng = np.random.default_rng(20260830)
    integers = rng.integers(-20, 21, size=1600, dtype=np.int64)
    floats = rng.standard_normal(1600)
    floats[[3, 700, 1500]] = np.nan
    datetimes = np.datetime64("2026-01-01") + np.arange(1600).astype(
        "timedelta64[D]"
    )
    complex_values = integers.astype(np.complex128) + 2j
    tensors = {
        "integers": integers,
        "floats": floats,
        "datetimes": datetimes,
        "complex": complex_values,
    }
    properties, _ = _properties(
        tensor_keys={"data": list(tensors)},
    )
    storage, metadata = protect_tensors(tensors, {}, properties, kind="query-test")
    path = tmp_path / "dtype-query.safetensors"
    _write_raw(path, storage, metadata)

    with SafetensorQuerySession(
        path,
        decryption_properties=properties,
    ) as query:
        cases = [
            ("integers", "==", 7, integers == 7),
            ("integers", "!=", -3, integers != -3),
            ("integers", "<", 0, integers < 0),
            ("integers", "<=", 10, integers <= 10),
            ("integers", ">", -10, integers > -10),
            ("integers", ">=", 15, integers >= 15),
            ("floats", "isnull", None, np.isnan(floats)),
            ("floats", "notnull", None, ~np.isnan(floats)),
            (
                "floats",
                ">=",
                0.25,
                (floats >= 0.25) & ~np.isnan(floats),
            ),
            (
                "datetimes",
                ">=",
                np.datetime64("2029-01-01"),
                datetimes >= np.datetime64("2029-01-01"),
            ),
            ("complex", "==", 7 + 2j, complex_values == 7 + 2j),
        ]
        for name, operator, value, expected in cases:
            np.testing.assert_array_equal(
                query.where(name, operator, value),
                np.flatnonzero(expected),
            )


def test_unmapped_new_tensor_defaults_to_footer_key(tmp_path):
    properties, _ = _properties(tensor_keys={"known": ["known"]})
    storage, metadata = protect_tensors(
        {"known": np.ones(2), "added_later": np.ones(3)},
        {"rex_meta": "{}"},
        properties,
        kind="test",
    )
    safe = pytest.importorskip("safetensors")
    numpy_api = pytest.importorskip("safetensors.numpy")
    path = tmp_path / "unmapped-default.safetensors"
    numpy_api.save_file(storage, str(path), metadata=metadata)
    with safe.safe_open(str(path), framework="numpy") as opened:
        manifest = open_encrypted_manifest(opened.metadata(), properties)
        validate_storage_inventory(opened, manifest)
    member = next(m for m in manifest["members"]
                  if m["logical_name"] == "added_later")
    assert member["protected"] is True and member["key_id"] == "footer"


def test_policy_rejects_unknown_overlap_and_unauthenticated_property(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    unknown, _ = _properties(tensor_keys={"grade": ["not_present"]})
    with pytest.raises(ValueError, match="absent tensors"):
        rex_to_safetensors(rex, tmp_path / "unknown", encryption_properties=unknown)

    overlap, _ = _properties(
        tensor_keys={"grade": ["boundary_ptr"]},
        plaintext_tensors=["boundary_ptr"],
    )
    with pytest.raises(ValueError, match="both encrypted and plaintext"):
        rex_to_safetensors(rex, tmp_path / "overlap", encryption_properties=overlap)

    class _NotAuthenticated:
        authenticated_encryption = False
        configuration = ContainerEncryptionConfig("footer", {})

        def seal(self, key_id, plaintext, aad):  # pragma: no cover - must not run
            return plaintext

    with pytest.raises(TypeError, match="authenticated_encryption"):
        rex_to_safetensors(
            rex,
            tmp_path / "not-aead",
            encryption_properties=_NotAuthenticated(),
        )

    properties, _ = _properties()
    with pytest.raises(TypeError, match="unsupported dtype"):
        protect_tensors(
            {"unsafe": np.array([object()], dtype=object)},
            {},
            properties,
            kind="test",
        )


def test_ciphertext_tamper_drop_and_cross_bundle_swap_fail(tmp_path):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    properties, _ = _properties()
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    rex_to_safetensors(rex, first, encryption_properties=properties)
    rex_to_safetensors(rex, second, encryption_properties=properties)
    first_tensors, first_meta = _raw_file(first)
    second_tensors, _ = _raw_file(second)
    storage_name = sorted(first_tensors)[0]

    tampered = {name: value.copy() for name, value in first_tensors.items()}
    tampered[storage_name][0] ^= np.uint8(1)
    tampered_path = tmp_path / "tampered.safetensors"
    _write_raw(tampered_path, tampered, first_meta)
    with pytest.raises(PermissionError, match="authentication"):
        load_safetensors(tampered_path, decryption_properties=properties)

    dropped = dict(first_tensors)
    dropped.pop(storage_name)
    dropped_path = tmp_path / "dropped.safetensors"
    _write_raw(dropped_path, dropped, first_meta)
    with pytest.raises(ContainerEncryptionError, match="inventory"):
        load_safetensors(dropped_path, decryption_properties=properties)

    swapped = dict(first_tensors)
    swapped[storage_name] = second_tensors[storage_name]
    swapped_path = tmp_path / "swapped.safetensors"
    _write_raw(swapped_path, swapped, first_meta)
    with pytest.raises(PermissionError, match="authentication"):
        load_safetensors(swapped_path, decryption_properties=properties)


def test_manifest_chunk_count_is_authenticated(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    properties, _ = _properties(plaintext_manifest=True)
    path = tmp_path / "count.safetensors"
    rex_to_safetensors(rex, path, encryption_properties=properties)
    tensors, metadata = _raw_file(path)
    descriptor = json.loads(metadata["rex_encryption"])
    descriptor["manifest"]["members"][0]["chunk_count"] += 1
    metadata["rex_encryption"] = json.dumps(descriptor, separators=(",", ":"))
    changed = tmp_path / "count-changed.safetensors"
    _write_raw(changed, tensors, metadata)
    with pytest.raises(PermissionError, match="authentication"):
        load_safetensors(changed, decryption_properties=properties)

    _, original_metadata = _raw_file(path)
    descriptor = json.loads(original_metadata["rex_encryption"])
    descriptor["footer_key"] = "substituted"
    original_metadata["rex_encryption"] = json.dumps(
        descriptor, separators=(",", ":")
    )
    substituted = tmp_path / "footer-substituted.safetensors"
    _write_raw(substituted, tensors, original_metadata)
    with pytest.raises(ContainerEncryptionError, match="footer_key"):
        load_safetensors(substituted, decryption_properties=properties)


def test_explicit_plaintext_tensor_is_still_integrity_checked(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    properties, _ = _properties(plaintext_tensors=["public"])
    path = tmp_path / "public.safetensors"
    rex_to_safetensors(
        rex,
        path,
        extra_tensors={"public": np.arange(10, dtype=np.int32)},
        encryption_properties=properties,
    )
    tensors, metadata = _raw_file(path)
    with pytest.importorskip("safetensors").safe_open(
        str(path), framework="numpy"
    ) as opened:
        manifest = open_encrypted_manifest(opened.metadata(), properties)
    public = next(member for member in manifest["members"]
                  if member["logical_name"] == "public")
    assert public["protected"] is False
    tensors[public["storage_name"]][0] ^= np.uint8(1)
    changed = tmp_path / "public-changed.safetensors"
    _write_raw(changed, tensors, metadata)
    with pytest.raises(ContainerEncryptionError, match="plaintext chunk"):
        read_safetensor_tensor(
            changed,
            "public",
            decryption_properties=properties,
        )


def test_empty_protected_tensor_still_requires_its_key(tmp_path):
    rex = RexGraph.from_graph([0], [1])
    properties, keys = _properties(tensor_keys={"empty-key": ["empty"]})
    path = tmp_path / "empty.safetensors"
    rex_to_safetensors(
        rex,
        path,
        extra_tensors={"empty": np.zeros((0, 3), dtype=np.float32)},
        encryption_properties=properties,
    )
    footer_only = _AeadProperties(properties.configuration, {"footer": keys["footer"]})
    with pytest.raises(PermissionError, match="authentication"):
        read_safetensor_tensor(
            path,
            "empty",
            decryption_properties=footer_only,
        )
    with pytest.raises(PermissionError, match="authentication"):
        read_safetensor_tensor(
            path,
            "empty",
            index=slice(0, 0),
            decryption_properties=footer_only,
        )
    got = read_safetensor_tensor(path, "empty", decryption_properties=properties)
    assert got.shape == (0, 3) and got.dtype == np.float32


def test_temporal_fingerprint_extra_and_generic_routes_do_not_bypass(tmp_path):
    properties, _ = _properties()
    temporal = TemporalRex([
        (np.array([0, 1], np.int32), np.array([1, 2], np.int32)),
        (np.array([0, 1, 2], np.int32), np.array([1, 2, 0], np.int32)),
    ])
    temporal_path = tmp_path / "temporal.safetensors"
    temporal_rex_to_safetensors(
        temporal,
        temporal_path,
        encryption_properties=properties,
    )
    loaded_temporal = safetensors_to_temporal_rex(
        temporal_path,
        decryption_properties=properties,
    )
    assert loaded_temporal.T == temporal.T
    assert loaded_temporal.reconstruct_at(1).nE == temporal.reconstruct_at(1).nE

    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    labels = np.array(["a", "b", "a", "b", "c", "c"])
    fingerprint_path = tmp_path / "fingerprints.safetensors"
    fingerprints_to_safetensors(
        features,
        labels,
        fingerprint_path,
        metadata={"purpose": "test"},
        encryption_properties=properties,
    )
    got_features, got_labels, _, metadata = safetensors_to_fingerprints(
        fingerprint_path,
        decryption_properties=properties,
    )
    np.testing.assert_array_equal(got_features, features)
    np.testing.assert_array_equal(got_labels, labels)
    assert metadata["purpose"] == "test"

    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    generic_path = tmp_path / "generic.safetensors"
    save(generic_path, rex, encryption_properties=properties)
    _assert_same_state(
        rex,
        load(generic_path, decryption_properties=properties),
    )

    extra_path = tmp_path / "extra.safetensors"
    rex_to_safetensors(
        rex,
        extra_path,
        extra_meta={"alpha": 3},
        encryption_properties=properties,
    )
    assert load_extra(extra_path, decryption_properties=properties) == {"alpha": 3}


def test_plaintext_compatibility_keeps_native_tensor_names(tmp_path):
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    path = tmp_path / "plain.safetensors"
    save_safetensors(path, rex)
    tensors, metadata = _raw_file(path)
    assert "boundary_idx" in tensors
    assert "rex_state_header" in metadata
    assert "rex_encryption" not in metadata
    _assert_same_state(rex, safetensors_to_rex(path))


def test_direct_common_reader_validates_inventory_before_payload(tmp_path):
    properties, _ = _properties()
    storage, metadata = protect_tensors(
        {"x": np.arange(12, dtype=np.int16).reshape(6, 2)},
        {"rex_meta": "{}"},
        properties,
        kind="test",
    )
    path = tmp_path / "direct.safetensors"
    _write_raw(path, storage, metadata)
    safe = pytest.importorskip("safetensors")
    with safe.safe_open(str(path), framework="numpy") as opened:
        manifest = open_encrypted_manifest(opened.metadata(), properties)
        validate_storage_inventory(opened, manifest)
        got = read_protected_tensor(
            opened,
            manifest,
            "x",
            properties,
            index=2,
        )
    np.testing.assert_array_equal(got, np.array([4, 5], dtype=np.int16))
