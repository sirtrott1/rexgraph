from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io.bundle import load_rex, save_rex
from rexgraph.io.catalog import FileCatalog, object_digest
from rexgraph.io.commit import CommitLink
from rexgraph.io.export import export_parquet, verify_export
from rexgraph.io.manifest import canonical_json, digest_parts, manifest_digest
from rexgraph.io.mutation import (
    MutationPackage,
    MutationPolicy,
    mutation_from_bytes,
    mutation_to_bytes,
    prepare_mutation,
    verify_mutation,
)
from rexgraph.io.partition_state import build_rex_partition
from rexgraph.io.privacy import (
    PrivacyProjection,
    StaticIdentityKeyProvider,
    project_rows,
    scoped_pseudonym,
)
from rexgraph.io.replication import (
    apply_replication,
    pack_replication,
    unpack_replication,
)
from rexgraph.io.rex_state import RexState, from_state, state_digest, to_state, verify_state
from rexgraph.io.safetensors_bridge import (
    safetensors_to_temporal_rex,
    temporal_rex_to_safetensors,
)
from rexgraph.io.security import (
    Ed25519Signer,
    StaticKeyProvider,
    decrypt_bytes,
    encrypt_bytes,
    envelope_info,
)
from rexgraph.io.temporal_state import (
    TemporalState,
    from_temporal_state,
    to_temporal_state,
    verify_temporal_state,
)
from rexgraph.io.transition import TransitionCommit
from rexgraph.io.transport import inspect, pack, unpack


def test_manifest_encoding_and_digest_are_canonical():
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert manifest_digest({"b": 2, "a": 1}) == manifest_digest({"a": 1, "b": 2})
    assert digest_parts("state", [("a", "12"), ("b", "3")]) != digest_parts(
        "state", [("a", "1"), ("b", "23")]
    )
    with pytest.raises(ValueError):
        canonical_json({"not_finite": float("nan")})


def test_privacy_projection_discloses_only_selected_fields_and_scopes_identity():
    keys = StaticIdentityKeyProvider({"identity": b"stable local identity key"})
    rows = [{"person": "alice", "score": 7, "secret": "not exported"}]
    projection = PrivacyProjection(
        ("person", "score"),
        pseudonym_fields=("person",),
        scope="study-a",
        key_id="identity",
    )
    projected = project_rows(rows, projection, keys=keys)
    assert list(projected[0]) == ["person", "score"]
    assert projected[0]["person"] == scoped_pseudonym(
        "alice", scope="study-a", key_id="identity", keys=keys
    )
    assert projected[0]["person"] != scoped_pseudonym(
        "alice", scope="study-b", key_id="identity", keys=keys
    )
    assert projected[0]["score"] == 7
    assert projection.digest == PrivacyProjection(
        ["person", "score"],
        pseudonym_fields=["person"],
        scope="study-a",
        key_id="identity",
    ).digest


def test_privacy_projection_rejects_ambiguous_or_unkeyed_disclosures():
    with pytest.raises(ValueError, match="unique"):
        PrivacyProjection(("person", "person"))
    with pytest.raises(ValueError, match="not projected"):
        PrivacyProjection(("score",), ("person",), "study", "identity")
    projection = PrivacyProjection(("person",), ("person",), "study", "identity")
    with pytest.raises(ValueError, match="IdentityKeyProvider"):
        project_rows([{"person": "alice"}], projection)
    with pytest.raises(ValueError, match="IdentityKeyProvider"):
        scoped_pseudonym("alice", scope="study", key_id="identity", keys=None)


def test_plain_parquet_export_seals_schema_payload_and_partition_lineage():
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        "embedding": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "node": np.array([4, 9], dtype=np.int64),
    }
    payload, manifest = export_parquet(data, partition_digest="partition-a")
    assert payload.startswith(b"PAR1") and payload.endswith(b"PAR1")
    assert verify_export(payload, manifest)
    assert not verify_export(payload + b"x", manifest)
    assert not verify_export(payload, replace(manifest, partition_digest="partition-b"))
    table = pq.read_table(pa.BufferReader(payload))
    assert table.column_names == ["embedding_0", "embedding_1", "node"]
    assert export_parquet(dict(reversed(list(data.items()))), partition_digest="partition-a") == (
        payload,
        manifest,
    )


def test_encrypted_parquet_export_roundtrips_without_plaintext_artifact():
    pytest.importorskip("pyarrow")
    pytest.importorskip("cryptography")
    keys = StaticKeyProvider({"export": os.urandom(32)})
    payload, manifest = export_parquet(
        {"node": np.array([1, 2], dtype=np.int64)},
        partition_digest="partition-a",
        key_id="export",
        keys=keys,
    )
    assert manifest.encrypted and manifest.key_id == "export"
    assert payload.startswith(b"REXENC\x00")
    assert verify_export(payload, manifest, keys=keys)
    assert not verify_export(payload, manifest)
    assert not verify_export(
        payload,
        manifest,
        keys=StaticKeyProvider({"export": os.urandom(32)}),
    )
    assert decrypt_bytes(payload, keys=keys).startswith(b"PAR1")


def test_parquet_export_rejects_generated_column_name_collisions():
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="collides"):
        export_parquet(
            {"value": np.ones((2, 1)), "value_0": np.ones(2)},
            partition_digest="partition-a",
        )
    with pytest.raises(ValueError, match="equal row counts"):
        export_parquet(
            {"left": np.ones(2), "right": np.ones(3)},
            partition_digest="partition-a",
        )


def _grade_four_rex():
    from rexgraph.graded_boundary import solid_octahedron_3rex

    cells = deepcopy(solid_octahedron_3rex())
    volume = deepcopy(cells[3][0])
    cells[3] = [deepcopy(volume), deepcopy(volume)]
    cells.append([[0, 1]])
    rex = RexGraph.from_cells(cells)
    rex._directed = True
    rex._g_channel = "normalized"
    rex._c_channel = "count"
    rex._w_E = np.arange(1, rex.nE + 1, dtype=np.float64)
    rex._signs = np.where(np.arange(rex.nE) % 2, -1.0, 1.0)
    rex._w_boundary = {(0, int(rex._boundary_idx[0])): 2.5}
    rex._agent_meta = {"secret": "application metadata"}
    rex.attach_metadata(1, 0, "private", "not selected")
    return rex


def test_partition_selected_grade_four_cell_closes_every_lower_grade():
    from rexgraph.graded_boundary import verify_chain

    source = _grade_four_rex()
    partition = build_rex_partition(
        source,
        np.zeros(source.nE, dtype=np.uint8),
        grade_masks={4: np.ones(1, dtype=np.uint8)},
        policy_digest="structural-only",
    )
    result = partition.rex
    source_tower = source.graded_boundaries()
    result_tower = result.graded_boundaries()
    assert len(result_tower) == 4
    assert verify_chain(result_tower)[0]
    for expected, actual in zip(source_tower, result_tower, strict=True):
        assert expected.shape == actual.shape
        assert (expected != actual).nnz == 0
    assert result._directed is True
    assert result.g_channel == "normalized" and result.c_channel == "count"
    assert np.array_equal(result._w_E, source._w_E)
    assert np.array_equal(result._signs, source._signs)
    assert result._w_boundary == source._w_boundary
    assert getattr(result, "_agent_meta", {}) == {}
    assert getattr(result, "_cell_metadata", {}).get(1, {}) == {}
    assert partition.state.source_state == object_digest(source)
    assert partition.state.result_state == object_digest(result)
    assert partition.state.policy_digest == "structural-only"


def test_partition_keeps_empty_higher_grade_slots_without_relabelling_them():
    from rexgraph.graded_boundary import verify_chain

    source = _grade_four_rex()
    edge_mask = np.zeros(source.nE, dtype=np.uint8)
    edge_mask[0] = 1
    result = build_rex_partition(source, edge_mask).rex
    tower = result.graded_boundaries()
    assert len(tower) == 4
    assert [matrix.shape[1] for matrix in tower] == [1, 0, 0, 0]
    assert verify_chain(tower)[0]


def test_partition_face_selection_is_downward_closed_and_rejects_invalid_source():
    triangle = RexGraph.from_simplicial(
        np.array([0, 0, 1], dtype=np.int32),
        np.array([1, 2, 2], dtype=np.int32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    partition = build_rex_partition(
        triangle,
        np.zeros(3, dtype=np.uint8),
        f_mask=np.ones(1, dtype=np.uint8),
    )
    assert partition.rex.nE == 3 and partition.rex.nF == 1
    assert partition.rex.chain_valid

    invalid = RexGraph(
        sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
        B2_col_ptr=np.array([0, 2], dtype=np.int32),
        B2_row_idx=np.array([0, 1], dtype=np.int32),
        B2_vals=np.array([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="chain condition"):
        build_rex_partition(invalid, np.ones(2), f_mask=np.ones(1))


def test_partition_self_loop_retains_its_vertex_even_when_b1_column_cancels():
    loop = RexGraph(
        sources=np.array([7], dtype=np.int32),
        targets=np.array([7], dtype=np.int32),
    )
    result = build_rex_partition(loop, np.ones(1, dtype=np.uint8)).rex
    assert result.nE == 1 and result.nV == 1
    assert result.relation_supports() == [[0, 0]]


def test_envelope_roundtrip_metadata_and_tamper_rejection():
    pytest.importorskip("cryptography")
    from cryptography.exceptions import InvalidTag

    keys = StaticKeyProvider({"workspace/key": os.urandom(32)})
    blob = encrypt_bytes(
        b"relational-state",
        key_id="workspace/key",
        keys=keys,
        object_type="RexState",
    )
    assert decrypt_bytes(blob, keys=keys) == b"relational-state"
    assert envelope_info(blob).object_type == "RexState"
    assert envelope_info(blob).key_id == "workspace/key"

    tampered = bytearray(blob)
    tampered[-1] ^= 1
    with pytest.raises(InvalidTag):
        decrypt_bytes(bytes(tampered), keys=keys)


def test_envelope_rejects_wrong_key_and_invalid_key_length():
    pytest.importorskip("cryptography")
    from cryptography.exceptions import InvalidTag

    keys = StaticKeyProvider({"k": os.urandom(32)})
    blob = encrypt_bytes(b"state", key_id="k", keys=keys)
    wrong = StaticKeyProvider({"k": os.urandom(32)})
    with pytest.raises(InvalidTag):
        decrypt_bytes(blob, keys=wrong)
    with pytest.raises(ValueError, match="32-byte"):
        encrypt_bytes(b"state", key_id="short", keys=StaticKeyProvider({"short": b"x"}))


def test_envelope_rejects_modified_authenticated_header():
    pytest.importorskip("cryptography")
    from cryptography.exceptions import InvalidTag

    keys = StaticKeyProvider({"k": os.urandom(32), "j": os.urandom(32)})
    blob = encrypt_bytes(b"state", key_id="k", keys=keys)
    header_start = len(b"REXENC\x00") + 4
    changed = bytearray(blob)
    key_at = changed.index(b'"k"', header_start)
    changed[key_at + 1] = ord("j")
    with pytest.raises(InvalidTag):
        decrypt_bytes(bytes(changed), keys=keys)


def test_ed25519_signature_roundtrip_and_tamper_rejection():
    pytest.importorskip("cryptography")
    signer = Ed25519Signer.generate("workspace/operator")
    verifier = signer.verifier()
    signature = signer.sign(b"canonical record")
    assert verifier.signer_id == "workspace/operator"
    assert verifier.verify(b"canonical record", signature)
    assert not verifier.verify(b"different record", signature)


def test_transition_and_lineage_signatures_are_distinct():
    pytest.importorskip("cryptography")
    signer = Ed25519Signer.generate("operator")
    verifier = signer.verifier()
    transition = TransitionCommit("before", "delta", "after", 1.0, actor="worker").signed(
        signer
    )
    assert transition.verify(verifier)
    assert transition.digest == TransitionCommit(
        "before", "delta", "after", 1.0, actor="worker"
    ).digest

    link = CommitLink(transition.digest, "parent").signed(signer)
    assert link.verify(verifier)
    moved = CommitLink(transition.digest, "other", link.signer_id, link.signature)
    assert not moved.verify(verifier)
    assert not verifier.verify(link.signing_bytes(), transition.signature)
    assert not verifier.verify(transition.signing_bytes(), link.signature)


def test_transition_rejects_changed_state_or_signer_identity():
    pytest.importorskip("cryptography")
    signer = Ed25519Signer.generate("operator")
    signed = TransitionCommit("before", "delta", "after", 2.0).signed(signer)
    changed = TransitionCommit(
        signed.previous_state,
        signed.delta_state,
        "different",
        signed.tx_time,
        signer_id=signed.signer_id,
        signature=signed.signature,
    )
    assert not changed.verify(signer.verifier())
    relabeled = TransitionCommit(
        signed.previous_state,
        signed.delta_state,
        signed.resulting_state,
        signed.tx_time,
        signer_id="other",
        signature=signed.signature,
    )
    assert not relabeled.verify(signer.verifier())


def test_transport_roundtrip_inspection_and_tamper_rejection():
    blob = pack(b"abc", object_type="TransitionCommit", metadata={"sequence": 3})
    info = inspect(blob)
    payload, header = unpack(blob)
    assert payload == b"abc"
    assert info.object_type == "TransitionCommit"
    assert info.payload_size == 3
    assert header["metadata"] == {"sequence": 3}

    tampered = bytearray(blob)
    tampered[-1] ^= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        unpack(bytes(tampered))


def test_transport_rejects_truncation_and_invalid_public_header():
    blob = pack(b"payload", object_type="MutationPackage")
    with pytest.raises(ValueError, match="length mismatch"):
        unpack(blob[:-1])

    header_start = len(b"REXPKG\x00") + 4
    changed = bytearray(blob)
    object_at = changed.index(b"MutationPackage", header_start)
    changed[object_at] = 0xFF
    with pytest.raises(ValueError, match="invalid transport header"):
        inspect(bytes(changed))


def test_transport_bounds_metadata_and_requires_an_object_type():
    with pytest.raises(ValueError, match="nonempty"):
        pack(b"payload", object_type="")
    with pytest.raises(ValueError, match="header length"):
        inspect(pack(b"payload", object_type="x"), max_header=1)


def _temporal_history():
    history = TemporalRex([], directed=True)
    history.append_snapshot(
        RexGraph(
            sources=np.array([0, 1], dtype=np.int32),
            targets=np.array([1, 2], dtype=np.int32),
            directed=True,
            g_channel="normalized",
            c_channel="count",
        ),
        at=10.0,
    )
    history.append_snapshot(
        RexGraph(
            sources=np.array([0, 2], dtype=np.int32),
            targets=np.array([2, 1], dtype=np.int32),
            directed=True,
            g_channel="normalized",
            c_channel="count",
        ),
        at=20.0,
    )
    return history


def test_temporal_state_roundtrip_preserves_history_semantics():
    state = to_temporal_state(_temporal_history())
    assert state.header["temporal_state_version"] == 2
    assert verify_temporal_state(state)
    restored = from_temporal_state(state)
    assert restored.T == 2
    assert restored._directed is True
    assert restored.times.tolist() == [10.0, 20.0]
    assert restored.reconstruct_at(0).g_channel == "normalized"
    assert restored.reconstruct_at(1).c_channel == "count"
    for time in range(restored.T):
        expected = _temporal_history().reconstruct_at(time)
        actual = restored.reconstruct_at(time)
        assert np.array_equal(actual._boundary_ptr, expected._boundary_ptr)
        assert np.array_equal(actual._boundary_idx, expected._boundary_idx)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("directed", False),
        ("general", True),
        ("T", 1),
        ("times", [11.0, 20.0]),
        ("checkpoint_times", []),
    ],
)
def test_temporal_state_v2_rejects_semantic_header_substitution(field, replacement):
    state = to_temporal_state(_temporal_history())
    changed = TemporalState(dict(state.tensors), deepcopy(state.header))
    changed.header[field] = replacement
    assert not verify_temporal_state(changed)
    with pytest.raises(ValueError, match="semantic payload"):
        from_temporal_state(changed)


def test_temporal_state_rejects_tensor_tamper_and_unsealed_extra_tensor():
    state = to_temporal_state(_temporal_history())
    changed = TemporalState(dict(state.tensors), deepcopy(state.header))
    name = next(iter(changed.tensors))
    changed.tensors[name] = changed.tensors[name].copy()
    changed.tensors[name].reshape(-1)[0] ^= 1
    assert not verify_temporal_state(changed)

    changed = TemporalState(dict(state.tensors), deepcopy(state.header))
    changed.tensors["unsealed"] = np.array([1], dtype=np.int8)
    assert not verify_temporal_state(changed)


def test_temporal_state_reads_reference_v1_only_by_explicit_migration():
    state = to_temporal_state(_temporal_history())
    legacy_header = deepcopy(state.header)
    legacy_header["temporal_state_version"] = 1
    legacy_header.pop("tensor_digest")
    legacy_header["digest"] = state_digest(
        state.tensors,
        legacy_header["digest_names"],
        algo=legacy_header["digest_algo"],
    )
    legacy = TemporalState(dict(state.tensors), legacy_header)
    assert not verify_temporal_state(legacy)
    with pytest.raises(ValueError, match="allow_legacy"):
        from_temporal_state(legacy)
    with pytest.raises(ValueError, match="allow_legacy"):
        from_temporal_state(legacy, verify=False)
    assert from_temporal_state(legacy, allow_legacy=True).times.tolist() == [10.0, 20.0]


def test_temporal_state_v1_downgrade_never_reports_verified():
    state = to_temporal_state(_temporal_history())
    legacy_header = deepcopy(state.header)
    legacy_header["temporal_state_version"] = 1
    legacy_header.pop("tensor_digest")
    legacy_header["digest"] = state_digest(state.tensors, legacy_header["digest_names"])
    legacy_header["directed"] = False
    downgraded = TemporalState(dict(state.tensors), legacy_header)
    assert not verify_temporal_state(downgraded)
    assert from_temporal_state(downgraded, allow_legacy=True)._directed is False


def _two_edge_rex(**kwargs):
    return RexGraph(
        sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
        **kwargs,
    )


def test_rex_state_roundtrip_preserves_coparticipation_channel():
    original = _two_edge_rex(c_channel="count")
    state = to_state(original)
    assert state.header["c_channel"] == "count"
    assert from_state(state).c_channel == "count"


def test_object_digest_binds_header_semantics_omitted_by_tensor_digest():
    undirected = _two_edge_rex(directed=False, c_channel="share")
    directed = _two_edge_rex(directed=True, c_channel="share")
    counted = _two_edge_rex(directed=False, c_channel="count")
    metadata = _two_edge_rex(directed=False, c_channel="share")
    metadata._agent_meta = {"source": "different"}

    states = [to_state(value) for value in (undirected, directed, counted, metadata)]
    assert len({state.header["digest"] for state in states}) == 1
    assert len({object_digest(value) for value in (undirected, directed, counted, metadata)}) == 4
    assert object_digest(undirected) == object_digest(_two_edge_rex())


def test_catalog_uses_relative_names_and_hashes_without_exposing_root(tmp_path):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"safe tensor bytes")
    catalog = FileCatalog([tmp_path])
    entry = catalog.list()[0]
    assert entry.name == "root0/weights.safetensors"
    assert str(tmp_path) not in repr(entry)
    assert catalog.roots == ("root0",)
    assert catalog.search("weights") == [entry]
    digest = catalog.hash(entry.name)
    assert digest == hashlib.sha256(b"safe tensor bytes").hexdigest()
    assert catalog.info(entry.name).sha256 == digest


def test_catalog_rejects_unknown_escape_names_and_symlinks(tmp_path):
    (tmp_path / "graph.safetensors").write_bytes(b"x")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
    outside.write_bytes(b"outside")
    link = tmp_path / "linked.safetensors"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    catalog = FileCatalog([tmp_path])
    assert [entry.name for entry in catalog.list()] == ["root0/graph.safetensors"]
    with pytest.raises(KeyError):
        catalog.info("root0/../outside.safetensors")


def test_catalog_injects_higher_package_loaders_without_importing_agent(tmp_path):
    store = tmp_path / "records"
    store.mkdir()
    (store / "records.log").write_bytes(b"record")
    (store / "blobs.pack").write_bytes(b"blob")
    name = "root0/records"

    catalog = FileCatalog([tmp_path])
    assert catalog.info(name).kind == "rcdb"
    with pytest.raises(ValueError, match="injected loader"):
        catalog.load(name)

    loaded = FileCatalog(
        [tmp_path], loaders={"rcdb": lambda path: (path.name, path.is_absolute())}
    ).load(name)
    assert loaded == ("records", True)


def _mutation_endpoints():
    kwargs = {"directed": True, "g_channel": "normalized", "c_channel": "count"}
    previous = RexGraph(
        sources=np.array([0, 1], dtype=np.int32),
        targets=np.array([1, 2], dtype=np.int32),
        **kwargs,
    )
    resulting = RexGraph(
        sources=np.array([0, 2], dtype=np.int32),
        targets=np.array([2, 1], dtype=np.int32),
        **kwargs,
    )
    return previous, resulting


def test_mutation_v2_binds_state_endpoints_lineage_and_signatures():
    pytest.importorskip("cryptography")
    previous, resulting = _mutation_endpoints()
    signer = Ed25519Signer.generate("operator")
    policy = MutationPolicy(True, True, ("operator",))
    package = prepare_mutation(
        previous,
        resulting,
        tx_time=42.0,
        actor="worker",
        policy=policy,
        parent_digest="parent",
        transition_signer=signer,
        lineage_signer=signer,
    )
    assert package.version == 2
    assert package.transition.delta_state == package.temporal_state.header["digest"]
    assert verify_mutation(
        package,
        previous=previous,
        policy=policy,
        verifiers={"operator": signer.verifier()},
        parent_digest="parent",
    )
    assert not verify_mutation(
        package,
        previous=previous,
        policy=policy,
        verifiers={"operator": signer.verifier()},
        parent_digest=None,
    )


def test_mutation_roundtrip_keeps_v2_binding_and_channel_semantics():
    pytest.importorskip("safetensors")
    previous, resulting = _mutation_endpoints()
    package = prepare_mutation(previous, resulting, tx_time=7.0)
    restored = mutation_from_bytes(mutation_to_bytes(package))
    assert restored.digest == package.digest
    assert verify_mutation(restored, previous=previous)
    history = from_temporal_state(restored.temporal_state)
    assert history.reconstruct_at(0).g_channel == "normalized"
    assert history.reconstruct_at(1).c_channel == "count"


def test_mutation_carries_rich_result_and_requires_prior_chain_state():
    pytest.importorskip("safetensors")
    previous, resulting = _mutation_endpoints()
    resulting._agent_meta = {"source": "sensor", "vertex_labels": ["a", "b", "c"]}
    resulting._signals = np.array([2.0, 3.0], dtype=np.float64)
    resulting.attach_metadata(1, 0, "role", "changed")
    package = prepare_mutation(previous, resulting, tx_time=11.0)
    restored = mutation_from_bytes(mutation_to_bytes(package))

    with pytest.raises(TypeError, match="previous"):
        verify_mutation(restored)
    assert verify_mutation(restored, previous=previous)
    assert not verify_mutation(restored, previous=_two_edge_rex(directed=True))
    carried = from_state(restored.resulting_state)
    assert carried._agent_meta == resulting._agent_meta
    assert np.array_equal(carried._signals, resulting._signals)
    assert carried._cell_metadata[1][0]["role"] == "changed"


def test_genesis_mutation_requires_explicit_absent_previous():
    _previous, resulting = _mutation_endpoints()
    package = prepare_mutation(None, resulting, tx_time=1.0)
    assert verify_mutation(package, previous=None)
    assert not verify_mutation(package, previous=resulting)


def test_temporal_containers_preserve_clock_and_channel_semantics(tmp_path):
    history = _temporal_history()
    bundle_path = tmp_path / "history.rex"
    save_rex(bundle_path, history)
    bundle_restored = load_rex(bundle_path)
    assert bundle_restored.times.tolist() == [10.0, 20.0]
    assert bundle_restored.at(0).g_channel == "normalized"
    assert bundle_restored.at(1).c_channel == "count"

    pytest.importorskip("safetensors")
    tensor_path = tmp_path / "history.safetensors"
    temporal_rex_to_safetensors(history, tensor_path)
    tensor_restored = safetensors_to_temporal_rex(tensor_path)
    assert tensor_restored.times.tolist() == [10.0, 20.0]
    assert tensor_restored.reconstruct_at(0).g_channel == "normalized"
    assert tensor_restored.reconstruct_at(1).c_channel == "count"


def test_mutation_rejects_semantic_header_delta_and_endpoint_substitution():
    pytest.importorskip("cryptography")
    previous, resulting = _mutation_endpoints()
    signer = Ed25519Signer.generate("operator")
    policy = MutationPolicy(True, True, ("operator",))
    package = prepare_mutation(
        previous,
        resulting,
        tx_time=3.0,
        policy=policy,
        transition_signer=signer,
        lineage_signer=signer,
    )

    changed_state = TemporalState(dict(package.temporal_state.tensors), deepcopy(package.temporal_state.header))
    changed_state.header["g_channels"] = ["raw"] * changed_state.header["T"]
    assert not verify_mutation(
        replace(package, temporal_state=changed_state),
        previous=previous,
        policy=policy,
        verifiers={"operator": signer.verifier()},
    )

    changed_result = RexState(
        dict(package.resulting_state.tensors), deepcopy(package.resulting_state.header)
    )
    changed_result.header["c_channel"] = "share"
    assert verify_state(changed_result)
    assert not verify_mutation(
        replace(package, resulting_state=changed_result),
        previous=previous,
        policy=policy,
        verifiers={"operator": signer.verifier()},
    )

    other = RexGraph(
        sources=np.array([0, 1, 2], dtype=np.int32),
        targets=np.array([2, 2, 0], dtype=np.int32),
        directed=True,
        g_channel="normalized",
        c_channel="count",
    )
    other_package = prepare_mutation(previous, other, tx_time=3.0)
    forged_transition = TransitionCommit(
        package.transition.previous_state,
        other_package.temporal_state.header["digest"],
        package.transition.resulting_state,
        3.0,
        policy=policy.digest,
    ).signed(signer)
    forged_link = CommitLink(forged_transition.digest).signed(signer)
    forged = MutationPackage(
        forged_transition,
        forged_link,
        other_package.temporal_state,
        package.resulting_state,
    )
    assert not verify_mutation(
        forged,
        previous=previous,
        policy=policy,
        verifiers={"operator": signer.verifier()},
    )


def test_mutation_required_signers_and_direction_mode_fail_closed():
    previous, resulting = _mutation_endpoints()
    policy = MutationPolicy(require_transition_signature=True)
    with pytest.raises(ValueError, match="transition signer"):
        prepare_mutation(previous, resulting, tx_time=1.0, policy=policy)

    undirected = _two_edge_rex(directed=False)
    with pytest.raises(ValueError, match="direction modes"):
        prepare_mutation(undirected, resulting, tx_time=1.0)
    with pytest.raises(ValueError, match="finite"):
        prepare_mutation(previous, resulting, tx_time=float("nan"))


def test_mutation_v2_cannot_be_relabelled_as_legacy():
    pytest.importorskip("safetensors")
    previous, resulting = _mutation_endpoints()
    blob = mutation_to_bytes(prepare_mutation(previous, resulting, tx_time=8.0))
    payload, outer = unpack(blob)
    metadata = deepcopy(outer["metadata"])
    metadata["version"] = 1
    downgraded = pack(payload, object_type="MutationPackage", metadata=metadata)
    with pytest.raises(ValueError, match="TemporalState v1"):
        mutation_from_bytes(downgraded, allow_legacy=True)


def test_reference_mutation_v1_is_migration_only_and_never_verifies():
    pytest.importorskip("safetensors")
    from safetensors.numpy import save

    from rexgraph.io.mutation import _legacy_delta_digest

    previous, resulting = _mutation_endpoints()
    history = TemporalRex([], directed=True)
    history.append_snapshot(previous)
    history.append_snapshot(resulting)
    current = to_temporal_state(history)
    header = deepcopy(current.header)
    header["temporal_state_version"] = 1
    header.pop("tensor_digest")
    header.pop("g_channels")
    header.pop("c_channels")
    header["digest"] = state_digest(current.tensors, header["digest_names"])
    legacy_state = TemporalState(dict(current.tensors), header)
    transition = TransitionCommit(
        object_digest(previous),
        _legacy_delta_digest(legacy_state),
        object_digest(resulting),
        9.0,
        policy=MutationPolicy().digest,
    )
    link = CommitLink(transition.digest)
    payload = save(
        legacy_state.tensors,
        metadata={"rex_meta": json.dumps(header, separators=(",", ":"), sort_keys=True)},
    )
    metadata = {
        "temporal_header": header,
        "transition": {
            **transition.manifest(),
            "signer_id": None,
            "signature": None,
        },
        "link": {**link.manifest(), "signer_id": None, "signature": None},
        "version": 1,
    }
    blob = pack(payload, object_type="MutationPackage", metadata=metadata)
    with pytest.raises(ValueError, match="allow_legacy"):
        mutation_from_bytes(blob)
    migrated = mutation_from_bytes(blob, allow_legacy=True)
    assert migrated.version == 1
    assert not verify_mutation(migrated, previous=previous)


def _replication_fixture():
    signer = Ed25519Signer.generate("replicator")
    policy = MutationPolicy(True, True, ("replicator",))
    checkpoint, first = _mutation_endpoints()
    second = RexGraph(
        sources=np.array([0, 1, 2], dtype=np.int32),
        targets=np.array([2, 2, 0], dtype=np.int32),
        directed=True,
        g_channel="normalized",
        c_channel="count",
    )
    second._agent_meta = {"replica": "terminal", "vertex_labels": ["a", "b", "c"]}
    first_package = prepare_mutation(
        checkpoint,
        first,
        tx_time=1.0,
        policy=policy,
        parent_digest="checkpoint-commit",
        transition_signer=signer,
        lineage_signer=signer,
    )
    second_package = prepare_mutation(
        first,
        second,
        tx_time=2.0,
        policy=policy,
        parent_digest=first_package.link.digest,
        transition_signer=signer,
        lineage_signer=signer,
    )
    mutations = tuple(
        mutation_to_bytes(package) for package in (first_package, second_package)
    )
    checkpoint_bytes = b"canonical checkpoint artifact"
    blob, manifest = pack_replication(
        checkpoint_bytes,
        mutations,
        checkpoint_state=object_digest(checkpoint),
        checkpoint_commit="checkpoint-commit",
    )
    return checkpoint, second, signer, policy, mutations, checkpoint_bytes, blob, manifest


def test_replication_applies_signed_mutations_against_real_prior_states():
    pytest.importorskip("cryptography")
    pytest.importorskip("safetensors")
    checkpoint, terminal, signer, policy, mutations, checkpoint_bytes, blob, manifest = (
        _replication_fixture()
    )
    restored_checkpoint, restored_mutations, restored_manifest = unpack_replication(blob)
    assert restored_checkpoint == checkpoint_bytes
    assert restored_mutations == mutations
    assert restored_manifest == manifest
    applied = apply_replication(
        blob,
        checkpoint_loader=lambda raw: checkpoint if raw == checkpoint_bytes else None,
        policy=policy,
        verifiers={"replicator": signer.verifier()},
    )
    assert len(applied.packages) == 2
    assert object_digest(applied.result) == object_digest(terminal)
    assert applied.result._agent_meta == terminal._agent_meta
    assert applied.manifest.terminal_state == object_digest(terminal)


def test_replication_rejects_checkpoint_substitution_and_missing_verifiers():
    pytest.importorskip("cryptography")
    pytest.importorskip("safetensors")
    checkpoint, _terminal, signer, policy, _mutations, _cp, blob, _manifest = (
        _replication_fixture()
    )
    with pytest.raises(ValueError, match="checkpoint state"):
        apply_replication(
            blob,
            checkpoint_loader=lambda _raw: _two_edge_rex(directed=True),
            policy=policy,
            verifiers={"replicator": signer.verifier()},
        )
    with pytest.raises(ValueError, match="policy verification"):
        apply_replication(
            blob,
            checkpoint_loader=lambda _raw: checkpoint,
            policy=policy,
        )


def test_replication_rejects_missing_reordered_duplicate_and_modified_packages():
    pytest.importorskip("cryptography")
    pytest.importorskip("safetensors")
    _checkpoint, _terminal, _signer, _policy, mutations, checkpoint_bytes, blob, manifest = (
        _replication_fixture()
    )
    kwargs = {
        "checkpoint_state": manifest.checkpoint_state,
        "checkpoint_commit": manifest.checkpoint_commit,
    }
    with pytest.raises(ValueError, match="lineage|previous state"):
        pack_replication(checkpoint_bytes, mutations[1:], **kwargs)
    with pytest.raises(ValueError, match="lineage|previous state"):
        pack_replication(checkpoint_bytes, reversed(mutations), **kwargs)
    with pytest.raises(ValueError, match="lineage|previous state"):
        pack_replication(checkpoint_bytes, (mutations[0], mutations[0]), **kwargs)
    modified = bytearray(blob)
    modified[-1] ^= 1
    with pytest.raises(ValueError, match="digest"):
        unpack_replication(bytes(modified))


def test_replication_supports_explicit_empty_genesis_checkpoint():
    pytest.importorskip("safetensors")
    _previous, resulting = _mutation_endpoints()
    mutation = mutation_to_bytes(prepare_mutation(None, resulting, tx_time=1.0))
    blob, manifest = pack_replication(b"", (mutation,))
    applied = apply_replication(blob, checkpoint_loader=lambda raw: None if raw == b"" else raw)
    assert applied.checkpoint is None
    assert object_digest(applied.result) == object_digest(resulting)
    assert manifest.terminal_state == object_digest(resulting)
    nonempty_checkpoint, _manifest = pack_replication(b"ignored", (mutation,))
    with pytest.raises(ValueError, match="empty bytes"):
        apply_replication(nonempty_checkpoint, checkpoint_loader=lambda _raw: None)
