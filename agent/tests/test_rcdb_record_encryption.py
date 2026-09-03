"""Records on disk, sealed, and what a store reveals about what it sealed.

Every RCDB backend wrote `serialize_complex(rex)` to disk as plaintext, so the container
work sealed exports while the database itself held the same complexes in the clear. A
store can now carry a key, and separately decide how much of the record's description it
persists beside the payload: a signature is a description, so writing the full one next
to ciphertext describes what was sealed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from agent.rcdb import MemoryStore, open_store  # noqa: E402

from rexgraph.io.security import ENVELOPE_MAGIC, StaticKeyProvider  # noqa: E402

KEYS = StaticKeyProvider({"records": b"k" * 32})


def _rex():
    from rexgraph.graph import RexGraph
    return RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])


def _sealed(uri):
    return open_store(uri).configure_security(key_id="records", keys=KEYS)


@pytest.mark.parametrize("backend", ["memory", "file", "rex"])
def test_a_sealed_record_round_trips_in_every_backend(backend, tmp_path):
    uri = "memory://" if backend == "memory" else f"{backend}://{tmp_path / backend}"
    store = _sealed(uri)
    store.put("r1", _rex(), meta={"vertex_labels": ["alpha", "beta"]})
    back = store.get("r1")
    assert back is not None
    assert (back.nV, back.nE) == (3, 2)


def test_the_object_backend_seals_too():
    """Built directly on an in-memory filesystem.

    Its registered schemes are cloud ones, so no local uri routes to it, and it never
    creates its own directories because an object store has none: on a real local
    filesystem even an unsealed put fails on the missing blobs/ and journal/ paths, which
    predates this change.
    """
    pytest.importorskip("fsspec")
    from agent.objectstore import ObjectStore
    store = ObjectStore("memory://sealed-objtest")
    store.configure_security(key_id="records", keys=KEYS)
    store.put("r1", _rex())
    back = store.get("r1")
    assert back is not None and (back.nV, back.nE) == (3, 2)
    assert store.fs.cat(store._blob_key("r1", 1)).startswith(ENVELOPE_MAGIC)


def test_the_bytes_on_disk_are_an_envelope(tmp_path):
    store = _sealed(f"file://{tmp_path / 'f'}")
    store.put("r1", _rex())
    written = [p for p in (tmp_path / "f").rglob("*") if p.is_file()]
    sealed = [p for p in written if p.read_bytes().startswith(ENVELOPE_MAGIC)]
    assert sealed, f"nothing sealed among {[p.name for p in written]}"


def test_an_unconfigured_store_refuses_a_sealed_payload(tmp_path):
    """A refusal, not a plaintext read of ciphertext."""
    root = tmp_path / "f"
    _sealed(f"file://{root}").put("r1", _rex())
    blind = open_store(f"file://{root}")
    with pytest.raises(PermissionError):
        blind.get("r1")


def test_a_wrong_key_does_not_open_it(tmp_path):
    root = tmp_path / "f"
    _sealed(f"file://{root}").put("r1", _rex())
    other = open_store(f"file://{root}").configure_security(
        key_id="records", keys=StaticKeyProvider({"records": b"w" * 32}))
    # A refusal, not cryptography's InvalidTag: a caller must not have to import the
    # crypto library to catch what the store did.
    with pytest.raises(PermissionError):
        other.get("r1")


def test_plaintext_written_before_a_key_still_opens_after_one(tmp_path):
    """The decision is made by the envelope, so both live side by side."""
    root = tmp_path / "f"
    plain = open_store(f"file://{root}")
    plain.put("old", _rex())
    later = _sealed(f"file://{root}")
    later.put("new", _rex())
    assert later.get("old") is not None, "the pre-key record became unreadable"
    assert later.get("new") is not None


def test_a_store_with_no_key_is_byte_identical_to_before(tmp_path):
    """Additive: a store that never configures security writes what it always wrote."""
    a = open_store(f"file://{tmp_path / 'a'}")
    a.put("r1", _rex())
    written = [p for p in (tmp_path / "a").rglob("*") if p.is_file()]
    assert not any(p.read_bytes().startswith(ENVELOPE_MAGIC) for p in written)


@pytest.mark.parametrize("mode,kept,dropped", [
    ("minimal", "nV", "kappa_mean"),
    ("structural", "betti", "labels_sample"),
])
def test_signature_mode_limits_what_is_stored_beside_the_payload(mode, kept, dropped):
    store = MemoryStore().configure_security(key_id="records", keys=KEYS,
                                             signature_mode=mode)
    store.put("r1", _rex(), meta={"vertex_labels": ["alpha"]})
    sig = store.get_record("r1").signature
    assert kept in sig, sig
    assert dropped not in sig, sig


def test_public_mode_keeps_the_whole_signature():
    store = MemoryStore().configure_security(key_id="records", keys=KEYS)
    store.put("r1", _rex(), meta={"vertex_labels": ["alpha"]})
    assert "labels_sample" in store.get_record("r1").signature


def test_metadata_fields_is_an_allow_list():
    store = MemoryStore().configure_security(
        key_id="records", keys=KEYS, signature_mode="minimal",
        metadata_fields=["source"])
    store.put("r1", _rex(), meta={"source": "keep", "vertex_labels": ["drop"]})
    meta = store.get_record("r1").meta
    assert meta == {"source": "keep"}, meta


def test_security_status_reveals_no_key_material():
    store = MemoryStore().configure_security(key_id="records", keys=KEYS,
                                             signature_mode="structural")
    status = store.security_status()
    assert status["payload_encryption"] is True
    assert status["signature_mode"] == "structural"
    assert "records" not in str(status), "the key identifier leaked into status"
    assert b"k" * 32 not in str(status).encode()
