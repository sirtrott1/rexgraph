"""Canonical local stores remain opaque catalog entries without Core importing RCDB."""
from contextlib import closing

import pytest

from rexgraph.io.catalog import FileCatalog


def test_native_store_is_one_entry_and_hashes_authoritative_files(tmp_path):
    rcdb = pytest.importorskip("rcdb")
    root = tmp_path / "native"
    with closing(rcdb.LocalStore(root)) as store:
        store.put_record("value", {"number": 2**53+1})
    catalog = FileCatalog([tmp_path])
    assert [(e.name, e.kind) for e in catalog.list()] == [("root0/native", "rcdb")]
    first = catalog.hash("root0/native")
    with closing(rcdb.LocalStore(root)) as store:
        store.put_record("value", {"number": 2**53+2})
    assert catalog.hash("root0/native") != first
    assert catalog.info("root0/native").size == sum(p.stat().st_size for p in root.rglob("*")
        if p.is_file() and p.name != ".rexgraph-publication.lock")
    with pytest.raises(ValueError, match="injected loader"):
        catalog.load("root0/native")


def test_native_marker_stops_payload_discovery_even_when_incomplete(tmp_path):
    root = tmp_path / "damaged"
    (root / "blobs").mkdir(parents=True)
    (root / "store.header").write_bytes(b"damaged")
    (root / "blobs" / "payload.safetensors").write_bytes(b"private")
    catalog = FileCatalog([tmp_path])
    assert [(e.name, e.kind) for e in catalog.list()] == [("root0/damaged", "rcdb")]


def test_symlink_native_marker_does_not_claim_a_store(tmp_path):
    root = tmp_path / "ordinary"; root.mkdir()
    target = tmp_path / "outside"; target.write_bytes(b"header")
    (root / "store.header").symlink_to(target)
    assert FileCatalog([tmp_path]).list() == []


def test_native_object_is_opaque_and_hashes_head_frames_and_owned_payloads(tmp_path):
    rcdb = pytest.importorskip("rcdb")
    pytest.importorskip("fsspec")
    root = tmp_path / "objects"
    uri = "file://"+str(root)
    with closing(rcdb.NativeObjectStore(uri)) as store:
        store.put_record("value", {"large": 2**90})
    catalog = FileCatalog([tmp_path])
    assert [(e.name, e.kind) for e in catalog.list()] == [("root0/objects", "rcdb")]
    first = catalog.hash("root0/objects")
    with closing(rcdb.NativeObjectStore(uri)) as store:
        store.delete("value")
    assert catalog.hash("root0/objects") != first
    assert catalog.info("root0/objects").size == sum(p.stat().st_size for p in root.rglob("*")
        if p.is_file() and p.name != ".rexgraph-publication.lock")
    with pytest.raises(ValueError, match="injected loader"): catalog.load("root0/objects")


def test_partial_object_head_prevents_payload_discovery(tmp_path):
    root = tmp_path / "partial"
    (root / "blobs").mkdir(parents=True)
    (root / "store.head").write_bytes(b"damaged")
    (root / "blobs" / "private.safetensors").write_bytes(b"private")
    catalog = FileCatalog([tmp_path])
    assert [(e.name, e.kind) for e in catalog.list()] == [("root0/partial", "rcdb")]


@pytest.mark.parametrize("manifest", [b'{"format":"rexdb-object","version":1}', b"damaged", None])
def test_legacy_object_store_stays_opaque_and_journal_changes_its_physical_hash(tmp_path, manifest):
    root = tmp_path / "objects"
    (root / "journal").mkdir(parents=True)
    (root / "blobs").mkdir()
    if manifest is not None: (root / "MANIFEST.json").write_bytes(manifest)
    (root / "blobs" / "bound.safetensors").write_bytes(b"owned payload")
    catalog = FileCatalog([tmp_path])
    assert [(entry.name, entry.kind) for entry in catalog.list()] == [("root0/objects", "rcdb")]
    before = catalog.hash("root0/objects")
    (root / "journal" / "000000000001.json").write_bytes(b"owned metadata")
    assert catalog.hash("root0/objects") != before
    assert catalog.info("root0/objects").size == sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    with pytest.raises(ValueError, match="injected loader"): catalog.load("root0/objects")
