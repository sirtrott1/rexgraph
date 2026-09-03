"""
Tests for rexgraph.io.bundle: .rex bundle format.

No heavy dependencies (only numpy and json). Uses temporary directories.

Verifies:
    - save/load roundtrip: RexGraph reconstructed with correct nV/nE/nF
    - MANIFEST.json: correct magic, object_type, metadata
    - Array access: bundle["boundary_ptr"], __contains__, list_arrays
    - Cache: written to cache/ subdirectory, readable
    - Weighted graph: w_E preserved
    - TemporalRex roundtrip
    - RexBundle.from_graph / .save / .load / .to_object
    - Memory-map mode
"""
import json
import multiprocessing
import os
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import rexgraph.io.bundle as bundle_module
from rexgraph.graph import RexGraph
from rexgraph.io import ContainerEncryptionConfig, load, save
from rexgraph.io._container_crypto import ContainerEncryptionError
from rexgraph.io.bundle import (
    RexBundle,
    load_rex,
    save_rex,
)


class _BundleAeadProperties:
    """Test-only opaque AEAD property; core never receives the keys."""

    authenticated_encryption = True

    def __init__(self, configuration, keys):
        self.configuration = configuration
        self._keys = dict(keys)
        self.open_calls: list[str] = []

    def seal(self, key_id, plaintext, aad):
        AESGCM = pytest.importorskip(
            "cryptography.hazmat.primitives.ciphers.aead"
        ).AESGCM
        nonce = os.urandom(12)
        encoded = key_id.encode("utf-8")
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
        self.open_calls.append(key_id)
        return AESGCM(self._keys[key_id]).decrypt(
            nonce,
            envelope[14 + size:],
            aad,
        )

    def open_with(self, key_id, envelope, aad):
        size = int.from_bytes(envelope[:2], "little")
        encoded = envelope[2:2 + size].decode("utf-8")
        if encoded != key_id:
            raise PermissionError("authenticated manifest chose the wrong key")
        return self.open(envelope, aad)


def _bundle_keys(*names):
    AESGCM = pytest.importorskip(
        "cryptography.hazmat.primitives.ciphers.aead"
    ).AESGCM
    return {name: AESGCM.generate_key(bit_length=256) for name in names}


def _bundle_properties(
    *,
    tensor_keys=None,
    plaintext_tensors=(),
    plaintext_manifest=False,
    keys=None,
):
    tensor_keys = tensor_keys or {}
    keys = keys or _bundle_keys("footer", *tensor_keys)
    config = ContainerEncryptionConfig(
        footer_key="footer",
        tensor_keys=tensor_keys,
        plaintext_tensors=plaintext_tensors,
        plaintext_manifest=plaintext_manifest,
        chunk_size=4096,
    )
    return _BundleAeadProperties(config, keys), keys


def _storage_files(path, suffix=None):
    files = sorted((path / "__rex_encrypted_storage__").iterdir())
    if suffix is not None:
        files = [file for file in files if file.suffix == suffix]
    return files


def _process_bundle_writer(path, barrier, keys, edge_count):
    properties, _ = _bundle_properties(keys=keys)
    sources = np.arange(edge_count, dtype=np.int32)
    targets = np.roll(sources, -1)
    barrier.wait()
    save_rex(path, RexGraph.from_graph(sources, targets),
             encryption_properties=properties)

# Fixtures

@pytest.fixture
def k4():
    return RexGraph.from_simplicial(
        sources=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
        targets=np.array([1, 2, 3, 2, 3, 3], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


@pytest.fixture
def triangle():
    return RexGraph.from_graph([0, 1, 0], [1, 2, 2])


@pytest.fixture
def rex_path(tmp_path):
    return str(tmp_path / "test.rex")


# Basic Roundtrip

class TestRoundtrip:

    def test_basic(self, k4, rex_path):
        save_rex(rex_path, k4)
        loaded = load_rex(rex_path)
        assert isinstance(loaded, RexGraph)
        assert loaded.nV == k4.nV
        assert loaded.nE == k4.nE
        assert loaded.nF == k4.nF

    def test_betti_preserved(self, k4, rex_path):
        save_rex(rex_path, k4)
        loaded = load_rex(rex_path)
        assert loaded.betti == k4.betti

    def test_weighted(self, tmp_path):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2], w_E=w)
        path = str(tmp_path / "weighted.rex")
        save_rex(path, rex)
        loaded = load_rex(path)
        assert np.allclose(loaded._w_E, w)

    def test_suffix_added(self, k4, tmp_path):
        """Saves with .rex suffix even if not provided."""
        path = str(tmp_path / "nosuffix")
        save_rex(path, k4)
        assert os.path.isdir(path + ".rex")


# MANIFEST.json

class TestManifest:

    def test_magic(self, k4, rex_path):
        save_rex(rex_path, k4)
        mf = json.loads(pathlib.Path(rex_path, "MANIFEST.json").read_text())
        assert mf["magic"] == "rex-bundle"

    def test_object_type(self, k4, rex_path):
        save_rex(rex_path, k4)
        mf = json.loads(pathlib.Path(rex_path, "MANIFEST.json").read_text())
        assert mf["object_type"] == "RexGraph"
        assert mf["nV"] == k4.nV
        assert mf["nE"] == k4.nE


# Array Access

class TestArrayAccess:

    def test_getitem(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        bp = bundle["boundary_ptr"]
        assert bp.shape == (k4.nE + 1,)

    def test_contains(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        assert "boundary_ptr" in bundle
        assert "nonexistent" not in bundle

    def test_list_arrays(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        names = bundle.list_arrays()
        assert "boundary_ptr" in names
        assert "boundary_idx" in names

    def test_missing_raises(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        with pytest.raises(KeyError):
            bundle["nonexistent_array"]


# Cache

class TestCache:

    def test_topology_cache(self, k4, rex_path):
        save_rex(rex_path, k4, cache=["topology"])
        bundle = RexBundle.load(rex_path)
        cache = bundle.read_cache()
        # Betti should be in scalar cache
        assert "betti" in cache or "edge_types" in bundle

    def test_cache_arrays_in_subdir(self, k4, rex_path):
        save_rex(rex_path, k4, cache=["algebra"])
        assert os.path.isdir(os.path.join(rex_path, "cache"))

    def test_all_cache(self, triangle, rex_path):
        save_rex(rex_path, triangle, cache="all")
        loaded = load_rex(rex_path)
        assert loaded.nE == triangle.nE


# TemporalRex

class TestTemporalRex:

    def test_roundtrip(self, tmp_path):
        from rexgraph.graph import TemporalRex
        snaps = [
            (np.array([0, 1, 0], dtype=np.int32),
             np.array([1, 2, 2], dtype=np.int32)),
            (np.array([0, 1, 0, 1], dtype=np.int32),
             np.array([1, 2, 2, 3], dtype=np.int32)),
        ]
        trex = TemporalRex(snaps)
        path = str(tmp_path / "temporal.rex")
        save_rex(path, trex)
        loaded = load_rex(path)
        assert isinstance(loaded, TemporalRex)
        assert loaded.T == 2

    def test_snapshot_files(self, tmp_path):
        from rexgraph.graph import TemporalRex
        snaps = [
            (np.array([0, 1], dtype=np.int32),
             np.array([1, 2], dtype=np.int32)),
        ]
        trex = TemporalRex(snaps)
        path = str(tmp_path / "temporal.rex")
        save_rex(path, trex)
        assert os.path.isdir(os.path.join(path, "snapshots", "0"))


# RexBundle API

class TestRexBundleAPI:

    def test_from_graph_and_save(self, k4, rex_path):
        bundle = RexBundle.from_graph(k4)
        assert bundle.object_type == "RexGraph"
        bundle.save(rex_path)
        assert os.path.exists(rex_path)

    def test_load_and_to_object(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        rex = bundle.to_object()
        assert isinstance(rex, RexGraph)
        assert rex.nV == k4.nV

    def test_repr(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path)
        r = repr(bundle)
        assert "RexBundle" in r
        assert "RexGraph" in r

    def test_mmap_mode(self, k4, rex_path):
        save_rex(rex_path, k4)
        bundle = RexBundle.load(rex_path, mmap=True)
        bp = bundle["boundary_ptr"]
        assert bp.shape == (k4.nE + 1,)


# Signed-topology round-trip fidelity (Wave-0 correctness)

def _signed_directed_faced_graph():
    """Signed, directed 2-rex with a filled face (negative B2 orientation
    entry) AND a branching edge. Exercises the full signed-complex contract."""
    boundary_ptr = np.array([0, 2, 4, 6, 9], dtype=np.int32)
    boundary_idx = np.array([0, 1, 0, 2, 1, 2, 1, 2, 3], dtype=np.int32)
    B2_col_ptr = np.array([0, 3], dtype=np.int32)
    B2_row_idx = np.array([0, 1, 2], dtype=np.int32)
    B2_vals = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    signs = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    return RexGraph(
        boundary_ptr=boundary_ptr, boundary_idx=boundary_idx,
        B2_col_ptr=B2_col_ptr, B2_row_idx=B2_row_idx, B2_vals=B2_vals,
        signs=signs, directed=True,
    )


class TestSignedBundleRoundtrip:

    def _assert_same(self, rex, rex2):
        assert np.array_equal(rex2._boundary_ptr, rex._boundary_ptr)
        assert np.array_equal(rex2._boundary_idx, rex._boundary_idx)
        assert np.array_equal(rex2._B2_col_ptr, rex._B2_col_ptr)
        assert np.array_equal(rex2._B2_row_idx, rex._B2_row_idx)
        assert np.allclose(rex2._B2_vals, rex._B2_vals)
        assert np.any(rex2._B2_vals < 0)  # negative orientation survives
        assert rex2._signs is not None
        assert np.allclose(np.asarray(rex2._signs), np.asarray(rex._signs))
        assert np.any(np.asarray(rex2._signs) < 0)  # negative edge sign survives
        assert rex2._directed is True

    def test_rex_bundle_roundtrip(self, rex_path):
        rex = _signed_directed_faced_graph()
        save_rex(rex_path, rex)
        rex2 = load_rex(rex_path)
        self._assert_same(rex, rex2)

    def test_to_dict_from_dict_roundtrip(self):
        rex = _signed_directed_faced_graph()
        rex2 = RexGraph.from_dict(rex.to_dict())
        self._assert_same(rex, rex2)


class TestEncryptedBundle:

    def test_roundtrip_hides_names_and_missing_grade_key_refuses(self, k4, tmp_path):
        properties, keys = _bundle_properties(
            tensor_keys={
                "edge": ["boundary_ptr", "boundary_idx"],
                "face": ["B2_col_ptr", "B2_row_idx", "B2_vals"],
            }
        )
        path = tmp_path / "encrypted.rex"
        save_rex(path, k4, encryption_properties=properties)

        public = json.loads((path / "MANIFEST.json").read_text())
        assert public["encrypted"] is True
        assert set(public) == {
            "encrypted", "magic", "version", "rex_encrypted", "rex_encryption",
        }
        assert "boundary_ptr" not in public["rex_encryption"]
        assert all(file.stem.isdigit() for file in _storage_files(path))

        footer_only = _BundleAeadProperties(
            properties.configuration,
            {"footer": keys["footer"]},
        )
        with pytest.raises(PermissionError, match="authentication"):
            load_rex(path, decryption_properties=footer_only)
        edge_only = _BundleAeadProperties(
            properties.configuration,
            {"footer": keys["footer"], "edge": keys["edge"]},
        )
        with pytest.raises(PermissionError, match="authentication"):
            load_rex(path, decryption_properties=edge_only)
        wrong, _ = _bundle_properties(
            tensor_keys=properties.configuration.tensor_keys,
        )
        with pytest.raises(PermissionError, match="authentication"):
            RexBundle.load(path, decryption_properties=wrong)

        loaded = load_rex(path, decryption_properties=properties)
        assert (loaded.nV, loaded.nE, loaded.nF) == (k4.nV, k4.nE, k4.nF)

    def test_random_bundle_id_and_allow_unsealed_cannot_bypass_auth(self, k4, tmp_path):
        properties, _ = _bundle_properties()
        first = tmp_path / "first.rex"
        second = tmp_path / "second.rex"
        save_rex(first, k4, encryption_properties=properties)
        save_rex(second, k4, encryption_properties=properties)
        first_descriptor = json.loads(
            json.loads((first / "MANIFEST.json").read_text())["rex_encryption"]
        )
        second_descriptor = json.loads(
            json.loads((second / "MANIFEST.json").read_text())["rex_encryption"]
        )
        assert first_descriptor["bundle_id"] != second_descriptor["bundle_id"]
        with pytest.raises(PermissionError, match="decryption properties"):
            load_rex(first, allow_unsealed=True)

    def test_cache_plaintext_mmap_and_protected_selective_read(self, k4, tmp_path):
        properties, _ = _bundle_properties(
            tensor_keys={"edge": ["boundary_ptr"]},
            plaintext_tensors=["cache/B1"],
        )
        path = tmp_path / "cache.rex"
        save_rex(path, k4, cache=["B1"], encryption_properties=properties)
        bundle = RexBundle.load(
            path,
            mmap=True,
            decryption_properties=properties,
        )
        assert "boundary_ptr" in bundle and "B1" in bundle
        assert isinstance(bundle["B1"], np.memmap)
        with pytest.raises(ValueError, match="read_slice"):
            bundle["boundary_ptr"]

        properties.open_calls.clear()
        selected = bundle.read_slice("boundary_ptr", slice(0, 2))
        np.testing.assert_array_equal(selected, k4._boundary_ptr[:2])
        assert properties.open_calls == ["edge"]
        assert "B1" in bundle.read_cache()

    def test_statistics_query_prunes_rex_chunks_and_reuses_opened_data(self, tmp_path):
        n_vertices = 2048
        boundary_ptr = np.arange(0, 2 * n_vertices + 1, 2, dtype=np.int32)
        boundary_idx = np.repeat(np.arange(n_vertices, dtype=np.int32), 2)
        weights = np.arange(n_vertices, dtype=np.float64)
        rex = RexGraph(
            boundary_ptr=boundary_ptr,
            boundary_idx=boundary_idx,
            w_E=weights,
        )
        properties, _ = _bundle_properties(
            tensor_keys={"edge": ["w_E"]},
        )
        path = tmp_path / "query.rex"
        save_rex(path, rex, encryption_properties=properties)

        properties.open_calls.clear()
        bundle = RexBundle.load(path, decryption_properties=properties)
        selected = bundle.select(
            "w_E",
            where=("w_E", ">=", 1792),
        )
        np.testing.assert_array_equal(
            selected["w_E"],
            weights[weights >= 1792],
        )
        first_calls = list(properties.open_calls)
        again = bundle.where("w_E", ">=", 1792)
        np.testing.assert_array_equal(again, np.arange(1792, 2048))
        assert properties.open_calls == first_calls
        assert first_calls == ["footer", "edge", "edge"]
        bundle.clear_query_cache()

    def test_plain_bundle_query_matches_encrypted_api(self, tmp_path):
        rex = RexGraph.from_graph([0, 2, 4], [1, 3, 0])
        path = tmp_path / "plain-query.rex"
        save_rex(path, rex)
        bundle = RexBundle.load(path)
        boundary = np.asarray(bundle["boundary_idx"])
        selected = bundle.select(
            "boundary_idx",
            where=("boundary_idx", ">=", 2),
        )
        np.testing.assert_array_equal(
            selected["boundary_idx"], boundary[boundary >= 2]
        )

    def test_tamper_drop_extra_and_cross_bundle_swap_fail(self, k4, tmp_path):
        properties, _ = _bundle_properties()
        first = tmp_path / "first.rex"
        second = tmp_path / "second.rex"
        save_rex(first, k4, encryption_properties=properties)
        save_rex(second, k4, encryption_properties=properties)
        member_name = _storage_files(first, ".rexenc")[0].name

        tampered = tmp_path / "tampered.rex"
        shutil.copytree(first, tampered)
        member = tampered / "__rex_encrypted_storage__" / member_name
        payload = bytearray(member.read_bytes())
        payload[0] ^= 1
        member.write_bytes(payload)
        with pytest.raises(ContainerEncryptionError, match="storage digest"):
            RexBundle.load(tampered, decryption_properties=properties)

        dropped = tmp_path / "dropped.rex"
        shutil.copytree(first, dropped)
        (dropped / "__rex_encrypted_storage__" / member_name).unlink()
        with pytest.raises(ContainerEncryptionError, match="inventory"):
            RexBundle.load(dropped, decryption_properties=properties)

        extra = tmp_path / "extra.rex"
        shutil.copytree(first, extra)
        (extra / "unlisted.bin").write_bytes(b"unlisted")
        with pytest.raises(ContainerEncryptionError, match="inventory"):
            RexBundle.load(extra, decryption_properties=properties)

        swapped = tmp_path / "swapped.rex"
        shutil.copytree(first, swapped)
        shutil.copyfile(
            second / "__rex_encrypted_storage__" / member_name,
            swapped / "__rex_encrypted_storage__" / member_name,
        )
        with pytest.raises(ContainerEncryptionError, match="storage digest"):
            RexBundle.load(swapped, decryption_properties=properties)

    def test_signed_manifest_and_plaintext_member_are_authenticated(self, k4, tmp_path):
        properties, _ = _bundle_properties(
            plaintext_manifest=True,
            plaintext_tensors=["boundary_ptr"],
        )
        path = tmp_path / "signed.rex"
        save_rex(path, k4, encryption_properties=properties)
        public = json.loads((path / "MANIFEST.json").read_text())
        descriptor = json.loads(public["rex_encryption"])
        assert descriptor["manifest_mode"] == "signed_plaintext"
        assert any(
            member["logical_name"] == "boundary_ptr"
            for member in descriptor["manifest"]["members"]
        )

        plaintext_file = _storage_files(path, ".npy")[0]
        array = np.load(plaintext_file, allow_pickle=False)
        array.flat[0] += 1
        np.save(plaintext_file, array)
        with pytest.raises(ContainerEncryptionError, match="storage digest"):
            RexBundle.load(path, decryption_properties=properties)

    def test_temporal_generic_registry_and_same_path_copy(self, tmp_path):
        from rexgraph.graph import TemporalRex

        temporal = TemporalRex([
            (np.array([0, 1, 2], np.int32), np.array([1, 2, 0], np.int32)),
            (np.array([0, 1, 2], np.int32), np.array([1, 2, 0], np.int32)),
        ], face_snapshots=[
            (
                np.array([0, 3], np.int32),
                np.array([0, 1, 2], np.int32),
                np.array([1.0, -1.0, 1.0]),
            ),
            (
                np.array([0, 3], np.int32),
                np.array([0, 1, 2], np.int32),
                np.array([1.0, -1.0, 1.0]),
            ),
        ])
        properties, _ = _bundle_properties()
        path = tmp_path / "temporal.rex"
        save(path, temporal, encryption_properties=properties)
        loaded = load(path, decryption_properties=properties)
        assert loaded.T == temporal.T
        assert loaded.reconstruct_at(1).nE == temporal.reconstruct_at(1).nE
        assert np.any(loaded.reconstruct_at(1)._B2_vals < 0)

        bundle = RexBundle.load(path, decryption_properties=properties)
        bundle.save(path)
        assert load_rex(path, decryption_properties=properties).T == temporal.T

    def test_concurrent_publication_leaves_one_complete_bundle(self, tmp_path):
        properties, _ = _bundle_properties()
        path = tmp_path / "race.rex"
        first = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
        second = RexGraph.from_graph([0, 1, 2, 3], [1, 2, 3, 0])
        barrier = Barrier(2)

        def write(rex):
            barrier.wait()
            save_rex(path, rex, encryption_properties=properties)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, rex) for rex in (first, second)]
            for future in futures:
                future.result()

        assert load_rex(path, decryption_properties=properties).nE in {3, 4}
        leftovers = [
            item.name for item in tmp_path.iterdir()
            if item.name.startswith(".race.rex.")
        ]
        assert leftovers == []

    def test_failed_publication_restores_the_previous_bundle(
        self,
        tmp_path,
        monkeypatch,
    ):
        properties, _ = _bundle_properties()
        path = tmp_path / "rollback.rex"
        original = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
        replacement = RexGraph.from_graph([0, 1, 2, 3], [1, 2, 3, 0])
        save_rex(path, original, encryption_properties=properties)

        replace = bundle_module.os.replace

        def fail_staging_publish(source, destination):
            source = os.fspath(source)
            if pathlib.Path(source).name.startswith(".rollback.rex.tmp-"):
                raise OSError("simulated publication failure")
            return replace(source, destination)

        monkeypatch.setattr(bundle_module.os, "replace", fail_staging_publish)
        with pytest.raises(OSError, match="simulated publication failure"):
            save_rex(path, replacement, encryption_properties=properties)

        assert load_rex(path, decryption_properties=properties).nE == original.nE
        assert not any(
            item.name.startswith(".rollback.rex.")
            for item in tmp_path.iterdir()
        )

    @pytest.mark.skipif(os.name != "posix", reason="cross-process flock is POSIX")
    @pytest.mark.filterwarnings(
        "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
    )
    def test_processes_cannot_interleave_one_directory(self, tmp_path):
        context = multiprocessing.get_context("fork")
        barrier = context.Barrier(2)
        keys = _bundle_keys("footer")
        path = tmp_path / "process-race.rex"
        processes = [
            context.Process(
                target=_process_bundle_writer,
                args=(str(path), barrier, keys, edge_count),
            )
            for edge_count in (25, 40)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
            assert process.exitcode == 0

        properties, _ = _bundle_properties(keys=keys)
        assert load_rex(path, decryption_properties=properties).nE in {25, 40}
        assert not any(
            item.name.startswith(".process-race.rex.")
            for item in tmp_path.iterdir()
        )
