"""
Tests for rexgraph.io.parquet_bridge: Parquet table export/import.

All tests require pyarrow. Skipped if pyarrow is not installed.

Verifies:
    - Generic write/read roundtrip: 1D, 2D arrays, metadata
    - Boundary table: write/read reconstructs boundary_ptr/boundary_idx
    - Edge table: correct columns, nE rows
    - Vertex table: correct columns, nV rows
    - Face table: write/read reconstructs B2 CSC arrays
    - Persistence table: pairs roundtrip with betti metadata
    - Filtration table: splits back into filt_v, filt_e, filt_f
    - Metrics table: generic per-cell roundtrip
"""

import base64
import hashlib
import hmac

import numpy as np
import pytest

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

pytestmark = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")

if HAS_PYARROW:
    from rexgraph.io.parquet_bridge import (
        parquet_encryption_properties,
        read_boundary_table,
        read_character_table,
        read_edge_table,
        read_face_table,
        read_filtration_table,
        read_metrics_table,
        read_parquet,
        read_persistence_table,
        read_vertex_character_table,
        read_vertex_table,
        read_void_table,
        write_boundary_table,
        write_character_table,
        write_edge_table,
        write_face_table,
        write_filtration_table,
        write_metrics_table,
        write_parquet,
        write_persistence_table,
        write_vertex_character_table,
        write_vertex_table,
        write_void_table,
    )

from rexgraph.graph import RexGraph


if HAS_PYARROW:
    try:
        from pyarrow.parquet.encryption import CryptoFactory, KmsClient, KmsConnectionConfig
        HAS_PARQUET_ENCRYPTION = True
    except ImportError:
        HAS_PARQUET_ENCRYPTION = False
else:
    HAS_PARQUET_ENCRYPTION = False

if HAS_PARQUET_ENCRYPTION:
    class _TestKmsClient(KmsClient):
        """Authenticated reversible wrapper for PME tests, never production crypto."""

        def __init__(self, master_keys):
            super().__init__()
            self._master_keys = master_keys

        def wrap_key(self, key_bytes, master_key_identifier):
            master = self._master_keys[master_key_identifier]
            mask = hashlib.sha256(master + b"rexgraph-pme-test").digest()
            wrapped = bytes(value ^ mask[i % len(mask)]
                            for i, value in enumerate(key_bytes))
            tag = hmac.new(master, wrapped, hashlib.sha256).digest()
            return base64.b64encode(tag + wrapped)

        def unwrap_key(self, wrapped_key, master_key_identifier):
            master = self._master_keys[master_key_identifier]
            payload = base64.b64decode(wrapped_key)
            tag, wrapped = payload[:32], payload[32:]
            expected = hmac.new(master, wrapped, hashlib.sha256).digest()
            if not hmac.compare_digest(tag, expected):
                raise ValueError("wrong test master key")
            mask = hashlib.sha256(master + b"rexgraph-pme-test").digest()
            return bytes(value ^ mask[i % len(mask)]
                         for i, value in enumerate(wrapped))


def _test_crypto(master_keys=None):
    keys = master_keys or {
        "footer-key": b"f" * 32,
        "column-key": b"c" * 32,
    }
    factory = CryptoFactory(lambda _config: _TestKmsClient(keys))
    connection = KmsConnectionConfig(kms_instance_id="rexgraph-tests")
    return factory, connection


def _test_properties(*, columns, plaintext_footer=False, master_keys=None):
    factory, connection = _test_crypto(master_keys)
    encryption = parquet_encryption_properties(
        factory,
        connection,
        footer_key="footer-key",
        column_keys={"column-key": list(columns)},
        plaintext_footer=plaintext_footer,
    )
    decryption = factory.file_decryption_properties(connection)
    return encryption, decryption


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
def tmp_path_pq(tmp_path):
    return str(tmp_path / "test.parquet")


# Generic write/read

class TestGenericParquet:

    def test_1d_roundtrip(self, tmp_path_pq):
        data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
        write_parquet(data, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        assert np.allclose(loaded["x"], data["x"])
        assert np.allclose(loaded["y"], data["y"])

    def test_2d_roundtrip(self, tmp_path_pq):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        write_parquet({"mat": arr}, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        assert loaded["mat"].shape == (3, 2)
        assert np.allclose(loaded["mat"], arr)

    def test_metadata(self, tmp_path_pq):
        write_parquet({"x": np.ones(3)}, tmp_path_pq,
                      metadata={"version": 2})
        # Just verify it doesn't crash; metadata is in schema

    def test_column_projection_is_pushed_into_pyarrow(self, tmp_path_pq, monkeypatch):
        import pyarrow.parquet as pq

        write_parquet({"x": np.arange(3), "secret": np.arange(3) + 10}, tmp_path_pq)
        real_read_table = pq.read_table
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("columns"))
            return real_read_table(*args, **kwargs)

        monkeypatch.setattr(pq, "read_table", spy)
        loaded = read_parquet(tmp_path_pq, columns=["x"])
        assert seen == [["x"]]
        assert set(loaded) == {"x"}

    def test_logical_2d_projection_expands_to_physical_columns(self, tmp_path_pq, monkeypatch):
        import pyarrow.parquet as pq

        mat = np.arange(12, dtype=np.float64).reshape(4, 3)
        write_parquet({"mat": mat, "secret": np.arange(4)}, tmp_path_pq)
        real_read_table = pq.read_table
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("columns"))
            return real_read_table(*args, **kwargs)

        monkeypatch.setattr(pq, "read_table", spy)
        loaded = read_parquet(tmp_path_pq, columns=["mat"])
        assert seen == [["mat_0", "mat_1", "mat_2"]]
        assert set(loaded) == {"mat"}
        assert np.array_equal(loaded["mat"], mat)

    def test_mixed_projection_reads_only_requested_physical_columns(self, tmp_path_pq, monkeypatch):
        import pyarrow.parquet as pq

        mat = np.arange(8).reshape(4, 2)
        write_parquet({"x": np.arange(4), "mat": mat, "secret": np.arange(4)}, tmp_path_pq)
        real_read_table = pq.read_table
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("columns"))
            return real_read_table(*args, **kwargs)

        monkeypatch.setattr(pq, "read_table", spy)
        loaded = read_parquet(tmp_path_pq, columns=["x", "mat"])
        assert seen == [["x", "mat_0", "mat_1"]]
        assert set(loaded) == {"x", "mat"}

    def test_physical_2d_component_can_be_projected_directly(self, tmp_path_pq, monkeypatch):
        import pyarrow.parquet as pq

        mat = np.arange(8).reshape(4, 2)
        write_parquet({"mat": mat}, tmp_path_pq)
        real_read_table = pq.read_table
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("columns"))
            return real_read_table(*args, **kwargs)

        monkeypatch.setattr(pq, "read_table", spy)
        loaded = read_parquet(tmp_path_pq, columns=["mat_1"])
        assert seen == [["mat_1"]]
        assert set(loaded) == {"mat_1"}
        assert np.array_equal(loaded["mat_1"], mat[:, 1])

    @pytest.mark.parametrize("requested", [[], ["missing"]])
    def test_empty_projection_never_falls_back_to_all_columns(
            self, tmp_path_pq, monkeypatch, requested):
        import pyarrow.parquet as pq

        write_parquet({"secret": np.arange(3)}, tmp_path_pq)
        real_read_table = pq.read_table
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("columns"))
            return real_read_table(*args, **kwargs)

        monkeypatch.setattr(pq, "read_table", spy)
        loaded = read_parquet(tmp_path_pq, columns=requested)
        assert seen == [[]]
        assert loaded == {}


@pytest.mark.skipif(
    not HAS_PARQUET_ENCRYPTION,
    reason="PyArrow was built without Parquet encryption",
)
class TestParquetModularEncryption:

    def test_plaintext_defaults_remain_compatible(self, tmp_path_pq):
        data = {"x": np.arange(4), "y": np.arange(4) + 10}
        write_parquet(data, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        assert np.array_equal(loaded["x"], data["x"])
        assert np.array_equal(loaded["y"], data["y"])

    def test_encrypted_footer_is_default_and_roundtrips_with_properties(self, tmp_path_pq):
        import pyarrow.parquet as pq

        encryption, decryption = _test_properties(columns=["secret"])
        write_parquet(
            {"public": np.arange(4), "secret": np.arange(4) + 10},
            tmp_path_pq,
            metadata={"classification": "test"},
            encryption_properties=encryption,
        )

        with pytest.raises(OSError, match="encrypted metadata"):
            pq.read_schema(tmp_path_pq)
        with pytest.raises(OSError, match="encrypted metadata"):
            read_parquet(tmp_path_pq)
        loaded = read_parquet(tmp_path_pq, decryption_properties=decryption)
        assert set(loaded) == {"public", "secret"}
        schema = pq.read_schema(tmp_path_pq, decryption_properties=decryption)
        assert b"classification" in schema.metadata[b"rex_metadata"]

    def test_plaintext_footer_exposes_schema_but_not_encrypted_column(self, tmp_path_pq):
        import pyarrow.parquet as pq

        encryption, decryption = _test_properties(
            columns=["secret"],
            plaintext_footer=True,
        )
        write_parquet(
            {"public": np.arange(4), "secret": np.arange(4) + 10},
            tmp_path_pq,
            metadata={"distribution": True},
            encryption_properties=encryption,
        )

        schema = pq.read_schema(tmp_path_pq)
        assert schema.names == ["public", "secret"]
        assert b"distribution" in schema.metadata[b"rex_metadata"]
        public = read_parquet(tmp_path_pq, columns=["public"])
        assert set(public) == {"public"}
        with pytest.raises(OSError, match="Cannot decrypt"):
            read_parquet(tmp_path_pq, columns=["secret"])
        secret = read_parquet(
            tmp_path_pq,
            columns=["secret"],
            decryption_properties=decryption,
        )
        assert np.array_equal(secret["secret"], np.arange(4) + 10)

    def test_wrong_key_is_refused(self, tmp_path_pq):
        encryption, _decryption = _test_properties(columns=["secret"])
        write_parquet(
            {"secret": np.arange(4)},
            tmp_path_pq,
            encryption_properties=encryption,
        )
        wrong_keys = {
            "footer-key": b"x" * 32,
            "column-key": b"y" * 32,
        }
        wrong_factory, wrong_connection = _test_crypto(wrong_keys)
        wrong_decryption = wrong_factory.file_decryption_properties(wrong_connection)
        with pytest.raises((OSError, ValueError), match="wrong test master key"):
            read_parquet(tmp_path_pq, decryption_properties=wrong_decryption)

    def test_ciphertext_tamper_is_refused(self, tmp_path):
        path = tmp_path / "tampered.parquet"
        encryption, decryption = _test_properties(columns=["secret"])
        write_parquet(
            {"secret": np.arange(1000)},
            path,
            encryption_properties=encryption,
        )
        payload = bytearray(path.read_bytes())
        payload[len(payload) // 2] ^= 1
        path.write_bytes(payload)
        with pytest.raises(OSError):
            read_parquet(path, decryption_properties=decryption)

    def test_logical_2d_projection_expands_under_encryption(self, tmp_path_pq):
        mat = np.arange(12, dtype=np.float64).reshape(4, 3)
        encryption, decryption = _test_properties(
            columns=["mat_0", "mat_1", "mat_2"],
            plaintext_footer=True,
        )
        write_parquet(
            {"public": np.arange(4), "mat": mat},
            tmp_path_pq,
            encryption_properties=encryption,
        )
        assert set(read_parquet(tmp_path_pq, columns=["public"])) == {"public"}
        with pytest.raises(OSError, match="Cannot decrypt"):
            read_parquet(tmp_path_pq, columns=["mat"])
        loaded = read_parquet(
            tmp_path_pq,
            columns=["mat"],
            decryption_properties=decryption,
        )
        assert np.array_equal(loaded["mat"], mat)

    def test_encrypted_batch_read(self, tmp_path_pq):
        from rexgraph.io.parquet_bridge import read_parquet_batches

        encryption, decryption = _test_properties(columns=["secret"])
        write_parquet(
            {"secret": np.arange(9)},
            tmp_path_pq,
            encryption_properties=encryption,
        )
        with pytest.raises(OSError, match="encrypted metadata"):
            list(read_parquet_batches(tmp_path_pq, batch_rows=3))
        batches = list(read_parquet_batches(
            tmp_path_pq,
            batch_rows=3,
            decryption_properties=decryption,
        ))
        assert np.array_equal(
            np.concatenate([batch["secret"] for batch in batches]),
            np.arange(9),
        )

    def test_typed_boundary_table_passes_properties_both_ways(self, k4, tmp_path_pq):
        encryption, decryption = _test_properties(columns=["vertex_idx"])
        write_boundary_table(k4, tmp_path_pq, encryption_properties=encryption)
        with pytest.raises(OSError, match="encrypted metadata"):
            read_boundary_table(tmp_path_pq)
        loaded = read_boundary_table(
            tmp_path_pq,
            decryption_properties=decryption,
        )
        assert np.array_equal(loaded["boundary_ptr"], k4._boundary_ptr)

    def test_character_and_void_writers_do_not_bypass_encryption(
            self, k4, triangle, tmp_path):
        cases = [
            (write_character_table, read_character_table, k4, "edge_idx"),
            (write_vertex_character_table, read_vertex_character_table, k4, "vertex_idx"),
            (write_void_table, read_void_table, triangle, "void_idx"),
            (write_void_table, read_void_table, k4, "void_idx"),  # empty-table branch
        ]
        for i, (writer, reader, rex, protected_column) in enumerate(cases):
            path = tmp_path / f"direct_{i}.parquet"
            encryption, decryption = _test_properties(columns=[protected_column])
            writer(rex, path, encryption_properties=encryption)
            with pytest.raises(OSError, match="encrypted metadata"):
                reader(path)
            loaded = reader(path, decryption_properties=decryption)
            assert protected_column in loaded

    def test_builder_rejects_an_empty_column_policy(self):
        factory, connection = _test_crypto()
        with pytest.raises(ValueError, match="column_keys"):
            parquet_encryption_properties(
                factory,
                connection,
                footer_key="footer-key",
                column_keys={},
            )


# Boundary Table

class TestBoundaryTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        write_boundary_table(k4, tmp_path_pq)
        loaded = read_boundary_table(tmp_path_pq)
        assert "boundary_ptr" in loaded
        assert "boundary_idx" in loaded
        bp_orig = np.asarray(k4._boundary_ptr)
        bp_loaded = loaded["boundary_ptr"]
        assert np.array_equal(bp_orig, bp_loaded)

    def test_correct_entry_count(self, k4, tmp_path_pq):
        write_boundary_table(k4, tmp_path_pq)
        loaded = read_parquet(tmp_path_pq)
        # Each standard edge has 2 boundary vertices
        assert loaded["edge_idx"].shape[0] == int(k4._boundary_ptr[-1])


# Edge Table

class TestEdgeTable:

    def test_columns_present(self, k4, tmp_path_pq):
        write_edge_table(k4, tmp_path_pq)
        loaded = read_edge_table(tmp_path_pq)
        for col in ["edge_idx", "source", "target", "boundary_size", "edge_type"]:
            assert col in loaded

    def test_row_count(self, k4, tmp_path_pq):
        write_edge_table(k4, tmp_path_pq)
        loaded = read_edge_table(tmp_path_pq)
        assert loaded["edge_idx"].shape[0] == k4.nE


# Vertex Table

class TestVertexTable:

    def test_columns_present(self, triangle, tmp_path_pq):
        write_vertex_table(triangle, tmp_path_pq)
        loaded = read_vertex_table(tmp_path_pq)
        assert "vertex_idx" in loaded

    def test_row_count(self, triangle, tmp_path_pq):
        write_vertex_table(triangle, tmp_path_pq)
        loaded = read_vertex_table(tmp_path_pq)
        assert loaded["vertex_idx"].shape[0] == triangle.nV


# Face Table

class TestFaceTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        write_face_table(k4, tmp_path_pq)
        loaded = read_face_table(tmp_path_pq)
        assert "B2_col_ptr" in loaded
        assert "B2_row_idx" in loaded
        assert "B2_vals" in loaded

    def test_empty_faces(self, triangle, tmp_path_pq):
        """Triangle with no faces writes an empty table."""
        write_face_table(triangle, tmp_path_pq)
        loaded = read_face_table(tmp_path_pq)
        assert loaded["nF"] == 0


# Persistence Table

class TestPersistenceTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        fv, fe, ff = k4.filtration(kind="dimension")
        result = k4.persistence(fv, fe, ff)
        write_persistence_table(result, tmp_path_pq)
        loaded = read_persistence_table(tmp_path_pq)
        assert "birth" in loaded
        assert "death" in loaded
        assert "dim" in loaded
        if "betti" in loaded:
            assert isinstance(loaded["betti"], tuple)


# Filtration Table

class TestFiltrationTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        fv = np.zeros(k4.nV, dtype=np.float64)
        fe = np.ones(k4.nE, dtype=np.float64)
        ff = np.full(k4.nF, 2.0, dtype=np.float64)
        write_filtration_table(k4, fv, fe, ff, tmp_path_pq, kind="dimension")
        loaded = read_filtration_table(tmp_path_pq)
        assert np.allclose(loaded["filt_v"], fv)
        assert np.allclose(loaded["filt_e"], fe)
        assert np.allclose(loaded["filt_f"], ff)
        assert loaded["kind"] == "dimension"


# Metrics Table

class TestMetricsTable:

    def test_roundtrip(self, tmp_path_pq):
        metrics = {
            "pagerank": np.array([0.1, 0.2, 0.3, 0.4]),
            "clustering": np.array([1.0, 0.5, 0.5, 1.0]),
        }
        write_metrics_table(metrics, tmp_path_pq)
        loaded = read_metrics_table(tmp_path_pq)
        assert "pagerank" in loaded
        assert np.allclose(loaded["pagerank"], metrics["pagerank"])
        assert "cell_idx" not in loaded  # excluded by default

    def test_empty_raises(self, tmp_path_pq):
        with pytest.raises(ValueError):
            write_metrics_table({}, tmp_path_pq)


# RCF Character and Void tables (v2)

class TestCharacterTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        chi = np.asarray(k4.structural_character, dtype=np.float64)
        write_character_table(k4, tmp_path_pq)
        loaded = read_character_table(tmp_path_pq)
        assert "edge_idx" in loaded
        assert loaded["edge_idx"].shape[0] == k4.nE
        chi_cols = [c for c in loaded if c.startswith("chi_")]
        assert len(chi_cols) == chi.shape[1]
        recon = np.column_stack([loaded[c] for c in chi_cols])
        assert np.allclose(recon, chi)


class TestVertexCharacterTable:

    def test_roundtrip(self, k4, tmp_path_pq):
        phi = np.asarray(k4.vertex_character, dtype=np.float64)
        kappa = np.asarray(k4.coherence, dtype=np.float64)
        write_vertex_character_table(k4, tmp_path_pq)
        loaded = read_vertex_character_table(tmp_path_pq)
        assert loaded["vertex_idx"].shape[0] == k4.nV
        assert np.allclose(loaded["kappa"], kappa)
        phi_cols = [c for c in loaded if c.startswith("phi_")]
        assert len(phi_cols) == phi.shape[1]
        recon = np.column_stack([loaded[c] for c in phi_cols])
        assert np.allclose(recon, phi)


class TestVoidTable:

    def test_roundtrip(self, triangle, tmp_path_pq):
        vc = triangle.void_complex
        assert vc.get("n_voids", 0) > 0  # fixture must exercise the value path
        eta = np.asarray(vc["eta"], dtype=np.float64)
        write_void_table(triangle, tmp_path_pq)
        loaded = read_void_table(tmp_path_pq)
        assert loaded["void_idx"].shape[0] == vc["n_voids"]
        assert np.allclose(loaded["eta"], eta)

    def test_empty_roundtrip(self, k4, tmp_path_pq):
        """k4 has no voids -> empty-schema write still round-trips."""
        assert k4.void_complex.get("n_voids", 0) == 0
        write_void_table(k4, tmp_path_pq)
        loaded = read_void_table(tmp_path_pq)
        assert loaded["void_idx"].shape[0] == 0
