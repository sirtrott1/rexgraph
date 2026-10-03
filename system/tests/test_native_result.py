"""System previews and artifact downloads come from the same RCQL execution."""
from fractions import Fraction as Q

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from rexgraph import RexGraph
from rcql import Result
from system.server.app import app
from system.state import sources


def test_system_native_result_is_decodable_and_preserves_exact_query_output():
    sources.register("portable_test", RexGraph.from_graph([0], [1]))
    try:
        with TestClient(app) as client:
            response = client.post("/api/query", json={"query": "FROM $portable_test RETURN 1 / 7", "result_format": "native"})
            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "application/vnd.rexgraph.rcql-result"
            assert response.headers["content-disposition"] == 'attachment; filename="result.rgqr"'
            decoded = Result.from_bytes(response.content)
            assert decoded.values == (Q(1, 7),)
            assert decoded.provenance and decoded.exactness[0].value == "rational"
            preview = client.post("/api/query", json={"query": "FROM $portable_test RETURN 1 / 7"})
            assert preview.status_code == 200
            assert preview.json()["values"] == [{"numerator": 1, "denominator": 7}]
    finally:
        sources.remove("portable_test")


def test_system_partition_download_retains_lineage_and_native_result_state():
    sources.register("portable_test", RexGraph.from_graph([0, 1], [1, 2]))
    try:
        with TestClient(app) as client:
            response = client.post("/api/query", json={"query": "FROM $portable_test RETURN PARTITION(CELL(1, 0))", "result_format": "native"})
            assert response.status_code == 200, response.text
            partition = Result.from_bytes(response.content).values[0]
            partition.check_state()
            assert partition.rex.nE == 1 and partition.cell_maps[1] == (0,)
    finally:
        sources.remove("portable_test")


def test_system_glue_preview_and_download_share_verified_original_basis():
    root = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]],
        [[(0, 1), (1, 1), (2, -1)]]])
    sources.register("portable_test", root)
    query = ('FROM $portable_test LET a=PARTITION(CELL(1,0)) '
             'LET b=PARTITION(CELL(1,1)) RETURN GLUE_PARTITIONS([a,b])')
    try:
        with TestClient(app) as client:
            preview = client.post("/api/query", json={"query": query})
            assert preview.status_code == 200, preview.text
            value = preview.json()["values"][0]
            assert value["kind"] == "RexPartition" and value["lineage"]["parent_count"] == 2
            assert value["cell_maps"][2] == {"grade": 2, "count": 0, "indices": [], "truncated": False}
            downloaded = client.post("/api/query", json={"query": query, "result_format": "native"})
            assert downloaded.status_code == 200, downloaded.text
            part = Result.from_bytes(downloaded.content).values[0]
            assert part.lineage.digest == value["lineage"]["digest"]
            assert part.lineage.verify(root, part.rex)
    finally:
        sources.remove("portable_test")


@pytest.mark.parametrize("kind", ["memory", "local", "sql", "object-file", "object-memory"])
def test_system_download_persists_through_native_rcdb_and_returns_the_same_result(kind, tmp_path):
    from contextlib import closing
    from rcdb import LocalStore, MemoryStore, NativeObjectStore, SQLStore
    from rcql import register_result_storage_codec
    if kind == "sql": pytest.importorskip("sqlalchemy")
    if kind.startswith("object-"): pytest.importorskip("fsspec")
    codec = register_result_storage_codec()
    store = (MemoryStore() if kind == "memory" else LocalStore(tmp_path / "records") if kind == "local"
             else NativeObjectStore(f"{kind[7:]}://{tmp_path / 'records'}") if kind.startswith("object-")
             else SQLStore(f"sqlite:///{tmp_path / 'records.sqlite'}"))
    sources.register("portable_test", RexGraph.from_graph([0], [1]))
    try:
        with closing(store), TestClient(app) as client:
            response = client.post("/api/query", json={"query": "FROM $portable_test RETURN 1 / 7",
                                                       "result_format": "native"})
            assert response.status_code == 200, response.text
            result = Result.from_bytes(response.content)
            store.put_record("query/result", result, codec=codec)
            loaded = store.read_record("query/result")
            assert loaded.value.to_bytes() == response.content
            assert loaded.value.exactness[0].value == "rational" and loaded.value.values == (Q(1, 7),)
            assert loaded.state_digest == loaded.record.envelope.object_digest
    finally:
        sources.remove("portable_test")
