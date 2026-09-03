"""Tests for the Relational Complex Database (agent.rcdb) - core backends,
structural query, and the HTTP routes."""

from contextlib import closing

import pytest
from agent.adapters.text import TextAdapter
from agent.auto import build_rex_from_edges
from agent.rcdb import (
    MemoryStore,
    SQLStore,
    open_store,
    register_backend,
    structural_signature,
)


def _rex(text):
    ec = TextAdapter().build(text, min_count=1, max_vocab=80)
    r = build_rex_from_edges(ec)
    r._agent_meta = {"vertex_labels": ec.vertex_labels, "input_type": "text"}
    return r


DENSE = _rex("cells signal receptors genes express proteins pathways regulate "
             "cell cycle tissues organs development growth division")
SPARSE = _rex("alpha beta gamma")


@pytest.fixture(params=["memory", "file", "sql"])
def backend_store(request, tmp_path):
    """An opened store per backend, closed when the test ends.

    This was a module-level _uris() built with tempfile.mkdtemp and mktemp, so the paths
    were created at COLLECTION time and left in /tmp, and every parametrization opened a
    store that nothing closed. A SQL store owns a connection pool, so those stayed open
    until the collector reached them.
    """
    kind = request.param
    if kind == "memory":
        uri = "memory://"
    elif kind == "file":
        uri = f"file://{tmp_path / 'store'}"
    else:
        uri = f"sqlite:///{tmp_path / 'store.db'}"
    store = open_store(uri)
    yield store
    store.close()


class TestBackends:
    def test_put_get_roundtrip(self, backend_store):
        st = backend_store
        st.put("d", DENSE, meta=DENSE._agent_meta, tags=["x"])
        got = st.get("d")
        assert got is not None and got.nV == DENSE.nV and got.nE == DENSE.nE

    def test_structural_query(self, backend_store):
        st = backend_store
        st.put("dense", DENSE, meta=DENSE._agent_meta, tags=["big"])
        st.put("sparse", SPARSE, meta=SPARSE._agent_meta, tags=["tiny"])
        # dense has many independent cycles; sparse has few
        big = {r.id for r in st.query(min_betti1=10)}
        assert "dense" in big and "sparse" not in big

    def test_tag_query(self, backend_store):
        st = backend_store
        st.put("a", DENSE, tags=["bio", "big"])
        st.put("b", SPARSE, tags=["toy"])
        assert {r.id for r in st.query(tags_any=["bio"])} == {"a"}
        assert {r.id for r in st.query(tags_all=["bio", "big"])} == {"a"}

    def test_list_delete_stats(self, backend_store):
        st = backend_store
        st.put("a", DENSE)
        st.put("b", SPARSE)
        assert len(st.list()) == 2
        s = st.stats()
        assert s["count"] == 2 and s["total_edges"] > 0
        assert st.delete("a") is True
        assert st.get("a") is None
        assert len(st.list()) == 1

    def test_open_store_scheme_dispatch(self, tmp_path):
        assert open_store("memory://").backend == "memory"
        assert open_store(f"file://{tmp_path / 'x_rcdb'}").backend == "file"
        with closing(open_store("sqlite:///" + str(tmp_path / "scheme.db"))) as sql:
            assert sql.backend == "sql"

    def test_register_custom_backend(self):
        register_backend("mymem", lambda uri: MemoryStore())
        assert open_store("mymem://whatever").backend == "memory"


class TestStructuralSearch:
    def _store_schema(self, client, sid, tables_fks, tags):
        spec = {"tables": tables_fks}
        return client.post("/api/v1/schema/analyze",
                           json={"spec": spec, "store_id": sid, "tags": tags})

    def test_similar_ranks_and_excludes(self, client):
        ecom = [{"name": "users", "primary_key": ["id"]},
                {"name": "orders", "foreign_keys": [{"columns": ["uid"], "references": "users"}]},
                {"name": "items", "foreign_keys": [{"columns": ["oid"], "references": "orders"}]}]
        ecom2 = [{"name": "users", "primary_key": ["id"]},
                 {"name": "orders", "foreign_keys": [{"columns": ["uid"], "references": "users"}]},
                 {"name": "payments", "foreign_keys": [{"columns": ["oid"], "references": "orders"}]}]
        blog = [{"name": "author", "primary_key": ["id"]},
                {"name": "post", "foreign_keys": [{"columns": ["aid"], "references": "author"}]}]
        self._store_schema(client, "ecom_v1", ecom, ["ecommerce"])
        self._store_schema(client, "ecom_v2", ecom2, ["ecommerce"])
        self._store_schema(client, "blog", blog, ["blog"])
        r = client.post("/api/v1/db/similar", json={"id": "ecom_v1", "top_k": 5}).json()
        ids = [m["id"] for m in r["matches"]]
        assert "ecom_v2" in ids            # structurally similar found
        assert "ecom_v1" not in ids        # self excluded
        assert "blog" not in ids           # no shared structure -> excluded
        assert 0 <= r["matches"][0]["match"] <= 1

    def test_lineage_versions_and_drift(self, client):
        v1 = [{"name": "users", "primary_key": ["id"]},
              {"name": "orders", "foreign_keys": [{"columns": ["u"], "references": "users"}]},
              {"name": "items", "foreign_keys": [{"columns": ["o"], "references": "orders"}]}]
        v2 = [{"name": "users", "primary_key": ["id"]},
              {"name": "orders", "foreign_keys": [{"columns": ["u"], "references": "users"}]},
              {"name": "payments", "foreign_keys": [{"columns": ["o"], "references": "orders"}]}]
        for tabs in (v1, v2):
            r = client.post("/api/v1/schema/analyze",
                            json={"spec": {"tables": tabs}, "lineage_id": "shop"}).json()
            assert r["version"]["lineage_id"] == "shop"
        lin = client.get("/api/v1/db/lineage/shop").json()
        assert [v["id"] for v in lin["versions"]] == ["shop@1", "shop@2"]
        assert lin["versions"][1]["parent_version"] == 1
        step = lin["trajectory"][0]
        assert step["added"] == ["payments"] and step["removed"] == ["items"]

    def test_compare_reports_drift(self, client):
        cmp = client.post("/api/v1/db/compare",
                          json={"a": "ecom_v1", "b": "ecom_v2"}).json()
        assert set(cmp["shared"]) == {"users", "orders"}
        assert cmp["only_in_a"] == ["items"]
        assert cmp["only_in_b"] == ["payments"]

    def test_clustering_groups_families(self, client):
        # ecom_v1/ecom_v2 already stored by test_similar; add two blog schemas
        for sid, extra in [("blog_v1", "comment"), ("blog_v2", "tag")]:
            client.post("/api/v1/schema/analyze", json={"store_id": sid, "tags": ["blog"],
                "spec": {"tables": [
                    {"name": "author", "primary_key": ["id"]},
                    {"name": "post", "foreign_keys": [{"columns": ["a"], "references": "author"}]},
                    {"name": extra, "foreign_keys": [{"columns": ["p"], "references": "post"}]}]}})
        r = client.post("/api/v1/db/cluster", json={"threshold": 0.7}).json()
        def cluster_of(x):
            for c in r["clusters"]:
                if x in c["members"]:
                    return frozenset(c["members"])
            return None
        # ecom_v1 and ecom_v2 land together; blog_v1 and blog_v2 land together
        assert cluster_of("ecom_v1") is not None
        assert cluster_of("ecom_v1") == cluster_of("ecom_v2")
        assert cluster_of("blog_v1") == cluster_of("blog_v2")
        # and the two families are distinct
        assert cluster_of("ecom_v1") != cluster_of("blog_v1")
        for c in r["clusters"]:
            assert c["avg_coherence"] > 0.5 and c["centroid"] in c["members"]
    def test_signature_has_topology(self):
        sig = structural_signature(DENSE, DENSE._agent_meta, tags=["t"])
        assert sig["nV"] > 0 and sig["nE"] > 0
        assert len(sig["betti"]) == 3
        assert "kappa_mean" in sig and sig["tags"] == ["t"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient
    db = tmp_path_factory.mktemp("rcdb") / "db.sqlite"
    os.environ["REXGRAPH_RCDB_URI"] = f"sqlite:///{db}"
    import agent.rcdb as R
    R.reset_default_store()  # drop the memo so the store picks up our URI
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    get_auth_manager().disable_auth(persist=False)
    with TestClient(app) as c:
        yield c
    R.reset_default_store()


class TestRoutes:
    def _upload(self, client, text):
        return client.post("/api/upload",
                           files={"file": ("d.txt", text.encode(), "text/plain")},
                           data={"options": "{}"}).json()["session_id"]

    def test_put_list_query_export(self, client):
        sid = self._upload(client, "cells signal receptors genes express proteins "
                                   "pathways regulate cell cycle tissues organs growth")
        put = client.post("/api/v1/db/put",
                          json={"id": "bio", "session_id": sid, "tags": ["biology"]})
        assert put.status_code == 200
        assert put.json()["signature"]["nV"] > 0

        info = client.get("/api/v1/db/info").json()
        assert info["count"] >= 1

        q = client.post("/api/v1/db/query", json={"tags_any": ["biology"]}).json()
        assert "bio" in [r["id"] for r in q["records"]]

        ex = client.get("/api/v1/db/export/bio")
        assert ex.status_code == 200 and len(ex.content) > 0

        assert client.delete("/api/v1/db/bio").json()["deleted"] is True

    def test_put_requires_input(self, client):
        assert client.post("/api/v1/db/put", json={}).status_code == 400


class TestSqlIndex:
    def _rexes(self):
        import numpy as np

        from rexgraph.graph import RexGraph
        cycle = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                         targets=np.array([1, 2, 0], dtype=np.int32))   # betti1=1
        tree = RexGraph(sources=np.array([0, 1], dtype=np.int32),
                        targets=np.array([2, 2], dtype=np.int32))        # betti1=0
        return cycle, tree

    def test_indexed_columns_and_pushdown(self, tmp_path):
        cycle, tree = self._rexes()
        with closing(SQLStore("sqlite:///" + str(tmp_path / "idx.db"))) as s:
            s.put("cyc", cycle, tags=["a"])
            s.put("tre", tree, tags=["b"])
            import sqlalchemy as sa
            cols = {c["name"] for c in sa.inspect(s.engine).get_columns("rc_complexes")}
            assert {"betti1", "kappa_mean", "source", "nV", "nE", "chain_valid"} <= cols
            assert len(sa.inspect(s.engine).get_indexes("rc_complexes")) >= 3
            assert [r.id for r in s.query(min_betti1=1)] == ["cyc"]
            assert [r.id for r in s.query(min_betti1=1, tags_any=["a"])] == ["cyc"]
            assert s.query(min_betti1=1, tags_any=["b"]) == []

    def test_parity_with_memory(self, tmp_path):
        cycle, tree = self._rexes()
        with closing(SQLStore("sqlite:///" + str(tmp_path / "p.db"))) as s:
            m = MemoryStore()
            for st in (s, m):
                st.put("cyc", cycle, tags=["a"])
                st.put("tre", tree, tags=["b"])
            for q in ({"min_betti1": 1}, {"max_betti1": 0}, {"tags_any": ["b"]},
                      {"chain_valid": True}):
                assert {r.id for r in s.query(**q)} == {r.id for r in m.query(**q)}

    def test_migration_backfills_old_table(self, tmp_path):
        import json
        import sqlite3
        import time
        dbf = str(tmp_path / "legacy.db")
        con = sqlite3.connect(dbf)
        con.execute("CREATE TABLE rc_complexes(id TEXT PRIMARY KEY, signature TEXT, "
                    "meta TEXT, created FLOAT, blob BLOB)")
        sig = json.dumps({"nV": 3, "nE": 3, "betti": [1, 1, 0], "kappa_mean": 0.0,
                          "chain_valid": True, "source": "legacy", "tags": ["old"]})
        con.execute("INSERT INTO rc_complexes VALUES(?,?,?,?,?)",
                    ("legacy", sig, "{}", time.time(), b"x"))
        con.commit()
        con.close()
        with closing(SQLStore("sqlite:///" + dbf)) as s:
            assert [r.id for r in s.query(min_betti1=1)] == ["legacy"]
            assert [r.id for r in s.query(source="legacy")] == ["legacy"]


class TestWorkAsComplex:
    def test_run_and_conversation_adapters(self):
        from agent.lineage_adapters import conversation_to_rex, run_to_rex
        r, meta = run_to_rex(["ocr", "corpus", "analysis", "export"])
        assert r is not None and r.nV == 4 and r.nE == 3
        assert meta["source"] == "pipeline-run"
        c, cmeta = conversation_to_rex(["q1", "a1", "q2"])
        assert c is not None and cmeta["source"] == "conversation"

    def test_record_work_and_similarity(self, client):
        client.post("/api/v1/db/record-work", json={
            "kind": "pipeline-run", "labels": ["ocr", "corpus", "analysis", "export"],
            "id": "run_A"})
        client.post("/api/v1/db/record-work", json={
            "kind": "pipeline-run", "labels": ["ocr", "corpus", "analysis", "export"],
            "id": "run_B"})
        client.post("/api/v1/db/record-work", json={
            "kind": "pipeline-run", "labels": ["upload", "validate", "reject"],
            "id": "run_C"})
        sim = client.post("/api/v1/db/similar", json={"id": "run_A"}).json()
        ids = [m["id"] for m in sim["matches"]]
        assert "run_B" in ids and "run_C" not in ids


class TestAutoLineage:
    def test_version_only_on_change(self, client):
        v1 = {"tables": [{"name": "u", "primary_key": ["id"]},
                         {"name": "o", "foreign_keys": [{"columns": ["u"], "references": "u"}]}]}
        # first store -> v1
        r1 = client.post("/api/v1/schema/analyze",
                         json={"spec": v1, "lineage_id": "auto"}).json()
        assert r1["version"]["version"] == 1 and r1["version"]["unchanged"] is False
        # re-store identical -> no new version
        r2 = client.post("/api/v1/schema/analyze",
                         json={"spec": v1, "lineage_id": "auto"}).json()
        assert r2["version"]["unchanged"] is True and r2["version"]["version"] == 1
        # store a changed schema -> v2
        v2 = {"tables": [{"name": "u", "primary_key": ["id"]},
                         {"name": "o", "foreign_keys": [{"columns": ["u"], "references": "u"}]},
                         {"name": "p", "foreign_keys": [{"columns": ["o"], "references": "o"}]}]}
        r3 = client.post("/api/v1/schema/analyze",
                         json={"spec": v2, "lineage_id": "auto"}).json()
        assert r3["version"]["version"] == 2 and r3["version"]["unchanged"] is False


# The default store: one resolver, shared, honoring REXGRAPH_RCDB_URI
@pytest.fixture
def isolated_default_store():
    """Reset the memoized default store around each test.

    `default_store()` caches one instance per process on purpose, so a test that
    repoints it must restore it or later tests inherit a store pointing at a deleted
    tmp_path.
    """
    from agent import rcdb as R
    R.reset_default_store()
    try:
        yield
    finally:
        R.reset_default_store()


def test_default_store_honors_the_env_uri(tmp_path, monkeypatch, isolated_default_store):
    """Without a shared default resolver, every non-HTTP consumer built its own
    MemoryStore() and silently discarded what it wrote."""
    from agent import rcdb as R

    monkeypatch.setenv("REXGRAPH_RCDB_URI", "file://" + str(tmp_path / "store"))
    R.reset_default_store()
    s = R.default_store()
    assert isinstance(s, R.FileStore)
    # the same process shares one instance
    assert R.default_store() is s


def test_default_store_falls_back_to_a_file_store(tmp_path, monkeypatch, isolated_default_store):
    """With no env override the default must still persist, not evaporate."""
    from agent import rcdb as R

    monkeypatch.delenv("REXGRAPH_RCDB_URI", raising=False)
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    R.reset_default_store()
    s = R.default_store()
    assert not isinstance(s, R.MemoryStore)


def test_hive_schema_and_query_manager_use_the_default_store(tmp_path, monkeypatch, isolated_default_store):
    """Constructing either without an explicit store must reach the default, so a
    versioned self-schema survives the request that created it."""
    from agent.hive_schema import HiveSchema
    from agent.query_manager import QueryManager

    from agent import rcdb as R

    monkeypatch.setenv("REXGRAPH_RCDB_URI", "file://" + str(tmp_path / "store"))
    R.reset_default_store()
    shared = R.default_store()

    class _Hive:
        name = "h"
        def roster(self):
            return []
        def list_bees(self):
            return []

    assert HiveSchema(_Hive()).store is shared
    assert QueryManager().store is shared


def test_file_store_ids_that_sanitize_alike_do_not_share_a_blob(tmp_path):
    """_blob_path replaced every non-alphanumeric character with '_', so 'core/alpha'
    and 'core_alpha' mapped to the same file. The index kept both records, but the
    second put silently overwrote the first blob and the first id read back as the
    wrong complex. Knowledge-core ids carry '/' and ':' routinely."""
    import numpy as np
    from agent.rcdb import open_store

    from rexgraph.graph import RexGraph

    st = open_store("file://" + str(tmp_path / "store"))
    tri = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    path = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 3], np.int32))
    st.put("core/alpha", tri)
    st.put("core_alpha", path)

    got_a, got_b = st.get("core/alpha"), st.get("core_alpha")
    assert (int(got_a.nV), int(got_a.nE)) == (3, 3), "core/alpha came back as the wrong complex"
    assert (int(got_b.nV), int(got_b.nE)) == (4, 3)


def test_file_store_round_trips_ids_with_path_and_scheme_characters(tmp_path):
    """The ids v1.0.5 will actually use."""
    import numpy as np
    from agent.rcdb import open_store

    from rexgraph.graph import RexGraph

    st = open_store("file://" + str(tmp_path / "store"))
    rex = RexGraph(sources=np.array([0, 1, 2], np.int32), targets=np.array([1, 2, 0], np.int32))
    for rid in ("doc:agent/agent/rcdb.py", "core/beta", "a%b", "sub/dir/thing.md"):
        st.put(rid, rex)
    for rid in ("doc:agent/agent/rcdb.py", "core/beta", "a%b", "sub/dir/thing.md"):
        assert st.get(rid) is not None, f"{rid} did not round-trip"
    assert {r.id for r in st.list()} >= {"doc:agent/agent/rcdb.py", "core/beta",
                                         "a%b", "sub/dir/thing.md"}
