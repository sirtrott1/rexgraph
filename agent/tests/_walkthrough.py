"""Full HTTP walkthrough: call every mounted route once and report what breaks.

Not collected by a normal run (leading underscore). Run it explicitly:

    micromamba run -n rexgraph python -m pytest tests/_walkthrough.py -s -p no:cacheprovider

Every route in ``app.openapi()`` (plus /docs, /redoc, /openapi.json and a static
asset) is called with a body built from the handler's own signature/docstring.
Results are classified:

    ok        2xx
    reject    4xx with a message (the route works, the input was refused)
    fault     5xx  -> printed loudly; correlated to the server-side log line so
              the sanitized {"detail": "Internal server error"} still names the
              real exception.

The route ORDER matters: fixtures (upload/corpus/rcdb/profile) are created before
the routes that read them, and the destructive routes (resets, deletes, auth
toggle) run last.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import traceback

import pytest

# a real 1x1 PNG, for the OCR multipart route
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_CSV = (
    "source,target,weight\n"
    "alpha,beta,1.0\n"
    "beta,gamma,2.0\n"
    "gamma,alpha,1.5\n"
    "alpha,delta,0.5\n"
    "delta,beta,0.75\n"
)

_TEXT = (
    "The relational complex encodes edges as signed boundary columns. "
    "Vertices are derived as the boundary of those edges. "
    "Geometry emerges from the weights, and frustration measures the sign mismatch. "
    "Coherence relates each edge to its neighbours through the shared vertices. "
    "The Hodge decomposition splits a flow into gradient, curl and harmonic parts."
)

_DDL = (
    "CREATE TABLE author (id INT PRIMARY KEY, name TEXT);\n"
    "CREATE TABLE book (id INT PRIMARY KEY, author_id INT REFERENCES author(id));\n"
    "CREATE TABLE review (id INT PRIMARY KEY, book_id INT REFERENCES book(id), "
    "author_id INT REFERENCES author(id));\n"
)

_SPEC = {
    "tables": [
        {"name": "author", "columns": ["id", "name"], "primary_key": "id",
         "foreign_keys": []},
        {"name": "book", "columns": ["id", "author_id"], "primary_key": "id",
         "foreign_keys": [{"from": "author_id", "to_table": "author", "to": "id"}]},
    ]
}

_TRIPLES = [
    ["Dog", "subClassOf", "Mammal"],
    ["Mammal", "subClassOf", "Animal"],
    ["Cat", "subClassOf", "Mammal"],
    ["Dog", "hasPart", "Tail"],
    ["Cat", "hasPart", "Tail"],
]

_EDGES = [["a", "b"], ["b", "c"], ["c", "a"], ["c", "d"]]


class _ErrCapture(logging.Handler):
    """Catch what the error sanitizer logs, so a masked 500 still names its cause."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.by_id: dict[str, str] = {}
        self.all: list[str] = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        detail = msg
        if record.exc_info:
            detail += "\n" + "".join(traceback.format_exception(*record.exc_info))
        self.all.append(detail)
        # error ids are 12 hex chars, emitted as "[<id>]" by the sanitizer
        for tok in msg.replace("[", " ").replace("]", " ").split():
            if len(tok) == 12 and all(c in "0123456789abcdef" for c in tok):
                self.by_id[tok] = detail


def _live_server(app):
    """A real uvicorn server on an ephemeral port.

    TestClient runs the app to completion inside a portal, so an endless SSE
    generator (``/api/v1/agents/events`` never returns) deadlocks it. Over a real
    socket the client can read one frame and hang up, which is also how the UI
    uses it.
    """
    import socket
    import threading

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    server.install_signal_handlers = lambda: None      # not the main thread
    t = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    t.start()
    for _ in range(400):
        if server.started:
            break
        time.sleep(0.05)
    return server, port


class Walk:
    def __init__(self, client, errlog, port=None):
        self.c = client
        self.errlog = errlog
        self.port = port
        self.rows: list[dict] = []

    def live(self, method, path, note="", read_timeout=8.0, max_bytes=4096, **kw):
        """Hit the route over a real socket and read only the first frame."""
        import httpx
        print(f"  ...   {method:6} {path}  (live server)", flush=True)
        t0 = time.time()
        status, body, exc = None, "", None
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            with httpx.stream(method, url, timeout=read_timeout, **kw) as r:
                status = r.status_code
                buf = b""
                for chunk in r.iter_raw():
                    buf += chunk
                    if len(buf) >= max_bytes or b"event: done" in buf or b"event: error" in buf:
                        break
                body = buf.decode("utf-8", "replace")
        except httpx.ReadTimeout:
            body += "  <no further data within the read timeout>"
        except BaseException as e:  # noqa: BLE001
            exc = f"{type(e).__name__}: {e}"
            body = traceback.format_exc()
        return self._record(method, path, status, body, exc, note, time.time() - t0)

    def call(self, method, path, note="", stream=False, max_chunks=40, **kw):
        print(f"  ...   {method:6} {path}", flush=True)
        t0 = time.time()
        status, body, exc = None, "", None
        try:
            if stream:
                with self.c.stream(method, path, **kw) as r:
                    status = r.status_code
                    parts = []
                    n = 0
                    for chunk in r.iter_raw():
                        parts.append(chunk.decode("utf-8", "replace"))
                        n += 1
                        blob = "".join(parts)
                        if n >= max_chunks or "event: done" in blob or "event: error" in blob:
                            break
                    body = "".join(parts)
            else:
                r = self.c.request(method, path, **kw)
                status = r.status_code
                body = r.text
        except BaseException as e:  # noqa: BLE001 - a raise IS the finding
            exc = f"{type(e).__name__}: {e}"
            body = traceback.format_exc()
        return self._record(method, path, status, body, exc, note, time.time() - t0)

    def _record(self, method, path, status, body, exc, note, elapsed):
        server_err = ""
        if exc:
            server_err = body                       # the traceback itself
        elif status is not None and status >= 500:
            try:
                eid = json.loads(body).get("error_id")
            except Exception:
                eid = None
            if eid:
                server_err = self.errlog.by_id.get(eid, "")

        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None

        row = {
            "method": method, "path": path, "status": status,
            "body": (body or "")[:150].replace("\n", " "),
            "exception": exc, "server_error": server_err[-2500:],
            "note": note, "secs": round(elapsed, 2), "json": parsed,
        }
        self.rows.append(row)
        flag = "RAISE" if exc else ("FAULT" if (status or 0) >= 500 else
                                    ("ok   " if (status or 0) < 400 else "rej  "))
        print(f"  {flag} {status} {method:6} {path:58} {elapsed:6.2f}s  {row['body'][:70]}",
              flush=True)
        return row


@pytest.fixture
def walk(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "sqlite:///" + str(tmp_path / "rcdb.sqlite"))
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "cfg"))
    # /models/pull otherwise blocks for minutes on a HuggingFace resolve; offline
    # makes it fail fast, which is what we want to observe here anyway.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "cfg").mkdir(parents=True, exist_ok=True)

    from agent.rcdb import reset_default_store
    reset_default_store()

    from fastapi.testclient import TestClient

    from agent.server import auth
    auth.reset_auth_manager()
    from agent.server.app import app

    cap = _ErrCapture()
    log = logging.getLogger("agent.server.errors")
    log.addHandler(cap)
    # loopback peer, so the host-local admin routes are genuinely exercised
    client = TestClient(app, client=("127.0.0.1", 40000))
    server, port = _live_server(app)
    try:
        yield Walk(client, cap, port=port)
    finally:
        server.should_exit = True
        log.removeHandler(cap)
        auth.reset_auth_manager()


def test_walkthrough(walk, tmp_path, capsys):
    w = walk
    F: dict = {}

    with capsys.disabled():
        print("\n=== fixtures ==================================================")

        # --- upload -> session -------------------------------------------------
        r = w.call("POST", "/api/upload", note="fixture: session",
                   files={"file": ("walk.csv", _CSV, "text/csv")},
                   data={"options": "{}"})
        F["sid"] = ((r["json"] or {}).get("session_id")) or "no-such-session"

        # --- corpus (workspace document + built corpus) ------------------------
        w.call("POST", "/api/v1/corpus/add-text", note="fixture: corpus doc",
               json={"text": _TEXT, "doc_id": "walkdoc", "depth": "quick"})
        w.call("POST", "/api/v1/corpus/add", note="fixture: 2nd doc",
               files={"file": ("walk_b.txt", _TEXT.encode(), "text/plain")},
               data={"doc_id": "walkdoc_b"})
        w.call("POST", "/api/v1/corpus/build", data={"depth": "quick", "ontology": "false"})
        F["doc"] = "walkdoc"

        # --- rcdb record + lineage --------------------------------------------
        w.call("POST", "/api/v1/db/put", note="fixture: record",
               json={"id": "walk-rec", "text": _TEXT, "tags": ["walkthrough"]})
        w.call("POST", "/api/v1/db/put", note="fixture: record 2",
               json={"id": "walk-rec-2", "text": _TEXT, "tags": ["walkthrough"]})
        F["rec"] = "walk-rec"
        w.call("POST", "/api/v1/db/record-work", note="fixture: lineage",
               json={"kind": "pipeline-run", "labels": ["ingest", "analyze", "store"],
                     "lineage_id": "walk-lineage", "tags": ["walkthrough"]})
        F["lineage"] = "walk-lineage"

        # --- hive profile ------------------------------------------------------
        r = w.call("POST", "/api/v1/hive/profiles", note="fixture: profile",
                   json={"name": "walkthrough-setup", "base": "attach"})
        F["pid"] = (((r["json"] or {}).get("profile") or {}).get("id")) or "attach"

        # --- dbmanager connection ---------------------------------------------
        sqlite_uri = "sqlite:///" + str(tmp_path / "walk.sqlite")
        w.call("POST", "/api/v1/dbmanager/connections", note="fixture: connection",
               json={"name": "walkconn", "uri": sqlite_uri, "kind": "sql"})

        # --- custom model path -------------------------------------------------
        mdir = tmp_path / "fake-model"
        mdir.mkdir()
        (mdir / "config.json").write_text("{}")
        w.call("POST", "/api/v1/models/set-path", note="fixture: model path",
               json={"model_id": "walkfake", "path": str(mdir),
                     "model_type": "transformers", "purpose": ""})

        print(f"\n  fixtures: {F}\n")
        print("=== read-only / no-dependency routes ==========================")

        w.call("GET", "/")
        w.call("GET", "/api/health")
        w.call("GET", "/docs")
        w.call("GET", "/redoc")
        w.call("GET", "/openapi.json")
        w.call("GET", "/static/index.html", note="static mount")

        w.call("GET", "/api/sessions")
        w.call("GET", "/api/v1/status")
        w.call("GET", "/api/v1/ocr/status")

        w.call("GET", "/api/v1/models/list")
        w.call("GET", "/api/v1/models/status")
        w.call("GET", "/api/v1/models/pipeline")
        w.call("GET", "/api/v1/models/cache")

        w.call("GET", "/api/v1/model/chat-config")
        w.call("GET", "/api/v1/model/local/status")
        w.call("GET", "/api/v1/model/local/discover")
        w.call("GET", "/api/v1/model/local/endpoints")
        w.call("GET", "/api/v1/model/embedder/status")
        w.call("GET", "/api/v1/model/introspect/attention/available")

        w.call("GET", "/api/v1/ml/archetypes")
        w.call("GET", "/api/v1/ml/components")

        w.call("GET", "/api/v1/ops/phases")
        w.call("GET", "/api/v1/ops/inventory")
        w.call("GET", "/api/v1/ops/compute")
        w.call("GET", "/api/v1/ops/runs")

        w.call("GET", "/api/v1/builder/steps")
        w.call("GET", "/api/v1/builder/templates")

        w.call("GET", "/api/v1/connectors")
        w.call("GET", "/api/v1/dbmanager/connections")

        w.call("GET", "/api/v1/db/info")
        w.call("GET", "/api/v1/db/list?limit=10&offset=0")
        w.call("GET", "/api/v1/db/recorded?workspace=default")

        w.call("GET", "/api/v1/hive/status?health=false")
        w.call("GET", "/api/v1/hive/monitor?embed=false")
        w.call("GET", "/api/v1/hive/plan")
        w.call("GET", "/api/v1/hive/profiles")

        w.call("GET", "/api/v1/agents/activity?limit=20")
        w.call("GET", "/api/v1/agents/monitor?embed=false")
        w.call("GET", "/api/v1/agents/usage")
        w.call("GET", "/api/v1/agents/dashboard")
        w.call("GET", "/api/v1/agents/network")

        w.call("GET", "/api/v1/admin/whoami")
        w.call("GET", "/api/v1/admin/members")
        w.call("GET", "/api/v1/admin/tokens")
        w.call("GET", "/api/v1/admin/workspaces")
        w.call("GET", "/api/v1/admin/workspace/activity")
        w.call("GET", "/api/v1/admin/workspace/complex")
        w.call("GET", "/api/v1/admin/workspace/overlap")
        w.call("GET", "/api/v1/admin/workspace/settings")
        w.call("GET", "/api/v1/admin/workspace/files")
        w.call("GET", "/api/v1/admin/workspace/stats")

        w.call("GET", "/api/v1/export/queries?limit=10")
        w.call("GET", "/api/v1/export/workspace?format=json")

        print("\n=== session / analysis / explore / chat =======================")
        sid = F["sid"]
        w.call("GET", f"/api/sessions/{sid}")
        w.call("POST", f"/api/sessions/{sid}/goto/0")
        w.call("GET", f"/api/analysis/{sid}?depth=quick")
        w.live("GET", f"/api/analysis/{sid}/stream?depth=quick", note="SSE")
        w.call("GET", f"/api/explore/{sid}/property/betti")
        w.call("GET", f"/api/explore/{sid}/explain/0/0")
        w.call("POST", f"/api/explore/{sid}/hodge", json={"signal": "uniform"})
        w.call("POST", f"/api/explore/{sid}/interfacing",
               json={"target_indices": [0], "target_weights": [1.0], "signal": "uniform"})
        w.call("POST", f"/api/explore/{sid}/context", json={"vertices": [0], "t": 1.0,
                                                            "max_cells": 20})
        w.call("POST", f"/api/explore/{sid}/reconfig", json={"threshold": 0.3})
        w.call("POST", f"/api/chat/{sid}", json={"message": "what are the betti numbers?"})
        w.call("GET", f"/api/chat/{sid}/metrics?structural=false")
        w.call("GET", f"/api/v1/export/session/{sid}?format=json")

        print("\n=== corpus ====================================================")
        w.call("GET", "/api/v1/corpus/summary")
        w.call("GET", "/api/v1/corpus/metrics")
        w.call("GET", "/api/v1/corpus/temporal")
        w.call("GET", "/api/v1/corpus/bridge/0/1")
        w.call("GET", "/api/v1/corpus/voids/0/1")
        w.call("GET", "/api/v1/corpus/persistence/0/1")
        w.call("POST", "/api/v1/corpus/query",
               data={"query": "hodge decomposition", "top_k": "3", "mode": "hybrid"})
        w.call("POST", "/api/v1/corpus/compare", data={"metric": "bottleneck"})
        w.call("POST", "/api/v1/corpus/trustgraph", data={"depth": "quick"})
        w.call("POST", "/api/v1/corpus/fusion", note="needs OCR backends",
               files={"file": ("walk.png", _PNG, "image/png")},
               data={"backends": "offline"})

        print("\n=== rcdb ======================================================")
        rec, lin = F["rec"], F["lineage"]
        w.call("GET", f"/api/v1/db/get/{rec}")
        w.call("GET", f"/api/v1/db/export/{rec}")
        w.call("POST", "/api/v1/db/query", json={"min_nV": 1, "limit": 10})
        w.call("POST", "/api/v1/db/similar", json={"id": rec, "top_k": 5})
        w.call("POST", "/api/v1/db/cluster", json={"threshold": 0.7})
        w.call("POST", "/api/v1/db/compare", json={"a": "walk-rec", "b": "walk-rec-2"})
        w.call("GET", f"/api/v1/db/lineage/{lin}")
        w.call("GET", f"/api/v1/db/recorded/{lin}/at?when={time.time()}")

        print("\n=== schema / ontology / dbmanager / connectors ================")
        w.call("POST", "/api/v1/schema/analyze", json={"ddl": _DDL, "store_id": "walk-schema",
                                                       "tags": ["walkthrough"]})
        w.call("POST", "/api/v1/schema/lint", json={"ddl": _DDL})
        w.call("POST", "/api/v1/schema/faces", json={"ddl": _DDL})
        w.call("POST", "/api/v1/schema/strain",
               json={"ddl": _DDL, "weights": {"book->author": 1000, "review->book": 5000,
                                              "review->author": 5000}})
        w.call("POST", "/api/v1/ontology/analyze",
               json={"triples": _TRIPLES, "store_id": "walk-onto"})

        w.call("POST", "/api/v1/dbmanager/test", json={"name": "walkconn"})
        w.call("POST", "/api/v1/dbmanager/tables", json={"name": "walkconn", "counts": False})
        w.call("POST", "/api/v1/dbmanager/import", json={"name": "walkconn",
                                                         "store_id": "walk-import"})
        w.call("POST", "/api/v1/dbmanager/strain", json={"name": "walkconn"})
        w.call("POST", "/api/v1/dbmanager/ddl", json={"spec": _SPEC, "dialect": "generic"})

        w.call("POST", "/api/v1/connectors/read", json={"scheme": "edges", "source": _EDGES})
        w.call("POST", "/api/v1/connectors/validate", json={"scheme": "edges", "source": _EDGES})
        w.call("POST", "/api/v1/connectors/ingest",
               json={"scheme": "edges", "source": _EDGES, "id": "walk-conn-rec",
                     "tags": ["walkthrough"]})

        print("\n=== integrations ==============================================")
        w.call("POST", "/api/v1/trustgraph/analyze", json={"triples": _TRIPLES})
        w.call("POST", "/api/v1/trustgraph/health", json={"triples": _TRIPLES})
        w.call("POST", "/api/v1/trustgraph/assess",
               json={"entities": ["Dog"], "triples": _TRIPLES})
        w.call("POST", "/api/v1/trustgraph/compare", note="needs a TrustGraph server",
               json={"flows": ["default"], "depth": "quick"})
        w.call("POST", "/api/v1/trustgraph/evolution", note="needs a TrustGraph server",
               json={"flow": "default"})
        w.call("GET", "/api/v1/trustgraph/cores", note="needs a TrustGraph server")
        w.call("POST", "/api/v1/trustgraph/core/analyze", note="needs a TrustGraph server",
               json={"core_id": "walk-core"})
        w.call("POST", "/api/v1/huggingface/analyze", json={"text": _TEXT})
        w.call("POST", "/api/v1/langchain/tools", json={})
        w.call("POST", "/api/v1/langchain/confidence", json={"text": _TEXT})
        w.call("POST", "/api/v1/langchain/analyze", json={"text": _TEXT})
        w.call("POST", "/api/v1/langgraph/state", json={})
        w.call("POST", "/api/v1/vllm/route", json={"text": "explain the proof step by step"})
        w.call("POST", "/api/v1/model/training", json={"target": "summary",
                                                       "format": "safetensors"})
        w.call("GET", "/api/v1/model/training/download?fmt=safetensors&target=summary")

        print("\n=== ml / ops / builder / deploy / pipeline ====================")
        w.call("POST", "/api/v1/ml/run",
               json={"archetype": "mlp", "mode": "single", "optimizer": "auto",
                     "steps": 3, "seed": 0, "device": "cpu"})
        w.call("POST", "/api/v1/ml/ingest", json={"triples": _TRIPLES, "train": False})

        w.call("POST", "/api/v1/hive/profiles/active", json={"id": F["pid"]})
        w.call("POST", "/api/v1/ops/compute", json={"threads": 2})
        r = w.call("POST", "/api/v1/ops/run", json={"phase": "deploy", "params": {}})
        F["run"] = ((r["json"] or {}).get("id")) or ((r["json"] or {}).get("run_id"))
        if not F["run"]:
            try:
                F["run"] = w.c.get("/api/v1/ops/runs?limit=1").json()["runs"][0]["id"]
            except Exception:
                F["run"] = "no-such-run"
        w.call("GET", f"/api/v1/ops/runs/{F['run']}")

        w.call("POST", "/api/v1/deploy/preview", json={"name": "walk-agent", "mode": "service",
                                                       "port": 8000})
        w.call("POST", "/api/v1/deploy/bundle", json={"name": "walk-agent", "mode": "service"})

        w.call("POST", "/api/v1/builder/run",
               data={"config": json.dumps({"name": "walk-agent",
                                           "steps": [{"type": "corpus",
                                                      "params": {"depth": "quick"}},
                                                     {"type": "chunk"}]}),
                     "query": "what is a relational complex?",
                     "workspace": "default"},
               files=[("files", ("walk.txt", _TEXT.encode(), "text/plain"))])

        w.live("POST", "/api/v1/pipeline/stream", note="SSE", read_timeout=180.0,
               max_bytes=200_000,
               files=[("files", ("walk.csv", _CSV, "text/csv"))],
               data={"query": "which nodes bridge", "depth": "quick",
                     "max_rechunk": "1", "workspace": "default",
                     "ontology": "false", "fusion": "false"})

        print("\n=== model / models (inference backends) =======================")
        w.call("POST", "/api/v1/model/chat-config", json={"url": ""})
        w.call("POST", "/api/v1/model/generate", note="needs a chat model server",
               json={"prompt": "hello", "max_tokens": 8, "stream": False})
        w.call("POST", "/api/v1/model/introspect", note="needs a local embedding server",
               json={"texts": ["alpha beta", "beta gamma", "gamma delta"]})
        w.call("POST", "/api/v1/model/introspect/attention", note="needs the Tier-2 host",
               json={"prompt": "hello"})
        w.call("POST", "/api/v1/model/local/start", note="needs a GGUF on disk",
               json={"model_path": str(tmp_path / "nope.gguf")})
        w.call("POST", "/api/v1/model/embedder/start", note="needs a GGUF on disk",
               json={"model_path": str(tmp_path / "nope-embed.gguf")})
        w.call("POST", "/api/v1/model/embedder/stop")
        w.call("POST", "/api/v1/model/local/stop")

        w.call("POST", "/api/v1/models/set-pipeline", json={"purpose": "chat",
                                                            "model_id": "walkfake"})
        w.call("POST", "/api/v1/models/pull", note="needs network + HF",
               json={"model_id": "walkthrough-not-a-real/model"})
        w.call("POST", "/api/v1/models/load", note="needs the model on disk/VRAM",
               json={"model_id": "walkthrough-not-a-real/model", "device": "cpu"})
        w.call("POST", "/api/v1/models/deploy", note="needs vLLM + the model",
               json={"model_id": "walkthrough-not-a-real/model", "port": 10000,
                     "backend": "vllm"})
        w.call("POST", "/api/v1/models/unload", json={"model_id": "walkthrough-not-a-real/model"})
        w.call("POST", "/api/v1/models/stop", json={})
        w.call("DELETE", "/api/v1/models/path/walkfake")
        w.call("DELETE", "/api/v1/models/cache/walkthrough-no-such-cached-model")

        print("\n=== hive / agents =============================================")
        w.call("GET", f"/api/v1/hive/profiles/{F['pid']}")
        w.call("POST", f"/api/v1/hive/profiles/{F['pid']}/apply", json={"reset": True})
        w.call("POST", "/api/v1/hive/attach-live")
        w.call("POST", "/api/v1/hive/auto", note="budget 0 -> plans, spawns nothing",
               json={"budget": 0})
        w.call("POST", "/api/v1/hive/spawn", note="needs a GGUF + llama.cpp binary",
               json={"name": "walkspawn", "model_path": str(tmp_path / "nope.gguf"),
                     "role": "worker", "specialties": []})
        w.call("POST", "/api/v1/hive/attach",
               json={"name": "walkbee", "url": "http://127.0.0.1:9", "role": "worker",
                     "model": "fake", "specialties": ["walk"]})
        w.call("POST", "/api/v1/hive/route", json={"query": "walk this", "top_k": 2})
        w.call("POST", "/api/v1/hive/dispatch", note="needs a reachable bee",
               json={"query": "walk this", "sender": "user"})
        w.call("POST", "/api/v1/hive/ask", json={"name": "no-such-bee", "prompt": "hi"})
        w.call("POST", "/api/v1/hive/remove", json={"name": "walkbee"})

        w.call("POST", "/api/v1/agents/message",
               json={"from": "router", "to": "worker", "text": "analyze the complex"})
        w.call("POST", "/api/v1/agents/message",
               json={"from": "worker", "to": "router", "text": "analysis complete"})
        w.call("POST", "/api/v1/agents/route", json={"query": "analyze", "top_k": 2})
        w.call("POST", "/api/v1/agents/network/hives", json={"name": "walkhive"})
        w.call("POST", "/api/v1/agents/command", json={"command": "status", "scope": "hive"})
        w.live("GET", "/api/v1/agents/events", note="SSE (endless)", read_timeout=6.0,
               max_bytes=1)

        print("\n=== ocr =======================================================")
        w.call("POST", "/api/v1/ocr", note="needs an OCR backend",
               files={"file": ("walk.png", _PNG, "image/png")},
               data={"dpi": "300"})

        print("\n=== admin (mutating) ==========================================")
        w.call("GET", f"/api/v1/admin/workspace/doc/{F['doc']}")
        w.call("POST", "/api/v1/admin/workspace/settings", json={"record_work": True})
        w.call("POST", "/api/v1/admin/members", json={"user_id": "walkuser", "role": "user"})
        w.call("POST", "/api/v1/admin/token", json={"user_id": "walktoken",
                                                     "workspaces": ["default"], "role": "user"})
        w.call("DELETE", "/api/v1/admin/members/walkuser")
        w.call("DELETE", f"/api/v1/admin/workspace/files/{F['doc']}")

        print("\n=== destructive / teardown ====================================")
        w.call("POST", "/api/v1/corpus/reset")
        w.call("POST", "/api/v1/agents/reset")
        w.call("POST", "/api/v1/hive/down")
        w.call("DELETE", f"/api/v1/hive/profiles/{F['pid']}")
        w.call("DELETE", "/api/v1/dbmanager/connections/walkconn")
        w.call("DELETE", f"/api/v1/db/{rec}")
        w.call("DELETE", f"/api/sessions/{sid}")

        print("\n=== auth toggle (LAST: turns auth on then off) ================")
        r = w.call("POST", "/api/v1/admin/recovery-key")
        key = (r["json"] or {}).get("recovery_key", "")
        w.call("POST", "/api/v1/admin/recover", json={"recovery_key": key or "not-a-key"})
        w.call("POST", "/api/v1/admin/auth/passphrase",
               json={"passphrase": "walkthrough-passphrase-123"})
        w.call("POST", "/api/v1/admin/auth/enable",
               json={"passphrase": "walkthrough-passphrase-123"})
        # auth is ON here; the disable call needs the admin token it just became
        from agent.server.auth import get_auth_manager
        tok = get_auth_manager().create_token("walkadmin", ["default"], "admin")
        w.call("POST", "/api/v1/admin/auth/disable",
               json={"passphrase": "walkthrough-passphrase-123"},
               headers={"Authorization": f"Bearer {tok}"})

        _report(w)


def _report(w: Walk):
    from agent.server.app import app

    spec = app.openapi()
    declared = set()
    for path, ops in spec["paths"].items():
        for m in ops:
            if m.upper() in ("HEAD", "OPTIONS", "PARAMETERS"):
                continue
            declared.add((m.upper(), path))
    for extra in (("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json")):
        declared.add(extra)

    import re
    called = set()
    for r in w.rows:
        p = r["path"].split("?")[0]
        hit = None
        for (m, dp) in declared:
            if m != r["method"]:
                continue
            pat = "^" + re.sub(r"\{[^}]+\}", "[^/]+", dp) + "$"
            if re.match(pat, p):
                hit = (m, dp)
                break
        if hit:
            called.add(hit)
    missing = sorted(declared - called)

    faults = [r for r in w.rows if r["exception"] or (r["status"] or 0) >= 500]
    rejects = [r for r in w.rows if 400 <= (r["status"] or 0) < 500]
    oks = [r for r in w.rows if (r["status"] or 0) < 400 and not r["exception"]]

    print("\n" + "=" * 78)
    print(f"routes declared: {len(declared)}   exercised: {len(called)}   "
          f"not exercised: {len(missing)}")
    print(f"calls: {len(w.rows)}   2xx/3xx: {len(oks)}   4xx: {len(rejects)}   "
          f"5xx or raised: {len(faults)}")
    if missing:
        print("\nNOT EXERCISED:")
        for m, p in missing:
            print(f"  {m:6} {p}")
    if faults:
        print("\n5xx / RAISED:")
        for r in faults:
            print(f"  {r['status']} {r['method']:6} {r['path']}")
            print(f"      note: {r['note'] or '-'}")
            print(f"      body: {r['body']}")
            if r["exception"]:
                print(f"      raised: {r['exception']}")
            if r["server_error"]:
                first = [ln for ln in r["server_error"].splitlines() if ln.strip()]
                print(f"      server: {first[0][:200]}")
                if len(first) > 1:
                    print(f"              {first[-1][:200]}")
    print("=" * 78)

    out = os.environ.get("WALKTHROUGH_REPORT")
    if out:
        slim = [{k: v for k, v in r.items() if k != "json"} for r in w.rows]
        with open(out, "w") as fh:
            json.dump({"rows": slim, "missing": sorted(missing)}, fh, indent=1)
        print(f"report written to {out}")
