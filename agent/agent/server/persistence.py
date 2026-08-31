"""
agent.server.persistence: workspace state persistence using rexgraph/io.

Uses the existing I/O layer directly:
    save_rex / load_rex     for RexGraph bundles (.rex)
    get_engine + write_*_sql  for analysis tables (SQLite)

Each workspace gets:
    ~/.config/rexgraph/workspaces/{name}/
    ├── state.db              SQLite: activity, queries, sessions
    ├── documents/
    │   ├── doc_0.rex         RexGraph bundle per document
    │   └── doc_1.rex
    └── conversations/
        └── {session_id}.json
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def _base_dir() -> Path:
    """Where workspaces live, read when asked rather than captured at import.

    Captured at import this disagreed with `handles.config_dir()`, which reads the
    environment on each call, so one process could hold two different ideas of where the
    configuration is and a containment test would compare paths from different roots.
    One source of truth instead.
    """
    from agent.server.handles import config_dir
    return config_dir() / "workspaces"


def _ws_dir(workspace: str) -> Path:
    """The directory holding one workspace's files.

    The name is validated here because this function both joins it onto a path and
    creates what it names: /api/v1/pipeline/stream took the workspace from a form field,
    so "../.." escaped the workspace root and mkdir made the directories on the way out.
    The predicate is the one `handles` already uses, not a second rule that can drift.
    """
    from agent.server.handles import valid_workspace
    if not valid_workspace(workspace):
        raise ValueError(f"not a usable workspace name: {workspace!r}")
    d = _base_dir() / workspace
    d.mkdir(parents=True, exist_ok=True)
    return d


def _docs_dir(workspace: str) -> Path:
    d = _ws_dir(workspace) / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _convs_dir(workspace: str) -> Path:
    d = _ws_dir(workspace) / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path(workspace: str) -> str:
    return str(_ws_dir(workspace) / "state.db")


# Document <-> Session linkage
#
# Sessions (~/.rexgraph-agent/sessions) and workspaces
# (~/.config/rexgraph/workspaces) were separate stores with no
# cross-reference. This index makes the mapping bidirectional so a
# workspace document can find its analysis session and vice-versa.

def _links_path(workspace: str) -> Path:
    return _ws_dir(workspace) / "doc_sessions.json"


def _settings_path(workspace: str) -> Path:
    return _ws_dir(workspace) / "settings.json"


def load_settings(workspace: str) -> dict:
    """Workspace settings. Absent or unreadable reads as no settings, so a missing
    file never turns a feature on."""
    p = _settings_path(workspace)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(workspace: str, settings: dict) -> dict:
    """Replace a workspace's settings."""
    _settings_path(workspace).write_text(json.dumps(settings, indent=2))
    return settings


def update_settings(workspace: str, changes: dict) -> dict:
    """Merge changes into a workspace's settings and return the result."""
    s = load_settings(workspace)
    s.update(changes or {})
    return save_settings(workspace, s)


def link_doc_session(workspace: str, doc_id: str, session_id: str) -> None:
    """Record ``doc_id -> session_id`` for a workspace."""
    if not doc_id or not session_id:
        return
    p = _links_path(workspace)
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        data = {}
    data[doc_id] = session_id
    with contextlib.suppress(Exception):
        p.write_text(json.dumps(data, indent=2))


def get_doc_session(workspace: str, doc_id: str):
    """Return the session_id linked to ``doc_id`` (or None)."""
    p = _links_path(workspace)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get(doc_id)
    except Exception:
        return None


def doc_session_map(workspace: str) -> dict:
    """Return the full ``{doc_id: session_id}`` map for a workspace."""
    p = _links_path(workspace)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# RexGraph persistence (uses rexgraph/io/bundle.py directly)

def staging_dir(workspace: str) -> Path:
    """Where a file this workspace uploaded is kept while a document refers to it.

    Not the shared temp directory. A staged upload becomes the document's source path,
    so it deliberately outlives the request, and gettempdir() is both nameable by anyone
    who may name a path and walked by the routes that accept a directory, so one
    tenant's upload sat next to another's indefinitely and enumerating it needed no
    guessed filename. Under the workspace instead, which no caller can name at all
    because path_within refuses the deployment's config directory outright.
    """
    d = _ws_dir(workspace) / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def doc_path(workspace: str, doc_id: str, suffix: str = ".rex") -> Path:
    """The path of one document, with the id held inside its own workspace.

    Validating the workspace name settled which directory this is; it said nothing about
    what gets joined onto it. A doc_id of "../../victim/documents/ordinary" resolved into
    another workspace, and RexBundle.save creates parents, so a tenant could plant or
    overwrite a document in a workspace they cannot otherwise reach. Reproduced, not
    theorised.

    `handles.path_within` is the wrong predicate here and deliberately so: it refuses
    anything under the deployment's config directory, which is exactly where every
    workspace lives. That predicate answers whether a caller may name a path on the
    filesystem at all. This one answers whether an id stays inside its own workspace.
    """
    base = _docs_dir(workspace).resolve()
    p = (base / f"{doc_id}{suffix}").resolve()
    if p == base or base not in p.parents:
        raise ValueError(f"not a document id: {doc_id!r}")
    return p


def save_document_rex(workspace: str, doc_id: str, rex, cache="all"):
    """Save a document's RexGraph as a .rex bundle."""
    from rexgraph.io import save_rex
    path = str(doc_path(workspace, doc_id))
    save_rex(path, rex, cache=cache)
    return path


def load_document_rex(workspace: str, doc_id: str):
    """Load a document's RexGraph from its .rex bundle."""
    from rexgraph.io import load_rex
    path = str(doc_path(workspace, doc_id))
    if not os.path.exists(path):
        return None
    return load_rex(path)


def list_document_bundles(workspace: str) -> list[str]:
    """List all saved document IDs in a workspace."""
    d = _docs_dir(workspace)
    return sorted(
        p.stem for p in d.iterdir()
        if p.suffix == ".rex" or p.is_dir()
    )


# Analysis persistence (uses rexgraph/io/sql_bridge.py directly)

def save_analysis_sql(workspace: str, doc_id: str, rex, analysis: dict):
    """Save analysis results to SQLite using the SQL bridge."""
    try:
        from rexgraph.io.sql_bridge import (
            get_engine,
            write_edge_sql,
            write_metrics_sql,
            write_vertex_sql,
        )
    except ImportError:
        logger.warning("SQL bridge not available (install sqlalchemy)")
        return

    engine = get_engine("sqlite:///" + _db_path(workspace))
    prefix = doc_id.replace("-", "_").replace(".", "_")

    try:
        write_edge_sql(rex, engine, prefix + "_edges")
    except Exception as e:
        logger.warning("write_edge_sql failed: %s", e)

    try:
        write_vertex_sql(rex, engine, prefix + "_vertices")
    except Exception as e:
        logger.warning("write_vertex_sql failed: %s", e)

    try:
        write_metrics_sql(analysis, engine, prefix + "_metrics")
    except Exception as e:
        logger.warning("write_metrics_sql failed: %s", e)


# Activity persistence

def save_activity(workspace: str, activity_edges: list):
    """Persist workspace activity to SQLite."""
    try:
        from rexgraph.io.sql_bridge import get_engine
        engine = get_engine("sqlite:///" + _db_path(workspace))
        sa = __import__("sqlalchemy")

        meta = sa.MetaData()
        table = sa.Table("activity", meta,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String),
            sa.Column("action", sa.String),
            sa.Column("target", sa.String),
            sa.Column("timestamp", sa.Float),
        )
        meta.create_all(engine)

        with engine.begin() as conn:
            for user, action, target, ts in activity_edges:
                conn.execute(table.insert().values(
                    user_id=user, action=action, target=target, timestamp=ts,
                ))
    except ImportError:
        # Fall back to JSON
        path = _ws_dir(workspace) / "activity.json"
        data = [{"user": u, "action": a, "target": t, "ts": ts}
                for u, a, t, ts in activity_edges]
        path.write_text(json.dumps(data))


def load_activity(workspace: str) -> list:
    """Load workspace activity from SQLite."""
    try:
        from rexgraph.io.sql_bridge import get_engine
        engine = get_engine("sqlite:///" + _db_path(workspace))
        sa = __import__("sqlalchemy")

        meta = sa.MetaData()
        table = sa.Table("activity", meta,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("user_id", sa.String),
            sa.Column("action", sa.String),
            sa.Column("target", sa.String),
            sa.Column("timestamp", sa.Float),
        )
        if not sa.inspect(engine).has_table("activity"):
            return []

        with engine.connect() as conn:
            rows = conn.execute(sa.select(table)).fetchall()
            return [(r.user_id, r.action, r.target, r.timestamp) for r in rows]
    except ImportError:
        path = _ws_dir(workspace) / "activity.json"
        if path.exists():
            data = json.loads(path.read_text())
            return [(d["user"], d["action"], d["target"], d["ts"]) for d in data]
        return []


# Query history persistence

def save_query(workspace: str, user_id: str, query_text: str,
               mode: str, results: list):
    """Persist a query and its results."""
    try:
        from rexgraph.io.sql_bridge import get_engine
        engine = get_engine("sqlite:///" + _db_path(workspace))
        sa = __import__("sqlalchemy")

        meta = sa.MetaData()
        table = sa.Table("queries", meta,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String),
            sa.Column("query", sa.String),
            sa.Column("mode", sa.String),
            sa.Column("results_json", sa.Text),
            sa.Column("timestamp", sa.Float),
        )
        meta.create_all(engine)

        with engine.begin() as conn:
            conn.execute(table.insert().values(
                user_id=user_id, query=query_text, mode=mode,
                results_json=json.dumps(results, default=str),
                timestamp=time.time(),
            ))
    except Exception as e:
        logger.warning("save_query failed: %s", e)


def load_query_history(workspace: str, limit: int = 50) -> list:
    """Load recent query history for a workspace."""
    try:
        from rexgraph.io.sql_bridge import get_engine
        engine = get_engine("sqlite:///" + _db_path(workspace))
        sa = __import__("sqlalchemy")

        meta = sa.MetaData()
        if not sa.inspect(engine).has_table("queries"):
            return []

        table = sa.Table("queries", meta, autoload_with=engine)
        with engine.connect() as conn:
            rows = conn.execute(
                sa.select(table).order_by(table.c.timestamp.desc()).limit(limit)
            ).fetchall()
            return [
                {"user": r.user_id, "query": r.query, "mode": r.mode,
                 "results": json.loads(r.results_json), "timestamp": r.timestamp}
                for r in rows
            ]
    except Exception:
        return []


# Conversation persistence

def save_conversation(workspace: str, session_id: str, exchanges: list):
    """Save conversation exchanges to disk."""
    path = _convs_dir(workspace) / (f"{session_id}.json")
    data = []
    for ex in exchanges:
        data.append({
            "n_shared": ex.n_shared,
            "exchange_edges": ex.n_exchange_edges,
            "kappa": ex.kappa_mean,
            "hodge": [ex.hodge_gradient, ex.hodge_curl, ex.hodge_harmonic],
        })
    path.write_text(json.dumps(data, indent=2))


# Export

def export_workspace(workspace: str, output_path: str, fmt: str = "rex"):
    """Export an entire workspace: all documents, analysis, queries.

    Formats: rex (bundle dir), json (single file), sql (SQLite copy).
    """
    ws_dir = _ws_dir(workspace)

    if fmt == "sql":
        import shutil
        db = _db_path(workspace)
        if os.path.exists(db):
            shutil.copy2(db, output_path)
            return output_path

    if fmt == "json":
        export = {
            "workspace": workspace,
            "documents": list_document_bundles(workspace),
            "activity": load_activity(workspace),
            "queries": load_query_history(workspace, limit=1000),
        }
        Path(output_path).write_text(json.dumps(export, indent=2, default=str))
        return output_path

    if fmt == "rex":
        import shutil
        shutil.copytree(str(ws_dir), output_path, dirs_exist_ok=True)
        return output_path

    raise ValueError(f"Unknown format: {fmt}")


# Shared file browsing

def list_workspace_files(workspace: str) -> list:
    """List all files in a workspace with metadata.

    Returns info about each document: doc_id, file size, format,
    when it was saved, whether analysis exists.
    """
    docs_dir = _docs_dir(workspace)
    files = []

    for entry in sorted(docs_dir.iterdir()):
        info = {"name": entry.name, "doc_id": entry.stem}

        if entry.is_dir() and (entry / "MANIFEST.json").exists():
            # .rex bundle
            info["format"] = "rex"
            try:
                manifest = json.loads((entry / "MANIFEST.json").read_text())
                info["nV"] = manifest.get("nV", 0)
                info["nE"] = manifest.get("nE", 0)
                info["nF"] = manifest.get("nF", 0)
            except Exception:
                pass
            info["size_bytes"] = sum(
                f.stat().st_size for f in entry.rglob("*") if f.is_file()
            )
            info["modified"] = entry.stat().st_mtime
        elif entry.suffix == ".rex":
            info["format"] = "rex"
            info["size_bytes"] = entry.stat().st_size
            info["modified"] = entry.stat().st_mtime
        else:
            continue

        # Check if SQL analysis exists
        db = _db_path(workspace)
        if os.path.exists(db):
            info["has_analysis"] = True
        else:
            info["has_analysis"] = False

        files.append(info)

    return files


def get_workspace_stats(workspace: str) -> dict:
    """Get aggregate statistics for a workspace."""
    files = list_workspace_files(workspace)
    queries = load_query_history(workspace, limit=10000)
    activity = load_activity(workspace)

    users = set()
    for user, _action, _target, _ts in activity:
        users.add(user)

    return {
        "workspace": workspace,
        "n_documents": len(files),
        "total_size_bytes": sum(f.get("size_bytes", 0) for f in files),
        "n_queries": len(queries),
        "n_users": len(users),
        "users": sorted(users),
        "has_database": os.path.exists(_db_path(workspace)),
    }
