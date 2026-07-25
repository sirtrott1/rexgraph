"""
agent.rcdb - the Relational Complex Database (RCDB).

A backend-agnostic store where **every record is a relational complex**.
One interface (:class:`RCStore`), several pluggable backends, and - the
part nobody else has - **structural query**: select complexes by their
topology (Betti numbers, coherence, voids), not just by id or column value.

The design separates two things:
  * the *blob* - the complex itself, serialized with the ``rexgraph.io``
    layer (safetensors by default; any supported format works);
  * the *signature* - a small, queryable structural summary
    (nV/nE/nF, Betti, κ, chain validity, types, tags, source).

Backends differ only in where blob + signature live, so an enterprise can
run the default file store on a laptop and the same code against Postgres
or S3 in production by changing one URI:

    open_store("memory://")                       # ephemeral / tests
    open_store("file:///var/lib/rexgraph/store")  # local, no deps
    open_store("sqlite:///rcdb.sqlite")           # embedded SQL
    open_store("postgresql://user@host/db")       # any SQLAlchemy backend

New backends register themselves via :func:`register_backend`, so a team
can plug in their own database or object store without touching the core.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np


# serialization (complex <-> bytes)

def serialize_complex(rex) -> bytes:
    """Serialize a RexGraph to safetensors bytes (cross-ecosystem, no pickle)."""
    from rexgraph.io.safetensors_bridge import rex_to_safetensors
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        rex_to_safetensors(rex, tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def deserialize_complex(blob: bytes):
    """Reconstruct a RexGraph from safetensors bytes."""
    from rexgraph.io.safetensors_bridge import safetensors_to_rex
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(blob)
        return safetensors_to_rex(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def structural_signature(rex, meta: Optional[dict] = None,
                         tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """A small, queryable structural summary of a complex."""
    meta = meta or (getattr(rex, "_agent_meta", {}) or {})
    sig: Dict[str, Any] = {
        "nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF),
        "tags": list(tags or []),
        "source": meta.get("input_type") or meta.get("source") or "",
    }
    try:
        sig["betti"] = [int(b) for b in rex.betti]
    except Exception:
        sig["betti"] = None
    try:
        sig["chain_valid"] = bool(rex.chain_valid)
    except Exception:
        pass
    try:
        sig["kappa_mean"] = round(float(np.asarray(rex.coherence).mean()), 6)
    except Exception:
        sig["kappa_mean"] = None
    try:
        vc = rex.void_complex
        sig["n_voids"] = int(vc.get("n_voids", 0))
    except Exception:
        pass
    # Per-document information metrics (structural perplexity = effective modes, the
    # varentropy reliability gap) - persisted so the corpus is queryable by them and
    # per-corpus aggregation is a cheap read of the stored signatures.
    try:
        from agent.metrics import structural_metrics
        sm = structural_metrics(rex)
        sig["structural_perplexity"] = sm["structural_perplexity"]
        sig["effective_modes"] = sm["effective_modes"]
        sig["varentropy_gap"] = sm["varentropy_gap"]
    except Exception:
        pass
    labels = meta.get("vertex_labels")
    if labels:
        sig["labels_sample"] = list(labels[:12])
        sig["n_labels"] = len(labels)
    return sig


@dataclass
class ComplexRecord:
    """A stored relational complex + its structural signature."""
    id: str
    signature: Dict[str, Any]
    created: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "signature": self.signature,
                "created": self.created, "meta": self.meta}


# structural predicate

def _sig_index_values(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the promoted-to-column values from a signature (for SQLStore)."""
    betti = sig.get("betti") or []
    return {
        "nV": int(sig.get("nV", 0) or 0),
        "nE": int(sig.get("nE", 0) or 0),
        "betti1": int(betti[1]) if len(betti) > 1 else 0,
        "kappa_mean": float(sig.get("kappa_mean") or 0.0),
        "chain_valid": bool(sig.get("chain_valid")),
        "source": sig.get("source") or "",
    }


def _priv(meta):
    """Apply engine label-privacy (tokenize names) before persisting, if enabled."""
    try:
        from agent.interfaces import apply_label_privacy
        return apply_label_privacy(meta or {})
    except Exception:
        return meta or {}


def _matches(sig: Dict[str, Any], q: Dict[str, Any]) -> bool:
    """Evaluate a structural query against a signature.

    Supported keys: min_nV/max_nV, min_nE/max_nE, min_nF,
    min_betti1/max_betti1, min_kappa/max_kappa, chain_valid,
    has_voids (bool), tags_any (list), tags_all (list), source.
    """
    def betti(i):
        b = sig.get("betti")
        return b[i] if (b and len(b) > i) else 0
    checks = [
        ("min_nV", lambda v: sig.get("nV", 0) >= v),
        ("max_nV", lambda v: sig.get("nV", 0) <= v),
        ("min_nE", lambda v: sig.get("nE", 0) >= v),
        ("max_nE", lambda v: sig.get("nE", 0) <= v),
        ("min_nF", lambda v: sig.get("nF", 0) >= v),
        ("min_betti1", lambda v: betti(1) >= v),
        ("max_betti1", lambda v: betti(1) <= v),
        ("min_kappa", lambda v: (sig.get("kappa_mean") or 0) >= v),
        ("max_kappa", lambda v: (sig.get("kappa_mean") or 0) <= v),
        ("chain_valid", lambda v: bool(sig.get("chain_valid")) == bool(v)),
        ("has_voids", lambda v: (sig.get("n_voids", 0) > 0) == bool(v)),
        ("source", lambda v: sig.get("source") == v),
        ("tags_any", lambda v: bool(set(sig.get("tags", [])) & set(v))),
        ("tags_all", lambda v: set(v).issubset(set(sig.get("tags", [])))),
    ]
    for key, pred in checks:
        if key in q and q[key] is not None:
            try:
                if not pred(q[key]):
                    return False
            except Exception:
                return False
    return True


# store interface

class RCStore:
    """Abstract Relational Complex Store."""

    backend = "abstract"

    def put(self, id: str, rex, meta: Optional[dict] = None,
            tags: Optional[List[str]] = None) -> ComplexRecord:
        raise NotImplementedError

    def get(self, id: str):
        """Return the reconstructed RexGraph, or None."""
        raise NotImplementedError

    def get_record(self, id: str) -> Optional[ComplexRecord]:
        raise NotImplementedError

    def list(self, limit: int = 100, offset: int = 0) -> List[ComplexRecord]:
        raise NotImplementedError

    def query(self, limit: int = 100, **predicate) -> List[ComplexRecord]:
        """Structural query - select complexes by their topology."""
        raise NotImplementedError

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        recs = self.list(limit=10 ** 9)
        n = len(recs)
        return {
            "backend": self.backend, "count": n,
            "total_vertices": sum(r.signature.get("nV", 0) for r in recs),
            "total_edges": sum(r.signature.get("nE", 0) for r in recs),
            "mean_kappa": (round(float(np.mean([r.signature.get("kappa_mean") or 0
                                                for r in recs])), 4) if n else None),
        }

    def close(self):
        pass


# in-memory backend

class MemoryStore(RCStore):
    backend = "memory"

    def __init__(self):
        self._blobs: Dict[str, bytes] = {}
        self._recs: Dict[str, ComplexRecord] = {}

    def put(self, id, rex, meta=None, tags=None):
        meta = _priv(meta)
        sig = structural_signature(rex, meta, tags)
        rec = ComplexRecord(id=id, signature=sig, meta=meta or {})
        self._blobs[id] = serialize_complex(rex)
        self._recs[id] = rec
        return rec

    def get(self, id):
        blob = self._blobs.get(id)
        return deserialize_complex(blob) if blob is not None else None

    def get_record(self, id):
        return self._recs.get(id)

    def list(self, limit=100, offset=0):
        recs = sorted(self._recs.values(), key=lambda r: -r.created)
        return recs[offset:offset + limit]

    def query(self, limit=100, **predicate):
        out = [r for r in self.list(limit=10 ** 9) if _matches(r.signature, predicate)]
        return out[:limit]

    def delete(self, id):
        self._blobs.pop(id, None)
        return self._recs.pop(id, None) is not None


# file backend (default, no deps beyond io)

class FileStore(RCStore):
    backend = "file"

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "blobs"), exist_ok=True)
        self._index_path = os.path.join(root, "index.json")

    def _index(self) -> Dict[str, dict]:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_index(self, idx: Dict[str, dict]):
        tmp = self._index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(idx, f)
        os.replace(tmp, self._index_path)

    def _blob_path(self, id: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in id)
        return os.path.join(self.root, "blobs", safe + ".safetensors")

    def put(self, id, rex, meta=None, tags=None):
        meta = _priv(meta)
        sig = structural_signature(rex, meta, tags)
        with open(self._blob_path(id), "wb") as f:
            f.write(serialize_complex(rex))
        idx = self._index()
        rec = ComplexRecord(id=id, signature=sig, meta=meta or {})
        idx[id] = rec.to_dict()
        self._write_index(idx)
        return rec

    def get(self, id):
        p = self._blob_path(id)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return deserialize_complex(f.read())

    def get_record(self, id):
        d = self._index().get(id)
        return ComplexRecord(**d) if d else None

    def list(self, limit=100, offset=0):
        recs = [ComplexRecord(**d) for d in self._index().values()]
        recs.sort(key=lambda r: -r.created)
        return recs[offset:offset + limit]

    def query(self, limit=100, **predicate):
        out = [r for r in self.list(limit=10 ** 9) if _matches(r.signature, predicate)]
        return out[:limit]

    def delete(self, id):
        idx = self._index()
        existed = idx.pop(id, None) is not None
        self._write_index(idx)
        try:
            os.unlink(self._blob_path(id))
        except OSError:
            pass
        return existed


# SQL backend (any SQLAlchemy database)

class SQLStore(RCStore):
    backend = "sql"

    # signature fields promoted to indexed columns for in-database queries
    _INDEX_COLS = {
        "nV": "INTEGER", "nE": "INTEGER", "betti1": "INTEGER",
        "kappa_mean": "FLOAT", "chain_valid": "BOOLEAN", "source": "VARCHAR(256)",
    }

    def __init__(self, conn_str: str, table: str = "rc_complexes"):
        from sqlalchemy import (create_engine, MetaData, Table, Column,
                                 String, Float, LargeBinary, Text, Integer, Boolean)
        self._sa = __import__("sqlalchemy")
        self.conn_str = conn_str
        self.engine = create_engine(conn_str)
        self.meta = MetaData()
        self.table = Table(
            table, self.meta,
            Column("id", String(256), primary_key=True),
            Column("signature", Text),
            Column("meta", Text),
            Column("created", Float),
            Column("blob", LargeBinary),
            Column("nV", Integer), Column("nE", Integer),
            Column("betti1", Integer), Column("kappa_mean", Float),
            Column("chain_valid", Boolean), Column("source", String(256)),
        )
        self.meta.create_all(self.engine)
        self._migrate_index_columns(table)

    def _migrate_index_columns(self, table):
        """Add indexed columns to a pre-existing table and backfill from the
        stored signature JSON, then index them. Idempotent."""
        from sqlalchemy import inspect, text
        insp = inspect(self.engine)
        have = {c["name"] for c in insp.get_columns(table)}
        missing = [c for c in self._INDEX_COLS if c not in have]
        with self.engine.begin() as conn:
            for col in missing:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col} {self._INDEX_COLS[col]}'))
            if missing:
                rows = conn.execute(text(f"SELECT id, signature FROM {table}")).fetchall()
                for rid, sigjson in rows:
                    vals = _sig_index_values(json.loads(sigjson or "{}"))
                    sets = ", ".join(f"{k} = :{k}" for k in self._INDEX_COLS)
                    conn.execute(text(f"UPDATE {table} SET {sets} WHERE id = :id"),
                                 dict(id=rid, **vals))
        existing_idx = {i["name"] for i in insp.get_indexes(table)}
        with self.engine.begin() as conn:
            for col in ("betti1", "kappa_mean", "source"):
                iname = f"ix_{table}_{col}"
                if iname not in existing_idx:
                    try:
                        conn.execute(text(f"CREATE INDEX {iname} ON {table} ({col})"))
                    except Exception:
                        pass

    def put(self, id, rex, meta=None, tags=None):
        meta = _priv(meta)
        sig = structural_signature(rex, meta, tags)
        rec = ComplexRecord(id=id, signature=sig, meta=meta or {})
        blob = serialize_complex(rex)
        from sqlalchemy import insert, update, select
        with self.engine.begin() as conn:
            exists = conn.execute(
                select(self.table.c.id).where(self.table.c.id == id)).first()
            values = dict(id=id, signature=json.dumps(sig),
                          meta=json.dumps(meta or {}), created=rec.created,
                          blob=blob, **_sig_index_values(sig))
            if exists:
                conn.execute(update(self.table).where(self.table.c.id == id).values(**values))
            else:
                conn.execute(insert(self.table).values(**values))
        return rec

    def _row_to_record(self, row) -> ComplexRecord:
        return ComplexRecord(id=row.id, signature=json.loads(row.signature or "{}"),
                             created=row.created or 0.0,
                             meta=json.loads(row.meta or "{}"))

    def get(self, id):
        from sqlalchemy import select
        with self.engine.connect() as conn:
            row = conn.execute(select(self.table.c.blob).where(
                self.table.c.id == id)).first()
        return deserialize_complex(row.blob) if row else None

    def get_record(self, id):
        from sqlalchemy import select
        with self.engine.connect() as conn:
            row = conn.execute(select(
                self.table.c.id, self.table.c.signature, self.table.c.meta,
                self.table.c.created).where(self.table.c.id == id)).first()
        return self._row_to_record(row) if row else None

    def list(self, limit=100, offset=0):
        from sqlalchemy import select
        with self.engine.connect() as conn:
            rows = conn.execute(select(
                self.table.c.id, self.table.c.signature, self.table.c.meta,
                self.table.c.created).order_by(self.table.c.created.desc())
                .limit(limit).offset(offset)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query(self, limit=100, **predicate):
        # push the indexed predicates into SQL; apply the rest (tags, voids)
        # in Python only over the narrowed candidate set.
        from sqlalchemy import select, and_
        t = self.table
        builders = {
            "min_nV": lambda v: t.c.nV >= v,
            "max_nV": lambda v: t.c.nV <= v,
            "min_nE": lambda v: t.c.nE >= v,
            "max_nE": lambda v: t.c.nE <= v,
            "min_betti1": lambda v: t.c.betti1 >= v,
            "max_betti1": lambda v: t.c.betti1 <= v,
            "min_kappa": lambda v: t.c.kappa_mean >= v,
            "max_kappa": lambda v: t.c.kappa_mean <= v,
            "source": lambda v: t.c.source == v,
            "chain_valid": lambda v: t.c.chain_valid == bool(v),
        }
        conds, pushed = [], set()
        for key, build in builders.items():
            if predicate.get(key) is not None:
                conds.append(build(predicate[key]))
                pushed.add(key)
        stmt = select(t.c.id, t.c.signature, t.c.meta, t.c.created)
        if conds:
            stmt = stmt.where(and_(*conds))
        stmt = stmt.order_by(t.c.created.desc())
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        residual = {k: v for k, v in predicate.items() if k not in pushed}
        out = [self._row_to_record(r) for r in rows]
        if residual:
            out = [r for r in out if _matches(r.signature, residual)]
        return out[:limit]

    def delete(self, id):
        from sqlalchemy import delete, select
        with self.engine.begin() as conn:
            existed = conn.execute(select(self.table.c.id).where(
                self.table.c.id == id)).first() is not None
            conn.execute(delete(self.table).where(self.table.c.id == id))
        return existed


# backend registry + URI opener

_BACKENDS: Dict[str, Callable[[str], RCStore]] = {}


def register_backend(scheme: str, factory: Callable[[str], RCStore]) -> None:
    """Register a backend factory for a URI scheme (e.g. 'redis')."""
    _BACKENDS[scheme] = factory


def _labels_of(rec: "ComplexRecord", rex) -> list:
    """Best-effort vertex labels for a record (from meta, else indices)."""
    labels = (rec.meta or {}).get("vertex_labels")
    if labels:
        return list(labels)
    n = int(getattr(rex, "nV", 0) or 0)
    return [str(i) for i in range(n)]


def find_similar(store: RCStore, query_rex, query_labels, top_k: int = 10,
                 exclude_id: str = None):
    """Rank stored complexes by structural similarity to a query complex.

    Uses the cross-complex bridge (aligns by shared labels, correlates the
    per-vertex coherence) - the real structural-similarity measure, not a
    scalar signature match. Returns a list of
    ``{id, match, shared, tags, source}`` sorted by match descending, where
    ``match`` is a 0-1 similarity a UI can show as a percentage.
    """
    from rexgraph.graph import cross_complex_bridge
    import math
    qset = set(query_labels)
    out = []
    for rec in store.list(limit=10 ** 9):
        if exclude_id is not None and rec.id == exclude_id:
            continue
        try:
            # lossless pre-filter: a record with no shared labels contributes
            # nothing (bridge n_shared=0), so skip the expensive deserialize.
            meta_labels = (rec.meta or {}).get("vertex_labels")
            if meta_labels is not None and qset and not (qset & set(meta_labels)):
                continue
            cand = store.get(rec.id)
            if cand is None:
                continue
            cand_labels = _labels_of(rec, cand)
            bridge = cross_complex_bridge(query_rex, cand, query_labels, cand_labels)
            n_shared = int(bridge.get("n_shared", 0) or 0)
            if n_shared == 0:
                continue
            corr = float(bridge.get("kappa", {}).get("correlation", 0.0) or 0.0)
            # combine agreement with how much overlaps (confidence)
            denom = max(len(query_labels), len(cand_labels), 1)
            overlap = n_shared / denom
            match = max(0.0, 0.5 * (corr + 1) * math.sqrt(overlap))
            out.append({
                "id": rec.id,
                "match": round(match, 4),
                "shared": n_shared,
                "tags": rec.signature.get("tags", []),
                "source": rec.signature.get("source", ""),
            })
        except Exception:
            continue
    out.sort(key=lambda r: -r["match"])
    return out[:top_k]


def version_if_changed(store: RCStore, lineage_id: str, rex, meta=None, tags=None):
    """Store a new version only if the schema actually changed vs the latest
    (different tables or different topology). Enables auto-lineage on repeated
    reflection without spamming identical versions. Returns version info with
    an ``unchanged`` flag."""
    versions = lineage(store, lineage_id)
    if versions:
        latest = versions[-1]
        latest_rex = store.get(latest["id"])
        latest_rec = store.get_record(latest["id"])
        if latest_rex is not None:
            new_labels = set((meta or {}).get("vertex_labels", []))
            old_labels = set(_labels_of(latest_rec, latest_rex))
            try:
                new_betti = [int(b) for b in getattr(rex, "betti", [])]
                old_betti = [int(b) for b in getattr(latest_rex, "betti", [])]
            except Exception:
                new_betti = old_betti = []
            if new_labels == old_labels and new_betti == old_betti:
                return {"id": latest["id"], "lineage_id": lineage_id,
                        "version": latest["version"], "unchanged": True}
    info = put_version(store, lineage_id, rex, meta=meta, tags=tags)
    info["unchanged"] = False
    return info


def put_version(store: RCStore, lineage_id: str, rex, meta=None, tags=None):
    """Store the next version of a lineage. Versions are records
    ``{lineage_id}@{version}`` linked by ``meta.lineage`` - no store change,
    since meta is schemaless. Returns the assigned version info."""
    import time
    existing = [r for r in store.list(limit=10 ** 9)
                if (r.meta or {}).get("lineage", {}).get("id") == lineage_id]
    versions = [int((r.meta or {}).get("lineage", {}).get("version", 0))
                for r in existing]
    v = (max(versions) + 1) if versions else 1
    parent = max(versions) if versions else None
    meta = dict(meta or {})
    meta["lineage"] = {"id": lineage_id, "version": v,
                       "parent_version": parent, "created": time.time()}
    rid = f"{lineage_id}@{v}"
    store.put(rid, rex, meta=meta, tags=list(tags or []) + ["lineage"])
    return {"id": rid, "lineage_id": lineage_id, "version": v,
            "parent_version": parent}


def lineage(store: RCStore, lineage_id: str):
    """Ordered version list for a lineage."""
    recs = [r for r in store.list(limit=10 ** 9)
            if (r.meta or {}).get("lineage", {}).get("id") == lineage_id]
    recs.sort(key=lambda r: (r.meta or {}).get("lineage", {}).get("version", 0))
    return [{"id": r.id,
             "version": (r.meta or {}).get("lineage", {}).get("version"),
             "parent_version": (r.meta or {}).get("lineage", {}).get("parent_version"),
             "created": (r.meta or {}).get("lineage", {}).get("created")}
            for r in recs]


def drift(store: RCStore, lineage_id: str):
    """Version list plus the drift trajectory (structural diff between each
    consecutive pair) - how the schema changed across versions."""
    versions = lineage(store, lineage_id)
    traj = []
    for a, b in zip(versions, versions[1:]):
        cmp = compare(store, a["id"], b["id"])
        if cmp:
            traj.append({"from": a["id"], "to": b["id"], "match": cmp["match"],
                         "added": cmp["only_in_b"], "removed": cmp["only_in_a"]})
    return {"lineage_id": lineage_id, "versions": versions, "trajectory": traj}


def cluster_complexes(store: RCStore, tags_any=None, threshold: float = 0.7):
    """Group stored complexes into structural families by cross-complex
    coherence (the crossing tensor). Builds the pairwise coherence matrix,
    then takes connected components at ``threshold``. Returns
    ``{clusters:[{members, avg_coherence, centroid, tags}], singletons, n}``.
    """
    from rexgraph.graph import cross_complex_bridge
    import math
    recs = store.list(limit=10 ** 9)
    if tags_any:
        tset = set(tags_any)
        recs = [r for r in recs if tset & set(r.signature.get("tags", []))]
    items = []
    for r in recs:
        try:
            rex = store.get(r.id)
            if rex is not None:
                items.append((r, rex, _labels_of(r, rex)))
        except Exception:
            continue
    m = len(items)
    label_sets = [set(labels) for (_, _, labels) in items]
    K = [[0.0] * m for _ in range(m)]
    for i in range(m):
        K[i][i] = 1.0
        for j in range(i + 1, m):
            if not (label_sets[i] & label_sets[j]):   # no shared labels -> skip bridge
                continue
            try:
                b = cross_complex_bridge(items[i][1], items[j][1],
                                         items[i][2], items[j][2])
                ns = int(b.get("n_shared", 0) or 0)
                if ns == 0:
                    continue
                corr = float(b.get("kappa", {}).get("correlation", 0.0) or 0.0)
                denom = max(len(items[i][2]), len(items[j][2]), 1)
                match = max(0.0, 0.5 * (corr + 1) * math.sqrt(ns / denom))
                K[i][j] = K[j][i] = match
            except Exception:
                continue
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            if K[i][j] >= threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    clusters, singletons = [], []
    for members in groups.values():
        if len(members) == 1:
            singletons.append(items[members[0]][0].id)
            continue
        pairs = [K[a][b] for ai, a in enumerate(members) for b in members[ai + 1:]]
        avg = sum(pairs) / len(pairs) if pairs else 0.0
        centroid = max(members, key=lambda a: sum(K[a][b] for b in members if b != a))
        tags = sorted({t for a in members
                       for t in items[a][0].signature.get("tags", [])})
        clusters.append({
            "members": [items[a][0].id for a in members],
            "avg_coherence": round(avg, 4),
            "centroid": items[centroid][0].id,
            "tags": tags})
    clusters.sort(key=lambda c: -len(c["members"]))
    return {"clusters": clusters, "singletons": singletons, "n": m}


def compare(store: RCStore, id_a: str, id_b: str):
    """Structurally compare two stored complexes (e.g. schema v1 vs v2).

    Returns a match score, the labels they share, and which side has labels
    the other lacks - a drift readout in plain terms.
    """
    from rexgraph.graph import cross_complex_bridge
    rex_a, rex_b = store.get(id_a), store.get(id_b)
    rec_a, rec_b = store.get_record(id_a), store.get_record(id_b)
    if rex_a is None or rex_b is None:
        return None
    la, lb = _labels_of(rec_a, rex_a), _labels_of(rec_b, rex_b)
    bridge = cross_complex_bridge(rex_a, rex_b, la, lb)
    corr = float(bridge.get("kappa", {}).get("correlation", 0.0) or 0.0)
    sa, sb = set(la), set(lb)
    return {
        "a": id_a, "b": id_b,
        "match": round(max(0.0, 0.5 * (corr + 1)), 4),
        "shared": sorted(sa & sb),
        "only_in_a": sorted(sa - sb),
        "only_in_b": sorted(sb - sa),
    }


def open_store(uri: str = "memory://") -> RCStore:
    """Open an RCStore from a URI.

    memory://                       -> MemoryStore
    file:///path  or  /path         -> FileStore
    sqlite:///f.db, postgresql://…  -> SQLStore (any SQLAlchemy backend)
    <custom>://…                    -> a registered backend
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"
    if scheme in _BACKENDS:
        return _BACKENDS[scheme](uri)
    if scheme == "memory":
        return MemoryStore()
    if scheme == "file":
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        return FileStore(path or "./rcdb")
    # anything SQLAlchemy understands
    return SQLStore(uri)


# built-in registrations
register_backend("memory", lambda uri: MemoryStore())
register_backend("file", lambda uri: FileStore(
    uri[len("file://"):] if uri.startswith("file://") else uri))
