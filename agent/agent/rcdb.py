"""
agent.rcdb: the Relational Complex Database (RCDB).

A backend-agnostic store where **every record is a relational complex**.
One interface (:class:`RCStore`), several pluggable backends, and
**structural query** (the part nobody else has): select complexes by their
topology (Betti numbers, coherence, voids), not just by id or column value.

The design separates two things:
  * the *blob*: the complex itself, serialized with the ``rexgraph.io``
    layer (safetensors by default; any supported format works);
  * the *signature*: a small, queryable structural summary
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
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


def _now():
    return time.time()


# serialization (complex <-> bytes)

def serialize_complex(obj) -> bytes:
    """Serialize a RexGraph or TemporalRex to safetensors bytes (cross-ecosystem,
    no pickle). A TemporalRex is written as its delta-compressed index via
    `temporal_rex_to_safetensors`; a plain RexGraph goes through the existing
    `rex_to_safetensors` path, unchanged."""
    from rexgraph.graph import TemporalRex
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, temporal_rex_to_safetensors
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        if isinstance(obj, TemporalRex):
            temporal_rex_to_safetensors(obj, tmp)
        else:
            rex_to_safetensors(obj, tmp)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def deserialize_complex(blob: bytes):
    """Reconstruct a RexGraph or TemporalRex from safetensors bytes.

    Routes on the file's own `object_type` metadata (written by `serialize_complex`)
    via `load_safetensors`, the object-type dispatch shared with `save_safetensors`
    (safetensors_bridge.py), so the reader never has to be told in advance which
    kind of complex the blob holds."""
    from rexgraph.io.safetensors_bridge import load_safetensors
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(blob)
        return load_safetensors(tmp)["object"]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def structural_signature(rex, meta: Optional[dict] = None,
                         tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """A small, queryable structural summary of a complex.

    A TemporalRex gets its own branch: the temporal fields (T, checkpoint_times)
    plus the structural signature of its latest snapshot (`reconstruct_at(T - 1)`),
    so a stored sequence is still queryable by the topology it currently holds.
    A plain RexGraph gets the existing signature, with "object_type": "RexGraph"
    added (additive: `_matches`/queries never read this key)."""
    from rexgraph.graph import TemporalRex
    if isinstance(rex, TemporalRex):
        rex._ensure_index()
        cp_times = ([int(x) for x in rex._index_cp_times]
                    if rex._index_cp_times is not None else [])
        # base = latest snapshot's own signature (its object_type is "RexGraph");
        # spread it FIRST so the temporal overrides applied after it (object_type,
        # T, checkpoint_times) are the ones that survive in the merged dict.
        base = structural_signature(rex.reconstruct_at(rex.T - 1), meta, tags)
        return {
            **base,
            "object_type": "TemporalRex",
            "T": int(rex.T),
            "checkpoint_times": cp_times,
        }
    meta = meta or (getattr(rex, "_agent_meta", {}) or {})
    sig: Dict[str, Any] = {
        "object_type": "RexGraph",
        "nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF),
        "tags": list(tags or []),
        "source": meta.get("input_type") or meta.get("source") or "",
    }
    try:
        sig["betti"] = [int(b) for b in rex.betti]
    except Exception:
        sig["betti"] = None
    b = sig.get("betti") or []
    sig["betti1"] = int(b[1]) if len(b) > 1 else 0
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
    # varentropy reliability gap), persisted so the corpus is queryable by them and
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
    created: float = field(default_factory=_now)
    meta: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    tx_from: float = field(default_factory=_now)
    tx_to: Optional[float] = None
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None

    def to_dict(self) -> dict:
        return {"id": self.id, "signature": self.signature, "created": self.created,
                "meta": self.meta, "version": self.version, "tx_from": self.tx_from,
                "tx_to": self.tx_to, "valid_from": self.valid_from, "valid_to": self.valid_to}

    @classmethod
    def from_dict(cls, d: dict) -> "ComplexRecord":
        created = d.get("created", _now())
        return cls(
            id=d["id"], signature=d.get("signature", {}), created=created,
            meta=d.get("meta", {}), version=d.get("version", 1),
            tx_from=d.get("tx_from", created), tx_to=d.get("tx_to"),
            valid_from=d.get("valid_from"), valid_to=d.get("valid_to"))


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

    def put(self, id, rex, meta=None, tags=None, *, valid_from=None, valid_to=None):
        """Append a new version of `id`. Template method: build the signature, delegate
        storage to _put_impl, then emit a best-effort change-feed event."""
        meta = _priv(meta)
        sig = structural_signature(rex, meta, tags)
        rec = self._put_impl(id, rex, sig, meta, tags, valid_from, valid_to)
        self._emit("rcdb.put", id, rec.version, sig)
        return rec

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to):
        raise NotImplementedError

    def get(self, id, *, as_of=None, valid_at=None):
        """Return the reconstructed RexGraph, or None."""
        raise NotImplementedError

    def get_record(self, id, *, as_of=None, valid_at=None):
        raise NotImplementedError

    def get_version(self, id, version):
        """Return the reconstructed RexGraph for one SPECIFIC version, or
        None. Unlike `get(as_of=...)`, this is keyed directly by version
        number, not by a timestamp that could collide across versions
        written on the same tick."""
        raise NotImplementedError

    def history(self, id):
        raise NotImplementedError

    def next_version(self, id):
        raise NotImplementedError

    @staticmethod
    def _select_version(records, as_of, valid_at):
        """Pick the version from `records` (all one id) satisfying the time selectors.
        as_of => tx_from <= as_of < (tx_to or +inf); valid_at => valid_from <= valid_at <
        (valid_to or +inf); both None => the live row (tx_to is None). Returns None if
        none match. When both selectors are given, both must hold."""
        def tx_ok(r):
            return as_of is None or (r.tx_from <= as_of and (r.tx_to is None or as_of < r.tx_to))
        def valid_ok(r):
            if valid_at is None:
                return True
            lo = r.valid_from if r.valid_from is not None else r.tx_from
            hi = r.valid_to
            return lo <= valid_at and (hi is None or valid_at < hi)
        if as_of is None and valid_at is None:
            live = [r for r in records if r.tx_to is None]
            return max(live, key=lambda r: r.version) if live else None
        cands = [r for r in records if tx_ok(r) and valid_ok(r)]
        return max(cands, key=lambda r: r.version) if cands else None

    @staticmethod
    def _split_versioned_id(id):
        """A display id like "base@3" -> ("base", 3); anything else -> None.
        Only a trailing @<positive-int> splits; a bare id or non-string is None."""
        if not isinstance(id, str):
            return None
        at = id.rfind("@")
        if at <= 0:
            return None
        tail = id[at + 1:]
        if not tail.isdigit():
            return None
        return id[:at], int(tail)

    def _emit(self, action, id, version, sig):
        try:
            from agent import activity
            activity.record("rcdb:" + self.backend, action, scope="network",
                            detail={"id": id, "version": version, "nV": sig.get("nV"),
                                    "nE": sig.get("nE"), "tags": sig.get("tags"),
                                    "lineage_id": id})
        except Exception:
            pass

    def list(self, limit: int = 100, offset: int = 0) -> List[ComplexRecord]:
        raise NotImplementedError

    def query(self, limit: int = 100, **predicate) -> List[ComplexRecord]:
        """Structural query: select complexes by their topology."""
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
        self._recs: Dict[str, List[ComplexRecord]] = {}
        self._blobs: Dict[Tuple[str, int], bytes] = {}

    def next_version(self, id):
        rs = self._recs.get(id)
        return (rs[-1].version + 1) if rs else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to):
        now = _now()
        v = self.next_version(id)
        rs = self._recs.setdefault(id, [])
        for r in rs:
            if r.tx_to is None:
                r.tx_to = now                       # close the prior open row
        rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {}, version=v,
                            tx_from=now, tx_to=None,
                            valid_from=valid_from if valid_from is not None else now,
                            valid_to=valid_to)
        rs.append(rec)
        self._blobs[(id, v)] = serialize_complex(rex)
        return rec

    def get_record(self, id, *, as_of=None, valid_at=None):
        rec = self._select_version(self._recs.get(id, []), as_of, valid_at)
        if rec is None:
            split = self._split_versioned_id(id)
            if split is not None:
                base, v = split
                # a lineage() display id: resolve the explicit version directly
                # (version is explicit, so as_of/valid_at do not apply)
                rec = next((r for r in self.history(base) if r.version == v), None)
        return rec

    def get(self, id, *, as_of=None, valid_at=None):
        rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
        if rec is None:
            return None
        # rec.id is the record's OWN stored id: for a direct hit that is `id`
        # itself, but for a display-id fallback (get_record resolved "base@v"
        # through history(base)) it is `base`, never the raw "base@v" string
        # the blob is never keyed by. Keying on rec.id is correct either way.
        blob = self._blobs.get((rec.id, rec.version))
        return deserialize_complex(blob) if blob is not None else None

    def get_version(self, id, version):
        blob = self._blobs.get((id, version))
        return deserialize_complex(blob) if blob is not None else None

    def history(self, id):
        return list(self._recs.get(id, []))

    def list(self, limit=100, offset=0, include_history=False):
        recs = ([r for rs in self._recs.values() for r in rs] if include_history
                else [self._select_version(rs, None, None) for rs in self._recs.values()])
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: -r.tx_from)
        return recs[offset:offset + limit]

    def query(self, limit=100, **predicate):
        return [r for r in self.list(limit=10 ** 9) if _matches(r.signature, predicate)][:limit]

    def delete(self, id):
        existed = id in self._recs
        self._recs.pop(id, None)
        for k in [k for k in self._blobs if k[0] == id]:
            self._blobs.pop(k, None)
        if existed:
            self._emit("rcdb.delete", id, 0, {})
        return existed


# file backend (default, no deps beyond io)

class FileStore(RCStore):
    backend = "file"

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "blobs"), exist_ok=True)
        self._index_path = os.path.join(root, "index.json")

    def _read_index(self) -> Dict[str, List[ComplexRecord]]:
        """Load index.json into `{id -> [ComplexRecord, ...]}`. A legacy
        `{id -> record_dict}` index (one dict per id, no version list) is
        wrapped as a single-element list; `from_dict` backfills the missing
        bitemporal fields so it reads as version 1."""
        if not os.path.exists(self._index_path):
            return {}
        try:
            with open(self._index_path) as f:
                raw = json.load(f)
        except Exception:
            return {}
        idx: Dict[str, List[ComplexRecord]] = {}
        for id, v in raw.items():
            if isinstance(v, dict):
                idx[id] = [ComplexRecord.from_dict(v)]
            else:
                idx[id] = [ComplexRecord.from_dict(x) for x in v]
        return idx

    def _write_index(self, idx: Dict[str, List[ComplexRecord]]):
        raw = {id: [r.to_dict() for r in versions] for id, versions in idx.items()}
        tmp = self._index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(raw, f)
        os.replace(tmp, self._index_path)

    @staticmethod
    def _safe_name(id: str) -> str:
        """Filesystem-safe, REVERSIBLE, collision-free encoding of a record id.

        The shared codec, with the path reserved set (which includes '@', the version
        separator used below). The previous scheme replaced every non-alphanumeric
        character with '_', which is lossy: 'core/alpha' and 'core_alpha' both became
        'core_alpha', so the second put silently overwrote the first blob while the
        index kept both records. Ids like 'doc:agent/rcdb.py' are exactly what a
        knowledge core is keyed by.
        """
        from rexgraph.io.rex_state import RESERVED_PATH, encode_name
        return encode_name(id, RESERVED_PATH)

    @staticmethod
    def _sanitized_name(id: str) -> str:
        """The pre-fix lossy encoding, kept so existing stores stay readable."""
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in id)

    def _blob_path(self, id: str, version: int) -> str:
        return os.path.join(self.root, "blobs",
                            "%s@%d.safetensors" % (self._safe_name(id), version))

    def _blob_read_paths(self, id: str, version: int):
        """Every path a blob for (id, version) may live at, newest scheme first: the
        reversible encoding, then the lossy one, then the pre-versioned layout."""
        b = os.path.join(self.root, "blobs")
        return [
            os.path.join(b, "%s@%d.safetensors" % (self._safe_name(id), version)),
            os.path.join(b, "%s@%d.safetensors" % (self._sanitized_name(id), version)),
            self._legacy_blob_path(id),
        ]

    def _legacy_blob_path(self, id: str) -> str:
        return os.path.join(self.root, "blobs", self._sanitized_name(id) + ".safetensors")

    def next_version(self, id):
        rs = self._read_index().get(id)
        return (rs[-1].version + 1) if rs else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to):
        now = _now()
        idx = self._read_index()
        versions = idx.setdefault(id, [])
        v = (versions[-1].version + 1) if versions else 1
        for r in versions:
            if r.tx_to is None:
                r.tx_to = now                       # close the prior open row
        rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {}, version=v,
                            tx_from=now, tx_to=None,
                            valid_from=valid_from if valid_from is not None else now,
                            valid_to=valid_to)
        versions.append(rec)
        with open(self._blob_path(id, v), "wb") as f:
            f.write(serialize_complex(rex))
        self._write_index(idx)
        return rec

    def get_record(self, id, *, as_of=None, valid_at=None):
        idx = self._read_index()
        rec = self._select_version(idx.get(id, []), as_of, valid_at)
        if rec is None:
            split = self._split_versioned_id(id)
            if split is not None:
                base, v = split
                # a lineage() display id: resolve the explicit version directly
                # (version is explicit, so as_of/valid_at do not apply)
                rec = next((r for r in idx.get(base, []) if r.version == v), None)
        return rec

    def _read_blob(self, id, version):
        # try every encoding a blob may have been written under, newest first
        p = next((q for q in self._blob_read_paths(id, version) if os.path.exists(q)),
                 self._blob_path(id, version))
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def get(self, id, *, as_of=None, valid_at=None):
        rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
        if rec is None:
            return None
        # rec.id is the record's OWN stored id (see MemoryStore.get for why
        # this, not the local `id`, is the correct blob key on a fallback hit).
        blob = self._read_blob(rec.id, rec.version)
        return deserialize_complex(blob) if blob is not None else None

    def get_version(self, id, version):
        blob = self._read_blob(id, version)
        return deserialize_complex(blob) if blob is not None else None

    def history(self, id):
        return list(self._read_index().get(id, []))

    def list(self, limit=100, offset=0, include_history=False):
        idx = self._read_index()
        recs = ([r for versions in idx.values() for r in versions] if include_history
                else [self._select_version(versions, None, None) for versions in idx.values()])
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: -r.tx_from)
        return recs[offset:offset + limit]

    def query(self, limit=100, **predicate):
        out = [r for r in self.list(limit=10 ** 9) if _matches(r.signature, predicate)]
        return out[:limit]

    def delete(self, id):
        idx = self._read_index()
        versions = idx.pop(id, None)
        existed = versions is not None
        if existed:
            self._write_index(idx)
            for r in versions:
                for p in self._blob_read_paths(id, r.version):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            self._emit("rcdb.delete", id, 0, {})
        return existed


# SQL backend (any SQLAlchemy database)

class SQLStore(RCStore):
    backend = "sql"

    # signature fields promoted to indexed columns for in-database queries
    _INDEX_COLS = {
        "nV": "INTEGER", "nE": "INTEGER", "betti1": "INTEGER",
        "kappa_mean": "FLOAT", "chain_valid": "BOOLEAN", "source": "VARCHAR(256)",
    }

    # bitemporal columns added on top of the (id, version) composite key
    _TEMPORAL_COLS = {
        "version": "INTEGER", "tx_from": "FLOAT", "tx_to": "FLOAT",
        "valid_from": "FLOAT", "valid_to": "FLOAT",
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
            Column("version", Integer, primary_key=True, default=1),
            Column("signature", Text),
            Column("meta", Text),
            Column("created", Float),
            Column("blob", LargeBinary),
            Column("nV", Integer), Column("nE", Integer),
            Column("betti1", Integer), Column("kappa_mean", Float),
            Column("chain_valid", Boolean), Column("source", String(256)),
            Column("tx_from", Float), Column("tx_to", Float),
            Column("valid_from", Float), Column("valid_to", Float),
        )
        self.meta.create_all(self.engine)
        self._migrate_index_columns(table)

    def _migrate_index_columns(self, table):
        """Add indexed columns to a pre-existing table and backfill from the
        stored signature JSON. Also ALTER-ADDs the five bitemporal columns
        onto a pre-Slice-C table and backfills legacy rows to version 1
        (open, tx_from/valid_from = created), then repairs a legacy id-only
        primary key to the composite (id, version) key the append-only
        design requires, and finally (re)creates the indexes on the
        promoted columns. Idempotent: re-opening an already-migrated table
        is a no-op."""
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
        have_t = {c["name"] for c in insp.get_columns(table)}
        missing_t = [c for c in self._TEMPORAL_COLS if c not in have_t]
        with self.engine.begin() as conn:
            for col in missing_t:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col} {self._TEMPORAL_COLS[col]}'))
            if missing_t:
                conn.execute(text(
                    f"UPDATE {table} SET version = 1, tx_from = created, "
                    f"valid_from = created WHERE version IS NULL"))
        self._ensure_composite_pk(table)
        self._create_promoted_indexes(table)

    def _create_promoted_indexes(self, table):
        """(Re)create the indexes on the promoted signature columns and the
        bitemporal lookup columns. Each CREATE INDEX is guarded by an
        existing-index check, so this is safe to call repeatedly: a table
        that already has the indexes is a no-op, and a table that just lost
        them (a primary-key rebuild drops indexes along with the table)
        gets them rebuilt."""
        from sqlalchemy import inspect, text
        insp = inspect(self.engine)
        existing_idx = {i["name"] for i in insp.get_indexes(table)}
        with self.engine.begin() as conn:
            for col in ("betti1", "kappa_mean", "source"):
                iname = f"ix_{table}_{col}"
                if iname not in existing_idx:
                    try:
                        conn.execute(text(f"CREATE INDEX {iname} ON {table} ({col})"))
                    except Exception:
                        pass
            for iname, cols_sql in ((f"ix_{table}_id_txto", "(id, tx_to)"),
                                     (f"ix_{table}_id_validfrom", "(id, valid_from)")):
                if iname not in existing_idx:
                    try:
                        conn.execute(text(f"CREATE INDEX {iname} ON {table} {cols_sql}"))
                    except Exception:
                        pass

    def _ensure_composite_pk(self, table):
        """Repair a legacy id-only primary key to the composite (id, version)
        key the append-only versioned schema requires. A table freshly
        created by this class already has the composite key (it is declared
        directly on self.table), so this only ever fires against a
        pre-Slice-C database. Idempotent: a table already keyed on
        (id, version) is left untouched, and an unrecognized primary-key
        shape is left alone rather than guessed at.

        SQLite cannot ALTER a primary key in place, so the table is rebuilt
        under a temporary name with the composite key declared, the data is
        copied over column by column, and the temporary table is swapped in
        for the original. Other dialects can ALTER the constraint directly."""
        from sqlalchemy import inspect, text, MetaData
        insp = inspect(self.engine)
        pk = insp.get_pk_constraint(table).get("constrained_columns") or []
        if sorted(pk) == ["id", "version"]:
            return                          # already composite: idempotent no-op
        if pk != ["id"]:
            return                          # unexpected PK shape: leave it alone
        dialect = self.engine.dialect.name
        if dialect == "sqlite":
            cols = [c["name"] for c in insp.get_columns(table)]
            collist = ", ".join(cols)
            tmp = table + "__pkmig"
            new_meta = MetaData()
            new_table = self.table.to_metadata(new_meta, name=tmp)
            with self.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
                new_table.create(conn)
                conn.execute(text(
                    f"INSERT INTO {tmp} ({collist}) SELECT {collist} FROM {table}"))
                conn.execute(text(f"DROP TABLE {table}"))
                conn.execute(text(f"ALTER TABLE {tmp} RENAME TO {table}"))
        elif dialect == "mysql":
            with self.engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} DROP PRIMARY KEY"))
                conn.execute(text(f"ALTER TABLE {table} ADD PRIMARY KEY (id, version)"))
        else:
            pk_name = insp.get_pk_constraint(table).get("name") or f"{table}_pkey"
            with self.engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {pk_name}"))
                conn.execute(text(f"ALTER TABLE {table} ADD PRIMARY KEY (id, version)"))

    def next_version(self, id):
        from sqlalchemy import text
        with self.engine.connect() as conn:
            row = conn.execute(text(
                f"SELECT COALESCE(MAX(version), 0) + 1 FROM {self.table.name} "
                f"WHERE id = :id"), {"id": id}).first()
        return int(row[0]) if row is not None else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to):
        from sqlalchemy import insert, text
        now = _now()
        blob = serialize_complex(rex)
        with self.engine.begin() as conn:
            conn.execute(text(
                f"UPDATE {self.table.name} SET tx_to = :now "
                f"WHERE id = :id AND tx_to IS NULL"), {"now": now, "id": id})
            row = conn.execute(text(
                f"SELECT COALESCE(MAX(version), 0) + 1 FROM {self.table.name} "
                f"WHERE id = :id"), {"id": id}).first()
            v = int(row[0]) if row is not None else 1
            vfrom = valid_from if valid_from is not None else now
            values = dict(id=id, signature=json.dumps(sig), meta=json.dumps(meta or {}),
                          created=now, blob=blob, version=v, tx_from=now, tx_to=None,
                          valid_from=vfrom, valid_to=valid_to, **_sig_index_values(sig))
            conn.execute(insert(self.table).values(**values))
        return ComplexRecord(id=id, signature=sig, created=now, meta=meta or {}, version=v,
                             tx_from=now, tx_to=None, valid_from=vfrom, valid_to=valid_to)

    def _row_to_record(self, row) -> ComplexRecord:
        created = row.created or 0.0
        return ComplexRecord(id=row.id, signature=json.loads(row.signature or "{}"),
                             created=created, meta=json.loads(row.meta or "{}"),
                             version=row.version if row.version is not None else 1,
                             tx_from=row.tx_from if row.tx_from is not None else created,
                             tx_to=row.tx_to, valid_from=row.valid_from, valid_to=row.valid_to)

    _RECORD_COLS = ("id", "signature", "meta", "created", "version",
                    "tx_from", "tx_to", "valid_from", "valid_to")

    def _record_cols(self):
        t = self.table
        return [getattr(t.c, name) for name in self._RECORD_COLS]

    def _records_for(self, id):
        from sqlalchemy import select
        with self.engine.connect() as conn:
            rows = conn.execute(select(*self._record_cols())
                                .where(self.table.c.id == id)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_record(self, id, *, as_of=None, valid_at=None):
        rec = self._select_version(self._records_for(id), as_of, valid_at)
        if rec is None:
            split = self._split_versioned_id(id)
            if split is not None:
                base, v = split
                # a lineage() display id: resolve the explicit version directly
                # (version is explicit, so as_of/valid_at do not apply)
                rec = next((r for r in self._records_for(base) if r.version == v), None)
        return rec

    def get(self, id, *, as_of=None, valid_at=None):
        rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
        if rec is None:
            return None
        # rec.id is the record's OWN stored id (see MemoryStore.get for why
        # this, not the local `id`, is the correct blob key on a fallback hit).
        from sqlalchemy import select
        with self.engine.connect() as conn:
            row = conn.execute(select(self.table.c.blob).where(
                self.table.c.id == rec.id, self.table.c.version == rec.version)).first()
        return deserialize_complex(row.blob) if row else None

    def get_version(self, id, version):
        from sqlalchemy import select
        with self.engine.connect() as conn:
            row = conn.execute(select(self.table.c.blob).where(
                self.table.c.id == id, self.table.c.version == version)).first()
        return deserialize_complex(row.blob) if row else None

    def history(self, id):
        from sqlalchemy import select
        with self.engine.connect() as conn:
            rows = conn.execute(select(*self._record_cols())
                                .where(self.table.c.id == id)
                                .order_by(self.table.c.version)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list(self, limit=100, offset=0, include_history=False):
        from sqlalchemy import select
        t = self.table
        stmt = select(*self._record_cols())
        if not include_history:
            stmt = stmt.where(t.c.tx_to.is_(None))
        stmt = stmt.order_by(t.c.tx_from.desc()).limit(limit).offset(offset)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query(self, limit=100, include_history=False, **predicate):
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
        if not include_history:
            conds.append(t.c.tx_to.is_(None))
        stmt = select(*self._record_cols())
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
        if existed:
            self._emit("rcdb.delete", id, 0, {})
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


def _get_ver(store: RCStore, id, version):
    """Deserialize one SPECIFIC version's blob, keyed by version number (not
    by an as_of timestamp, which can collide when two versions are written on
    the same tick and misresolve to the wrong one). Every backend already
    keys its blob storage by (id, version), so this is a direct fetch via
    `store.get_version` rather than a scan through the bitemporal selector."""
    # Use get_version only if the concrete backend actually overrides it; the ABC
    # defines a NotImplementedError stub, so a plain getattr always finds SOMETHING.
    if type(store).get_version is not RCStore.get_version:
        return store.get_version(id, version)
    # last-resort fallback for a backend that hasn't implemented get_version:
    # not safe under same-tick collisions, only reached for an unknown type.
    rec = next((r for r in store.history(id) if r.version == version), None)
    return store.get(id, as_of=rec.tx_from) if rec is not None else None


def _num(x) -> float:
    """Coerce a signature value to a float scalar. A signature's `betti` is
    stored as a list ([b0, b1, b2, ...]); when one of those slips in here,
    use its b1 (betti1) entry rather than the list itself."""
    if isinstance(x, (list, tuple)):
        return float(x[1]) if len(x) > 1 else (float(x[0]) if x else 0.0)
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _pair_match(ra, rb) -> float:
    """The relational match between two reconstructed complexes: the same
    cross_complex_bridge kappa-correlation score `compare` returns, rescaled
    to [0, 1]. Labels are plain vertex-index labels (no per-version meta
    needed for a trend read). Guarded to 0.0 on any failure (missing complex,
    degenerate bridge, etc.)."""
    try:
        from rexgraph.graph import cross_complex_bridge
        la = [str(i) for i in range(int(getattr(ra, "nV", 0) or 0))]
        lb = [str(i) for i in range(int(getattr(rb, "nV", 0) or 0))]
        bridge = cross_complex_bridge(ra, rb, la, lb)
        corr = float(bridge.get("kappa", {}).get("correlation", 0.0) or 0.0)
        return round(max(0.0, 0.5 * (corr + 1)), 4)
    except Exception:
        return 0.0


def trajectory(store: RCStore, id):
    """The version history of `id` as a directional path in the relational
    field: per-version structural signature, and per-step the signed change
    in each structural quantity (existence/direction over time) plus the
    relational match (cross_complex_bridge similarity) between consecutive
    versions (how close, and moving toward/away)."""
    hist = store.history(id)
    versions = []
    rexes = []
    for r in hist:
        rex = store.get(id, as_of=None) if r.version == hist[-1].version else _get_ver(store, id, r.version)
        rexes.append(rex)
        versions.append({"version": r.version, "tx_from": r.tx_from,
                         "signature": r.signature})
    steps = []
    quant = ("nV", "nE", "nF", "betti1", "kappa_mean")
    for i in range(1, len(hist)):
        a, b = hist[i - 1].signature, hist[i].signature
        dsig = {k: _num(b.get(k)) - _num(a.get(k)) for k in quant if b.get(k) is not None and a.get(k) is not None}
        match = _pair_match(rexes[i - 1], rexes[i])
        prev_match = steps[-1]["match"] if steps else None
        steps.append({"from": hist[i - 1].version, "to": hist[i].version, "d": dsig,
                      "match": match,
                      "direction": (None if prev_match is None else
                                    ("toward" if match > prev_match else
                                     "away" if match < prev_match else "level"))})
    return {"id": id, "versions": versions, "steps": steps}


def trend_between(store: RCStore, id_a, id_b):
    """How two records' relational similarity moves over their aligned
    version timelines (converging vs diverging, and by how much per step)."""
    ha, hb = store.history(id_a), store.history(id_b)
    n = min(len(ha), len(hb))
    series = []
    for i in range(n):
        ra = _get_ver(store, id_a, ha[i].version)
        rb = _get_ver(store, id_b, hb[i].version)
        series.append(_pair_match(ra, rb))
    steps = [{"step": i, "match": series[i],
              "direction": ("toward" if series[i] > series[i - 1] else
                            "away" if series[i] < series[i - 1] else "level")}
             for i in range(1, n)]
    return {"a": id_a, "b": id_b, "match_series": series, "steps": steps,
            "net": (series[-1] - series[0]) if series else 0.0}


def find_similar(store: RCStore, query_rex, query_labels, top_k: int = 10,
                 exclude_id: str = None):
    """Rank stored complexes by structural similarity to a query complex.

    Uses the cross-complex bridge (aligns by shared labels, correlates the
    per-vertex coherence): the real structural-similarity measure, not a
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
    an ``unchanged`` flag.

    Compares against ``store.get(lineage_id)`` (the current version) directly,
    over the native version chain (no scan over other lineages)."""
    latest_rex = store.get(lineage_id)
    if latest_rex is not None:
        latest_rec = store.get_record(lineage_id)
        new_labels = set((meta or {}).get("vertex_labels", []))
        old_labels = set(_labels_of(latest_rec, latest_rex))
        try:
            new_betti = [int(b) for b in getattr(rex, "betti", [])]
            old_betti = [int(b) for b in getattr(latest_rex, "betti", [])]
        except Exception:
            new_betti = old_betti = []
        if new_labels == old_labels and new_betti == old_betti:
            return {"id": f"{lineage_id}@{latest_rec.version}", "lineage_id": lineage_id,
                    "version": latest_rec.version, "unchanged": True}
    info = put_version(store, lineage_id, rex, meta=meta, tags=tags)
    info["unchanged"] = False
    return info


def put_version(store: RCStore, lineage_id: str, rex, meta=None, tags=None, *, valid_from=None):
    """Store the next version of a lineage over the store's own native version
    chain (one id, appended versions); the version number comes from
    ``ComplexRecord.version`` (an O(1) lookup on that id), not a scan over
    every stored complex. Returns the assigned version info."""
    rec = store.put(lineage_id, rex, meta=meta,
                    tags=list(tags or []) + ["lineage"], valid_from=valid_from)
    v = rec.version
    parent = v - 1 if v > 1 else None
    return {"id": f"{lineage_id}@{v}", "lineage_id": lineage_id, "version": v,
            "parent_version": parent}


def _legacy_lineage_records(store: RCStore, lineage_id: str):
    """Old-scheme fallback: under the legacy scheme, each version was a SEPARATE record
    id "{lineage_id}@{v}" carrying meta["lineage"]={"id","version",...}. Collect
    those, oldest version first. Empty list if none (i.e. not a legacy store)."""
    out = []
    for r in store.list(limit=10 ** 9):
        meta = r.meta if isinstance(r.meta, dict) else {}
        lin = meta.get("lineage")
        if isinstance(lin, dict) and lin.get("id") == lineage_id:
            out.append((r, lin))
    out.sort(key=lambda rl: rl[1].get("version", 1))
    return out


def lineage(store: RCStore, lineage_id: str):
    """Ordered version list for a lineage. Reads the store's native version
    chain for this id; for a store populated under the legacy scheme (each version stored
    as a separate "{id}@{v}" record grouped by meta.lineage) it falls back to
    that legacy scheme so old data still reads."""
    hist = store.history(lineage_id)
    if hist:
        return [{"id": f"{lineage_id}@{r.version}", "version": r.version,
                 "parent_version": r.version - 1 if r.version > 1 else None,
                 "created": r.tx_from}
                for r in hist]
    legacy = _legacy_lineage_records(store, lineage_id)
    return [{"id": r.id, "version": lin.get("version", i + 1),
             "parent_version": lin.get("parent_version"),
             "created": lin.get("created", r.created)}
            for i, (r, lin) in enumerate(legacy)]


def drift(store: RCStore, lineage_id: str):
    """Version list plus the drift trajectory (structural diff between each
    consecutive pair): how the schema changed across versions. Walks
    ``store.history(lineage_id)`` directly (the native version chain for this
    one id), reconstructing each version by its own version number (via
    ``_get_ver``, not a same-tick ``tx_from`` that could misresolve across
    versions written in the same instant). For a legacy store with no native
    chain under this id, each version is instead reconstructed through its
    own display/real id, so the ``trajectory`` diff still populates for
    legacy data (``trajectory_steps`` stays history-based, so it is ``[]``
    for a legacy lineage; the native path is unaffected).

    Also carries the relational trend layer: ``trajectory_steps`` is
    ``trajectory(store, lineage_id)["steps"]`` (signed movement per
    structural quantity, plus the toward/away/level relational direction).
    The existing keys (``lineage_id``/``versions``/``trajectory``, and each
    ``trajectory`` entry's ``from``/``to``/``match``/``added``/``removed``)
    are unchanged; this only adds a key."""
    from rexgraph.graph import cross_complex_bridge
    versions = lineage(store, lineage_id)
    hist = store.history(lineage_id)
    traj = []
    if hist:
        for v_a, v_b, rec_a, rec_b in zip(versions, versions[1:], hist, hist[1:]):
            try:
                rex_a = _get_ver(store, lineage_id, rec_a.version)
                rex_b = _get_ver(store, lineage_id, rec_b.version)
                if rex_a is None or rex_b is None:
                    continue
                la, lb = _labels_of(rec_a, rex_a), _labels_of(rec_b, rex_b)
                bridge = cross_complex_bridge(rex_a, rex_b, la, lb)
                corr = float(bridge.get("kappa", {}).get("correlation", 0.0) or 0.0)
                sa, sb = set(la), set(lb)
                traj.append({"from": v_a["id"], "to": v_b["id"],
                             "match": round(max(0.0, 0.5 * (corr + 1)), 4),
                             "added": sorted(sb - sa), "removed": sorted(sa - sb)})
            except Exception:
                continue
    else:
        # legacy store: no native chain under `lineage_id`, so reconstruct each
        # version through its own display/real id (store.get resolves both).
        for v_a, v_b in zip(versions, versions[1:]):
            try:
                rex_a, rex_b = store.get(v_a["id"]), store.get(v_b["id"])
                if rex_a is None or rex_b is None:
                    continue
                rec_a, rec_b = store.get_record(v_a["id"]), store.get_record(v_b["id"])
                la, lb = _labels_of(rec_a, rex_a), _labels_of(rec_b, rex_b)
                bridge = cross_complex_bridge(rex_a, rex_b, la, lb)
                corr = float(bridge.get("kappa", {}).get("correlation", 0.0) or 0.0)
                sa, sb = set(la), set(lb)
                traj.append({"from": v_a["id"], "to": v_b["id"],
                             "match": round(max(0.0, 0.5 * (corr + 1)), 4),
                             "added": sorted(sb - sa), "removed": sorted(sa - sb)})
            except Exception:
                continue
    trajectory_steps = trajectory(store, lineage_id)["steps"]
    return {"lineage_id": lineage_id, "versions": versions, "trajectory": traj,
            "trajectory_steps": trajectory_steps}


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
    the other lacks: a drift readout in plain terms.
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


# The process-wide default store
#
# Before this existed, the only code resolving REXGRAPH_RCDB_URI lived inside
# server/routes/rcdb.py, so every non-HTTP consumer fell back to its own
# `MemoryStore()` and silently discarded whatever it wrote. Callers that want a
# specific store still pass one; callers that just want "the store" get this.

_DEFAULT_STORE: Optional[RCStore] = None


def default_store_uri() -> str:
    """The configured store URI: REXGRAPH_RCDB_URI, else a file store under the
    config dir (REXGRAPH_CONFIG_DIR, else ~/.config/rexgraph)."""
    uri = os.environ.get("REXGRAPH_RCDB_URI")
    if uri:
        return uri
    base = os.environ.get("REXGRAPH_CONFIG_DIR",
                          os.path.join(os.path.expanduser("~"), ".config", "rexgraph"))
    return "file://" + os.path.join(base, "rcdb")


def default_store() -> RCStore:
    """The shared default store for this process, opened once.

    Persistent by default: a caller that omits a store keeps its data instead of
    writing into a throwaway MemoryStore.
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = open_store(default_store_uri())
    return _DEFAULT_STORE


def reset_default_store() -> None:
    """Drop the memoized default so the next call re-reads the environment."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = None
