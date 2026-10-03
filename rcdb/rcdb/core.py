"""RCDB: native relational complex and typed record persistence.

Memory, Local and new SQLite stores share the checked state engine, exact
metadata, version/tombstone rules, envelope ownership and logical change cursors.
Graph signatures support structural queries; generic values declare installed
codecs and do not impersonate graph grades. Backends register public factories.

    open_store("memory://")                   # native ephemeral store
    open_store("local:///var/lib/rexgraph/db") # canonical durable local engine
    open_store("sqlite:///rcdb.sqlite")       # native new SQLite store
    open_store("file:///existing/legacy-db")  # read-only migration source

Legacy file/packed readers preserve existing data without implicit migration.
Object and non native SQL adapters retain their compatibility contracts; their
engine adoption and provider durability qualification remain separate work.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
import zlib
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import wraps
from math import isfinite
from numbers import Integral, Real
from threading import RLock
from typing import Any
from urllib.parse import urlparse

import numpy as np
from .envelope import RecordEnvelope, encode_mapping, decode_mapping, dumps_mapping, loads_mapping, snapshot_mapping, wire_mapping

from .analytics import coherence_greens_mean, coherence_mean

#: Policy an application injects, so this package interoperates with one without
#: importing it. All four are optional: with none set the store works alone, which is
#: what makes it installable and testable on its own.
_ACTIVITY_HOOK = None
_SCOPE_HOOK = None
_PRIVACY_HOOK = None
_SIMILARITY_HOOK = None
_MUTATION_COMMIT = object()


class PublicationUncertainError(RuntimeError):
    """Publication failed and durable rollback could not be established.

    Recovery is required before retrying. A staged attestation must be retained:
    the corresponding record may already be visible after reopening the store.
    """


def configure_hooks(*, activity=None, scope=None, privacy=None, similarity=None) -> None:
    """Supply application policy without this package importing the application.

    `activity` records a change into an application's own feed, `scope` narrows the
    default store to what a request may see, `privacy` projects metadata before it is
    stored, and `similarity` replaces the default scoring. Each is a plain callable, so
    nothing here depends on where it came from.
    """
    global _ACTIVITY_HOOK, _SCOPE_HOOK, _PRIVACY_HOOK, _SIMILARITY_HOOK
    _ACTIVITY_HOOK = activity
    _SCOPE_HOOK = scope
    _PRIVACY_HOOK = privacy
    _SIMILARITY_HOOK = similarity


def _now():
    return time.time()


# serialization (complex <-> bytes)

class _Prepared:
    """Prepared native complex bytes accepted by serialize_complex.

    The payload carries serialized graph state without reconstructing a RexGraph
    in the receiving process.
    """

    __slots__ = ("blob",)

    def __init__(self, blob: bytes):
        self.blob = bytes(blob)


#: Frame for a compressed blob: magic, then one byte naming the codec.
#: A raw safetensors file opens with a little endian u64 header length, so its first
#: four bytes would have to read 0x315A5852 (an 823 MB header) to collide with this
#: magic. The format caps a header at 100 MB, so the two are distinguishable exactly and
#: not by a heuristic: a blob either starts with this or it is a legacy raw one.
_BLOB_MAGIC = b"RXZ1"
_CODEC_ZLIB = b"z"
_CODEC_ZSTD = b"s"


def _codec():
    """Return the zstd codec marker when installed, otherwise the zlib marker.

    Reading a frame requires the compressor named by its marker.
    """
    try:
        import zstandard  # noqa: F401
        return _CODEC_ZSTD
    except ImportError:
        return _CODEC_ZLIB


def compress_blob(raw: bytes, level: int | None = None, *, codec: str | None = None) -> bytes:
    """Frame and compress a serialized complex using the selected codec and level."""
    if codec is not None and (type(codec) is not str or codec not in {"none", "zlib", "zstd"}):
        raise ValueError("unknown blob compression codec")
    if level is not None and type(level) is not int:
        raise ValueError("blob compression level requires a native integer")
    if codec == "none":
        if level is not None:
            raise ValueError("uncompressed blobs cannot declare a compression level")
        return raw
    # None is retained for callers of the legacy ambient selection API. The
    # record engine supplies the codec sealed in its store header.
    c = _codec() if codec is None else (_CODEC_ZSTD if codec == "zstd" else _CODEC_ZLIB)
    if c == _CODEC_ZSTD:
        import zstandard
        body = zstandard.ZstdCompressor(level=3 if level is None else level).compress(raw)
    else:
        body = zlib.compress(raw, 6 if level is None else level)
    return _BLOB_MAGIC + c + body


def decompress_blob(blob: bytes, *, max_output_bytes: int = 256*1024*1024) -> bytes:
    """The inverse, passing a legacy uncompressed blob through untouched."""
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("blob output limit must be positive")
    if not blob[:4] == _BLOB_MAGIC:
        if len(blob) > max_output_bytes:
            raise ValueError("blob exceeds its declared output limit")
        return blob                                         # written before compression
    c, body = blob[4:5], blob[5:]
    if c == _CODEC_ZSTD:
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "this blob was written with zstd; install `zstandard` to read it "
                "(pip install zstandard)") from exc
        decoder = zstandard.ZstdDecompressor()
        try:
            size = zstandard.frame_content_size(body)
            if size > max_output_bytes:
                raise ValueError("decompressed blob exceeds its declared output limit")
            # max_output_size bounds frames without a declared size; zstd ignores
            # it when the header supplies a size, which is why the check is above.
            raw = decoder.decompress(body, max_output_size=max_output_bytes, allow_extra_data=False)
            if size < 0:
                # Some zstd bindings ignore allow_extra_data for unknown size
                # frames. The bounded decode above has established the first
                # frame's output size before this explicit framing check.
                framing = decoder.decompressobj()
                framing.decompress(body)
                if not framing.eof or framing.unused_data:
                    raise ValueError("compressed blob is incomplete or contains trailing bytes")
        except zstandard.ZstdError as exc:
            raise ValueError("compressed blob is invalid, incomplete, or exceeds its output limit") from exc
        return raw
    if c == _CODEC_ZLIB:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(body, max_output_bytes+1)
        if len(raw) > max_output_bytes or decoder.unconsumed_tail:
            raise ValueError("decompressed blob exceeds its declared output limit")
        if not decoder.eof or decoder.unused_data:
            raise ValueError("compressed blob is incomplete or contains trailing bytes")
        return raw
    raise ValueError(f"unknown blob codec {c!r}")


def serialize_complex(obj) -> bytes:
    """Serialize a RexGraph or TemporalRex to compressed safetensors bytes
    (cross ecosystem, no pickle). A TemporalRex is written as its delta compressed index
    via `temporal_rex_to_safetensors`; a plain RexGraph goes through the existing
    `rex_to_safetensors` path, unchanged. A `_Prepared` is already those bytes.

    The result is framed and compressed, so every store gets it from the one place they
    all serialize through rather than three stores each deciding separately.
    """
    if isinstance(obj, _Prepared):
        return obj.blob
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
            return compress_blob(f.read())
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def deserialize_complex(blob: bytes, *, verify: bool = True):
    """Reconstruct a RexGraph or TemporalRex from safetensors bytes.

    `verify=False` skips the load time integrity check, which is a REAL check and not a
    formality: since the Merkle digests are derived rather than stored, `from_state`
    rebuilds the layer tree from the complex and compares it to the recorded root, so it
    catches a re signed boundary column. It also costs one `_leaf_digests` pass: 37.6 s
    on the largest record in the Gutenberg corpus.

    So it is worth paying where the answer is COMMITTED and not on every speculative
    read. A retrieval opens many candidates and keeps a few; verifying the discarded ones
    buys nothing. This is the same split `score_document(reading=False)` makes, for the
    same reason. Callers that skip it here must verify what survives.

    Routes on the file's own `object_type` metadata (written by `serialize_complex`)
    via `load_safetensors`, the object type dispatch shared with `save_safetensors`
    (safetensors_bridge.py), so the reader never has to be told in advance which
    kind of complex the blob holds."""
    from rexgraph.io.safetensors_bridge import load_safetensors
    fd, tmp = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(decompress_blob(blob))
        return load_safetensors(tmp, verify=verify)["object"]
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


class _SkipAnalytics(Exception):
    """Not an error: the marker that lets the analytics blocks share their own `except`.

    Both readings already swallow their failures, so a plain guard would have duplicated
    each `try` around a second condition. Raising into the handler that is there keeps one
    exit per block.
    """


def structural_signature(rex, meta: dict | None = None,
                         tags: list[str] | None = None, *,
                         voids: bool = False, analytics: bool = True) -> dict[str, Any]:
    """A small, queryable structural summary of a complex.

    `voids` adds `n_voids`, and is OFF by default because it is the one reading here that
    is not cheap at every size: see the note at its call site.

    `analytics` covers `kappa_mean` and the information metrics: the readings that cost
    more than building the document does. `betti` is NOT among them: it is
    structure, it is queried, and its cost was a fixable one in the rank path. They are
    ANALYTICS
    columns (`analytics.SCHEMA`, queried as `avg(kappa_mean) GROUP BY source`), not
    retrieval inputs: the query prefilter reads `labels_sample`. They also cost 2.17 s
    a document against 0.39 s to build the document itself, so at corpus scale they are
    85% of the ingest and 37 of its 44 hours. On by default, because a caller reading one
    document wants them; a corpus ingest turns them off and backfills what it needs.

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
        base = structural_signature(rex.reconstruct_at(rex.T - 1), meta, tags,
                                    voids=voids, analytics=analytics)
        times = [float(x) for x in getattr(rex, "_times", [])]
        return {
            **base,
            "object_type": "TemporalRex",
            "T": int(rex.T),
            "checkpoint_times": cp_times,
            # the history's span on its own clock, so a store can be asked which
            # records cover a moment without opening a single blob.
            "t_first": times[0] if times else None,
            "t_last": times[-1] if times else None,
        }
    meta = meta or (getattr(rex, "_agent_meta", {}) or {})
    sig: dict[str, Any] = {
        "object_type": "RexGraph",
        "nV": int(rex.nV), "nE": int(rex.nE), "nF": int(rex.nF),
        "tags": list(tags or []),
        "source": meta.get("input_type") or meta.get("source") or "",
    }
    # BETTI is structure, not analytics: `min_betti1`/`max_betti1` query on it and a
    # record without it cannot answer whether the evidence closes. It was briefly moved
    # behind the flag on cost (4.31s of a 4.49s signature) but that cost was the rank
    # path, not Betti. A span gated document's pairs do not span its wider columns, so
    # the union find identity correctly refuses and exact elimination runs; the
    # elimination was carrying Fractions with denominator 1 and reducing its widest
    # columns first. Both are fixed at the source in `_exact_rank_reduction`.
    try:
        sig["betti"] = [int(b) for b in rex.betti]
    except Exception:
        sig["betti"] = None
    b = sig.get("betti") or []
    sig["betti1"] = int(b[1]) if len(b) > 1 else 0
    # The layers, summarised. A document's chapter and paragraph sectionings are
    # partitions of THIS field, and without them here the index knows a book only as one
    # undifferentiated complex: asking which documents carry a paragraph layer, or how
    # many sections one has, would mean deserialising every blob to find out.
    with contextlib.suppress(Exception):
        from rexgraph.sectioning import sectioning_summary
        sect = sectioning_summary(rex)
        if sect:
            sig["sectionings"] = sect
            sig["sectioning_names"] = [s["name"] for s in sect]
            # the layer tree's root, so a store can be asked whether two documents share
            # a chapter or whether one changed, without opening either blob
            from rexgraph.merkle import build_merkle
            m = build_merkle(rex)
            sig["merkle_root"] = m.root.hex()
            sig["merkle_chain"] = list(m.chain)
    with contextlib.suppress(Exception):
        sig["chain_valid"] = bool(rex.chain_valid)
    # kappa_mean is a queried column (analytics.SCHEMA, temporal.QUANTITIES) and is
    # averaged across records, so it must mean ONE thing at every scale: the local
    # read. The global Green's read goes under its own key, and is absent rather than
    # substituted when the complex is over budget.
    try:
        if not analytics:
            raise _SkipAnalytics
        sig["kappa_mean"] = round(coherence_mean(rex), 6)
        sig["coherence_method"] = "local"
        kg = coherence_greens_mean(rex)
        if kg is not None:
            sig["kappa_greens_mean"] = round(kg, 6)
    except Exception:
        sig["kappa_mean"] = None
    if voids:
        try:
            # Check the dense allocation guard before the optional void reading.
            from rexgraph.core._common import check_dense_allocation
            check_dense_allocation("void_complex nE x nE", int(rex.nE), int(rex.nE))
            vc = rex.void_complex
            sig["n_voids"] = int(vc.get("n_voids", 0))
        except Exception:
            pass
    # Per document information metrics (structural perplexity = effective modes, the
    # varentropy reliability gap), persisted so the corpus is queryable by them and
    # per corpus aggregation is a cheap read of the stored signatures.
    try:
        if not analytics:
            raise _SkipAnalytics
        from .analytics import structural_metrics
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
    """Owned stored object metadata; complex records carry structural signatures.

    StoredRecord is the general public alias. Historical ComplexRecord imports
    remain the same type and serialization contract.
    """
    id: str
    signature: dict[str, Any]
    created: float = field(default_factory=_now)
    meta: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    tx_from: float = field(default_factory=_now)
    tx_to: float | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    envelope: RecordEnvelope | None = None

    @property
    def is_complex(self):
        return self.envelope is None or self.envelope.codec == "rexgraph.safetensors"

    @property
    def object_type(self):
        return self.envelope.object_type if self.envelope is not None else self.signature.get("object_type")

    def __post_init__(self):
        if self.envelope is not None:
            if not isinstance(self.envelope, RecordEnvelope):
                raise TypeError("record envelope must be a declared RecordEnvelope")
            if (type(self.version) is not int or self.id != self.envelope.record_id
                    or self.version != self.envelope.record_version):
                raise ValueError("record envelope differs from its metadata address")

    def to_dict(self) -> dict:
        """Caller owned semantic fields, independent of the storage envelope."""
        result = {"id": self.id, "signature": snapshot_mapping(self.signature), "created": self.created,
                "meta": snapshot_mapping(self.meta), "version": self.version, "tx_from": self.tx_from,
                "tx_to": self.tx_to, "valid_from": self.valid_from, "valid_to": self.valid_to}
        if self.envelope is not None:
            result["envelope"] = self.envelope.as_record()
        return result

    def to_storage_dict(self) -> dict:
        result = self.to_dict()
        result.update(record_values_version=1, signature=encode_mapping(self.signature), meta=encode_mapping(self.meta))
        return result

    def detached(self):
        """Return caller owned metadata without copying any stored payload."""
        return replace(self, signature=snapshot_mapping(self.signature), meta=snapshot_mapping(self.meta))

    def to_wire_dict(self) -> dict:
        """JSON transport with explicit native envelopes for exact fields.

        Read with ``ComplexRecord.from_dict``. Ordinary JSON signatures and
        metadata keep their existing shape; storage and HTTP share one codec.
        """
        result = self.to_dict()
        for name in ("signature", "meta"):
            value = result[name]
            encoded = wire_mapping(value)
            result[name] = encoded
            if encoded is not value:
                result["record_values_version"] = 1
        return result

    @classmethod
    def from_dict(cls, d: dict) -> ComplexRecord:
        created = d.get("created", _now())
        decode = decode_mapping if d.get("record_values_version") == 1 else snapshot_mapping
        if "record_values_version" in d and (type(d["record_values_version"]) is not int or d["record_values_version"] != 1):
            raise ValueError("unknown record values version")
        return cls(
            id=d["id"], signature=decode(d.get("signature", {})), created=created,
            meta=decode(d.get("meta", {})), version=d.get("version", 1),
            tx_from=d.get("tx_from", created), tx_to=d.get("tx_to"),
            valid_from=d.get("valid_from"), valid_to=d.get("valid_to"),
            envelope=None if d.get("envelope") is None else RecordEnvelope.from_record(d["envelope"]))


StoredRecord = ComplexRecord

# structural predicate


@dataclass(frozen=True)
class RecordSnapshot:
    """One selected published version, detached payload, metadata and state identity.

    The envelope is frozen; its Rex value and copied metadata belong to the caller.
    Mutating them does not mutate the store. The digest names the state at read time.
    """

    record: ComplexRecord
    value: object
    state_digest: str


def _read_selector(id, version, as_of, valid_at):
    if not isinstance(id, str) or not id:
        raise TypeError("record id must be a nonempty string")
    if version is not None:
        if isinstance(version, bool) or not isinstance(version, Integral) or version < 1:
            raise ValueError("record version must be a positive integer")
        if as_of is not None or valid_at is not None:
            raise ValueError("an exact version cannot be combined with time selectors")
    for name, value in (("as_of", as_of), ("valid_at", valid_at)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not isfinite(float(value))):
            raise ValueError(f"{name} must be a finite real time")


def _write_interval(valid_from, valid_to):
    for name, value in (("valid_from", valid_from), ("valid_to", valid_to)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not isfinite(float(value))):
            raise ValueError(f"{name} must be a finite real time")
    # RCDB clock coordinates are binary64 times, not exact field coefficients.
    valid_from = None if valid_from is None else float(valid_from)
    valid_to = None if valid_to is None else float(valid_to)
    if valid_from is not None and valid_to is not None and valid_to <= valid_from:
        raise ValueError("valid_to must follow valid_from (half-open interval)")
    return valid_from, valid_to


class VersionConflictError(ValueError):
    """An expected record version no longer matches the published current state."""

def _sig_index_values(sig: dict[str, Any], *, is_complex=True) -> dict[str, Any]:
    """Extract the promoted to column values from a signature (for SQLStore)."""
    betti = sig.get("betti") or []
    source = sig.get("source") or ""
    if not is_complex:
        return {"nV": None, "nE": None, "betti1": None, "kappa_mean": None, "chain_valid": None,
                "source": source if isinstance(source, str) else None}
    return {
        "nV": int(sig.get("nV", 0) or 0),
        "nE": int(sig.get("nE", 0) or 0),
        "betti1": int(betti[1]) if len(betti) > 1 else 0,
        "kappa_mean": float(sig.get("kappa_mean") or 0.0),
        "chain_valid": bool(sig.get("chain_valid")),
        "source": source if isinstance(source, str) else None,
    }


def _priv(meta):
    """Apply engine label privacy (tokenize names) before persisting, if enabled."""
    try:
        if _PRIVACY_HOOK is None:
            return dict(meta or {})
        apply_label_privacy = _PRIVACY_HOOK
        return apply_label_privacy(meta or {})
    except Exception:
        return meta or {}


def _record_labels(sig: dict[str, Any], meta: dict[str, Any] | None = None) -> set:
    """The record's vocabulary, lowercased.

    Prefers meta["vertex_labels"], which is the FULL set. `labels_sample` is what its
    name says (twelve entries), so a prefilter built on it silently misses any
    document whose matching term falls outside them. Falls back to it only when meta
    carries nothing, where a lossy filter still beats no filter.
    """
    labels = (meta or {}).get("vertex_labels") or sig.get("labels_sample") or []
    return {str(x).lower() for x in labels}


def _matches(sig: dict[str, Any], q: dict[str, Any],
             meta: dict[str, Any] | None = None, *, is_complex=True, record_type=None) -> bool:
    """Evaluate a structural query against a signature.

    Supported keys: min_nV/max_nV, min_nE/max_nE, min_nF,
    min_betti1/max_betti1, min_kappa/max_kappa, chain_valid,
    has_voids (bool), tags_any (list), tags_all (list), source,
    labels_any (list), labels_all (list).

    An unsupported key raises TypeError. Skipping it instead would make the query
    match every record, which returns a wrong answer that looks like a right one:
    `query(nE=4)` reads as a filter but the bound is spelled `max_nE`.
    """
    checks = _query_checks(sig, meta, record_type=record_type)
    unknown = sorted(set(q) - _QUERY_KEYS)
    if unknown:
        raise TypeError(
            f"unsupported query key(s): {', '.join(unknown)}. Supported: "
            f"{', '.join(sorted(_QUERY_KEYS))}")
    if (not is_complex and any(v is not None and (k.startswith(("min_", "max_"))
                                      or k in {"chain_valid", "has_voids"}) for k, v in q.items())):
        return False  # Generic declarations carry no graph grades or invariants.
    for key, pred in checks:
        if key in q and q[key] is not None:
            try:
                if not pred(q[key]):
                    return False
            except Exception:
                # A predicate that cannot be evaluated against this record's value
                # has not matched it. Raising here would make one malformed record
                # fail a query over the whole store.
                return False
    return True


def _query_checks(sig: dict[str, Any], meta: dict[str, Any] | None, *, record_type=None):
    """(key, predicate) for every supported query key, bound to one record."""
    def betti(i):
        b = sig.get("betti")
        return b[i] if (b and len(b) > i) else 0
    return [
        ("record_type", lambda v: (sig.get("object_type") if record_type is None else record_type) == v),
        ("labels_any", lambda v: bool(_record_labels(sig, meta)
                                      & {str(x).lower() for x in v})),
        ("labels_all", lambda v: {str(x).lower() for x in v}
                                 <= _record_labels(sig, meta)),
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


_QUERY_KEYS = frozenset(key for key, _pred in _query_checks({}, None))


# store interface

def _recompress_one(path: str, verify: bool = True, force: bool = False) -> tuple:
    """Rewrite one blob in place. Module level so a worker process can import it.

    Returns `(path, before, after, error, rewrote)`. `rewrote` is reported rather than
    inferred from the sizes: a blob can re encode to exactly the same length, and
    reading that as "skipped" undercounts the work. Only file CONTENTS change (no
    record, no index and no log entry is touched) which is why this is safe to run in
    parallel while the store object stays untouched in the parent.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return (path, 0, 0, f"{type(exc).__name__}: {exc}", False)
    from rexgraph.io.security import ENVELOPE_MAGIC
    from .envelope import RECORD_MAGIC
    if raw.startswith(RECORD_MAGIC):
        # The claim names exact payload bytes. Changing compression would also
        # require a transactionally published replacement metadata envelope.
        try:
            RecordEnvelope.from_bytes(raw)
        except ValueError as exc:
            return (path, len(raw), len(raw), f"ValueError: {exc}", False)
        return (path, len(raw), len(raw), None, False)
    if raw.startswith(ENVELOPE_MAGIC):
        # Recompression deserializes and rewrites. A sealed blob cannot be opened here,
        # and forcing it would replace ciphertext with whatever the failure produced.
        return (path, len(raw), len(raw), "sealed payload, not recompressible", False)
    if raw[:4] == _BLOB_MAGIC and not force:
        return (path, len(raw), len(raw), None, False)      # already framed
    try:
        rex = deserialize_complex(raw)
        fresh = serialize_complex(rex)
        if verify:
            back = deserialize_complex(fresh)
            if (int(back.nV), int(back.nE)) != (int(rex.nV), int(rex.nE)):
                raise ValueError("shape changed through the rewrite")
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(fresh)
        os.replace(tmp, path)                               # atomic
        return (path, len(raw), len(fresh), None, True)
    except Exception as exc:
        with contextlib.suppress(OSError):
            os.unlink(f"{path}.tmp")
        return (path, len(raw), len(raw), f"{type(exc).__name__}: {exc}", False)


def _serialized(method):
    """Serialize public writes on one live store handle, including commit staging."""
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._transaction_lock:
            self._check_open()
            if self._publication_uncertain:
                raise PublicationUncertainError("RCDB publication is uncertain; reopen and verify before writing")
            self._corpus_cache = None
            try:
                return method(self, *args, **kwargs)
            except PublicationUncertainError:
                self._publication_uncertain = True
                raise
            finally:
                self._corpus_cache = None
    return run


def _owned_records(method):
    """All public record reads return snapshots under the store handle's lock."""
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._transaction_lock:
            self._check_open()
            if self._publication_uncertain:
                raise PublicationUncertainError("RCDB publication is uncertain; reopen and verify before reading")
            value = method(self, *args, **kwargs)
            if value is None:
                return None
            if isinstance(value, ComplexRecord):
                return value.detached()
            return [record.detached() for record in value]
    return run


class RCStore:
    """Abstract Relational Complex Store."""

    backend = "abstract"
    _cache_corpus = False

    @property
    def logical_state_version(self):
        """Format identity advertised with this provider's logical state digest."""
        return 2

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._transaction_lock = RLock()
        instance._publication_uncertain = False
        instance._corpus_cache = None
        instance._store_identity = None
        return instance

    @property
    def store_id(self):
        """Opaque identity of this logical store, stable across durable reopenings."""
        with self._transaction_lock:
            if self._store_identity is None:
                self._store_identity = self._load_store_identity()
            return self._store_identity.id

    def _load_store_identity(self):
        raise NotImplementedError("this backend must supply a durable store identity")

    def _check_open(self):
        """Providers with an explicit closed state refuse further operations here."""
        return None

    def _check_writable(self):
        if getattr(self, "read_only", False):
            raise PermissionError("legacy RCDB stores are read-only; migrate into LocalStore, MemoryStore or native SQLite")

    def corpus_snapshot(self, *, as_of=None, valid_at=None, signature_fields=None):
        """Capture projected accession terms on one visible version per record."""
        from .corpus import capture
        with self.read_transaction():
            return capture(self, as_of=as_of, valid_at=valid_at, signature_fields=signature_fields)

    @contextlib.contextmanager
    def read_transaction(self):
        """Pin reads on this handle; providers may supply stronger arbitration."""
        with self._transaction_lock:
            self._check_open()
            if self._publication_uncertain:
                raise PublicationUncertainError("RCDB publication is uncertain; reopen before reading")
            yield self

    @contextlib.contextmanager
    def write_scope(self):
        """Reserve a native provider's writer through one conditional publication."""
        raise NotImplementedError("this backend has no native writer scope")
        yield self

    @_serialized
    def configure_security(self, *, key_id=None, keys=None, mutation_policy=None,
                           verifiers=None, transition_signer=None, lineage_signer=None,
                           signature_mode="public", metadata_fields=None,
                           require_commits=False):
        """Configure payload encryption and what a stored record reveals.

        Keys and signers stay process capabilities: only opaque identities and signed
        artifacts are ever persisted. Calling this is additive, so a store that never
        calls it reads and writes exactly as before, which is what keeps every existing
        store readable.
        """
        mode = str(signature_mode).lower()
        if mode not in {"public", "structural", "minimal"}:
            raise ValueError("signature_mode must be public, structural, or minimal")
        selected_fields = (None if metadata_fields is None
                           else frozenset(str(x) for x in metadata_fields))
        selected_verifiers = dict(verifiers or {})
        for signer in (transition_signer, lineage_signer):
            if signer is not None and hasattr(signer, "verifier"):
                verifier = signer.verifier()
                selected_verifiers.setdefault(str(verifier.signer_id), verifier)
        selected_key_id = None if key_id is None else str(key_id)
        selected_require_commits = bool(require_commits)
        self._security_key_id = selected_key_id
        self._security_keys = keys
        self._mutation_policy = mutation_policy
        self._mutation_verifiers = selected_verifiers
        self._transition_signer = transition_signer
        self._lineage_signer = lineage_signer
        self._require_commits = selected_require_commits
        self._signature_mode = mode
        self._metadata_fields = selected_fields
        return self

    def _stored_meta(self, meta):
        """The meta this store persists, which need not be all of it."""
        value = _priv(meta) or {}
        if (getattr(self, "_signature_mode", "public") == "public"
                and getattr(self, "_metadata_fields", None) is None):
            return value
        allowed = getattr(self, "_metadata_fields", None) or frozenset()
        return {k: value[k] for k in sorted(allowed) if k in value}

    def _stored_signature(self, sig):
        """The signature this store persists.

        A signature describes the data, so a store that seals its records and then writes
        the full signature beside them has described what it sealed. `structural` keeps
        the shape and the invariants, `minimal` keeps only what is needed to address a
        record at all.
        """
        mode = getattr(self, "_signature_mode", "public")
        if mode == "public":
            return sig
        minimal = {"object_type", "nV", "nE", "nF"}
        structural = minimal | {
            "betti", "betti1", "chain_valid", "kappa_mean", "kappa_greens_mean",
            "n_voids", "structural_perplexity", "effective_modes", "varentropy_gap",
            "T", "checkpoint_times", "t_first", "t_last", "n_labels",
        }
        allowed = structural if mode == "structural" else minimal
        return {k: sig[k] for k in sorted(allowed) if k in sig}

    def _encode_payload(self, raw: bytes, *, object_type: str) -> bytes:
        """Seal a payload, or hand it back unchanged when no key is configured."""
        key_id = getattr(self, "_security_key_id", None)
        keys = getattr(self, "_security_keys", None)
        if not key_id:
            return bytes(raw)
        if keys is None:
            raise ValueError("encrypted RCDB writes require a KeyProvider")
        from rexgraph.io.security import encrypt_bytes
        return encrypt_bytes(raw, key_id=key_id, keys=keys, object_type=object_type)

    def _decode_payload(self, raw: bytes) -> bytes:
        """Open a payload, deciding by the envelope rather than by configuration.

        A store reads what is on disk, so a sealed blob in an unconfigured store is a
        refusal rather than a plaintext read of ciphertext, and a plaintext blob in a
        configured store still opens: both live side by side after a key is introduced.
        """
        from rexgraph.io.security import ENVELOPE_MAGIC, decrypt_bytes
        if not bytes(raw).startswith(ENVELOPE_MAGIC):
            return bytes(raw)
        keys = getattr(self, "_security_keys", None)
        if keys is None:
            raise PermissionError("encrypted RCDB payload requires a KeyProvider")
        try:
            return decrypt_bytes(raw, keys=keys)
        except PermissionError:
            raise
        except Exception as exc:                     # noqa: BLE001  # wrong key or tamper
            # A wrong key is a refusal, not a library exception. decrypt_bytes lets
            # cryptography's InvalidTag through, so without this a caller would have to
            # import cryptography to catch what the store did, and a store configured
            # with the wrong key would look like a corrupt one.
            raise PermissionError(
                "this key does not open the stored payload") from exc

    def _serialize_payload(self, rex) -> bytes:
        return self._encode_payload(serialize_complex(rex), object_type="RCDB.Rex")

    def _deserialize_payload(self, blob: bytes, *, verify: bool = True):
        return deserialize_complex(self._decode_payload(blob), verify=verify)

    def _record_payload(self, record, rex=None, *, payload=None, identity=None):
        """Bind a serialized object before any backend publishes its metadata."""
        from .engine import bind_record
        from .codecs import EncodedRecord
        if isinstance(rex, EncodedRecord):
            self.header.check_record_codec(rex.reference.name, rex.reference.version)
            payload = self.blob_codec.encode(rex.payload)
            record, raw = bind_record(record, payload, store_id=self.store_id,
                identity=(rex.object_type, rex.digest), codec=rex.reference.name, codec_version=rex.reference.version)
            return record, self._encode_payload(raw, object_type="RCDB.Record")
        if payload is None:
            # Keep the existing serialization hook's single argument contract.
            payload = self._decode_payload(self._serialize_payload(rex))
        record, raw = bind_record(record, payload, store_id=self.store_id, identity=identity,
                                  compression=getattr(self, "blob_codec", None))
        return record, self._encode_payload(raw, object_type="RCDB.Record")

    def _read_record_payload(self, record, blob, *, verify=True):
        from .engine import open_record, rebuild_record
        from .envelope import RECORD_MAGIC
        raw = self._decode_payload(blob)
        store_id = self.store_id if raw.startswith(RECORD_MAGIC) or record.envelope is not None else None
        payload, state = open_record(record, raw, store_id=store_id,
                                     compression=getattr(self, "blob_codec", None))
        return rebuild_record(payload, state, verify=verify)

    def _encode_commit(self, package) -> bytes:
        from rexgraph.io.mutation import mutation_to_bytes
        return self._encode_payload(mutation_to_bytes(package),
                                    object_type="RCDB.MutationPackage")

    def _decode_commit(self, blob: bytes):
        from rexgraph.io.mutation import mutation_from_bytes
        return mutation_from_bytes(self._decode_payload(blob))

    def _store_commit_bytes(self, id: str, version: int, blob: bytes) -> None:
        raise NotImplementedError

    def _load_commit_bytes(self, id: str, version: int):
        raise NotImplementedError

    def _delete_commit_bytes(self, id: str, version: int) -> None:
        """Remove an unpublished commit artifact after a failed write."""
        return None

    def _claim_delete(self, id) -> list[int]:
        """Refuse a raw deletion where commits are required; name the versions to reclaim.

        Called at the top of every backend's delete. Refusing matters because deletion is
        the one operation a commit chain cannot describe: there is no artifact attesting
        that a record was meant to stop existing, so where commits are required a raw
        delete would silently end a lineage the chain still claims is intact.
        """
        self._check_writable()
        if getattr(self, "_require_commits", False):
            raise PermissionError(
                "raw deletion is disabled when this store requires mutation commits")
        return [int(r.version) for r in self.history(str(id))]

    def _reclaim_commits(self, id, versions) -> None:
        """Drop the artifacts for versions that no longer exist.

        An artifact outliving its record is not merely litter: commit_history would hand
        back a package for a version nothing can produce, and a later record reusing that
        id and version would inherit an attestation it never earned.
        """
        for version in versions:
            self._delete_commit_bytes(str(id), int(version))

    def commit_history(self, id: str):
        """The mutation artifacts held for one record, aligned to its versions."""
        out = []
        for rec in self.history(str(id)):
            raw = self._load_commit_bytes(rec.id, rec.version)
            if raw is not None:
                out.append(self._decode_commit(raw))
        return out

    def verify_commits(self, id: str) -> bool:
        """Whether this record's history is the one its commits attest to.

        Walks forward, so each package is checked against the version that actually
        precedes it in the store rather than against whatever it claims. That is the
        whole point of the chain: a package proves a transition only when the endpoint it
        started from is the one on disk, which is why `previous` is passed here and why
        verify_mutation requires it rather than defaulting.
        """
        from rexgraph.object_identity import object_digest
        from rexgraph.io.mutation import MutationPolicy, verify_mutation
        policy = getattr(self, "_mutation_policy", None) or MutationPolicy()
        verifiers = getattr(self, "_mutation_verifiers", {})
        parent = None
        previous_state = ""
        previous_rex = None
        for rec in self.history(str(id)):
            if self._commit_predecessor_reset(rec):
                parent, previous_state, previous_rex = None, "", None
            try:
                rex = self.get_version(rec.id, rec.version)
            except ValueError:
                return False  # A refused record binding cannot certify a commit chain.
            if rex is None:
                return False
            if rec.envelope is not None and rec.envelope.codec != "rexgraph.safetensors":
                return False  # Current Core mutation packages attest complex states.
            current_state = object_digest(rex)
            raw = self._load_commit_bytes(rec.id, rec.version)
            if raw is None:
                # A version with no artifact is ordinary where commits are optional, and
                # evidence where they are required: every write went through
                # commit_mutation, so a missing one means it was removed. Skipping it
                # would let deleting an artifact hide the version it attested to.
                if getattr(self, "_require_commits", False):
                    return False
            else:
                package = self._decode_commit(raw)
                if package.transition.previous_state != previous_state:
                    return False
                if package.transition.resulting_state != current_state:
                    return False
                if not verify_mutation(package, previous=previous_rex, policy=policy,
                                       verifiers=verifiers, parent_digest=parent):
                    return False
                parent = package.link.digest
            previous_state = current_state
            previous_rex = rex
        return True

    def _commit_predecessor_reset(self, record):
        """Engine stores may retain old incarnations across durable tombstones."""
        return False

    @_serialized
    def commit_mutation(self, id, rex, meta=None, tags=None, *, valid_from=None,
                        valid_to=None, actor="", tx_time=None, analytics=True,
                        voids=False, expected_version=None, signature=None):
        """Append a version together with the signed artifact that attests to it.

        The artifact is staged BEFORE the record and rolled back if the record write
        fails, so a version is never published without one. A normal failure therefore
        leaves neither. Only a failure of the rollback itself can leave an artifact
        behind, and that direction is the safe one: an unreferenced artifact is inert,
        while a version with no commit is a hole in the chain.
        """
        self._check_writable()
        from rexgraph.io.mutation import (
            MutationPolicy,
            prepare_mutation,
            verify_mutation,
        )
        _read_selector(id, None, tx_time, None)
        valid_from, valid_to = _write_interval(valid_from, valid_to)
        if not isinstance(actor, str):
            raise TypeError("mutation actor must be a string")
        if signature is not None:
            signature = snapshot_mapping(signature)
        if expected_version is not None and (
            isinstance(expected_version, bool) or not isinstance(expected_version, Integral)
            or expected_version < 0
        ):
            raise ValueError("expected_version must be a nonnegative integer or None")
        now = _now() if tx_time is None else float(tx_time)
        # Mutation targets are literal identities, not legacy display aliases.
        current_rec = self._select_version(self.history(id), None, None)
        actual_version = 0 if current_rec is None else current_rec.version
        if expected_version is not None and expected_version != actual_version:
            raise VersionConflictError(
                f"RCDB record {id!r} expected version {expected_version}, current version {actual_version}")
        if current_rec is not None and now < float(current_rec.tx_from):
            raise ValueError(
                "mutation transaction time precedes the current RCDB version")
        previous = self.get_version(id, current_rec.version) if current_rec is not None else None
        if current_rec is not None and previous is None:
            raise ValueError(f"published RCDB record {id!r}@{current_rec.version} has no payload")
        history = self.commit_history(str(id)) if current_rec is not None else []
        parent = history[-1].link.digest if history else None
        policy = getattr(self, "_mutation_policy", None) or MutationPolicy()
        package = prepare_mutation(
            previous, rex, tx_time=now, actor=str(actor), policy=policy,
            parent_digest=parent,
            transition_signer=getattr(self, "_transition_signer", None),
            lineage_signer=getattr(self, "_lineage_signer", None))
        # Unconditional. The reference gated this on the store demanding a signature,
        # reasoning that a locally built unsigned package would not verify under a
        # permissive policy. It does: verify_mutation already accounts for the policy and
        # returns True for an unsigned package when none is required. So False here means
        # a real endpoint, lineage, temporal or policy failure, and gating the raise meant
        # that in the DEFAULT configuration a package that failed verification was written
        # anyway, which makes the chain vacuous exactly where it is most likely to be run.
        if not verify_mutation(package, previous=previous, policy=policy,
                               verifiers=getattr(self, "_mutation_verifiers", {}),
                               parent_digest=parent):
            raise ValueError(
                "mutation package does not verify against this store's history")
        version = self.next_version(str(id))
        self._store_commit_bytes(str(id), version, self._encode_commit(package))
        try:
            if signature is None:
                rec = self.put(id, rex, meta, tags, valid_from=valid_from,
                               valid_to=valid_to, analytics=analytics, voids=voids,
                               _tx_time=now, _mutation_commit=_MUTATION_COMMIT)
            else:
                from .engine import encode_native_payload
                from .header import BlobCodecSpec
                rec = self.put_prepared(id, encode_native_payload(rex, BlobCodecSpec()), signature, meta, tags,
                                       valid_from=valid_from, valid_to=valid_to, _tx_time=now,
                                       _mutation_commit=_MUTATION_COMMIT)
            if int(rec.version) != int(version):
                raise RuntimeError("RCDB version changed while publishing mutation")
            return rec
        except PublicationUncertainError:
            raise
        except Exception:
            self._delete_commit_bytes(str(id), version)
            raise

    def security_status(self) -> dict[str, Any]:
        """Bounded security configuration, with no key material and no storage paths."""
        policy = getattr(self, "_mutation_policy", None)
        return {
            "payload_encryption": bool(getattr(self, "_security_key_id", None)),
            "require_commits": bool(getattr(self, "_require_commits", False)),
            "signature_mode": str(getattr(self, "_signature_mode", "public")),
            "metadata_projection": getattr(self, "_metadata_fields", None) is not None,
            "transition_signature_required": bool(
                getattr(policy, "require_transition_signature", False)),
            "lineage_signature_required": bool(
                getattr(policy, "require_lineage_signature", False)),
            "allowed_signer_count": len(getattr(policy, "allowed_signers", ()) or ()),
        }

    def transfer_policy_digest(self):
        """Pin the destination's logical copy policy without exporting capabilities."""
        from rexgraph.value_codec import pack_value
        with self._transaction_lock:
            self._check_open()
            policy = getattr(self, "_mutation_policy", None)
            fields = getattr(self, "_metadata_fields", None)
            value = {"security": self.security_status(),
                     "metadata_fields": None if fields is None else sorted(fields),
                     "mutation_policy": None if policy is None else policy.manifest()}
            return hashlib.sha256(b"rexgraph-migration-destination-policy\x00"
                                  + pack_value(value)).hexdigest()

    @_serialized
    def put(self, id, rex, meta=None, tags=None, *, valid_from=None, valid_to=None,
            analytics=True, voids=False, _tx_time=None, _mutation_commit=False):
        """Append a new version of `id`. Template method: build the signature, delegate
        storage to _put_impl, then emit a best effort change feed event."""
        self._check_writable()
        if getattr(self, "_require_commits", False) and _mutation_commit is not _MUTATION_COMMIT:
            # An ungoverned write into a store that requires commits would create a
            # version with nothing attesting to it, which is the hole the requirement
            # exists to prevent.
            raise PermissionError(
                "this RCDB store requires TemporalRex mutation commits")
        _read_selector(id, None, _tx_time, None)
        valid_from, valid_to = _write_interval(valid_from, valid_to)
        raw_meta = snapshot_mapping(_priv(meta) or {})
        # `analytics=False` writes the structural facts (nV/nE/betti/chain_valid/
        # sectionings/merkle_root/labels_sample) and leaves the analytics columns unset.
        # It exists for corpus ingest, where those columns are 85% of the cost and
        # nothing on the retrieval path reads them. `backfill_analytics` fills them in
        # afterwards for whatever subset is worth it.
        raw_sig = structural_signature(rex, raw_meta, tags, analytics=analytics,
                                       voids=voids)
        # Taken from the RAW pair, before any minimisation of what gets stored, because a
        # store that protects its vocabulary keeps no plaintext labels to recompute them
        # from and the index must answer for the real term even when the record no longer
        # carries it.
        search_terms = tuple(sorted(_record_labels(raw_sig, raw_meta)))
        meta = snapshot_mapping(self._stored_meta(raw_meta))
        sig = snapshot_mapping(self._stored_signature(raw_sig))
        rec = self._put_impl(id, rex, sig, meta, tags, valid_from, valid_to,
                             search_terms, tx_time=_tx_time)
        self._emit("rcdb.put", id, rec.version, sig)
        return rec.detached()

    @_serialized
    def put_prepared(self, id, blob, sig, meta=None, tags=None, *,
                     valid_from=None, valid_to=None, _tx_time=None, _mutation_commit=False):
        """`put` for a complex already built, serialized and signed elsewhere.

        Same record, same versioning, same change feed event: it skips only the two
        steps the caller has already paid for. The caller owns the signature, so it also
        owns whether `analytics` were computed into it.
        """
        self._check_writable()
        if getattr(self, "_require_commits", False) and _mutation_commit is not _MUTATION_COMMIT:
            raise PermissionError("this RCDB store requires mutation commits")
        _read_selector(id, None, _tx_time, None)
        valid_from, valid_to = _write_interval(valid_from, valid_to)
        raw_meta = snapshot_mapping(_priv(meta) or {})
        search_terms = tuple(sorted(_record_labels(sig, raw_meta)))
        meta = snapshot_mapping(self._stored_meta(raw_meta))
        sig = snapshot_mapping(self._stored_signature(sig))
        rec = self._put_impl(id, _Prepared(blob), sig, meta, tags, valid_from, valid_to,
                             search_terms, tx_time=_tx_time)
        self._emit("rcdb.put", id, rec.version, sig)
        return rec.detached()

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        raise NotImplementedError

    def put_record(self, id, value, *, codec=None, meta=None, tags=None, signature=None,
                   valid_from=None, valid_to=None, tx_time=None, expected_version=None):
        """Publish a declared native value through this provider's shared engine.

        Codecs are installed process capabilities and must be declared in the
        immutable store header. Current graph mutation commits cannot govern
        generic values; required commit stores refuse this operation.
        """
        self._check_writable()
        from .codecs import VALUE_CODEC, record_codec
        from .header import StoreHeader
        _read_selector(id, None, tx_time, None)
        valid_from, valid_to = _write_interval(valid_from, valid_to)
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise ValueError("expected version requires a nonnegative native integer")
        if not isinstance(getattr(self, "header", None), StoreHeader):
            raise ValueError("generic records require native record-state storage")
        with self.write_scope():
            if not isinstance(getattr(self, "header", None), StoreHeader):
                raise ValueError("generic records require native record-state storage")
            if getattr(self, "_require_commits", False):
                raise PermissionError("current mutation commits cannot govern generic native records")
            reference = VALUE_CODEC if codec is None else codec
            provider = record_codec(reference)
            self.header.check_record_codec(reference.name, reference.version)
            current = self.get_record(id)
            if current is not None and current.id != id:
                current = None  # A legacy @version display alias is not this address.
            actual = 0 if current is None else current.version
            if expected_version is not None and actual != expected_version:
                raise VersionConflictError("record expected version differs from current native version")
            if self._load_commit_bytes(id, self.next_version(id)) is not None:
                raise ValueError("native record address has an unpublished mutation artifact")
            raw_meta = snapshot_mapping(_priv(meta) or {})
            raw_signature = ({"object_type": provider.object_type, "tags": list(tags or []),
                              "source": raw_meta.get("source", "")} if signature is None else snapshot_mapping(signature))
            if raw_signature.get("object_type") != provider.object_type:
                raise ValueError("native record signature must declare its codec's object type")
            terms = tuple(sorted(_record_labels(raw_signature, raw_meta)))
            encoded, _ = provider.prepare(value)
            stored_meta = snapshot_mapping(self._stored_meta(raw_meta))
            stored_signature = snapshot_mapping(self._stored_signature(raw_signature))
            record = self._put_impl(id, encoded, stored_signature, stored_meta, tags,
                                    valid_from, valid_to, terms, tx_time=tx_time)
            self._corpus_cache = None
            self._emit("rcdb.put", id, record.version, stored_signature)
            return record.detached()

    def get(self, id, *, as_of=None, valid_at=None, verify: bool = True):
        """Return the reconstructed RexGraph, or None.

        `verify=False` skips the load time integrity rebuild; see
        `deserialize_complex` for what that check is and when skipping it is right.
        """
        raise NotImplementedError

    def get_record(self, id, *, as_of=None, valid_at=None):
        raise NotImplementedError

    def get_version(self, id, version):
        """Return the reconstructed RexGraph for one SPECIFIC version, or
        None. Unlike `get(as_of=...)`, this is keyed directly by version
        number, not by a timestamp that could collide across versions
        written on the same tick."""
        raise NotImplementedError

    def checkpoint(self, *, max_frames=256, target_bytes=8*1024*1024):
        """Providers may coalesce physical replay without retiring logical history."""
        raise NotImplementedError("this provider has no persistent replay checkpoint")

    def plan_retention(self, policy=None):
        """Plan bounded collection of unpublished staging; preserve all history."""
        from .retention import plan_retention
        return plan_retention(self, policy)

    def apply_retention(self, plan):
        """Recheck a retention plan under provider writer arbitration, then apply."""
        from .retention import apply_retention
        return apply_retention(self, plan)

    def _record_at_version(self, id, version):
        """Select literal address metadata; compatibility providers retain history lookup.

        Native providers override this within their pinned read scope. Decoding
        stays in _read_published_record so payload checks and identity stay shared.
        """
        return next((row for row in self.history(id) if row.version == version), None)

    def read_record(self, id: str, *, version=None, as_of=None, valid_at=None):
        """Select and decode one published version under this handle's write lock.

        Exact ids win over legacy ``id@version`` display aliases. Explicit version
        and time selectors are different operations, never competing filters.
        None means no selected record; an existing record without its payload is
        an integrity failure, not a missing record. No numerical analytics run.
        """
        from .header import StoreHeader
        _read_selector(id, version, as_of, valid_at)
        with self._transaction_lock:
            if self._publication_uncertain:
                raise PublicationUncertainError("RCDB publication is uncertain; reopen and verify before reading")
            if version is not None:
                rec = self._record_at_version(id, version)
            elif isinstance(getattr(self, "header", None), StoreHeader):
                # Native providers share literal-ID/alias and bitemporal selection.
                # Their pinned read scopes and point selection avoid cloning an
                # entire retained lineage before decoding one published payload.
                rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
            else:
                # Backend legacy aliases vary; check the literal id's own history
                # first so a stored id containing @ is never silently shadowed.
                records = self.history(id)
                rec = self._select_version(records, as_of, valid_at)
                if not records:
                    split = self._split_versioned_id(id)
                    if split is not None:
                        if as_of is not None or valid_at is not None:
                            raise ValueError("a version display alias cannot be combined with time selectors")
                        base, selected = split
                        rec = next((r for r in self.history(base) if r.version == selected), None)
            if rec is None:
                return None
            return self._read_published_record(rec)

    def _read_published_record(self, rec):
        """Decode a selected record while the caller holds the handle's lock."""
        metadata = rec.detached()
        value = self.get_version(metadata.id, metadata.version)
        if value is None and metadata.is_complex:
            raise ValueError(f"published RCDB record {metadata.id!r}@{metadata.version} has no payload")
        from rexgraph.object_identity import object_digest
        if metadata.envelope is not None and metadata.envelope.codec != "rexgraph.safetensors":
            from .codecs import record_codec
            from .header import CodecRef
            provider = record_codec(CodecRef(metadata.envelope.codec, metadata.envelope.codec_version))
            digest = provider.identity(provider.encode(value))
        else:
            digest = object_digest(value)
        return RecordSnapshot(metadata, value, digest)

    def state_manifest(self):
        """Canonical logical history, excluding backend layout and derived analytics.

        This reads every published payload under this handle's lock. It is not an
        index hash, a cheap statistic, or a multi process snapshot. Commit identities
        are included when present; hashing them is not signature/lineage verification.
        """
        with self._transaction_lock:
            if self._publication_uncertain:
                raise PublicationUncertainError("RCDB publication is uncertain; reopen and verify before reading")
            rows = []
            for rec in sorted(self.list(limit=sys.maxsize, include_history=True),
                              key=lambda r: (r.id, r.version)):
                snapshot = self._read_published_record(rec)
                metadata = snapshot.record
                artifact = self._load_commit_bytes(metadata.id, metadata.version)
                rows.append({
                    "id": metadata.id, "version": int(metadata.version),
                    "state_digest": snapshot.state_digest,
                    "created": float(metadata.created), "tx_from": float(metadata.tx_from),
                    "tx_to": None if metadata.tx_to is None else float(metadata.tx_to),
                    "valid_from": None if metadata.valid_from is None else float(metadata.valid_from),
                    "valid_to": None if metadata.valid_to is None else float(metadata.valid_to),
                    "meta": metadata.meta,
                    "tags": sorted(set(metadata.signature.get("tags", []))),
                    "commit": None if artifact is None else self._decode_commit(artifact).digest,
                })
            return {"object_type": "RCDBLogicalState", "version": self.logical_state_version, "records": rows}

    def state_digest(self):
        """Hash logical state v2 without coercing native exact metadata to JSON."""
        from rexgraph.value_codec import pack_value
        return hashlib.sha256(b"rexgraph-rcdb-logical-state\x00\x02"
                              + pack_value(self.state_manifest())).hexdigest()

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
        Only a trailing @<positive int> splits; a bare id or non string is None."""
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
            if _ACTIVITY_HOOK is None:
                return
            _ACTIVITY_HOOK("rcdb:" + self.backend, action,
                           {"id": id, "version": version, "nV": sig.get("nV"),
                            "nE": sig.get("nE"), "tags": sig.get("tags"),
                            "lineage_id": id})
        except Exception:
            pass

    def list(self, limit: int = 100, offset: int = 0, *, as_of=None,
             valid_at=None, include_history: bool = False) -> builtins.list[ComplexRecord]:
        """The store's records. `as_of`/`valid_at` read it AS IT STOOD, selecting the
        version current at that transaction/validity time instead of the latest."""
        raise NotImplementedError

    def query(self, limit: int = 100, *, as_of=None, valid_at=None,
              **predicate) -> builtins.list[ComplexRecord]:
        """Structural query: select complexes by their topology.

        `as_of`/`valid_at` apply the predicate to the version that was current then,
        not to the latest one. That distinction is the whole point: matching a
        predicate against today's record and then reading yesterday's blob silently
        drops anything whose structure or vocabulary has since changed."""
        raise NotImplementedError

    def delete(self, id: str) -> bool:
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        with self.read_transaction():
            recs = self.list(limit=sys.maxsize)
            graphs = [r for r in recs if r.is_complex]
            return {
                "backend": self.backend, "count": len(recs), "n_records": len(recs),
                "n_versions": len(self.list(limit=sys.maxsize, include_history=True)),
                "total_vertices": sum(r.signature.get("nV", 0) for r in graphs),
                "total_edges": sum(r.signature.get("nE", 0) for r in graphs),
                "mean_kappa": (round(float(np.mean([r.signature.get("kappa_mean") or 0
                                                    for r in graphs])), 4) if graphs else None),
            }

    def close(self):
        pass


# in memory backend

class MemoryStore(RCStore):
    backend = "memory"
    _cache_corpus = True

    def __init__(self, *, compression=None, record_codecs=None):
        from .engine import StoreState
        from .header import BlobCodecSpec, StoreHeader, record_codec_inventory
        from .store_identity import StoreIdentity
        if compression is not None and not isinstance(compression, BlobCodecSpec):
            raise TypeError("memory compression requires a declared BlobCodecSpec")
        self._header = StoreHeader(StoreIdentity.create(self.backend), compression or BlobCodecSpec(),
                                   record_codec_inventory(record_codecs))
        self._store_identity = self.header.identity
        self._state = StoreState(self.header)
        self._recs = self._state._rows  # Compatibility inspection, not an import/publication API.
        self._blobs: dict[tuple[str, int], bytes] = {}
        self._commit_blobs: dict[tuple[str, int], bytes] = {}
        self._closed = False

    def _load_store_identity(self):
        return self.header.identity

    @property
    def header(self):
        return self._header

    @property
    def blob_codec(self):
        return self.header.compression

    def _check_open(self):
        if self._closed:
            raise ValueError("memory store handle is closed")

    @contextlib.contextmanager
    def write_scope(self):
        with self.read_transaction():
            yield self

    def _serialize_payload(self, rex):
        from .engine import encode_native_payload
        return self._encode_payload(encode_native_payload(rex, self.blob_codec), object_type="RCDB.Rex")

    def _store_commit_bytes(self, id, version, blob):
        if self._state.change_for(id, version) is not None:
            raise ValueError("cannot replace a published memory mutation artifact")
        raw = bytes(blob)
        existing = self._commit_blobs.get((id, version))
        if existing is not None and existing != raw:
            raise ValueError("unpublished mutation artifact already occupies this memory address")
        self._commit_blobs[(id, version)] = raw

    def _load_commit_bytes(self, id, version):
        import hashlib
        raw = self._commit_blobs.get((id, version))
        change = self._state.change_for(id, version)
        if change is not None:
            if raw is None and change.commit_digest is not None:
                raise ValueError("published memory mutation artifact is missing")
            if raw is not None and hashlib.sha256(raw).hexdigest() != change.commit_digest:
                raise ValueError("memory mutation artifact differs from its published digest")
        return raw

    def _delete_commit_bytes(self, id, version):
        if self._state.change_for(id, version) is not None:
            raise ValueError("cannot remove a published memory mutation artifact")
        self._commit_blobs.pop((id, version), None)

    def next_version(self, id):
        with self.read_transaction():
            if id in self._recs and id not in self._state._highwater:
                raise ValueError("unpublished legacy memory metadata requires explicit migration before writing")
            return self._state.next_version(id)

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        now = _now() if tx_time is None else float(tx_time)
        v = self.next_version(id)
        rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {}, version=v,
                            tx_from=now, tx_to=None,
                            valid_from=valid_from if valid_from is not None else now,
                            valid_to=valid_to)
        rec, blob = self._record_payload(rec, rex)
        commit = self._commit_blobs.get((id, v))
        import hashlib
        frame = self._state.prepare_put(rec, blob_digest=hashlib.sha256(blob).hexdigest(),
            commit_digest=None if commit is None else hashlib.sha256(commit).hexdigest())
        self._blobs[(id, v)] = blob  # Becomes visible only after the checked state transition.
        self._state.apply(frame)
        return rec

    @_owned_records
    def get_record(self, id, *, as_of=None, valid_at=None):
        _read_selector(id, None, as_of, valid_at)
        split = self._split_versioned_id(id) if id not in self._recs else None
        if id in self._state._highwater or (split is not None and split[0] in self._state._highwater):
            return self._state.selected(id, as_of=as_of, valid_at=valid_at)
        rec = self._select_version(self._recs.get(id, []), as_of, valid_at)
        if rec is None and id not in self._recs:
            split = self._split_versioned_id(id)
            if split is not None:
                if as_of is not None or valid_at is not None:
                    raise ValueError("version display aliases cannot combine time selectors")
                base, v = split
                # a lineage() display id: resolve the explicit version directly
                # (version is explicit, so as_of/valid_at do not apply)
                rec = next((r for r in self.history(base) if r.version == v), None)
        return rec

    def get(self, id, *, as_of=None, valid_at=None, verify: bool = True):
        with self.read_transaction():
            rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
            return None if rec is None else self._read_version(rec, verify=verify)

    def get_version(self, id, version):
        _read_selector(id, version, None, None)
        if version is None:
            raise ValueError("record version must be a positive integer")
        with self.read_transaction():
            rec = (self._state.record(id, version) if id in self._state._highwater else
                   next((r for r in self._recs.get(id, []) if r.version == version), None))
            return None if rec is None else self._read_version(rec)

    def _record_at_version(self, id, version):
        with self.read_transaction():
            if id in self._state._highwater or id not in self._recs:
                return self._state.record(id, version)
            # Injected legacy rows need not have contiguous native allocations.
            return super()._record_at_version(id, version)

    def _read_version(self, record, *, verify=True):
        import hashlib
        blob = self._blobs.get((record.id, record.version))
        if blob is None:
            raise ValueError("published memory record payload is missing")
        change = self._state.change_for(record.id, record.version)
        if change is None:
            if record.envelope is not None:
                raise ValueError("bound memory record has no published engine transition")
        else:
            if hashlib.sha256(blob).hexdigest() != change.blob_digest:
                raise ValueError("memory record payload differs from its published content digest")
            if change.commit_digest is not None and self._load_commit_bytes(record.id, record.version) is None:
                raise ValueError("published memory mutation artifact is missing")
        return self._read_record_payload(record, blob, verify=verify)

    @_owned_records
    def history(self, id):
        _read_selector(id, None, None, None)
        return list(self._recs.get(id, []))

    @_owned_records
    def list(self, limit=100, offset=0, *, as_of=None, valid_at=None,
             include_history=False):
        if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
            raise ValueError("memory collection bounds require nonnegative native integers")
        _read_selector("selector", None, as_of, valid_at)
        return self._state.select_records(limit=limit, offset=offset, as_of=as_of,
                                         valid_at=valid_at, include_history=include_history)

    @_owned_records
    def query(self, limit=100, *, as_of=None, valid_at=None, include_history=False, **predicate):
        if type(limit) is not int or limit < 0:
            raise ValueError("memory query limit requires a nonnegative native integer")
        if set(predicate)-_QUERY_KEYS:
            raise TypeError("unsupported query keys: "+", ".join(sorted(set(predicate)-_QUERY_KEYS))
                            +". Supported: "+", ".join(sorted(_QUERY_KEYS)))
        return self._state.select_records(limit=limit, as_of=as_of, valid_at=valid_at,
                                         include_history=include_history, predicate=predicate)

    @_serialized
    def delete(self, id, *, tx_time=None, expected_version=None):
        _read_selector(id, None, tx_time, None)
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise ValueError("expected version requires a nonnegative native integer")
        self._claim_delete(id)
        current = self._state.current(id)
        actual = 0 if current is None else current.version
        if expected_version is not None and expected_version != actual:
            raise VersionConflictError("deletion expected version differs from current memory record")
        frame = self._state.prepare_delete(id, _now() if tx_time is None else tx_time)
        if frame is None:
            return False
        self._state.apply(frame)
        self._emit("rcdb.delete", id, frame.mutation.version, {})
        return True

    @property
    def change_cursor(self):
        with self.read_transaction():
            return self._state.cursor

    def changes(self, after=None, *, limit=100):
        with self.read_transaction():
            return self._state.changes(after, limit=limit)

    def cursor_for(self, change):
        with self.read_transaction():
            return self._state.cursor_for(change)

    def tombstone(self, id):
        _read_selector(id, None, None, None)
        with self.read_transaction():
            return self._state.tombstone(id)

    def commit_history(self, id):
        with self.read_transaction():
            return [self._decode_commit(raw) for row in self._state.incarnation_history(id)
                    if (raw := self._load_commit_bytes(id, row.version)) is not None]

    def _commit_predecessor_reset(self, record):
        change = self._state.change_for(record.id, record.version)
        return change is not None and change.previous_version == 0

    def close(self):
        with self._transaction_lock:
            self._closed = True


# file backend (default, no deps beyond io)

def _version_selector(as_of, valid_at):
    """The rows `_select_version` would admit, as a mask over the cochains.

    Mirrors that method's rules rather than approximating them, but the caller still
    runs `_select_version` on what comes back, so this only has to be a superset for
    the answer to stay right.
    """
    def select(c):
        tx_to = c["tx_to"]
        if as_of is None and valid_at is None:
            return np.isnan(tx_to)                       # the live row
        keep = np.ones(tx_to.shape, bool)
        if as_of is not None:
            keep &= (c["tx_from"] <= as_of) & (np.isnan(tx_to) | (as_of < tx_to))
        if valid_at is not None:
            lo = np.where(np.isnan(c["valid_from"]), c["tx_from"], c["valid_from"])
            hi = c["valid_to"]
            keep &= (lo <= valid_at) & (np.isnan(hi) | (valid_at < hi))
        return keep
    return select


class _LazyIndex(dict):
    """`{id -> [ComplexRecord]}` that builds an id's records the first time it is asked
    for, from the snapshot cochains and the log frames that land on it.

    A get costs one id, a scan costs the ids it scans, and opening a store costs
    neither. Built lists are cached and mutated in place, which `put` relies on to close
    the prior version's transaction bound.

    Every lookup and every view is overridden: CPython reaches into the underlying
    storage directly for `in`, `len`, iteration and `values`, so inheriting them would
    report only the ids that happen to have been built already.

    The live id set is MAINTAINED rather than recomputed. The snapshot row map is kept
    intact for the same reason: `list` orders on the cochains, which needs to know which
    rows belong to which id without building anything.
    """

    __slots__ = ("_snap", "_rows", "_pending", "_gone", "_ids", "_order")

    def __init__(self, snap, rows_by_id, pending, eager):
        super().__init__(eager)
        self._snap, self._rows, self._pending = snap, rows_by_id, pending
        self._gone: set = set()
        self._ids: set = set(rows_by_id) | set(pending) | set(eager)
        self._order = None

    def _build(self, key):
        """The records for `key`, or None if the store does not hold it."""
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._gone:
            return None
        rows = self._rows.get(key)
        pend = self._pending.get(key)
        if rows is None and pend is None:
            return None
        recs = []
        if rows:
            from . import index as _ix
            recs = sorted((_ix.record_at(self._snap, r) for r in rows),
                          key=lambda r: r.version)
        for rec in pend or ():
            recs = [r for r in recs if r.version != rec.version]
            for prior in recs:
                if prior.tx_to is None:
                    prior.tx_to = rec.tx_from
            recs.append(rec)
            recs.sort(key=lambda r: r.version)
        dict.__setitem__(self, key, recs)
        return recs

    def _build_all(self):
        for key in self._ids:
            self._build(key)

    def __missing__(self, key):
        recs = self._build(key)
        if recs is None:
            raise KeyError(key)
        return recs

    def get(self, key, default=None):
        recs = self._build(key)
        return default if recs is None else recs

    def setdefault(self, key, default=None):
        recs = self._build(key)
        if recs is None:
            dict.__setitem__(self, key, default)
            self._ids.add(key)
            self._gone.discard(key)
            self._order = None
            return default
        return recs

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._ids.add(key)
        self._gone.discard(key)
        self._order = None

    def pop(self, key, *default):
        recs = self._build(key)
        if recs is None:
            if default:
                return default[0]
            raise KeyError(key)
        self._gone.add(key)
        self._ids.discard(key)
        self._order = None
        return dict.pop(self, key)

    def __contains__(self, key):
        return key in self._ids

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        self._build_all()
        return iter(list(self._ids))

    def keys(self):
        self._build_all()
        return dict.keys(self)

    def values(self):
        self._build_all()
        return dict.values(self)

    def items(self):
        self._build_all()
        return dict.items(self)

    def copy(self):
        self._build_all()
        return dict(self)

    def narrow(self, predicate, as_of=None):
        """The ids that could match, or None if there is nothing to narrow against.

        A superset. Ids the log knows and ids put since are always included, since the
        cochains only describe the snapshot, and the caller still evaluates the full
        predicate on what comes back.
        """
        if self._snap is None:
            return None
        from . import index as _ix
        rows = _ix.rows_for(self._snap, as_of=as_of, **predicate)
        out = set(_ix.ids_of(self._snap, rows)) | set(self._pending)
        out |= {k for k in dict.keys(self) if k not in self._rows}
        # the snapshot still holds the rows of a since deleted id, so the live set
        # decides. Maintained, so this is the size of the hit, not of the store.
        return out & self._ids

    #### ordering without materialising
    #
    # `list` wants the newest N records. Sorting requires every tx_from, but tx_from is
    # a cochain, so the ordering is a read over an array and only the page has to become
    # a record. Ids the log has touched are already built and merge in directly.

    def _order_arrays(self):
        """(owner_code per snapshot row, the ids those codes index). Cached."""
        if self._order is None:
            import numpy as np
            n = self._snap["n"] if self._snap else 0
            owner = np.full(n, -1, np.int64)
            ids = []
            for code, (rid, rows) in enumerate(self._rows.items()):
                ids.append(rid)
                owner[np.asarray(rows, np.int64)] = code
            self._order = (owner, ids)
        return self._order

    def ordered_ids(self, select):
        """Ids newest first, where `select(measures)` marks the rows a selector admits.

        Returns None when there is no snapshot to order on, so the caller keeps its
        own path.
        """
        if self._snap is None:
            return None
        import numpy as np
        owner, ids = self._order_arrays()
        c = self._snap["measures"]
        ok = select(c) & (owner >= 0)
        rows = np.flatnonzero(ok)
        out = []
        if rows.size:
            # newest version per id, then that version's tx_from
            order = np.lexsort((c["version"][rows], owner[rows]))
            rows = rows[order]
            o = owner[rows]
            last = np.flatnonzero(np.r_[o[1:] != o[:-1], True])
            pick = rows[last]
            tx = c["tx_from"][pick]
            out = [(float(t), ids[int(owner[r])])
                   for t, r in zip(tx, pick, strict=True)]
        # anything the log touched or a put created is already a record: merge it in
        # and let it supersede the snapshot's reading of the same id
        touched = set(self._pending) | {k for k in dict.keys(self)}
        out = [(t, i) for t, i in out if i not in touched]
        for rid in touched:
            if rid not in self._ids:
                continue
            recs = self._build(rid) or []
            if recs:
                out.append((max(r.tx_from for r in recs), rid))
        out.sort(key=lambda p: -p[0])
        return [i for _t, i in out]


class FileStore(RCStore):
    """Legacy file layout reader. ``read_only=False`` opts into temporary compatibility writing."""
    backend = "file"
    _cache_corpus = True

    def __init__(self, root: str, *, read_only=True):
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool")
        self.read_only = read_only
        self.root = os.fspath(root)
        if _existing_backend(self.root) not in {None, "file"}:
            raise ValueError("FileStore cannot adopt another store layout")
        if read_only:
            from .legacy import check_source_root, source_fingerprint
            check_source_root(self.root, "file")
            self._legacy_source_fingerprint = source_fingerprint(self.root)
        else:
            os.makedirs(os.path.join(root, "blobs"), exist_ok=True)
        # The binary pair is authoritative. The json pair is read when it is all that
        # exists, so a store written by an earlier version opens unchanged and is
        # converted on the next compaction.
        self._index_path = os.path.join(root, "index.rexidx")
        self._log_path = os.path.join(root, "index.rexlog")
        self._legacy_index = os.path.join(root, "index.json")
        self._legacy_log = os.path.join(root, "index.log")
        # Loaded once. Re reading the index on every call, and rewriting it on every
        # put, is what made ingest quadratic; the log means the cache stays authoritative
        # and each change costs one line.
        self._idx = self._read_index()
        if not os.path.exists(os.path.join(self.root, ".rcdb-identity")):
            from .store_identity import bound_identity
            if bound_identity(record for versions in self._idx.values() for record in versions) is not None:
                self._load_store_identity()  # Refuse missing ownership; never create a new UUID.
        if read_only and source_fingerprint(self.root) != self._legacy_source_fingerprint:
            raise ValueError("legacy source changed while opening its read-only snapshot")

    def _load_store_identity(self):
        from .store_identity import bound_identity, local_identity
        from .journal import journal_identity
        path = os.path.join(self.root, ".rcdb-identity")
        known = journal_identity(self._log_path)
        if known is None and not os.path.exists(path) and hasattr(self, "_idx"):
            known = bound_identity(record for versions in self._idx.values() for record in versions)
        return local_identity(path, backend="file", existing_id=known, read_only=self.read_only)

    def _read_index(self) -> dict[str, builtins.list[ComplexRecord]]:
        """`{id -> [ComplexRecord, ...]}`, the snapshot with the log layered on top.

        Lazy over the snapshot: opening a store reads the columns, and a record is
        built when its id is asked for, so open no longer costs one object per stored
        record. A legacy json index is materialised eagerly, since a document has to be
        parsed in full before any of it can be read.
        """
        eager: dict[str, list[ComplexRecord]] = {}
        snap, rows_by_id = None, {}
        # A snapshot, if one exists: written by an older version of this store, or
        # left by compaction. Read first so the log layers on top of it.
        if os.path.exists(self._index_path):
            from . import index as _ix
            # A log after compaction contains only the new tail. Its existence
            # cannot establish that it can replace an unreadable base snapshot.
            snap = _ix.read(self._index_path)
            for row, rid in enumerate(snap["ids"]):
                rows_by_id.setdefault(str(rid), []).append(row)
        elif os.path.exists(self._legacy_index):
            from .legacy import loads_metadata, record_metadata
            with open(self._legacy_index, "rb") as f:
                raw = loads_metadata(f.read())
            if type(raw) is not dict:
                raise ValueError("legacy index must declare a record mapping")
            for id, v in raw.items():
                if type(id) is not str or not id or type(v) not in (dict, list):
                    raise ValueError("invalid legacy index record address or versions")
                eager[id] = [record_metadata(x, id) for x in ([v] if type(v) is dict else v)]
        # then the append only log. Rewriting the whole index on every put made the
        # cost of a put grow with the store: 4 ms at a hundred records, 35 ms at
        # sixteen hundred, which is quadratic ingest. One frame per change instead.
        entries = []
        if snap is None and os.path.exists(self._legacy_log):
            from .legacy import line_changes
            entries.extend({"op": op, "id": rid, "record": rec}
                           for op, rid, rec, _extra in line_changes(self._legacy_log))
        if os.path.exists(self._log_path):
            from . import index as _ix
            from .journal import journal_identity
            owner = self.store_id if journal_identity(self._log_path) is not None else None
            entries.extend({"op": op, "id": rid, "record": rec}
                           for op, rid, rec, _x in _ix.log_read(self._log_path, store_id=owner))
        pending: dict[str, list[ComplexRecord]] = {}
        for entry in entries:
            rid = entry.get("id")
            if entry.get("op") == "delete":
                rows_by_id.pop(rid, None)
                pending.pop(rid, None)
                eager.pop(rid, None)
                continue
            raw = entry["record"]
            # the frame log decodes to a record; the line log it replaces gave a dict
            rec = raw if isinstance(raw, ComplexRecord) else ComplexRecord.from_dict(raw)
            versions = pending.setdefault(rid, eager.pop(rid, []))
            versions[:] = [r for r in versions if int(r.version) != int(rec.version)]
            versions.append(rec)
            versions.sort(key=lambda r: int(r.version))
        if not rows_by_id and not pending:
            return eager
        return _LazyIndex(snap, rows_by_id, pending, eager)

    def _append_log(self, entry: dict) -> None:
        """One frame per change: a length, an op, and the payload once. The line log
        rewrote every field name and every number as text on each put."""
        from . import index as _ix
        rec = entry.get("record")
        if rec is not None and not isinstance(rec, ComplexRecord):
            rec = ComplexRecord.from_dict(rec)
        _ix.log_append(self._log_path, entry.get("op", "put"), entry["id"], rec, store_id=self.store_id)

    def _write_index(self, idx: dict[str, builtins.list[ComplexRecord]]):
        """Write a full snapshot and drop the log. This is compaction, not the write
        path: callers that used it to persist one change now append instead."""
        from . import index as _ix
        rows = [(rid, r) for rid, versions in idx.items() for r in versions]
        tmp = self._index_path + ".tmp"
        _ix.write(tmp, _ix.build(rows))
        os.replace(tmp, self._index_path)
        with contextlib.suppress(OSError):
            os.remove(self._log_path)           # written here, and folded in above
        # The json files this store was read from are now redundant, and leaving them
        # in place is a hazard: lose the snapshot and the store silently reopens on a
        # stale index instead of an empty one. Renamed rather than removed, so a
        # rollback to a version that reads them still has them.
        for legacy in (self._legacy_log, self._legacy_index):
            if os.path.exists(legacy):
                with contextlib.suppress(OSError):
                    os.replace(legacy, legacy + ".migrated")

    def recompress(self, *, verify: bool = True, force: bool = False,
                   workers: int | None = None, progress=None) -> dict:
        """Rewrite blobs written before compression, in place.

        Reading is already correct without this (`decompress_blob` passes a legacy blob
        through) so this is purely about the bytes on disk. A store written before
        `serialize_complex` framed its output keeps whatever it has until asked.

        VERIFY THEN REPLACE, never the other way round: the new bytes are deserialized
        and checked against the old complex's shape and Merkle root before anything is
        written, and the write is a temp file plus an atomic rename. A blob that fails
        any of that is left exactly as it was and counted in `failed`. Nothing is
        deleted: an entry either gets smaller or stays the same.
        """
        self._check_writable()
        out = {"total": 0, "rewritten": 0, "skipped": 0, "failed": 0,
               "before": 0, "after": 0, "failures": []}
        seen, paths = set(), []
        for id_ in list(self._idx):
            for rec in self.history(id_):
                key = (rec.id, rec.version)
                if key in seen:
                    continue
                seen.add(key)
                out["total"] += 1
                path = next((q for q in self._blob_read_paths(rec.id, rec.version)
                             if os.path.exists(q)), None)
                if path is None:
                    out["skipped"] += 1
                    continue
                paths.append(path)

        def record(res):
            _p, before, after, err, rewrote = res
            out["before"] += before
            out["after"] += after
            if err:
                out["failed"] += 1
                out["failures"].append({"path": _p, "error": err})
            elif rewrote:
                out["rewritten"] += 1
            else:
                out["skipped"] += 1
            if progress is not None:
                progress(out)

        if not workers or workers <= 1 or len(paths) < 2:
            for path in paths:
                record(_recompress_one(path, verify, force))
            return out
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
            for res in ex.map(_recompress_one, paths, [verify] * len(paths),
                              [force] * len(paths), chunksize=32):
                record(res)
        return out

    def compact(self) -> dict:
        """Fold the log into a snapshot. Optional: reading is correct either way."""
        self._check_writable()
        before = os.path.getsize(self._log_path) if os.path.exists(self._log_path) else 0
        self._write_index(self._read_index())
        return {"log_bytes_reclaimed": before}

    @staticmethod
    def _safe_name(id: str) -> str:
        """Filesystem safe, REVERSIBLE, collision free encoding of a record id.

        The shared codec, with the path reserved set (which includes '@', the version
        separator used below). The previous scheme replaced every non alphanumeric
        character with '_', which is lossy: 'core/alpha' and 'core_alpha' both became
        'core_alpha', so the second put silently overwrote the first blob while the
        index kept both records. Ids like 'doc:agent/rcdb.py' are exactly what a
        knowledge core is keyed by.
        """
        from rexgraph.state import RESERVED_PATH, encode_name
        return encode_name(id, RESERVED_PATH)

    @staticmethod
    def _sanitized_name(id: str) -> str:
        """Encode legacy path components by replacing unsupported characters with underscores."""
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in id)

    def _blob_path(self, id: str, version: int) -> str:
        return os.path.join(self.root, "blobs",
                            "%s@%d.safetensors" % (self._safe_name(id), version))

    def _blob_read_paths(self, id: str, version: int):
        """Every path a blob for (id, version) may live at, newest scheme first: the
        reversible encoding, then the lossy one, then the pre versioned layout."""
        b = os.path.join(self.root, "blobs")
        paths = [self._blob_path(id, version)]
        if not self._legacy_commit_collides(id, version):
            paths.append(os.path.join(b, "%s@%d.safetensors" % (self._sanitized_name(id), version)))
        if not any(str(rid) != str(id) and self._sanitized_name(str(rid)) == self._sanitized_name(id)
                   for rid in self._idx):
            paths.append(self._legacy_blob_path(id))
        return list(dict.fromkeys(paths))

    def _legacy_blob_path(self, id: str) -> str:
        return os.path.join(self.root, "blobs", self._sanitized_name(id) + ".safetensors")

    def _commit_path(self, id, version) -> str:
        d = os.path.join(self.root, "commits")
        return os.path.join(d, f"{self._safe_name(str(id))}@{int(version)}.rexpkg")

    def _legacy_commit_path(self, id, version) -> str:
        return os.path.join(self.root, "commits",
                            f"{self._sanitized_name(str(id))}@{int(version)}.rexpkg")

    def _legacy_commit_collides(self, id, version) -> bool:
        name = self._sanitized_name(str(id))
        return any(str(rid) != str(id) and self._sanitized_name(str(rid)) == name
                   and any(int(r.version) == int(version) for r in self.history(rid))
                   for rid in self._idx)

    def _store_commit_bytes(self, id, version, blob):
        self._check_writable()
        path = self._commit_path(id, version)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".rcdb-commit-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(bytes(blob))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _load_commit_bytes(self, id, version):
        path = self._commit_path(id, version)
        if not os.path.exists(path):
            legacy = self._legacy_commit_path(id, version)
            if not os.path.exists(legacy):
                return None
            if self._legacy_commit_collides(id, version):
                raise ValueError(f"ambiguous legacy commit for {id!r} version {version}")
            if self.read_only:
                path = legacy
            else:
                os.replace(legacy, path)
        with open(path, "rb") as fh:
            return fh.read()

    def _delete_commit_bytes(self, id, version):
        with contextlib.suppress(OSError):
            os.unlink(self._commit_path(id, version))
        if not self._legacy_commit_collides(id, version):
            with contextlib.suppress(OSError):
                os.unlink(self._legacy_commit_path(id, version))

    def next_version(self, id):
        rs = self._idx.get(id)
        return (rs[-1].version + 1) if rs else 1

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        now = _now() if tx_time is None else float(tx_time)
        idx = self._idx
        versions = idx.get(id, [])
        v = (versions[-1].version + 1) if versions else 1
        rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {}, version=v,
                            tx_from=now, tx_to=None,
                            valid_from=valid_from if valid_from is not None else now,
                            valid_to=valid_to)
        rec, blob = self._record_payload(rec, rex)
        path = self._blob_path(id, v)
        fd, temporary = tempfile.mkstemp(prefix=".rcdb-payload-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        # Payload first, durable publication next, cached metadata last. An
        # unreferenced payload after a log failure is not a published version.
        self._append_log({"op": "put", "id": id, "record": rec.to_storage_dict()})
        for r in versions:
            if r.tx_to is None:
                r.tx_to = now
        idx.setdefault(id, []).append(rec)
        return rec

    @_owned_records
    def get_record(self, id, *, as_of=None, valid_at=None):
        idx = self._idx
        rec = self._select_version(idx.get(id, []), as_of, valid_at)
        if rec is None and id not in idx:
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

    def get(self, id, *, as_of=None, valid_at=None, verify: bool = True):
        rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
        if rec is None:
            return None
        # rec.id is the record's OWN stored id (see MemoryStore.get for why
        # this, not the local `id`, is the correct blob key on a fallback hit).
        blob = self._read_blob(rec.id, rec.version)
        return self._read_record_payload(rec, blob, verify=verify) if blob is not None else None

    def get_version(self, id, version):
        rec = next((r for r in self.history(id) if r.version == version), None)
        if rec is None:
            return None
        blob = self._read_blob(id, version)
        return self._read_record_payload(rec, blob) if blob is not None else None

    @_owned_records
    def history(self, id):
        return list(self._idx.get(id, []))

    @_owned_records
    def list(self, limit=100, offset=0, *, as_of=None, valid_at=None,
             include_history=False):
        """The store's records, newest first.

        Ordering runs on the tx_from cochain and only the requested window becomes a
        record, so asking for a page costs the page rather than the store. The full
        walk stays for `include_history`, which wants every version by definition.
        """
        idx = self._idx
        # only when the window is narrower than the store. Asked for everything,
        # building in bulk beats resolving ids one at a time, and "the window is
        # everything" is a fact about the call rather than a tuned size. Tested before
        # the ordering runs, or the full listing pays for both paths.
        if (not include_history and isinstance(idx, _LazyIndex)
                and offset + limit < len(idx)):
            order = idx.ordered_ids(_version_selector(as_of, valid_at))
            if order is not None:
                out = []
                for rid in order:
                    rec = self._select_version(idx.get(rid, []), as_of, valid_at)
                    if rec is None:
                        continue                  # the cochain read is a superset
                    out.append(rec)
                    if len(out) >= offset + limit:
                        break
                return out[offset:offset + limit]
        recs = ([r for versions in idx.values() for r in versions] if include_history
                else [self._select_version(versions, as_of, valid_at)
                      for versions in idx.values()])
        recs = [r for r in recs if r is not None]
        recs.sort(key=lambda r: -r.tx_from)
        return recs[offset:offset + limit]

    @_owned_records
    def query(self, limit=100, *, as_of=None, valid_at=None, **predicate):
        """Narrow on the columns, then evaluate the full predicate on the survivors.

        The narrowing pass is a superset, so `_matches` still decides every record and
        the answer does not depend on which keys happen to have a column.
        """
        unknown = sorted(set(predicate) - _QUERY_KEYS)
        if unknown:
            raise TypeError(
                f"unsupported query key(s): {', '.join(unknown)}. Supported: "
                f"{', '.join(sorted(_QUERY_KEYS))}")
        ids = (self._idx.narrow(predicate, as_of=as_of)
               if isinstance(self._idx, _LazyIndex) else None)
        if ids is None:
            recs = self.list(limit=10 ** 9, as_of=as_of, valid_at=valid_at)
        else:
            recs = [self._select_version(self._idx.get(i, []), as_of, valid_at)
                    for i in ids]
            recs = [r for r in recs if r is not None]
            recs.sort(key=lambda r: -r.tx_from)
        out = [r for r in recs if _matches(r.signature, predicate, r.meta)]
        return out[:limit]

    @_serialized
    def delete(self, id):
        reclaim = self._claim_delete(id)
        idx = self._idx
        versions = idx.pop(id, None)
        existed = versions is not None
        if existed:
            self._append_log({"op": "delete", "id": id})
            for r in versions:
                for p in self._blob_read_paths(id, r.version):
                    with contextlib.suppress(OSError):
                        os.unlink(p)
            self._reclaim_commits(id, reclaim)
            self._emit("rcdb.delete", id, 0, {})
        return existed


# SQL backend (any SQLAlchemy database)

class SQLStore(RCStore):
    """Indexed SQL records with the common state engine for new SQLite stores.

    Populated headerless databases keep their compatibility semantics. They need
    explicit migration before native tombstones/cursors can be claimed. ``native``
    can select that compatibility path for an empty database; it cannot downgrade
    an existing native store. Compression is persisted in the native store header.
    """
    backend = "sql"

    # signature fields promoted to indexed columns for in database queries
    _INDEX_COLS = {
        "nV": "INTEGER", "nE": "INTEGER", "betti1": "INTEGER",
        "kappa_mean": "FLOAT", "chain_valid": "BOOLEAN", "source": "VARCHAR(256)",
    }

    # bitemporal columns added on top of the (id, version) composite key
    _TEMPORAL_COLS = {
        "version": "INTEGER", "tx_from": "FLOAT", "tx_to": "FLOAT",
        "valid_from": "FLOAT", "valid_to": "FLOAT",
    }

    def _load_store_identity(self):
        from .store_identity import sql_identity
        from .sql_journal import sql_state_header
        from sqlalchemy import select
        self._check_open()
        with self.engine.connect() as connection:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN")
            else:
                connection.begin()
            claim = connection.execute(select(self.table.c.record_envelope).where(
                self.table.c.record_envelope.is_not(None)).limit(1)).scalar_one_or_none()
            native = sql_state_header(connection, self.table.name)
        known = None if claim is None else RecordEnvelope.from_record(json.loads(claim)).store_id
        if native is not None:
            if known is not None and known != native.identity.id:
                raise ValueError("SQL record and journal ownership claims name different stores")
            known = native.identity.id
        return sql_identity(self.engine, self.table.name, existing_id=known)

    def __init__(self, conn_str: str, table: str = "rc_complexes", *, compression=None, native=None,
                 record_codecs=None):
        from sqlalchemy import (
            Boolean,
            Column,
            Float,
            Integer,
            LargeBinary,
            MetaData,
            String,
            Table,
            Text,
            create_engine,
        )
        from sqlalchemy.engine import make_url
        from sqlalchemy.pool import StaticPool
        from .store_identity import sql_identity_table
        from .header import BlobCodecSpec, record_codec_inventory
        inventory = record_codec_inventory(record_codecs)
        if compression is not None and not isinstance(compression, BlobCodecSpec):
            raise TypeError("SQL compression requires a declared BlobCodecSpec")
        if native is not None and type(native) is not bool:
            raise TypeError("SQL native mode requires a boolean or None")
        self._sa = __import__("sqlalchemy")
        self._native = None
        self._sql_opening = True
        self.conn_str = conn_str
        self._closed = False
        self._sql_connection = None
        self._sql_writing = False
        self._sql_rollback_only = False
        self._sql_events = []
        self._sql_schema_connection = None
        url = make_url(conn_str)
        options = {}
        if url.get_backend_name() == "sqlite" and url.database in {None, "", ":memory:"}:
            # A handle's RLock serializes use of the one private in memory database.
            # SQLite's default per thread pool would give each thread a different DB.
            options = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        self.engine = create_engine(conn_str, **options)
        self.identity_table = sql_identity_table(table)
        self.meta = MetaData()
        self.table = Table(
            table, self.meta,
            Column("id", String(256), primary_key=True),
            Column("version", Integer, primary_key=True, default=1),
            Column("signature", Text),
            Column("meta", Text),
            Column("created", Float),
            Column("blob", LargeBinary),
            Column("record_envelope", Text),
            Column("nV", Integer), Column("nE", Integer),
            Column("betti1", Integer), Column("kappa_mean", Float),
            Column("chain_valid", Boolean), Column("source", String(256)),
            Column("tx_from", Float), Column("tx_to", Float),
            Column("valid_from", Float), Column("valid_to", Float),
        )
        # Inverted index over the vocabulary. Retrieval's prefilter is "which records
        # share a token with this query", and answering that by reading every row back
        # into Python is the whole scale problem. One indexed row per (record, label)
        # lets the database answer it.
        self.labels_table = Table(
            f"{table}_labels", self.meta,
            Column("id", String(256), primary_key=True),
            Column("version", Integer, primary_key=True),
            Column("label", String(256), primary_key=True),
        )
        self.commits_table = Table(
            f"{table}_commits", self.meta,
            Column("id", String(256), primary_key=True),
            Column("version", Integer, primary_key=True),
            Column("artifact", LargeBinary),
        )
        try:
            with self._schema_transaction() as conn:
                from sqlalchemy import insert, inspect
                from .header import StoreHeader
                from .sql_engine import SQLRecordEngine
                from .sql_journal import sql_state_header
                from .store_identity import StoreIdentity
                self._sql_table_existed = inspect(conn).has_table(table)
                self._sql_owner_schema_existed = inspect(conn).has_table(self.identity_table.name)
                claimed = sql_state_header(conn, table)
                if claimed is not None:
                    if native is False:
                        raise ValueError("native SQL state cannot be downgraded; explicit migration required")
                    if compression is not None and compression != claimed.compression:
                        raise ValueError("SQL compression differs from persisted header; explicit migration required")
                    if record_codecs is not None and inventory != claimed.record_codecs:
                        raise ValueError("SQL record codecs differ from persisted header; explicit migration required")
                    if any(not inspect(conn).has_table(projection.name)
                            for projection in (self.table, self.labels_table, self.commits_table)):
                        raise ValueError("native SQL projection schema is missing; explicit recovery required")
                    # Verify claimed native metadata before any schema repair. Its
                    # owner was checked against the durable identity by discovery.
                    self._store_identity = claimed.identity
                    self._native = SQLRecordEngine(self, conn, claimed)
                self.meta.create_all(conn)
                self._migrate_index_columns(table)
                self._create_label_index(table)
                if (claimed is None and not self._sql_table_existed and not self._sql_owner_schema_existed
                        and native is not False and self.engine.dialect.name == "sqlite"):
                    # Schema, identity, header and genesis head are one publication.
                    # Another constructor must never observe a headerless interval.
                    self.identity_table.create(conn)
                    self._store_identity = StoreIdentity.create(self.backend)
                    conn.execute(insert(self.identity_table).values(key="identity", value=self._store_identity.to_bytes()))
                    header = StoreHeader(self._store_identity, compression or BlobCodecSpec(), inventory)
                    self._native = SQLRecordEngine(self, conn, header)
            if self._native is None:
                self._open_record_engine(compression=compression, native=native, record_codecs=record_codecs)
            self._sql_opening = False
        except BaseException:
            self._closed = True
            self.engine.dispose()
            raise

    def _open_record_engine(self, *, compression, native, record_codecs=None):
        from sqlalchemy import select
        from .header import BlobCodecSpec, StoreHeader, record_codec_inventory
        from .sql_engine import SQLRecordEngine
        from .sql_journal import sql_state_header
        from .store_identity import StoreIdentity
        with self._sql_transaction(write=True) as conn:
            header = sql_state_header(conn, self.table.name)
            if header is not None:
                if native is False:
                    raise ValueError("native SQL state cannot be downgraded; explicit migration required")
                if compression is not None and compression != header.compression:
                    raise ValueError("SQL compression differs from persisted header; explicit migration required")
                if record_codecs is not None and record_codec_inventory(record_codecs) != header.record_codecs:
                    raise ValueError("SQL record codecs differ from persisted header; explicit migration required")
            elif native is False:
                if compression is not None or record_codecs is not None:
                    raise ValueError("declared SQL compression requires native state")
                return
            else:
                populated = any(conn.execute(select(table.c.id).limit(1)).first() is not None
                                for table in (self.table, self.labels_table, self.commits_table))
                if populated or self._sql_table_existed or self._sql_owner_schema_existed:
                    if native is True or compression is not None or record_codecs is not None:
                        raise ValueError("existing headerless SQL store requires explicit native-state migration")
                    return
                if self.engine.dialect.name != "sqlite":
                    if native is True or compression is not None or record_codecs is not None:
                        raise ValueError("native SQL state currently requires the qualified SQLite provider")
                    return
                header = StoreHeader(StoreIdentity(self.store_id, self.backend), compression or BlobCodecSpec(),
                                     record_codec_inventory(record_codecs))
            self._native = SQLRecordEngine(self, conn, header)

    @property
    def header(self):
        return None if self._native is None else self._native.header

    @property
    def blob_codec(self):
        return None if self._native is None else self.header.compression

    def _serialize_payload(self, rex):
        if self._native is None:
            return super()._serialize_payload(rex)
        from .engine import encode_native_payload
        return self._encode_payload(encode_native_payload(rex, self.blob_codec), object_type="RCDB.Rex")

    @contextlib.contextmanager
    def _schema_transaction(self):
        """Keep SQLite DDL and legacy backfills in one actual DB transaction."""
        with self._transaction_lock:
            self._check_open()
            if self._sql_schema_connection is not None:
                yield self._sql_schema_connection
                return
            with self.engine.connect() as conn:
                if self.engine.dialect.name == "sqlite":
                    conn.exec_driver_sql("BEGIN IMMEDIATE")
                else:
                    conn.begin()
                self._sql_schema_connection = conn
                try:
                    yield conn
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
                finally:
                    self._sql_schema_connection = None

    def _check_open(self):
        if self._closed:
            raise RuntimeError("SQLStore is closed")

    @contextlib.contextmanager
    def _sql_transaction(self, *, write=False):
        """Own one connection from selection through publication or snapshot capture.

        SQLite writers reserve the database before checking versions; readers use
        an explicit BEGIN, including with Python's legacy sqlite transaction mode.
        Other SQL dialects serialize provider operations on the durable identity
        row. Their isolation, driver and failure behavior still require qualification.
        """
        from sqlalchemy import select, update
        from .store_identity import StoreIdentity, sql_identity_payload
        from .sql_journal import sql_state_header
        with self._transaction_lock:
            self._check_open()
            if self._publication_uncertain:
                raise PublicationUncertainError("RCDB publication is uncertain; reopen and verify before accessing SQL")
            if self._sql_connection is not None:
                if write and not self._sql_writing:
                    raise RuntimeError("cannot write inside a SQL read transaction")
                try:
                    yield self._sql_connection
                except BaseException:
                    if self._sql_writing:
                        self._sql_rollback_only = True
                    raise
                return
            # Creating the owner row with another connection during a SQLite
            # write transaction would deadlock against our own reservation.
            owner = self.store_id
            events = []
            with self.engine.connect() as conn:
                if self.engine.dialect.name == "sqlite":
                    conn.exec_driver_sql("BEGIN IMMEDIATE" if write else "BEGIN")
                else:
                    conn.begin()
                    # A no operation UPDATE is a portable write lock even for dialects
                    # where SELECT FOR UPDATE is ignored. It covers reads too so
                    # provider writers cannot alter a dependency closure midway.
                    conn.execute(update(self.identity_table).where(
                        self.identity_table.c.key == "identity").values(value=self.identity_table.c.value))
                self._sql_connection = conn
                self._sql_writing = write
                self._sql_rollback_only = False
                self._sql_events = events
                try:
                    payload = conn.execute(select(self.identity_table.c.value).where(
                        self.identity_table.c.key == "identity")).scalar_one_or_none()
                    if payload is None:
                        raise ValueError("store identity is missing from bound durable state; explicit recovery required")
                    if StoreIdentity.from_bytes(sql_identity_payload(payload), backend="sql").id != owner:
                        raise ValueError("store identity differs from this SQL handle's durable ownership claim")
                    native = sql_state_header(conn, self.table.name)
                    if native is not None and native.identity.id != owner:
                        raise ValueError("SQL record and journal ownership claims name different stores")
                    if self._native is not None:
                        self._native.refresh(conn)
                    elif native is not None and not self._sql_opening:
                        raise ValueError("SQL store acquired native state after this compatibility handle opened; reopen")
                    yield conn
                    if self._sql_rollback_only:
                        raise RuntimeError("SQL transaction was aborted by a nested operation")
                    try:
                        conn.commit()
                    except BaseException as exc:
                        if write:
                            self._publication_uncertain = True
                            raise PublicationUncertainError("SQL commit outcome is uncertain; reopen and verify") from exc
                        raise
                except BaseException as exc:
                    if self._native is not None:
                        self._native.discard()
                    try:
                        conn.rollback()
                    except BaseException as rollback:
                        if write:
                            self._publication_uncertain = True
                            raise PublicationUncertainError("SQL rollback failed; reopen and verify") from rollback
                        raise rollback from exc
                    raise
                finally:
                    self._sql_connection = None
                    self._sql_writing = False
                    self._sql_rollback_only = False
                    self._sql_events = []
                    if write:
                        self._corpus_cache = None
            for action, record_id, version, signature in events:
                super()._emit(action, record_id, version, signature)

    @contextlib.contextmanager
    def read_transaction(self):
        """Pin record, artifact and RCQL dependency reads to one SQL snapshot."""
        with self._sql_transaction():
            yield self

    @contextlib.contextmanager
    def write_scope(self):
        with self._sql_transaction(write=True):
            self._require_record_engine()
            yield self

    def put(self, *args, **kwargs):
        with self._sql_transaction(write=True):
            return super().put(*args, **kwargs)

    def put_prepared(self, *args, **kwargs):
        with self._sql_transaction(write=True):
            return super().put_prepared(*args, **kwargs)

    def commit_mutation(self, *args, **kwargs):
        with self._sql_transaction(write=True):
            return super().commit_mutation(*args, **kwargs)

    def read_record(self, *args, **kwargs):
        with self.read_transaction():
            return super().read_record(*args, **kwargs)

    def _record_at_version(self, id, version):
        with self._sql_transaction():
            if self._native is not None:
                return self._native.state.record(id, version)
            return super()._record_at_version(id, version)

    def state_manifest(self):
        with self.read_transaction():
            return super().state_manifest()

    def commit_history(self, id):
        with self.read_transaction():
            if self._native is not None:
                _read_selector(id, None, None, None)
                return [self._decode_commit(raw) for row in self._native.state.incarnation_history(id)
                        if (raw := self._load_commit_bytes(id, row.version)) is not None]
            return super().commit_history(id)

    def _commit_predecessor_reset(self, record):
        if self._native is None:
            return False
        change = self._native.state.change_for(record.id, record.version)
        return change is not None and change.previous_version == 0

    def verify_commits(self, id):
        with self.read_transaction():
            return super().verify_commits(id)

    def _emit(self, action, id, version, sig):
        if self._sql_connection is not None:
            self._sql_events.append((action, id, version, snapshot_mapping(sig)))
        else:
            super()._emit(action, id, version, sig)

    def _store_commit_bytes(self, id, version, blob):
        from sqlalchemy import insert, select
        with self._sql_transaction(write=True) as conn:
            if conn.execute(select(self.table.c.id).where(
                self.table.c.id == str(id), self.table.c.version == int(version))).first() is not None:
                raise ValueError("cannot overwrite a published SQL mutation artifact")
            prior = conn.execute(select(self.commits_table.c.artifact).where(
                self.commits_table.c.id == str(id), self.commits_table.c.version == int(version))).scalar_one_or_none()
            if prior is not None:
                if bytes(prior) != bytes(blob):
                    raise ValueError("SQL mutation artifact address contains different bytes")
                return
            conn.execute(insert(self.commits_table).values(
                id=str(id), version=int(version), artifact=bytes(blob)))

    def _load_commit_bytes(self, id, version):
        from sqlalchemy import select
        with self._sql_transaction() as conn:
            row = conn.execute(select(self.commits_table.c.artifact).where(
                self.commits_table.c.id == str(id),
                self.commits_table.c.version == int(version))).first()
            if self._native is not None:
                return self._native.commit_bytes(id, version, None if row is None else row[0])
        return None if row is None else bytes(row[0])

    def _delete_commit_bytes(self, id, version):
        from sqlalchemy import delete, select
        with self._sql_transaction(write=True) as conn:
            if self._sql_rollback_only:
                # RCStore's compatibility cleanup runs after a failed nested put.
                # Its provisional row may still be visible on this connection;
                # the outer rollback discards both row and artifact together.
                return
            if conn.execute(select(self.table.c.id).where(
                self.table.c.id == str(id), self.table.c.version == int(version))).first() is not None:
                raise ValueError("cannot delete a published SQL mutation artifact")
            conn.execute(delete(self.commits_table).where(
                self.commits_table.c.id == str(id),
                self.commits_table.c.version == int(version)))

    def _create_label_index(self, table):
        """The label index is part of the declared SQL schema, with literal names."""
        from sqlalchemy import Index
        with self._schema_transaction() as conn:
            Index(f"ix_{table}_labels_label", self.labels_table.c.label).create(conn, checkfirst=True)

    def _migrate_index_columns(self, table):
        """Add indexed columns to a pre existing table and backfill from the
        stored signature JSON. Also ALTER ADDs the five bitemporal columns
        onto a pre Slice-C table and backfills legacy rows to version 1
        (open, tx_from/valid_from = created), then repairs a legacy id only
        primary key to the composite (id, version) key the append only
        design requires, and finally (re)creates the indexes on the
        promoted columns. Idempotent: re opening an already migrated table
        is a no op."""
        from sqlalchemy import inspect, select, text, update
        with self._schema_transaction() as conn:
            quote = self.engine.dialect.identifier_preparer.quote
            named = quote(table)
            have = {c["name"] for c in inspect(conn).get_columns(table)}
            missing = [c for c in self._INDEX_COLS if c not in have]
            if "record_envelope" not in have:
                conn.execute(text(f'ALTER TABLE {named} ADD COLUMN record_envelope TEXT'))
            for col in missing:
                conn.execute(text(f'ALTER TABLE {named} ADD COLUMN {quote(col)} {self._INDEX_COLS[col]}'))
            if missing:
                rows = conn.execute(select(self.table.c.id, self.table.c.signature)).fetchall()
                for rid, sigjson in rows:
                    vals = _sig_index_values(loads_mapping(sigjson or "{}"))
                    conn.execute(update(self.table).where(self.table.c.id == rid).values(**vals))
            have_t = {c["name"] for c in inspect(conn).get_columns(table)}
            missing_t = [c for c in self._TEMPORAL_COLS if c not in have_t]
            for col in missing_t:
                conn.execute(text(f'ALTER TABLE {named} ADD COLUMN {quote(col)} {self._TEMPORAL_COLS[col]}'))
            if missing_t:
                conn.execute(update(self.table).where(self.table.c.version.is_(None)).values(
                    version=1, tx_from=self.table.c.created, valid_from=self.table.c.created))
            self._ensure_composite_pk(table)
            self._create_promoted_indexes(table)

    def _create_promoted_indexes(self, table):
        """(Re)create the indexes on the promoted signature columns and the
        bitemporal lookup columns. Each CREATE INDEX is guarded by an
        existing index check, so this is safe to call repeatedly: a table
        that already has the indexes is a no op, and a table that just lost
        them (a primary key rebuild drops indexes along with the table)
        gets them rebuilt."""
        from sqlalchemy import Index
        with self._schema_transaction() as conn:
            for suffix, columns in (("betti1", ("betti1",)), ("kappa_mean", ("kappa_mean",)),
                                    ("source", ("source",)), ("id_txto", ("id", "tx_to")),
                                    ("id_validfrom", ("id", "valid_from"))):
                Index(f"ix_{table}_{suffix}", *(self.table.c[col] for col in columns)).create(conn, checkfirst=True)

    def _ensure_composite_pk(self, table):
        """Repair a legacy id only primary key to the composite (id, version)
        key the append only versioned schema requires. A table freshly
        created by this class already has the composite key (it is declared
        directly on self.table), so this only ever fires against a
        pre Slice-C database. Idempotent: a table already keyed on
        (id, version) is left untouched. An unrecognized primary key
        requires explicit migration rather than an inferred rewrite.

        SQLite cannot ALTER a primary key in place, so the table is rebuilt
        under a temporary name with the composite key declared, the data is
        copied over column by column, and the temporary table is swapped in
        for the original. Other dialects can ALTER the constraint directly."""
        from sqlalchemy import MetaData, inspect, text
        from uuid import uuid4
        with self._schema_transaction() as conn:
            insp = inspect(conn)
            quote = self.engine.dialect.identifier_preparer.quote
            named = quote(table)
            pk = insp.get_pk_constraint(table).get("constrained_columns") or []
            if sorted(pk) == ["id", "version"]:
                return
            if pk != ["id"]:
                raise ValueError("SQL record primary key is incompatible; explicit migration required")
            dialect = self.engine.dialect.name
            if dialect == "sqlite":
                cols = [c["name"] for c in insp.get_columns(table)]
                collist = ", ".join(quote(col) for col in cols)
                # Never drop an occupied name to make room for a migration.
                tmp = "rcdb_pkmig_"+uuid4().hex
                temporary = quote(tmp)
                new_table = self.table.to_metadata(MetaData(), name=tmp)
                new_table.create(conn)
                conn.execute(text(
                    f"INSERT INTO {temporary} ({collist}) SELECT {collist} FROM {named}"))
                conn.execute(text(f"DROP TABLE {named}"))
                conn.execute(text(f"ALTER TABLE {temporary} RENAME TO {named}"))
            elif dialect in {"mysql", "mariadb"}:
                conn.execute(text(f"ALTER TABLE {named} DROP PRIMARY KEY"))
                conn.execute(text(f"ALTER TABLE {named} ADD PRIMARY KEY (id, version)"))
            else:
                pk_name = insp.get_pk_constraint(table).get("name") or f"{table}_pkey"
                conn.execute(text(f"ALTER TABLE {named} DROP CONSTRAINT IF EXISTS {quote(pk_name)}"))
                conn.execute(text(f"ALTER TABLE {named} ADD PRIMARY KEY (id, version)"))

    def next_version(self, id):
        from sqlalchemy import func, select
        with self._sql_transaction() as conn:
            if self._native is not None:
                return self._native.state.next_version(id)
            return int(conn.execute(select(func.coalesce(func.max(self.table.c.version), 0)+1)
                                    .where(self.table.c.id == id)).scalar_one())

    def _put_impl(self, id, rex, sig, meta, tags, valid_from, valid_to,
                  search_terms=None, tx_time=None):
        if self._native is not None:
            with self._sql_transaction(write=True) as conn:
                return self._native.put(conn, id, rex, sig, meta or {}, valid_from, valid_to, tx_time)
        from sqlalchemy import insert, update
        from .engine import payload_identity
        now = _now() if tx_time is None else float(tx_time)
        payload = self._decode_payload(self._serialize_payload(rex))
        identity = payload_identity(payload)
        with self._sql_transaction(write=True) as conn:
            conn.execute(update(self.table).where(
                self.table.c.id == id, self.table.c.tx_to.is_(None)).values(tx_to=now))
            v = self.next_version(id)
            vfrom = valid_from if valid_from is not None else now
            rec = ComplexRecord(id=id, signature=sig, created=now, meta=meta or {}, version=v,
                                tx_from=now, tx_to=None, valid_from=vfrom, valid_to=valid_to)
            rec, blob = self._record_payload(rec, payload=payload, identity=identity)
            values = dict(id=id, signature=dumps_mapping(sig), meta=dumps_mapping(meta or {}),
                          created=now, blob=blob, version=v, tx_from=now, tx_to=None,
                          valid_from=vfrom, valid_to=valid_to,
                          record_envelope=json.dumps(rec.envelope.as_record()), **_sig_index_values(sig))
            conn.execute(insert(self.table).values(**values))
            labels = sorted(_record_labels(sig, meta))
            if labels:
                conn.execute(insert(self.labels_table),
                             [{"id": id, "version": v, "label": lab} for lab in labels])
        return rec

    def _row_to_record(self, row) -> ComplexRecord:
        created = row.created or 0.0
        return ComplexRecord(id=row.id, signature=loads_mapping(row.signature or "{}"),
                             created=created, meta=loads_mapping(row.meta or "{}"),
                             version=row.version if row.version is not None else 1,
                             tx_from=row.tx_from if row.tx_from is not None else created,
                             tx_to=row.tx_to, valid_from=row.valid_from, valid_to=row.valid_to,
                             envelope=None if row.record_envelope is None else RecordEnvelope.from_record(json.loads(row.record_envelope)))

    _RECORD_COLS = ("id", "signature", "meta", "created", "version",
                    "tx_from", "tx_to", "valid_from", "valid_to", "record_envelope")

    def _record_cols(self):
        t = self.table
        return [getattr(t.c, name) for name in self._RECORD_COLS]

    def _records_for(self, id):
        from sqlalchemy import select
        with self._sql_transaction() as conn:
            rows = conn.execute(select(*self._record_cols())
                                .where(self.table.c.id == id)).fetchall()
        return [self._row_to_record(r) for r in rows]

    @_owned_records
    def get_record(self, id, *, as_of=None, valid_at=None):
        with self._sql_transaction():
            if self._native is not None:
                return self._native.selected(id, as_of, valid_at)
            literal_versions = self._records_for(id)
            rec = self._select_version(literal_versions, as_of, valid_at)
            if rec is None and not literal_versions:
                split = self._split_versioned_id(id)
                if split is not None:
                    base, v = split
                    # a lineage() display id: resolve the explicit version directly
                    # (version is explicit, so as_of/valid_at do not apply)
                    rec = next((r for r in self._records_for(base) if r.version == v), None)
            return rec

    def get(self, id, *, as_of=None, valid_at=None, verify: bool = True):
        from sqlalchemy import select
        with self._sql_transaction() as conn:
            rec = self.get_record(id, as_of=as_of, valid_at=valid_at)
            if rec is None:
                return None
            if self._native is not None:
                return self._native.read(conn, rec, verify=verify)
            # rec.id is the literal stored id, including display alias fallback.
            row = conn.execute(select(self.table.c.blob).where(
                self.table.c.id == rec.id, self.table.c.version == rec.version)).first()
            if row is None or row.blob is None:
                raise ValueError("published SQL record payload is missing")
            return self._read_record_payload(rec, row.blob, verify=verify)

    def get_version(self, id, version):
        from sqlalchemy import select
        with self._sql_transaction() as conn:
            if self._native is not None:
                rec = self._native.state.record(id, version)
                return None if rec is None else self._native.read(conn, rec)
            row = conn.execute(select(self.table).where(
                self.table.c.id == id, self.table.c.version == version)).first()
            if row is not None and row.blob is None:
                raise ValueError("published SQL record payload is missing")
            return self._read_record_payload(self._row_to_record(row), row.blob) if row else None

    @_owned_records
    def history(self, id):
        from sqlalchemy import select
        with self._sql_transaction() as conn:
            if self._native is not None:
                _read_selector(id, None, None, None)
                return self._native.state.history(id)
            rows = conn.execute(select(*self._record_cols())
                                .where(self.table.c.id == id)
                                .order_by(self.table.c.version)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _temporal_conds(self, as_of, valid_at, include_history):
        """Row selection for a point in transaction and/or validity time.

        A version is current AT a time when it opened on or before it and had not yet
        been closed. With neither given this is the open row, which is what the store
        always did; the difference is that the caller can now name a different one.
        """
        from sqlalchemy import or_
        t = self.table
        conds = []
        if as_of is not None:
            conds.append(t.c.tx_from <= as_of)
            conds.append(or_(t.c.tx_to.is_(None), t.c.tx_to > as_of))
        if valid_at is not None:
            conds.append(or_(t.c.valid_from.is_(None), t.c.valid_from <= valid_at))
            conds.append(or_(t.c.valid_to.is_(None), t.c.valid_to > valid_at))
        if not conds and not include_history:
            conds.append(t.c.tx_to.is_(None))
        return conds

    def _collection_conds(self, as_of, valid_at, include_history):
        """Choose one native version per id before applying query predicates."""
        conds = self._temporal_conds(as_of, valid_at, include_history)
        if self._native is None:
            return conds
        _read_selector("selector", None, as_of, valid_at)
        if type(include_history) is not bool:
            raise TypeError("include_history must be a bool")
        if include_history:
            if as_of is not None or valid_at is not None:
                raise ValueError("history collection cannot combine time selectors")
            return conds
        from sqlalchemy import and_, func, select
        t = self.table
        selected = select(t.c.id, func.max(t.c.version).label("version"))
        if conds:
            selected = selected.where(and_(*conds))
        selected = selected.group_by(t.c.id).subquery()
        return [select(selected.c.id).where(selected.c.id == t.c.id,
                                            selected.c.version == t.c.version).exists()]

    def _canonical_time_selection(self, as_of, valid_at):
        """SQL Float parameters must not change the point selector's comparison.

        Fractions, extended/custom reals and integers rounded by binary64 use
        checked engine selection. Native floats and exactly represented integers
        retain SQL predicates and indexes.
        """
        return self._native is not None and any(
            value is not None and (type(value) not in (int, float)
                                   or (type(value) is int and float(value) != value))
            for value in (as_of, valid_at))

    def _canonical_time_page(self, *, limit, offset=0, as_of=None, valid_at=None,
                             include_history=False, predicate=None):
        with self._sql_transaction() as conn:
            if self._sql_writing:
                self._native.check_projection(conn)
            return self._native.state.select_records(limit=limit, offset=offset, as_of=as_of,
                    valid_at=valid_at, include_history=include_history, predicate=predicate)

    @_owned_records
    def list(self, limit=100, offset=0, *, as_of=None, valid_at=None,
             include_history=False):
        from sqlalchemy import and_, select
        if self._native is not None and (type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0):
            raise ValueError("SQL collection bounds require nonnegative native integers")
        if self._canonical_time_selection(as_of, valid_at):
            return self._canonical_time_page(limit=limit, offset=offset, as_of=as_of,
                                             valid_at=valid_at, include_history=include_history)
        t = self.table
        stmt = select(*self._record_cols())
        conds = self._collection_conds(as_of, valid_at, include_history)
        if conds:
            stmt = stmt.where(and_(*conds))
        order = (t.c.tx_from.desc(), t.c.id, t.c.version.desc()) if self._native is not None else (t.c.tx_from.desc(),)
        stmt = stmt.order_by(*order).limit(limit).offset(offset)
        with self._sql_transaction() as conn:
            if self._native is not None and self._sql_writing:
                self._native.check_projection(conn)
            rows = conn.execute(stmt).fetchall()
        return [self._row_to_record(r) for r in rows]

    @_owned_records
    def query(self, limit=100, include_history=False, *, as_of=None,
              valid_at=None, **predicate):
        # push the indexed predicates into SQL; apply the rest (tags, voids)
        # in Python only over the narrowed candidate set.
        from sqlalchemy import and_, select
        if self._native is not None:
            if type(limit) is not int or limit < 0:
                raise ValueError("SQL query limit requires a nonnegative native integer")
            if set(predicate)-_QUERY_KEYS:
                raise TypeError("unsupported query keys: "+", ".join(sorted(set(predicate)-_QUERY_KEYS))
                                +". Supported: "+", ".join(sorted(_QUERY_KEYS)))
            if self._canonical_time_selection(as_of, valid_at):
                return self._canonical_time_page(limit=limit, as_of=as_of, valid_at=valid_at,
                                                include_history=include_history, predicate=predicate)
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
            value = predicate.get(key)
            if value is not None:
                if key == "source" and not isinstance(value, str):
                    continue
                conds.append(build(value))
                # Text indexes narrow candidates. The signature retains the type.
                if key != "source":
                    pushed.add(key)
        # the vocabulary predicate resolves in the indexed label table, so a
        # "which records share a token" prefilter never leaves the database.
        lt = self.labels_table
        for key, op in (("labels_any", "any"), ("labels_all", "all")):
            vals = predicate.get(key)
            if not vals:
                continue
            wanted = sorted({str(x).lower() for x in vals})
            sub = (select(lt.c.id)
                   .where(and_(lt.c.id == t.c.id, lt.c.version == t.c.version,
                               lt.c.label.in_(wanted))))
            if op == "all":
                from sqlalchemy import func
                sub = (sub.group_by(lt.c.id)
                          .having(func.count(func.distinct(lt.c.label)) == len(wanted)))
            conds.append(sub.exists())
            pushed.add(key)
        conds.extend(self._collection_conds(as_of, valid_at, include_history))
        stmt = select(*self._record_cols())
        if conds:
            stmt = stmt.where(and_(*conds))
        order = (t.c.tx_from.desc(), t.c.id, t.c.version.desc()) if self._native is not None else (t.c.created.desc(),)
        stmt = stmt.order_by(*order)
        residual = {k: v for k, v in predicate.items() if k not in pushed}
        if not residual:
            # nothing left for Python to reject, so stop the database at `limit`
            # instead of materializing every match and slicing afterwards.
            stmt = stmt.limit(limit)
        with self._sql_transaction() as conn:
            if self._native is not None:
                if self._sql_writing:
                    self._native.check_projection(conn)
                if not limit:
                    return []
                out = []
                # Residual predicates may reject early SQL candidates. Stream
                # until enough complete matches survive, then close the cursor
                # before releasing this same checked provider snapshot.
                with conn.execute(stmt) as rows:
                    for row in rows:
                        record = self._row_to_record(row)
                        if residual and not _matches(record.signature, residual, record.meta,
                                is_complex=record.is_complex, record_type=record.object_type):
                            continue
                        out.append(record)
                        if len(out) == limit:
                            break
                return out
            rows = conn.execute(stmt).fetchall()
        out = [self._row_to_record(r) for r in rows]
        if residual:
            out = [r for r in out if _matches(r.signature, residual, r.meta,
                                            is_complex=r.is_complex, record_type=r.object_type)]
        return out[:limit]

    @_serialized
    def delete(self, id, *, tx_time=None, expected_version=None):
        from sqlalchemy import delete, select
        with self._sql_transaction(write=True) as conn:
            if self._native is not None:
                return self._native.delete(conn, id, tx_time=tx_time, expected_version=expected_version)
            if tx_time is not None or expected_version is not None:
                raise ValueError("native deletion selectors require explicit migration of this SQL store")
            reclaim = self._claim_delete(id)
            existed = conn.execute(select(self.table.c.id).where(
                self.table.c.id == id)).first() is not None
            conn.execute(delete(self.table).where(self.table.c.id == id))
            # the label index is a projection of the record; it must not outlive it
            conn.execute(delete(self.labels_table).where(self.labels_table.c.id == id))
            if existed:
                self._reclaim_commits(id, reclaim)
                self._emit("rcdb.delete", id, 0, {})
            return existed

    def _require_record_engine(self):
        if self._native is None:
            raise ValueError("SQL change history requires explicit native-state migration")
        return self._native.state

    @property
    def change_cursor(self):
        with self.read_transaction():
            return self._require_record_engine().cursor

    def changes(self, after=None, *, limit=100):
        with self.read_transaction():
            return self._require_record_engine().changes(after, limit=limit)

    def cursor_for(self, change):
        with self.read_transaction():
            return self._require_record_engine().cursor_for(change)

    def tombstone(self, id):
        _read_selector(id, None, None, None)
        with self.read_transaction():
            return self._require_record_engine().tombstone(id)

    def close(self):
        """Idempotently close this handle and release its connection pool."""
        with self._transaction_lock:
            if self._closed:
                return
            if self._sql_connection is not None:
                raise RuntimeError("cannot close SQLStore inside an active transaction")
            if self._native is not None:
                self._native.discard()
            self._closed = True
            self.engine.dispose()



# backend registry + URI opener

from rexgraph.registry import Registry

_BACKENDS = Registry("rcdb backend")


def register_backend(scheme: str, factory: Callable[[str], RCStore]) -> None:
    """Register a backend factory for a URI scheme (e.g. 'redis')."""
    _BACKENDS.register(scheme, factory)


def unregister_backend(scheme: str):
    """Remove a backend factory. Returns it, or None if it was not registered."""
    return _BACKENDS.unregister(scheme)


def available_backends() -> list[str]:
    """Every registered URI scheme."""
    return _BACKENDS.available()


def _labels_of(rec: ComplexRecord, rex) -> list:
    """Best effort vertex labels for a record (from meta, else indices)."""
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
    # last resort fallback for a backend that hasn't implemented get_version:
    # not safe under same tick collisions, only reached for an unknown type.
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
    cross_complex_bridge kappa correlation score `compare` returns, rescaled
    to [0, 1]. Labels are plain vertex index labels (no per version meta
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
    field: per version structural signature, and per step the signed change
    in each structural quantity (existence/direction over time) plus the
    relational match (cross_complex_bridge similarity) between consecutive
    versions (how close, and moving toward/away)."""
    hist = store.history(id)
    versions = []
    rexes = []
    for r in hist:
        rex = current_rex(store.get(id, as_of=None) if r.version == hist[-1].version
                          else _get_ver(store, id, r.version))
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
        ra = current_rex(_get_ver(store, id_a, ha[i].version))
        rb = current_rex(_get_ver(store, id_b, hb[i].version))
        series.append(_pair_match(ra, rb))
    steps = [{"step": i, "match": series[i],
              "direction": ("toward" if series[i] > series[i - 1] else
                            "away" if series[i] < series[i - 1] else "level")}
             for i in range(1, n)]
    return {"a": id_a, "b": id_b, "match_series": series, "steps": steps,
            "net": (series[-1] - series[0]) if series else 0.0}


def current_rex(obj):
    """The RexGraph an analysis should read from a stored object.

    A lineage recorded over time is stored as a TemporalRex, so the object a
    structural read gets back is the whole history rather than a complex. Every
    analytic here works on one complex, and the one it means is the latest state.
    Anything that is already a RexGraph passes through, so callers do not branch.
    """
    from rexgraph.graph import TemporalRex
    if isinstance(obj, TemporalRex):
        if obj.T <= 0:
            return None
        return obj.reconstruct_at(int(obj.T) - 1)
    return obj


def find_similar(store: RCStore, query_rex, query_labels, top_k: int = 10,
                 exclude_id: str = None):
    """Rank stored complexes by structural similarity to a query complex.

    Scores through `rcdb.analytics.interfacing_score` by default, or through the
    similarity hook when an application injected one. It reads the query's
    footprint under each candidate's own coherence field by demand driven
    diffusion. Returns ``{id, match, score, shared, context_size, tags, source}``
    sorted by match descending, where ``match`` is a 0-1 number a UI can show as
    a percentage.
    """
    from .analytics import interfacing_score
    score_fn = _SIMILARITY_HOOK or interfacing_score
    qset = {str(x).lower() for x in (query_labels or [])}
    out = []
    for rec in store.list(limit=10 ** 9):
        if not rec.is_complex:
            continue
        if exclude_id is not None and rec.id == exclude_id:
            continue
        try:
            # lossless pre filter: a record with no shared labels contributes
            # nothing (bridge n_shared=0), so skip the expensive deserialize.
            meta_labels = (rec.meta or {}).get("vertex_labels")
            if meta_labels is not None and qset and not (qset & set(meta_labels)):
                continue
            cand = current_rex(store.get(rec.id))
            if cand is None:
                continue
            cand_labels = _labels_of(rec, cand)
            r = score_fn(cand, cand_labels, query_labels)
            if r["n_shared"] == 0:
                continue
            # `match` is documented as a 0-1 number a UI shows as a percentage, but
            # ||iv|| is unbounded. s/(1+s) is monotone, so the ranking is the
            # scorer's ranking exactly, and bounded, so the percentage means
            # something. Same map the retrieval path uses for the same reason.
            s_raw = r["score"]
            match = s_raw / (1.0 + s_raw) if s_raw > 0 else 0.0
            out.append({
                "id": rec.id,
                "match": round(match, 4),
                "score": round(s_raw, 6),
                "kappa_mean": round(r["kappa_mean"], 4),
                "context_size": r["context_size"],
                "shared": r["n_shared"],
                "tags": rec.signature.get("tags", []),
                "source": rec.signature.get("source", ""),
            })
        except (KeyError, TypeError) as e:
            # a missing key here is a contract break between this and the scorer,
            # not a bad record: swallowing it silently returns an empty ranking
            # and looks like "nothing matched".
            raise RuntimeError(f"find_similar: scorer contract changed ({e})") from e
        except Exception:
            continue
    out.sort(key=lambda r: (-r["match"], str(r["id"])))
    return out[:top_k]


def version_if_changed(store: RCStore, lineage_id: str, rex, meta=None, tags=None,
                       *, valid_from=None):
    """Store a new version only if the schema actually changed vs the latest
    (different tables or different topology). Enables auto lineage on repeated
    reflection without spamming identical versions. Returns version info with
    an ``unchanged`` flag.

    Compares against ``store.get(lineage_id)`` (the current version) directly,
    over the native version chain (no scan over other lineages)."""
    latest_rex = current_rex(store.get(lineage_id))
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
    info = put_version(store, lineage_id, rex, meta=meta, tags=tags,
                       valid_from=valid_from)
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
    """Old scheme fallback: under the legacy scheme, each version was a SEPARATE record
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
    ``_get_ver``, not a same tick ``tx_from`` that could misresolve across
    versions written in the same instant). For a legacy store with no native
    chain under this id, each version is instead reconstructed through its
    own display/real id, so the ``trajectory`` diff still populates for
    legacy data (``trajectory_steps`` stays history based, so it is ``[]``
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
        for v_a, v_b, rec_a, rec_b in zip(versions, versions[1:], hist, hist[1:], strict=False):
            try:
                rex_a = current_rex(_get_ver(store, lineage_id, rec_a.version))
                rex_b = current_rex(_get_ver(store, lineage_id, rec_b.version))
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
        for v_a, v_b in zip(versions, versions[1:], strict=False):
            try:
                rex_a = current_rex(store.get(v_a["id"]))
                rex_b = current_rex(store.get(v_b["id"]))
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
    """Group stored complexes into structural families by cross complex
    coherence (the crossing tensor). Builds the pairwise coherence matrix,
    then takes connected components at ``threshold``. Returns
    ``{clusters:[{members, avg_coherence, centroid, tags}], singletons, n}``.
    """
    import math

    from rexgraph.graph import cross_complex_bridge
    recs = [record for record in store.list(limit=10 ** 9) if record.is_complex]
    if tags_any:
        tset = set(tags_any)
        recs = [r for r in recs if tset & set(r.signature.get("tags", []))]
    items = []
    for r in recs:
        try:
            rex = current_rex(store.get(r.id))
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
    rex_a, rex_b = current_rex(store.get(id_a)), current_rex(store.get(id_b))
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


def copy_record(src: RCStore, dst: RCStore, record, *, meta=None, tags=None,
                governed=None, actor="", expected_version=None, return_receipt=False,
                tx_time=None, expected_digest=None, expected_cursor=None,
                expected_policy_digest=None, preserve_signature=False, destination_id=None):
    """Copy one stored version of one record from `src` into `dst`.

    The single place a record crosses between stores. `migrate` walks a whole history
    through here and a courier carries one current version through here, so the two
    differ in WHICH versions they select and in nothing else. That matters because what
    has to travel with a complex is easy to forget one carrier at a time: the meta the
    record was stored under, the tags that keep it queryable, and the valid time it was
    true for, which is not the time it is being copied at.

    `record` is a ComplexRecord from `src`, current or historical. `meta` and `tags`
    override what it carried, which is how a caller stamps provenance without having to
    reassemble the rest. Returns the destination record, or None when the source cannot
    produce the complex.

    Read the requested published version through `read_record`, never pair its
    metadata with a newer payload. `governed=None` adopts the destination's
    require_commits setting; True always makes a new destination mutation commit.
    Source commit packages are not transplanted into a different lineage. False
    uses put and cannot bypass a required commit destination. actor and
    expected_version apply only to governed copies. Payload integrity failures raise.
    With return_receipt=True, return a closed CopyReceipt naming the actual source
    and published destination versions instead of the destination ComplexRecord.
    It records Core object identities, not physical container or commit digests.
    Checked migration can supply tx_time, expected_digest, expected_cursor and
    expected_policy_digest. Source selection finishes before the destination writer
    is acquired; cursor/policy conditions are checked under that writer. A
    preserve_signature copy refuses metadata/tag overrides or lossy projection.
    destination_id optionally names a different literal destination address; the
    receipt continues to name the original source address and version.
    """
    if governed is not None and not isinstance(governed, bool):
        raise TypeError("governed must be a bool or None")
    if type(return_receipt) is not bool:
        raise TypeError("return_receipt must be a bool")
    if type(preserve_signature) is not bool:
        raise TypeError("preserve_signature must be a bool")
    _read_selector(record.id, None, tx_time, None)
    target_id = record.id if destination_id is None else destination_id
    _read_selector(target_id, None, None, None)
    if expected_digest is not None:
        from .envelope import _hex_identity
        _hex_identity(expected_digest, 64, "expected source object digest")
    if expected_policy_digest is not None:
        from .envelope import _hex_identity
        _hex_identity(expected_policy_digest, 64, "expected destination policy digest")
    if expected_cursor is not None:
        from .engine import ChangeCursor
        if not isinstance(expected_cursor, ChangeCursor):
            raise TypeError("conditional copy requires a declared ChangeCursor")
    if preserve_signature and (meta is not None or tags is not None):
        raise ValueError("signature-preserving copy cannot override source metadata or tags")
    snapshot = src.read_record(record.id, version=record.version)
    if snapshot is None:
        return None
    if expected_digest is not None and snapshot.state_digest != expected_digest:
        raise ValueError("copied source object differs from its pinned digest")
    source_owner = src.store_id if return_receipt else None
    record = snapshot.record
    options = {
        "meta": deepcopy(record.meta or {}) if meta is None else meta,
        "tags": list((record.signature or {}).get("tags", [])) if tags is None else tags,
        "valid_from": record.valid_from, "valid_to": record.valid_to,
    }
    # Source selection is complete before acquiring a destination publication gate.
    # This avoids nested directory locks or a source SQL snapshot blocking COMMIT.
    scope = dst._transaction_lock if expected_cursor is None else dst.write_scope()
    with scope:
        if expected_cursor is not None and dst.change_cursor != expected_cursor:
            raise VersionConflictError("destination cursor changed before conditional copy")
        if expected_policy_digest is not None and dst.transfer_policy_digest() != expected_policy_digest:
            raise ValueError("destination policy differs from the conditional copy")
        governed = bool(getattr(dst, "_require_commits", False)) if governed is None else governed
        if not governed and (expected_version is not None or actor):
            raise ValueError("actor and expected_version require a governed copy")
        if preserve_signature:
            from rexgraph.value_codec import pack_value
            if (pack_value(snapshot_mapping(dst._stored_meta(_priv(record.meta)))) != pack_value(record.meta)
                    or pack_value(snapshot_mapping(dst._stored_signature(record.signature))) != pack_value(record.signature)):
                raise ValueError("destination policy would change the source metadata or signature")
        if preserve_signature and dst._load_commit_bytes(target_id, dst.next_version(target_id)) is not None:
            raise ValueError("destination record address has an unpublished mutation artifact")
        if record.envelope is not None and record.envelope.codec != "rexgraph.safetensors":
            if governed:
                raise ValueError("current mutation commits cannot govern generic native copies")
            from .header import CodecRef
            copied = dst.put_record(target_id, snapshot.value,
                                    codec=CodecRef(record.envelope.codec, record.envelope.codec_version),
                                    signature=record.signature if preserve_signature else None,
                                    tx_time=tx_time, **options)
        elif governed:
            copied = dst.commit_mutation(target_id, snapshot.value, actor=actor,
                                         expected_version=expected_version, tx_time=tx_time,
                                         signature=record.signature if preserve_signature else None, **options)
        elif preserve_signature:
            from .engine import encode_native_payload
            from .header import BlobCodecSpec
            copied = dst.put_prepared(target_id, encode_native_payload(snapshot.value, BlobCodecSpec()),
                                      record.signature, _tx_time=tx_time, **options)
        else:
            copied = dst.put(target_id, snapshot.value, _tx_time=tx_time, **options)
    if not return_receipt:
        return copied
    from .transfer import CopyReceipt
    if copied.envelope is None:
        raise ValueError("published destination has no record binding for a copy receipt")
    copied.envelope.check_address(store_id=dst.store_id, record_id=copied.id, record_version=copied.version)
    return CopyReceipt(source_owner, record.id, int(record.version), snapshot.state_digest,
                       dst.store_id, copied.id, int(copied.version), copied.envelope.object_digest)


def migrate(src: RCStore, dst: RCStore, *, ids=None, limit: int = 10 ** 9) -> dict:
    """Copy records from one store into another, every version, oldest first.

    Written against the RCStore contract alone, so it works for any pair of backends
    without either knowing the other exists, which is what makes the choice of
    backend reversible rather than a commitment. Existing records in `dst` are left
    alone; a colliding id gains versions rather than losing its own.
    Each successful version has a closed identity receipt in the report. This
    compatibility history copy selects current ids by default and assigns new
    transaction times; it does not replay native tombstones or whole store clocks.
    Legacy callers should use plan_legacy_migration/migrate_legacy_batch for
    pinned inventory, explicit loss acceptance and verified resume.
    """
    if src.store_id == dst.store_id:
        raise ValueError("migration source and destination name the same logical store")
    wanted = list(ids) if ids is not None else [r.id for r in src.list(limit=limit)]
    n_records = n_versions = 0
    receipts = []
    for rid in wanted:
        history = src.history(rid)
        if not history:
            continue
        n_records += 1
        for rec in sorted(history, key=lambda r: r.version):
            receipt = copy_record(src, dst, rec, return_receipt=True)
            if receipt is not None:
                n_versions += 1
                receipts.append(receipt.as_record())
    from .legacy_migration import LEGACY_MIGRATION_LIMITATIONS
    return {"records": n_records, "versions": n_versions,
            "src": getattr(src, "backend", "?"), "dst": getattr(dst, "backend", "?"),
            "receipts": tuple(receipts), "scope": "compatibility_history_copy",
            "limitations": LEGACY_MIGRATION_LIMITATIONS}


def _existing_backend(path: str):
    """Which backend already owns `path`, if any. Choosing must never orphan data:
    a directory written by one store has to reopen as that store."""
    import os
    if not os.path.isdir(path):
        return None
    candidates = []
    native_object = os.path.lexists(os.path.join(path, "store.head"))
    header_path = os.path.join(path, "store.header")
    if os.path.lexists(header_path):
        from .header import HEADER_LIMIT, StoreHeader
        if os.path.islink(header_path) or not os.path.isfile(header_path):
            raise ValueError("store header requires a regular file")
        with open(header_path, "rb") as stream:
            header = StoreHeader.from_bytes(stream.read(HEADER_LIMIT+65))
        native_object = native_object or header.identity.backend == "object"
    if native_object:
        candidates.append("object")
    markers = {"local": ("store.header", "records.journal"),
               "rex": ("records.log", "blobs.pack"),
               "file": ("index.rexidx", "index.rexlog", "index.json", "index.log")}
    if native_object:
        markers["local"] = ("records.journal",)
    manifest_path = os.path.join(path, "MANIFEST.json")
    if os.path.lexists(manifest_path):
        from .legacy import loads_metadata
        if os.path.islink(manifest_path) or not os.path.isfile(manifest_path):
            raise ValueError("store manifest requires a regular file")
        with open(manifest_path, "rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError("store manifest exceeds its byte limit")
        manifest = loads_metadata(raw)
        if type(manifest) is not dict or manifest.get("format") not in {"rexdb-object", "rexstore"}:
            raise ValueError("unknown legacy store manifest declaration")
        owner = "object" if manifest["format"] == "rexdb-object" else "rex"
        if owner not in candidates:
            candidates.append(owner)
        if owner == "object":
            markers["file"] = tuple(name for name in markers["file"] if name != "index.json")
    for backend, names in markers.items():
        if any(os.path.lexists(os.path.join(path, name)) for name in names):
            if backend not in candidates:
                candidates.append(backend)
    if len(candidates) > 1:
        raise ValueError("directory contains conflicting RCDB store layouts")
    return candidates[0] if candidates else None


def recommend_backend(path: str = "", *, uri: str = "") -> dict:
    """Which backend to use, and why.

    Order of deference: an explicit URI scheme, then whatever already lives at the
    path, then the embedded store. SQL is not chosen automatically even when a
    driver is installed: it needs a server or a file the caller names, and
    guessing a database is not a decision a library should make for someone.
    """
    if uri:
        scheme = urlparse(uri).scheme
        if scheme and scheme != "auto":
            return {"backend": scheme, "reason": f"explicit in the uri ({scheme}://)"}
        if scheme == "auto" and not path:
            path = uri[len("auto://"):] or "./rcdb"
    found = _existing_backend(path) if path else None
    if found:
        return {"backend": found,
                "reason": f"a {found} store already exists at this path"}
    return {"backend": "local",
            "reason": "canonical checked local engine with retained history and tombstones, no server"}


def open_store(uri: str = "memory://", **options) -> RCStore:
    """Open an RCStore from a URI.

    auto:///path                    -> existing layout, else canonical LocalStore
    s3://…, gs://…, az://…          -> object adapter (driver and native publication capability required)
    memory://                       -> MemoryStore
    local:///path                   -> canonical checked local engine
    rex:///path                     -> read only legacy packed source
    file:///path  or  /path         -> read only legacy file source
    sqlite:///f.db, postgresql://…  -> SQLStore (any SQLAlchemy backend)
    <custom>://…                    -> a registered backend

    Explicit keyword options are passed to that backend's constructor/factory.
    Legacy writing requires read_only=False for the compatibility transition;
    new application stores should use local://, memory:// or native SQLite.
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"
    if scheme == "auto":
        path = uri[len("auto://"):] or "./rcdb"
        chosen = recommend_backend(path)["backend"]
        if chosen == "object":
            from .objectstore import open_object_store
            return open_object_store("file://"+os.path.abspath(path), **options)
        return open_store(f"{chosen}://{path}", **options)
    if scheme in _BACKENDS:
        return _BACKENDS.require(scheme)(uri, **options)
    if scheme == "memory":
        return MemoryStore(**options)
    if scheme == "file":
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        return FileStore(path or "./rcdb", **options)
    # anything SQLAlchemy understands
    return SQLStore(uri, **options)


# built in registrations
def _open_rexstore(uri: str, **options):
    from .rexstore import RexStore
    path = uri[len("rex://"):] if uri.startswith("rex://") else uri
    return RexStore(path or "./rexdb", **options)


def _open_objectstore(uri: str, **options):
    from .objectstore import open_object_store
    return open_object_store(uri, **options)


def _open_localstore(uri: str, **options):
    from .localstore import LocalStore
    return LocalStore(uri[len("local://"):] or "./rcdb-local", **options)


register_backend("memory", lambda uri, **options: MemoryStore(**options))
register_backend("rex", _open_rexstore)
register_backend("local", _open_localstore)
# one backend, every provider: fsspec routes the wire protocol to its driver.
for _scheme in ("s3", "gs", "gcs", "az", "abfs", "adl"):
    register_backend(_scheme, _open_objectstore)
register_backend("file", lambda uri, **options: FileStore(
    uri[len("file://"):] if uri.startswith("file://") else uri, **options))


# The process wide default store
#
# Resolve REXGRAPH_RCDB_URI once for callers using the process default store.

_DEFAULT_STORE: RCStore | None = None


def default_store_uri() -> str:
    """Configured URI, else auto:// under the config directory.

    New paths use LocalStore; existing legacy data opens read only for migration.
    """
    uri = os.environ.get("REXGRAPH_RCDB_URI")
    if uri:
        return uri
    base = os.environ.get("REXGRAPH_CONFIG_DIR",
                          os.path.join(os.path.expanduser("~"), ".config", "rexgraph"))
    return "auto://" + os.path.join(base, "rcdb")


def default_store() -> RCStore:
    """The shared default store for this process, opened once.

    Persistent by default: a caller that omits a store keeps its data instead of
    writing into a throwaway MemoryStore.

    Inside a request served with auth on, this is the store as that WORKSPACE may see
    it: records belonging to another one are absent rather than refused. The narrowing
    happens here because the store is one namespace shared by every workspace, and a
    rule applied at each of the routes that reach it is a rule the next route will not
    have. Outside a request, and whenever auth is off, the store is returned whole,
    which is what the CLI and anything running in process want.
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = open_store(default_store_uri())
    try:
        if _SCOPE_HOOK is None:
            return _DEFAULT_STORE
        scoped = _SCOPE_HOOK
        return scoped(_DEFAULT_STORE)
    except ImportError:                          # core install, no server
        return _DEFAULT_STORE


def reset_default_store() -> None:
    """Close the memoized default and drop it, so the next call re reads the environment.

    Dropping the reference alone left a SQL store's pool open until the collector got to
    it, which is what made the connections surface as warnings from unrelated frames.
    The close is best effort: a store that fails to close must not stop the reset, or a
    single bad store would pin the default forever.
    """
    global _DEFAULT_STORE
    store, _DEFAULT_STORE = _DEFAULT_STORE, None
    if store is not None:
        with contextlib.suppress(Exception):
            store.close()
