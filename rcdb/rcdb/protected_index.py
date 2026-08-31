"""
rcdb.protected_index: exact search over a vocabulary the index does not store.

The canonical RCDB snapshot reconstructs records, so it is not a security boundary: its
labels are written in the clear, both into `RexStore`'s in-memory map and into the SQL
labels table. This builds a SEPARATE, disposable relation for exact term lookup whose
vocabulary is fixed-width tokens, so a persisted index contains neither a plaintext term
nor a plaintext record id and still answers "which records carry this exact term".

It is the search half of the container work. `rexgraph.io` seals what a container holds;
this seals what an index reveals about it. Sealing the records and then shipping a
plaintext term index beside them protects nothing.

Three modes per accession kind, chosen per deployment:

    public      sha256 over the framed term. Anyone can recompute a token from a guess,
                so this hides nothing from a dictionary attack and is for terms that are
                not secret but should not be greppable.
    structural  the same, in a separate domain, so a structural token can never collide
                with a public one for the same term.
    keyed       HMAC-SHA256 under a key this workspace can name. Without the key a token
                cannot be produced from a guessed term, which is the mode that actually
                resists enumeration.
    none        the kind is not indexed at all.

The key arrives as an `IndexKeyProvider`, never as key bytes. That protocol is the whole
interoperation: an application supplies a provider that resolves a key identifier however
its own tenancy requires, and this package never learns how. A multi-tenant server
typically resolves per workspace, so the same identifier in two workspaces is two
different keys and one tenant's tokens are meaningless to another.

Ported from the standalone rcdb package. The framing, the domain separators and the
persisted layout are unchanged and FORMAT_VERSION stays 2, so an index written by that
package still loads here.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Protocol

import numpy as np

FORMAT_VERSION = 2
TOKEN_BYTES = 32
_PUBLIC = "public"
_KEYED = "keyed"
_STRUCTURAL = "structural"
_NONE = "none"
_MODES = frozenset((_PUBLIC, _KEYED, _STRUCTURAL, _NONE))


class IndexKeyProvider(Protocol):
    """Resolve a search index key by opaque key identity."""

    def key(self, key_id: str) -> bytes: ...


@dataclass(frozen=True)
class StaticIndexKeyProvider:
    """In-process search key provider for tests and single-operator deployments.

    A server supplies its own `IndexKeyProvider` instead, one that namespaces a key
    identifier by tenant. Constructing this one inside a request would hand every tenant
    the same key.
    """

    keys: dict[str, bytes]

    def key(self, key_id: str) -> bytes:
        try:
            value = self.keys[str(key_id)]
        except KeyError as exc:
            raise KeyError(f"no search index key for {key_id!r}") from exc
        return bytes(value)


@dataclass(frozen=True)
class IndexPolicy:
    """Choose how each accession kind is represented in the derived search relation."""

    modes: dict[str, str]
    key_id: str | None = None

    def __post_init__(self):
        for kind, mode in self.modes.items():
            if mode not in _MODES:
                raise ValueError(f"invalid index mode {mode!r} for {kind!r}")
        if any(mode == _KEYED for mode in self.modes.values()) and not self.key_id:
            raise ValueError("keyed index policy requires key_id")

    def mode(self, kind: str) -> str:
        return self.modes.get(str(kind), _NONE)

    @property
    def digest(self) -> str:
        from rexgraph.io.manifest import manifest_digest
        return manifest_digest({
            "key_id": self.key_id,
            "modes": dict(sorted(self.modes.items())),
            "object_type": "RCDBIndexPolicy",
            "version": FORMAT_VERSION,
        })


def _frame(kind: str, term: str) -> bytes:
    """Length-prefixed framing, so ("ab", "c") and ("a", "bc") cannot frame alike."""
    kb = str(kind).encode("utf-8")
    tb = str(term).encode("utf-8")
    return (b"rcdb-search-term\x00" + len(kb).to_bytes(4, "big") + kb
            + len(tb).to_bytes(8, "big") + tb)


def term_token(kind: str, term: str, *, mode: str,
               key_id: str | None = None,
               keys: IndexKeyProvider | None = None) -> bytes:
    """Return one domain separated exact search token."""
    framed = _frame(kind, term)
    if mode == _PUBLIC:
        return hashlib.sha256(b"public\x00" + framed).digest()
    if mode == _STRUCTURAL:
        return hashlib.sha256(b"structural\x00" + framed).digest()
    if mode == _KEYED:
        if keys is None or not key_id:
            raise ValueError("keyed term lookup requires an index key provider")
        return hmac.new(keys.key(key_id), b"keyed\x00" + framed, hashlib.sha256).digest()
    if mode == _NONE:
        raise ValueError("term kind is not indexed")
    raise ValueError(f"unknown index mode {mode!r}")


def record_token(record_id: str, *, key_id: str | None = None,
                 keys: IndexKeyProvider | None = None) -> bytes:
    """Return a protected identity token for one canonical RCDB record id."""
    framed = _frame("record", record_id)
    if keys is not None and key_id:
        return hmac.new(keys.key(key_id), b"record\x00" + framed, hashlib.sha256).digest()
    return hashlib.sha256(b"record\x00" + framed).digest()


def version_record_token(record_id: str, version: int, *, key_id: str | None = None,
                         keys: IndexKeyProvider | None = None) -> bytes:
    """Return a protected identity token for one RCDB record version."""
    rid = str(record_id).encode("utf-8")
    framed = (b"rcdb-search-record-version\x00" + len(rid).to_bytes(8, "big") + rid
              + int(version).to_bytes(8, "big", signed=False))
    if keys is not None and key_id:
        return hmac.new(keys.key(key_id), b"record-version\x00" + framed,
                        hashlib.sha256).digest()
    return hashlib.sha256(b"record-version\x00" + framed).digest()


@dataclass
class SearchRelation:
    """One derived record to search token relation.

    `rel_ptr` and `rel_idx` use the same record-first incidence convention as the
    canonical RCDB index. Both records and searchable terms are fixed width tokens in
    persisted form. `record_ids` is an optional in-memory resolver and is never required
    for serialization, which is what keeps identities out of the file.
    """

    record_tokens: np.ndarray
    token_bytes: np.ndarray
    rel_ptr: np.ndarray
    rel_idx: np.ndarray
    rel_kind: np.ndarray
    kinds: tuple[str, ...]
    policy_digest: str
    record_ids: tuple[str, ...] | None = None

    @property
    def n_records(self) -> int:
        return int(self.record_tokens.shape[0])

    def tokens_for(self, kind: str, term: str, *, policy: IndexPolicy,
                   keys: IndexKeyProvider | None = None) -> list[bytes]:
        """Return protected record tokens naming one exact term."""
        rows = self._rows_for(kind, term, policy=policy, keys=keys)
        return [bytes(self.record_tokens[r]) for r in rows]

    def ids_for(self, kind: str, term: str, *, policy: IndexPolicy,
                keys: IndexKeyProvider | None = None) -> list[str]:
        """Return record identities when an in-memory resolver is attached."""
        if self.record_ids is None:
            raise ValueError("this persisted search relation has no record id resolver")
        rows = self._rows_for(kind, term, policy=policy, keys=keys)
        return [self.record_ids[r] for r in rows]

    def _rows_for(self, kind: str, term: str, *, policy: IndexPolicy,
                  keys: IndexKeyProvider | None = None) -> list[int]:
        # The policy digest is checked first: a relation built under one policy answers
        # nothing under another, because the same term tokenises differently and a miss
        # would read as "no such record" rather than as the mismatch it is.
        if policy.digest != self.policy_digest:
            raise ValueError("search policy does not match this index")
        try:
            kind_code = self.kinds.index(str(kind))
        except ValueError:
            return []
        mode = policy.mode(kind)
        if mode == _NONE:
            return []
        token = term_token(kind, term, mode=mode, key_id=policy.key_id, keys=keys)
        if self.token_bytes.size == 0:
            return []
        rows = self.token_bytes.view(f"S{TOKEN_BYTES}").reshape(-1)
        target = np.bytes_(token)
        pos = int(np.searchsorted(rows, target))
        if pos >= len(rows) or bytes(rows[pos]) != token:
            return []
        vertex = self.n_records + pos
        out = []
        seen = set()
        for rel in range(len(self.rel_ptr) - 1):
            if int(self.rel_kind[rel]) != kind_code:
                continue
            lo, hi = int(self.rel_ptr[rel]), int(self.rel_ptr[rel + 1])
            span = self.rel_idx[lo:hi]
            if np.any(span[1:] == vertex):
                row = int(span[0])
                if row not in seen:
                    seen.add(row)
                    out.append(row)
        return out


def _token_matrix(raw) -> np.ndarray:
    """Fixed width token rows, or an empty matrix of the right shape."""
    if not raw:
        return np.zeros((0, TOKEN_BYTES), dtype=np.uint8)
    return np.frombuffer(b"".join(raw), dtype=np.uint8).reshape(-1, TOKEN_BYTES).copy()


def build_search_relation(records, policy: IndexPolicy,
                          *, keys: IndexKeyProvider | None = None) -> SearchRelation:
    """Build a disposable protected search relation from `(id, ComplexRecord)` rows."""
    from .index import KINDS, _terms_of

    rows = list(records)
    record_ids = tuple(f"{rid}\x00{int(rec.version)}" for rid, rec in rows)
    record_tokens = _token_matrix([
        version_record_token(rid, int(rec.version), key_id=policy.key_id, keys=keys)
        for rid, rec in rows])
    kinds = tuple(KINDS)
    raw_relations = []
    token_set = set()
    for row, (_rid, rec) in enumerate(rows):
        for kind_code, terms in _terms_of(rec):
            kind = kinds[int(kind_code)]
            mode = policy.mode(kind)
            if mode == _NONE:
                continue
            tokens = [term_token(kind, term, mode=mode, key_id=policy.key_id, keys=keys)
                      for term in terms]
            if not tokens:
                continue
            token_set.update(tokens)
            raw_relations.append((row, int(kind_code), tokens))

    tokens = sorted(token_set)
    token_pos = {token: i for i, token in enumerate(tokens)}
    n_records = len(record_ids)
    rel_ptr = [0]
    rel_idx = []
    rel_kind = []
    for row, kind_code, terms in raw_relations:
        rel_idx.append(row)
        rel_idx.extend(n_records + token_pos[token] for token in terms)
        rel_ptr.append(len(rel_idx))
        rel_kind.append(kind_code)

    return SearchRelation(
        record_tokens, _token_matrix(tokens), np.asarray(rel_ptr, np.int64),
        np.asarray(rel_idx, np.int64), np.asarray(rel_kind, np.uint8), kinds,
        policy.digest, record_ids)


def build_search_relation_from_tokens(records, policy: IndexPolicy, *, kind: str,
                                      keys: IndexKeyProvider | None = None) -> SearchRelation:
    """Build a protected search relation from record version token rows.

    `records` contains `(id, version, tokens)` triples. Tokens are already protected
    exact term tokens and are never decoded by this function.
    """
    from .index import KINDS

    rows = [(str(rid), int(version), tuple(sorted({bytes(x) for x in tokens})))
            for rid, version, tokens in records]
    record_ids = tuple(f"{rid}\x00{version}" for rid, version, _tokens in rows)
    record_tokens = _token_matrix([
        version_record_token(rid, version, key_id=policy.key_id, keys=keys)
        for rid, version, _tokens in rows])
    try:
        kind_code = tuple(KINDS).index(str(kind))
    except ValueError as exc:
        raise ValueError(f"unknown protected search kind {kind!r}") from exc
    token_set = set()
    for _rid, _version, tokens in rows:
        for token in tokens:
            if len(token) != TOKEN_BYTES:
                raise ValueError("protected search tokens must be 32 bytes")
            token_set.add(token)
    tokens = sorted(token_set)
    token_pos = {token: i for i, token in enumerate(tokens)}
    rel_ptr = [0]
    rel_idx = []
    rel_kind = []
    n_records = len(rows)
    for row, (_rid, _version, values) in enumerate(rows):
        if not values:
            continue
        rel_idx.append(row)
        rel_idx.extend(n_records + token_pos[token] for token in values)
        rel_ptr.append(len(rel_idx))
        rel_kind.append(kind_code)
    return SearchRelation(
        record_tokens, _token_matrix(tokens), np.asarray(rel_ptr, np.int64),
        np.asarray(rel_idx, np.int64), np.asarray(rel_kind, np.uint8), tuple(KINDS),
        policy.digest, record_ids)


def save_search_relation(path, relation: SearchRelation) -> None:
    """Write a protected search relation without plaintext terms or record identities."""
    import json

    from safetensors.numpy import save_file
    metadata = {"rcdb_search": json.dumps({
        "format_version": FORMAT_VERSION, "kinds": list(relation.kinds),
        "policy_digest": relation.policy_digest}, separators=(",", ":"), sort_keys=True)}
    save_file({
        "record_tokens": np.ascontiguousarray(relation.record_tokens, dtype=np.uint8),
        "token_bytes": np.ascontiguousarray(relation.token_bytes, dtype=np.uint8),
        "rel_ptr": np.ascontiguousarray(relation.rel_ptr, dtype=np.int64),
        "rel_idx": np.ascontiguousarray(relation.rel_idx, dtype=np.int64),
        "rel_kind": np.ascontiguousarray(relation.rel_kind, dtype=np.uint8),
    }, str(path), metadata=metadata)


def load_search_relation(path) -> SearchRelation:
    """Load one protected search relation and verify its structural bounds.

    The bounds are checked because this file is derived and disposable: a malformed one
    should be refused rather than indexed into, and an index that points outside its own
    vertex range would read someone else's row.
    """
    import json

    from safetensors import safe_open
    from safetensors.numpy import load_file
    arrays = load_file(str(path))
    with safe_open(str(path), framework="numpy") as f:
        raw = (f.metadata() or {}).get("rcdb_search")
    if raw is None:
        raise ValueError("not an RCDB protected search relation")
    meta = json.loads(raw)
    if int(meta.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError("unsupported RCDB protected search relation version")
    record_tokens = np.asarray(arrays["record_tokens"], dtype=np.uint8)
    token_bytes = np.asarray(arrays["token_bytes"], dtype=np.uint8)
    rel_ptr = np.asarray(arrays["rel_ptr"], dtype=np.int64)
    rel_idx = np.asarray(arrays["rel_idx"], dtype=np.int64)
    rel_kind = np.asarray(arrays["rel_kind"], dtype=np.uint8)
    if record_tokens.ndim != 2 or record_tokens.shape[1] != TOKEN_BYTES:
        raise ValueError("invalid protected record token matrix")
    if token_bytes.ndim != 2 or token_bytes.shape[1] != TOKEN_BYTES:
        raise ValueError("invalid protected term token matrix")
    if rel_ptr.ndim != 1 or len(rel_ptr) == 0 or int(rel_ptr[0]) != 0:
        raise ValueError("invalid protected relation pointer")
    if np.any(np.diff(rel_ptr) < 0) or int(rel_ptr[-1]) != len(rel_idx):
        raise ValueError("invalid protected relation pointer bounds")
    if len(rel_kind) != len(rel_ptr) - 1:
        raise ValueError("protected relation kind length mismatch")
    nV = len(record_tokens) + len(token_bytes)
    if len(rel_idx) and (int(rel_idx.min()) < 0 or int(rel_idx.max()) >= nV):
        raise ValueError("protected relation index is outside its vertex range")
    return SearchRelation(record_tokens, token_bytes, rel_ptr, rel_idx, rel_kind,
                          tuple(meta.get("kinds", ())), str(meta["policy_digest"]), None)
