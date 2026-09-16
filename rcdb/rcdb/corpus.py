"""Owned accession snapshots over the store's visible record versions."""
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
import sys

from . import index


def options(query, reading="existence", exact=True, as_of=None, valid_at=None):
    if not isinstance(query, (list, tuple)) or any(not isinstance(x, str) for x in query):
        raise TypeError("corpus query must be a list of complete vocabulary terms")
    if reading not in ("share", "existence"):
        raise ValueError("corpus reading must be share or existence")
    if not isinstance(exact, bool):
        raise TypeError("corpus exact must be boolean")
    for value in (as_of, valid_at):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not isfinite(value)):
            raise ValueError("corpus timestamps must be finite numbers or None")
    return tuple(sorted({x.lower() for x in query})), reading, exact, as_of, valid_at


@dataclass(frozen=True)
class CorpusSnapshot:
    """A fixed record axis and projected accession relation, without payload reads."""

    ids: tuple
    versions: tuple
    digest: str
    as_of: float | None
    valid_at: float | None
    fields: tuple
    _index: dict = field(repr=False, compare=False)

    def response(self, query, *, reading="existence", exact=True):
        from fractions import Fraction
        query, reading, exact, _, _ = options(query, reading, exact)
        if exact:
            nonzero = index.record_response_exact(self._index, query, reading=reading)
            scores = tuple(nonzero.get(i, Fraction(0)) for i in range(len(self.ids)))
        else:
            values, _ = index.record_response(self._index, query, reading=reading)
            scores = tuple(map(float, values))
        return {"scores": scores, "ids": self.ids, "versions": self.versions,
                "snapshot_digest": self.digest, "as_of": self.as_of, "valid_at": self.valid_at,
                "fields": self.fields, "reading": reading, "exact": exact,
                "coefficient_domain": "Q" if exact else "real"}


def capture(store, *, as_of=None, valid_at=None, signature_fields=None):
    """Select and copy one version per id while holding the store handle's lock.

    Bounded signature fields exclude metadata vertex labels. SQL and custom
    backends capture anew; only handles with an owned local index reuse a cache.
    This is not a transaction spanning independent handles or processes.
    """
    from .core import ComplexRecord, PublicationUncertainError
    from rexgraph.io.manifest import manifest_digest
    options([], as_of=as_of, valid_at=valid_at)
    fields = tuple(name for name in index.KINDS if signature_fields is None
                   or (name not in index.FROM_META and name in signature_fields))
    key = as_of, valid_at, fields
    with store._transaction_lock:
        if store._publication_uncertain:
            raise PublicationUncertainError("RCDB publication is uncertain; reopen before capturing a corpus")
        cached = store._corpus_cache
        if store._cache_corpus and cached is not None and cached[0] == key:
            return cached[1]
        rows = store.list(limit=sys.maxsize, as_of=as_of, valid_at=valid_at)
        by_id = {}
        for row in rows:
            by_id.setdefault(row.id, []).append(row)
        selected, identities = [], []
        for rid in sorted(by_id):
            row = store._select_version(by_id[rid], as_of, valid_at)
            if row is None:
                continue
            # Copy only accession terms, not unrelated metadata or payload state.
            projected = ComplexRecord(id=rid, created=row.created, version=row.version,
                signature={}, meta={})
            visible = ComplexRecord(id=rid, created=row.created,
                signature={name: row.signature[name] for name in fields
                           if name not in index.FROM_META and name in row.signature},
                meta={name: row.meta[name] for name in fields
                      if name in index.FROM_META and name in row.meta})
            for kind, terms in index._terms_of(visible):
                name = index.KINDS[kind]
                if name in fields:
                    target = projected.meta if name in index.FROM_META else projected.signature
                    target[name] = terms[0] if name in index.SINGLE else list(terms)
            selected.append((rid, projected))
            identities.append((rid, int(row.version), index._terms_of(projected)))
        ix = index.build(selected)
        snapshot = CorpusSnapshot(tuple(ix["ids"]), tuple(r.version for _, r in selected),
            manifest_digest({"object_type": "RCDBCorpusSnapshot", "version": 1,
                             "records": identities, "fields": fields,
                             "as_of": as_of, "valid_at": valid_at}),
            as_of, valid_at, fields, ix)
        if store._cache_corpus:
            store._corpus_cache = key, snapshot
        return snapshot
