"""Documents into an RCDB, one record per document, prose left on disk.

The split is the one the storage layer already makes everywhere else, and the reason it
matters here is scale: 61,354 Gutenberg texts are 24.35 GB of prose, and a record that
carries its own text doubles that for bytes nothing reads until someone asks for one
section.

    the record    the field, its layers, their digests and the Merkle root. Tensors, and
                  1.86x the raw size measured across 24 books.
    the heap      the file on disk, untouched. Every section carries a byte span into it,
                  so recovering prose is one seek: `rexgraph.document.section_text`.

`source_text` is deliberately NOT written. It is the thing that made retrieval read prose
out of a blob instead of seeking, and putting it back here would undo the point of having
spans at all.

Resumable in the same sense the fetcher is: a document whose content digest already
matches the stored one is skipped, so a re-run costs a hash and not a rebuild.
"""
from __future__ import annotations

import hashlib
import os
import time

__all__ = ["ingest_document", "ingest_directory", "document_meta",
           "backfill_analytics"]

#: extensions read as plain text. Anything else needs an adapter and is not this
#: module's job. Silently treating a PDF as utf-8 would produce a complex over mojibake.
TEXT_EXT = (".txt", ".md", ".rst", ".text")


def _content_digest(raw):
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def document_meta(rex, info, *, doc_id, source, heap, content_digest):
    """What a caller can read WITHOUT deserialising the complex.

    The layer methods are in here for the same reason they are on the sectioning: a
    reader has to be able to tell a chapter layer that matched real headings from a
    document that simply has none, and that is a query, not an inspection.
    """
    from rexgraph.io.rex_state import to_state
    from rexgraph.merkle import build_merkle

    st = to_state(rex)
    m = build_merkle(rex)
    return {
        "doc_id": doc_id, "source": source, "object_type": "document",
        # the POINTER layer: where the prose actually lives
        "heap": os.path.abspath(str(heap)) if heap else "",
        "content_digest": content_digest,
        "span_units": "bytes",
        "state_digest": st.header.get("digest", ""),
        "merkle_root": m.root.hex(),
        "merkle_chain": list(m.chain),
        "layers": list(info["layers"]),
        "methods": dict(info["methods"]),
        "n_sentences": int(info["n_sentences"]),
        "n_dropped": int(info["n_dropped"]),
        "pair_mode": info["pair_mode"],
        "vertex_labels": list(info["vocab"]),
        "built": time.time(),
    }


def ingest_document(store, source, *, doc_id=None, raw=None, heap=None,
                    tags=(), skip_unchanged=True, analytics=False, **build_kw):
    """One document into `store`. Returns `(record_id, meta)` or `(record_id, None)`
    when it was already present and unchanged.

    `source` is a path, or a name when `raw` is given. `heap` defaults to `source` when
    that is a real file, because the span pointers are only meaningful against the file
    they were computed from.

    `analytics` is OFF here, unlike the signature's own default. Measured per book:
    building the document is 0.39 s and the analytics columns (`kappa_mean` and the
    information metrics) are 2.17 s, so they are 85% of a corpus ingest (37 of its 44
    hours) and nothing on the retrieval path reads them. With them off the corpus lands
    in 6.7 h on one thread or ~34 min on twelve, and `backfill_analytics` fills them in
    for whatever subset earns it.
    """
    from rexgraph.document import build_document, read_document

    path = str(source)
    is_file = raw is None or os.path.exists(path)
    exact = True
    if raw is None:
        # read_document, NOT open(): text mode translates CRLF and shifts every span
        raw, exact = read_document(path, build_kw.get("encoding", "utf-8"))
    rid = str(doc_id or os.path.splitext(os.path.basename(path))[0] or "doc")
    if heap is None and is_file and os.path.exists(path):
        # a heap pointer is only published when the text re-encodes to the file
        # byte-for-byte; otherwise the spans are valid against `raw` and nothing else,
        # and handing out a path would produce confidently wrong prose.
        heap = path if exact else None

    cd = _content_digest(raw)
    if skip_unchanged:
        try:
            rec = store.get_record(rid)
            if rec is not None and (rec.meta or {}).get("content_digest") == cd:
                return rid, None
        except Exception:
            pass                       # absent, or a store with no get_record: build it

    rex, info = build_document(raw, **build_kw)
    meta = document_meta(rex, info, doc_id=rid, source=path, heap=heap,
                         content_digest=cd)
    store.put(rid, rex, meta=meta, tags=[*tags, "document"], analytics=analytics)
    return rid, meta


def ingest_directory(store, root, *, recursive=True, extensions=TEXT_EXT, limit=None,
                     prefix="", skip_unchanged=True, analytics=False, log=print,
                     **build_kw):
    """Every text file under `root`, resumably. Returns a small run summary.

    A document that fails to build does not stop the run: a corpus of 61k files will
    contain some that segment to nothing, and refusing the whole ingest over one of them
    would be the wrong trade. Failures are counted and the first few reported.
    """
    root = os.path.expanduser(str(root))
    exts = tuple(e.lower() for e in extensions)
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith(exts):
                files.append(os.path.join(dirpath, n))
        if not recursive:
            break
    files.sort()
    if limit:
        files = files[:int(limit)]

    n_new = n_skip = n_fail = 0
    failures = []
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        rid = prefix + os.path.splitext(os.path.basename(f))[0]
        try:
            _rid, meta = ingest_document(store, f, doc_id=rid, analytics=analytics,
                                         skip_unchanged=skip_unchanged, **build_kw)
            if meta is None:
                n_skip += 1
            else:
                n_new += 1
        except Exception as exc:                       # noqa: BLE001 - corpus scale
            n_fail += 1
            if len(failures) < 10:
                failures.append((rid, f"{type(exc).__name__}: {exc}"))
        if log and i % 500 == 0:
            el = time.perf_counter() - t0
            log(f"  {i:,}/{len(files):,}  new {n_new:,}  skip {n_skip:,}  "
                f"fail {n_fail:,}  {i / max(el, 1e-9):.1f}/s")
    el = time.perf_counter() - t0
    if log:
        log(f"  done: {n_new:,} ingested, {n_skip:,} unchanged, {n_fail:,} failed, "
            f"{el:.1f}s")
        for rid, err in failures:
            log(f"    {rid}: {err}")
    return {"n_files": len(files), "n_new": n_new, "n_skipped": n_skip,
            "n_failed": n_fail, "failures": failures, "seconds": el}


def backfill_analytics(store, ids=None, *, voids=False, log=print):
    """Compute the analytics columns for records ingested without them.

    Separated from ingest because the two answer different questions. Getting the corpus
    IN is structural and cheap; `avg(kappa_mean) GROUP BY source` is a reporting query and
    costs 2.17 s a document, which is worth paying for the records someone actually asks
    about and not for 61,354 of them up front.

    Re-puts each record with a full signature. Returns a small run summary.
    """
    from agent.rcdb import structural_signature

    rows = list(ids) if ids is not None else [r.id for r in store.list()]
    n_ok = n_fail = 0
    t0 = time.perf_counter()
    for i, rid in enumerate(rows, 1):
        try:
            rec = store.get_record(rid)
            rex = store.get(rid)
            sig = structural_signature(rex, rec.meta, list(rec.signature.get("tags") or []),
                                       analytics=True, voids=voids)
            store.put(rid, rex, meta=rec.meta,
                      tags=list(rec.signature.get("tags") or []), analytics=True,
                      voids=voids)
            del sig
            n_ok += 1
        except Exception:                              # noqa: BLE001 - corpus scale
            n_fail += 1
        if log and i % 200 == 0:
            log(f"  {i:,}/{len(rows):,}  ok {n_ok:,}  fail {n_fail:,}")
    el = time.perf_counter() - t0
    if log:
        log(f"  backfilled {n_ok:,} of {len(rows):,} in {el:.1f}s ({n_fail:,} failed)")
    return {"n": len(rows), "n_ok": n_ok, "n_failed": n_fail, "seconds": el}
