"""Corpus ingest: many documents into one relational complex store.

Each document becomes ONE complex over its own raw text, built by
`rexgraph.document.build_document`: spans as the base relation, sentence, paragraph
and chapter as coarsenings by parent map over that same field. Nothing here decides
anything about a document; it drives the construction that already exists and writes
what comes out.

**Why processes.** `build_document` and the exact rank reduction behind `betti` are pure
Python, and `rexgraph.compute.parallel_map` says in as many words that a thread pool
does nothing for a pure-Python body. So the fan-out is a process pool: each child reads,
builds, serializes and signs one document, and the parent does nothing but write. The
parent stays single-threaded because a `FileStore`'s index and append log are
per-process state, and two processes appending to one log is a corrupted store.

**Why it is resumable.** `put` APPENDS A VERSION rather than replacing, so re-running an
interrupted ingest over the same paths would give every already-stored document a second
identical version: twice the blobs, and a lineage that records a revision that never
happened. `pending()` filters the work list against the ids the store already holds, so
a resumed run does the remainder and nothing else.

**What is NOT stored.** The text. A document's prose stays in the file it came from and
the record carries a heap pointer to it, which is what makes the corpus 86 GiB of
structure over 23 GiB of source rather than both. The pointer is published only when the
text re-encodes to the file byte-for-byte, because a byte span into a file whose bytes
were not what we decoded addresses the wrong prose. `read_document` decides that, not
this module.
"""
from __future__ import annotations

import os
import time
import traceback

__all__ = ["DOC_TAGS", "ingest_corpus", "ingest_one", "pending"]

#: Applied to every record, so the corpus is separable from anything else in the store.
DOC_TAGS = ("document", "corpus")


def doc_id_for(path: str) -> str:
    """A stable id for a source file: its basename without extension.

    Stable across runs is the whole requirement: it is what lets `pending` recognise
    an already-ingested document, and what makes a re-ingest a new VERSION of the same
    document rather than a second document.
    """
    return os.path.splitext(os.path.basename(str(path)))[0]


def ingest_one(path: str, *, profile=None, analytics: bool = False,
               build_kw: dict | None = None) -> dict:
    """Build, serialize and sign ONE document. Runs in the worker process.

    Returns a plain dict so it survives the pipe without carrying a complex back. On
    failure it returns the same shape with `ok` False and the traceback, because one
    unreadable file out of 61,354 must not end the run: the caller records it and
    moves on.
    """
    from agent.rcdb import serialize_complex, structural_signature
    from rexgraph.document import build_document, read_document

    t0 = time.perf_counter()
    doc_id = doc_id_for(path)
    try:
        raw, exact = read_document(path)
        rex, info = build_document(raw, profile=profile, **(build_kw or {}))
        meta = {
            "input_type": "document",
            "source": os.path.abspath(str(path)),
            "layers": list(info["layers"]),
            "base_layer": info.get("base_layer"),
            "methods": dict(info["methods"]),
            "n_sentences": int(info["n_sentences"]),
            "n_spans": int(info.get("n_spans", info["n_sentences"])),
            "pair_mode": info["pair_mode"],
            "encoding_exact": bool(exact),
            # Resolvable prose, and only when a span into this file is addressable.
            "heap": os.path.abspath(str(path)) if exact else "",
            "vertex_labels": list(info["vocab"]),
        }
        sig = structural_signature(rex, meta, list(DOC_TAGS), analytics=analytics)
        return {"ok": True, "id": doc_id, "path": str(path),
                "blob": serialize_complex(rex), "sig": sig, "meta": meta,
                "seconds": time.perf_counter() - t0}
    except Exception as exc:                       # one bad file is a record, not an end
        return {"ok": False, "id": doc_id, "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
                "seconds": time.perf_counter() - t0}


def pending(store, paths) -> list[str]:
    """The paths whose documents the store does not already hold.

    This is what makes a re-run a resume instead of a duplicate: `put` appends a
    version, so an unfiltered re-run silently doubles the corpus.
    """
    have = set()
    try:
        have = {str(k) for k in store._idx}         # FileStore / MemoryStore index
    except AttributeError:
        for rec in store.list(limit=10 ** 9):
            have.add(str(rec.id))
    return [p for p in paths if doc_id_for(p) not in have]


def ingest_corpus(paths, store, *, profile=None, workers: int | None = None,
                  analytics: bool = False, build_kw: dict | None = None,
                  resume: bool = True, progress=None, compact_every: int = 5000,
                  limit: int | None = None) -> dict:
    """Ingest `paths` into `store`, building in parallel and writing serially.

    `workers` defaults to the CPU count less one, leaving the parent a core to write
    with. `progress(done, total, result)` is called after each write.

    Returns a summary carrying every failure, so a run that ends with failures ends
    LOUDLY rather than reporting a count that quietly excludes them.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    paths = [str(p) for p in paths]
    if resume:
        paths = pending(store, paths)
    if limit is not None:
        paths = paths[:limit]
    total = len(paths)
    out = {"total": total, "written": 0, "failed": 0, "failures": [],
           "bytes": 0, "seconds": 0.0}
    if total == 0:
        return out

    n = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    n = min(n, total)
    t0 = time.perf_counter()
    # forkserver: a clean single-threaded child, which is the pattern the pipeline
    # workers already use. `fork` inherits this process's threads and its BLAS pools.
    ctx = multiprocessing.get_context("forkserver")
    # The window is BOUNDED. Submitting every path at once lets the workers run ahead of
    # the parent's writes, and each finished result holds its blob: 1.4 MiB on the
    # measured average, so an unbounded queue is the whole corpus in memory. Keeping a
    # few tasks per worker in flight is enough to never leave one idle.
    window = max(2 * n, 32)
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
        from collections import deque
        it = iter(paths)
        futures = deque(
            ex.submit(ingest_one, p, profile=profile, analytics=analytics,
                      build_kw=build_kw)
            for p in (next(it, None) for _ in range(window)) if p is not None)
        i = 0
        while futures:
            fut = futures.popleft()
            nxt = next(it, None)
            if nxt is not None:
                futures.append(ex.submit(ingest_one, nxt, profile=profile,
                                         analytics=analytics, build_kw=build_kw))
            i += 1
            try:
                r = fut.result()
            except Exception as exc:               # a child that died, not a file that failed
                out["failed"] += 1
                out["failures"].append({"path": "<worker>", "error": repr(exc)})
                if progress is not None:
                    progress(i, total, {"ok": False, "path": "<worker>"})
                continue
            if not r["ok"]:
                out["failed"] += 1
                out["failures"].append({"path": r["path"], "error": r["error"],
                                        "traceback": r.get("traceback", "")})
            else:
                store.put_prepared(r["id"], r["blob"], r["sig"], r["meta"],
                                   list(DOC_TAGS))
                out["written"] += 1
                out["bytes"] += len(r["blob"])
            if progress is not None:
                progress(i, total, r)
            # Fold the append log into the snapshot periodically. Without this the log
            # carries every record written this run and the pending map holds them all.
            if compact_every and out["written"] and out["written"] % compact_every == 0:
                with _suppress():
                    store.compact()
    with _suppress():
        store.compact()
    out["seconds"] = time.perf_counter() - t0
    return out


class _suppress:
    """`compact` is an optimisation; a store that cannot do it is still correct."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
