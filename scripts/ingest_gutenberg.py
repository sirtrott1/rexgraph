"""Ingest a Gutenberg mirror into a record store."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

DEFAULT_TEXTS = os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/texts")
DEFAULT_STORE = os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/rcdb")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", default=DEFAULT_TEXTS)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analytics", action="store_true",
                    help="also compute the analytics columns (much slower)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    from agent.corpus.ingest import ingest_corpus
    from agent.rcdb import FileStore
    from rexgraph.corpus_profile import ENGLISH_GUTENBERG

    paths = sorted(glob.glob(os.path.join(args.texts, "*", "*.txt")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.texts, "*.txt")))
    print(f"{len(paths):,} source files under {args.texts}", flush=True)

    os.makedirs(args.store, exist_ok=True)
    store = FileStore(args.store)

    t0 = time.time()
    state = {"last": 0.0, "bytes": 0}

    def progress(done, total, r):
        state["bytes"] += len(r.get("blob") or b"")
        now = time.time()
        if now - state["last"] < 5.0 and done != total:
            return
        state["last"] = now
        el = now - t0
        rate = done / el if el else 0.0
        eta = (total - done) / rate if rate else 0.0
        print(f"  {done:,}/{total:,}  {rate:5.1f} doc/s  "
              f"{state['bytes'] / 2**30:6.2f} GiB  "
              f"elapsed {el/60:5.1f}m  eta {eta/60:6.1f}m", flush=True)

    out = ingest_corpus(paths, store, profile=ENGLISH_GUTENBERG,
                        workers=args.workers, analytics=args.analytics,
                        resume=not args.no_resume, limit=args.limit,
                        progress=progress)

    print(f"\nwritten {out['written']:,}   failed {out['failed']:,}   "
          f"{out['bytes']/2**30:.1f} GiB   {out['seconds']/60:.1f} min")
    for f in out["failures"][:20]:
        print(f"  FAIL {f['path']}: {f['error']}")
    if len(out["failures"]) > 20:
        print(f"  ... and {len(out['failures']) - 20:,} more")
    return 1 if out["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
