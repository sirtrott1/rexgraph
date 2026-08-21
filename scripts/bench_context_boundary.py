"""The threshold-free context boundary against top_k."""
import os

import numpy as np

os.environ.setdefault("REXGRAPH_RCDB_URI", "file://" + os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/rcdb"))

from rexgraph.graph import RexGraph
from rexgraph.partition import grade_leverage

QUERIES = [
    "who lives at 221b baker street",
    "where is the whale hunted",
    "what happens to gregor samsa",
    "the adventures of sherlock holmes",
    "moby dick or the whale",
    "how does photosynthesis work",
]
STOP = {"where", "is", "the", "what", "to", "a", "of", "at", "who", "are", "there",
        "how", "does", "did", "in", "and", "or", "was", "were", "happens", "lives"}


def neighbourhood(store, terms, limit=24):
    from agent.query_engine import _field_candidates

    hits = _field_candidates(store, set(terms), limit) or []
    docs, vocab, rows = [], {}, []

    def vid(x):
        if x not in vocab:
            vocab[x] = len(vocab)
        return vocab[x]

    for rec, _s, _p, _c in hits:
        did = str(rec.id)
        held = {str(w).lower() for w in (rec.meta or {}).get("vertex_labels", ())}
        shared = sorted(held & set(terms))
        if not shared:
            continue
        docs.append(did)
        for term in shared:
            rows.append((vid("D:" + did), vid("T:" + term)))
    if len(rows) < 2:
        return None, None, None
    src = np.array([a for a, _b in rows], np.int32)
    tgt = np.array([b for _a, b in rows], np.int32)
    rex = RexGraph(sources=src, targets=tgt)
    rex._ensure_clean()
    inv = {v: k for k, v in vocab.items()}
    return rex, rows, inv


def gap_cut(values):
    v = np.sort(np.asarray(values, float))[::-1]
    if v.size < 2:
        return 0, 1.0
    ratios = v[:-1] / np.maximum(v[1:], 1e-300)
    i = int(np.argmax(ratios))
    return i + 1, float(ratios[i])


def main():
    from agent.rcdb import default_store
    store = default_store()

    print(f"{'query':<36} {'rels':>5} {'R_eff range':>18} {'gap':>8} {'keep':>6} "
          f"{'>0.9':>5}")
    for q in QUERIES:
        terms = [w for w in q.lower().split() if w not in STOP]
        rex, rows, _inv = neighbourhood(store, terms)
        if rex is None:
            print(f"{q:<36} (no neighbourhood)")
            continue
        r = np.asarray(grade_leverage(rex, 1)[0], float)
        keep, gap = gap_cut(r)
        n_bridge = int((r > 0.9).sum())
        print(f"{q:<36} {int(rex.nE):>5} [{r.min():.4f}, {r.max():.4f}] "
              f"{gap:>8.3f} {keep:>6} {n_bridge:>5}")

    print("""
READ:

  gap ~ 1.0   the R_eff values are a continuum: no natural boundary exists, and the
              honest answer is that this neighbourhood does not decompose. A top_k here
              is picking an arbitrary point on a smooth curve.
  gap >> 1    a real separation: the relations above the gap are load-bearing and the
              rest are redundant, and the cut is a property of the structure.

The '>0.9' column is what engine.py:746 counts today. Where it disagrees with the gap
reading, one of them is a constant and the other is a measurement.""")


if __name__ == "__main__":
    main()
