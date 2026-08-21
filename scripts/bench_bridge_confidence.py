"""Bridge fraction as a confidence signal."""
import os

import numpy as np

os.environ.setdefault("REXGRAPH_RCDB_URI", "file://" + os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/rcdb"))

from rexgraph.graph import RexGraph
from rexgraph.partition import grade_leverage

STOP = {"where", "is", "the", "what", "to", "a", "of", "at", "who", "are", "there",
        "how", "does", "did", "in", "and", "or", "was", "were", "happens", "lives"}

CASES = [
    ("alice's adventures in wonderland", "pg11"),
    ("the adventures of sherlock holmes", "pg1661"),
    ("frankenstein or the modern prometheus", "pg84"),
    ("moby dick or the whale", "pg2701"),
    ("pride and prejudice", "pg1342"),
    ("the metamorphosis", "pg5200"),
    ("a tale of two cities", "pg98"),
    ("the picture of dorian gray", "pg174"),
    ("dracula", "pg345"),
    ("the war of the worlds", "pg36"),
    ("who lives at 221b baker street", "pg1661"),
    ("what happens to gregor samsa", "pg5200"),
    ("where is the whale hunted", "pg2701"),
]


def bridge_fraction(store, terms, limit=24):
    from agent.query_engine import _field_candidates

    hits = _field_candidates(store, set(terms), limit) or []
    vocab, rows = {}, []

    def vid(x):
        if x not in vocab:
            vocab[x] = len(vocab)
        return vocab[x]

    for rec, _s, _p, _c in hits:
        held = {str(w).lower() for w in (rec.meta or {}).get("vertex_labels", ())}
        for term in sorted(held & set(terms)):
            rows.append((vid("D:" + str(rec.id)), vid("T:" + term)))
    if len(rows) < 2:
        return None, None
    rex = RexGraph(sources=np.array([a for a, _ in rows], np.int32),
                   targets=np.array([b for _, b in rows], np.int32))
    rex._ensure_clean()
    r = np.asarray(grade_leverage(rex, 1)[0], float)
    return float((r >= 1.0 - 1e-9).mean()), int(rex.nE)


def rank_of(snap, ids, B, deg, codes, terms, doc):
    x = np.zeros(B.shape[0])
    hit = False
    for t in terms:
        c = codes.get(t)
        if c is not None:
            x[c] = 1.0 / max(deg[c], 1.0)
            hit = True
    if not hit:
        return None
    n = int(snap["n"])
    v = np.abs(B @ (B.T @ x))[:n]
    o = np.argsort(v)[::-1]
    rk = np.empty(len(o), np.int64)
    rk[o] = np.arange(len(o))
    return int(rk[ids.index(doc)]) + 1


def main():
    from agent import rcdb_index as ix
    from agent.rcdb import default_store

    store = default_store()
    snap = store._idx._snap
    ids = list(snap["ids"])
    B = ix.boundary_operator(snap)
    deg = ix._vertex_degree(snap, B)
    codes = ix._term_codes(snap)

    print(f"{'query':<38} {'bridge frac':>12} {'rels':>5} {'rank':>8}  correct?")
    rows = []
    for q, doc in CASES:
        terms = [w for w in q.lower().replace("'s", " ").split() if w not in STOP]
        frac, nE = bridge_fraction(store, terms)
        rk = rank_of(snap, ids, B, deg, codes, terms, doc)
        if frac is None or rk is None:
            continue
        rows.append((frac, rk, q))
        ok = "YES" if rk == 1 else ("near" if rk <= 10 else "no")
        print(f"{q:<38} {frac:>12.3f} {nE:>5} {rk:>8,}  {ok}")

    fr = np.array([r[0] for r in rows])
    rk = np.array([r[1] for r in rows], float)
    top1 = fr[rk == 1]
    miss = fr[rk > 10]
    print(f"\n  bridge fraction where retrieval hit rank 1 : "
          f"{sorted(np.round(top1, 3).tolist())}")
    print(f"  bridge fraction where it missed (rank>10)  : "
          f"{sorted(np.round(miss, 3).tolist())}")
    if len(top1) and len(miss):
        print(f"\n  lowest among the hits  = {top1.min():.3f}")
        print(f"  highest among the misses = {miss.max():.3f}")
        sep = top1.min() > miss.max()
        print(f"  SEPARABLE: {sep}"
              + (" : a bridge fraction above the misses predicts a hit"
                 if sep else " : the ranges overlap, so it is not a clean signal"))


if __name__ == "__main__":
    main()
