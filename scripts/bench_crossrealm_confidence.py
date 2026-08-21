"""Cross-realm agreement as a confidence signal."""
import os
from itertools import combinations

import numpy as np

os.environ.setdefault("REXGRAPH_RCDB_URI", "file://" + os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/rcdb"))

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


def lexical_neighbours(term, index, cap=40):
    out = set()
    for _kind, _head, members in index.get(str(term).lower(), ())[:cap]:
        out |= {str(m).lower() for m in members if " " not in str(m)}
    return out


def coherence(terms, index):
    ts = sorted(set(terms))
    if len(ts) < 2:
        return None, 0, 0
    nb = {t: lexical_neighbours(t, index) for t in ts}
    linked = 0
    total = 0
    for a, b in combinations(ts, 2):
        total += 1
        if b in nb[a] or a in nb[b] or (nb[a] & nb[b]):
            linked += 1
    return linked / total, linked, total


def main():
    from agent import rcdb_index as ix
    from agent.answerers import _default_registry
    from agent.rcdb import default_store

    store = default_store()
    snap = store._idx._snap
    ids = list(snap["ids"])
    n = int(snap["n"])
    B = ix.boundary_operator(snap)
    deg = ix._vertex_degree(snap, B)
    codes = ix._term_codes(snap)
    index = _default_registry()["wiktionary"][0]._index()

    print(f"{'query':<38} {'rank':>7} {'coh':>6} {'pairs':>7}  shared terms")
    rows = []
    for q, doc in CASES:
        terms = [w for w in q.lower().replace("'s", " ").split() if w not in STOP]
        x = np.zeros(B.shape[0])
        seen = []
        for t in terms:
            c = codes.get(t)
            if c is not None:
                x[c] = 1.0 / max(deg[c], 1.0)
                seen.append(t)
        if not seen:
            continue
        v = np.abs(B @ (B.T @ x))[:n]
        order = np.argsort(v)[::-1]
        rk = int(np.where(order == ids.index(doc))[0][0]) + 1
        top = ids[int(order[0])]

        rec = store.get_record(top)
        held = {str(w).lower() for w in (rec.meta or {}).get("vertex_labels", ())}
        shared = sorted(set(seen) & held)
        coh, linked, total = coherence(shared, index)
        if coh is None:
            print(f"{q:<38} {rk:>7,} {'--':>6} {'<2':>7}  {shared}")
            continue
        rows.append((coh, rk, q))
        print(f"{q:<38} {rk:>7,} {coh:>6.2f} {linked:>3}/{total:<3}  {shared[:5]}")

    coh = np.array([r[0] for r in rows])
    rk = np.array([r[1] for r in rows], float)
    hit, miss = coh[rk <= 5], coh[rk > 10]
    print(f"\n  coherence where retrieval hit top-5 : {sorted(np.round(hit, 2).tolist())}")
    print(f"  coherence where it missed (rank>10) : {sorted(np.round(miss, 2).tolist())}")
    if len(hit) and len(miss):
        print(f"\n  lowest hit  = {hit.min():.2f}    highest miss = {miss.max():.2f}")
        print(f"  SEPARABLE: {hit.min() > miss.max()}")


if __name__ == "__main__":
    main()
