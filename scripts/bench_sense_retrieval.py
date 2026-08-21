"""Sense expansion on the retrieval path."""
import os

import numpy as np

os.environ.setdefault("REXGRAPH_RCDB_URI", "file://" + os.path.expanduser(
    "~/projects/rexgraph/data/corpora/gutenberg/rcdb"))

from agent import rcdb_index as ix
from agent.rcdb import default_store
from agent.senses import inventory, sense_expansion, SenseModel

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


def main():
    snap = default_store()._idx._snap
    ids = list(snap["ids"])
    n = int(snap["n"])
    B = ix.boundary_operator(snap)
    deg = ix._vertex_degree(snap, B)
    codes = ix._term_codes(snap)

    def rank(weights, doc):
        x = np.zeros(B.shape[0])
        hit = False
        for t, w in weights.items():
            c = codes.get(t)
            if c is not None:
                x[c] = max(x[c], w / max(deg[c], 1.0))
                hit = True
        if not hit:
            return None
        r = np.abs(B @ (B.T @ x))[:n]
        o = np.argsort(r)[::-1]
        rk = np.empty(len(o), np.int64)
        rk[o] = np.arange(len(o))
        return int(rk[ids.index(doc)]) + 1

    inventory()          # load once, outside the timing
    syn, by_lemma, rel = inventory()

    print(f"{'query':<38} {'raw':>7} {'blind':>7} {'filt':>7}  fired")
    arms = {"raw": [], "blind": [], "filtered": []}
    fired = 0
    for q, doc in CASES:
        seeds = [w for w in q.lower().replace("'s", " ").split() if w not in STOP]
        raw_w = {t: 1.0 for t in seeds}
        blind = sense_expansion(seeds, blind=True)
        filt = sense_expansion(seeds, blind=False)
        differs = set(blind) != set(filt)
        fired += differs
        a = rank(raw_w, doc)
        b = rank({t: v[0] for t, v in blind.items()}, doc)
        c = rank({t: v[0] for t, v in filt.items()}, doc)
        if None in (a, b, c):
            continue
        arms["raw"].append(a)
        arms["blind"].append(b)
        arms["filtered"].append(c)
        print(f"{q:<38} {a:>7,} {b:>7,} {c:>7,}  {'yes' if differs else 'no'}")

    print()
    for name, v in arms.items():
        v = np.array(v)
        print(f"{name:>9}: top-1 {int((v == 1).sum())}/{len(v)}  "
              f"top-5 {int((v <= 5).sum())}/{len(v)}  "
              f"top-20 {int((v <= 20).sum())}/{len(v)}  median {int(np.median(v))}")
    print(f"\nfiltering DIFFERED from blind on {fired}/{len(CASES)} queries")

    # why: how often does a query term even get disambiguated?
    print("\n--- why: does the query's own context reach a sense's extent? ---")
    tot = amb = decided = 0
    for q, _doc in CASES:
        seeds = [w for w in q.lower().replace("'s", " ").split() if w not in STOP]
        for t in seeds:
            if not by_lemma.get(t):
                continue
            tot += 1
            m = SenseModel.for_word(t, syn, by_lemma, rel, hops=1)
            if m.d < 2:
                continue
            amb += 1
            if not m.disambiguate([s for s in seeds if s != t])["abstain"]:
                decided += 1
    print(f"  query terms in WordNet     : {tot}")
    print(f"  ambiguous (>=2 senses)     : {amb}")
    print(f"  disambiguated (not abstain): {decided}  "
          f"({100*decided/max(amb,1):.0f}% of ambiguous)")


if __name__ == "__main__":
    main()
