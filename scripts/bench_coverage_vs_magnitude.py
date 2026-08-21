"""Coverage against magnitude as a section reading."""
import glob, os, random, time
import numpy as np
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.core._sparse import to_scipy_csr

ARMS = ("magnitude", "imbalance", "mass", "evenness", "coverage", "overlap")


def per_section(arm, B, absB, Bcoo, owner, ns, x, seedmask):
    if arm == "magnitude":
        resp = np.abs(B @ (B.T @ x))
        out = np.zeros(ns)
        co = owner[Bcoo.col]
        keep = co >= 0
        np.add.at(out, co[keep], resp[Bcoo.row[keep]])
        return out
    g = np.abs(B.T @ x)
    m = absB.T @ x
    if arm == "imbalance":
        v = g
    elif arm == "mass":
        v = m
    elif arm == "evenness":
        v = np.where(m > 0, 1.0 - np.minimum(g / np.maximum(m, 1e-300), 1.0), 0.0)
    elif arm == "coverage":
        v = m - g
    else:
        v = absB.T @ seedmask          # plain count of shared vertices
    out = np.zeros(ns)
    keep = owner >= 0
    np.add.at(out, owner[keep], v[keep])
    return out


paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(p) < 900_000][:10]

for regime in ("full", "partial"):
    res = {a: [] for a in ARMS}
    secs = {a: 0.0 for a in ARMS}
    for p in paths:
        raw, _ = read_document(p)
        rex, info = build_document(raw, profile=ENGLISH_GUTENBERG)
        base = info["base_layer"]
        sect = sectionings_of(rex)[base]
        vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
        owner = np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64)
        ns = len(sect)
        B = to_scipy_csr(rex._B1_dual).tocsr()
        absB = abs(B)
        Bcoo = B.tocoo()
        deg = np.asarray(rex.degree, dtype=np.float64)
        rng = random.Random(31)
        for _ in range(30):
            i = rng.randrange(ns)
            q = section_text(rex, base, i, raw).strip()
            if not (60 < len(q) < 300):
                continue
            toks = [w for w, _a, _b in tokenize(q, ENGLISH_GUTENBERG) if w in vocab]
            if regime == "partial":
                toks = toks[: max(1, len(toks) // 2)]
            seeds = [vocab[w] for w in toks]
            if not seeds:
                continue
            x = np.zeros(int(rex.nV))
            sd = np.asarray(seeds)
            x[sd] = 1.0 / np.maximum(deg[sd], 1.0)
            mask = np.zeros(int(rex.nV))
            mask[sd] = 1.0
            for a in ARMS:
                t = time.perf_counter()
                sc = per_section(a, B, absB, Bcoo, owner, ns, x, mask)
                secs[a] += time.perf_counter() - t
                res[a].append(int((sc > sc[i]).sum()) + 1)
    print(f"\n=== {regime} query ===")
    print(f"{'arm':11s} {'n':>4} {'top-1':>7} {'top-5':>7} {'median':>8} {'total s':>9}")
    for a in ARMS:
        r = np.array(res[a])
        print(f"{a:11s} {len(r):>4} {(r == 1).mean()*100:6.1f}% {(r <= 5).mean()*100:6.1f}% "
              f"{int(np.median(r)):>8} {secs[a]:8.3f}s", flush=True)
