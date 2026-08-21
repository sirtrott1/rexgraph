"""Spread as a retrieval reading."""
import glob, os, random
import numpy as np
import scipy.sparse as sp
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.core._sparse import to_scipy_csr
from rexgraph.rational_trig import spread as exact_spread

#### the vectorised form must BE the library's definition, not a lookalike ####
_rng = np.random.default_rng(0)
for _ in range(200):
    k = int(_rng.integers(2, 7))
    u = _rng.normal(size=k)
    v = _rng.normal(size=k)
    mine = 1.0 - (u @ v) ** 2 / ((u @ u) * (v @ v))
    assert abs(mine - float(exact_spread(u, v))) < 1e-9, "not the library's spread"
print("vectorised spread == rational_trig.spread on 200 random pairs", flush=True)

ARMS = ("magnitude", "coverage", "spread", "spread-mass", "align-mass")


def readings(B, absB, supp, Bcoo, owner, ns, x, Qc):
    out = {}
    resp = np.abs(B @ (B.T @ x))
    a = np.zeros(ns)
    co = owner[Bcoo.col]
    k = co >= 0
    np.add.at(a, co[k], resp[Bcoo.row[k]])
    out["magnitude"] = a

    g = B.T @ x
    m = absB.T @ x
    Qx = supp.T @ (x * x)
    denom = Qx * Qc
    s = np.where(denom > 0, 1.0 - np.minimum((g * g) / np.maximum(denom, 1e-300), 1.0), 0.0)
    ok = owner >= 0
    for name, v in (("coverage", m - np.abs(g)),
                    ("spread", s),
                    ("spread-mass", s * m),
                    ("align-mass", (1.0 - s) * m)):
        o = np.zeros(ns)
        np.add.at(o, owner[ok], v[ok])
        out[name] = o
    return out


paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(p) < 900_000][:25]

for regime in ("full", "partial", "quarter"):
    res = {a: [] for a in ARMS}
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
        supp = sp.csr_matrix((np.ones_like(B.data), B.indices, B.indptr), shape=B.shape)
        Bcoo = B.tocoo()
        Qc = np.asarray(B.multiply(B).sum(axis=0)).ravel()
        deg = np.asarray(rex.degree, dtype=np.float64)
        rng = random.Random(31)
        for _ in range(40):
            i = rng.randrange(ns)
            q = section_text(rex, base, i, raw).strip()
            if not (60 < len(q) < 300):
                continue
            toks = [w for w, _a, _b in tokenize(q, ENGLISH_GUTENBERG) if w in vocab]
            if regime == "partial":
                toks = toks[: max(1, len(toks) // 2)]
            elif regime == "quarter":
                toks = toks[: max(1, len(toks) // 4)]
            if not toks:
                continue
            sd = np.asarray([vocab[w] for w in toks])
            x = np.zeros(int(rex.nV))
            x[sd] = 1.0 / np.maximum(deg[sd], 1.0)
            sc = readings(B, absB, supp, Bcoo, owner, ns, x, Qc)
            for a in ARMS:
                res[a].append(int((sc[a] > sc[a][i]).sum()) + 1)

    n = len(res["magnitude"])
    print(f"\n=== {regime} query ===  n={n}")
    print(f"{'arm':13s} {'top-1':>7} {'top-5':>7} {'median':>8}")
    for a in ARMS:
        r = np.array(res[a])
        print(f"{a:13s} {(r == 1).mean()*100:6.1f}% {(r <= 5).mean()*100:6.1f}% "
              f"{int(np.median(r)):>8}", flush=True)
