"""Agreement between two section readings."""
import glob, os, random
import numpy as np
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.core._sparse import to_scipy_csr

paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(p) < 900_000][:25]

for regime in ("full", "partial", "quarter"):
    rows = []
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
        co = owner[Bcoo.col]
        kmask = co >= 0
        deg = np.asarray(rex.degree, dtype=np.float64)
        okeep = owner >= 0
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

            resp = np.abs(B @ (B.T @ x))
            a = np.zeros(ns)
            np.add.at(a, co[kmask], resp[Bcoo.row[kmask]])

            v = (absB.T @ x) - np.abs(B.T @ x)
            b = np.zeros(ns)
            np.add.at(b, owner[okeep], v[okeep])

            om = np.argsort(a)[::-1]
            rows.append({
                "truth": i,
                "top_mag": int(om[0]),
                "second_mag": int(om[1]) if ns > 1 else int(om[0]),
                "top_cov": int(np.argmax(b)),
                "rank_mag": int((a > a[i]).sum()) + 1,
            })

    n = len(rows)
    agree = [r for r in rows if r["top_mag"] == r["top_cov"]]
    dis = [r for r in rows if r["top_mag"] != r["top_cov"]]

    def hit(rs, key):
        return sum(1 for r in rs if r[key] == r["truth"])

    def hit2(rs, k1, k2):
        return sum(1 for r in rs if r["truth"] in (r[k1], r[k2]))

    print(f"\n=== {regime} query ===  n={n}")
    print(f"  magnitude top-1 overall      {hit(rows,'top_mag')/n*100:5.1f}%")
    print(f"  coverage  top-1 overall      {hit(rows,'top_cov')/n*100:5.1f}%")
    print(f"  they AGREE on {len(agree)}/{n} ({len(agree)/n*100:.0f}%)")
    if agree:
        print(f"    P(magnitude correct | agree)    {hit(agree,'top_mag')/len(agree)*100:5.1f}%")
    if dis:
        print(f"    P(magnitude correct | disagree) {hit(dis,'top_mag')/len(dis)*100:5.1f}%")
        print(f"    on disagreement, coverage's pick is right "
              f"{hit(dis,'top_cov')/len(dis)*100:5.1f}%")
    print(" : two guesses, and the control --")
    print(f"    {{top_mag, top_cov}}   {hit2(rows,'top_mag','top_cov')/n*100:5.1f}%")
    print(f"    magnitude top-2      {hit2(rows,'top_mag','second_mag')/n*100:5.1f}%  <- control")
