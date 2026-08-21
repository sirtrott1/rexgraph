"""Retrieval across construction arms."""
import glob, os, time
import numpy as np

CORPUS = os.path.expanduser('~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')
N_DOCS, N_QUERIES_PER_DOC, SEED = 50, 4, 20260818


def pool():
    fs = [x for x in sorted(glob.glob(CORPUS)) if 60_000 < os.path.getsize(x) < 130_000]
    return fs[:N_DOCS]


def queries(paths):
    import random
    from rexgraph.corpus_profile import ENGLISH_GUTENBERG
    from rexgraph.document import build_document, read_document, section_text
    rng = random.Random(SEED)
    out = []
    for di, p in enumerate(paths):
        raw, _ = read_document(p)
        rex, info = build_document(raw, profile=ENGLISH_GUTENBERG, pair_mode="none")
        base = info["base_layer"]
        n = info["n_spans"] if base == "span" else info["n_sentences"]
        got, tries = 0, 0
        while got < N_QUERIES_PER_DOC and tries < 400:
            tries += 1
            i = rng.randrange(n)
            q = section_text(rex, base, i, raw).strip()
            if 60 < len(q) < 300:
                out.append((di, q)); got += 1
    return out


def build_arm(paths, mode):
    from rexgraph.corpus_profile import ENGLISH_GUTENBERG
    from rexgraph.document import build_document, read_document
    from rexgraph.sectioning import sectionings_of
    docs = []
    for p in paths:
        raw, _ = read_document(p)
        rex, info = build_document(raw, profile=ENGLISH_GUTENBERG, pair_mode=mode)
        base = info["base_layer"]
        sect = sectionings_of(rex)[base]
        vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
        owner = np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64)
        docs.append({"rex": rex, "sect": sect, "vocab": vocab, "owner": owner,
                     "n_sections": len(sect)})
    return docs


def score_query(docs, q_tokens):
    from rexgraph.partition import section_response
    n = len(docs)
    best, total, spread = np.zeros(n), np.zeros(n), np.zeros(n)
    ret = np.zeros(n)
    for j, d in enumerate(docs):
        seeds = [d["vocab"][t] for t in q_tokens if t in d["vocab"]]
        if not seeds:
            continue
        # SEED RETURN, indexed by VERTICES, which are identical in all three arms. The
        # section readout can be biased by construction: under clique expansion a
        # section owns C(k,2) cells instead of one, so its summed response grows with
        # span length squared. This one cannot be: it is the diffused field read back at
        # the query's own terms, and the query's terms are the same vertices whatever
        # the relations between them are.
        rex = d["rex"]
        ind = np.zeros(int(rex.nV))
        deg = np.asarray(rex.degree, dtype=np.float64)
        sd = np.asarray(seeds, dtype=int)
        ind[sd] = 1.0 / np.maximum(deg[sd], 1.0)
        resp = np.abs(np.asarray(rex.propagate_signal(ind, mode="heat", t=1.0),
                                 dtype=np.float64).ravel())
        ret[j] = float(resp[sd].sum())
        # PINNED to the propagator these recorded numbers were measured under. The
        # default has since moved to "boundary", which was measured equal on section
        # localisation and is 1154x cheaper; pinning keeps the table below meaning what
        # it says rather than silently becoming a different experiment.
        sc, _lab = section_response(d["rex"], d["sect"], seeds, t=1.0,
                                    seed_weight="invdeg", n_sections=d["n_sections"],
                                    owner=d["owner"], propagator="rl4")
        if sc.size:
            tot = float(sc.sum())
            best[j] = float(sc.max()); total[j] = tot
            # SPREAD = peak/total. Magnitude is not comparable across documents: they
            # differ in size and in degree, but concentration is. The true source has
            # the passage in ONE section, so its response concentrates; elsewhere the
            # same words are scattered over the book. This is the model's own `spread`,
            # a ratio of two exact sums, not a normalisation chosen to make it work.
            spread[j] = (best[j] / tot) if tot > 0 else 0.0
    return best, total, spread, ret


def score_slice(mode, lo, hi):
    from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
    paths = pool()
    qs = queries(paths)
    docs = build_arm(paths[lo:hi], mode)
    nb = np.zeros((len(qs), hi - lo)); nt = np.zeros_like(nb); ns = np.zeros_like(nb)
    nr = np.zeros_like(nb)
    for i, (_di, q) in enumerate(qs):
        toks = [t for t, _a, _b in tokenize(q, ENGLISH_GUTENBERG)]
        b, t_, sp, rr = score_query(docs, toks)
        nb[i], nt[i], ns[i], nr[i] = b, t_, sp, rr
    return lo, hi, nb, nt, ns, nr


def cost(mode, lo, hi):
    from rexgraph.core._sparse import to_scipy_csr
    paths = pool()[lo:hi]
    t0 = time.perf_counter()
    docs = build_arm(paths, mode)
    tb = time.perf_counter() - t0
    return {"nE": sum(int(d["rex"].nE) for d in docs),
            "nnz": sum(to_scipy_csr(d["rex"]._B1_dual).nnz for d in docs),
            "build_s": tb}


if __name__ == "__main__":
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    paths = pool()
    qs = queries(paths)
    print(f"{len(paths)} documents, {len(qs)} queries, chance = "
          f"{100/len(paths):.1f}%", flush=True)
    W = 10                                   # slices of documents, not of queries
    edges = np.linspace(0, len(paths), W + 1).astype(int)
    slices = [(int(edges[i]), int(edges[i + 1])) for i in range(W)
              if edges[i + 1] > edges[i]]
    truth = np.array([di for di, _q in qs])
    ctx = multiprocessing.get_context("forkserver")
    results = {}
    for mode in ("none", "spanning", "clique"):
        t0 = time.perf_counter()
        B = np.zeros((len(qs), len(paths))); T = np.zeros_like(B); S = np.zeros_like(B)
        R = np.zeros_like(B)
        with ProcessPoolExecutor(max_workers=W, mp_context=ctx) as ex:
            for lo, hi, nb, nt, ns, nr in ex.map(
                    score_slice, [mode] * len(slices), [a for a, _ in slices],
                    [b for _, b in slices]):
                B[:, lo:hi], T[:, lo:hi], S[:, lo:hi], R[:, lo:hi] = nb, nt, ns, nr
            cs = list(ex.map(cost, [mode] * len(slices), [a for a, _ in slices],
                             [b for _, b in slices]))
        c = {k: sum(x[k] for x in cs) for k in ("nE", "nnz", "build_s")}

        def ranks(M):
            own = M[np.arange(len(qs)), truth][:, None]
            return (M > own).sum(axis=1) + 1

        rb, rt, rs, rr = ranks(B), ranks(T), ranks(S), ranks(R)
        results[mode] = (rb, rt, rs, c, time.perf_counter() - t0, rr)
        print(f"\n=== {mode} ===  {len(qs)} queries x {len(paths)} documents in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  cost: nE {c['nE']:,}  nnz {c['nnz']:,}  build {c['build_s']:.1f}s",
              flush=True)
        for nm, r in (("best-section", rb), ("seed-return", rr), ("total", rt),
                      ("spread pk/tot", rs)):
            print(f"  {nm:13s} top-1 {int((r == 1).sum()):>4}/{len(r)} "
                  f"({(r == 1).mean()*100:5.1f}%)   top-5 {(r <= 5).mean()*100:5.1f}%   "
                  f"median rank {int(np.median(r))}", flush=True)
    base_nE = results["none"][3]["nE"]
    for which, ix in (("best-section", 0), ("seed-return (vertex-indexed)", 5)):
        print(f"\n=== summary: {which} ===")
        print(f"{'mode':10s} {'top-1':>8} {'top-5':>8} {'med rank':>9} {'nE':>10} "
              f"{'nnz':>10} {'x none':>8} {'score s':>9}")
        for mode in ("none", "spanning", "clique"):
            r = results[mode][ix]; c = results[mode][3]; dt = results[mode][4]
            print(f"{mode:10s} {(r == 1).mean()*100:7.1f}% {(r <= 5).mean()*100:7.1f}% "
                  f"{int(np.median(r)):>9} {c['nE']:>10,} {c['nnz']:>10,} "
                  f"{c['nE']/base_nE:>7.1f}x {dt:>8.0f}s")
