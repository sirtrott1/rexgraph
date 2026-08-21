"""Locating the section a query was lifted from."""
import glob, os, random
import numpy as np
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.partition import section_response

paths = [x for x in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(x) < 130_000][:10]
rng = random.Random(5)

print(f"{'mode':10s} {'true-section rank':>18} {'top-1':>7} {'argmax == longest':>19} "
      f"{'cells/section':>14}")
for mode in ("none", "spanning", "clique"):
    true_ranks, longest_hits, n, cps = [], 0, 0, []
    for p in paths:
        raw, _ = read_document(p)
        rex, info = build_document(raw, profile=ENGLISH_GUTENBERG, pair_mode=mode)
        base = info["base_layer"]
        sect = sectionings_of(rex)[base]
        vocab = {str(v).lower(): i for i, v in enumerate(info["vocab"])}
        owner = np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64)
        ns = len(sect)
        # how many CELLS each section owns: one per span in `none`, k-1 / C(k,2) after
        cells = np.bincount(owner[owner >= 0], minlength=ns).astype(float)
        cps.append(cells.mean())
        biggest = int(np.argmax(cells))
        for _ in range(30):
            i = rng.randrange(ns)
            q = section_text(rex, base, i, raw).strip()
            if not (60 < len(q) < 300):
                continue
            toks = [t for t, _a, _b in tokenize(q, ENGLISH_GUTENBERG)]
            seeds = [vocab[t] for t in toks if t in vocab]
            if not seeds:
                continue
            # PINNED, for the same reason as the construction benchmark: the recorded
            # table is an "rl4" measurement. On 46 identical queries over 10 documents
            # "boundary" matched it exactly (97.8% top-1, 100% top-5, median 1) at
            # 0.1 s against 115.4 s, which is why it is now the default elsewhere.
            sc, _l = section_response(rex, sect, seeds, t=1.0, seed_weight="invdeg",
                                      n_sections=ns, owner=owner, propagator="rl4")
            r = int((sc > sc[i]).sum()) + 1        # rank of the section it came FROM
            true_ranks.append(r); n += 1
            longest_hits += (int(np.argmax(sc)) == biggest)
    tr = np.array(true_ranks)
    print(f"{mode:10s} {'median ' + str(int(np.median(tr))):>18} "
          f"{(tr == 1).mean()*100:6.1f}% {longest_hits/max(n,1)*100:18.1f}% "
          f"{np.mean(cps):14.1f}   n={n}  top-5 {(tr <= 5).mean()*100:.1f}%", flush=True)
