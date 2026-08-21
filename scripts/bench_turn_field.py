"""The conversation as a complex."""
import glob, os, random
import numpy as np
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.partition import section_response
from rexgraph.core._sparse import to_scipy_csr
from agent.turn_field import TurnField

paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(p) < 250_000][:12]
docs = []
for p in paths:
    raw, _ = read_document(p)
    rex, info = build_document(raw, profile=ENGLISH_GUTENBERG)
    base = info["base_layer"]
    sect = sectionings_of(rex)[base]
    docs.append({"rex": rex, "raw": raw, "base": base, "sect": sect, "name": os.path.basename(p),
                 "owner": np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64),
                 "vocab": {str(v).lower(): i for i, v in enumerate(info["vocab"])}})
print(f"{len(docs)} documents, chance = {100/len(docs):.1f}%", flush=True)


def sec_terms(d, i):
    return [w for w, _a, _b in
            tokenize(section_text(d["rex"], d["base"], i, d["raw"]).strip(),
                     ENGLISH_GUTENBERG)]


def doc_score(d, words, conv_w=None):
    idx, w = [], []
    for k, t in enumerate(words):
        j = d["vocab"].get(t)
        if j is not None:
            idx.append(j)
            w.append(1.0 if conv_w is None else float(conv_w[k]))
    if not idx:
        return 0.0
    if conv_w is None:
        sc, _n = section_response(d["rex"], d["sect"], idx,
                                  n_sections=len(d["sect"]), owner=d["owner"])
        return float(sc.max()) if sc.size else 0.0
    rex = d["rex"]
    B = to_scipy_csr(rex._B1_dual).tocsr()
    deg = np.asarray(rex.degree, dtype=np.float64)
    x = np.zeros(int(rex.nV))
    ii = np.asarray(idx)
    x[ii] = np.asarray(w) / np.maximum(deg[ii], 1.0)
    resp = np.abs(B @ (B.T @ x))
    out = np.zeros(len(d["sect"]))
    C = B.tocoo()
    co = d["owner"][C.col]
    keep = co >= 0
    np.add.at(out, co[keep], resp[C.row[keep]])
    return float(out.max()) if out.size else 0.0


def rank_of(target, words, conv_w=None):
    s = [doc_score(d, words, conv_w) for d in docs]
    return int((np.asarray(s) > s[target]).sum()) + 1


rng = random.Random(17)
ranks_a, ranks_c, ranks_w, ranks_1 = [], [], [], []
gate_followup, gate_change = [], []

for di, d in enumerate(docs):
    ns = len(d["sect"])
    for _ in range(30):
        i1 = rng.randrange(ns)
        t1 = sec_terms(d, i1)
        if not (5 < len(t1) < 80):
            continue
        i2 = rng.randrange(ns)
        t2 = sec_terms(d, i2)[:4]                      # the thin follow-up
        if len(t2) < 3:
            continue
        oj = rng.choice([k for k in range(len(docs)) if k != di])
        t3 = sec_terms(docs[oj], rng.randrange(len(docs[oj]["sect"])))[:4]
        if len(t3) < 3:
            continue

        tf = TurnField()
        tf.observe(" ".join(t1), profile=ENGLISH_GUTENBERG)
        o2 = tf.observe(" ".join(t2), profile=ENGLISH_GUTENBERG)
        # the gate needs a baseline before it can fire at all (warmup=3), so give the
        # conversation a real length: several more on-topic turns, THEN the change
        gt = TurnField()
        gt.observe(" ".join(t1), profile=ENGLISH_GUTENBERG)
        fired_follow = False
        for _k in range(5):
            tk = sec_terms(d, rng.randrange(ns))[:4]
            if len(tk) < 3:
                continue
            fired_follow |= bool(gt.observe(" ".join(tk),
                                            profile=ENGLISH_GUTENBERG)["event"])
        o3 = gt.observe(" ".join(t3), profile=ENGLISH_GUTENBERG)
        gate_followup.append(fired_follow)
        gate_change.append(bool(o3["event"]))

        ranks_1.append(rank_of(di, t1))                 # DIAGNOSTIC: does turn 1 work?
        ra = rank_of(di, t2)                            # turn 2, alone
        rc = rank_of(di, o2["seeds"])                   # carried, UNWEIGHTED
        rw = rank_of(di, o2["seeds"], o2["weights"])    # carried, conversation-weighted
        ranks_a.append(ra); ranks_c.append(rc); ranks_w.append(rw)

n = len(ranks_a)
ra, rc, rw = np.array(ranks_a), np.array(ranks_c), np.array(ranks_w)
print(f"\n=== CARRY ===  n={n}")
print(f"  follow-up alone     top-1 {(ra == 1).mean()*100:5.1f}%  "
      f"top-3 {(ra <= 3).mean()*100:5.1f}%  median rank {int(np.median(ra))}")
print(f"  follow-up + path    top-1 {(rc == 1).mean()*100:5.1f}%  "
      f"top-3 {(rc <= 3).mean()*100:5.1f}%  median rank {int(np.median(rc))}")
print(f"  follow-up + path,   top-1 {(rw == 1).mean()*100:5.1f}%  "
      f"top-3 {(rw <= 3).mean()*100:5.1f}%  median rank {int(np.median(rw))}   <- WEIGHTED")
print(f"  unweighted vs alone: better {int((rc < ra).sum())} same {int((rc == ra).sum())}"
      f" worse {int((rc > ra).sum())}")
print(f"  WEIGHTED   vs alone: better {int((rw < ra).sum())} same {int((rw == ra).sum())}"
      f" worse {int((rw > ra).sum())}")
r1 = np.array(ranks_1)
print(f"\n  DIAGNOSTIC: turn 1 alone (the opening question, a full section):")
print(f"    top-1 {(r1 == 1).mean()*100:5.1f}%  top-3 {(r1 <= 3).mean()*100:5.1f}%  "
      f"median rank {int(np.median(r1))}")
print(f"    if turn 1 does not identify the document, carrying it CANNOT help turn 2")
print(f"\n=== GATE ===  a follow-up should be quiet, a topic change an event")
print(f"  fired on the follow-up   {sum(gate_followup)}/{len(gate_followup)}")
print(f"  fired on the topic change {sum(gate_change)}/{len(gate_change)}")
