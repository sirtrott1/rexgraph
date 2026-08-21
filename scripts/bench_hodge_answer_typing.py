"""Typing an answer by its Hodge parts."""
import glob, os, random
import numpy as np
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.partition import section_response
from rexgraph.flow.navigator import flow_step

paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(p) < 250_000][:6]
docs = []
for p in paths:
    raw, _ = read_document(p)
    rex, info = build_document(raw, profile=ENGLISH_GUTENBERG)
    base = info["base_layer"]
    sect = sectionings_of(rex)[base]
    docs.append({"rex": rex, "raw": raw, "base": base, "sect": sect,
                 "owner": np.asarray(sect.owner_cochain(int(rex.nE)), dtype=np.int64),
                 "vocab": {str(v).lower(): i for i, v in enumerate(info["vocab"])},
                 "name": os.path.basename(p)})


def terms_of(d, i):
    return [w for w, _a, _b in
            tokenize(section_text(d["rex"], d["base"], i, d["raw"]).strip(),
                     ENGLISH_GUTENBERG)]


def type_response(d, words, topk=3):
    seeds = [d["vocab"][w] for w in words if w in d["vocab"]]
    if not seeds:
        return None
    ns = len(d["sect"])
    resp, _n = section_response(d["rex"], d["sect"], seeds,
                                n_sections=ns, owner=d["owner"])
    if not resp.size or resp.max() <= 0:
        return None
    top = np.argsort(resp)[::-1][:topk]
    # the ANSWER's support: the relations the responding sections own
    region = np.flatnonzero(np.isin(d["owner"], top))
    if region.size == 0 or region.size == int(d["rex"].nE):
        return None
    out = flow_step(d["rex"], region)
    dr = np.asarray(out["draining"], float); ci = np.asarray(out["circulating"], float)
    tot = float(dr @ dr) + float(ci @ ci)
    if tot <= 0:
        return None
    return float(ci @ ci) / tot, int(region.size)


rng = random.Random(5)
res = {"answerable": [], "foreign": []}
sz = {"answerable": [], "foreign": []}
for di, d in enumerate(docs):
    ns = len(d["sect"])
    for _ in range(8):
        r = type_response(d, terms_of(d, rng.randrange(ns)))
        if r:
            res["answerable"].append(r[0]); sz["answerable"].append(r[1])
        o = docs[rng.choice([k for k in range(len(docs)) if k != di])]
        r = type_response(d, terms_of(o, rng.randrange(len(o["sect"]))))
        if r:
            res["foreign"].append(r[0]); sz["foreign"].append(r[1])
    print(f"  {d['name']:12s} {len(res['answerable'])}/{len(res['foreign'])}", flush=True)

print(f"\n{'condition':12s} {'n':>4} {'harmonic share':>26} {'region':>8}")
for k in ("answerable", "foreign"):
    v = np.array(res[k]); z = np.array(sz[k])
    print(f"{k:12s} {len(v):>4}  median {np.median(v):8.4f}  "
          f"min {v.min():.3f} max {v.max():.3f}   {int(np.median(z)):>6}")
a, f = np.array(res["answerable"]), np.array(res["foreign"])
if len(a) and len(f):
    print(f"\n  answerable above the foreign median: {int((a > np.median(f)).sum())}/{len(a)}")
    print(f"  foreign above the answerable median: {int((f > np.median(a)).sum())}/{len(f)}")
