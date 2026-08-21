"""The channel profile against a scalar reading."""
import glob, os, random
import numpy as np
from rexgraph.corpus_profile import ENGLISH_GUTENBERG, tokenize
from rexgraph.document import build_document, read_document, section_text
from rexgraph.sectioning import sectionings_of
from rexgraph.partition import section_response

paths = [p for p in sorted(glob.glob(os.path.expanduser(
    '~/projects/rexgraph/data/corpora/gutenberg/texts/*/*.txt')))
    if 60_000 < os.path.getsize(p) < 250_000][:10]
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
    _ = rex.structural_character           # warm the per-document cache
print(f"{len(docs)} documents", flush=True)


def terms(d, i):
    return [w for w, _a, _b in tokenize(
        section_text(d["rex"], d["base"], i, d["raw"]).strip(), ENGLISH_GUTENBERG)]


def top_profile(d, words):
    seeds = [d["vocab"][w] for w in words if w in d["vocab"]]
    if not seeds:
        return None
    out = section_response(d["rex"], d["sect"], seeds, channels=True,
                           n_sections=len(d["sect"]), owner=d["owner"])
    if len(out) != 3:
        return None
    prof, _l, names = out
    tot = prof.sum(axis=1)
    if not tot.size or tot.max() <= 0:
        return None
    j = int(np.argmax(tot))
    return prof[j] / tot[j], names           # DIRECTION: what kind of response


rng = random.Random(5)
rows = {"answerable": [], "foreign": []}
names = None
for di, d in enumerate(docs):
    ns = len(d["sect"])
    for _ in range(12):
        r = top_profile(d, terms(d, rng.randrange(ns)))
        if r:
            rows["answerable"].append(r[0]); names = r[1]
        o = docs[rng.choice([k for k in range(len(docs)) if k != di])]
        r = top_profile(d, terms(o, rng.randrange(len(o["sect"]))))
        if r:
            rows["foreign"].append(r[0]); names = r[1]
    print(f"  {d['name']:12s} {len(rows['answerable'])}/{len(rows['foreign'])}", flush=True)

A = np.array(rows["answerable"]); F = np.array(rows["foreign"])
print(f"\nn: answerable {len(A)}  foreign {len(F)}   channels {names}")
print(f"\n{'channel':18s} {'answerable':>22} {'foreign':>22}  {'separates?':>12}")
for k, nm in enumerate(names):
    a, f = A[:, k], F[:, k]
    overlap = (a > np.median(f)).mean()
    print(f"{nm:18s} med {np.median(a):7.4f} [{a.min():.3f},{a.max():.3f}]  "
          f"med {np.median(f):7.4f} [{f.min():.3f},{f.max():.3f}]  "
          f"{overlap*100:6.1f}% above")

# and the whole DIRECTION, not one axis: how far apart are the two clouds?
from rexgraph.rational_trig import spread
ca, cf = A.mean(axis=0), F.mean(axis=0)
print(f"\n  answerable direction {np.round(ca, 4)}")
print(f"  foreign    direction {np.round(cf, 4)}")
print(f"  spread between them  {float(spread(ca, cf)):.6f}   (0 = same direction)")

# --- can the DIRECTION classify, held out? Leave-one-out, no threshold anywhere:
# each sample is assigned to whichever class centroid it has the smaller SPREAD to,
# and the centroids are computed WITHOUT it.
print("\n=== leave-one-out classification on the profile direction ===")
X = np.vstack([A, F])
y = np.array([0] * len(A) + [1] * len(F))
correct = 0
for i in range(len(X)):
    m = np.ones(len(X), dtype=bool); m[i] = False
    ca = X[m & (y == 0)].mean(axis=0)
    cf = X[m & (y == 1)].mean(axis=0)
    da, df = float(spread(X[i], ca)), float(spread(X[i], cf))
    correct += int((0 if da <= df else 1) == y[i])
print(f"  accuracy {correct/len(X)*100:5.1f}%   (chance {max(len(A),len(F))/len(X)*100:.1f}%)")

# and the control the scalar gives: the SUM over channels, which is what shipped
print("\n=== the same test on the SCALAR the profile sums to ===")
sa, sf = A.sum(axis=1), F.sum(axis=1)     # == 1 by construction for a direction
print(f"  the direction sums to 1 by construction: answerable {np.unique(np.round(sa,6))}")
print(f"  so a scalar built from it carries NOTHING. There is no scalar control here,")
print(f"  which is the point: the information is in the axes, not the magnitude.")
