"""
End-to-end biology-pipeline test through the RexGraph agent.

Mirrors the manual workflow in NEXT_SESSION_BRIEF.md:
  - multiple 10X scRNA-seq datasets (stand-ins for GSE121861/72056/123366)
  - marker-gene cell typing
  - curated ligand-receptor interaction scoring between cell types
  - full relational-complex analysis (Hodge / void / sigma-sweep / ...)
  - TrustGraph ontology enrichment
  - cross-dataset Poincaré-style structural comparison

Uses synthetic-but-structured 10X data (real GEO downloads aren't available
here) with planted cell populations and signaling so every stage has real
structure to find. Everything runs through the actual agent entry points.
"""

import gzip
import os
import tempfile

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

# A compact but realistic gene panel
MARKERS = {
    "T_cell":      ["CD3D", "CD3E", "CD2", "TRAC", "CD8A"],
    "B_cell":      ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Myeloid":     ["LYZ", "CD68", "ITGAM", "CSF1R", "C1QA"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "FLT1"],
    "Fibroblast":  ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB"],
    "Tumor":       ["EPCAM", "KRT8", "KRT18", "KRT19", "MKI67"],
}

# Curated L-R panel (31 pairs; ligand-producing -> receptor-bearing).
LR_PAIRS = [
    ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
    ("VEGFA", "FLT1"), ("VEGFA", "KDR"),
    ("PDGFB", "PDGFRB"), ("PDGFA", "PDGFRA"),
    ("IL6", "IL6R"), ("IL1B", "IL1R1"),
    ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B"),
    ("CXCL12", "CXCR4"), ("CXCL9", "CXCR3"), ("CXCL10", "CXCR3"),
    ("CCL2", "CCR2"), ("CCL5", "CCR5"), ("CCL19", "CCR7"),
    ("EGF", "EGFR"), ("HBEGF", "EGFR"),
    ("WNT5A", "FZD1"), ("DLL4", "NOTCH1"), ("JAG1", "NOTCH2"),
    ("CD274", "PDCD1"), ("PDCD1LG2", "PDCD1"),
    ("LGALS9", "HAVCR2"), ("CD80", "CTLA4"), ("CD86", "CTLA4"),
    ("ICAM1", "ITGAL"), ("SELP", "SELPLG"),
    ("CXCL13", "CXCR5"), ("IL15", "IL2RB"), ("CSF1", "CSF1R"),
]

# Which cell type predominantly *produces* each ligand / *bears* each receptor.
# (Just enough biology to plant a directed signaling structure.)
LIGAND_SOURCE = {
    "TGFB1": "Fibroblast", "VEGFA": "Tumor", "PDGFB": "Endothelial",
    "PDGFA": "Tumor", "IL6": "Myeloid", "IL1B": "Myeloid", "TNF": "Myeloid",
    "CXCL12": "Fibroblast", "CXCL9": "Myeloid", "CXCL10": "Myeloid",
    "CCL2": "Tumor", "CCL5": "T_cell", "CCL19": "Fibroblast",
    "EGF": "Fibroblast", "HBEGF": "Myeloid", "WNT5A": "Fibroblast",
    "DLL4": "Endothelial", "JAG1": "Endothelial", "CD274": "Tumor",
    "PDCD1LG2": "Myeloid", "LGALS9": "Myeloid", "CD80": "Myeloid",
    "CD86": "B_cell", "ICAM1": "Endothelial", "SELP": "Endothelial",
    "CXCL13": "Fibroblast", "IL15": "Myeloid", "CSF1": "Tumor",
}
RECEPTOR_BEARER = {
    "TGFBR1": "T_cell", "TGFBR2": "Endothelial", "FLT1": "Endothelial",
    "KDR": "Endothelial", "PDGFRB": "Fibroblast", "PDGFRA": "Fibroblast",
    "IL6R": "T_cell", "IL1R1": "Fibroblast", "TNFRSF1A": "Endothelial",
    "TNFRSF1B": "T_cell", "CXCR4": "T_cell", "CXCR3": "T_cell",
    "CCR2": "Myeloid", "CCR5": "T_cell", "CCR7": "T_cell", "EGFR": "Tumor",
    "FZD1": "Tumor", "NOTCH1": "Endothelial", "NOTCH2": "Tumor",
    "PDCD1": "T_cell", "HAVCR2": "T_cell", "CTLA4": "T_cell",
    "ITGAL": "T_cell", "SELPLG": "Myeloid", "CXCR5": "B_cell",
    "IL2RB": "T_cell", "CSF1R": "Myeloid",
}


def _all_genes():
    genes = []
    for gs in MARKERS.values():
        genes += gs
    for lig, rec in LR_PAIRS:
        genes += [lig, rec]
    # de-dup, stable order
    seen, out = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g); out.append(g)
    return out


def make_dataset(path, n_cells, composition, seed):
    """Write a synthetic 10X triplet with planted cell types + signaling."""
    rng = np.random.default_rng(seed)
    genes = _all_genes()
    gidx = {g: i for i, g in enumerate(genes)}
    types = list(MARKERS.keys())

    # assign each cell a type by the requested composition
    probs = np.array([composition.get(t, 0.0) for t in types], float)
    probs = probs / probs.sum()
    cell_types = rng.choice(types, size=n_cells, p=probs)

    G = np.zeros((len(genes), n_cells))  # genes x cells (10X convention)
    for ci, ct in enumerate(cell_types):
        # background
        G[:, ci] = rng.poisson(0.2, len(genes))
        # markers for this type high
        for m in MARKERS[ct]:
            G[gidx[m], ci] += rng.poisson(8)
        # ligands this type produces
        for lig, src in LIGAND_SOURCE.items():
            if src == ct and lig in gidx:
                G[gidx[lig], ci] += rng.poisson(6)
        # receptors this type bears
        for rec, bear in RECEPTOR_BEARER.items():
            if bear == ct and rec in gidx:
                G[gidx[rec], ci] += rng.poisson(6)

    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "matrix.mtx")
    mmwrite(tmp, sparse.csr_matrix(G))
    with open(tmp, "rb") as f:
        data = f.read()
    with gzip.open(os.path.join(path, "matrix.mtx.gz"), "wb") as f:
        f.write(data)
    os.remove(tmp)
    with gzip.open(os.path.join(path, "features.tsv.gz"), "wt") as f:
        for g in genes:
            f.write("ENSG_%s\t%s\tGene Expression\n" % (g, g))
    with gzip.open(os.path.join(path, "barcodes.tsv.gz"), "wt") as f:
        for i in range(n_cells):
            f.write("BC%d-1\n" % i)
    return dict(zip(*np.unique(cell_types, return_counts=True)))


def main():
    from agent.auto import auto_rex
    from agent.pipeline import AnalysisPipeline
    from agent.corpus import CorpusBuilder

    root = tempfile.mkdtemp(prefix="bio_")
    # three datasets with different tumor-microenvironment compositions
    datasets = {
        "GSE121861": (dict(T_cell=.3, Myeloid=.25, Tumor=.2, Fibroblast=.1,
                           Endothelial=.1, B_cell=.05), 220, 11),
        "GSE72056":  (dict(T_cell=.4, Tumor=.25, Myeloid=.15, B_cell=.1,
                           Fibroblast=.05, Endothelial=.05), 200, 22),
        "GSE123366": (dict(Tumor=.35, Fibroblast=.2, Myeloid=.2, T_cell=.15,
                           Endothelial=.07, B_cell=.03), 240, 33),
    }

    print("=" * 64)
    print("BIO PIPELINE - end to end through the agent")
    print("=" * 64)
    print("L-R panel: %d pairs | cell types: %d | genes: %d\n"
          % (len(LR_PAIRS), len(MARKERS), len(_all_genes())))

    corpus = CorpusBuilder()
    # With the spectral/quotient void path, the dense L-R complex is now
    # tractable at full depth (~seconds instead of ~3 min), so run the
    # complete analysis on every dataset.
    for name, (comp, n, seed) in datasets.items():
        d = os.path.join(root, name)
        planted = make_dataset(d, n, comp, seed)

        # STEP 1-3: 10X -> cell typing -> L-R complex (auto_rex)
        rex = auto_rex(d, markers=MARKERS, lr_pairs=LR_PAIRS)
        print("[%s] %d cells -> types=%s" % (
            name, n, {k: int(v) for k, v in planted.items()}))
        print("    complex: nV=%d nE=%d nF=%d  betti=%s" % (
            rex.nV, rex.nE, rex.nF, list(rex.betti)))

        # STEP 4: full analysis (Hodge / void / sigma-sweep / ...)
        import time as _t
        _t0 = _t.time()
        res = AnalysisPipeline(rex).run(depth="full")
        elapsed = _t.time() - _t0
        hodge = res.get("hodge", {})
        void = res.get("void", {})
        ss = res.get("sigma_sweep", {})
        print("    Hodge: grad=%.2f curl=%.2f harm=%.2f | κ=%s" % (
            hodge.get("pct_gradient", 0), hodge.get("pct_curl", 0),
            hodge.get("pct_harmonic", 0),
            res.get("relational", {}).get("kappa_mean")))
        if ss.get("available"):
            print("    sigma-sweep: %d pts, strain %.3f->%.3f (min t=%s)" % (
                len(ss.get("sweep", [])), ss.get("strain_norm_max", 0),
                ss.get("strain_norm_min", 0), ss.get("t_min_strain")))
        print("    void[%s]: n_voids=%s (β1) shadow_dim=%s congruence_classes=%s"
              % (void.get("method"), void.get("n_voids"),
                 void.get("shadow_dim"), void.get("congruence_classes")))
        print("    full-depth analysis: %.1fs\n" % elapsed)

        corpus.add_document(source=d, doc_id=name)

    # STEP 5-6: corpus-level enrichment + Poincaré comparison
    print("-" * 64)
    print("CROSS-DATASET (corpus of %d datasets)" % corpus.n_documents)
    # standard depth is now affordable end-to-end (spectral void path);
    # this gives kappa / Hodge invariants alongside the Poincaré matrix.
    corpus.build(depth="standard")

    tg = corpus.trustgraph_analysis(depth="standard")
    print("TrustGraph enrichment: available=%s  triples=%s"
          % (tg.get("available"), tg.get("n_triples")))

    cmp = corpus.cross_dataset_comparison(metric="bottleneck")
    print("Poincaré cross-dataset distance matrix (bottleneck):")
    ids = cmp["doc_ids"]
    print("    " + "  ".join("%10s" % i[:10] for i in ids))
    for i, row in enumerate(cmp["distance_matrix"]):
        print("    %10s " % ids[i][:10]
              + "  ".join("%10.3f" % v for v in row))
    print("\ninvariants per dataset:")
    for inv in cmp["invariants"]:
        print("    %s: nV=%d nE=%d nF=%d betti=%s kappa=%s"
              % (inv["doc_id"], inv["nV"], inv["nE"], inv["nF"],
                 inv["betti"], inv["kappa_mean"]))

    print("\nOK - bio pipeline ran end to end through the agent.")


if __name__ == "__main__":
    main()
