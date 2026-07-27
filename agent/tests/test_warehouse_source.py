import numpy as np
from agent.agent.warehouse import source as S


# a tiny inline generic weighted edge list TSV fixture (no pandas, no large file)
FIXTURE = ("src\tx1\tdst\tx2\tw\n"
           "A1\t.\tB1\t.\t0.5\nA1\t.\tB2\t.\t1.0\nA1\t.\tB3\t.\t50.0\n"
           "A2\t.\tB1\t.\t2.0\nA2\t.\tB2\t.\t4.0\n"
           "A3\t.\tB4\t.\t0.2\nA3\t.\tB5\t.\t0.3\nA3\t.\tB6\t.\t900.0\n")


def _fixture(tmp_path):
    p = tmp_path / "e.tsv"; p.write_text(FIXTURE); return str(p)


def test_load_edges_dedup_and_ids(tmp_path):
    ed = S.load_edges(_fixture(tmp_path), source="src", target="dst", weight="w")
    assert ed.n_src == 3
    assert len(ed.src_idx) == 8 and len(ed.weight) == 8
    assert ed.src_idx.min() == 0 and ed.dst_idx.min() >= ed.n_src   # destination nodes indexed after source nodes


def test_load_edges_dedup_drops_duplicate_pairs(tmp_path):
    # A1/B1 appears twice with different w values; keep the FIRST occurrence only.
    dup_fixture = ("src\tx1\tdst\tx2\tw\n"
                   "A1\t.\tB1\t.\t0.5\n"
                   "A1\t.\tB1\t.\t9.0\n"
                   "A2\t.\tB1\t.\t1.0\n")
    p = tmp_path / "dup.tsv"; p.write_text(dup_fixture)
    ed = S.load_edges(str(p), source="src", target="dst", weight="w")
    assert len(ed.src_idx) == 2 and len(ed.weight) == 2   # two distinct (source,destination) pairs
    assert ed.weight[0] == 0.5     # kept weight matches the FIRST row's raw w, not the second


def test_tier_split_partitions_edges(tmp_path):
    ed = S.load_edges(_fixture(tmp_path), source="src", target="dst", weight="w")
    tiers = S.tier_split(ed, n_tiers=3)
    assert isinstance(tiers, list) and len(tiers) == 3
    for tier in tiers:
        assert isinstance(tier, np.ndarray)
        assert np.issubdtype(tier.dtype, np.integer)
    all_idx = np.sort(np.concatenate(tiers))
    assert np.array_equal(all_idx, np.arange(len(ed.src_idx)))   # every edge assigned exactly once


def test_edge_complex_is_edge_primal(tmp_path):
    ed = S.load_edges(_fixture(tmp_path), source="src", target="dst", weight="w")
    rex = S.edge_complex(ed)
    assert rex.nE == 8                       # one edge per row
    assert rex.nV == ed.n_src + ed.n_dst


def test_edge_features_are_per_edge_and_named(tmp_path):
    ed = S.load_edges(_fixture(tmp_path), source="src", target="dst", weight="w")
    rex = S.edge_complex(ed)
    mask = np.arange(ed.src_idx.shape[0])
    X, names = S.edge_features(rex, ed, mask)
    assert X.shape[0] == mask.shape[0]
    assert X.shape[1] == len(names) and X.shape[1] >= 8
    assert np.isfinite(X).all()


def test_diffused_features_vary_with_time(tmp_path):
    # the "signal diffusion in a tensor field" smoke test: larger t spreads the signal
    ed = S.load_edges(_fixture(tmp_path), source="src", target="dst", weight="w")
    rex = S.edge_complex(ed)
    mask = np.arange(ed.src_idx.shape[0])
    X_small, names = S.edge_features(rex, ed, mask, t_scales=(0.01,))
    X_large, _ = S.edge_features(rex, ed, mask, t_scales=(5.0,))
    di = [i for i, n in enumerate(names) if "diffus" in n or "heat" in n or "dirac" in n]
    assert di, "expected diffused-signal feature columns"
    assert not np.allclose(X_small[:, di], X_large[:, di])


def test_hypergraph_bundle_shape(tmp_path):
    ed = S.load_edges(_fixture(tmp_path), source="src", target="dst", weight="w")
    rex = S.edge_complex(ed)
    mask = np.arange(ed.src_idx.shape[0])
    X, _ = S.edge_features(rex, ed, mask)
    y = S.labels(ed, mask)
    b = S.hypergraph_bundle(ed, mask, X, y)
    assert b.kind == "hypergraph"
    assert b.meta["n_nodes"] == mask.shape[0] and b.meta["n_classes"] == 2
    assert b.extra["he_ptr"].shape[0] >= 2      # at least one hyperedge
