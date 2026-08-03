import numpy as np
import pytest

pytest.importorskip("safetensors")
from agent.rcdb import open_store

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io.safetensors_bridge import safetensors_to_temporal_rex, temporal_rex_to_safetensors


def test_dogfood_temporal_store_through_rcdb(tmp_path):
    # an evolving complex: grow + reweight over 8 steps (a stand in for a live change stream)
    graphs = [RexGraph(sources=np.arange(k, dtype=np.int32),
                       targets=np.arange(1, k + 1, dtype=np.int32),
                       w_E=(np.arange(k, dtype=np.float64) + 1.0)) for k in range(2, 10)]
    trex = TemporalRex([(g.sources, g.targets) for g in graphs])
    trex._ensure_index()

    # delta serialize the temporal store, reload it delta backed, reconstruct any snapshot
    p = tmp_path / "trex.safetensors"
    temporal_rex_to_safetensors(trex, str(p))
    back = safetensors_to_temporal_rex(str(p))

    # store the reconstructed snapshots in a REAL FileStore RCDB and read them back
    store = open_store(f"file://{tmp_path}/rcdb")
    try:
        for t in range(back._T):
            r = back.reconstruct_at(t) if not back._snapshots_materialized else back.at(t)
            rec = store.put("snap@%d" % t, r, meta={"t": t})
            assert rec.id == "snap@%d" % t
        got = store.get("snap@7")
        ref = graphs[7]
        assert set(zip(got.sources.tolist(), got.targets.tolist(), strict=False)) == \
               set(zip(ref.sources.tolist(), ref.targets.tolist(), strict=False))
        assert len(store.list(limit=100)) == back._T
    finally:
        store.close()
