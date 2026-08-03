import pickle

import numpy as np
from agent.coordinator_adapter import _to_type
from agent.warehouse.foundry_tasks import train_one


def _spec():
    # a tiny 2-class hypergraph: 6 binding-nodes, 2 hyperedges
    X = np.random.RandomState(0).standard_normal((6, 8)).astype(np.float32)
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    return {"archetype": "hgnn", "params": {"d_hid": 8, "n_layers": 1},
            "device": "cpu", "save_path": None,
            "he_ptr": np.array([0, 3, 6], np.int32),
            "he_idx": np.array([0, 1, 2, 3, 4, 5], np.int32),
            "X": X, "y": y, "feat_dim": 8, "n_classes": 2,
            "tier": 0, "config_id": "hgnn#0"}


def test_train_one_is_picklable_and_trains():
    spec = _spec()
    pickle.dumps(spec)                       # the whole spec crosses the forkserver boundary
    out = train_one(spec)
    assert out["tier"] == 0 and out["config_id"] == "hgnn#0"
    assert isinstance(out["metric"], float) and np.isfinite(out["metric"])


def test_train_task_lane_mapping():
    assert _to_type("train:hgnn") == "gpu_kernel"     # GPU-capable -> igpu lane
    assert _to_type("train:cnn") == "cpu_coordination"  # CPU-only -> proc lane
