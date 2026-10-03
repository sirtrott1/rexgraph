"""Bounded table I/O through the real Agent training, prediction and fusion paths."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from agent.models import load_mapped_table
from agent.models import data as D, train as T


def table(tmp_path, *, task="classification", labels=True):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) if task == "classification" else X[:, 0] * 2
    xp, yp = tmp_path / "x.npy", tmp_path / "y.npy"
    np.save(xp, X); np.save(yp, y)
    b = load_mapped_table(xp, yp if labels else None, task=task, n_classes=2,
                          eval_batch_size=4)
    return b, X, y


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_mapped_loss_training_and_prediction_match_in_memory(tmp_path, task):
    b, X, y = table(tmp_path, task=task)
    assert isinstance(b.X, np.memmap) and not b.X.flags.writeable
    assert all(isinstance(idx, range) for idx in b.splits.values())
    assert sum(map(len, b.splits.values())) == len(X)
    ordinary = D.DataBundle("vector", torch.from_numpy(X), torch.from_numpy(y), dict(b.meta),
                            {k: torch.tensor(list(v)) for k, v in b.splits.items()})
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2 if task == "classification" else 1)
    other = copy.deepcopy(model)
    r1 = T.train_one(model, b, steps=4, batch=5, lr=.01, device="cpu", seed=91)
    r2 = T.train_one(other, ordinary, steps=4, batch=5, lr=.01, device="cpu", seed=91)
    assert r1 == r2
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for split in (None, "train", "test"):
        p, metric = T.predict_on(model, b, "vector", split)
        q, oracle = T.predict_on(other, ordinary, "vector", split)
        np.testing.assert_allclose(p, q, atol=1e-7)
        assert metric == oracle


def test_mapped_values_are_validated_at_batch_access_without_class_remapping(tmp_path):
    b, X, y = table(tmp_path)
    y[0] = 2
    X[1, 0] = np.inf
    np.save(tmp_path / "x.npy", X); np.save(tmp_path / "y.npy", y)
    b = load_mapped_table(tmp_path / "x.npy", tmp_path / "y.npy", n_classes=2)
    # Loading reads metadata, not every value; bad rows fail when reached.
    with pytest.raises(ValueError, match="class IDs"):
        b.fetch([0])
    with pytest.raises(ValueError, match="features"):
        b.fetch([1])
    xb, yb = b.fetch([2, 3])
    np.testing.assert_array_equal(yb.numpy(), y[[2, 3]])
    assert xb.shape == (2, 3)


@pytest.mark.parametrize("mode", ["ensemble", "split", "stack"])
def test_fusion_and_stages_keep_feature_reads_bounded(tmp_path, mode):
    b, _, _ = table(tmp_path)
    reads = []
    original = b.fetch
    def fetch(indices):
        reads.append(len(indices))
        return original(indices)
    b.fetch = fetch
    # Base trainer defaults to batch64; restrict the split to four rows so this
    # test can assert every read, including meta head epochs, has a four row bound.
    b.splits["train"] = range(4)
    result = T.train_fusion([("mlp", {"d_hid": 4, "depth": 1}),
                             ("mlp", {"d_hid": 4, "depth": 1})],
                            b, mode=mode, steps=1, device="cpu")
    assert np.isfinite(result["fused_metric"])
    assert reads and max(reads) <= 4
    assert isinstance(T._sub_bundle(b, range(2)), D.MappedDataBundle)


def test_mapped_unlabelled_prediction_and_compact_partition(tmp_path):
    b, _, _ = table(tmp_path, labels=False)
    p, metric = T.predict_on(torch.nn.Linear(3, 2), b, "vector")
    assert len(p) == 30 and metric is None
    parts = T._split_indices(range(100, 1000000, 3), 5, seed=1)
    assert all(isinstance(p, range) for p in parts)
    assert sum(map(len, parts)) == len(range(100, 1000000, 3))
    assert len(set(v for p in T._split_indices(range(15), 4, 1) for v in p)) == 15


@pytest.mark.parametrize("options", [{"n_classes": None}, {"eval_batch_size": 0},
                                       {"ratios": (1, 1, 1)}, {"task": "unknown"}])
def test_mapped_loader_rejects_invalid_contract(tmp_path, options):
    table(tmp_path)
    with pytest.raises(ValueError):
        load_mapped_table(tmp_path / "x.npy", tmp_path / "y.npy", **{"n_classes": 2, **options})
