"""Declared relation identities and row domains survive the HGNN projection."""
import numpy as np
import pytest

from agent.knowledge import Knowledge
from agent.warehouse import source as S


def knowledge(edges):
    entities = dict.fromkeys(v for s, _, t, _ in edges for v in (s, t))
    return Knowledge({v: [v] for v in entities}, edges, {}, {})


def groups(bundle):
    ptr, idx = bundle.extra["he_ptr"], bundle.extra["he_idx"]
    return [idx[a:b].tolist() for a, b in zip(ptr[:-1], ptr[1:], strict=True)]


def build_model(bundle):
    pytest.importorskip("torch")
    from agent.models.archetypes import get
    spec = get("hgnn")
    cfg = dict(spec["defaults"], d_hid=4, n_layers=1)
    return spec["build"](cfg, bundle)


def test_uniform_weight_target_retains_both_declared_classes():
    torch = pytest.importorskip("torch")
    k = knowledge([("a", "r", "b", "p"), ("a", "r", "c", "p")])
    b = S.knowledge_bundle(k, target="weight", weight_by="uniform")
    logits = build_model(b)(b.X)
    loss = torch.nn.functional.cross_entropy(logits, b.y)
    loss.backward()
    assert torch.isfinite(loss)
    assert b.y.tolist() == [1, 1]
    assert logits.shape == (2, len(b.meta["classes"])) == (2, 2)


def test_single_space_chain_participates_at_the_shared_entity():
    pytest.importorskip("torch")
    k = knowledge([("a", "r", "b", "p"), ("b", "r", "c", "p")])
    b = S.knowledge_bundle(k)
    assert groups(b) == [[0, 1]]
    assert build_model(b)(b.X).shape == (2, 1)


def test_bipartite_columns_keep_distinct_vertex_identities(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "edges.tsv"
    path.write_text("src\tdst\tw\na\tb\t1\nb\tc\t2\n")
    ed = S.load_edges(path, source="src", target="dst", weight="w")
    b = S.hypergraph_bundle(ed, np.arange(2), np.ones((2, 1)), np.array([0, 1]))
    assert groups(b) == []  # source b and destination b occupy disjoint domains


@pytest.mark.parametrize("src,dst", [([0, 0, 3], [1, 2, 4]), ([0, 2, 4], [1, 3, 5])])
def test_hgnn_preserves_isolated_relation_rows(src, dst):
    torch = pytest.importorskip("torch")
    ed = S.EdgeData(np.array(src), np.array(dst), np.ones(3), 6, 0)
    b = S.hypergraph_bundle(ed, np.arange(3), np.ones((3, 2)), np.array([0, 1, 0]))
    model = build_model(b)
    logits = model(b.X)
    assert model.B1.shape[0] == b.meta["n_nodes"] == 3
    assert logits.shape == (3, 2)
    loss = torch.nn.functional.cross_entropy(logits, b.y)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mask", [np.array([1, 0]), np.array([0, 0]), np.array([-1]),
    np.array([2]), np.array([0.0]), np.ma.array([0, 1], mask=[False, True])])
def test_bundle_and_labels_refuse_ambiguous_tier_rows(mask):
    ed = S.EdgeData(np.array([0, 0]), np.array([1, 2]), np.ones(2), 1, 2)
    with pytest.raises(ValueError, match="tier indices"):
        S.labels(ed, mask)
    with pytest.raises(ValueError, match="tier indices"):
        S.hypergraph_bundle(ed, mask, np.ones((len(mask), 2)), np.zeros(len(mask), dtype=int))


@pytest.mark.parametrize("X,y", [
    (np.ones((1, 2)), np.array([0, 1])),
    (np.ones((2, 2)), np.array([0])),
    (np.ones((2, 2)), np.array([0.5, 1.5])),
    (np.ones((2, 2)), np.array([-1, 1])),
    (np.ones((2, 2)), np.array([0, 2])),
    (np.array([[np.inf], [1.0]]), np.array([0, 1])),
    (np.array([[1e100], [1.0]]), np.array([0, 1])),
    (np.ma.array([[1.0], [2.0]], mask=[[False], [True]]), np.array([0, 1])),
    (np.ones((2, 2)), np.ma.array([0, 1], mask=[False, True])),
])
def test_bundle_refuses_misaligned_or_lossy_training_values(X, y):
    ed = S.EdgeData(np.array([0, 0]), np.array([1, 2]), np.ones(2), 1, 2)
    with pytest.raises(ValueError):
        S.hypergraph_bundle(ed, np.arange(2), X, y)


def test_hgnn_refuses_a_declared_node_domain_that_disagrees_with_features():
    pytest.importorskip("torch")
    ed = S.EdgeData(np.array([0, 0]), np.array([1, 2]), np.ones(2), 1, 2)
    b = S.hypergraph_bundle(ed, np.arange(2), np.ones((2, 2)), np.array([0, 1]))
    b.meta["n_nodes"] = 3
    with pytest.raises(ValueError, match="node count"):
        build_model(b)


def test_isolated_row_domain_survives_checkpoint_reload(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from agent.models.archetypes import get
    from agent.models.store import load_checkpoint, save_checkpoint
    ed = S.EdgeData(np.array([0, 0, 3]), np.array([1, 2, 4]), np.ones(3), 5, 0)
    b = S.hypergraph_bundle(ed, np.arange(3), np.ones((3, 2)), np.array([0, 1, 0]))
    cfg = dict(get("hgnn")["defaults"], d_hid=4, n_layers=1)
    model = get("hgnn")["build"](cfg, b)
    save_checkpoint(tmp_path / "model", model, "hgnn", cfg, bundle=b)
    restored, _ = load_checkpoint(tmp_path / "model", device="cpu")
    torch.testing.assert_close(restored(b.X), model(b.X))
    assert restored.B1.shape[0] == 3


def test_isolated_rows_survive_the_training_worker():
    pytest.importorskip("torch")
    from agent.warehouse.foundry_tasks import train_one
    out = train_one({"archetype": "hgnn", "params": {"d_hid": 4, "n_layers": 1},
        "device": "cpu", "save_path": None, "steps": 2, "tier": 0, "config_id": "isolates",
        "he_ptr": np.array([0], dtype=np.int32), "he_idx": np.array([], dtype=np.int32),
        "X": np.ones((6, 2), dtype=np.float32), "y": np.array([0, 1, 0, 1, 0, 1]),
        "feat_dim": 2, "n_classes": 2})
    assert out.get("error") is None
    assert np.isfinite(out["metric"])
