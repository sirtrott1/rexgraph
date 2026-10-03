"""Custom model training and I/O honor their declared shapes and task settings."""
import numpy as np
import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def hypergraph_bundle():
    from agent.models.data import DataBundle
    return DataBundle("hypergraph", np.ones((4, 2), np.float32), np.array([0, 1, 0, 1]),
        {"n_nodes": 4, "feat_dim": 2, "n_classes": 2},
        extra={"he_ptr": np.array([0, 2], np.int32), "he_idx": np.array([1, 0], np.int32)})


@pytest.mark.parametrize("sink", ["bundle", "rcdb"])
def test_hypergraph_export_keeps_isolated_nodes_and_participant_order(sink, tmp_path):
    from agent.models import store
    from agent.rcdb import open_store
    bundle = hypergraph_bundle()
    if sink == "bundle":
        store.save_complex_rex(bundle, tmp_path / "g.rcbd")
        graph = store.rio.load_rcbd(tmp_path / "g.rcbd")
    else:
        uri = f"sqlite:///{tmp_path / 'catalog.db'}"
        store.to_rcdb(bundle, uri, name="g")
        graph = open_store(uri).get("g")
    assert graph.nV == 4
    assert graph.boundary_idx.tolist() == [1, 0]


def test_sql_model_loader_reads_beyond_first_batch(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    from agent.models import store
    uri = f"sqlite:///{tmp_path / 'training.db'}"
    engine = store.rio.get_engine(uri)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE samples (feature REAL, label INTEGER)"))
        conn.execute(sa.text("INSERT INTO samples VALUES (:feature, :label)"),
                     [{"feature": float(i), "label": i % 2} for i in range(100005)])
    bundle = store.load_bundle(uri, table="samples")
    assert bundle.X.shape == (100005, 1)
    assert bundle.y.shape == (100005,)
    assert bundle.X[-1, 0] == 100004


def test_fractional_regression_labels_survive_table_loading(torch, tmp_path):
    from agent import models
    path = tmp_path / "regression.csv"
    path.write_text("x,label\n1,0.25\n2,1.75\n3,-0.5\n4,2.5\n5,3.25\n")
    model, _, bundle = models.build("mlp", params={"task": "regression"}, data=path)
    np.testing.assert_array_equal(bundle.y.cpu(), [.25, 1.75, -.5, 2.5, 3.25])
    from agent.models.train import _forward_loss
    loss, out = _forward_loss(model, bundle, bundle.splits["train"], "vector")
    expected = torch.nn.functional.mse_loss(out.squeeze(-1), bundle.y[bundle.splits["train"]].float())
    torch.testing.assert_close(loss, expected)


def test_fractional_class_labels_are_rejected_instead_of_truncated(tmp_path):
    from agent.models.store import load_bundle
    path = tmp_path / "classification.csv"
    path.write_text("x,label\n1,0.5\n2,1.75\n")
    with pytest.raises(ValueError, match="class"):
        load_bundle(path)


def test_regression_run_uses_mse_and_restores_configured_input_shape(torch, tmp_path):
    from agent import models
    result = models.run("mlp", params={"task": "regression", "feat_dim": 3, "d_hid": 4},
                        steps=2, device="cpu", save_to=tmp_path / "regression")
    assert result["metric_name"] == "-test MSE" and np.isfinite(result["final"])
    predicted = models.predict(tmp_path / "regression")
    assert predicted["predictions"].dtype.kind == "f"
    assert predicted["metric"] <= 0 and predicted["n"] == 800


def test_checkpoint_default_prediction_uses_saved_archetype_parameters(torch, tmp_path):
    from agent import models
    models.run("mlp", params={"feat_dim": 3, "n_classes": 2, "d_hid": 4}, steps=1,
               save_to=tmp_path / "classification")
    assert models.predict(tmp_path / "classification")["n"] == 800


def test_hgnn_flow_false_disables_cross_grade_parameters(torch):
    from agent import models
    from agent.models.data import _as, make_splits
    bundle = hypergraph_bundle()
    bundle.X, bundle.y = _as(bundle.X), _as(bundle.y)
    bundle.splits = make_splits(4)
    model, _, _ = models.build("hgnn", params={"flow": False, "d_hid": 4, "n_layers": 1}, data=bundle)
    model(bundle.X).sum().backward()
    layer = model.layers[0]
    assert layer.Wd.weight.grad is not None
    assert layer.Wg.weight.grad is None and layer.Wc.weight.grad is None
    assert layer.Wv.weight.grad is None and layer.log_t1.grad is None


def test_multistep_does_not_remove_caller_bundle_from_stage(torch):
    from agent import models
    from agent.models import train
    model, _, bundle = models.build("mlp", params={"d_hid": 4})
    stage = {"bundle": bundle, "steps": 1}
    train.train_multistep(model, bundle, [stage], device="cpu")
    assert stage["bundle"] is bundle and stage["steps"] == 1


@pytest.mark.parametrize("raises", [False, True])
def test_prediction_restores_individual_module_training_modes(torch, raises):
    from agent.models import train
    from agent.models.data import DataBundle
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
        def forward(self, x):
            if raises:
                raise RuntimeError("fixture inference failure")
            return self.layer(x)
    model = Predictor().eval()
    model.layer.train()
    before = [m.training for m in model.modules()]
    bundle = DataBundle("vector", torch.ones(2, 2))
    if raises:
        with pytest.raises(RuntimeError, match="fixture"):
            train.predict_on(model, bundle, "vector")
    else:
        train.predict_on(model, bundle, "vector")
    assert [m.training for m in model.modules()] == before


@pytest.mark.parametrize("mode", ["ensemble", "split", "stack"])
def test_regression_fusion_keeps_real_predictions_and_mse_objective(torch, mode):
    from agent.models.data import synth_vectors
    from agent.models.train import train_fusion
    bundle = synth_vectors(n=20, feat_dim=2, task="regression")
    result = train_fusion([("mlp", {"task": "regression", "d_hid": 4})] * 2,
                          bundle, mode=mode, steps=1, device="cpu")
    assert result["metric_name"] == "-test MSE"
    assert np.isfinite(result["fused_metric"]) and result["fused_metric"] <= 0


def test_sequence_stacking_trains_over_the_vocabulary_axis(torch):
    from agent.models.data import synth_sequences
    from agent.models.train import train_fusion
    bundle = synth_sequences(n=20, vocab=4, seq_len=3, period=2)
    result = train_fusion([("lm", {"d": 4, "n_head": 1, "n_layer": 1})],
                          bundle, mode="stack", steps=1, device="cpu")
    assert np.isfinite(result["fused_metric"])


def test_nonfinite_training_loss_is_rejected_before_parameter_update(torch):
    from agent.models.data import DataBundle
    from agent.models.train import train_one
    model = torch.nn.Linear(2, 2)
    before = [p.detach().clone() for p in model.parameters()]
    bundle = DataBundle("vector", torch.full((3, 2), float("nan")), torch.zeros(3, dtype=torch.long),
                        splits={"train": torch.tensor([0, 1]), "test": torch.tensor([2])})
    with pytest.raises(ArithmeticError, match="nonfinite"):
        train_one(model, bundle, steps=1, device="cpu")
    for parameter, saved in zip(model.parameters(), before, strict=True):
        assert torch.equal(parameter, saved)


def test_unlabelled_vector_corpus_does_not_invent_ground_truth(tmp_path):
    pytest.importorskip("safetensors")
    from agent.models import store
    path = tmp_path / "unlabelled.safetensors"
    store.rio.save_vectors(np.ones((6, 2), np.float32), None, str(path))
    bundle = store.load_bundle(path)
    assert bundle.y is None


def test_numeric_string_vector_labels_keep_their_class_indices(tmp_path):
    pytest.importorskip("safetensors")
    from agent.models import store
    path = tmp_path / "labelled.safetensors"
    store.rio.save_vectors(np.ones((6, 2), np.float32), np.array(["0", "1"] * 3), str(path))
    bundle = store.load_bundle(path)
    assert bundle.y.tolist() == [0, 1] * 3


@pytest.mark.parametrize("bad", [np.array([1e100]), np.array([np.nan])])
def test_vector_loader_rejects_nonfinite_or_unrepresentable_features(bad, tmp_path):
    from agent.models import store
    path = tmp_path / "bad.npz"
    np.savez(path, X=bad.reshape(1, 1), y=np.array([0]))
    with pytest.raises(ValueError, match="features"):
        store.load_bundle(path)
