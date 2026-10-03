"""Assembly retires its generated losing outputs without touching other runs."""
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest

from agent.warehouse import source as S
from rcdb import MemoryStore

A = importlib.import_module("agent.warehouse.assemble")


@pytest.fixture
def assembly(monkeypatch):
    import agent.coordinator_adapter as adapter
    import rexgraph.coordinator as coordinator

    ed = S.EdgeData(np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3]), np.arange(1., 5.), 2, 2)
    state = SimpleNamespace(specs=[], produce=None, run=0)

    class Pools:
        def __init__(self, *args): pass
        def shutdown(self): pass
        def run(self, units, placement, **kwargs):
            state.run += 1
            specs = [unit["fn"].args[0] for unit in units]
            state.specs.append(specs)
            return state.produce(specs, state.run)

    class Coordinator:
        def __init__(self, pools): self.pools = pools; self.cost = None
        def plan(self, units): return None

    monkeypatch.setattr(S, "load_edges", lambda *a, **kw: ed)
    monkeypatch.setattr(S, "edge_features", lambda *a: (np.ones((4, 2)), ["a", "b"]))
    monkeypatch.setattr(S, "hypergraph_bundle", lambda *a: SimpleNamespace(extra={"he_ptr": np.array([0]), "he_idx": np.array([], dtype=int)}))
    monkeypatch.setattr(coordinator, "LanePools", Pools)
    monkeypatch.setattr(coordinator, "Coordinator", Coordinator)
    monkeypatch.setattr(adapter, "work_units", lambda tasks: tasks)

    def invoke(save_dir, *, store=None, sweep=None):
        if store is not None:
            return A.assemble("fixture", store=store, hive=SimpleNamespace(add_model=lambda *a, **kw: None),
                save_dir=save_dir, n_tiers=1, sweep=sweep or [{"archetype": "hgnn"}] * 3)
        with closing(MemoryStore()) as memory:
            return invoke(save_dir, store=memory, sweep=sweep)

    state.invoke = invoke
    return state


def outputs(specs, run=1, *, directory=True):
    result = {}
    for i, spec in enumerate(specs):
        path = Path(spec["save_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            path.mkdir(exist_ok=True)
            (path / "weights.safetensors").write_text(f"run {run}, candidate {i}")
            (path / "config.json").write_text("{}")
        else:
            path.write_text(f"run {run}, candidate {i}")
        result[spec["config_id"]] = {"tier": spec["tier"], "config_id": spec["config_id"],
            "archetype": spec["archetype"], "device": "cpu", "saved": str(path), "metric": float(3-i)}
    return result


@pytest.mark.parametrize("directory", [True, False])
def test_only_winning_checkpoint_survives(assembly, tmp_path, directory):
    assembly.produce = lambda specs, run: outputs(specs, run, directory=directory)
    report = assembly.invoke(tmp_path)
    winner = Path(report["tiers"][0]["best"]["saved"])
    assert winner.exists()
    assert all(not Path(s["save_path"]).exists() for s in assembly.specs[0][1:])


def test_partial_failed_checkpoint_is_removed(assembly, tmp_path):
    def produce(specs, run):
        result = outputs(specs)
        result[specs[1]["config_id"]].update(saved=None, metric=None, error="failed after saving")
        return result
    assembly.produce = produce
    assembly.invoke(tmp_path)
    assert not Path(assembly.specs[0][1]["save_path"]).exists()


def test_unreported_partial_outputs_are_removed(assembly, tmp_path):
    def produce(specs, run):
        outputs(specs)
        return {}  # coordinator isolated every failed task
    assembly.produce = produce
    report = assembly.invoke(tmp_path)
    assert report["tiers"][0]["best"] is None
    assert all(not Path(s["save_path"]).exists() for s in assembly.specs[0])


def test_worker_report_cannot_target_an_unowned_file(assembly, tmp_path):
    foreign = tmp_path / "other-model.pt"
    foreign.write_text("keep this model")
    def produce(specs, run):
        result = outputs(specs)
        result[specs[1]["config_id"]]["saved"] = str(foreign)
        return result
    assembly.produce = produce
    assembly.invoke(tmp_path / "outputs")
    assert foreign.read_text() == "keep this model"
    assert not Path(assembly.specs[0][1]["save_path"]).exists()


def test_runs_under_same_parent_do_not_overwrite_prior_winner(assembly, tmp_path):
    assembly.produce = outputs
    first = assembly.invoke(tmp_path)
    winner = Path(first["tiers"][0]["best"]["saved"])
    expected = (winner / "weights.safetensors").read_bytes()
    second = assembly.invoke(tmp_path)
    assert winner != Path(second["tiers"][0]["best"]["saved"])
    assert (winner / "weights.safetensors").read_bytes() == expected


def test_cleanup_error_does_not_prevent_other_losers_being_removed(assembly, monkeypatch, tmp_path):
    import shutil
    original_remove, original_tree = A.os.remove, shutil.rmtree
    def refuse_file(path, *a, **kw):
        if str(path) == assembly.specs[0][1]["save_path"]:
            raise PermissionError("fixture refuses this candidate")
        return original_remove(path, *a, **kw)
    def refuse_tree(path, *a, **kw):
        if str(path) == assembly.specs[0][1]["save_path"]:
            raise PermissionError("fixture refuses this candidate")
        return original_tree(path, *a, **kw)
    monkeypatch.setattr(A.os, "remove", refuse_file)
    monkeypatch.setattr(shutil, "rmtree", refuse_tree)
    assembly.produce = outputs
    assembly.invoke(tmp_path)
    assert Path(assembly.specs[0][1]["save_path"]).exists()
    assert not Path(assembly.specs[0][2]["save_path"]).exists()


def test_nested_and_top_level_symlinks_do_not_remove_their_targets(assembly, tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep").write_text("keep")
    def produce(specs, run):
        result = outputs(specs[:2])
        (Path(specs[1]["save_path"]) / "outside").symlink_to(foreign, target_is_directory=True)
        Path(specs[2]["save_path"]).symlink_to(foreign, target_is_directory=True)
        result[specs[2]["config_id"]] = dict(result[specs[1]["config_id"]],
            config_id=specs[2]["config_id"], saved=specs[2]["save_path"], metric=0.)
        return result
    assembly.produce = produce
    assembly.invoke(tmp_path / "outputs")
    assert (foreign / "keep").read_text() == "keep"
    assert all(not Path(s["save_path"]).is_symlink() and not Path(s["save_path"]).exists()
               for s in assembly.specs[0][1:])


def test_model_names_do_not_become_output_paths(assembly, tmp_path):
    assembly.produce = outputs
    assembly.invoke(tmp_path / "outputs", sweep=[{"archetype": "custom/model"}, {"archetype": "hgnn"}])
    parents = {Path(spec["save_path"]).parent for spec in assembly.specs[0]}
    assert len(parents) == 1


def test_publication_failure_preserves_outputs_for_recovery(assembly, tmp_path):
    class RefusingStore:
        def put(self, *a, **kw): raise OSError("fixture publication failed")
    assembly.produce = outputs
    with pytest.raises(OSError, match="publication failed"):
        assembly.invoke(tmp_path, store=RefusingStore())
    assert all(Path(s["save_path"]).exists() for s in assembly.specs[0])


@pytest.mark.parametrize("symlink", [False, True])
def test_replaced_run_directory_is_not_cleaned(assembly, tmp_path, symlink):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    foreign_paths = []
    def produce(specs, run):
        result = outputs(specs)
        root = Path(specs[0]["save_path"]).parent
        root.rename(tmp_path / "original-run")
        if symlink:
            root.symlink_to(foreign, target_is_directory=True)
        else:
            root.mkdir()
        for spec in specs[1:]:
            unrelated = root / Path(spec["save_path"]).name
            unrelated.write_text("unowned replacement")
            foreign_paths.append(unrelated)
        return result
    assembly.produce = produce
    assembly.invoke(tmp_path / "outputs")
    assert all(path.read_text() == "unowned replacement" for path in foreign_paths)


def test_all_failed_run_removes_its_empty_output_directory(assembly, tmp_path):
    def produce(specs, run):
        outputs(specs)
        return {}
    assembly.produce = produce
    assembly.invoke(tmp_path / "outputs")
    assert list((tmp_path / "outputs").iterdir()) == []


def test_no_trainable_tier_allocates_no_output_directory(assembly, monkeypatch, tmp_path):
    monkeypatch.setattr(S, "tier_split", lambda *a: [np.array([0])])
    assembly.produce = lambda specs, run: {}
    report = assembly.invoke(tmp_path / "missing-parent")
    assert report == {"tiers": [], "store_ids": []}
    assert not (tmp_path / "missing-parent").exists()
