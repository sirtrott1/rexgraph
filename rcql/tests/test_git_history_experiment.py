"""History experiment retains primary identities and states through real stores."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "experiment_git_history.py"
    spec = importlib.util.spec_from_file_location("history_experiment", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.STDOUT)
    git("init", "-b", "main")
    git("config", "user.name", "fixture")
    git("config", "user.email", "fixture@example.invalid")
    def commit(names):
        for name in names:
            target = path / name
            target.write_text((target.read_text() if target.exists() else "") + "x")
        git("add", "--", *names) if names else None
        git("commit", "--allow-empty", "-m", "fixture")
    commit(["one"])
    commit(["one", "two", " leading and trailing ", "line\nbreak", "quote\"name"])
    commit(["one", "two"])
    commit(["one", "two"])
    commit([])
    git("mv", "one", "renamed")
    git("commit", "-m", "rename")
    return path


def test_literal_paths_witnesses_parallel_ids_and_empty_commits(repo):
    tool = module()
    data = tool.history(repo)
    files, cells = tool.projection(data)
    assert len(data["commits"]) == 6
    assert len(cells) == 5
    assert " leading and trailing " in files and "line\nbreak" in files and 'quote"name' in files
    assert len(cells[0]["support"]) == 1
    assert cells[2]["support"] == cells[3]["support"]
    assert cells[2]["relation_id"] != cells[3]["relation_id"]
    assert data["commits"][-1]["paths"] == ["one", "renamed"]
    _, filtered = tool.projection(data, 4)
    assert all(len(c["support"]) == 1 for c in filtered)
    assert [c["relation_id"] for c in filtered] == [c["relation_id"] for c in cells]
    assert tool.snapshot(files, cells[:1]).nV == len(files)


def test_real_store_reopen_and_all_identified_transition_channels(repo, tmp_path):
    tool = module()
    output = tmp_path / "report"
    report = tool.run(repo, output, stride=1)
    assert report["witnesses"] == 1
    assert report["retained_relations"] == 5
    assert len(report["transitions"]) == 4
    assert all(t["births"] == 1 for t in report["transitions"])
    assert all(s["diagonal_T_equals_G_exact"] for s in report["snapshots"])
    assert report["clocks"]["as_of_before_ingestion_absent"] is True
    assert json.loads((output / "report.json").read_text())["input_revision"] == report["input_revision"]
    with pytest.raises(FileExistsError):
        tool.run(repo, output)


def test_betti_zero_is_not_branching_connectivity_and_diagonal_is_not_full_operator():
    from fractions import Fraction
    import numpy as np
    from rexgraph.graph import RexGraph
    from rexgraph.channel_operator import channel_operator
    tool = module()
    rex = RexGraph.from_cells([3, [[0, 1, 2]]])
    result = tool.measure(rex)
    assert result["betti0"] == 2
    assert result["boundary_quadrance_by_arity"][3] == "3/2"
    parallel = RexGraph.from_cells([3, [[0, 1, 2], [1, 0, 2]]])
    t, g = (channel_operator(parallel, key) for key in ("T", "G"))
    assert np.array_equal(t.diagonal(exact=True), g.diagonal(exact=True))
    v = np.array([Fraction(1), Fraction(0)], dtype=object)
    assert not np.array_equal(t.apply(v, exact=True), g.apply(v, exact=True))


def test_merge_uses_first_parent_difference(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT)
    git("checkout", "-b", "side")
    (repo / "side file").write_text("side")
    git("add", "--", "side file")
    git("commit", "-m", "side")
    git("checkout", "main")
    (repo / "main file").write_text("main")
    git("add", "--", "main file")
    git("commit", "-m", "main")
    git("merge", "--no-ff", "side", "-m", "merge")
    data = module().history(repo)
    assert len(data["commits"][-1]["parents"]) == 2
    assert data["commits"][-1]["paths"] == ["side file"]
    assert len(data["commits"]) == 8


@pytest.mark.parametrize("future", [False, True])
def test_git_clock_ties_and_reversals_do_not_reorder_identity_steps(repo, tmp_path, monkeypatch, future):
    tool = module()
    data = tool.history(repo)
    for commit, timestamp in zip(data["commits"], (30, 10, 10, 20, 25, 5), strict=True):
        commit["time"] = timestamp + (4102444800 if future else 0)
    monkeypatch.setattr(tool, "history", lambda *args: data)
    report = tool.run(repo, tmp_path / "report", stride=1)
    assert [s["valid_time"] for s in report["snapshots"]] == [c["time"] for c in data["commits"] if c["paths"]]
    assert report["clocks"]["as_of_before_ingestion_absent"] is (None if future else True)
    assert [t["step"] for t in report["transitions"]] == [1, 2, 3, 4]


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_history_bounds_fail_before_output(repo, tmp_path, value):
    tool = module()
    with pytest.raises(ValueError):
        tool.history(repo, limit=value)
    with pytest.raises(ValueError):
        tool.projection({"commits": []}, value)
    with pytest.raises(ValueError):
        tool.run(repo, tmp_path / "absent", stride=value)
    assert not (tmp_path / "absent").exists()
