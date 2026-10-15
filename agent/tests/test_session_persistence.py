"""Session writes retain previous history on failure and refuse stale updates."""
from pathlib import Path
import shutil

import pytest
from rexgraph import RexGraph
from agent.session import Session, create_session
from agent.server.state import SessionStore


def graph():
    return RexGraph.from_graph([0, 1, 2], [1, 2, 0])


def test_missing_session_lookup_does_not_create_a_directory(tmp_path):
    store = SessionStore(str(tmp_path))
    assert store.get("missing") is None
    assert not (tmp_path / "missing").exists()
    with pytest.raises(FileNotFoundError):
        Session.load("missing", str(tmp_path))
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "/outside", "a/b", "a\\b", "C:drive", "bad\x00id"])
def test_session_ids_cannot_escape_storage(tmp_path, name):
    with pytest.raises(ValueError):
        Session(name, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_session_directory_links_are_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        Session("linked", str(root))
    assert not list(outside.iterdir())


def test_graph_write_failure_is_reported_without_recording_a_snapshot(tmp_path, monkeypatch):
    session = create_session(str(tmp_path))
    def fail(*args, **kwargs):
        raise OSError("synthetic graph write failure")
    monkeypatch.setattr("rexgraph.io.save_rcbd", fail)
    with pytest.raises(OSError, match="graph write"):
        session.add_snapshot(graph(), "upload")
    assert session.snapshots == [] and session.current_step == -1
    assert not (session.session_dir / "session_index.json").exists()


def test_index_write_failure_retains_history_and_allows_retry(tmp_path, monkeypatch):
    session = create_session(str(tmp_path))
    first = graph()
    session.add_snapshot(first, "upload")
    index = session.session_dir / "session_index.json"
    previous = index.read_bytes()
    from rexgraph.io import publication
    replace = publication.os.replace
    def fail_index(source, target):
        if Path(target) == index:
            raise OSError("synthetic index write failure")
        return replace(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(publication.os, "replace", fail_index)
        with pytest.raises(OSError, match="index write"):
            session.add_snapshot(graph(), "analyze")
    assert index.read_bytes() == previous
    assert len(session.snapshots) == 1 and session.current() is first
    restored = Session.load(session.session_id, str(tmp_path))
    assert restored.current().nE == first.nE
    session.add_snapshot(graph(), "analyze")
    assert len(Session.load(session.session_id, str(tmp_path)).snapshots) == 2


def test_a_stale_loaded_session_cannot_overwrite_new_history(tmp_path):
    session = create_session(str(tmp_path))
    session.add_snapshot(graph(), "upload")
    stale = Session.load(session.session_id, str(tmp_path))
    session.add_snapshot(graph(), "analyze")
    with pytest.raises(RuntimeError, match="changed on disk"):
        stale.add_snapshot(graph(), "reconfig")
    assert len(Session.load(session.session_id, str(tmp_path)).snapshots) == 2


def test_session_allocation_does_not_reuse_an_existing_id(tmp_path, monkeypatch):
    occupied = tmp_path / "aaaaaaaa"
    occupied.mkdir()
    marker = occupied / "user-data"
    marker.write_text("preserve")
    values = iter(["aaaaaaaa-0000", "bbbbbbbb-0000"])
    monkeypatch.setattr("agent.session.uuid.uuid4", lambda: next(values))
    session = create_session(str(tmp_path))
    assert session.session_id == "bbbbbbbb"
    assert marker.read_text() == "preserve"


def test_a_moved_session_uses_its_copied_snapshot(tmp_path):
    original = tmp_path / "original"
    session = create_session(str(original))
    session.add_snapshot(graph(), "upload")
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    shutil.rmtree(original)
    restored = Session.load(session.session_id, str(moved))
    assert restored.current().nE == 3


def test_snapshot_parameters_retain_numeric_types(tmp_path):
    import numpy as np
    session = create_session(str(tmp_path))
    session.add_snapshot(graph(), "reconfig", params={"threshold": np.float32(0.25),
                         "selected": np.array([0, 2]), "enabled": np.bool_(True)})
    params = Session.load(session.session_id, str(tmp_path)).snapshots[0].params
    assert params == {"threshold": 0.25, "selected": [0, 2], "enabled": True}
