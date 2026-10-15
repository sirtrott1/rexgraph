"""Check cache serialization, retention, filesystem failures and concurrent writes."""


import numpy as np
import pytest

from rexgraph.graph import RexGraph


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("REXGRAPH_NO_CACHE", raising=False)
    monkeypatch.delenv("REXGRAPH_CACHE_MAX_ENTRIES", raising=False)
    import importlib

    from agent import cache as mod
    importlib.reload(mod)
    return mod


def _rex():
    r = RexGraph(sources=np.array([0, 1, 2], np.int32),
                 targets=np.array([1, 2, 0], np.int32))
    r._agent_meta = {"vertex_labels": ["a", "b", "c"], "source_text": "hello"}
    return r


def test_nothing_on_disk_is_a_pickle(cache, tmp_path):
    cache.store_rex_and_analysis("k", _rex(), {"topology": {"betti": [1, 1]}},
                                 {"vertex_labels": ["a", "b", "c"]})
    written = list(tmp_path.rglob("*"))
    assert written, "nothing was written"
    for p in written:
        if not p.is_file():
            continue
        assert p.suffix != ".pkl", f"{p.name} is a pickle"
        if p.suffix == ".rexblob":
            # A safetensors header length can begin with pickle protocol bytes.
            # Validate the declared format rather than identifying it from those bytes.
            from agent.rcdb import deserialize_complex
            assert deserialize_complex(p.read_bytes()) is not None
            continue
        head = p.read_bytes()[:2]
        assert head not in (b"\x80\x04", b"\x80\x05"), f"{p.name} has a pickle header"


def test_the_rex_round_trips_with_its_metadata(cache):
    rex = _rex()
    cache.store_rex_and_analysis("k", rex, {"ok": 1}, dict(rex._agent_meta))
    back, analysis, meta = cache.get_rex_and_analysis("k")
    assert back is not None
    assert int(back.nE) == int(rex.nE)
    assert (back._agent_meta or {})["vertex_labels"] == ["a", "b", "c"]
    assert (back._agent_meta or {})["source_text"] == "hello"
    assert analysis == {"ok": 1}
    assert meta["vertex_labels"] == ["a", "b", "c"]


def test_a_nested_analysis_dict_survives(cache):
    analysis = {"construction": {"n": 3}, "topology": {"betti": [1, 1, 0]},
                "spectral": {"eigenvalues_L0": [0.0, 1.5, 3.0]}}
    cache.store_rex_and_analysis("k", _rex(), analysis, {})
    _, back, _ = cache.get_rex_and_analysis("k")
    assert back == analysis


def test_a_miss_is_a_miss_not_an_exception(cache):
    assert cache.get_rex_and_analysis("never-written") == (None, None, None)


@pytest.mark.parametrize("text", ["a" * 10000, "text\x00content", "a directory"])
def test_literal_text_and_its_encoded_bytes_have_the_same_key(cache, tmp_path, text):
    if text == "a directory":
        text = str(tmp_path)
    assert cache.content_key(text) == cache.content_key(text.encode())


def test_the_key_changes_when_the_format_version_does(cache):
    """A changed format version invalidates keys for the same input."""
    k1 = cache.content_key("some content", depth="standard")
    original = cache.CACHE_VERSION
    try:
        cache.CACHE_VERSION = original + "-next"
        k2 = cache.content_key("some content", depth="standard")
    finally:
        cache.CACHE_VERSION = original
    assert k1 != k2, "the cache key ignores the format version"


def test_depth_still_separates_entries(cache):
    assert cache.content_key("x", depth="quick") != cache.content_key("x", depth="standard")


def test_a_corrupt_entry_is_a_miss_not_a_crash(cache, tmp_path):
    cache.store_rex_and_analysis("k", _rex(), {"ok": 1}, {})
    for p in tmp_path.rglob("*"):
        if p.is_file():
            p.write_bytes(b"garbage")
    assert cache.get_rex_and_analysis("k") == (None, None, None)


def test_a_half_written_entry_is_a_miss(cache, tmp_path):
    """Two files means the pair can tear. The sidecar is written last, so its
    absence has to read as a miss rather than a rex with no analysis."""
    cache.store_rex_and_analysis("k", _rex(), {"ok": 1}, {})
    for p in tmp_path.rglob("*.json"):
        p.unlink()
    assert cache.get_rex_and_analysis("k") == (None, None, None)


def test_concurrent_writers_do_not_share_a_temp_path(cache, tmp_path):
    """Concurrent writes within one process each publish a complete JSON payload."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from unittest.mock import patch

    barrier = Barrier(4)
    replace = cache.os.replace
    temporary_names = []

    def synchronized_replace(source, target):
        temporary_names.append(source)
        barrier.wait(timeout=10)
        replace(source, target)

    with patch.object(cache.os, "replace", synchronized_replace), ThreadPoolExecutor(4) as pool:
        assert all(pool.map(lambda i: cache.set("shared", {"i": i}), range(4)))
    assert len(set(temporary_names)) == 4
    assert cache.get("shared") in [{"i": i} for i in range(4)]
    assert not list(tmp_path.glob("*.tmp"))


def test_the_cache_can_be_pruned(cache, tmp_path):
    for i in range(6):
        cache.store_rex_and_analysis(f"k{i}", _rex(), {"i": i}, {})
    assert cache.entry_count() == 6
    removed = cache.prune(max_entries=2)
    assert removed >= 4
    assert cache.entry_count() <= 2


def test_pruning_keeps_the_most_recent(cache):
    import time
    for i in range(4):
        cache.store_rex_and_analysis(f"k{i}", _rex(), {"i": i}, {})
        time.sleep(0.01)
    cache.prune(max_entries=1)
    _, analysis, _ = cache.get_rex_and_analysis("k3")
    assert analysis == {"i": 3}, "pruning discarded the newest entry"


def test_disabling_the_cache_is_honoured(cache, monkeypatch):
    monkeypatch.setenv("REXGRAPH_NO_CACHE", "1")
    assert cache.store_rex_and_analysis("k", _rex(), {"ok": 1}, {}) is False
    assert cache.get_rex_and_analysis("k") == (None, None, None)


def test_cache_miss_does_not_create_a_directory(cache, tmp_path, monkeypatch):
    directory = tmp_path / "absent"
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(directory))
    assert cache.get("missing") is None
    assert cache.get_rex_and_analysis("missing") == (None, None, None)
    assert cache.entry_count() == 0
    assert cache.prune() == 0
    assert not directory.exists()


def test_unavailable_cache_directory_is_a_miss(cache, tmp_path, monkeypatch):
    file = tmp_path / "file"
    file.write_text("not a directory")
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(file))
    assert cache.get("k") is None
    assert cache.get_rex_and_analysis("k") == (None, None, None)
    assert not cache.set("k", {"ok": 1})
    assert not cache.store_rex_and_analysis("k", _rex(), {}, {})


def test_mismatched_blob_and_sidecar_are_a_miss(cache):
    assert cache.store_rex_and_analysis("k", _rex(), {"first": True}, {})
    first_sidecar = cache._side_path("k").read_bytes()
    changed = RexGraph(sources=np.array([0], np.int32), targets=np.array([1], np.int32))
    assert cache.store_rex_and_analysis("k", changed, {"first": False}, {})
    cache._side_path("k").write_bytes(first_sidecar)
    assert cache.get_rex_and_analysis("k") == (None, None, None)


def test_failed_publication_removes_temporary_file(cache, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(cache.os, "replace", fail)
    assert not cache.set("k", {"ok": True})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("key", ["../outside", "/absolute", "a/b", "a\\b", "C:outside", "", "..", "a\x00b"])
def test_invalid_keys_do_not_access_other_paths(cache, key):
    assert cache.get(key) is None
    assert cache.get_rex_and_analysis(key) == (None, None, None)
    assert not cache.set(key, {})
    assert not cache.store_rex_and_analysis(key, _rex(), {}, {})


def test_pruning_keeps_entry_types_separate_and_removes_dotted_key_blobs(cache, tmp_path):
    assert cache.set("same", {"generic": True})
    assert cache.store_rex_and_analysis("same", _rex(), {}, {})
    assert cache.store_rex_and_analysis("dotted.key", _rex(), {}, {})
    (tmp_path / "server.pid").write_text('{"pid": 1}')
    (tmp_path / "config.json").write_text('{}')
    models = tmp_path / "models"
    models.mkdir()
    (models / "weights.safetensors").write_bytes(b"synthetic")
    assert cache.entry_count() == 3
    assert cache.prune(0) == 5
    assert not list(tmp_path.glob("*.rexblob"))
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "server.pid").exists()
    assert (models / "weights.safetensors").exists()


def test_key_parameters_have_unambiguous_boundaries(cache):
    assert cache.content_key(b"x", depth="a|b", extra="c") != cache.content_key(b"x", depth="a", extra="b|c")


def test_invalid_entry_limit_uses_default(cache, monkeypatch):
    import importlib
    monkeypatch.setenv("REXGRAPH_CACHE_MAX_ENTRIES", "invalid")
    importlib.reload(cache)
    assert cache.MAX_ENTRIES == 2000


def test_zero_entry_limit_disables_cache(cache, monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 0)
    assert not cache.set("k", {})
    assert not cache.enabled()


def test_first_write_in_a_new_process_enforces_retention(cache, monkeypatch):
    import importlib
    for i in range(4):
        assert cache.set(f"old{i}", {"i": i})
    assert cache.entry_count() == 4
    with monkeypatch.context() as settings:
        settings.setenv("REXGRAPH_CACHE_MAX_ENTRIES", "2")
        importlib.reload(cache)
        assert cache.set("new", {"new": True})
        assert cache.entry_count() == 2
        assert cache.get("new") == {"new": True}
    importlib.reload(cache)
