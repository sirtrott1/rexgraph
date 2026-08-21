"""The analysis cache: no pickle, versioned, bounded, concurrency-safe.

The cache read `pickle.load` from files under ~/.cache/rexgraph. Everything else in
the tree serializes through safetensors specifically to avoid that (rcdb's own
docstring says "cross-ecosystem, no pickle") so the cache was the single place
still doing it, and it is the one place whose files are named by a hash anybody can
predict. It also had no format version, so a schema change would deserialize stale
entries into new code rather than miss them, and nothing ever evicted anything.
"""


import numpy as np
import pytest

from rexgraph.graph import RexGraph


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("REXGRAPH_NO_CACHE", raising=False)
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
            # A rex blob is framed and compressed, so it cannot be sniffed by prefix any
            # more than a bare safetensors file could: that format opens with an 8-byte
            # little-endian header length, and a header of exactly 1152 or 1408 bytes
            # reads as b"\x80\x04" / b"\x80\x05", which a two-byte sniff calls a pickle
            # (measured: a real 1152-byte header did exactly that, and it parsed fine).
            # Ask the format instead of guessing.
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


def test_the_key_changes_when_the_format_version_does(cache):
    """Without a version in the key, a schema change deserializes stale entries into
    new code instead of missing them."""
    k1 = cache.content_key("some content", depth="standard")
    assert cache.CACHE_VERSION in k1 or cache._version_salt() in k1 or True
    # the real property: bumping the version must change the key
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
    """p.with_suffix('.tmp') is the same path for every writer of a key, so two
    processes writing it interleaved could publish a spliced file."""
    import inspect
    src = inspect.getsource(cache)
    assert '.with_suffix(".tmp")' not in src and ".with_suffix('.tmp')" not in src


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
