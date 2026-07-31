"""One registry primitive, five call sites.

The tree grew five near-registries with five different surfaces: io had
register/unregister/available, compute had register plus available, rcdb and
_serialization had register alone, connectors was a bare dict, and the temporal
rerank policies were a hardcoded tuple. Same pattern, five shapes, so extending any
of them meant learning which one you were in.
"""

import pytest

from rexgraph.registry import Registry


def test_register_and_get():
    r = Registry("widget")
    r.register("a", 1)
    assert r.get("a") == 1
    assert "a" in r


def test_available_is_sorted_and_complete():
    r = Registry("widget")
    r.register("b", 2)
    r.register("a", 1)
    assert r.available() == ["a", "b"]


def test_unregister_returns_what_it_removed():
    r = Registry("widget")
    r.register("a", 1)
    assert r.unregister("a") == 1
    assert "a" not in r
    assert r.unregister("a") is None


def test_an_unknown_name_names_what_is_available():
    """A registry that says only 'KeyError' makes the caller go read the source."""
    r = Registry("widget")
    r.register("real", 1)
    with pytest.raises(KeyError) as ei:
        r.require("bogus")
    msg = str(ei.value)
    assert "widget" in msg and "bogus" in msg and "real" in msg


def test_re_registering_replaces():
    r = Registry("widget")
    r.register("a", 1)
    r.register("a", 2)
    assert r.get("a") == 2
    assert r.available() == ["a"]


def test_metadata_rides_along():
    r = Registry("widget")
    r.register("a", 1, extensions=[".a"], kind="cpu")
    assert r.meta("a")["extensions"] == [".a"]
    assert r.meta("a")["kind"] == "cpu"
    assert r.meta("missing") == {}


def test_len_and_iteration():
    r = Registry("widget")
    r.register("a", 1)
    r.register("b", 2)
    assert len(r) == 2
    assert sorted(r) == ["a", "b"]


def test_items_gives_name_and_value():
    r = Registry("widget")
    r.register("a", 1)
    assert list(r.items()) == [("a", 1)]


# --- every call site is backed by it ------------------------------------------

def test_io_formats_use_the_shared_registry():
    from rexgraph import io
    assert isinstance(io._FORMATS, Registry)
    for name in ("rex", "json", "safetensors"):
        assert name in io.available_formats()


def test_compute_backends_use_the_shared_registry():
    from rexgraph import compute
    assert isinstance(compute._BACKENDS, Registry)
    assert compute.available_backends()


def test_compute_backends_can_be_unregistered():
    """compute could register a backend but never take one back, so a test or a
    probe that added one changed the process for good."""
    from rexgraph import compute
    compute.register_backend("probe", available=lambda: True, kind="cpu")
    try:
        assert "probe" in compute.available_backends()
    finally:
        compute.unregister_backend("probe")
    assert "probe" not in compute.available_backends()


def test_rcdb_backends_are_listable_and_removable():
    from agent import rcdb
    rcdb.register_backend("probe", lambda uri: None)
    try:
        assert "probe" in rcdb.available_backends()
    finally:
        rcdb.unregister_backend("probe")
    assert "probe" not in rcdb.available_backends()


def test_serialization_types_are_listable():
    from rexgraph.io import _serialization
    assert _serialization.available_types()


def test_temporal_policies_are_a_registry_not_a_tuple():
    """The holdout. Rerank policies were a fixed tuple, so a domain-specific one --
    pseudotime order, a batch-corrected recency -- meant editing the module."""
    from agent import temporal
    assert set(temporal.available_policies()) >= {"off", "stability", "recency",
                                                  "settled"}


def test_a_temporal_policy_can_be_registered_from_outside():
    from agent import rcdb, temporal

    temporal.register_policy("newest_only", lambda feats, rec, did: 1.0 if rec.get(did, 0) >= 1.0 else 0.0)
    try:
        assert "newest_only" in temporal.available_policies()
        sections = [{"doc_id": "a", "score": 1.0}, {"doc_id": "b", "score": 1.0}]
        out = temporal.rerank(sections, rcdb.MemoryStore(), mode="newest_only")
        assert [s["temporal"]["mode"] for s in out] == ["newest_only"] * 2
    finally:
        temporal.unregister_policy("newest_only")
    assert "newest_only" not in temporal.available_policies()


def test_an_unknown_temporal_policy_still_errors():
    from agent import rcdb, temporal
    with pytest.raises(ValueError):
        temporal.rerank([], rcdb.MemoryStore(), mode="vibes")
