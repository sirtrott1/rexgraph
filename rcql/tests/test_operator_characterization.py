"""Check direct storage, catalog, metadata and RCDB operator behavior.

These cases call registered adapters directly, including catalog entry errors
and cached content hashes.
"""

from __future__ import annotations

import pathlib

import pytest

from rcql.operators import _REGISTRY

rcdb = pytest.importorskip("rcdb")
pytest.importorskip("safetensors")


def _op(name):
    return _REGISTRY[name].fn


@pytest.fixture
def rex():
    from rexgraph.graph import RexGraph

    return RexGraph.from_graph(sources=[0, 1, 2], targets=[1, 2, 0])


@pytest.fixture
def catalog(tmp_path):
    """A catalog indexes loadable kinds only, which is narrower than its name suggests."""
    import numpy  # a safetensors bundle IS a numpy binary, which is where numpy belongs
    from rexgraph.io.catalog import FileCatalog
    from safetensors.numpy import save_file

    save_file({"w": numpy.arange(6, dtype=numpy.float32).reshape(2, 3)},
              str(tmp_path / "m.safetensors"))
    (tmp_path / "notes.txt").write_text("alpha beta gamma", encoding="utf-8")
    cat = FileCatalog([tmp_path])
    cat.refresh()
    return cat


@pytest.fixture
def store(tmp_path, rex):
    s = rcdb.open_store(f"rex://{tmp_path / 'store'}", read_only=False)
    s.put("r1", rex, meta={"note": "first"}, tags=["t"])
    yield s
    s.close()


# source binding


def test_rex_returns_the_bound_name_rather_than_a_typed_complex(rex):
    """REX is a name passthrough today; the executor resolves it separately.

    The native contract makes REX(name) yield a typed Rex carrying source state and policy.
    Until then the operator cannot be asked anything about the complex it names.
    """
    assert _op("REX")(rex, "graph_a") == "graph_a"
    assert isinstance(_op("REX")(rex, "graph_a"), str)


# file catalog


def test_the_catalog_indexes_loadable_kinds_only(catalog):
    """A plain text file beside an indexed tensor file is simply not in the catalog."""
    names = [entry.name for entry in _op("FILES")(catalog)]
    assert any(name.endswith("m.safetensors") for name in names)
    assert not any(name.endswith("notes.txt") for name in names)


def test_file_info_and_hash_raise_for_an_unindexed_path_that_exists(catalog):
    """Direct catalog adapters raise KeyError for existing but unindexed paths."""
    for name in ("FILE_INFO", "FILE_HASH", "TENSORS"):
        with pytest.raises(KeyError):
            _op(name)(catalog, "notes.txt")


def test_catalog_readings_are_bounded_and_structural(catalog):
    entry = _op("FILES")(catalog)[0]
    # FILE_INFO returns a CatalogEntry dataclass rather than a mapping, so a caller
    # rendering it as JSON converts that dataclass.
    info = _op("FILE_INFO")(catalog, entry.name)
    assert info.kind == "safetensors"
    assert _op("HASH_FILES")(catalog) == 1
    assert _op("SEARCH")(catalog, "nothing-matches-this") == []
    assert isinstance(_op("TENSORS")(catalog, entry.name), list)
    assert isinstance(_op("SEARCH_TENSORS")(catalog, entry.name, "w"), list)


def test_file_info_is_not_idempotent_and_depends_on_what_ran_before_it(catalog):
    """FILE_HASH populates a cached entry's digest for subsequent FILE_INFO calls."""
    name = _op("FILES")(catalog)[0].name

    assert _op("FILE_INFO")(catalog, name).sha256 is None
    digest = _op("FILE_HASH")(catalog, name)
    assert _op("FILE_INFO")(catalog, name).sha256 == digest


def test_state_hash_takes_a_complex_and_not_a_catalog(rex, catalog):
    """STATE_HASH sits among the catalog operators but is a reading of a Rex.

    It answers object_digest(source), so it belongs with the persistence readings. Passing
    the catalog it is filed next to fails inside the digest rather than at the boundary.
    """
    digest = _op("STATE_HASH")(rex)
    assert isinstance(digest, str) and len(digest) == 64

    with pytest.raises(AttributeError):
        _op("STATE_HASH")(catalog)


# RCDB readings


def test_rcdb_list_and_history_project_structure_without_the_complex(store):
    """List and history return metadata views; neither decodes a stored complex."""
    listed = _op("RCDB_LIST")(store)
    assert [row["id"] for row in listed] == ["r1"]
    assert listed[0]["version"] == 1
    assert "signature" in listed[0]
    assert "rex" not in listed[0]

    history = _op("RCDB_HISTORY")(store, "r1")
    assert [row["version"] for row in history] == [1]


def test_rcdb_get_returns_the_decoded_complex(store, rex):
    got = _op("RCDB_GET")(store, "r1")
    assert int(got.nV) == int(rex.nV)
    assert int(got.nE) == int(rex.nE)


def test_rcdb_hash_and_verify_and_stats_and_security(store):
    assert isinstance(_op("RCDB_HASH")(store, "r1"), str)
    assert _op("RCDB_VERIFY")(store, "r1") is True

    stats = _op("RCDB_STATS")(store)
    assert stats["backend"] == "rex"
    assert stats["n_records"] == 1

    security = _op("RCDB_SECURITY")(store)
    assert "payload_encryption" in security
    assert not any("key" in str(value).lower() for value in security.values()), (
        "the security reading must stay free of key material"
    )


def test_rcdb_commits_is_empty_for_a_plain_put(store):
    """A put is not a governed transition, so it contributes no commit link."""
    assert _op("RCDB_COMMITS")(store, "r1") == []


def test_rcdb_search_matches_nothing_without_an_index(store):
    assert _op("RCDB_SEARCH")(store, "r1") == []


def test_rcdb_state_hash_uses_the_provider_logical_identity_and_reports_its_version(store):
    """The once refused name now has a real RCDB owned contract, not an adapter hash."""
    from fractions import Fraction
    from rcql.execution_trace import capture_methods
    store.put("exact", store.get("r1"), meta={"rational": Fraction(1, 7), "tuple": (1, 2)},
              analytics=False)
    with capture_methods() as methods:
        actual = _op("RCDB_STATE_HASH")(store)
    assert actual == store.state_digest()
    assert methods == [{"method": "rcdb-logical-state-digest",
                        "manifest_version": store.logical_state_version, "state_digest": actual}]
    assert store.logical_state_version == store.state_manifest()["version"] == 2


# catalogue shape


def test_every_storage_operator_is_characterized_here():
    """Nothing this file owns can enter the registry without a characterization.

    The check is deliberately one sided. The Rex mathematics operators have their own
    direct suite, so asserting an exhaustive registry equality here would fail on every
    addition there and teach whoever hits it to widen a set without reading. What must not drift is the other direction: an operator in the storage,
    catalog or metadata group that no test in this file exercises.
    """
    storage = {
        "FILES", "SEARCH", "FILE_INFO", "FILE_HASH", "HASH_FILES", "TENSORS",
        "SEARCH_TENSORS", "STATE_HASH", "RCDB_LIST", "RCDB_SEARCH", "RCDB_GET",
        "RCDB_HISTORY", "RCDB_STATS", "RCDB_HASH", "RCDB_COMMITS", "RCDB_VERIFY",
        "RCDB_STATE_HASH", "RCDB_SECURITY",
    }
    missing_from_registry = sorted(storage - set(_REGISTRY))
    assert not missing_from_registry, (
        f"the inventory names operators that no longer exist: {missing_from_registry}"
    )

    source = pathlib.Path(__file__).read_text()
    unexercised = sorted(name for name in storage if f'_op("{name}")' not in source)
    assert not unexercised, (
        f"storage operators with no characterization in this file: {unexercised}"
    )
