"""Phase 0 characterization of the storage, catalog and metadata operators.

These record what the current adapters DO, not what the native contract will require.
That is the point of a characterization suite: an operator cannot be moved behind a static
signature safely unless its present behaviour is pinned first, so a Phase 1 change is
visible as a diff rather than as a silent shift in meaning.

Scope is deliberately the 19 operators that are not Rex mathematics: REX, the file catalog
and metadata readings, and the RCDB readings. The 19 Rex-math adapters are excluded on
purpose. Their exactness and grade contracts are being corrected in the same phase, so
pinning today's floating and grade-1-only behaviour would produce tests written to fail.

Where an operator's current behaviour is wrong against the native contract, the test says
so and pins the wrong behaviour anyway, naming what Phase 1 owes. A characterization test
that quietly asserts the target contract is not characterizing anything.
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
    s = rcdb.open_store(f"rex://{tmp_path / 'store'}")
    s.put("r1", rex, meta={"note": "first"}, tags=["t"])
    yield s
    s.close()


# ---------------------------------------------------------------- source binding


def test_rex_returns_the_bound_name_rather_than_a_typed_complex(rex):
    """REX is a name passthrough today; the executor resolves it separately.

    The native contract makes REX(name) yield a typed Rex carrying source state and policy.
    Until then the operator cannot be asked anything about the complex it names.
    """
    assert _op("REX")(rex, "graph_a") == "graph_a"
    assert isinstance(_op("REX")(rex, "graph_a"), str)


# ---------------------------------------------------------------- file catalog


def test_the_catalog_indexes_loadable_kinds_only(catalog):
    """A plain text file beside an indexed tensor file is simply not in the catalog."""
    names = [entry.name for entry in _op("FILES")(catalog)]
    assert any(name.endswith("m.safetensors") for name in names)
    assert not any(name.endswith("notes.txt") for name in names)


def test_file_info_and_hash_raise_for_an_unindexed_path_that_exists(catalog):
    """KeyError, not an empty result, and the file is genuinely on disk.

    Phase 1 owes this a signature precondition so an unknown entry is refused during
    binding rather than surfacing as a runtime KeyError from inside the adapter.
    """
    for name in ("FILE_INFO", "FILE_HASH", "TENSORS"):
        with pytest.raises(KeyError):
            _op(name)(catalog, "notes.txt")


def test_catalog_readings_are_bounded_and_structural(catalog):
    entry = _op("FILES")(catalog)[0]
    # FILE_INFO returns a CatalogEntry dataclass rather than a mapping, so a caller
    # rendering it as JSON has to convert. Phase 1 owes it a declared result type.
    info = _op("FILE_INFO")(catalog, entry.name)
    assert info.kind == "safetensors"
    assert _op("HASH_FILES")(catalog) == 1
    assert _op("SEARCH")(catalog, "nothing-matches-this") == []
    assert isinstance(_op("TENSORS")(catalog, entry.name), list)
    assert isinstance(_op("SEARCH_TENSORS")(catalog, entry.name, "w"), list)


def test_file_info_is_not_idempotent_and_depends_on_what_ran_before_it(catalog):
    """The same reading answers differently depending on call history.

    A catalog refreshed without hashing carries sha256 = None. FILE_HASH computes the
    digest and writes it back into the cached entry, so a FILE_INFO issued afterwards
    reports a digest that the identical call reported as None a moment earlier.

    This matters more for RCQL than for the catalog. The blueprint requires provenance to
    travel with a result and forbids EXPLAIN implying a field came from a state other than
    the one read, and common-subplan elimination assumes identical inputs give identical
    results. A reading whose field appears only after an unrelated operator has run
    satisfies neither. Phase 1 should either make the digest an explicit request that
    always computes, or declare it absent and leave FILE_HASH as the only way to get it.
    """
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


# ---------------------------------------------------------------- RCDB readings


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


def test_rcdb_state_hash_cannot_execute_against_any_store(store):
    """Registered, catalogued, and unreachable.

    RCDB_STATE_HASH requires ``source.state_digest()``. No RCStore has that method and
    none of the nine registered backends provides it, so the operator raises for every
    possible RCDB source rather than for a badly typed one. Phase 1 must either bind it to
    a real store-level digest or refuse it at signature time, because a catalogue entry
    that cannot run is worse than an absent one: it type-checks and then fails.
    """
    from rcdb.core import RCStore

    assert not hasattr(RCStore, "state_digest")
    assert not hasattr(store, "state_digest")

    with pytest.raises(TypeError, match="expects an RCDB store"):
        _op("RCDB_STATE_HASH")(store)


# ---------------------------------------------------------------- catalogue shape


def test_every_storage_operator_is_characterized_here():
    """Nothing this file owns can enter the registry without a characterization.

    The check is deliberately one-sided. Rex-mathematics operators are Codex's and their
    own direct suite covers them, so asserting an exhaustive registry equality here would
    fail on every addition they make and teach whoever hits it to widen a set without
    reading. What must not drift is the other direction: an operator in the storage,
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
