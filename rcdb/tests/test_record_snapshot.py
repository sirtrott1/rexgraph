"""All backends select one native state, metadata and digest by the same contract."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io.catalog import object_digest

from rcdb import FileStore, MemoryStore, ObjectStore, RecordSnapshot, RexStore, SQLStore


@pytest.fixture(params=["memory", "file", "rex", "sql", "object"])
def store(request, tmp_path):
    if request.param == "memory":
        result = MemoryStore()
    elif request.param == "file":
        result = FileStore(str(tmp_path / "file"))
    elif request.param == "rex":
        result = RexStore(str(tmp_path / "rex"))
    elif request.param == "sql":
        pytest.importorskip("sqlalchemy")
        result = SQLStore(f"sqlite:///{tmp_path / 'sql.db'}")
    else:
        pytest.importorskip("fsspec")
        result = ObjectStore(f"file://{tmp_path / 'object'}")
    yield result
    result.close()


def rex(n):
    return RexGraph.from_graph(list(range(n)), list(range(1, n + 1)))


def test_selectors_and_detached_metadata(store):
    first = store.put("r", rex(2), meta={"user": {"name": "original"}},
                      _tx_time=100.0, valid_from=10.0, valid_to=20.0, analytics=False)
    store.put("r", rex(3), _tx_time=200.0, valid_from=20.0, analytics=False)
    for options, version, edges in [
        ({}, 2, 3), ({"version": 1}, 1, 2), ({"as_of": 150.0}, 1, 2),
        ({"valid_at": Fraction(15)}, 1, 2), ({"valid_at": 20.0}, 2, 3),
        ({"as_of": 150.0, "valid_at": 15.0}, 1, 2),
    ]:
        snapshot = store.read_record("r", **options)
        assert isinstance(snapshot, RecordSnapshot)
        assert (snapshot.record.id, snapshot.record.version, snapshot.value.nE) == ("r", version, edges)
        assert snapshot.state_digest == object_digest(snapshot.value)
        assert snapshot.record is not first
    assert store.read_record("r", as_of=200.0, valid_at=15.0) is None
    assert store.read_record("r", as_of=99.0) is None
    assert store.read_record("r", version=3) is None
    assert store.read_record("missing") is None
    snapshot = store.read_record("r", version=1)
    snapshot.record.meta["user"]["name"] = "changed"
    snapshot.record.signature["nE"] = 999
    snapshot.record.version = 999
    again = store.read_record("r", version=1)
    assert again.record.meta["user"]["name"] == "original"
    assert again.record.signature["nE"] == 2
    assert again.record.version == 1
    assert again.value is not snapshot.value


def test_display_alias_and_literal_id_are_distinct(store):
    store.put("r", rex(2), analytics=False)
    store.put("r", rex(3), analytics=False)
    alias = store.read_record("r@1")
    assert (alias.record.id, alias.record.version, alias.value.nE) == ("r", 1, 2)
    with pytest.raises(ValueError, match="display alias"):
        store.read_record("r@1", as_of=10.0)
    store.put("r@1", rex(4), analytics=False)
    literal = store.read_record("r@1")
    assert (literal.record.id, literal.record.version, literal.value.nE) == ("r@1", 1, 4)
    assert store.read_record("r", version=1).value.nE == 2


def test_published_metadata_with_missing_payload_is_an_integrity_error(store, monkeypatch):
    store.put("r", rex(2), analytics=False)
    monkeypatch.setattr(store, "get_version", lambda *args: None)
    with pytest.raises(ValueError, match="published RCDB record.*has no payload"):
        store.read_record("r")


def test_temporal_payload_uses_native_state_digest(store):
    timeline = TemporalRex([])
    timeline.append_snapshot(rex(2), at=1.0)
    timeline.append_snapshot(rex(3), at=4.0)
    store.put("timeline", timeline, analytics=False)
    snapshot = store.read_record("timeline")
    assert isinstance(snapshot.value, TemporalRex)
    assert snapshot.value.T == 2
    assert snapshot.value.reconstruct_at(1).nE == 3
    assert snapshot.state_digest == object_digest(timeline)


def test_conversation_capture_keeps_labels_and_turn_ids(store):
    from rexgraph.flow.turn_field import TurnField
    from rcql import Executor, parse
    field = TurnField()
    for text in ("alpha beta", "beta gamma delta", "alpha beta"):
        field.observe(text)
    history = Executor(sources={"chat": field}).execute(parse(
        'FROM $chat RETURN TURN_FIELD()')).values[0]
    store.put("chat", history, analytics=False)
    captured = store.read_record("chat")
    assert captured.state_digest == object_digest(history)
    for step in range(3):
        rex = captured.value.at(step)
        assert rex.relation_ids.tolist() == list(range(step + 1))
        assert rex._agent_meta == history.at(step)._agent_meta
    result = Executor(sources={"history": captured.value}).execute(parse(
        'FROM AT($history,2) RETURN COUNT(CELLS(1)),CHARACTER(true)'))
    assert result.values[0] == 3


def test_branching_orientation_and_exact_channel_state_survive_selection(store):
    from rexgraph.rational_trig import exact_character
    value = RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                     boundary_idx=np.array([2, 0, 1, 3, 0, 2], np.int32),
                     w_E=np.array([Fraction(2, 3), Fraction(5, 7)], dtype=object))
    expected = exact_character(value)
    store.put("branch", value, analytics=False)
    snapshot = store.read_record("branch")
    assert snapshot.state_digest == object_digest(value)
    np.testing.assert_array_equal(snapshot.value.boundary_ptr, value.boundary_ptr)
    np.testing.assert_array_equal(snapshot.value.boundary_idx, value.boundary_idx)
    actual = exact_character(snapshot.value)
    assert actual[1] == expected[1]
    np.testing.assert_array_equal(actual[0], expected[0])


def test_native_rank_betti_queries_use_selected_primary_state(store):
    from rcql import Executor, parse
    first = RexGraph.from_hypergraph([0, 4], [0, 1, 2, 3])
    second = RexGraph.from_graph([0, 1, 2], [1, 2, 3])
    store.put("rank-state", first, analytics=False)
    store.put("rank-state", second, analytics=False)
    executor = Executor(sources={"db": store})
    for version, expected in [(1, (1, 0, 3)), (2, (3, 0, 1))]:
        result = executor.execute(parse(
            f'FROM RCDB_VERSION($db,"rank-state",{version}) RETURN RANK(1), NULLITY(1), BETTI(0)'))
        assert result.values == expected
        assert result.native_plan["source"]["state"]["record_version"] == version
        assert result.execution[0]["methods"][0]["method"] == "native-exact-rank"


def test_higher_native_integer_tower_roundtrips_to_rank_queries(store):
    from rcql import Executor, parse
    from rexgraph.native_sparse import csr_carrier, empty_native
    n = 2**60
    value = RexGraph.from_graph([0], [1])
    value._graded_duals = [empty_native((0, 2)).dual, csr_carrier(np.array([0, 2, 4], np.int32),
        np.array([0, 1, 0, 1], np.int32), np.array([n, n+1, n-1, n], np.int64), (2, 2))]
    store.put("native-tower", value, analytics=False)
    snapshot = store.read_record("native-tower")
    assert snapshot.state_digest == object_digest(value)
    result = Executor(sources={"db": store}).execute(parse(
        'FROM RCDB_VERSION($db,"native-tower",1) RETURN RANK(4), NULLITY(4), BETTI(3), BETTI(4)'))
    assert result.values == (2, 0, 0, 0)


@pytest.mark.parametrize("channel,expected", [
    ("T", [Fraction(16, 27), Fraction(40, 63)]),
    ("G", [Fraction(16, 27), Fraction(-40, 63)]),
    ("F", [Fraction(80, 63), Fraction(80, 63)]),
    ("C", [Fraction(4, 3), Fraction(-4, 3)]),
])
def test_exact_full_channel_query_keeps_selected_weighted_state(store, channel, expected):
    rcql = pytest.importorskip("rcql")
    for sign in (-1, 1):
        value = RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                         boundary_idx=np.array([2, 0, 1, 3, 0, 2], np.int32),
                         w_E=np.array([Fraction(2, 3), sign*Fraction(5, 7)], dtype=object))
        store.put("branch", value, analytics=False)
    executor = rcql.Executor(sources={"db": store})
    result = executor.execute(rcql.parse('FROM RCDB_VERSION(RCDB("db"), "branch", 1) '
        f'RETURN APPLY(CHANNEL("{channel}"),INDICATOR(CELL(1,0)),true)'))
    assert result.values[0].values.tolist() == expected
    state = result.provenance[0]["source_state"]
    assert state["record_version"] == 1 and state["state_digest"] == object_digest(store.get_version("branch", 1))
    latest = executor.execute(rcql.parse('FROM RCDB_GET(RCDB("db"), "branch") '
        f'RETURN APPLY(CHANNEL("{channel}"),INDICATOR(CELL(1,0)),true)'))
    assert latest.values[0].values.tolist() == [expected[0], expected[1] if channel == "C" else -expected[1]]
    assert result.exactness[0].value == latest.exactness[0].value == "rational"


def test_rectangular_phrase_check_keeps_both_selected_native_versions(store):
    rcql = pytest.importorskip("rcql")

    for sign in (1, -1):
        value = RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                         boundary_idx=np.array([2, 0, 1, 3, 0, 2], np.int32),
                         w_E=np.array([sign*Fraction(2, 3), Fraction(5, 7)], dtype=object))
        store.put("branch", value, analytics=False)
    snapshots = [store.read_record("branch", version=v) for v in (1, 2)]
    stalks = tuple(rcql.PhraseStalk(f"v{v}", rcql.bind(
        f"branch@{v}", snap.value, rcql.SourcePolicy.allow("read", "identity"),
        source_ref=rcql.SourceRef(name=f"branch@{v}", record_id="branch", record_version=v,
                                  state_digest=snap.state_digest)))
        for v, snap in zip((1, 2), snapshots, strict=True))
    sh = rcql.PhraseSheaf(stalks, (rcql.PhraseCorrespondence("metric-view", ("v1", "v2")),),
                          stalk_dims={"v1": 2, "v2": 2}, correspondence_dims={"metric-view": 1})
    for name, snap, sign in zip(("v1", "v2"), snapshots, (1, -1), strict=True):
        sh.assign(name, snap.value.edge_metric_exact)
        sh.restrict(name, "metric-view", [[sign, 0]])
    executor = rcql.Executor(params={"section": sh})
    text = rcql.parse('FROM PHRASE($section) RETURN SECTION_CHECK($section), GLUE($section).ratio')
    result = executor.execute(text)
    assert result.values[0].compatible and result.values[1] == 1
    assert [r.record_version for r in result.values[0].contributors] == [1, 2]
    assert [r.state_digest for r in result.values[0].contributors] == [s.state_digest for s in snapshots]
    store.put("branch", rex(5), analytics=False)
    assert executor.execute(text).values == result.values  # latest is not reselected
    snapshots[0].value.w_E[0] = Fraction(9)
    with pytest.raises(rcql.PhraseMapError, match="selected state.*changed"):
        executor.execute(text)


def test_logical_manifest_and_digest_match_independent_memory_reference(store):
    from rexgraph.io.manifest import manifest_digest
    reference = MemoryStore()
    assert store.state_manifest() == {"object_type": "RCDBLogicalState", "version": 1, "records": []}
    for target in (store, reference):
        # Deliberately out of order IDs and tag order; analytics are derived, not identity.
        target.put("z", rex(2), _tx_time=100.0, valid_from=10.0, tags=["b", "a"],
                   meta={"name": "native"}, analytics=target is reference)
        target.put("a", rex(3), _tx_time=120.0, analytics=False)
        target.put("z", rex(4), _tx_time=200.0, valid_from=20.0, analytics=False)
    assert store.state_manifest() == reference.state_manifest()
    assert store.state_digest() == manifest_digest(reference.state_manifest())
    manifest = store.state_manifest()
    assert [(r["id"], r["version"]) for r in manifest["records"]] == [("a", 1), ("z", 1), ("z", 2)]
    before = store.state_digest()
    manifest["records"][1]["meta"]["name"] = "detached"
    assert store.state_digest() == before
    store.put("z", rex(4), _tx_time=300.0, valid_from=30.0, analytics=False)
    assert store.state_digest() != before


def test_governed_copy_and_migration_preserve_data_through_one_commit_seam(store):
    from rcdb import copy_record, migrate
    source = MemoryStore()
    first = source.put("r", rex(2), valid_from=10.0, valid_to=20.0,
                       meta={"origin": "source"}, tags=["test"], analytics=False)
    source.put("r", rex(3), valid_from=20.0, analytics=False)
    store.configure_security(require_commits=True)
    copied = copy_record(source, store, first, actor="copy", expected_version=0)
    assert copied.meta["origin"] == "source"
    assert copied.valid_from == 10.0 and copied.valid_to == 20.0
    assert store.read_record("r").state_digest == source.read_record("r", version=1).state_digest
    assert store.verify_commits("r")
    assert store.state_manifest()["records"][0]["commit"] == store.commit_history("r")[0].digest
    result = migrate(source, store)
    assert result["versions"] == 2
    assert len(store.history("r")) == len(store.commit_history("r")) == 3
    assert store.verify_commits("r")
    with pytest.raises(PermissionError, match="requires.*mutation commits"):
        copy_record(source, store, first, governed=False)


@pytest.mark.parametrize("id, options", [
    ("", {}), (1, {}), ("r", {"version": 0}), ("r", {"version": True}),
    ("r", {"version": 1.0}), ("r", {"version": 1, "as_of": 2}),
    ("r", {"version": 1, "valid_at": 2}), ("r", {"as_of": float("nan")}),
    ("r", {"valid_at": float("inf")}), ("r", {"valid_at": True}),
])
def test_invalid_selector_is_refused_before_storage(id, options, monkeypatch):
    store = MemoryStore()
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid selector reached storage")
    monkeypatch.setattr(store, "history", forbidden)
    with pytest.raises((TypeError, ValueError)):
        store.read_record(id, **options)
