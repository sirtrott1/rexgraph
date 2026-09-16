"""Governed writes validate identities, time domains and version expectations."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pytest
from rexgraph.graph import RexGraph

from rcdb import MemoryStore, VersionConflictError


def rex(n=2):
    return RexGraph.from_graph(list(range(n)), list(range(1, n + 1)))


def test_expected_version_prevents_stale_publication_and_staging(monkeypatch):
    store = MemoryStore().configure_security(require_commits=True)
    store.commit_mutation("r", rex(), expected_version=0, analytics=False)
    def forbidden(*args):
        pytest.fail("stale write staged an artifact")
    monkeypatch.setattr(store, "_store_commit_bytes", forbidden)
    with pytest.raises(VersionConflictError, match="expected version 0, current version 1"):
        store.commit_mutation("r", rex(3), expected_version=0, analytics=False)
    assert len(store.history("r")) == len(store.commit_history("r")) == 1


def test_same_handle_competing_expected_versions_admit_only_one():
    store = MemoryStore().configure_security(require_commits=True)
    def write(n):
        try:
            return store.commit_mutation("r", rex(n), expected_version=0, analytics=False)
        except VersionConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [2, 3]))
    assert sum(value is not None for value in results) == 1
    assert store.verify_commits("r")
    assert len(store.history("r")) == 1


def test_mutation_target_is_literal_not_a_version_display_alias():
    store = MemoryStore()
    store.commit_mutation("r", rex(), analytics=False)
    store.commit_mutation("r@1", rex(3), expected_version=0, analytics=False)
    assert store.read_record("r@1").value.nE == 3
    assert store.read_record("r").value.nE == 2
    assert store.verify_commits("r@1")
    assert store.commit_history("r@1")[0].transition.previous_state == ""


@pytest.mark.parametrize("method", ["put", "put_prepared", "commit_mutation"])
@pytest.mark.parametrize("options", [
    {"valid_from": float("nan")}, {"valid_to": float("inf")}, {"valid_from": True},
    {"valid_from": 1, "valid_to": 1}, {"valid_from": 2, "valid_to": 1},
])
def test_invalid_time_interval_never_reaches_publication(method, options, monkeypatch):
    store = MemoryStore()
    def forbidden(*args, **kwargs):
        pytest.fail("invalid interval reached publication")
    monkeypatch.setattr(store, "_put_impl", forbidden)
    args = ("r", b"unused", {}) if method == "put_prepared" else ("r", rex())
    with pytest.raises(ValueError):
        getattr(store, method)(*args, **options)


@pytest.mark.parametrize("options", [{"expected_version": -1}, {"expected_version": True},
                                    {"expected_version": 1.0}, {"actor": 1}, {"tx_time": True}])
def test_invalid_mutation_metadata_never_stages(options, monkeypatch):
    store = MemoryStore()
    def forbidden(*args, **kwargs):
        pytest.fail("invalid mutation staged an artifact")
    monkeypatch.setattr(store, "_store_commit_bytes", forbidden)
    with pytest.raises((ValueError, TypeError)):
        store.commit_mutation("r", rex(), **options)


def test_validity_times_have_one_clock_representation_not_object_metadata():
    store = MemoryStore()
    rec = store.commit_mutation("r", rex(), valid_from=Fraction(1, 3), valid_to=Fraction(2, 3), analytics=False)
    assert type(rec.valid_from) is type(rec.valid_to) is float
    assert rec.valid_from == 1 / 3
    assert store.read_record("r", valid_at=0.5).record.version == 1
    with pytest.raises(ValueError, match="follow valid_from"):
        store.commit_mutation("r", rex(), valid_from=2**54, valid_to=2**54 + 1, analytics=False)
