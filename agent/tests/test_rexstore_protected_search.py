"""A store whose vocabulary is tokens still answers a vocabulary query.

RexStore kept `label -> {id}` in the clear and wrote the same labels into the record log,
so an index built beside protected records handed back what the protection was for. With
a search policy the labels become fixed width tokens: the log carries tokens, the
persisted relation carries tokens, and `query(labels_any=...)` still resolves.
"""
from __future__ import annotations

import pytest
from agent.rcdb_protected_index import IndexPolicy, StaticIndexKeyProvider
from agent.rexstore import SEARCH, RexStore

pytest.importorskip("safetensors")

SECRET_TERMS = ("oncology", "cardiology")


def _rex():
    from rexgraph.graph import RexGraph
    return RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2, 2, 0])


def _keyed():
    return (IndexPolicy({"vertex_labels": "keyed"}, "search"),
            StaticIndexKeyProvider({"search": b"s" * 32}))


def _seed(store):
    store.put("r1", _rex(), meta={"vertex_labels": ["oncology", "shared"]})
    store.put("r2", _rex(), meta={"vertex_labels": ["cardiology", "shared"]})


@pytest.fixture
def protected(tmp_path):
    policy, keys = _keyed()
    store = RexStore(str(tmp_path / "s"), search_policy=policy, search_keys=keys)
    _seed(store)
    return store, tmp_path / "s", policy, keys


def _ids(store, term):
    return sorted(r.id for r in store.query(labels_any=[term]))


def test_a_vocabulary_query_still_resolves(protected):
    store, _root, _p, _k = protected
    assert _ids(store, "oncology") == ["r1"]
    assert _ids(store, "cardiology") == ["r2"]
    assert _ids(store, "shared") == ["r1", "r2"]
    assert _ids(store, "never-stored") == []


def test_the_log_carries_tokens_so_replay_does_not_need_the_key(protected):
    """The frame's `extra` row carries the tokens behind a magic word, so a replay
    re-adds terms it could not itself compute."""
    store, root, _p, _k = protected
    raw = (root / "records.log").read_bytes()
    assert raw.startswith(b"REXLOG")
    assert store._search_tail, "no tokens were admitted"
    for refs in store._search_tail.values():
        assert all(isinstance(v, int) for _rid, v in refs)


def test_the_record_itself_still_carries_its_labels(protected):
    """A deliberate boundary, recorded rather than hidden.

    Protecting the SEARCH INDEX does not minimise what the RECORD stores: the log frame
    still holds meta["vertex_labels"] and the signature's labels_sample in the clear.
    Removing those is `_stored_meta` and `signature_mode`, which are not ported yet, so
    a protected index alone does not make the store's own bytes term-free.
    """
    store, root, _p, _k = protected
    raw = (root / "records.log").read_bytes()
    assert b"oncology" in raw, \
        "if this now passes, record minimisation landed and this test should assert it"


def test_the_persisted_relation_carries_no_plaintext_term(protected):
    store, root, _p, _k = protected
    store.write_index()
    assert (root / SEARCH).exists()
    raw = (root / SEARCH).read_bytes()
    for term in SECRET_TERMS:
        assert term.encode() not in raw, term


def test_a_reopened_store_still_answers(protected):
    """Replay has to reconstruct the tail from the tokens the log carries."""
    store, root, policy, keys = protected
    store.write_index()
    again = RexStore(str(root), search_policy=policy, search_keys=keys)
    assert _ids(again, "oncology") == ["r1"]
    assert _ids(again, "shared") == ["r1", "r2"]


def test_compaction_carries_the_tokens_forward(protected):
    store, root, policy, keys = protected
    store.write_index()
    store.compact()
    assert _ids(store, "oncology") == ["r1"]
    again = RexStore(str(root), search_policy=policy, search_keys=keys)
    assert _ids(again, "shared") == ["r1", "r2"]


def test_a_store_opened_without_the_key_cannot_read_the_vocabulary(protected):
    """The relation is on disk either way; without the key a term does not tokenise to
    anything the index holds."""
    store, root, policy, _keys = protected
    store.write_index()
    wrong = StaticIndexKeyProvider({"search": b"w" * 32})
    blind = RexStore(str(root), search_policy=policy, search_keys=wrong)
    assert _ids(blind, "oncology") == []


def test_a_store_with_no_policy_is_unchanged(tmp_path):
    """The default path keeps plaintext labels and the behaviour it always had."""
    store = RexStore(str(tmp_path / "plain"))
    _seed(store)
    assert _ids(store, "oncology") == ["r1"]
    assert store._labels.get("oncology") == {"r1"}
    assert not (tmp_path / "plain" / SEARCH).exists()


def test_a_protected_index_and_a_minimal_signature_compose(tmp_path):
    """The two halves together are what makes the store's own bytes term-free.

    Protecting the index stops the INDEX naming a term; minimising the signature and meta
    stops the RECORD carrying it. Either alone leaves the term on disk, which is why the
    sibling test above asserts it is still there under the default public mode.
    """
    pytest.importorskip("cryptography")
    from rexgraph.io.security import StaticKeyProvider

    policy, keys = _keyed()
    root = tmp_path / "both"
    store = RexStore(str(root), search_policy=policy, search_keys=keys)
    store.configure_security(
        key_id="records", keys=StaticKeyProvider({"records": b"r" * 32}),
        signature_mode="minimal")
    _seed(store)
    store.write_index()

    assert _ids(store, "oncology") == ["r1"], "the index stopped answering"
    for path in (root / "records.log", root / SEARCH, root / "blobs.pack"):
        raw = path.read_bytes()
        for term in SECRET_TERMS:
            assert term.encode() not in raw, f"{term} survives in {path.name}"
