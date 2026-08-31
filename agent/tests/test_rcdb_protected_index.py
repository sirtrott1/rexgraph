"""The search index must not become the copy of the data that has no protection.

The canonical RCDB snapshot writes labels in the clear, so an index built beside sealed
records would hand back exactly what the sealing was for. A protected relation stores
fixed width tokens instead: no plaintext term and no plaintext record id, in memory or on
disk, and exact lookup still works.

The source tests loaded the module by path to avoid importing a compiled package. Here it
imports normally, so these say the same things directly.
"""
from __future__ import annotations

import numpy as np
import pytest
from agent.rcdb_protected_index import (
    IndexPolicy,
    StaticIndexKeyProvider,
    build_search_relation,
    build_search_relation_from_tokens,
    load_search_relation,
    save_search_relation,
    term_token,
)

pytest.importorskip("safetensors")

SECRETS = ("alice", "oncology", "hospital-a", "r1")


class _Rec:
    """The subset of ComplexRecord that _terms_of reads."""

    def __init__(self, version=1):
        self.version = version
        self.signature = {"source": "hospital-a", "tags": ["oncology"]}
        self.meta = {"vertex_labels": ["alice", "diagnosis"]}


@pytest.fixture
def relation(monkeypatch):
    import agent.rcdb_index as idx
    monkeypatch.setattr(idx, "KINDS", ("source", "tags", "vertex_labels"), raising=False)
    monkeypatch.setattr(idx, "_terms_of", lambda rec: [
        (0, [rec.signature["source"]]), (1, rec.signature["tags"]),
        (2, rec.meta["vertex_labels"])], raising=False)
    keys = StaticIndexKeyProvider({"search": b"x" * 32})
    policy = IndexPolicy({"source": "public", "tags": "keyed",
                          "vertex_labels": "keyed"}, "search")
    return build_search_relation([("r1", _Rec())], policy, keys=keys), policy, keys


def test_exact_lookup_resolves_through_the_tokens(relation):
    rel, policy, keys = relation
    assert rel.ids_for("vertex_labels", "alice", policy=policy, keys=keys) == ["r1\x001"]
    assert rel.ids_for("tags", "oncology", policy=policy, keys=keys) == ["r1\x001"]
    assert rel.ids_for("vertex_labels", "never-stored", policy=policy, keys=keys) == []


def test_the_relation_holds_no_plaintext_in_memory(relation):
    rel, _policy, _keys = relation
    raw = rel.token_bytes.tobytes() + rel.record_tokens.tobytes()
    for secret in SECRETS:
        assert secret.encode() not in raw, secret
    assert rel.rel_idx.dtype == np.int64


def test_the_relation_holds_no_plaintext_on_disk(relation, tmp_path):
    rel, policy, keys = relation
    path = tmp_path / "search.safetensors"
    save_search_relation(path, rel)
    persisted = path.read_bytes()
    for secret in SECRETS:
        assert secret.encode() not in persisted, secret
    loaded = load_search_relation(path)
    assert (loaded.tokens_for("vertex_labels", "alice", policy=policy, keys=keys)
            == rel.tokens_for("vertex_labels", "alice", policy=policy, keys=keys))


def test_a_persisted_relation_cannot_resolve_identities(relation, tmp_path):
    """record_ids is in-memory only, which is what keeps identities out of the file."""
    rel, policy, keys = relation
    path = tmp_path / "search.safetensors"
    save_search_relation(path, rel)
    with pytest.raises(ValueError):
        load_search_relation(path).ids_for("tags", "oncology", policy=policy, keys=keys)


def test_two_versions_of_one_record_stay_distinct(monkeypatch):
    import agent.rcdb_index as idx
    monkeypatch.setattr(idx, "KINDS", ("vertex_labels",), raising=False)
    keys = StaticIndexKeyProvider({"search": b"k" * 32})
    policy = IndexPolicy({"vertex_labels": "keyed"}, "search")
    old = term_token("vertex_labels", "old", mode="keyed", key_id="search", keys=keys)
    new = term_token("vertex_labels", "new", mode="keyed", key_id="search", keys=keys)
    rel = build_search_relation_from_tokens(
        [("r1", 1, (old,)), ("r1", 2, (new,))], policy, kind="vertex_labels", keys=keys)
    assert rel.ids_for("vertex_labels", "old", policy=policy, keys=keys) == ["r1\x001"]
    assert rel.ids_for("vertex_labels", "new", policy=policy, keys=keys) == ["r1\x002"]
    assert bytes(rel.record_tokens[0]) != bytes(rel.record_tokens[1])


def test_a_relation_answers_nothing_under_a_different_policy(relation):
    """A miss under the wrong policy would read as "no such record" rather than as the
    mismatch it is, so the digest is checked before the lookup."""
    rel, _policy, keys = relation
    other = IndexPolicy({"source": "public", "tags": "public",
                         "vertex_labels": "keyed"}, "search")
    with pytest.raises(ValueError):
        rel.ids_for("tags", "oncology", policy=other, keys=keys)


def test_a_keyed_policy_without_a_key_id_is_refused():
    with pytest.raises(ValueError):
        IndexPolicy({"tags": "keyed"})


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        IndexPolicy({"tags": "sideways"})


def test_a_keyed_token_is_scoped_to_its_workspace(tmp_path, monkeypatch):
    """The adaptation that makes this a port rather than a copy.

    The reference took key bytes through a static provider. Here the identifier resolves
    through the workspace-scoped secret store, so the same name in two workspaces derives
    two different keys and one tenant's tokens are meaningless to another. A static
    provider constructed inside a request would hand every tenant the same key.
    """
    pytest.importorskip("cryptography")
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", f"file://{tmp_path}/connections.json")
    from agent.kms import WorkspaceKeyring
    from agent.server import auth, scope
    scope.reset_secret_store(); auth.reset_auth_manager()
    auth.get_auth_manager().enable_auth()

    def token_in(workspace):
        held = scope.set_workspace(workspace)
        try:
            scope.secret_store().put("idx", f"secret-of-{workspace}", "key")
            return term_token("tags", "oncology", mode="keyed",
                              key_id="idx", keys=WorkspaceKeyring())
        finally:
            scope.reset_workspace(held)

    alpha, beta = token_in("alpha"), token_in("beta")
    assert alpha != beta, "the same key id in two workspaces produced the same token"
    assert len(alpha) == len(beta) == 32
    scope.reset_secret_store(); auth.reset_auth_manager()
