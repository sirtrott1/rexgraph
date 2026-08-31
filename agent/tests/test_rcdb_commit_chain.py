"""A version and the signed artifact that attests to it.

The audit chain proves an audit LINE was not edited. It says nothing about whether record
X at version 3 is the state the commit after version 2 says follows it. That is a
different object, and this is it: each version carries a mutation package binding the
endpoint it started from to the one it produced, so a rewritten history stops verifying
even when every record on its own looks well formed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from agent.rcdb import MemoryStore, open_store  # noqa: E402

from rexgraph.io.security import Ed25519Signer  # noqa: E402


def _rex(n=2):
    from rexgraph.graph import RexGraph
    ptr = [0] + [2 * (i + 1) for i in range(n)]
    idx = []
    for i in range(n):
        idx += [i, (i + 1) % (n + 1)]
    return RexGraph.from_hypergraph(ptr, idx)


@pytest.fixture
def store():
    return MemoryStore()


def test_a_committed_version_verifies(store):
    store.commit_mutation("r1", _rex(2), actor="alice")
    assert store.verify_commits("r1") is True
    assert len(store.commit_history("r1")) == 1


def test_a_chain_of_versions_verifies(store):
    for n in (2, 3, 4):
        store.commit_mutation("r1", _rex(n), actor="alice")
    assert store.verify_commits("r1") is True
    assert len(store.commit_history("r1")) == 3


def test_the_chain_links_each_package_to_the_one_before(store):
    for n in (2, 3, 4):
        store.commit_mutation("r1", _rex(n), actor="alice")
    history = store.commit_history("r1")
    assert history[0].link.parent_digest is None, "the first commit has no parent"
    for earlier, later in zip(history, history[1:], strict=False):
        assert later.link.parent_digest == earlier.link.digest


def test_a_rewritten_record_stops_verifying(store):
    """The whole point: the bytes still deserialize, the chain still refuses."""
    store.commit_mutation("r1", _rex(2), actor="alice")
    store.commit_mutation("r1", _rex(3), actor="alice")
    assert store.verify_commits("r1") is True
    # replace the stored complex at version 2 with a different one
    store._blobs[("r1", 2)] = store._serialize_payload(_rex(5))
    assert store.verify_commits("r1") is False


def test_a_store_that_requires_commits_refuses_a_plain_put(store):
    store.configure_security(require_commits=True)
    with pytest.raises(PermissionError):
        store.put("r1", _rex(2))
    store.commit_mutation("r1", _rex(2), actor="alice")
    assert store.get("r1") is not None


def test_requiring_commits_is_off_by_default(store):
    store.put("r1", _rex(2))
    assert store.get("r1") is not None
    assert store.security_status()["require_commits"] is False


def test_a_signing_policy_is_satisfied_by_a_signed_package():
    from rexgraph.io.mutation import MutationPolicy
    signer = Ed25519Signer.generate("alice")
    policy = MutationPolicy(require_transition_signature=True,
                            allowed_signers=("alice",))
    store = MemoryStore().configure_security(
        mutation_policy=policy, transition_signer=signer)
    store.commit_mutation("r1", _rex(2), actor="alice")
    assert store.verify_commits("r1") is True


def test_a_signing_policy_refuses_an_unsigned_package():
    from rexgraph.io.mutation import MutationPolicy
    policy = MutationPolicy(require_transition_signature=True,
                            allowed_signers=("alice",))
    store = MemoryStore().configure_security(mutation_policy=policy)
    with pytest.raises(ValueError):
        store.commit_mutation("r1", _rex(2), actor="alice")


@pytest.mark.parametrize("backend", ["memory", "file", "rex"])
def test_every_backend_holds_the_artifact(backend, tmp_path):
    uri = "memory://" if backend == "memory" else f"{backend}://{tmp_path / backend}"
    s = open_store(uri)
    s.commit_mutation("r1", _rex(2), actor="alice")
    s.commit_mutation("r1", _rex(3), actor="alice")
    assert len(s.commit_history("r1")) == 2
    assert s.verify_commits("r1") is True


def test_an_unattested_version_is_ordinary_where_commits_are_optional(store):
    """With commits optional a version may simply not have one, so the walk continues."""
    store.commit_mutation("r1", _rex(2), actor="alice")
    store.commit_mutation("r1", _rex(3), actor="alice")
    store._commit_blobs.pop(("r1", 2))
    assert store.verify_commits("r1") is True
    assert len(store.commit_history("r1")) == 1


def test_deleting_an_artifact_cannot_hide_a_version_where_commits_are_required(store):
    """Where every write went through commit_mutation, a missing artifact is evidence.

    Skipping it would make deletion the cheapest attack on the chain: remove the artifact
    for the version you rewrote and the walk never looks at it.
    """
    store.configure_security(require_commits=True)
    store.commit_mutation("r1", _rex(2), actor="alice")
    store.commit_mutation("r1", _rex(3), actor="alice")
    assert store.verify_commits("r1") is True
    store._commit_blobs.pop(("r1", 2))
    assert store.verify_commits("r1") is False


def test_a_package_that_does_not_verify_is_never_published(store, monkeypatch):
    """The reference gated this raise on the store demanding a signature, reasoning that
    an unsigned package would not verify under a permissive policy. It does verify:
    verify_mutation already accounts for the policy. So a False here is a real failure,
    and gating meant the DEFAULT configuration published packages that did not verify."""
    import agent.rcdb as rcdb_mod
    store.commit_mutation("r1", _rex(2), actor="alice")

    import rexgraph.io.mutation as mut
    monkeypatch.setattr(mut, "verify_mutation", lambda *a, **k: False)
    with pytest.raises(ValueError):
        store.commit_mutation("r1", _rex(3), actor="alice")
    assert rcdb_mod is not None
    # and nothing was published: neither the version nor a stray artifact
    assert store.get_record("r1").version == 1
    assert ("r1", 2) not in store._commit_blobs


def test_an_unsigned_package_verifies_under_a_permissive_policy():
    """The fact the reference's gate was built on, stated as a test so the reasoning
    above cannot silently stop being true."""
    from rexgraph.io.mutation import MutationPolicy, prepare_mutation, verify_mutation
    policy = MutationPolicy()
    package = prepare_mutation(None, _rex(2), tx_time=1.0, actor="a", policy=policy,
                               parent_digest=None)
    assert verify_mutation(package, previous=None, policy=policy, verifiers={},
                           parent_digest=None) is True


@pytest.mark.parametrize("backend", ["memory", "file", "rex"])
def test_deleting_a_record_reclaims_its_artifacts(backend, tmp_path):
    """An artifact outliving its record is not litter.

    commit_history would hand back a package for a version nothing can produce, and a
    later record reusing that id and version would inherit an attestation it never
    earned.
    """
    uri = "memory://" if backend == "memory" else f"{backend}://{tmp_path / backend}"
    s = open_store(uri)
    s.commit_mutation("r1", _rex(2), actor="alice")
    s.commit_mutation("r1", _rex(3), actor="alice")
    assert len(s.commit_history("r1")) == 2
    s.delete("r1")
    assert s.commit_history("r1") == []
    s.commit_mutation("r1", _rex(4), actor="alice")
    assert len(s.commit_history("r1")) == 1, "a reused id inherited an old attestation"


def test_a_raw_delete_is_refused_where_commits_are_required(store):
    """Deletion is the one operation a commit chain cannot describe: no artifact attests
    that a record was meant to stop existing, so a raw delete would silently end a
    lineage the chain still claims is intact."""
    store.commit_mutation("r1", _rex(2), actor="alice")
    store.configure_security(require_commits=True)
    with pytest.raises(PermissionError):
        store.delete("r1")
    assert store.get("r1") is not None
