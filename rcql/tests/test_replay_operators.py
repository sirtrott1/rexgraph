"""Local replay, provider timing and explicit RCDB publication."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rcql import Executor, parse, lookup
from rcql.artifact_services import ArtifactServices
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io.catalog import object_digest
from rexgraph.io.mutation import MutationPolicy, prepare_mutation, mutation_to_bytes
from rexgraph.io.replication import pack_replication


def fixture(name, signed=False):
    from rexgraph.io.security import Ed25519Signer
    prior = RexGraph.from_cells([4, [[0, 1, 2], [0], [0, 1, 2]]], relation_ids=[11, 12, 13])
    target = RexGraph.from_cells([4, [[0, 2, 1], [0], [0, 1, 2]]], relation_ids=[11, 12, 13],
        w_E=np.array([Q(1, 3), 2**100+1, Q(-2, 7)], object), signs=[1, -1, 1])
    target._agent_meta = {"nested": {"reading": "new"}}
    signer = Ed25519Signer.generate("author") if signed else None
    policy = MutationPolicy(signed, signed, ("author",) if signed else ())
    package = prepare_mutation(prior, target, tx_time=1, parent_digest="parent", policy=policy,
        transition_signer=signer, lineage_signer=signer)
    blob = mutation_to_bytes(package)
    if name == "REPLICATE":
        blob, _ = pack_replication(b"already loaded", [blob], checkpoint_state=object_digest(prior),
                                   checkpoint_commit="parent")
    return prior, target, blob, policy, signer


def engine(name, signed=False):
    prior, target, blob, policy, signer = fixture(name, signed)
    options = {key: value for key, value in policy.manifest().items() if key != "version"}
    service = ArtifactServices(verifiers={"author": signer.verifier()}) if signed else None
    return Executor(sources={"r": prior}, params={"blob": blob, "p": options}, artifacts=service), target


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
@pytest.mark.parametrize("signed", [False, True])
def test_query_replay_matches_full_state_and_is_not_memoized(name, signed):
    executor, target = engine(name, signed)
    q = parse(f'FROM $r RETURN {name}($blob,"parent",$p),{name}($blob,"parent",$p)')
    result = executor.execute(q)
    a, b = result.values
    assert object_digest(a) == object_digest(b) == object_digest(target)
    assert a is not b and not lookup(name).memoizable
    a._agent_meta["nested"]["reading"] = "changed"
    assert b._agent_meta == target._agent_meta
    assert Executor(sources={"next": b}).execute(parse('FROM $next RETURN HASH()')).values[0] == object_digest(target)


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
def test_explain_defers_core_and_verifiers(name, monkeypatch):
    import rexgraph.io.mutation as mutation
    import rexgraph.io.replication as replication
    executor, _ = engine(name, True)
    def fail(*args, **kwargs):
        pytest.fail("replay or provider ran during EXPLAIN")
    monkeypatch.setattr(mutation, "apply_mutation", fail)
    monkeypatch.setattr(replication, "apply_replication", fail)
    monkeypatch.setattr(ArtifactServices, "provider", fail)
    result = executor.execute(parse(f'EXPLAIN FROM $r RETURN {name}($blob,"parent",$p)'))
    assert result.execution == ()


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
def test_signature_verification_runs_for_each_call(name):
    executor, _ = engine(name, True)
    original = executor.artifacts.provider("verifiers", "author")
    calls = []
    class Verifier:
        signer_id = "author"
        def verify(self, payload, signature):
            calls.append(True)
            return original.verify(payload, signature)
    executor.artifacts = ArtifactServices(verifiers={"author": Verifier()})
    q = parse(f'FROM $r RETURN {name}($blob,"parent",$p),{name}($blob,"parent",$p)')
    executor.execute(q)
    assert len(calls) == 4
    executor.execute(q)
    assert len(calls) == 8


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
def test_bad_provider_identity_is_not_used(name):
    executor, _ = engine(name, True)
    original = executor.artifacts.provider("verifiers", "author")
    executor.artifacts = ArtifactServices(verifiers={"wrong": original})
    with pytest.raises(ValueError, match="identity differs"):
        executor.execute(parse(f'FROM $r RETURN {name}($blob,"parent",$p)'))


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
@pytest.mark.parametrize("failure", ["parent", "root", "state", "bytes", "provider", "downgrade"])
def test_replay_rejects_wrong_endpoints_and_missing_provider(name, failure):
    executor, target = engine(name, True)
    executor.params["parent"] = "other" if failure == "parent" else None if failure == "root" else "parent"
    if failure == "state":
        executor.sources["r"] = target
    elif failure == "bytes":
        executor.params["blob"] = executor.params["blob"][:-1] + bytes([executor.params["blob"][-1] ^ 1])
    elif failure == "provider":
        executor.artifacts = None
    elif failure == "downgrade":
        executor.params["p"] = None
    with pytest.raises((TypeError, ValueError)):
        executor.execute(parse(f'FROM $r RETURN {name}($blob,$parent,$p)'))


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
@pytest.mark.parametrize("options", [{"unknown": 1}, {"require_transition_signature": 1},
    {"allowed_signers": "author"}, {"allowed_signers": ["a", "a"]}, {"allowed_signers": [""]}])
def test_bad_policy_refused_during_explain(name, options):
    executor, _ = engine(name)
    executor.params["p"] = options
    with pytest.raises((TypeError, ValueError)):
        executor.execute(parse(f'EXPLAIN FROM $r RETURN {name}($blob,"parent",$p)'))


def test_temporal_signal_is_not_a_canonical_mutation():
    timeline = TemporalRex([])
    timeline.append_snapshot(RexGraph.from_graph([0], [1]))
    timeline.append_snapshot(RexGraph.from_graph([1], [2]))
    delta = Executor(sources={"t": timeline}).execute(parse('FROM $t RETURN TEMPORAL_DELTA(1)')).values[0]
    executor = Executor(sources={"r": timeline.at(0)}, params={"d": delta})
    with pytest.raises(TypeError):
        executor.execute(parse('EXPLAIN FROM $r RETURN APPLY_DELTA($d)'))


@pytest.mark.parametrize("name", ["APPLY_DELTA", "REPLICATE"])
def test_replayed_state_is_published_only_by_explicit_rcdb_commit(tmp_path, name):
    from rcdb import RexStore
    executor, target = engine(name)
    path = str(tmp_path / "db")
    store = RexStore(path).configure_security(require_commits=True)
    try:
        prior = executor.sources["r"]
        Executor(sources={"db": store}, params={"r": prior}).execute(parse(
            'FROM $db MUTATE "r" SET state=$r,expected_version=0 COMMIT'))
        candidate = executor.execute(parse(f'FROM $r RETURN {name}($blob,"parent",$p)')).values[0]
        assert store.read_record("r").state_digest == object_digest(prior)
        Executor(sources={"db": store}, params={"r": candidate}).execute(parse(
            'FROM $db MUTATE "r" SET state=$r,expected_version=1 COMMIT'))
        assert store.verify_commits("r")
    finally:
        store.close()
    store = RexStore(path)
    try:
        result = Executor(sources={"db": store}).execute(parse('FROM $db RETURN RCDB_GET("r")')).values[0]
        assert object_digest(result) == object_digest(target)
        assert store.verify_commits("r")
    finally:
        store.close()
