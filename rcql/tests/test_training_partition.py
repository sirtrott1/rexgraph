"""Training selections are explicit Core partitions, not implicit fitting."""
from dataclasses import replace

import pytest

from rexgraph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.partition_state import partition_from_policy
from rcql import BoundSource, Executor, SourcePolicy, parse


def setup(permissions=("read", "identity", "train")):
    rex = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]], relation_ids=[17, 91])
    policy = {"source_state": object_digest(rex), "cells": [[1, [0]]]}
    authority = SourcePolicy.allow(*permissions)
    return rex, policy, authority, Executor(sources={"r": BoundSource(rex, authority)}, params={"p": policy})


def test_text_policy_members_and_core_equivalence():
    rex, policy, authority, engine = setup()
    result = engine.execute(parse('FROM $r LET p=TRAINING_PARTITION(policy=$p) '
                                 'RETURN p,p.rex,p.manifest,p.cell_maps,p.digest'))
    p, child, manifest, maps, digest = result.values
    expected = partition_from_policy(rex, policy, authority_digest=authority.digest)
    assert digest == expected.digest and maps == ((0, 1, 2), (0,))
    assert child is p.rex and child.relation_ids.tolist() == [17]
    assert manifest == expected.state.manifest()
    assert tuple(e.value for e in result.exactness) == (
        "structural", "structural", "structural", "integer", "structural")
    assert object_digest(rex) == policy["source_state"]


@pytest.mark.parametrize("permissions", [(), ("read",), ("read", "identity"),
    ("read", "train"), ("identity", "train")])
@pytest.mark.parametrize("explain", [False, True])
def test_each_capability_required_before_adapter(permissions, explain, monkeypatch):
    import rcql.executor
    _, _, _, engine = setup(permissions)
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises(PermissionError):
        engine.execute(replace(parse('FROM $r RETURN TRAINING_PARTITION($p)'), explain=explain))


def test_explain_validates_but_does_not_construct(monkeypatch):
    import rexgraph.io.partition_state as core
    _, _, _, engine = setup()
    monkeypatch.setattr(core, "build_rex_partition", lambda *a, **kw: pytest.fail("partition built"))
    result = engine.execute(parse('EXPLAIN FROM $r RETURN TRAINING_PARTITION($p)'))
    assert result.execution == ()
    assert result.values[0]["returns"][0]["result"]["kind"] == "RexPartition"


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("bad", [None, {}, {"source_state": "wrong", "cells": []},
    {"cells": [[1, [0]]]}, "all", [1]])
def test_bad_policy_refused_before_adapter(bad, explain, monkeypatch):
    import rcql.executor
    _, _, _, engine = setup()
    engine.params["p"] = bad
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((ValueError, TypeError)):
        engine.execute(replace(parse('FROM $r RETURN TRAINING_PARTITION($p)'), explain=explain))


def test_changed_source_policy_refused_at_execution_and_explain():
    rex, _, _, engine = setup()
    rex.add_edges([0], [3], relation_ids=[101])
    for prefix in ("", "EXPLAIN "):
        with pytest.raises(ValueError, match="source_state"):
            engine.execute(parse(prefix+'FROM $r RETURN TRAINING_PARTITION($p)'))


def test_rcdb_source_and_owned_child_roundtrip(tmp_path):
    import rcdb
    from contextlib import closing
    rex, policy, _, _ = setup()
    path = f"rex://{tmp_path / 'db'}"
    with closing(rcdb.open_store(path).configure_security(require_commits=True)) as store:
        engine = Executor(sources={"db": store}, params={"r": rex, "p": policy})
        engine.execute(parse('FROM $db MUTATE "r" SET state=$r,actor="Art" COMMIT'))
        result = engine.execute(parse('FROM RCDB_GET($db,"r") RETURN TRAINING_PARTITION($p)')).values[0]
        engine.params["child"] = result.rex
        engine.execute(parse('FROM $db MUTATE "training" SET state=$child,actor="Art" COMMIT'))
        assert store.read_record("r").record.version == 1 and store.verify_commits("training")
    with closing(rcdb.open_store(path)) as store:
        assert object_digest(store.get("training")) == result.state.result_state
        assert object_digest(store.get("r")) == policy["source_state"]
        assert store.get("training").relation_ids.tolist() == [17]
