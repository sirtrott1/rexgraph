"""Canonical replay owns full state and verifies actual prior identities."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.mutation import apply_mutation, prepare_mutation, mutation_from_bytes, mutation_to_bytes
from rexgraph.io.replication import apply_replication, pack_replication
from rexgraph.io.partition_state import partition_tower


def state(version=0):
    simplices = [list(combinations(range(5), grade + 1)) for grade in range(5)]
    cells = [6, simplices[1] + [(0, 1, 2), (0,), (0, 1, 2)]]
    for grade in range(2, 5):
        lower = {cell: i for i, cell in enumerate(simplices[grade - 1])}
        cells.append([[(lower[cell[:i] + cell[i+1:]], (-1)**i) for i in range(len(cell))]
                      for cell in simplices[grade]])
    weights = [1]*10 + [Q(1, 3) + version * Q(1, 2**100), 2**100 + version, Q(-2, 7)]
    rex = RexGraph.from_cells(cells, w_E=np.array(weights, object),
                             relation_ids=np.arange(11, 24), signs=[1]*11 + [-1, 1])
    rex._agent_meta = {"nested": {"version": version}}
    rex._signals = np.arange(13)
    rex.attach_metadata(1, 0, "version", version)
    return rex


def test_apply_preserves_full_canonical_state_without_aliasing():
    prior, target = state(), state(1)
    before = object_digest(prior)
    package = mutation_from_bytes(mutation_to_bytes(prepare_mutation(prior, target, tx_time=1)))
    saved = deepcopy(package)
    result = apply_mutation(package, previous=prior)
    assert object_digest(result) == object_digest(target)
    assert len(partition_tower(result)[0]) == 4
    np.testing.assert_array_equal(result.w_E, target.w_E)
    result._agent_meta["nested"]["version"] = 99
    result._signals[0] = 99
    result.w_E[0] = Q(99)
    assert object_digest(prior) == before
    assert package.resulting_state.header == saved.resulting_state.header
    for name in saved.resulting_state.tensors:
        np.testing.assert_array_equal(package.resulting_state.tensors[name], saved.resulting_state.tensors[name])


@pytest.mark.parametrize("failure", ["previous", "parent", "root", "result", "version", "type", "policy"])
def test_apply_rejects_invalid_state_and_options(failure):
    prior, target = state(), state(1)
    package = prepare_mutation(prior, target, tx_time=1, parent_digest="parent")
    kwargs = {"previous": prior, "parent_digest": "parent"}
    if failure == "previous":
        kwargs["previous"] = target
    elif failure in {"parent", "root"}:
        kwargs["parent_digest"] = "other" if failure == "parent" else None
    elif failure == "result":
        package.resulting_state.header["agent_meta"]["nested"]["version"] = 99
    elif failure == "version":
        package = replace(package, version=1)
    elif failure == "type":
        package = b"not a decoded mutation"
    else:
        kwargs["policy"] = {}
    with pytest.raises((TypeError, ValueError)):
        apply_mutation(package, **kwargs)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_replication_full_state_and_empty_stream_ownership(count):
    prior = state()
    current, parent, packages = prior, "checkpoint", []
    for step in range(count):
        target = state(step + 1)
        package = prepare_mutation(current, target, tx_time=step + 1, parent_digest=parent)
        packages.append(mutation_to_bytes(package))
        current, parent = target, package.link.digest
    blob, _ = pack_replication(b"checkpoint artifact", packages,
        checkpoint_state=object_digest(prior), checkpoint_commit="checkpoint")
    result = apply_replication(blob, checkpoint_loader=lambda _: prior, checkpoint_commit="checkpoint")
    assert object_digest(result.result) == object_digest(current)
    assert result.result is not prior
    result.result._agent_meta["nested"]["version"] = 99
    assert prior._agent_meta["nested"]["version"] == 0
    with pytest.raises(ValueError, match="explicit parent"):
        apply_replication(blob, checkpoint_loader=lambda _: pytest.fail("loader ran"), checkpoint_commit=None)
