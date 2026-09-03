"""A relation's support is a set of participants, so no column names one of them twice.

The vocabulary is many to one: two terms of a record that carry the same string resolve to
the same vertex. Emitting one participation per TERM therefore produced a column naming
that vertex twice, which is not a boundary. The complex rejects it now, but before that
check existed the duplicate was silently collapsed, so an index could be written that no
longer described the corpus it came from.

Multiplicity is not incidence: a record mentioning a term twice participates once.
"""
from __future__ import annotations

import numpy as np
import pytest
from rcdb import index as ix
from rcdb.core import ComplexRecord


def _rec(rid, tags, labels):
    return (rid, ComplexRecord(id=rid, created=0.0,
                               signature={"source": "s", "tags": list(tags)},
                               meta={"vertex_labels": list(labels)}))


def _relations(index):
    ptr = np.asarray(index["rel_ptr"], dtype=np.int64)
    idx = np.asarray(index["rel_idx"], dtype=np.int64)
    return [idx[ptr[e]:ptr[e + 1]].tolist() for e in range(ptr.size - 1)]


def test_a_repeated_label_is_one_participant():
    """Two vertices sharing a display label are one vocabulary vertex, named once."""
    index = ix.build([_rec("r0", ["t0", "t1"], ["gene1", "gene1"]),
                      _rec("r1", ["t1", "t2"], ["gene2", "gene3"])])

    for e, support in enumerate(_relations(index)):
        assert len(set(support)) == len(support), (
            f"relation {e} names a participant twice: {support}"
        )


def test_the_deduped_relation_keeps_the_record_and_its_distinct_terms():
    """Dedupe drops the repeat, not the participation: arity falls by exactly one."""
    index = ix.build([_rec("r0", ["t0"], ["gene1", "gene1"])])
    relations = _relations(index)
    n = index["n"]
    labels = next(r for r in relations if len(r) > 1 and r[0] == 0 and len(r) == 2)

    assert labels[0] == 0, "the record stays the distinguished participant"
    assert labels[1] >= n, "its one distinct term is the other participant"


def test_first_seen_order_is_preserved():
    """Dedupe must not reorder: the head is positional, so order carries meaning."""
    index = ix.build([_rec("r0", ["t0"], ["b", "a", "b", "c"])])
    n = index["n"]
    codes = {t: n + i for i, t in enumerate(index["vocab"])}
    support = next(r for r in _relations(index) if r[0] == 0 and len(r) == 4)

    assert support == [0, codes["b"], codes["a"], codes["c"]]


def test_the_index_still_builds_a_valid_complex():
    """The end to end claim: a corpus with repeated labels becomes a usable complex."""
    from rexgraph.graph import RexGraph  # noqa: F401  the complex is the point of the index

    index = ix.build([_rec("r0", ["t0", "t1"], ["gene1", "gene1"]),
                      _rec("r1", ["t1", "t2"], ["gene2", "gene3"])])
    rex = ix.complex_of(index)

    assert int(rex.nE) == len(_relations(index))
    assert bool(rex.chain_valid)


def _legacy(index, relation):
    """An index as the old build would have written it: one participant repeated.

    Built by reintroducing the duplicate into a correct index rather than by pinning a
    stored file, so the fixture cannot drift away from the current on disk layout.
    """
    ptr = np.asarray(index["rel_ptr"], np.int64).tolist()
    idx = np.asarray(index["rel_idx"], np.int64).tolist()
    at = ptr[relation + 1]
    idx = idx[:at] + [idx[at - 1]] + idx[at:]
    ptr = ptr[:relation + 1] + [p + 1 for p in ptr[relation + 1:]]
    out = dict(index)
    out["rel_ptr"] = np.asarray(ptr, np.int64)
    out["rel_idx"] = np.asarray(idx, np.int64)
    return out


def test_a_legacy_index_is_refused_with_the_repair_path_named():
    """It fails at complex_of, not at read, and the message says what to do about it."""
    index = _legacy(ix.build([_rec("r0", ["t0", "t1"], ["gene1", "gene2"])]), 0)

    with pytest.raises(ValueError) as caught:
        ix.complex_of(index)

    message = str(caught.value)
    assert "repeats a C0 participant" in message, "the cause is still stated"
    assert "rcdb.index.repair" in message, "the remedy is named"


def test_repair_returns_a_copy_and_leaves_the_original_alone():
    original = _legacy(ix.build([_rec("r0", ["t0", "t1"], ["gene1", "gene2"])]), 0)
    before = np.asarray(original["rel_idx"], np.int64).copy()

    repaired, report = ix.repair(original)

    assert np.array_equal(np.asarray(original["rel_idx"], np.int64), before), (
        "repair must not mutate the index it was handed"
    )
    assert report["relations_repaired"] == 1
    assert report["participants_removed"] == 1
    assert report["relations"] == (0,)
    assert repaired is not original


def test_repair_preserves_the_relation_count_and_first_seen_order():
    """Only repeats are removed. The head is positional, so order is part of the data."""
    index = ix.build([_rec("r0", ["t0"], ["b", "a", "c"])])
    n_relations = len(_relations(index))
    legacy = _legacy(index, 1)

    repaired, _report = ix.repair(legacy)

    assert len(_relations(repaired)) == n_relations, "no relation is dropped"
    assert _relations(repaired) == _relations(index), "order and support are restored"


def test_the_full_migration_round_trip(tmp_path):
    """legacy on disk -> read -> refusal -> repair -> write -> a chain valid complex."""
    clean = ix.build([_rec("r0", ["t0", "t1"], ["gene1", "gene2"]),
                      _rec("r1", ["t1", "t2"], ["gene3", "gene4"])])
    path = str(tmp_path / "legacy.safetensors")
    ix.write(path, _legacy(clean, 2))

    stored = ix.read(path)
    assert len(_relations(stored)) == len(_relations(clean)), "read returns it as stored"
    with pytest.raises(ValueError, match="rcdb.index.repair"):
        ix.complex_of(stored)

    repaired, report = ix.repair(stored)
    assert report["participants_removed"] == 1

    out = str(tmp_path / "repaired.safetensors")
    ix.write(out, repaired)
    rex = ix.complex_of(ix.read(out))

    assert bool(rex.chain_valid)
    assert int(rex.nE) == len(_relations(clean))
    for e, support in enumerate(_relations(ix.read(out))):
        assert len(set(support)) == len(support), f"relation {e} still repeats"


def test_repair_is_a_no_op_on_an_index_that_is_already_correct():
    index = ix.build([_rec("r0", ["t0", "t1"], ["gene1", "gene2"])])

    repaired, report = ix.repair(index)

    assert report == {"relations_repaired": 0, "participants_removed": 0, "relations": ()}
    assert _relations(repaired) == _relations(index)
