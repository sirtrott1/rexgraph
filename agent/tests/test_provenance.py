"""What an answer rests on, and that the readings are the theorems and not a score."""
import numpy as np
import pytest
from agent.rcdb import ComplexRecord

from agent import provenance as pv
from agent import rcdb_index as ix


def _rec(rid, source, tags, labels):
    return (rid, ComplexRecord(id=rid, created=0.0,
                               signature={"source": source, "tags": list(tags)},
                               meta={"vertex_labels": list(labels)}))


def _index(n=8, shared=3):
    """n records over a shared vocabulary, so the corpus has real cycles."""
    rows = [_rec(f"r{i}", "s", [f"t{i % shared}", f"t{(i + 1) % shared}"],
                 [f"v{i % shared}", f"v{i}"]) for i in range(n)]
    return ix.build(rows)


def test_rel_owner_is_derived_not_stored():
    """A built index and a read one are the same object.

    `rel_owner` used to be written by `read` and omitted by `build`, so a freshly built
    index could not answer `relations_of` at all. It is a read of position 0 of each
    span, so it is derived in one place and both paths agree by construction.
    """
    index = _index()
    own = ix.rel_owner(index)
    assert own.shape == (len(index["rel_ptr"]) - 1,)
    assert np.all(np.diff(own) >= 0), "relations are emitted in row order"
    for row in range(index["n"]):
        rels = list(ix.relations_of(index, row))
        assert rels, "every record here has at least one accession"
        assert all(own[e] == row for e in rels)


def test_rel_owner_survives_a_round_trip(tmp_path):
    index = _index()
    p = tmp_path / "i.safetensors"
    ix.write(str(p), index)
    back = ix.read(str(p))
    assert "rel_owner" not in back, "derived data is not stored"
    assert np.array_equal(ix.rel_owner(index), ix.rel_owner(back))
    for row in range(index["n"]):
        assert list(ix.relations_of(index, row)) == list(ix.relations_of(back, row))


def test_leverage_is_cached_against_the_digest():
    index = _index()
    rex1, lev1 = pv.index_leverage(index)
    rex2, lev2 = pv.index_leverage(index)
    assert rex1 is rex2 and lev1 is lev2
    index["state_digest"] = "changed"
    rex3, _ = pv.index_leverage(index)
    assert rex3 is not rex1, "a write invalidates the reading"


def test_provenance_readings_obey_their_theorems():
    index = _index(n=10, shared=4)
    rex, lev = pv.index_leverage(index)
    p = pv.store_provenance(index, [f"r{i}" for i in range(4)])
    assert p["n_records"] == 4 and not p["missing"]
    # Theorem 23/24, the same bounds partition.section_readings asserts
    assert p["mass"] <= p["own_rank"] + 1e-9
    assert p["own_cycles"] <= p["share"] + 1e-9
    assert 0.0 <= p["share_of_corpus"] <= 1.0
    assert p["n_irreplaceable"] == sum(1 for c in
                                       np.asarray(lev)[p["irreplaceable"]] if c > 1 - 1e-9)


def test_whole_corpus_holds_the_whole_rank():
    """Every record retrieved is the whole complex, so the mass IS rank(B1)."""
    index = _index(n=6, shared=3)
    rex, _ = pv.index_leverage(index)
    p = pv.store_provenance(index, [f"r{i}" for i in range(6)])
    assert p["n"] == rex.nE
    assert p["mass"] == pytest.approx(p["corpus_rank"], abs=1e-6)
    assert p["gap"] == pytest.approx(0.0, abs=1e-6), (
        "nothing is outside the whole, so nothing closes for it")


def test_a_bridge_is_irreplaceable_and_a_cycle_is_not():
    """R_eff = 1 exactly on a relation no other path reaches."""
    index = ix.build([_rec("a", "s", ["x"], ["p", "q"]),
                      _rec("b", "s", ["y"], ["p", "q"]),
                      _rec("c", "s", ["z"], ["lonely"])])
    rex, lev = pv.index_leverage(index)
    solo = list(ix.relations_of(index, 2))
    assert all(lev[e] == pytest.approx(1.0) for e in solo), (
        "record c shares no term, so every relation it owns is a bridge")
    p = pv.store_provenance(index, ["c"])
    assert p["n_irreplaceable"] == p["n"]
    assert p["own_cycles"] == 0


def test_missing_ids_are_reported_not_ignored():
    index = _index(n=4)
    p = pv.store_provenance(index, ["r0", "nope"])
    assert p["missing"] == ["nope"] and p["n_records"] == 1


def test_no_support_is_stated_plainly():
    index = _index(n=4)
    p = pv.store_provenance(index, ["nothing-here"])
    assert p["n"] == 0
    assert "No relation" in pv.format_provenance(p)


def test_hodge_split_needs_the_right_length():
    index = _index(n=5)
    rex, _ = pv.index_leverage(index)
    with pytest.raises(ValueError, match="response is"):
        pv.query_provenance(rex, [0, 1], response=np.zeros(3))
    p = pv.query_provenance(rex, [0, 1], response=np.ones(int(rex.nE)))
    h = p["hodge"]
    assert h["gradient"] >= 0 and h["curl"] >= 0 and h["unaccounted"] >= 0
    total = h["gradient"] ** 2 + h["curl"] ** 2 + h["unaccounted"] ** 2
    assert total == pytest.approx(1.0, abs=1e-6), "the decomposition is orthogonal"


def test_out_of_range_relation_refuses():
    index = _index(n=4)
    rex, _ = pv.index_leverage(index)
    with pytest.raises(IndexError):
        pv.query_provenance(rex, [int(rex.nE) + 5])


def test_coupling_is_opt_in_because_it_costs_a_solve():
    """Every other reading is a lookup or a rank; this one solves the field.

    Measured on a real store it was 1.33s of a 1.35s query, so computing it unasked
    made every retrieval pay for a coordinate most callers never read.
    """
    index = _index(n=8, shared=3)
    p = pv.store_provenance(index, [f"r{i}" for i in range(4)])
    assert "coupling" not in p, "the default must not pay for a field solve"
    q = pv.store_provenance(index, [f"r{i}" for i in range(4)], coupling=True)
    assert "coupling" in q


def test_a_coupling_failure_is_reported_not_swallowed():
    """A bare except made a broken solve read the same as a section with no pairs."""
    index = _index(n=6, shared=3)
    rex, lev = pv.index_leverage(index)
    # at least three relations, or coupling_fraction returns NaN for having no pairs
    # to read and the field is never touched
    p = pv.query_provenance(rex, [0, 1, 2, 3], leverage=lev, coupling=True,
                            field=np.zeros((2, 2)))       # wrong shape on purpose
    assert np.isnan(p["coupling"])
    assert "coupling_error" in p, "the reason has to survive"


def test_the_corpus_rank_is_cached_on_the_index():
    """It is a property of the store, not of the query."""
    index = _index(n=8, shared=3)
    pv.index_leverage(index)
    assert "_corpus_rank" in index
    p = pv.store_provenance(index, ["r0", "r1"])
    assert p["corpus_rank"] == index["_corpus_rank"]
