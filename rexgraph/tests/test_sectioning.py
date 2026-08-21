"""Sectionings: partitions of ONE field, carried as a cochain and digested per layer.

The claim is that a document's layers do not need nested complexes. They are partitions
of the document's own field, several of them coexist over it, and each one survives
serialisation with a digest a caller can check without opening the complex.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.sectioning import (Sectioning, add_sectioning, drop_sectioning,
                                 sectioning_summary, sectionings_of)


@pytest.fixture
def rex():
    # two triangles sharing a vertex: 6 relations, so a sectioning has something to cut
    src = [0, 1, 2, 2, 3, 4]
    tgt = [1, 2, 0, 3, 4, 2]
    return RexGraph(sources=src, targets=tgt)


def test_a_partition_reports_itself_as_one(rex):
    s = add_sectioning(rex, "half", {"lo": [0, 1, 2], "hi": [3, 4, 5]})
    assert s.is_partition()
    assert s.n_sections == 2
    assert list(s.cells(0)) == [0, 1, 2]


def test_a_cover_is_not_a_partition_and_says_so(rex):
    """Sentence spans really are a cover: a pair recurring in two sentences is in both."""
    s = add_sectioning(rex, "overlap", {"x": [0, 1, 2], "y": [2, 3, 4, 5]})
    assert not s.is_partition()


def test_a_gap_is_not_a_partition_either(rex):
    s = add_sectioning(rex, "gap", {"x": [0, 1]})
    assert not s.is_partition()


def test_the_owner_cochain_is_the_partition_written_as_a_grade_one_field(rex):
    add_sectioning(rex, "half", {"lo": [0, 1, 2], "hi": [3, 4, 5]})
    owner = sectionings_of(rex)["half"].owner_cochain()
    assert owner.shape == (int(rex.nE),)
    assert list(owner) == [0, 0, 0, 1, 1, 1]


def test_a_sectioning_cannot_name_a_cell_the_complex_does_not_have(rex):
    with pytest.raises(ValueError, match="does not introduce"):
        add_sectioning(rex, "bad", {"x": [0, 999]})


def test_several_sectionings_coexist_over_one_field(rex):
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    add_sectioning(rex, "paragraph", {"p0": [0, 1, 2, 3], "p1": [4, 5]})
    got = sectionings_of(rex)
    assert set(got) == {"sentence", "paragraph"}
    assert got["sentence"].is_partition() and got["paragraph"].is_partition()


def test_a_coarsening_keeps_the_same_cells(rex):
    """A paragraph layer is a coarsening of the sentence layer over the SAME field, not a
    different complex, so the two cover exactly the same cells."""
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    add_sectioning(rex, "paragraph", {"p0": [0, 1, 2, 3], "p1": [4, 5]})
    got = sectionings_of(rex)
    assert sorted(got["sentence"].indices) == sorted(got["paragraph"].indices)


def test_the_summary_is_queryable_without_opening_the_complex(rex):
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3, 4, 5]},
                   method="agreement>=2")
    (s,) = sectioning_summary(rex)
    assert s["name"] == "sentence" and s["n_sections"] == 2
    assert s["is_partition"] and s["max_section"] == 4 and s["min_section"] == 2
    assert s["method"] == "agreement>=2"


def test_sectionings_survive_the_state_round_trip_with_their_own_digests(rex):
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]},
                   spans={"s0": (0, 10), "s1": (10, 12), "s2": (22, 8)},
                   method="agreement>=2")
    add_sectioning(rex, "paragraph", {"p0": [0, 1, 2, 3], "p1": [4, 5]})
    st = to_state(rex)
    entries = {e["name"]: e for e in st.header["sectionings"]}
    assert set(entries) == {"sentence", "paragraph"}
    assert all(e["digest"] for e in entries.values())
    assert entries["sentence"]["digest"] != entries["paragraph"]["digest"]

    back = from_state(st, verify=True)
    got = sectionings_of(back)
    assert set(got) == {"sentence", "paragraph"}
    assert got["sentence"].method == "agreement>=2"
    assert list(got["sentence"].cells(1)) == [2, 3]
    assert got["sentence"].spans is not None
    assert tuple(got["sentence"].spans[0]) == (0, 10)
    assert got["paragraph"].spans is None


def test_a_tampered_layer_is_caught_by_its_own_digest(rex):
    """The per-layer digest has to add protection the container digest does not.

    Rewriting a layer AND refreshing the whole-state digest is exactly what a rewrite
    through a legitimate writer looks like, so the outer seal passes. The layer's own
    digest is what still says the layer is not what was written.
    """
    from rexgraph.io.rex_state import state_digest
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    st = to_state(rex)
    # `indptr` is stored as its first difference, which is the section sizes; rewriting
    # those regroups the cells. (`indices` is an exact arange here and the codec keeps it
    # as its endpoints, so there is no array of it to rewrite: see the codec test.)
    key = "sections/sentence/indptr"
    st.tensors[key] = np.asarray([0, 3, 3, 0], dtype=st.tensors[key].dtype)
    st.header["digest"] = state_digest(st.tensors, st.header["digest_names"])
    with pytest.raises(ValueError, match="does not match its digest"):
        from_state(st, verify=True)


def test_a_rewritten_codec_spec_is_caught_too(rex):
    """The codec spec is an input to reconstruction, so it is a tensor and not a header
    key: `state_digest` covers the tensors and nothing else. Rewriting an `arange`
    codec's start would otherwise hand the loader a different array with the container
    seal still checking out. The layer digest catches it in either case, which is the
    point of having both."""
    import json

    from rexgraph.io.rex_state import CODEC_TENSOR, state_digest
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    st = to_state(rex)
    spec = json.loads(bytes(np.asarray(st.tensors[CODEC_TENSOR]).tobytes()).decode())
    assert spec["sections/sentence/indices"]["c"] == "arange"
    spec["sections/sentence/indices"]["start"] = 3          # shift every index
    st.tensors[CODEC_TENSOR] = np.frombuffer(
        json.dumps(spec, sort_keys=True).encode(), dtype=np.uint8).copy()
    st.header["digest"] = state_digest(st.tensors, st.header["digest_names"])
    with pytest.raises(ValueError, match="does not match its digest"):
        from_state(st, verify=True)


def test_the_codec_spec_is_sealed_by_the_container_digest(rex):
    """Without refreshing the outer digest, the same edit is caught one layer earlier."""
    import json

    from rexgraph.io.rex_state import CODEC_TENSOR
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    st = to_state(rex)
    spec = json.loads(bytes(np.asarray(st.tensors[CODEC_TENSOR]).tobytes()).decode())
    spec["sections/sentence/indices"]["start"] = 3
    st.tensors[CODEC_TENSOR] = np.frombuffer(
        json.dumps(spec, sort_keys=True).encode(), dtype=np.uint8).copy()
    with pytest.raises(ValueError, match="do not match the digest"):
        from_state(st, verify=True)


def test_a_complex_with_no_sectionings_writes_no_header_key(rex):
    st = to_state(rex)
    assert "sectionings" not in st.header
    assert not sectionings_of(from_state(st, verify=True))


def test_dropping_a_sectioning_removes_it(rex):
    add_sectioning(rex, "sentence", {"s0": [0, 1, 2, 3, 4, 5]})
    assert drop_sectioning(rex, "sentence") is not None
    assert sectionings_of(rex) == {}


def test_as_sections_feeds_section_readings(rex):
    from rexgraph.partition import section_readings
    s = add_sectioning(rex, "half", {"lo": [0, 1, 2], "hi": [3, 4, 5]})
    rd = section_readings(rex, s.as_sections(), verify=False)
    assert set(rd) == {"lo", "hi"}
    assert all("mass" in v for v in rd.values())


def test_a_sectioning_is_reconstructible_from_its_own_pieces(rex):
    s = add_sectioning(rex, "half", {"lo": [0, 1, 2], "hi": [3, 4, 5]})
    again = Sectioning(s.name, s.grade, s.indptr, s.indices, s.labels,
                       n_cells=s.n_cells)
    assert again.as_sections() == s.as_sections()


#### coarsening: a layer that owns sections, not cells ##########################

def test_a_coarsening_stores_the_parent_map_not_the_memberships(rex):
    from rexgraph.sectioning import add_coarsening
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    c = add_coarsening(rex, "paragraph", "sentence", [0, 0, 1], ["p0", "p1"])
    assert c.is_derived and c.indices.size == 0
    assert list(c.parent) == [0, 0, 1]


def test_a_derived_layer_resolves_to_the_union_of_its_children(rex):
    from rexgraph.sectioning import add_coarsening
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    add_coarsening(rex, "paragraph", "sentence", [0, 0, 1], ["p0", "p1"])
    got = sectionings_of(rex)["paragraph"].as_sections(sectionings_of(rex))
    assert got == {"p0": [0, 1, 2, 3], "p1": [4, 5]}


def test_a_derived_layer_refuses_to_answer_cells_alone(rex):
    from rexgraph.sectioning import add_coarsening
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    c = add_coarsening(rex, "paragraph", "sentence", [0, 0, 1], ["p0", "p1"])
    with pytest.raises(ValueError, match="derived"):
        c.cells(0)


def test_a_coarsening_must_cover_every_finer_section_exactly_once(rex):
    from rexgraph.sectioning import add_coarsening
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    with pytest.raises(ValueError, match="exactly once"):
        add_coarsening(rex, "paragraph", "sentence", [0, 0], ["p0"])


def test_a_coarsening_cannot_refine_a_layer_that_is_not_there(rex):
    from rexgraph.sectioning import add_coarsening
    with pytest.raises(ValueError, match="not attached"):
        add_coarsening(rex, "paragraph", "sentence", [0], ["p0"])


def test_a_coarsening_survives_the_round_trip_and_still_resolves(rex):
    from rexgraph.sectioning import add_coarsening
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    add_coarsening(rex, "paragraph", "sentence", [0, 0, 1], ["p0", "p1"],
                   method="blank_line")
    st = to_state(rex)
    assert "sections/paragraph/indices" not in st.tensors, "derived layers store no CSR"
    assert "sections/paragraph/parent" in st.tensors
    back = from_state(st, verify=True)
    got = sectionings_of(back)
    assert got["paragraph"].refines == "sentence"
    assert got["paragraph"].as_sections(got) == {"p0": [0, 1, 2, 3], "p1": [4, 5]}


def test_nesting_three_deep_resolves_through_both_levels(rex):
    from rexgraph.sectioning import add_coarsening
    add_sectioning(rex, "sentence", {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5]})
    add_coarsening(rex, "paragraph", "sentence", [0, 0, 1], ["p0", "p1"])
    add_coarsening(rex, "chapter", "paragraph", [0, 0], ["c0"])
    got = sectionings_of(rex)
    assert got["chapter"].as_sections(got) == {"c0": [0, 1, 2, 3, 4, 5]}
