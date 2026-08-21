"""The hybrid tree: semantic levels outside, binary trees inside.

The claims under test are that the path names real layers, that the proof costs log of
each fanout rather than all of it, that the interior nodes ARE the layer digests, and
that the whole thing rests on the base layer being a partition.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.merkle import (build_merkle, layer_chain, verify_proof)
from rexgraph.sectioning import add_coarsening, add_sectioning, sectionings_of


@pytest.fixture
def doc():
    """8 relations -> 4 sentences -> 2 paragraphs -> 1 chapter."""
    r = RexGraph(sources=[0, 1, 2, 3, 4, 5, 6, 7], targets=[1, 2, 3, 4, 5, 6, 7, 0])
    add_sectioning(r, "sentence",
                   {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5], "s3": [6, 7]},
                   spans={"s0": (0, 10), "s1": (10, 10), "s2": (20, 10),
                          "s3": (30, 10)})
    add_coarsening(r, "paragraph", "sentence", [0, 0, 1, 1], ["p0", "p1"])
    add_coarsening(r, "chapter", "paragraph", [0, 0], ["c0"])
    return r


def test_the_chain_is_discovered_from_refines(doc):
    assert layer_chain(sectionings_of(doc)) == ["sentence", "paragraph", "chapter"]


def test_a_named_base_is_required_when_several_layers_own_cells(doc):
    add_sectioning(doc, "speaker", {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]})
    with pytest.raises(ValueError, match="name one as base="):
        layer_chain(sectionings_of(doc))
    assert layer_chain(sectionings_of(doc), "speaker") == ["speaker"]


def test_every_proof_verifies_against_the_root(doc):
    m = build_merkle(doc)
    for i in range(len(m.leaves)):
        assert verify_proof(m.leaves[i], m.proof(i), m.root)


def test_a_wrong_leaf_does_not_verify(doc):
    m = build_merkle(doc)
    assert not verify_proof(m.leaves[1], m.proof(0), m.root)


def test_a_tampered_sibling_does_not_verify(doc):
    m = build_merkle(doc)
    pr = m.proof(0)
    bad = [(b"\x01" * 32, r, lay) for (_s, r, lay) in pr[:1]] + list(pr[1:])
    assert not verify_proof(m.leaves[0], bad, m.root)


def test_the_path_names_the_layers_it_climbs(doc):
    """The whole reason for the hybrid: a verifier can say WHICH paragraph and WHICH
    chapter it went through, not merely that a path exists."""
    m = build_merkle(doc)
    climbed = [lay for _s, _r, lay in m.proof(0)]
    assert climbed and set(climbed) <= {"sentence", "paragraph", "chapter"}
    assert climbed == sorted(climbed, key=["sentence", "paragraph", "chapter"].index)


def test_the_interior_nodes_are_the_layer_digests(doc):
    """Coarsening digests are not computed separately, they are this tree."""
    m = build_merkle(doc)
    assert len(m.roots["paragraph"]) == 2
    assert len(m.roots["chapter"]) == 1
    assert m.roots["chapter"][0] == m.root
    assert m.layer_digest("sentence", 0) == m.leaves[0]
    assert m.layer_digest("paragraph", 1) == m.roots["paragraph"][1]


def test_the_proof_is_logarithmic_in_each_fanout_not_linear(doc):
    """A wide sibling set must cost log2 of it. Flat siblings would cost all 16."""
    r = RexGraph(sources=list(range(32)), targets=[(i + 1) % 32 for i in range(32)])
    add_sectioning(r, "sentence", {f"s{i}": [i] for i in range(32)})
    add_coarsening(r, "paragraph", "sentence", [0] * 32, ["p0"])
    m = build_merkle(r)
    assert len(m.proof(0)) == 5, "32 siblings must cost log2(32)=5, not 32"


def test_changing_one_leaf_changes_the_root(doc):
    before = build_merkle(doc).root
    doc._sectionings["sentence"].labels[2] = "renamed"
    assert build_merkle(doc).root != before


def test_a_cover_is_refused_because_a_leaf_would_have_two_parents(doc):
    """The coupling to the partition decision, pinned. Under a cover the tree is not
    well defined and building one anyway would certify a shape that is not there."""
    r = RexGraph(sources=[0, 1, 2, 3], targets=[1, 2, 3, 0])
    add_sectioning(r, "sentence", {"s0": [0, 1, 2], "s1": [2, 3]})
    with pytest.raises(ValueError, match="cover, not a partition"):
        build_merkle(r)


def test_a_complex_with_no_sectionings_has_no_tree():
    r = RexGraph(sources=[0, 1], targets=[1, 0])
    with pytest.raises(ValueError, match="no sectionings"):
        build_merkle(r)


def test_the_root_survives_the_state_round_trip_and_rebuilds(doc):
    st = to_state(doc)
    assert st.header["merkle"]["chain"] == ["sentence", "paragraph", "chapter"]
    assert st.header["merkle"]["n_leaves"] == 4
    back = from_state(st, verify=True)
    assert build_merkle(back).root.hex() == st.header["merkle"]["root"]
    assert [bytes(x) for x in back._merkle_leaves] == build_merkle(doc).leaves


def test_no_digest_is_stored_because_every_one_of_them_is_derived(doc):
    """Leaves and interior alike. `_leaf_digests` builds from the labels, the spans and
    the boundary columns, all of which are already tensors here, so a stored digest is a
    cached hash of its own neighbours, and digests are the one thing in the file at
    full entropy, so they survive compression whole and dominate the compressed size."""
    st = to_state(doc)
    assert not [k for k in st.tensors if "merkle" in k], sorted(st.tensors)
    assert set(st.header["merkle"]) == {"chain", "root", "n_leaves"}


def test_a_proof_still_travels_carrying_its_own_leaf(doc):
    """A verifier needs the leaf, the path and the root, which is what a Merkle proof
    has always been. What changed is that the leaf is derived from the bundle rather
    than read out of it; verification does not involve the source text either way."""
    m = build_merkle(doc)
    root = bytes.fromhex(to_state(doc).header["merkle"]["root"])
    assert verify_proof(m.leaves[2], m.proof(2), root)
    assert not verify_proof(m.leaves[1], m.proof(2), root)


def test_the_leaf_commits_to_orientation_not_only_to_support(doc):
    """Re-signing a column leaves the support alone, so a digest over the support would
    miss it. Orientation is the content of the boundary, so the leaf carries the data."""
    before = build_merkle(doc).leaves[0]
    doc._B1_dual = None                      # force a rebuild from the flipped edge
    r = RexGraph(sources=[1, 0, 2, 3, 4, 5, 6, 7], targets=[0, 2, 3, 4, 5, 6, 7, 1])
    add_sectioning(r, "sentence",
                   {"s0": [0, 1], "s1": [2, 3], "s2": [4, 5], "s3": [6, 7]},
                   spans={"s0": (0, 10), "s1": (10, 10), "s2": (20, 10),
                          "s3": (30, 10)})
    add_coarsening(r, "paragraph", "sentence", [0, 0, 1, 1], ["p0", "p1"])
    add_coarsening(r, "chapter", "paragraph", [0, 0], ["c0"])
    assert build_merkle(r).leaves[0] != before


#### the loader actually checks, rather than saying it does ####################

def test_a_rewritten_boundary_column_is_caught_at_load(doc):
    """This is what deriving the leaves buys. The check is now 'does this complex make
    this root', so re-signing a column (which leaves the support untouched and passes
    a refreshed container digest) fails it. Comparing stored leaves to a stored root
    could not see this at all."""
    from rexgraph.io.rex_state import state_digest
    st = to_state(doc)
    idx = np.array(st.tensors["boundary_idx"])
    idx[0], idx[1] = idx[1], idx[0]          # flip one column's orientation
    st.tensors["boundary_idx"] = idx
    st.header["digest"] = state_digest(st.tensors, st.header["digest_names"])
    with pytest.raises(ValueError, match="does not hash to the stored Merkle root"):
        from_state(st, verify=True)


def test_a_rewritten_span_is_caught_by_the_closer_guard(doc):
    """A leaf commits to where the section lives, so the root would catch this too,
    but the sectioning carries its own digest and that one fires first. Worth pinning
    which guard speaks: two checks covering one tamper is defence, one check believed to
    cover a tamper it does not reach is not."""
    from rexgraph.io.rex_state import state_digest
    st = to_state(doc)
    sp = np.array(st.tensors["sections/sentence/spans"])
    sp[1, 0] += 1
    st.tensors["sections/sentence/spans"] = sp
    st.header["digest"] = state_digest(st.tensors, st.header["digest_names"])
    with pytest.raises(ValueError, match="does not match its digest"):
        from_state(st, verify=True)


def test_a_rewritten_root_is_caught_at_load(doc):
    from rexgraph.io.rex_state import state_digest
    st = to_state(doc)
    st.header["merkle"] = {**st.header["merkle"], "root": "00" * 32}
    st.header["digest"] = state_digest(st.tensors, st.header["digest_names"])
    with pytest.raises(ValueError, match="does not hash to the stored Merkle root"):
        from_state(st, verify=True)


def test_the_check_rebuilds_from_the_complex(doc):
    """'Does this complex make this root'. Passing leaves in is still supported, because
    that is how a rebuild of the interior alone is expressed."""
    m = build_merkle(doc)
    assert build_merkle(doc, leaves=m.leaves).root == m.root
    with pytest.raises(ValueError, match="leaves for"):
        build_merkle(doc, leaves=m.leaves[:-1])


def _layered_rex(n_base=64):
    """`n_base` relations -> n_base/2 sentences -> n_base/4 paragraphs -> 1 chapter."""
    n = int(n_base)
    r = RexGraph(sources=list(range(n)), targets=[(i + 1) % n for i in range(n)])
    add_sectioning(r, "sentence",
                   {f"s{i}": [2 * i, 2 * i + 1] for i in range(n // 2)},
                   spans={f"s{i}": (10 * i, 10) for i in range(n // 2)})
    add_coarsening(r, "paragraph", "sentence", [i // 2 for i in range(n // 2)],
                   [f"p{i}" for i in range(n // 4)])
    add_coarsening(r, "chapter", "paragraph", [0] * (n // 4), ["c0"])
    return r


#### the header carries structure, not digests ##################################

def test_interior_nodes_are_not_written_into_the_json_header():
    """`pack_merkle` says it stores the leaves and the root because the interior is a
    pure function of them. It also used to write every coarser layer's roots as hex
    strings into the header, which is the same interior it says it omits, at two hex
    characters per byte and then serialised twice. On a large document that alone
    exceeded safetensors' 100 MB header limit and the document could not be stored."""
    import json

    from rexgraph.io.rex_state import to_state

    rex = _layered_rex(n_base=256)
    st = to_state(rex)
    h = json.dumps(st.header)
    assert "layer_roots" not in st.header
    # the header is STRUCTURE: it must not scale with the number of sections
    small = json.dumps(to_state(_layered_rex(n_base=32)).header)
    assert len(h) - len(small) < 512, (
        f"header grew {len(h) - len(small)} bytes for 224 more sections")


def test_the_layer_digests_still_resolve_after_a_round_trip():
    """Dropping the stored roots is only safe because `build_merkle` recomputes them."""
    from rexgraph.merkle import build_merkle

    rex = _layered_rex(n_base=64)
    m = build_merkle(rex)
    chain = list(m.chain)
    assert len(chain) > 1
    coarser = chain[1]
    before = m.layer_digest(coarser, 0)
    m2 = build_merkle(rex, leaves=list(m.leaves))
    assert m2.layer_digest(coarser, 0) == before
    assert m2.root == m.root
