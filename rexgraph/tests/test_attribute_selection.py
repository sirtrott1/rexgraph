"""Selecting on what a source SAID, beside selecting on what the complex computes.

`select` reads quantities the complex computes and its docstring says so: "over
quantities the complex computes rather than over stored attributes". The stored half was
missing. `select_by_attribute` returns the same uint8 mask, so the two compose through
`select_and`/`select_or`/`select_not` without either knowing about the other.

`subcomplex_by_criteria` read `_cell_metadata[1]` directly and was relation-only, so a
vertex attribute could be stored and never filtered on. Both now go through one criteria
evaluator, so the mask form and the subcomplex form cannot drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph


@pytest.fixture
def rex():
    """A path with attributes at every grade."""
    r = RexGraph(sources=np.array([0, 1, 2], dtype=np.int32),
                 targets=np.array([1, 2, 0], dtype=np.int32))
    r.add_faces([np.array([0, 1, 2], dtype=np.int32)], signs=None)
    r._ensure_clean()
    for v, element in enumerate(["C", "C", "N"]):
        r.attach_metadata(0, v, "element", element)
    r.attach_metadata(0, 0, "score", 10.0)
    r.attach_metadata(0, 1, "score", 2.0)
    for e, order in enumerate([1, 2, 1]):
        r.attach_metadata(1, e, "bond_order", order)
    r.attach_metadata(2, 0, "ring", "aromatic")
    return r


#### the criteria forms


def test_equality(rex):
    assert rex.select_by_attribute({"element": "C"}, dim=0).tolist() == [1, 1, 0]


def test_a_bound(rex):
    assert rex.select_by_attribute({"score": {"min": 5}}, dim=0).tolist() == [1, 0, 0]
    assert rex.select_by_attribute({"score": {"max": 5}}, dim=0).tolist() == [0, 1, 0]


def test_membership(rex):
    assert rex.select_by_attribute({"element": ["C", "N"]}, dim=0).tolist() == [1, 1, 1]


def test_presence(rex):
    """Whatever the value, so long as the key is there."""
    assert rex.select_by_attribute({"score": None}, dim=0).tolist() == [1, 1, 0]


def test_a_missing_key_never_matches(rex):
    """Absence and a non-matching value are different, and a filter treating them alike
    would select the cells nobody has said anything about."""
    assert rex.select_by_attribute({"element": "C"}, dim=1).tolist() == [0, 0, 0]


def test_several_criteria_are_conjunctive(rex):
    assert rex.select_by_attribute({"element": "C", "score": {"min": 5}},
                                   dim=0).tolist() == [1, 0, 0]


#### every grade


def test_it_reads_relations(rex):
    assert rex.select_by_attribute({"bond_order": 2}, dim=1).tolist() == [0, 1, 0]


def test_it_reads_faces(rex):
    assert rex.select_by_attribute({"ring": "aromatic"}, dim=2).tolist() == [1]


def test_an_unknown_grade_is_refused(rex):
    with pytest.raises(ValueError, match="dim must be"):
        rex.select_by_attribute({"element": "C"}, dim=7)


#### it composes with the computed half


def test_the_two_kinds_of_fact_combine(rex):
    """One expression over a stored attribute and a computed quantity, because both are
    the same mask."""
    stored = rex.select_by_attribute({"bond_order": 1}, dim=1)
    computed = rex.select("chi", ">", 0.0)
    assert rex.select_and(stored, computed).tolist() == [1, 0, 1]
    assert rex.select_not(stored).tolist() == [0, 1, 0]


def test_the_mask_is_the_same_type_as_select(rex):
    assert (rex.select_by_attribute({"bond_order": 1}, dim=1).dtype
            == rex.select("chi", ">", 0.0).dtype)


#### the subcomplex form agrees with the mask form


def test_a_relation_selection_keeps_those_relations(rex):
    assert rex.subcomplex_by_criteria({"bond_order": 1}, dim=1).nE == 2


def test_a_vertex_selection_is_the_INDUCED_subcomplex(rex):
    """The relations whose WHOLE boundary matches, which is the only reading that does
    not leave a relation with an endpoint outside the selection."""
    sub = rex.subcomplex_by_criteria({"element": "C"}, dim=0)
    assert sub.nE == 1, "only the C-C relation is induced"


def test_the_two_forms_use_one_evaluator(rex):
    mask = rex.select_by_attribute({"bond_order": 1}, dim=1)
    assert rex.subcomplex_by_criteria({"bond_order": 1}, dim=1).nE == int(mask.sum())


def test_a_face_selection_is_refused_with_a_reason(rex):
    """A set of faces does not determine a set of relations without a choice."""
    with pytest.raises(ValueError, match="does not determine a subcomplex"):
        rex.subcomplex_by_criteria({"ring": "aromatic"}, dim=2)


def test_a_complex_with_no_attributes_selects_nothing(rex):
    bare = RexGraph(sources=np.array([0], dtype=np.int32),
                    targets=np.array([1], dtype=np.int32))
    assert bare.select_by_attribute({"element": "C"}, dim=0).tolist() == [0, 0]
