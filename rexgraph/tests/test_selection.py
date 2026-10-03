"""Selections share native closure, carried state and checked original bases."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import Selection, Lineage, RexGraph, restrict, glue
from rexgraph.cells import CellSet, GradedCellPattern
from rexgraph.object_identity import object_digest
from rexgraph.partition_state import build_rex_partition


def tower():
    face = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    rex = RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]],
        [face, face], [difference, difference], [difference]], relation_ids=[91, 17, 53])
    rex.set_provenance({"vertex_labels": ["a", "b", "c", "isolated"], "exact": Q(1, 7)})
    rex.attach_metadata(1, 2, "exact", [Q(2, 7), 2**1000])
    rex.attach_metadata(3, 1, "upper", Q(3, 11))
    return rex


def test_public_multigrade_selection_matches_legacy_partition_and_keeps_isolates():
    source = tower()
    selected = Selection(source, {0: [3], 3: [1]})
    result = restrict(source, selected, carried_state="all")
    old = build_rex_partition(source, [0, 0, 0], v_mask=[0, 0, 0, 1],
                             grade_masks={3: [0, 1]}, carried_state="all")
    assert object_digest(result.rex) == object_digest(old.rex)
    assert result.cell_maps == old.cell_maps == ((0, 1, 2, 3), (0, 1, 2), (0, 1), (1,), ())
    assert result.source_sizes == (4, 3, 2, 2, 1)
    assert selected.request_digest == result.state.selection_digest
    assert result.lineage.verify(source, result.rex)
    assert result.old_to_new[3].tolist() == [-1, 0]
    assert result.rex.get_metadata(3, 0, "upper") == Q(3, 11)
    assert result.rex.provenance["vertex_labels"][-1] == "isolated"


def test_mask_and_index_inputs_have_one_canonical_selection_identity():
    source = tower()
    array = np.array([0, 1, 0], np.uint8)
    selected = Selection.from_masks(source, {1: array})
    array[:] = 0
    expected = Selection(source, {1: [1, 1]})
    assert selected.digest == expected.digest
    assert selected.indices[1] == (1,)
    assert Selection.from_record(selected.as_record(), source) == selected
    assert Selection.from_cells(GradedCellPattern(source, (CellSet(source, 1, (1,)),))).digest == selected.digest


@pytest.mark.parametrize("cells", [{True: []}, {1.5: []}, {"1": []}, {-1: []}, {5: []},
                                    {1: [0.5]}, {1: [True]}, {1: ["1"]}, {1: [-1]}, {1: [3]}])
def test_invalid_addresses_are_refused_without_coercion(cells):
    with pytest.raises((TypeError, ValueError)):
        Selection(tower(), cells)


@pytest.mark.parametrize("mask", [[256, 0, 0], [-1, 0, 0], [0.5, 0, 0], [float("nan"), 0, 0],
                                  [[1, 0, 0]], [1, 0], ["1", "0", "0"]])
def test_invalid_masks_are_refused_before_conversion(mask):
    with pytest.raises(ValueError):
        Selection.from_masks(tower(), {1: mask})


@pytest.mark.parametrize("bad", [True, 1.5, "1", Q(3, 2)])
def test_cell_facades_cannot_coerce_invalid_coordinates_before_selection(bad):
    from rexgraph import Cell
    source = tower()
    for constructor in (lambda: Cell(source, bad, 0), lambda: Cell(source, 1, bad),
                        lambda: CellSet(source, bad, (0,)), lambda: CellSet(source, 1, (bad,))):
        with pytest.raises(TypeError, match="integer"):
            constructor()


def test_repeated_pattern_grades_are_refused_instead_of_losing_the_first_set():
    source = tower()
    with pytest.raises(ValueError, match="one declared selection"):
        GradedCellPattern(source, (CellSet(source, 1, (0,)), CellSet(source, 1, (1,))))


def test_selection_requires_explicit_rebinding_and_refuses_changed_sources():
    source = tower()
    selected = Selection(source, {1: [0]})
    copy = source.copy()
    with pytest.raises(ValueError, match="another source"):
        restrict(copy, selected)
    assert selected.bind(copy).digest == selected.digest
    source.attach_metadata(1, 0, "new", Q(1, 13))
    with pytest.raises(ValueError, match="source changed"):
        restrict(source, selected)


@pytest.mark.parametrize("mode", ["structural", "all"])
def test_lineage_composes_exact_maps_and_verifies_the_root_projection(mode):
    source = tower()
    parent = restrict(source, Selection(source, {0: [3], 3: [1]}), carried_state=mode)
    child = restrict(parent.rex, Selection(parent.rex, {1: [2]}), carried_state=mode)
    composed = parent.lineage.compose(child.lineage)
    assert composed.cell_maps == ((0, 2), (2,), (), (), ())
    assert composed.old_to_new[0].tolist() == [0, -1, 1, -1]
    assert composed.parents == (parent.lineage.digest, child.lineage.digest)
    assert composed.verify(source, child.rex)
    assert Lineage.from_bytes(composed.to_bytes()) == composed
    with pytest.raises(ValueError, match="intermediate"):
        child.lineage.compose(parent.lineage)


def test_lineage_refuses_inconsistent_dimensions_and_tampered_content():
    source = tower()
    part = restrict(source, Selection(source, {1: [2]}))
    with pytest.raises(ValueError, match="basis"):
        replace(part.lineage, result_sizes=(1, 1, 0, 0, 0))
    changed = part.lineage.as_record()
    changed["policy_digest"] = "modified"
    with pytest.raises(ValueError, match="digest mismatch"):
        Lineage.from_record(changed)
    with pytest.raises(ValueError, match="reproduce"):
        replace(part.lineage, selection_digest="0"*64).verify(source, part.rex)


def test_glue_unions_verified_original_cells_and_records_parent_lineage():
    source = tower()
    a = restrict(source, Selection(source, {2: [0]}), carried_state="all")
    b = restrict(source, Selection(source, {0: [3], 2: [1]}), carried_state="all")
    united = glue([b, a, a], source=source)
    assert united.cell_maps == ((0, 1, 2, 3), (0, 1, 2), (0, 1), (), ())
    assert united.rex.relation_ids.tolist() == [91, 17, 53]
    assert united.lineage.verify(source, united.rex)
    assert united.lineage.parents == tuple(sorted({a.lineage.digest, b.lineage.digest}))
    a.rex.attach_metadata(1, 0, "edited", 1)
    with pytest.raises(ValueError, match="changed"):
        glue([a, b], source=source)


def test_glue_requires_explicit_mixed_policy_and_never_restores_omitted_state():
    source = tower()
    a = restrict(source, Selection(source, {1: [0]}), carried_state="all")
    b = restrict(source, Selection(source, {1: [1]}))
    with pytest.raises(ValueError, match="mixed"):
        glue([a, b], source=source)
    with pytest.raises(ValueError, match="omitted"):
        glue([a, b], source=source, carried_state="all")
    assert not glue([a, b], source=source, carried_state="structural").rex.provenance
    with pytest.raises(ValueError, match="state changed"):
        glue([a], source=RexGraph.from_graph([0], [1]))


def test_empty_tower_grades_and_empty_selections_are_preserved():
    source = RexGraph.from_cells([4, [[0, 1]]])
    empty = restrict(source, Selection(source))
    assert empty.source_sizes == (4, 1) and empty.cell_maps == ((), ())
    assert tuple(empty.old_to_new) == (0, 1)
    assert empty.lineage.verify(source, empty.rex)


def test_restriction_and_roundtrip_preserve_explicit_empty_face_grade():
    from rexgraph.state import from_state, to_state, semantic_header
    from rexgraph.partition_state import partition_tower
    source = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    part = restrict(source, Selection(source, {1: [0]}))
    assert part.lineage.result_sizes == (2, 1, 0)
    state = to_state(part.rex)
    assert semantic_header(state)["empty_face_grade"] is True
    restored = from_state(state)
    assert [boundary.shape for boundary in partition_tower(restored)[0]] == [(2, 1), (1, 0)]
    assert object_digest(restored) == part.state.result_state
    declared_empty = RexGraph.from_cells([3, [[0, 1]], []])
    assert restrict(declared_empty, Selection(declared_empty)).lineage.result_sizes == (0, 0, 0)


@pytest.mark.parametrize("indices", [((), (0, 65535, 65536, 69999)),
                                     ((0, 65536), (1,)), ((), ())])
def test_sparse_request_hash_matches_legacy_masks_with_bounded_allocation(indices, monkeypatch):
    from rexgraph.partition_state import _selection_digest, _selection_digest_indices
    sizes = (70000, 70000)
    masks = [np.zeros(size, np.uint8) for size in sizes]
    for mask, selected in zip(masks, indices, strict=True):
        mask[list(selected)] = 1
    expected = _selection_digest(masks)
    original = np.zeros
    def bounded(shape, *args, **kwargs):
        assert shape <= 65536
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, "zeros", bounded)
    assert _selection_digest_indices(sizes, indices) == expected
