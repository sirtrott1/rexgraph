from fractions import Fraction as Q

import numpy as np
import pytest

from rcql import Executor, parse, mutation, param
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.io.catalog import FileCatalog, object_digest
from rexgraph.io import save
from rexgraph.structural_edit import edit_relations


def branching():
    return RexGraph.from_hypergraph([0, 2, 5, 9], [0, 1, 1, 2, 3, 0, 2, 3, 4],
                                   w_E=[Q(1, 3), Q(2, 7), Q(5, 11)])


def test_match_preserves_native_cells_exact_readings_and_pagination():
    r = branching()
    executor = Executor(sources={"r": r})
    text = 'FROM $r MATCH e IN CELLS(1) WHERE ARITY(e) >= 3 AND e.index != 1 RETURN e, ARITY(e) ORDER BY e.index DESC LIMIT 1'
    result = executor.execute(parse(text))
    cell, arity = result.values[0][0]
    assert cell.source is r and cell.grade == 1 and cell.index == 2 and arity == 4
    assert executor.execute(parse('FROM $r MATCH e IN CELLS(1) RETURN e.index LIMIT 1 OFFSET 1')).values == (((1,),),)
    assert executor.execute(parse('FROM $r MATCH e IN CELLS(1) WHERE e.index < 0 RETURN e')).values == ((),)
    assert executor.execute(parse('FROM $r MATCH e IN CELLS(1) RETURN e LIMIT 0')).values == ((),)
    assert not executor.execute(parse('EXPLAIN ' + text)).execution


def test_dependent_native_match_and_exact_predicate():
    r = branching()
    executor = Executor(sources={"r": r})
    result = executor.execute(parse('FROM $r MATCH a IN CELLS(1), b IN CELLS(1) '
        'WHERE a.index < b.index AND 1152921504606846977 > 1152921504606846976 RETURN a.index, b.index'))
    assert result.values == (((0, 1), (0, 2), (1, 2)),)
    assert executor.execute(parse('FROM $r RETURN 1/3 < 1/2')).values == (True,)


def test_catalog_match(tmp_path):
    save(str(tmp_path / 'b.rcbd'), branching())
    save(str(tmp_path / 'a.rcbd'), branching())
    result = Executor(sources={"files": FileCatalog([tmp_path])}).execute(parse(
        'FROM $files MATCH f IN FILES() WHERE f.kind = "rcbd" RETURN f.name ORDER BY f.name DESC'))
    assert result.values == ((("root0/b.rcbd",), ("root0/a.rcbd",)),)


def test_store_match_and_whole_plan_validation_precede_readers(monkeypatch):
    rcdb = pytest.importorskip('rcdb')
    store = rcdb.MemoryStore()
    store.put('r', branching(), analytics=False)
    executor = Executor(sources={'db': store})
    assert executor.execute(parse('FROM $db MATCH r IN RCDB_LIST() WHERE r.id = "r" RETURN r.id')).values == ((("r",),),)
    def fail(*args, **kwargs):
        pytest.fail('reader ran during EXPLAIN or before body validation')
    monkeypatch.setattr(store, 'list', fail)
    executor.execute(parse('EXPLAIN FROM $db MATCH r IN RCDB_LIST() RETURN r.id'))
    with pytest.raises(TypeError):
        executor.execute(parse('FROM $db MATCH r IN RCDB_LIST() RETURN ARITY(r)'))


@pytest.mark.parametrize("text", [
    'FROM $r MATCH e IN CELLS(1) WHERE e RETURN e',
    'FROM $r MATCH e IN DESCRIBE() RETURN e',
    'FROM $r MATCH e IN CELLS(1) RETURN e.source',
    'FROM $r MATCH e IN CELLS(1) RETURN e LIMIT -1',
    'FROM $r MATCH e IN CELLS(1) RETURN e ORDER BY e LIMIT 2',
])
def test_invalid_match_fails(text):
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        Executor(sources={"r": branching()}).execute(parse(text))


def test_inline_edits_commit_exact_state_without_touching_source():
    rcdb = pytest.importorskip('rcdb')
    store = rcdb.MemoryStore().configure_security(require_commits=True)
    original = branching()
    before = object_digest(original)
    executor = Executor(sources={"db": store}, params={"r": original})
    text = 'FROM $db MUTATE "r" SET state=$r, expected_version=0 REMOVE [0] ADD [[0,1,2,3,4]] COMMIT'
    built = mutation(param('db'), 'r', param('r'), expected_version=0,
                     edits=(('REMOVE', [0]), ('ADD', [[0, 1, 2, 3, 4]])))
    assert parse(text) == built
    executor.execute(parse('EXPLAIN ' + text))
    assert store.read_record('r') is None
    executor.execute(parse(text))
    assert object_digest(original) == before
    loaded = store.read_record('r').value
    assert loaded.nE == 3 and loaded.w_E[0] == Q(2, 7)
    assert store.verify_commits('r')
    # Ordered clauses use the post deletion basis, not the caller's old indices.
    assert np.diff(loaded._boundary_ptr).tolist() == [3, 4, 5]


def test_weighted_edit_mapping_and_bad_edit_leave_source_unchanged():
    r = branching()
    before = object_digest(r)
    added = edit_relations(r, 'ADD', {"columns": [[0, 5, 6]], "weights": [Q(1, 2**70)]})
    assert added.w_E[-1] == Q(1, 2**70)
    with pytest.raises(ValueError):
        edit_relations(r, 'REMOVE', [99])
    assert object_digest(r) == before


def test_compaction_maps_faces_metadata_and_upper_cofaces():
    from rexgraph.native_sparse import csr_carrier, sparse_arrays
    from rexgraph.sectioning import add_sectioning, sectionings_of
    r = RexGraph.from_simplicial([0, 1, 0, 3, 4, 3], [1, 2, 2, 4, 5, 5], [[0, 1, 2], [3, 4, 5]])
    # Zero columns are valid upper cycles; a nonzero coface dependency is
    # tested separately at the transport seam, not called a valid chain here.
    r._graded_duals = [csr_carrier(np.array([0, 1, 2]), np.array([0, 1]), np.array([1, 1]), (2, 2))]
    r.attach_metadata(2, 1, 'name', 'surviving face')
    r.attach_metadata(3, 1, 'name', 'surviving upper cell')
    add_sectioning(r, 'parts', {'a': [0, 1, 2], 'b': [3, 4, 5]})
    r.remove_edges([1, 0, 0, 0, 0, 0])
    remap = r.compact()
    assert remap.face_map.tolist() == [-1, 0]
    assert r.get_metadata(2, 0, 'name') == 'surviving face'
    assert r.get_metadata(3, 0, 'name') == 'surviving upper cell'
    assert sparse_arrays(r._graded_duals[0])[3] == (1, 1)
    assert sectionings_of(r)['parts'].cells(1).tolist() == [2, 3, 4]


@pytest.mark.parametrize('at', [-1, float('nan'), float('inf'), True])
def test_invalid_clock_is_atomic(at):
    r = branching()
    history = TemporalRex([], general=True)
    history.append_snapshot(r, at=5)
    before = object_digest(history)
    with pytest.raises(ValueError):
        history.append_snapshot(r, at=at)
    assert object_digest(history) == before
    for index in [-2, 1]:
        with pytest.raises(IndexError):
            history.reconstruct_at(index)
