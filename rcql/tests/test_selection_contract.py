"""RCQL delegates multi grade selection and lineage to Core's public contract."""
from dataclasses import replace
import hashlib

import pytest

from rcql import Executor, Result, parse
from rcql.query_cache import QueryCache
from rcdb import FileStore
from rexgraph import RexGraph, Selection, restrict
from rexgraph.cells import CellSet, GradedCellPattern
from rexgraph.value_codec import pack_value, unpack_value


def graph():
    rex = RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]],
                              relation_ids=[91, 17, 53])
    rex.attach_metadata(1, 1, "name", "retained")
    return rex


@pytest.mark.parametrize("carrier", ["selection", "pattern"])
@pytest.mark.parametrize("operator", ["PARTITION", "RESTRICT", "QUOTIENT"])
def test_multigrade_inputs_share_the_declared_core_basis(carrier, operator):
    source = graph()
    chosen = (Selection(source, {0: [3], 2: [0]}) if carrier == "selection" else
              GradedCellPattern(source, (CellSet(source, 0, (3,)), CellSet(source, 2, (0,)))))
    executor = Executor(sources={"r": source}, params={"chosen": chosen})
    query = parse(f'FROM $r RETURN {operator}($chosen)')
    executor.execute(replace(query, explain=True))
    result = executor.execute(query).values[0]
    if operator == "QUOTIENT":
        assert result.sizes == (0, 0, 0)
    else:
        output = result.rex if operator == "PARTITION" else result
        assert (output.nV, output.nE, output.nF) == (4, 3, 1)
        assert output.relation_ids.tolist() == [91, 17, 53]


def test_lineage_members_are_explicit_read_only_records():
    source = graph()
    executor = Executor(sources={"r": source})
    result = executor.execute(parse('FROM $r LET p=PARTITION(CELL(1, 1)) '
                                    'RETURN p.lineage, p.lineage.digest, p.lineage.cell_maps'))
    lineage, digest, maps = result.values
    assert lineage.digest == digest and lineage.cell_maps == maps == ((1, 2), (1,), ())
    with pytest.raises(TypeError, match="declared member"):
        executor.execute(parse('FROM $r RETURN PARTITION(CELL(1, 1)).lineage.to_bytes'))


def test_selection_partition_and_lineage_roundtrip_share_one_source_graph():
    source = graph()
    chosen = Selection(source, {0: [3], 1: [1]})
    part = restrict(source, chosen, carried_state="all")
    output = Result.from_bytes(Result((source, chosen, part, part.lineage)).to_bytes())
    root, selection, restored, lineage = output.values
    assert selection.source is root
    assert restored.lineage == lineage == part.lineage
    assert restored.lineage.verify(root, restored.rex)
    assert restored.rex.get_metadata(1, 0, "name") == "retained"


def test_legacy_partition_result_keeps_its_existing_members_without_inventing_a_basis():
    from rexgraph.object_identity import object_digest
    source = graph()
    part = replace(restrict(source, Selection(source, {1: [0]})), source_sizes=(), requested=())
    part.rex._face_grade = False
    part = replace(part, state=replace(part.state, result_state=object_digest(part.rex)))
    restored = Result.from_bytes(Result((part,)).to_bytes()).values[0]
    assert restored.cell_maps == part.cell_maps and restored.manifest == part.manifest
    with pytest.raises(ValueError, match="legacy partition"):
        _ = restored.lineage


@pytest.mark.parametrize("field", ["cell_maps", "selection_digest", "result_state"])
def test_reframed_partition_cannot_disagree_with_its_lineage(field):
    source = graph()
    part = restrict(source, Selection(source, {1: [1]}))
    payload = Result((part,)).to_bytes()
    record = unpack_value(payload[37:])
    node = record["values"][1][0]
    if field == "cell_maps":
        record["values"] = ("tuple", ((node[0], node[1], node[2], ((0, 1), (0,), ()), node[4]),))
    else:
        node[2][field] = "0"*64
    encoded = pack_value(record)
    with pytest.raises(ValueError, match="lineage"):
        Result.from_bytes(payload[:5]+hashlib.sha256(encoded).digest()+encoded)


def test_selection_parameter_and_result_survive_cache_reopening_and_rebinding(tmp_path):
    store = FileStore(str(tmp_path / "cache"), read_only=False)
    root = graph()
    query = parse('FROM $r RETURN $chosen, PARTITION($chosen,carried_state="all")')
    executor = Executor(sources={"r": root}, params={"chosen": Selection(root, {1: [1]})})
    try:
        first = QueryCache(store).execute(executor, query)
        store.close()
        store = FileStore(str(tmp_path / "cache"), read_only=False)
        copy = root.copy()
        current = Executor(sources={"r": copy}, params={"chosen": Selection(copy, {1: [1]})})
        second = QueryCache(store).execute(current, query)
        assert second.native_plan["cache"]["hit"]
        assert second.values[0].source is copy
        assert first.values[1].lineage == second.values[1].lineage
        assert second.values[1].lineage.verify(copy, second.values[1].rex)
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["structural", "all"])
def test_partition_glue_uses_the_bound_source_and_survives_portable_results(mode):
    root = graph()
    executor = Executor(sources={"r": root})
    query = parse(f'FROM $r LET a=PARTITION(CELL(1,0),carried_state="{mode}") '
                  f'LET b=PARTITION(CELL(1,1),carried_state="{mode}") '
                  f'RETURN GLUE_PARTITIONS([a,b],carried_state="{mode}")')
    executor.execute(replace(query, explain=True))
    result = executor.execute(query)
    united = Result.from_bytes(result.to_bytes()).values[0]
    assert united.cell_maps == ((0, 1, 2), (0, 1), ())
    assert len(united.lineage_parents) == 2
    assert united.lineage.verify(root, united.rex)
    assert united.rex.get_metadata(1, 1, "name") == ("retained" if mode == "all" else None)


@pytest.mark.parametrize("parts", [[], [1]])
def test_partition_glue_refuses_invalid_families_during_planning(parts):
    executor = Executor(sources={"r": graph()}, params={"parts": parts})
    with pytest.raises((TypeError, ValueError), match="partition|parts"):
        executor.execute(parse('FROM $r RETURN GLUE_PARTITIONS($parts)'))


def test_partition_glue_refuses_modified_parts_and_omitted_application_state():
    root = graph()
    part = restrict(root, Selection(root, {1: [1]}))
    executor = Executor(sources={"r": root}, params={"parts": [part]})
    with pytest.raises(ValueError, match="omitted"):
        executor.execute(parse('FROM $r RETURN GLUE_PARTITIONS($parts,carried_state="all")'))
    part.rex.attach_metadata(1, 0, "edited", 1)
    with pytest.raises(ValueError, match="changed"):
        executor.execute(parse('FROM $r RETURN GLUE_PARTITIONS($parts)'))


def test_artifact_operators_accept_complete_lineage_without_changing_legacy_partition_hashes():
    root = graph()
    executor = Executor(sources={"r": root})
    result = executor.execute(parse('FROM $r LET p=PARTITION(CELL(1,1)) '
        'RETURN p.lineage, HASH(p.lineage,"lineage"), LINEAGE(p.lineage), MANIFEST(p.lineage), '
        'HASH(p,"lineage"), LINEAGE(p)'))
    lineage, digest, description, manifest, legacy_digest, legacy = result.values
    assert lineage.digest == digest == description["digest"]
    assert description == manifest and manifest["manifest"] == lineage.as_record()
    assert manifest["object_type"] == "Lineage" and not manifest["signature_verified"]
    assert legacy_digest == legacy["digest"] and legacy["object_type"] == "PartitionState"
    assert lineage.digest != legacy_digest
