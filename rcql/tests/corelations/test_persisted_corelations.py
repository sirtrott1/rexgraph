from fractions import Fraction as Q

import pytest

from rcql import Executor, NameRelation, RelationTopology, parse
from rcql.ast import Call, ListExpr, Literal, Parameter
from rcql.program_codec import dumps, loads
from rexgraph.chain_map import _chain_residual


def add(n):
    return NameRelation('add', ('x',), Call('SUM', (ListExpr((Parameter('x'), Literal(n))),)))


def double():
    return NameRelation('double', ('x',), Call('SUM', (ListExpr((Parameter('x'), Parameter('x'))),)))


def source():
    from rexgraph.graph import RexGraph
    return RexGraph.from_graph([0, 1], [1, 2])


def test_schema_two_persists_exact_corelations_without_rebuild(monkeypatch):
    topology = add(1).then(double(), 'x', name='twice_after_add').topology()
    record = topology.to_record()
    meta = record._agent_meta
    assert meta['rcql_relation_schema'] == 2
    assert meta['rcql_operation_top_grade'] == 2
    assert isinstance(meta['rcql_operation_complex'], str)
    original = topology.corelations()
    assert original and original[0].identity == 'expansion/name/twice_after_add'
    assert _chain_residual(topology.boundary_tower()) == 0

    monkeypatch.setattr(RelationTopology, '_build_boundary_tower', lambda self: pytest.fail('reload rebuilt co relations'))
    restored = RelationTopology.from_record(record)
    assert restored.corelations() == original
    assert restored.boundary_tower().coefficient_digest == meta['rcql_operation_complex_digest']
    assert _chain_residual(restored.boundary_tower()) == 0


def test_primary_carrier_remains_canonical_while_exact_tower_retains_composites():
    topology = add(1).then(double(), 'x', name='twice_after_add').topology()
    record = topology.to_record()
    tower = topology.boundary_tower()
    assert record.nE == len(topology.declaration['nodes'])
    assert tower.sizes[1] == record.nE + len(topology.declaration['composites'])
    assert tuple(tower.spaces[1].keys[:record.nE]) == tuple(n['id'] for n in topology.declaration['nodes'])
    assert tuple(tower.spaces[1].keys[record.nE:]) == ('name/twice_after_add',)


def test_declared_corelation_is_persisted_as_an_exact_grade_two_cell():
    topology = add(1).then(double(), 'x', name='twice_after_add').topology()
    first = topology.corelations()[0]
    copied = topology.declare_filling('same_boundary', first.boundary)
    record = copied.to_record()
    restored = RelationTopology.from_record(record)
    cells = restored.corelations()
    assert tuple(c.identity for c in cells) == ('expansion/name/twice_after_add', 'same_boundary')
    assert cells[0].boundary == cells[1].boundary
    assert _chain_residual(restored.boundary_tower()) == 0


def test_persisted_chain_violation_is_rejected_even_if_payload_digest_is_updated():
    topology = add(1).then(double(), 'x', name='twice_after_add').topology()
    record = topology.to_record()
    meta = dict(record._agent_meta)
    payload = loads(meta['rcql_operation_complex'])
    boundaries = [list(level) for level in payload['boundaries']]
    row, column, value = boundaries[1][0]
    boundaries[1][0] = (row, column, value + Q(1))
    payload['boundaries'] = tuple(tuple(level) for level in boundaries)
    # The declared digest is deliberately removed from consideration by replacing it
    # with a syntactically valid value. The exact chain check must still reject this.
    payload['digest'] = 'tampered'
    meta['rcql_operation_complex'] = dumps(payload).decode('utf8')
    meta['rcql_operation_complex_digest'] = 'tampered'
    record._agent_meta = meta
    with pytest.raises(ValueError):
        RelationTopology.from_record(record)


def test_legacy_schema_one_record_still_restores_and_upgrades():
    topology = add(2).then(double(), 'x', name='legacy').topology()
    legacy = topology._record(1)
    restored = RelationTopology.from_record(legacy)
    assert restored.coefficient_digest == topology.coefficient_digest
    upgraded = restored.to_record()
    assert upgraded._agent_meta['rcql_relation_schema'] == 2
    assert RelationTopology.from_record(upgraded).corelations() == topology.corelations()


def test_rcql_topology_result_carries_persisted_grade_two_operation_relations():
    r = source()
    operation = add(1).then(double(), 'x', name='twice_after_add')
    result = Executor(sources={'r': r}, params={'op': operation}).execute(parse(
        'FROM $r RETURN NAME_TOPOLOGY($op)'
    ))
    restored = RelationTopology.from_record(result.values[0])
    cells = restored.corelations()
    assert len(cells) == 1
    assert cells[0].grade == 2
    assert cells[0].identity == 'expansion/name/twice_after_add'
    assert cells[0].boundary


@pytest.mark.parametrize('kind', ['rcbd', 'safetensors'])
def test_exact_operation_tower_survives_portable_storage(tmp_path, kind):
    topology = add(3).then(double(), 'x', name='portable').topology()
    record = topology.to_record()
    path = tmp_path / ('topology.' + kind)
    if kind == 'rcbd':
        from rexgraph.io.bundle import save_rcbd, load_rcbd
        save_rcbd(path, record)
        loaded = load_rcbd(path)
    else:
        from rexgraph.io.safetensors_bridge import rex_to_safetensors, load_safetensors
        rex_to_safetensors(record, path)
        loaded = load_safetensors(path)['object']
    restored = RelationTopology.from_record(loaded)
    assert restored.boundary_tower().coefficient_digest == topology.boundary_tower().coefficient_digest
    assert restored.corelations() == topology.corelations()


def test_exact_operation_tower_survives_rcdb_roundtrip(tmp_path):
    from contextlib import closing
    from rcdb import open_store
    topology = add(4).then(double(), 'x', name='stored').topology()
    with closing(open_store('rex://' + str(tmp_path / 'db'))) as store:
        store.commit_mutation('operation_topology', topology.to_record(), expected_version=0, tx_time=1)
        loaded = store.read_record('operation_topology').value
    restored = RelationTopology.from_record(loaded)
    assert restored.boundary_tower().coefficient_digest == topology.boundary_tower().coefficient_digest
    assert restored.corelations() == topology.corelations()
