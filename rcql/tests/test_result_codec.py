"""Portable exact results do not execute query syntax or densify lazy actions."""
from fractions import Fraction as Q
import hashlib

import numpy as np
import pytest
from rexgraph.cochain import Chain, Cochain
from rexgraph.exact_array import ExactArray
from rexgraph.graph import RexGraph
from rexgraph.relations import Relations, VertexTable
from rexgraph.value import Absent, Approx
from rexgraph.value_codec import pack_value, unpack_value

from rcql import Executor, Result, call, query, source
from rcql.ast import Literal
from rcql.optimizer import Rewrite
from rcql.result_codec import pack_result, unpack_result
from rcql.types import Exactness, PredicateResult


def test_exact_result_retains_values_kinds_aliases_plans_and_provenance():
    values = (2**16000, Q(2, 3), Absent, Approx(.1, "measured"),
              ExactArray.from_values([1, Q(1), Absent]), np.array([Q(1, 7)], dtype=object))
    original = Result(values, plan=("declared",), aliases=("big", "q", None, None, None, None),
                      exactness=(Exactness.INTEGER, Exactness.RATIONAL, None, Exactness.APPROXIMATE, None, None),
                      native_plan={"method": "sparse"}, provenance=({"state": "source", "q": Q(1, 3)},),
                      execution=({"cells": 5},))
    decoded = Result.from_bytes(original.to_bytes())
    assert decoded.values[:4] == values[:4]
    assert decoded.values[4] == values[4]
    assert np.array_equal(decoded.values[5], values[5])
    assert decoded.named_values == {"big": 2**16000, "q": Q(2, 3)}
    for name in ("plan", "aliases", "exactness", "native_plan", "provenance", "execution"):
        assert getattr(decoded, name) == getattr(original, name)
    assert decoded.to_bytes() == original.to_bytes()


def test_graph_and_chains_share_one_sealed_source_basis():
    rex = RexGraph.from_relations(Relations.from_supports([[0, 1, 2, 3]],
        vertices=VertexTable(("a", "b", "c", "d", "isolate")),
        shares=[0, Q(1, 4), Q(1, 2), Q(1, 4)], weights=[Q(2, 3)], relation_ids=["r"]))
    c = Chain(1, np.array([Q(1, 7)], dtype=object), ("r",), rex)
    d = Cochain(0, np.arange(5), source=rex)
    result = unpack_result(pack_result(Result((rex, c, d, {"nested": rex}))))
    back, chain, cochain, nested = result.values
    assert chain.source is back and cochain.source is back and nested["nested"] is back
    assert back.nV == 5 and back.relations.vertices.ids[-1] == "isolate"
    assert back.relations.share.values().tolist() == [0, Q(1, 4), Q(1, 2), Q(1, 4)]
    assert back.edge_metric_exact == [Q(2, 3)]
    assert chain.values.tolist() == [Q(1, 7)]
    assert isinstance(chain, Chain) and isinstance(cochain, Cochain)


def test_rewrites_are_syntax_records_and_decoding_never_executes(monkeypatch):
    before, after = call("SUM", [1, 2]), Literal(3)
    rewrite = Rewrite(before, after, "scalar fold", (PredicateResult("domain", "verified", "integer"),))
    payload = pack_result(Result((3,), (rewrite,)))
    monkeypatch.setattr(Executor, "execute", lambda *a, **k: pytest.fail("query executed while decoding"))
    assert unpack_result(payload).rewrites == (rewrite,)


def test_real_executor_result_is_portable():
    rex = RexGraph.from_graph([0], [1])
    result = Executor(sources={"r": rex}).execute(query(source("r"), call("BETTI", 0)))
    back = unpack_result(pack_result(result))
    assert np.array_equal(back.values[0], result.values[0])
    assert back.provenance == result.provenance and back.execution == result.execution


def test_canonical_map_order_and_explicit_limits():
    assert pack_result(Result(({"a": 1, "b": Q(2, 3)},))) == pack_result(Result(({"b": Q(2, 3), "a": 1},)))
    raw = pack_result(Result((1,)))
    with pytest.raises(ValueError, match="limit"):
        pack_result(Result((1,)), max_bytes=10)
    with pytest.raises(ValueError, match="oversized"):
        unpack_result(raw, max_bytes=10)
    with pytest.raises(ValueError, match="digest"):
        unpack_result(raw[:-1] + bytes([raw[-1] ^ 1]))


@pytest.mark.parametrize("mutation", ["unknown", "unaligned", "graph-reference", "grade-basis"])
def test_unknown_or_misaligned_records_are_refused_even_with_a_new_digest(mutation):
    rex = RexGraph.from_graph([0], [1])
    values = (rex, Chain(1, np.array([1]), source=rex)) if mutation == "grade-basis" else (1,)
    payload = pack_result(Result(values))
    record = unpack_value(payload[37:])
    if mutation == "unknown":
        record["values"] = ("tuple", (("execute", "SUM"),))
    elif mutation == "unaligned":
        record["aliases"] = ("a", "b")
    elif mutation == "graph-reference":
        record["values"] = ("tuple", (("graph", 0),))
    else:
        nodes = list(record["values"][1])
        nodes[1] = ("Chain", 0, *nodes[1][2:])
        record["values"] = ("tuple", tuple(nodes))
    raw = pack_value(record)
    with pytest.raises(ValueError):
        unpack_result(payload[:5] + hashlib.sha256(raw).digest() + raw)


def test_unsupported_live_handles_and_cycles_are_refused():
    class Handle:
        def __repr__(self):
            pytest.fail("opaque repr exposed")
    with pytest.raises(TypeError, match="portable codec"):
        pack_result(Result((Handle(),)))
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        pack_result(Result((cyclic,)))


def test_partition_lineage_maps_and_carried_metadata_are_portable():
    from rexgraph.partition_state import build_rex_partition
    from rexgraph.object_identity import object_digest
    rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2], relation_ids=[10, 11, 12])
    rex.attach_metadata(1, 0, "exact", Q(1, 7))
    partition = build_rex_partition(rex, [1, 1, 0], carried_state="all")
    restored = unpack_result(pack_result(Result((partition,)))).values[0]
    assert restored.manifest == partition.manifest
    assert restored.cell_maps == partition.cell_maps
    assert object_digest(restored.rex) == object_digest(partition.rex)
    assert restored.rex.get_metadata(1, 0, "exact") == Q(1, 7)


def test_retained_field_sources_share_the_portable_graph_table():
    from rexgraph.tensor_field import FieldSource, TensorField, TensorChannels
    from rexgraph.type_accession import CoordinateSpace
    rex = RexGraph.from_graph([0], [1])
    ref = FieldSource(rex, "record", 7)
    field = TensorField(CoordinateSpace("edge", ("e",)), [Q(1, 7)], source=ref, grade=1)
    channels = TensorChannels(("measured",), (field,), "0"*64, (ref,))
    result = unpack_result(pack_result(Result((rex, ref, field, channels))))
    graph, source_ref, retained, channel_set = result.values
    assert source_ref.source is graph and retained.source.source is graph
    assert channel_set.fields[0].source.source is graph
    assert channel_set.endpoint_sources[0].source is graph
    assert retained.coefficient_digest == field.coefficient_digest
    assert channel_set.names == channels.names and channel_set.declaration_digest == channels.declaration_digest
    assert retained.values.tolist() == [Q(1, 7)]
    assert result.to_bytes() == pack_result(Result((rex, ref, field, channels)))


def test_detached_field_sources_remain_detached():
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.type_accession import CoordinateSpace
    rex = RexGraph.from_graph([0], [1])
    ref = FieldSource(None, "record", 7, FieldSource(rex).state_digest)
    field = TensorField(CoordinateSpace("edge", ("e",)), [Q(1, 7)], source=ref, grade=1)
    restored = unpack_result(pack_result(Result((field,)))).values[0]
    assert restored.source.source is None and restored.source.as_record() == ref.as_record()


def test_temporal_and_relations_results_preserve_native_identity():
    from rexgraph.graph import TemporalRex
    from rexgraph.object_identity import object_digest
    rex = RexGraph.from_relations(Relations.from_supports([[0, 1, 2]], heads=[1],
        shares=[Q(1, 3), 0, Q(2, 3)], weights=[Absent], relation_ids=["r"],
        vertices=VertexTable(("a", "b", "c", "isolate")), attributes={1: {0: {"exact": Q(1, 7)}}}))
    temporal = TemporalRex([])
    temporal.append_snapshot(rex)
    temporal.append_snapshot(rex.copy())
    back_temporal, back_relations = unpack_result(pack_result(Result((temporal, rex.relations)))).values
    assert object_digest(back_temporal) == object_digest(temporal)
    assert back_relations.weight.values().tolist() == [Absent]
    assert back_relations.vertices.ids[-1] == "isolate"
    assert back_relations.attributes[1][0]["exact"] == Q(1, 7)


@pytest.mark.parametrize("mutation", ["binding-digest", "unknown-component", "partition-maps"])
def test_native_result_carriers_refuse_forged_declarations(mutation):
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.partition_state import build_rex_partition
    rex = RexGraph.from_graph([0], [1])
    value = (build_rex_partition(rex, [1]) if mutation == "partition-maps" else
             TensorField(CoordinateSpace("edge", ("e",)), [Q(1, 7)], source=FieldSource(rex)))
    payload = pack_result(Result((value,)))
    record = unpack_value(payload[37:])
    node = list(record["values"][1][0])
    if mutation == "binding-digest":
        binding = dict(node[3][0][0]); binding["state_digest"] = "0"*64
        node[3] = ((binding, node[3][0][1]),)
    elif mutation == "unknown-component":
        node[1] = "arbitrary.class.import"
    else:
        node[3] = ((0,), (0,))
    record["values"] = ("tuple", (tuple(node),))
    raw = pack_value(record)
    with pytest.raises(ValueError):
        unpack_result(payload[:5]+hashlib.sha256(raw).digest()+raw)
