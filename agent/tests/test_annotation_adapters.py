"""Annotation adapters and their native temporal query route."""
from fractions import Fraction as Q
import gzip
import json
from contextlib import closing
import shutil

import numpy as np
import pytest

from agent.adapters.annotations import read_brat, read_conll
from agent.adapters.annotation_store import ingest_annotations, verify_annotation_sources
from agent.adapters.formats import available_readers, reader_for, read
from agent.auto import auto_rex, build_rex_from_edges
from rexgraph.attachment_field import AttachmentField, common_attachment_observations
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.span import SpanBlock
from rexgraph.tensor_moment import CoordinatePairing, MomentSpan
from rexgraph.io.rex_state import to_state, from_state
from rexgraph.chain_map import CoordinateComplex
from rcdb import open_store
from rcdb.core import VersionConflictError
from rcql import Executor, parse


TEXT = "Drug α began in 2017, paused, and resumed in 2024.\r\nNo nausea."


def fixture(path, *, replace=None):
    path.mkdir(parents=True, exist_ok=True)
    def line(id, label, fragments):
        spans = [(TEXT.index(fragment), TEXT.index(fragment) + len(fragment)) for fragment in fragments]
        return f"{id}\t{label} " + ";".join(f"{a} {b}" for a, b in spans) + "\t" + " ".join(fragments)
    lines = [line("T1", "Drug", ["Drug α"]), line("T2", "Treatment", ["began"]),
             line("T3", "Time", ["2017", "2024"]), line("T4", "Finding", ["nausea"]),
             line("T5", "Drug", ["Drug α"]), "E1\tTreatment:T2 Agent:T1 Time:T3",
             "E2\tTreatment:T2 Agent:T1", "R1\tBefore Arg1:E1 Arg2:E2",
             "A1\tNegation E2", "N1\tReference T1 CHEBI:123\tDrug alpha",
             "#1\tAnnotatorNotes T3\tSeparated references", "*\tEquiv T1 T5"]
    ann = "\r\n".join(lines) + "\r\n"
    if replace:
        ann = replace(ann)
    (path / "case.txt").write_bytes(TEXT.encode("utf-8"))
    (path / "case.ann").write_bytes(ann.encode("utf-8"))
    return path / "case.ann"


def temporal_options(path, *, second=2024):
    base = read_brat(path, document_id="test_document")
    a = next(a for _, a in base.attachments if a.annotation_id == "E1/Time/0")
    block = SpanBlock("event_support", "event_time", "abstract_year",
                      (("first", 2017, 2019), ("second", second, 2026)))
    g = CoordinateMap(a.text.coordinates, block.coordinates, ((0, 0, 1), (1, 1, 1)))
    return {"time_supports": {a.annotation_id: block}, "groundings": {a.annotation_id: g}}


def test_registration():
    assert {"brat", "conll_ner"}.issubset(available_readers())
    assert reader_for("x.ann.gz") == "brat"
    assert reader_for("x.conll") == "conll_ner"
    assert reader_for("x.bio.gz") == "conll_ner"


def test_source_namespace_required(tmp_path):
    p = fixture(tmp_path)
    with pytest.raises(TypeError): read(p)
    with pytest.raises(ValueError): read_brat(p, document_id="")


def test_brat_components_and_references(tmp_path):
    b = read_brat(fixture(tmp_path), document_id="doc")
    by_id = {r.identifier: r for r in b.records}
    assert len(b.records) == 12
    assert len(by_id["T3"].components) == 2
    assert by_id["E1"].references == (("Trigger", "T2"), ("Agent", "T1"), ("Time", "T3"))
    assert by_id["A1"].value == "true"
    assert json.loads(by_id["N1"].value)[0] == "CHEBI:123"
    assert all(r.raw_line.endswith("\r\n") for r in b.records)
    assert b.manifest["unit"] == "unicode_codepoint"
    assert not any(a.time for _, a in b.attachments)


def test_native_build_and_exact_boundaries(tmp_path):
    b = read_brat(fixture(tmp_path), document_id="doc")
    rex = b.to_rex()
    assert rex.nE == len(b.records)
    assert rex.nF == 0
    # The native exact boundary reader checks the chosen primary construction.
    tower = CoordinateComplex.from_rex(rex)
    assert tower.spaces[1].keys
    assert len(set(map(int, rex.relation_ids))) == rex.nE
    restored = from_state(to_state(rex))
    field = AttachmentField.from_source(restored)
    assert len(field.attachments) == len(b.attachments)
    assert {a.annotation_id for a in field.attachments} == {a.annotation_id for _, a in b.attachments}
    assert restored._agent_meta["source_manifest"] == b.manifest


def test_auto_dispatch_uses_same_construction(tmp_path):
    p = fixture(tmp_path)
    ec = read(p, document_id="doc")
    expected = build_rex_from_edges(ec)
    actual = auto_rex(str(p), document_id="doc")
    assert actual.nE == expected.nE
    assert np.array_equal(actual.relation_ids, expected.relation_ids)
    assert actual._agent_meta["source_manifest"] == expected._agent_meta["source_manifest"]


def test_parallel_source_ids_are_not_merged(tmp_path):
    b = read_brat(fixture(tmp_path), document_id="doc")
    ec = b.construction()
    ids = {r.identifier: i for i, r in enumerate(b.records)}
    assert ec.branching[ids["T1"]] == ec.branching[ids["T5"]]
    assert ec.relation_ids[ids["T1"]] != ec.relation_ids[ids["T5"]]
    ev = [a for _, a in b.attachments if a.role == "Trigger"]
    assert ev[0].text == ev[1].text
    assert ev[0].annotation_id != ev[1].annotation_id
    assert ev[0].owner_id != ev[1].owner_id


def test_record_reorder_preserves_explicit_primary_ids(tmp_path):
    p = fixture(tmp_path)
    a = read_brat(p, document_id="doc")
    lines = p.read_bytes().splitlines(keepends=True)
    p.write_bytes(b"".join(reversed(lines)))
    b = read_brat(p, document_id="doc")
    ma = {r.identifier: int(n) for r, n in zip(a.records, a.construction().relation_ids, strict=False)}
    mb = {r.identifier: int(n) for r, n in zip(b.records, b.construction().relation_ids, strict=False)}
    assert ma == mb
    assert a.digest != b.digest


def test_declared_time_and_grounding_roundtrip(tmp_path):
    p = fixture(tmp_path)
    options = temporal_options(p)
    b = read_brat(p, document_id="test_document", **options)
    a = next(a for _, a in b.attachments if a.annotation_id == "E1/Time/0")
    assert a.time.components == (("first", Q(2017), Q(2019)), ("second", Q(2024), Q(2026)))
    actual = AttachmentField.from_source(from_state(to_state(b.to_rex())), annotation_ids=[a.annotation_id])
    assert actual.attachments[0] == a
    o = actual.observe(support="grounded_time", local=False)
    y = o.action.apply(actual.amplitudes().values)
    assert y.tolist() == [1, 0, 1]
    assert o.metric.moment(y, y) == 4


def test_qualifiers_do_not_change_asserted_support(tmp_path):
    b = read_brat(fixture(tmp_path), document_id="doc")
    a = next(a for _, a in b.attachments if a.annotation_id == "E2/Trigger/0")
    assert dict(a.qualifiers)["attribute:A1"] == '["Negation","true"]'
    assert a.text.components


@pytest.mark.parametrize("bad", [
    "T1\tDrug 0 999\tDrug α\n", "T1\tDrug 1 1\t\n", "T1\tDrug 0 6\twrong\n",
    "T1 Drug 0 6 Drug α\n",
    "T1\tDrug 0 6\tDrug α\nT1\tDrug 0 6\tDrug α\n", "A1\tNegation E99\n",
    "E1\tDrug:T999\n", "T1\tDrug 0 6\tDrug α\nE1\tTreatment:T1\n",
    "T1\tDrug 0 6\tDrug α\nR1\tRel Arg1:T1\n", "Z1\tUnknown T1\n",
    "N1\tReference T1 invalid\tname\n", "*\tEquiv T1 T1\n"])
def test_brat_rejects_malformed_records(tmp_path, bad):
    (tmp_path / "x.txt").write_bytes(TEXT.encode())
    p = tmp_path / "x.ann"; p.write_bytes(bad.encode())
    with pytest.raises(ValueError): read_brat(p, document_id="doc")


def test_invalid_unicode_not_replaced(tmp_path):
    p = fixture(tmp_path)
    (tmp_path / "case.txt").write_bytes(b"\xfftext")
    with pytest.raises(UnicodeDecodeError): read_brat(p, document_id="doc")


def test_gzip_and_explicit_text_path(tmp_path):
    p = fixture(tmp_path)
    text = tmp_path / "renamed.txt.gz"
    text.write_bytes(gzip.compress((tmp_path / "case.txt").read_bytes()))
    ann = tmp_path / "case.ann.gz"; ann.write_bytes(gzip.compress(p.read_bytes()))
    b = read_brat(ann, document_id="doc", text_path=text)
    assert len(b.records) == 12
    assert b.manifest["files"][0]["name"] == "renamed.txt.gz"


def test_text_change_requires_explicit_alignment(tmp_path):
    p = fixture(tmp_path)
    a = AttachmentField.from_source(read_brat(p, document_id="doc").to_rex(), annotation_ids=["T1"])
    (tmp_path / "case.txt").write_bytes((TEXT + " later").encode())
    b = AttachmentField.from_source(read_brat(p, document_id="doc").to_rex(), annotation_ids=["T1"])
    with pytest.raises(ValueError): common_attachment_observations(a, b, support="text")


def test_undeclared_grounding_identity_rejected(tmp_path):
    p = fixture(tmp_path)
    span = SpanBlock("time", "event_time", "day", (("x", 0, 1),))
    with pytest.raises(ValueError): read_brat(p, document_id="doc", time_supports={"unknown": span})


@pytest.mark.parametrize("scheme,data,count", [
    ("IOB2", "echo B-X\necho B-X\n", 2),
    ("IOB2", "a B-X\nb I-X\n\nc B-X\n", 2),
    ("IOB1", "a I-X\nb I-X\nc B-X\n", 2),
    ("BIOES", "a B-X\nb E-X\nc S-X\n", 2),
    ("BIOES", "a O\n", 0)])
def test_conll_declared_scheme(tmp_path, scheme, data, count):
    p = tmp_path / "tokens.conll"; p.write_bytes(data.encode())
    b = read_conll(p, document_id="doc", scheme=scheme)
    assert len(b.records) == count
    assert b.unit == "token"
    assert all(a.text.unit == "token" for _, a in b.attachments)
    if count:
        assert b.to_rex().nE == count


@pytest.mark.parametrize("scheme,data", [
    ("IOB2", "a I-X\n"), ("IOB2", "a B-X\nb I-Y\n"),
    ("BIOES", "a B-X\n"), ("BIOES", "a B-X\nb O\n"),
    ("BIOES", "a E-X\n"), ("BIOES", "a B-X\n\nb E-X\n"),
    ("BIOES", "a B-X\nb B-X\n"), ("IOB1", "a Q-X\n")])
def test_conll_rejects_invalid_transitions(tmp_path, scheme, data):
    p = tmp_path / "x.conll"; p.write_bytes(data.encode())
    with pytest.raises(ValueError): read_conll(p, document_id="doc", scheme=scheme)


def test_conll_no_guessed_character_offsets_or_lineage(tmp_path):
    p = tmp_path / "x.conll"; p.write_bytes(b"a B-X\nb O\n")
    a = read_conll(p, document_id="doc")
    p.write_bytes(b"a B-X\nb I-X\n")
    b = read_conll(p, document_id="doc")
    assert a.axis == b.axis
    assert a.records[0].identifier != b.records[0].identifier
    assert a.manifest["profile"]["identity_origin"] == "span_content_not_inferred_lineage"


def test_conll_sentence_boundary_and_custom_columns(tmp_path):
    p = tmp_path / "x.conll"
    p.write_bytes(b"-DOCSTART- -X- O\na NN B-X\nb NN I-X\n\nc NN B-X\n")
    b = read_conll(p, document_id="doc", token_column=0, tag_column=2)
    assert [r.components for r in b.records] == [((0, 2),), ((2, 3),)]


def test_ingestion_sidecar_change_and_expected_version(tmp_path):
    p = fixture(tmp_path)
    with closing(open_store("memory://")) as store:
        rid, one = ingest_annotations(store, p, document_id="doc", tx_time=20, expected_version=0)
        assert one["version"] == 1
        assert ingest_annotations(store, p, document_id="doc", tx_time=21)[1] is None
        with pytest.raises(VersionConflictError):
            ingest_annotations(store, p, document_id="doc", expected_version=0)
        p.write_bytes(p.read_bytes().replace(b"Negation E2", b"Negation E1"))
        _, two = ingest_annotations(store, p, document_id="doc", tx_time=30)
        assert two["version"] == 2
        assert one["bundle_digest"] != two["bundle_digest"]
        assert store.verify_commits(rid)


def test_ingestion_temporal_declaration_change(tmp_path):
    p = fixture(tmp_path)
    with closing(open_store("memory://")) as store:
        _, a = ingest_annotations(store, p, document_id="test_document", tx_time=20, **temporal_options(p))
        _, b = ingest_annotations(store, p, document_id="test_document", tx_time=30, **temporal_options(p, second=2025))
        assert (a["version"], b["version"]) == (1, 2)
        assert a["bundle_digest"] != b["bundle_digest"]


def test_source_heap_relocation_and_tampering(tmp_path):
    p = fixture(tmp_path / "before")
    rex = read_brat(p, document_id="doc").to_rex()
    relocated = tmp_path / "elsewhere μ"; shutil.copytree(p.parent, relocated)
    assert len(verify_annotation_sources(rex, relocated)) == 2
    (relocated / "case.ann").write_bytes(b"different")
    with pytest.raises(ValueError): verify_annotation_sources(rex, relocated)


@pytest.mark.parametrize("backend", ["memory", "file", "rex", "sql"])
def test_native_store_roundtrip(tmp_path, backend):
    p = fixture(tmp_path / "inputs")
    uri = {"memory": "memory://", "file": f"file://{tmp_path / 'files'}",
           "rex": str(tmp_path / "data.rexdb"), "sql": f"sqlite:///{tmp_path / 'data.sqlite'}"}[backend]
    store = open_store(uri)
    try:
        _, meta = ingest_annotations(store, p, document_id="test_document", **temporal_options(p))
        snapshot = store.read_record("test_document", version=meta["version"])
        assert len(AttachmentField.from_source(snapshot.value).attachments) == 10
        assert verify_annotation_sources(snapshot.value, p.parent)
        assert store.verify_commits("test_document")
    finally:
        store.close()


def test_bitemporal_selection_does_not_read_later_annotation(tmp_path):
    p = fixture(tmp_path)
    with closing(open_store("memory://")) as store:
        ingest_annotations(store, p, document_id="test_document", tx_time=20,
                           valid_from=2017, valid_to=2027, **temporal_options(p))
        query = parse('FROM RCDB_AS_OF($db,"test_document",25) RETURN ATTACHMENTS()')
        before = Executor(sources={"db": store}).execute(query).values[0]
        ingest_annotations(store, p, document_id="test_document", tx_time=30,
                           valid_from=2017, valid_to=2027, **temporal_options(p, second=2025))
        after = Executor(sources={"db": store}).execute(query).values[0]
        assert before.coefficient_digest == after.coefficient_digest
        assert before.source.version == after.source.version == 1
        new = store.read_record("test_document", as_of=35, valid_at=2025)
        assert new.record.version == 2
        assert store.read_record("test_document", as_of=19, valid_at=2025) is None


def test_import_query_retained_temporal_moment(tmp_path):
    p = fixture(tmp_path)
    with closing(open_store("memory://")) as store:
        ingest_annotations(store, p, document_id="test_document", tx_time=20, **temporal_options(p))
        ingest_annotations(store, p, document_id="test_document", tx_time=30, **temporal_options(p, second=2025))
        def select(v):
            return Executor(sources={"db": store}).execute(parse(
                f'FROM RCDB_VERSION($db,"test_document",{v}) RETURN ATTACHMENTS()')).values[0]
        old, new = select(1), select(2)
        old = AttachmentField(tuple(a for a in old.attachments if a.annotation_id == "E1/Time/0"), old.source)
        new = AttachmentField(tuple(a for a in new.attachments if a.annotation_id == "E1/Time/0"), new.source)
        old_o, new_o = common_attachment_observations(old, new, support="grounded_time", local=False)
        old_y, new_y = old_o.field.amplitudes(), new_o.field.amplitudes()
        from rexgraph.tensor_field import apply_tensor
        a, b = apply_tensor(old_o.action, old_y), apply_tensor(new_o.action, new_y)
        from rexgraph.tensor_field import TensorField
        delta = TensorField(a.space, b.values-a.values, source=b.source, dependencies=(old.source,new.source))
        assert delta.values.tolist() == [0, 0, -1, 0]
        moment = MomentSpan(delta, delta, CoordinatePairing.metric(new_o.metric))
        assert moment.support().values.tolist() == [0, 0, 1, 0]
        assert moment.scalar() == 1


def test_byte_coordinates_and_record_reload(tmp_path):
    from agent.adapters.annotations import annotation_records
    b = read_brat(fixture(tmp_path), document_id="doc")
    row = next(r for r in b.records if r.identifier == "T2")
    assert row.components == ((7, 12),)
    assert row.byte_components == ((8, 13),)
    restored = from_state(to_state(b.to_rex()))
    assert annotation_records(restored) == b.records


def test_event_role_order_is_not_attachment_identity(tmp_path):
    p = fixture(tmp_path)
    a = read_brat(p, document_id="doc")
    p.write_bytes(p.read_bytes().replace(b"Agent:T1 Time:T3", b"Time:T3 Agent:T1"))
    b = read_brat(p, document_id="doc")
    aa = {v.annotation_id: v for _, v in a.attachments}
    ab = {v.annotation_id: v for _, v in b.attachments}
    assert aa == ab
    assert a.digest != b.digest


def test_nested_event_reference_is_retained_without_invented_span(tmp_path):
    p = fixture(tmp_path)
    with p.open("ab") as f:
        f.write(b"E3\tTreatment:T2 Cause:E1\r\n")
    b = read_brat(p, document_id="doc")
    row = next(r for r in b.records if r.identifier == "E3")
    assert row.references[-1] == ("Cause", "E1")
    assert [a.role for owner, a in b.attachments if owner == "E3"] == ["Trigger"]
    assert b.to_rex().nE == 13


def test_repeated_components_are_not_silently_deduplicated(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"alpha")
    p = tmp_path / "x.ann"; p.write_bytes(b"T1\tThing 0 5;0 5\talpha alpha\n")
    b = read_brat(p, document_id="doc")
    assert len(b.records[0].components) == 2
    assert len(b.construction().branching[0]) == 2
    field = AttachmentField.from_source(b.to_rex())
    summed = field.observe(support="text", local=False, mode="sum").evaluate()
    union = field.observe(support="text", local=False, mode="union").evaluate()
    assert summed.values.tolist() == [2]
    assert union.values.tolist() == [1]


def test_empty_annotations_are_a_valid_native_state(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"unannotated text")
    p = tmp_path / "x.ann"; p.write_bytes(b"")
    with closing(open_store("memory://")) as store:
        ingest_annotations(store, p, document_id="empty")
        rex = store.read_record("empty").value
        assert rex.nE == 0
        assert AttachmentField.from_source(rex).attachments == ()


def test_empty_conll_annotations_do_not_invent_entities(tmp_path):
    p = tmp_path / "x.conll"; p.write_bytes(b"alpha O\nbeta O\n")
    rex = read_conll(p, document_id="empty").to_rex()
    assert rex.nE == 0


@pytest.mark.parametrize("arguments", [
    {"valid_from": True}, {"valid_to": float("nan")}, {"tx_time": float("inf")},
    {"valid_from": 2, "valid_to": 1}, {"expected_version": True}])
def test_ingestion_rejects_invalid_selection(tmp_path, arguments):
    p = fixture(tmp_path)
    with closing(open_store("memory://")) as store:
        with pytest.raises((ValueError, TypeError)):
            ingest_annotations(store, p, document_id="doc", **arguments)
        assert store.read_record("doc") is None


def test_exact_event_endpoints_not_converted_by_record_clock(tmp_path):
    p = fixture(tmp_path)
    options = temporal_options(p)
    old = options["time_supports"]["E1/Time/0"]
    b = SpanBlock(old.name, old.axis, old.unit, (("a", Q(1, 3), Q(5, 7)), ("b", 2**65, 2**65 + 1)))
    options["time_supports"]["E1/Time/0"] = b
    base = read_brat(p, document_id="test_document")
    a = next(v for _, v in base.attachments if v.annotation_id == "E1/Time/0")
    options["groundings"] = {a.annotation_id: CoordinateMap(a.text.coordinates, b.coordinates, ((0, 0, 1), (1, 1, 1)))}
    with closing(open_store("memory://")) as store:
        ingest_annotations(store, p, document_id="test_document", valid_from=Q(1, 3), **options)
        rex = store.read_record("test_document").value
        actual = AttachmentField.from_source(rex, annotation_ids=[a.annotation_id]).attachments[0]
        assert actual.time == b


