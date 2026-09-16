"""Core artifact semantics, provider isolation and observable execution."""
import hashlib
import json
from dataclasses import replace

import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.commit import CommitLink
from rexgraph.io.manifest import canonical_json, manifest_digest
from rexgraph.io.privacy import StaticIdentityKeyProvider, scoped_pseudonym
from rexgraph.io.security import StaticKeyProvider, envelope_info
from rexgraph.io.transport import inspect, pack, unpack

from rcql import ArtifactServices, BoundSource, Executor, SourcePolicy, call, parse, query, source
from rcql.artifact_contracts import OBSERVABLE
from rcql.execution_trace import capture_methods, current_artifact_services
from rcql.executor import value_exactness
from rcql.operators import get_operator


@pytest.fixture
def rex():
    return RexGraph.from_cells([4, [[0, 1, 2, 3], [0, 0], [3]]], relation_ids=[7, 19, 53])


@pytest.fixture
def services():
    pytest.importorskip("cryptography")
    from rexgraph.io.security import Ed25519Signer
    signer = Ed25519Signer.generate("author")
    return ArtifactServices(keys=StaticKeyProvider({"artifact": b"K"*32}),
        identity_keys=StaticIdentityKeyProvider({"scope-key": b"scope-secret"}),
        signers={"author": signer}, verifiers={"author": signer.verifier()})


def engine(rex, services=None, **params):
    return Executor(sources={"r": rex}, params=params, artifacts=services)


def test_hash_domains_and_transport_preserve_original_bytes(rex):
    payload = b"\x00\xff exact payload"
    metadata = {"grade": 4, "fraction": {"numerator": "1", "denominator": "3"}}
    e = engine(rex, payload=payload, meta=metadata)
    result = e.execute(parse('FROM $r LET p=TRANSPORT($payload,"native-state",$meta) '
                             'RETURN p, MANIFEST(p), HASH(p,kind="bytes"), HASH($meta,"manifest"), HASH(), STATE_HASH()'))
    framed, info, digest, mdigest, state, expected = result.values
    assert framed == pack(payload, object_type="native-state", metadata=metadata)
    assert unpack(framed) == (payload, {"object_type": "native-state", "metadata": metadata,
        "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_size": len(payload), "version": 1})
    assert info["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert info["verification"] == "header-only"
    assert digest == hashlib.sha256(framed).hexdigest() and mdigest == manifest_digest(metadata)
    assert state == expected and state != digest
    assert all(v.value == "structural" for v in result.exactness)


def test_lineage_is_explicit_and_is_not_signature_verification(rex):
    link = CommitLink("transition", "parent")
    e = engine(rex, link=link)
    out = e.execute(parse('FROM $r LET p=PARTITION(CELL(1,0)) '
        'RETURN LINEAGE(p), MANIFEST(p), HASH(p,"lineage"), LINEAGE($link), HASH($link,"lineage")')).values
    assert out[0] == out[1] and out[0]["digest"] == out[2]
    assert out[0]["manifest"]["result_state"] and out[0]["signature_verified"] is False
    assert out[3]["manifest"] == link.manifest() and out[4] == link.digest
    assert out[3]["signature_verified"] is False


def test_capabilities_describe_actual_binding_without_granting_read(rex):
    policy = SourcePolicy.allow(record_fields={"nE"})
    e = engine(BoundSource(rex, policy))
    value = e.execute(parse('FROM $r RETURN SHOW_CAPABILITIES()')).values[0]
    assert value["permissions"] == () and value["record_fields"] == ("nE",)
    assert value["policy_digest"] == policy.digest
    with pytest.raises(PermissionError):
        e.execute(parse('FROM $r RETURN HASH()'))


def test_authenticated_roundtrip_and_corruption_failures(rex, services):
    from cryptography.exceptions import InvalidTag
    e = engine(rex, services, payload=b"untouched binary \x00\xff")
    out = e.execute(parse('FROM $r LET t=TRANSPORT($payload,"bytes") '
        'LET a=ENCRYPT(t,key_id="artifact",object_type="REXPKG") '
        'RETURN a, DECRYPT(a), MANIFEST(a)')).values
    encrypted, plaintext, public = out
    assert unpack(plaintext)[0] == e.params["payload"]
    assert public["object_type"] == "REXPKG" and public["verification"] == "header-only"
    assert envelope_info(encrypted).key_id == "artifact"
    e.params["bad"] = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
    with pytest.raises(InvalidTag):
        e.execute(parse('FROM $r RETURN DECRYPT($bad)'))
    wrong = ArtifactServices(keys=StaticKeyProvider({"artifact": b"X"*32}))
    with pytest.raises(InvalidTag):
        engine(rex, wrong, a=encrypted).execute(parse('FROM $r RETURN DECRYPT($a)'))


def test_signature_core_parity_and_wrong_bytes_return_false(rex, services):
    link = CommitLink("transition", "parent")
    e = engine(rex, services, payload=link.signing_bytes(), altered=b"other")
    signature, good, bad = e.execute(parse('FROM $r LET s=SIGN($payload,"author") '
        'RETURN s, VERIFY_SIGNATURE($payload,s,"author"), VERIFY_SIGNATURE($altered,s,"author")')).values
    assert signature == services.signers["author"].sign(link.signing_bytes())
    assert good is True and bad is False
    assert replace(link, signer_id="author", signature=signature).verify(services.verifiers["author"])


def test_pseudonym_scope_and_core_parity(rex, services):
    values = engine(rex, services).execute(parse('FROM $r RETURN '
        'PSEUDONYMIZE("sample","one","scope-key"), PSEUDONYMIZE("sample","two","scope-key")')).values
    assert values[0] == scoped_pseudonym("sample", scope="one", key_id="scope-key", keys=services.identity_keys)
    assert values[0] != values[1]


def test_random_envelopes_are_not_implicitly_reused_but_let_captures_once(rex, services):
    e = engine(rex, services, x=b"data")
    result = e.execute(parse('FROM $r RETURN ENCRYPT($x,"artifact"), ENCRYPT($x,"artifact")'))
    assert result.values[0] != result.values[1]
    calls = [n for n in result.native_plan["nodes"] if n.get("operator") == "ENCRYPT"]
    assert len(calls) == 2 and all(not n["reusable"] for n in calls)
    result = e.execute(parse('FROM $r LET a=ENCRYPT($x,"artifact") RETURN a,a'))
    assert result.values[0] == result.values[1]


@pytest.mark.parametrize("name", sorted(OBSERVABLE))
def test_provider_operations_have_direct_and_planned_parity(rex, services, name):
    encrypted = engine(rex, services, x=b"data").execute(parse('FROM $r RETURN ENCRYPT($x,"artifact")')).values[0]
    signature = services.signers["author"].sign(b"data")
    args = {"ENCRYPT": (b"data", "artifact"), "DECRYPT": (encrypted,),
            "SIGN": (b"data", "author"), "VERIFY_SIGNATURE": (b"data", signature, "author"),
            "PSEUDONYMIZE": ("item", "scope", "scope-key")}[name]
    q = query(source("r"), call(name, *args))
    with capture_methods(artifact_services=services):
        direct = get_operator(name).fn(rex, *args)
    e = engine(rex, services)
    actual = e.execute(q)
    declared = e.execute(replace(q, explain=True)).values[0]["returns"][0]
    assert value_exactness(direct) == actual.exactness[0]
    assert declared["result"]["exactness"] == actual.exactness[0].value == "structural"
    assert not declared["memoizable"]
    if name != "ENCRYPT":
        assert direct == actual.values[0]


@pytest.mark.parametrize("prefix", ["", "EXPLAIN "])
@pytest.mark.parametrize("expression", ['ENCRYPT(12,"artifact")', 'DECRYPT("text")',
    'SIGN($x,"")', 'VERIFY_SIGNATURE($x,3,"author")', 'PSEUDONYMIZE(42,"s","k")',
    'PSEUDONYMIZE("v","","k")', 'TRANSPORT($x,"")', 'HASH($x,"state")',
    'HASH($x,"unknown")', 'LINEAGE($meta)', 'MANIFEST(123)'])
def test_invalid_contracts_refused_before_any_adapter(rex, prefix, expression, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises((TypeError, ValueError)):
        engine(rex, x=b"bytes", meta={}).execute(parse(prefix+'FROM $r RETURN '+expression))


@pytest.mark.parametrize("metadata", [{1: "value"}, {"nested": {2: "value"}}, {"v": float("nan")}])
@pytest.mark.parametrize("name", ["HASH", "TRANSPORT"])
@pytest.mark.parametrize("prefix", ["", "EXPLAIN "])
def test_json_mapping_does_not_coerce_keys_or_nonfinite_values(rex, metadata, name, prefix, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    text = 'HASH($meta,"manifest")' if name == "HASH" else 'TRANSPORT($x,"bytes",$meta)'
    with pytest.raises((TypeError, ValueError)):
        engine(rex, x=b"x", meta=metadata).execute(parse(prefix+'FROM $r RETURN '+text))


def test_explain_never_reads_keys_or_signers_and_services_are_not_query_values(rex):
    class Forbidden:
        signer_id = "author"
        def __repr__(self):
            pytest.fail("provider repr inspected")
        def key(self, *args):
            pytest.fail("provider invoked")
        sign = key
        verify = key
    services = ArtifactServices(keys=Forbidden(), identity_keys=Forbidden(),
                                signers={"author": Forbidden()}, verifiers={"author": Forbidden()})
    e = engine(rex, services, x=b"private bytes", s=b"private signature")
    q = parse('EXPLAIN FROM $r RETURN ENCRYPT($x,"artifact"), DECRYPT($x), SIGN($x,"author"), '
              'VERIFY_SIGNATURE($x,$s,"author"), PSEUDONYMIZE("value","scope","key")')
    result = e.execute(q)
    serialized = json.dumps(result.native_plan)
    assert "private bytes" not in serialized and "private signature" not in serialized
    assert "Forbidden" not in serialized and result.execution == ()
    e.params["services"] = services
    with pytest.raises(TypeError, match="configuration"):
        e.execute(parse('FROM $r RETURN $services'))


@pytest.mark.parametrize("prefix", ["", "EXPLAIN "])
@pytest.mark.parametrize("expression", ['ENCRYPT($x,"k")', 'DECRYPT($x)', 'SIGN($x,"s")',
    'VERIFY_SIGNATURE($x,$x,"s")', 'PSEUDONYMIZE("v","scope","k")', 'HASH()',
    'MANIFEST(PARTITION(CELL(1,0)))', 'LINEAGE(PARTITION(CELL(1,0)))'])
def test_missing_permissions_refused_before_provider_resolution(rex, prefix, expression, monkeypatch):
    import rcql.executor
    monkeypatch.setattr(rcql.executor, "get_operator", lambda *a: pytest.fail("adapter reached"))
    with pytest.raises(PermissionError):
        engine(BoundSource(rex, SourcePolicy.allow("read")), x=b"x").execute(parse(prefix+'FROM $r RETURN '+expression))


def test_context_is_reset_on_failure_and_between_executors(rex, services):
    with pytest.raises(ValueError, match="configured"):
        current_artifact_services()
    with pytest.raises(ValueError):
        engine(rex, services, x=b"bad").execute(parse('FROM $r RETURN DECRYPT($x)'))
    with pytest.raises(ValueError, match="configured"):
        current_artifact_services()
    with pytest.raises(ValueError, match="configured"):
        engine(rex, x=b"x").execute(parse('FROM $r RETURN SIGN($x,"author")'))
    outer = ArtifactServices()
    with capture_methods(artifact_services=outer):
        engine(rex, services, x=b"x").execute(parse('FROM $r RETURN SIGN($x,"author")'))
        assert current_artifact_services() is outer


@pytest.mark.parametrize("version", [True, 1.9, "1", None])
def test_envelope_versions_are_not_silently_coerced(rex, services, version):
    from rexgraph.io.security import ENVELOPE_MAGIC
    a = engine(rex, services, x=b"x").execute(parse('FROM $r RETURN ENCRYPT($x,"artifact")')).values[0]
    offset = len(ENVELOPE_MAGIC)
    length = int.from_bytes(a[offset:offset+4], "big")
    header = json.loads(a[offset+4:offset+4+length])
    header["version"] = version
    encoded = canonical_json(header)
    bad = ENVELOPE_MAGIC + len(encoded).to_bytes(4, "big") + encoded + a[offset+4+length:]
    with pytest.raises(ValueError, match="version"):
        envelope_info(bad)
    with pytest.raises(ValueError, match="version"):
        engine(rex, services, a=bad).execute(parse('FROM $r RETURN MANIFEST($a)'))


def test_native_transport_header_is_not_a_payload_integrity_claim(rex):
    a = pack(b"data", object_type="bytes")
    corrupt = a[:-1] + b"x"
    result = engine(rex, a=corrupt).execute(parse('FROM $r RETURN MANIFEST($a)')).values[0]
    assert result["verification"] == "header-only" and inspect(corrupt).payload_size == 4
    with pytest.raises(ValueError, match="digest"):
        unpack(corrupt)


def test_provider_calls_are_not_suppressed_through_pure_parents(rex):
    class Counting:
        signer_id = "author"
        calls = 0
        def sign(self, payload):
            self.calls += 1
            return payload + bytes([self.calls])
    signer = Counting()
    e = engine(rex, ArtifactServices(signers={"author": signer}), x=b"x")
    result = e.execute(parse('FROM $r RETURN HASH(SIGN($x,"author"),"bytes"), '
                             'HASH(SIGN($x,"author"),"bytes")'))
    assert signer.calls == 2 and result.values[0] != result.values[1]


def test_concurrent_executors_keep_their_provider_context(rex):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    class Provider:
        signer_id = "author"
        def __init__(self, result):
            self.result = result
        def sign(self, payload):
            barrier.wait(timeout=10)
            return self.result
    def work(result):
        services = ArtifactServices(signers={"author": Provider(result)})
        return engine(rex, services, x=b"x").execute(parse('FROM $r RETURN SIGN($x,"author")')).values[0]
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(work, [b"first", b"second"])) == [b"first", b"second"]


def test_rcdb_partition_artifact_roundtrip_preserves_state_and_lineage(rex, services, tmp_path):
    import rcdb
    from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
    from rexgraph.io.catalog import object_digest
    uri = f"rex://{tmp_path / 'store'}"
    store = rcdb.open_store(uri)
    try:
        store.put("r", rex, analytics=False)
        before = store.read_record("r")
        e = Executor(sources={"db": store}, artifacts=services)
        partition = e.execute(parse('FROM RCDB_GET($db,"r") RETURN PARTITION(CELLS(1))')).values[0]
        original = tmp_path / "original.safetensors"
        rex_to_safetensors(partition.rex, original)
        e.params.update(payload=original.read_bytes(), lineage=partition.manifest)
        q = parse('FROM RCDB_GET($db,"r") LET t=TRANSPORT($payload,"safetensors",$lineage) '
                  'LET a=ENCRYPT(t,"artifact") LET s=SIGN(a,"author") '
                  'RETURN a,s,VERIFY_SIGNATURE(a,s,"author"),DECRYPT(a),HASH(),SHOW_CAPABILITIES()')
        encrypted, signature, valid, opened, digest, caps = e.execute(q).values
        assert valid and services.verifiers["author"].verify(encrypted, signature)
        assert digest == before.state_digest and caps["scope"] == "bound-source"
        payload, header = unpack(opened)
        assert payload == original.read_bytes() and header["metadata"] == partition.manifest
        restored = tmp_path / "restored.safetensors"
        restored.write_bytes(payload)
        child = safetensors_to_rex(restored)
        assert object_digest(child) == partition.state.result_state
        assert child.relation_ids.tolist() == rex.relation_ids.tolist()
        store.put("child", child, meta={"partition": partition.manifest}, analytics=False)
        assert store.read_record("r").record.version == before.record.version
    finally:
        store.close()
    store = rcdb.open_store(uri)
    try:
        result = Executor(sources={"db": store}).execute(parse('FROM RCDB_GET($db,"child") RETURN HASH(), ARITY(CELL(1,0))'))
        assert result.values == (partition.state.result_state, 4)
        assert store.read_record("child").record.meta["partition"] == partition.manifest
    finally:
        store.close()
