"""State inspection shares normal safetensors ownership rules without graph builds."""
import json

import numpy as np
import pytest

from rexgraph import RexGraph, TemporalRex
from rexgraph.object_identity import object_digest
from rexgraph.io.safetensors_bridge import state_from_safetensors_bytes, rex_to_safetensors, temporal_rex_to_safetensors


@pytest.mark.parametrize("temporal", [False, True])
def test_native_state_bytes_roundtrip_uses_no_file_publication(temporal, monkeypatch):
    from rexgraph.io.safetensors_bridge import state_to_safetensors_bytes
    from rexgraph.state import to_state
    from rexgraph.temporal_state import to_temporal_state
    from rexgraph.io import publication
    value = (TemporalRex([(np.array([0]), np.array([1]))]) if temporal
             else RexGraph.from_graph([0], [1]))
    state = to_temporal_state(value) if temporal else to_state(value)
    monkeypatch.setattr(publication, "staged_publication", lambda *a, **k: pytest.fail("native bytes wrote a file"))
    raw = state_to_safetensors_bytes(state)
    restored = state_from_safetensors_bytes(raw)
    assert restored.header["digest"] == state.header["digest"]
    assert set(restored.tensors) == set(state.tensors)


def test_native_state_bytes_refuse_a_tampered_semantic_state():
    from rexgraph.io.safetensors_bridge import state_to_safetensors_bytes
    from rexgraph.state import to_state, RexState
    state = to_state(RexGraph.from_graph([0], [1]))
    bad = RexState(state.tensors, {**state.header, "digest": "0"*64})
    with pytest.raises(ValueError):
        state_to_safetensors_bytes(bad)


def encoded(path):
    raw = path.read_bytes()
    length = int.from_bytes(raw[:8], "little")
    return raw, json.loads(raw[8:8+length])


@pytest.mark.parametrize("temporal", [False, True])
def test_verified_state_identity_without_reconstructing_graph(temporal, tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    from rexgraph import state as static_state
    from rexgraph import temporal_state
    value = RexGraph.from_graph([0], [1])
    if temporal:
        value = TemporalRex([(np.array([0]), np.array([1]))])
    expected = object_digest(value)
    writer = temporal_rex_to_safetensors if temporal else rex_to_safetensors
    path = writer(value, tmp_path / "state.safetensors")
    def forbidden(*args, **kwargs):
        pytest.fail("state inspection reconstructed a graph")
    monkeypatch.setattr(static_state, "from_state", forbidden)
    monkeypatch.setattr(temporal_state, "from_temporal_state", forbidden)
    restored = state_from_safetensors_bytes(path.read_bytes())
    assert restored.header["digest"] == expected


def test_auxiliary_digest_and_unclaimed_tensor_refusals_match_normal_reader(tmp_path):
    st = pytest.importorskip("safetensors.numpy")
    path = rex_to_safetensors(RexGraph.from_graph([0], [1]), tmp_path / "state.safetensors",
                             extra_tensors={"field": np.array([1, 2])})
    raw, header = encoded(path)
    assert state_from_safetensors_bytes(raw).header["object_type"] == "RexGraph"
    tensors = st.load(raw)
    tensors["field"][0] = 9
    with pytest.raises(ValueError, match="auxiliary payload digest"):
        state_from_safetensors_bytes(st.save(tensors, metadata=header["__metadata__"]))
    tensors["unclaimed"] = np.array([1])
    with pytest.raises(ValueError, match="auxiliary payloads"):
        state_from_safetensors_bytes(st.save(tensors, metadata=header["__metadata__"]))


def test_duplicate_native_metadata_declaration_is_refused(tmp_path):
    st = pytest.importorskip("safetensors.numpy")
    path = rex_to_safetensors(RexGraph.from_graph([0], [1]), tmp_path / "state.safetensors")
    raw, header = encoded(path)
    metadata = header["__metadata__"]
    metadata["rex_state_header"] = '{"object_type":"other",'+metadata["rex_state_header"][1:]
    with pytest.raises(ValueError, match="duplicate"):
        state_from_safetensors_bytes(st.save(st.load(raw), metadata=metadata))


@pytest.mark.parametrize("raw", [b"", b"bad state", b"\xff"*8, b"\x01"+b"\x00"*7+b"0"])
def test_invalid_safetensors_state_has_a_consistent_refusal(raw):
    pytest.importorskip("safetensors")
    with pytest.raises(ValueError):
        state_from_safetensors_bytes(raw)
