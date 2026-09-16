"""Exact coefficient identity and storage do not depend on Python object addresses."""
from __future__ import annotations

from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.rex_state import (
    CODEC_TENSOR,
    decode_tensors,
    encode_tensors,
    from_state,
    state_digest,
    to_state,
    verify_state,
)


def weighted():
    return RexGraph(boundary_ptr=np.array([0, 4, 6], np.int32),
                    boundary_idx=np.array([2, 0, 1, 3, 0, 2], np.int32),
                    w_E=np.array([Q(2, 3), Q(-5, 7)], dtype=object))


@pytest.mark.parametrize("shape", [(0,), (2, 0), (), (1, 2, 2)])
def test_codec_preserves_shape_exact_values_and_numeric_kinds(shape):
    values = [Q(-17, 20), 1 << 16000, Q(3), -(1 << 16001)]
    array = np.array(values[:int(np.prod(shape))], dtype=object).reshape(shape)
    tensors = {"x": array.copy()}
    spec = encode_tensors(tensors)
    assert tensors["x"].dtype == np.uint8
    assert spec == {"x": {"c": "exact", "shape": list(shape)}}
    decode_tensors(tensors, spec)
    np.testing.assert_array_equal(tensors["x"], array)
    assert [type(x) for x in tensors["x"].flat] == [type(x) for x in array.flat]


def test_independent_objects_share_identity_and_coefficient_change_moves_it():
    first, second = weighted(), weighted()
    assert first._w_E[0] is not second._w_E[0]
    assert object_digest(first) == object_digest(second)
    state = to_state(first)
    assert state.header["format_version"] == 3
    assert verify_state(state)
    assert all(not a.dtype.hasobject for a in state.tensors.values())
    restored = from_state(state)
    assert object_digest(restored) == object_digest(first)
    np.testing.assert_array_equal(restored._w_E, first._w_E)
    second._w_E[0] = Q(3, 4)
    assert object_digest(first) != object_digest(second)
    state.tensors["w_E"][0] ^= np.uint8(1)
    assert not verify_state(state)


def test_numeric_states_keep_version_two_and_no_new_codec_payload():
    state = to_state(RexGraph.from_graph([0, 1], [1, 2]))
    assert state.header["format_version"] == 2
    assert b'"exact"' not in state.tensors[CODEC_TENSOR].tobytes()


@pytest.mark.parametrize("kind", ["safetensors", "rcbd"])
def test_framework_containers_preserve_rational_weights(kind, tmp_path):
    value = weighted()
    if kind == "safetensors":
        pytest.importorskip("safetensors")
        from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
        path = rex_to_safetensors(value, tmp_path / "r.safetensors")
        restored = safetensors_to_rex(path)
    else:
        from rexgraph.io.bundle import load_rcbd, save_rcbd
        path = tmp_path / "r.rcbd"
        save_rcbd(path, value)
        restored = load_rcbd(path)
    np.testing.assert_array_equal(restored._w_E, value._w_E)
    assert object_digest(restored) == object_digest(value)


@pytest.mark.parametrize("raw", [b"q:1/0\n", b"i:01\n", b"q:2/4\n", b"i:1", b"f:1.2\n"])
def test_noncanonical_or_invalid_exact_payload_is_refused(raw):
    tensors = {"x": np.frombuffer(raw, np.uint8)}
    with pytest.raises(ValueError, match="exact coefficient"):
        decode_tensors(tensors, {"x": {"c": "exact", "shape": [1]}})


@pytest.mark.parametrize("value", [object(), "one", 1.25, True, complex(1, 2)])
def test_other_objects_are_not_serialized_or_hashed_as_addresses(value):
    tensors = {"x": np.array([value], dtype=object)}
    with pytest.raises(TypeError, match="integer or Fraction"):
        encode_tensors(dict(tensors))
    with pytest.raises(TypeError, match="before hashing"):
        state_digest(tensors)


def test_nested_rational_rex_keeps_its_independent_state_identity():
    outer = RexGraph.from_graph([0], [1])
    child = weighted()
    outer.attach_metadata(0, 0, "child", child)
    restored = from_state(to_state(outer))
    assert object_digest(restored.get_metadata(0, 0, "child")) == object_digest(child)
    assert object_digest(restored) == object_digest(outer)
