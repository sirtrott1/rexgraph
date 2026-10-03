"""Shared adversarial values for readers, state, stores and exact result payloads."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction as Q
import pickle

import numpy as np
import pytest

from rexgraph.exact_array import ExactArray
from rexgraph.value import Absent, Approx, ExactTime, NumberRule, TimeRange, ValueRules, convert_number
from rexgraph.value_codec import pack_value, unpack_value


@pytest.mark.parametrize("value,rule,expected", [
    ("0.1", NumberRule.DECIMAL_EXACT, Q(1, 10)),
    ("1.23e-4", NumberRule.DECIMAL_EXACT, Q(123, 1_000_000)),
    (Decimal("0.1"), NumberRule.EXACT, Q(1, 10)),
    (2**53+1, NumberRule.EXACT, Q(2**53+1)),
    (np.int64(2**53+1), NumberRule.EXACT, Q(2**53+1)),
    (.1, NumberRule.BINARY_EXACT, Q.from_float(.1)),
    (.1, NumberRule.XLSX_SHORTEST_DECIMAL, Q(1, 10)),
    ({"numerator": 1, "denominator": 4}, NumberRule.JSON_RATIONAL, Q(1, 4)),
    ([1, 4], NumberRule.JSON_RATIONAL, Q(1, 4)),
    (np.float32(.1), NumberRule.TYPED_COLUMNAR, Q(*np.float32(.1).as_integer_ratio())),
    (-0., NumberRule.BINARY_EXACT, Q(0)),
])
def test_declared_number_rules(value, rule, expected):
    assert convert_number(value, rule) == expected


@pytest.mark.parametrize("complex_values", [False, True])
def test_extended_float_padding_is_not_value_data(complex_values):
    real = np.dtype(np.longdouble)
    info = np.finfo(real)
    if info.nmant != 63 or info.iexp != 15 or real.itemsize <= 10:
        pytest.skip("this platform does not use padded x87 extended precision")
    dtype = np.clongdouble if complex_values else np.longdouble
    value = np.array([1, np.longdouble(1)+info.eps], dtype=dtype)
    first, second = value.copy(), value.copy()
    first.view(np.uint8).reshape(-1, real.itemsize)[:, 10:] = 37
    second.view(np.uint8).reshape(-1, real.itemsize)[:, 10:] = 91
    np.testing.assert_array_equal(first, second)
    assert pack_value(first) == pack_value(second)
    restored = unpack_value(pack_value(first))
    np.testing.assert_array_equal(restored, value)
    assert pack_value(restored) == pack_value(first)
    malformed = bytearray(pack_value(first))
    malformed[-1] = 1
    with pytest.raises(ValueError, match="padding"):
        unpack_value(bytes(malformed))


@pytest.mark.parametrize("container", ["state", "wire", "safetensors"])
def test_extended_graph_weights_keep_value_and_stable_identity(tmp_path, container):
    from rexgraph.graph import RexGraph
    from rexgraph.sealed_state import state_identity
    from rexgraph.state import from_state, to_state
    value = np.longdouble(1)+np.finfo(np.longdouble).eps
    weights = np.array([value], dtype=np.longdouble)
    first, second = weights.copy(), weights.copy()
    if weights.dtype.itemsize > 10 and np.finfo(weights.dtype).nmant == 63:
        first.view(np.uint8).reshape(-1, weights.dtype.itemsize)[:, 10:] = 37
        second.view(np.uint8).reshape(-1, weights.dtype.itemsize)[:, 10:] = 91
    a = RexGraph(sources=[0], targets=[1], w_E=first)
    b = RexGraph(sources=[0], targets=[1], w_E=second)
    assert state_identity(to_state(a)) == state_identity(to_state(b))
    if container == "state":
        back = from_state(to_state(a))
    elif container == "wire":
        from rexgraph.protocol import decode, encode, to_complex
        back = to_complex(decode(encode(a)))
    else:
        pytest.importorskip("safetensors")
        from rexgraph.io.safetensors_bridge import rex_to_safetensors, safetensors_to_rex
        path = tmp_path / "extended.safetensors"
        rex_to_safetensors(a, path)
        back = safetensors_to_rex(path)
    assert back.edge_metric_exact == [Q(*value.as_integer_ratio())]
    assert state_identity(to_state(back)) == state_identity(to_state(a))


def test_extended_binary_scalar_never_narrows_to_double():
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    assert convert_number(value, NumberRule.BINARY_EXACT) == Q(*value.as_integer_ratio())


def test_absence_and_typed_text_remain_distinct():
    rules = ValueRules(NumberRule.DECIMAL_EXACT, ("", "NA", "null"))
    for value in (None, "", "NA", "null", Absent):
        assert rules.convert(value) is Absent
    assert rules.convert("00123", kind="string") == "00123"
    assert rules.convert("0") == 0
    assert pickle.loads(pickle.dumps(Absent)) is Absent
    with pytest.raises(TypeError):
        bool(Absent)


@pytest.mark.parametrize("rule", list(NumberRule))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_invalid_numeric_sources_are_refused(value, rule):
    with pytest.raises((TypeError, ValueError)):
        convert_number(value, rule)


def test_float_requires_declared_source_and_approx_stays_approx():
    with pytest.raises(TypeError):
        convert_number(.1)
    assert convert_number(Approx(.1, "measured")) == Approx(.1, "measured")
    with pytest.raises(TypeError):
        ExactArray.from_values([Approx(.1)])


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (2, 3)])
def test_exact_array_shapes_presence_bigints_and_kinds(shape):
    values = [Q(-1), Q(1, 4), Absent, 2**1000, Q(1, 2**1001+1), 0]
    array = np.array(values[:int(np.prod(shape))], dtype=object).reshape(shape)
    exact = ExactArray.from_values(array)
    for restored in (ExactArray.from_bytes(exact.to_bytes()), ExactArray.from_tensors(exact.to_tensors())):
        assert restored == exact
        np.testing.assert_array_equal(restored.values(), array)
        assert [type(v) for v in restored.values().flat] == [type(v) for v in array.flat]
    for tensor in exact.to_tensors().values():
        assert not tensor.dtype.hasobject
        with pytest.raises(ValueError):
            tensor.flags.writeable = True


def test_asymmetric_column_and_absent_weight_are_exact():
    array = ExactArray.from_values([Q(-1), Q(1, 4), Q(1, 2), Q(1, 4)])
    assert sum(v*v for v in ExactArray.from_bytes(array.to_bytes()).values()) == Q(11, 8)
    assert ExactArray.from_values([Absent]).to_bytes() != ExactArray.from_values([1]).to_bytes()


@pytest.mark.parametrize("value", [Absent, None, False, True, pytest.param(2**16000, id="bigint"), Q(2, 3), "00123", b"\0\xff", -.0,
                                     Approx(.1, "instrument"), ExactTime(Q(1, 1_000_000_000)),
                                     TimeRange(ExactTime(1)), (Absent, Q(1, 4)), [1, "one"]])
def test_closed_value_roundtrip_and_canonical_bytes(value):
    raw = pack_value(value)
    restored = unpack_value(raw)
    if value is Absent:
        assert restored is Absent
    else:
        assert restored == value
    assert pack_value(restored) == raw
    with pytest.raises(ValueError):
        unpack_value(raw + b"\0")


@pytest.mark.parametrize("value", [np.array([1, 2], ">i8"), np.array([1+2j]), np.array("text"),
                                     np.array([Q(1, 4), Absent], object), np.float32(.1), np.longdouble(.1)])
def test_native_tensors_and_numpy_scalars(value):
    restored = unpack_value(pack_value(value))
    np.testing.assert_array_equal(restored, value)
    assert pack_value(restored) == pack_value(value)


def test_canonical_mapping_order_retains_typed_keys_and_sequence_order():
    assert pack_value({"b": [2, 1], (1, 2): Q(1, 4), "a": None}) == pack_value({"a": None, (1, 2): Q(1, 4), "b": [2, 1]})
    assert pack_value([1, 2]) != pack_value([2, 1])
    assert pack_value([1]) != pack_value((1,))


@pytest.mark.parametrize("value", [object(), {1, 2}, float("nan"), np.array([float("inf")]), np.array([object()], object)])
def test_native_values_have_no_string_fallback(value):
    with pytest.raises((TypeError, ValueError)):
        pack_value(value)


def test_exact_utc_time_and_half_open_range():
    stamp = datetime(2026, 9, 29, 12, 0, 0, 1, tzinfo=timezone(timedelta(hours=-5)))
    exact = ExactTime.from_datetime(stamp)
    assert exact == ExactTime.from_datetime(stamp.astimezone(timezone.utc))
    assert exact.seconds.denominator == 1_000_000
    interval = TimeRange(exact, ExactTime(exact.seconds + Q(1, 1000)))
    assert interval.contains(exact)
    assert not interval.contains(interval.end)
    with pytest.raises(ValueError):
        ExactTime.from_datetime(stamp.replace(tzinfo=None))
