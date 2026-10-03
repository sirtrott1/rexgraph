"""Native callers must be safe even when they bypass the public wrappers."""
import numpy as np
import pytest

from rexgraph.core import _quotient
from rexgraph.graph import RexGraph


@pytest.mark.parametrize("complex_signal", [False, True])
@pytest.mark.parametrize("mask_length", [1, 4])
def test_native_restrict_refuses_wrong_mask_length(complex_signal, mask_length):
    fn = _quotient.restrict_signal_complex if complex_signal else _quotient.restrict_signal
    signal = np.arange(3, dtype=np.complex128 if complex_signal else np.float64)
    with pytest.raises(ValueError, match="length"):
        fn(signal, np.zeros(mask_length, np.uint8))


@pytest.mark.parametrize("complex_signal", [False, True])
@pytest.mark.parametrize("signal_length", [1, 3])
def test_native_lift_refuses_wrong_quotient_length(complex_signal, signal_length):
    fn = _quotient.lift_signal_complex if complex_signal else _quotient.lift_signal
    signal = np.arange(signal_length, dtype=np.complex128 if complex_signal else np.float64)
    with pytest.raises(ValueError, match="length"):
        fn(signal, np.array([0, 1, 0], np.uint8))


@pytest.mark.parametrize("grade", [0, 1])
def test_native_field_restrict_validates_both_masks(grade):
    signals = [np.arange(3, dtype=float), np.arange(2, dtype=float)]
    masks = [np.zeros(3, np.uint8), np.zeros(2, np.uint8)]
    masks[grade] = np.zeros(1, np.uint8)
    with pytest.raises(ValueError, match="length"):
        _quotient.restrict_field_state(*signals, *masks)


@pytest.mark.parametrize("grade", [0, 1])
def test_native_field_lift_validates_both_quotients(grade):
    signals = [np.arange(2, dtype=float), np.arange(1, dtype=float)]
    masks = [np.array([0, 1, 0], np.uint8), np.array([1, 0], np.uint8)]
    signals[grade] = np.zeros(0)
    with pytest.raises(ValueError, match="length"):
        _quotient.lift_field_state(*signals, *masks)


def test_public_signal_restrict_lift_round_trip():
    rex = RexGraph.from_graph([0, 1, 2], [1, 2, 3])
    mask = np.array([0, 1, 0], np.uint8)
    assert rex.restrict_signal([2, 4, 8], mask).tolist() == [2, 8]
    assert rex.lift_signal([2, 8], mask).tolist() == [2, 0, 8]
    with pytest.raises(ValueError, match="length"):
        rex.restrict_signal([2, 4, 8], [0])
    with pytest.raises(ValueError, match="length"):
        rex.lift_signal([2], mask)
