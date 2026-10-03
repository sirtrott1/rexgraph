"""One source aware scalar to rational contract for exact RexGraph semantics."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from rexgraph.exact_green import _rational
from rexgraph.exact_value import (
    binary_fraction, decimal_fraction, exact_fraction, validate_exact_array,
)


def test_exact_scalar_accepts_native_and_numpy_integers_without_float_round_trip():
    assert exact_fraction(2**60 + 1) == Fraction(2**60 + 1)
    assert exact_fraction(np.int64(2**60 + 1)) == Fraction(2**60 + 1)
    assert _rational(np.int64(1)) == Fraction(1)


def test_exact_scalar_refuses_approximate_binary_input_when_exact_was_promised():
    with pytest.raises(TypeError, match="binary"):
        exact_fraction(0.1)
    with pytest.raises(TypeError, match="boolean"):
        exact_fraction(True)


def test_source_policy_distinguishes_binary_float_from_decimal_text():
    assert binary_fraction(0.25) == Fraction(1, 4)
    assert binary_fraction(0.1) == Fraction.from_float(0.1)
    assert decimal_fraction("0.1") == Fraction(1, 10)
    assert exact_fraction(Decimal("0.1")) == Fraction(1, 10)


def test_exact_structure_and_evaluator_policy_are_independent():
    from rexgraph import RexGraph
    from rexgraph.core._common import configure_algorithms, get_algorithm_config
    from rexgraph.exact_green import exact_evaluator_available, exact_structure_available

    rex = RexGraph.from_graph(np.asarray([0], np.int32), np.asarray([1], np.int32))
    before = get_algorithm_config()["exact_field_limit"]
    try:
        assert exact_structure_available(rex, 1)
        configure_algorithms(exact_field_limit=0)
        assert exact_structure_available(rex, 1)  # carrier did not change
        assert not exact_evaluator_available(rex, 1)
        assert not exact_structure_available(rex, 0)  # no exact grade 0 solver contract yet
    finally:
        configure_algorithms(exact_field_limit=before)


def test_exact_array_validation_is_semantic_and_io_independent():
    values = np.asarray([1, np.int64(2), Fraction(3, 4)], dtype=object)
    checked = validate_exact_array(values)
    assert checked.tolist() == [1, 2, Fraction(3, 4)]
    with pytest.raises(TypeError, match="integer or Fraction"):
        validate_exact_array(np.asarray([Fraction(1, 2), 0.5], dtype=object))
    with pytest.raises(TypeError, match="integer or Fraction"):
        validate_exact_array(np.asarray([True], dtype=object))


def test_model_state_exact_array_validation_does_not_import_rex_state(monkeypatch):
    import sys
    from rexgraph.model_state import freeze_tree

    # The semantic object carrier must be valid without reaching into a storage codec.
    monkeypatch.delitem(sys.modules, "rexgraph.io.rex_state", raising=False)
    out = freeze_tree(np.asarray([Fraction(1, 3), 2], dtype=object))
    assert out.tolist() == [Fraction(1, 3), 2]
    assert "rexgraph.io.rex_state" not in sys.modules
