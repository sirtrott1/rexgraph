"""Accession ratios use integer support and round only after grouping."""
from fractions import Fraction

import numpy as np
import pytest

from rcdb import index as ix
from rcdb.core import ComplexRecord


def _index(widths):
    """Several accession relations owned by one record, plus an isolated record."""
    ptr, idx = [0], []
    for width in widths:
        idx.extend([0, *range(2, width + 2)])
        ptr.append(len(idx))
    n_terms = max(widths, default=0)
    return dict(n=2, nV=2 + n_terms, ids=["a", "b"],
                vocab=[f"t{i}" for i in range(n_terms)],
                rel_ptr=np.array(ptr, dtype=np.int64),
                rel_idx=np.array(idx, dtype=np.int64))


def _reference(index, terms, reading):
    supports = [index["rel_idx"][a:b].tolist()
                for a, b in zip(index["rel_ptr"][:-1], index["rel_ptr"][1:], strict=False)]
    seeds = {index["n"] + i for i, term in enumerate(index["vocab"]) if term in terms}
    out = [Fraction(0) for _ in range(index["n"])]
    for support in supports:
        for seed in seeds.intersection(support[1:]):
            degree = sum(seed in other for other in supports)
            den = len(support) - 1 if reading == "share" else 1
            out[support[0]] += Fraction(1, degree * den)
    return out


@pytest.mark.parametrize("reading", ["share", "existence"])
@pytest.mark.parametrize("fallback", [False, True])
def test_record_width_product_cannot_overflow(reading, fallback, monkeypatch):
    index = _index([2] * 64 + [3, 5, 7])
    if fallback:
        monkeypatch.setattr(ix, "_exact_ratio", None)
    monkeypatch.setattr(ix, "boundary_operator", lambda *_: pytest.fail("matrix requested"))
    terms = ["t0", "t1", "t4"]
    expected = _reference(index, terms, reading)
    exact = ix.record_response_exact(index, terms, reading=reading)
    scores, ids = ix.record_response(index, terms, reading=reading)
    assert exact == {r: value for r, value in enumerate(expected) if value}
    np.testing.assert_array_equal(scores, [float(value) for value in expected])
    assert ids is index["ids"]
    assert "_den" not in index and "_B1_csc" not in index
    cached = index["_accession_v2e"]
    ix.record_response(index, terms, reading=reading)
    assert index["_accession_v2e"] is cached


@pytest.mark.parametrize("reading", ["share", "existence"])
def test_built_and_reopened_index_agree(tmp_path, reading):
    records = [(str(i), ComplexRecord(
        id=str(i), created=0.0, signature={"tags": ["a", f"b{i}"]},
        meta={"vertex_labels": ["a", "a", "c"]})) for i in range(4)]
    index = ix.build(records)
    path = tmp_path / "index.safetensors"
    ix.write(path, index)
    reopened = ix.read(path)
    expected = _reference(index, ["a", "b1"], reading)
    for current in (index, reopened):
        scores, _ = ix.record_response(current, ["A", "a", "b1"], reading=reading)
        np.testing.assert_array_equal(scores, list(map(float, expected)))
        assert ix.record_response_exact(current, ["a", "b1"], reading=reading) == {
            r: value for r, value in enumerate(expected) if value}


@pytest.mark.parametrize("terms", [[], ["absent"], ["t0"]])
@pytest.mark.parametrize("reader", [ix.record_response, ix.record_response_exact])
def test_unknown_reading_is_refused_even_without_seeds(terms, reader):
    with pytest.raises(ValueError, match="reading must"):
        reader(_index([2]), terms, reading="unknown")


@pytest.mark.parametrize("reader", [ix.record_response, ix.record_response_exact])
def test_empty_corpus_and_absent_seeds(reader):
    result = reader(_index([]), ["absent"])
    if reader is ix.record_response:
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
    else:
        assert result == {}


@pytest.mark.parametrize("reading", ["share", "existence"])
def test_seed_permutation_preserves_final_rounding(reading):
    index = _index([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    terms = [f"t{i}" for i in range(47)]
    expected = _reference(index, terms, reading)
    for query in (terms, list(reversed(terms)), terms + terms):
        scores, _ = ix.record_response(index, query, reading=reading)
        np.testing.assert_array_equal(scores, list(map(float, expected)))


@pytest.mark.parametrize("field,value", [
    ("rel_idx", [0, 2, 2**32 + 3]), ("rel_idx", [0, 2, -1]),
    ("rel_idx", [0.0, 2.0, 3.0]), ("rel_ptr", [0, 2**32 + 3]),
    ("rel_ptr", [0, 0, 3]), ("rel_idx", [2, 2, 3]),
])
def test_malformed_accession_is_not_narrowed_into_valid_data(field, value):
    index = _index([2])
    index[field] = np.array(value)
    with pytest.raises(ValueError):
        ix.record_response(index, ["t0"])
