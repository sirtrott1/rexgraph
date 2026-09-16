"""Native section state keeps the established digest framing and channel values."""
import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.io.catalog import object_digest
from rexgraph.io.rex_state import from_state, to_state
from rexgraph.merkle import _leaf_digests, _h, build_merkle
from rexgraph.partition import document_field, section_response
from rexgraph.sectioning import add_sectioning, sectionings_of


def fixture(g_channel="raw", loop=True):
    r = RexGraph.from_cells([4, [[0, 1, 2], [0, 0] if loop else [0, 2], [2], [1, 3]]], g_channel=g_channel)
    add_sectioning(r, "s", {"a": [0, 1], "b": [2, 3]}, spans={"a": (0, 7), "b": (7, 9)})
    return r, sectionings_of(r)["s"]


def test_section_state_roundtrip_and_merkle_are_native():
    r, s = fixture()
    expected = object_digest(r)
    loaded = from_state(to_state(r))
    assert object_digest(loaded) == expected
    assert build_merkle(loaded).root == build_merkle(r).root
    a = section_response(r, s, [0, 2], exact=True)
    b = section_response(loaded, sectionings_of(loaded)["s"], [0, 2], exact=True)
    np.testing.assert_array_equal(a[0], b[0])


def test_native_leaf_bytes_match_the_existing_scipy_encoding_oracle():
    pytest.importorskip("scipy.sparse")
    from rexgraph.core._sparse import to_scipy_csr
    r, s = fixture()
    old = to_scipy_csr(r._B1_dual).tocsc()
    expected = []
    for i in range(len(s)):
        parts = [s.labels[i].encode("utf-8"), np.asarray(s.spans[i], dtype=np.int64).tobytes()]
        for c in sorted(s.cells(i)):
            lo, hi = old.indptr[c:c+2]
            parts.extend([old.indices[lo:hi].astype(np.int64).tobytes(),
                          np.ascontiguousarray(old.data[lo:hi], dtype=np.float64).tobytes()])
        expected.append(_h(*parts))
    assert _leaf_digests(r, s) == expected


@pytest.mark.parametrize("g_channel", ["raw", "normalized"])
def test_native_section_profiles_match_the_existing_character_oracle(g_channel):
    pytest.importorskip("scipy.sparse")
    r, s = fixture(g_channel, loop=False)
    field = document_field(r, [0, 1], exact=False).numpy()
    expected = np.vstack([sum((field[i]*r.structural_character[i] for i in s.cells(j)),
                             np.zeros(4)) for j in range(len(s))])
    got, _, names = section_response(r, s, [0, 1], channels=True)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    assert len(names) == 4


def test_normalized_profiles_keep_core_repeated_participant_refusal():
    r, s = fixture("normalized")
    with pytest.raises(ValueError, match="distinct participants"):
        section_response(r, s, [0], channels=True)
