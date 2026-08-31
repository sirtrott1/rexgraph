"""The container seal: what it answers, and the framing that makes the answer true.

A digest exists to say "these are the bytes that were written". If two different tensor
sets can produce one digest, it does not say that, and the failure is silent in exactly
the case it was installed to catch.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import (
    DIGEST_ALGO,
    RexState,
    from_state,
    state_digest,
    to_state,
    verify_state,
)


def test_unframed_concatenation_collides_and_framed_does_not():
    """The constructed collision. Writing name, dtype, shape and payload end to end
    leaves the field boundaries ambiguous, so a name can absorb the next tensor's header:
    {"a": zeros(0), "b": zeros(0)} and {"auint8(0,)b": zeros(0)} produce the same stream
    and therefore the same sha256."""
    two = {"a": np.zeros(0, np.uint8), "b": np.zeros(0, np.uint8)}
    one = {"auint8(0,)b": np.zeros(0, np.uint8)}
    assert state_digest(two, algo=1) == state_digest(one, algo=1), (
        "the legacy path must still reproduce the old stream, collision included")
    assert state_digest(two) != state_digest(one), "framing must separate the fields"


def test_the_name_is_still_bound_to_its_payload():
    """Framing must not cost what the seal already had: moving bytes between tensors has
    to change the digest."""
    a = {"x": np.array([1, 2, 3], np.uint8), "y": np.array([4], np.uint8)}
    b = {"x": np.array([1, 2], np.uint8), "y": np.array([3, 4], np.uint8)}
    assert state_digest(a) != state_digest(b)


def test_dtype_and_shape_are_still_covered():
    a = {"x": np.zeros(4, np.uint8)}
    b = {"x": np.zeros(4, np.int8)}
    c = {"x": np.zeros((2, 2), np.uint8)}
    assert len({state_digest(a), state_digest(b), state_digest(c)}) == 3


def test_insertion_order_still_does_not_matter():
    a = {"a": np.arange(3, dtype=np.uint8), "b": np.arange(2, dtype=np.uint8)}
    b = {"b": np.arange(2, dtype=np.uint8), "a": np.arange(3, dtype=np.uint8)}
    assert state_digest(a) == state_digest(b)


#### migration: an old bundle is old, not corrupt ###############################

@pytest.fixture
def rex():
    return RexGraph(sources=[0, 1, 2], targets=[1, 2, 0])


def test_a_fresh_state_stamps_the_framing_it_used(rex):
    st = to_state(rex)
    assert st.header["digest_algo"] == DIGEST_ALGO
    assert verify_state(st)


def test_a_bundle_written_before_the_fix_still_verifies(rex):
    """An unstamped header means algo 1. Checking it under the new rule would report
    every previously stored object as corrupt, which turns a fix into data loss."""
    st = to_state(rex)
    legacy = RexState(dict(st.tensors),
                      {k: v for k, v in st.header.items() if k != "digest_algo"})
    legacy.header["digest"] = state_digest(
        legacy.tensors, legacy.header["digest_names"], algo=1)
    assert verify_state(legacy)


def test_an_unsealed_state_is_not_successful_verification(rex):
    """The .rex loader owns legacy migration; the general verifier fails closed."""
    st = to_state(rex)
    unsealed = RexState(
        dict(st.tensors),
        {k: v for k, v in st.header.items()
         if k not in {"digest", "digest_names", "digest_algo"}},
    )

    assert not verify_state(unsealed)
    with pytest.raises(ValueError, match="no content digest"):
        from_state(unsealed)


def test_a_precanonical_bundle_reports_age_before_the_absent_seal(rex):
    """The unsealed migration flag cannot decode the old array naming scheme, so the
    reader must not recommend it for a bundle that predates canonical RexState."""
    st = to_state(rex)
    legacy_header = {
        k: v for k, v in st.header.items()
        if k not in {"format_version", "digest", "digest_names", "digest_algo"}
    }
    legacy_header["version"] = "0.9.0"

    with pytest.raises(ValueError, match="older version"):
        from_state(RexState(dict(st.tensors), legacy_header))
    with pytest.raises(ValueError, match="older version"):
        from_state(
            RexState(dict(st.tensors), legacy_header),
            _allow_unsealed=True,
        )


@pytest.mark.parametrize("field,value", [
    ("digest", ""),
    ("digest", 0),
    ("digest", "not-a-sha256"),
    ("digest_names", None),
    ("digest_names", ["boundary_idx", "boundary_idx"]),
    ("digest_algo", 999),
])
def test_malformed_digest_metadata_is_not_a_legacy_state(rex, field, value):
    st = to_state(rex)
    st.header[field] = value
    assert not verify_state(st)
    with pytest.raises(ValueError, match="do not match"):
        from_state(st, _allow_unsealed=True)


def test_tampering_is_still_caught_under_the_legacy_rule(rex):
    st = to_state(rex)
    legacy = RexState(dict(st.tensors),
                      {k: v for k, v in st.header.items() if k != "digest_algo"})
    legacy.header["digest"] = state_digest(
        legacy.tensors, legacy.header["digest_names"], algo=1)
    legacy.tensors["boundary_idx"] = np.asarray(
        legacy.tensors["boundary_idx"])[::-1].copy()
    assert not verify_state(legacy)


def test_a_dropped_tensor_is_a_failure_not_an_absence(rex):
    st = to_state(rex)
    st.tensors.pop("boundary_idx")
    assert not verify_state(st)


def test_a_state_round_trips_under_the_new_framing(rex):
    st = to_state(rex)
    back = from_state(st, verify=True)
    assert int(back.nE) == int(rex.nE)
    assert to_state(back).header["digest"] == st.header["digest"]


def test_a_rex_bundle_cannot_downgrade_integrity_by_deleting_the_seal(rex, tmp_path):
    """Reproduce the on-disk downgrade: altered tensors plus deleted declarations
    must be refused before reconstruction. A genuine unsealed legacy bundle can still
    be migrated only through the explicit opt-in."""
    import json

    from rexgraph.io import load_rex, save_rex

    path = tmp_path / "unsealed.rex"
    save_rex(str(path), rex)
    manifest_path = path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("digest", "digest_names", "digest_algo"):
        manifest.pop(name, None)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="no content digest"):
        load_rex(str(path))
    assert load_rex(str(path), allow_unsealed=True).nE == rex.nE

    boundary_path = path / "boundary_idx.npy"
    boundary = np.load(boundary_path)
    boundary[0] = (int(boundary[0]) + 1) % rex.nV
    np.save(boundary_path, boundary)
    with pytest.raises(ValueError, match="no content digest"):
        load_rex(str(path))
