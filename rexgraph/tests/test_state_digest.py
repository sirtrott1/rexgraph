"""The container seal: what it answers, and the framing that makes the answer true.

A digest exists to say "these are the bytes that were written". If two different tensor
sets can produce one digest, it does not say that, and the failure is silent in exactly
the case it was installed to catch.
"""
from __future__ import annotations

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.io.rex_state import (DIGEST_ALGO, RexState, from_state, state_digest,
                                   to_state, verify_state)


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
