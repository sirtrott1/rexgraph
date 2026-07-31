"""Mutation: existence changing into another structure, not two unrelated events.

Findings from a review of the temporal kernels. Each test names the defect it pins.

The through-line is that the kernels tracked BIRTH and DEATH well and PERSISTENCE
and MUTATION poorly. A cell that dies as another is born is a topology mutating, and
reading it as two independent events loses exactly the thing worth knowing; a cell
that flickers off and back is not one continuous life; a face that deforms has not
merged with anything.
"""

import numpy as np
import pytest

from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.core import _temporal

B, I, O, E, S = 0, 1, 2, 3, 4
PERSIST, BORN, DIED, SPLIT, MERGE, MUTATE = 0, 1, 2, 3, 4, 5


# --- BIOES is blind to edge-level mutation ------------------------------------

def _pendant_moves():
    """Triangle 0-1-2 plus one pendant to vertex 3, which moves at t=2.
    beta0 and beta1 both hold, so only the churn reveals that anything happened."""
    tr = TemporalRex([])
    for p in (0, 0, 1, 1):
        tr.append_snapshot(RexGraph(sources=np.array([0, 1, 2, p], np.int32),
                                    targets=np.array([1, 2, 0, 3], np.int32)))
    return tr


def test_the_fixture_is_genuinely_betti_preserving():
    tr = _pendant_moves()
    betti = [tuple(tr.reconstruct_at(t).betti[:2]) for t in range(tr.T)]
    assert len(set(betti)) == 1, betti


def test_a_betti_preserving_edge_mutation_breaks_a_phase():
    """compute_bioes_unified measured edge_born/edge_died and returned them, but
    passed only FACE events to the phase detector, so an edge dying as another was
    born produced one unbroken phase."""
    tr = _pendant_moves()
    betti = np.array([list(tr.reconstruct_at(t).betti[:2]) for t in range(tr.T)],
                     np.int64)
    tags = list(tr.bioes(betti)[0])
    assert tags != [B, I, I, E], "the mutation at t=2 still leaves one phase"
    assert tags[2] in (B, S), f"t=2 should open a phase, got {tags}"


def test_edge_events_reach_the_phase_detector():
    counts = np.zeros(4, np.int32)
    edge_born = np.array([0, 0, 1, 0], np.int32)
    edge_died = np.array([0, 0, 1, 0], np.int32)
    betti = np.ones((4, 2), np.int64)
    ps, pe, _, reasons = _temporal.detect_phases_with_events(
        betti, counts, counts, counts, counts,
        edge_born=edge_born, edge_died=edge_died)
    assert len(ps) > 1, "a pure edge mutation did not break a phase"


def test_a_quiet_history_is_still_one_phase():
    """The complement: nothing happening must not be split up."""
    z = np.zeros(4, np.int32)
    betti = np.ones((4, 2), np.int64)
    ps, _, _, _ = _temporal.detect_phases_with_events(betti, z, z, z, z,
                                                      edge_born=z, edge_died=z)
    assert len(ps) == 1


# --- face correspondence ------------------------------------------------------

_SRC = np.array([0, 1, 2, 0, 3, 4], np.int32)
_TGT = np.array([1, 2, 0, 3, 4, 0], np.int32)


def _track(prev_cols, curr_cols, min_shared=1):
    def b2(cols):
        cp, ri = [0], []
        for c in cols:
            ri.extend(c)
            cp.append(len(ri))
        return np.array(cp, np.int32), np.array(ri, np.int32)
    pcp, pri = b2(prev_cols)
    ccp, cri = b2(curr_cols)
    return _temporal.track_faces_i32(pcp, pri, _SRC, _TGT, ccp, cri, _SRC, _TGT,
                                     False, min_shared)


def test_a_deforming_face_is_a_mutation_not_a_merge():
    """One face partially matching one successor had nothing to merge with. It was
    reported MERGE, and its predecessor reported DIED."""
    ev_prev, ev_curr, p2c, c2p, _ = _track([[0, 1, 2]], [[0, 1, 3]])
    assert ev_curr[0] == MUTATE, f"curr event was {ev_curr[0]}"
    assert ev_prev[0] == MUTATE, f"prev event was {ev_prev[0]}"


def test_a_jaccard_match_records_the_lineage_both_ways():
    """prev_to_curr was only ever written by the exact-match pass, so every
    approximate correspondence left the predecessor looking dead."""
    _, _, p2c, c2p, _ = _track([[0, 1, 2]], [[0, 1, 3]])
    assert c2p[0] == 0
    assert p2c[0] == 0, "prev_to_curr lost the correspondence"


def test_a_genuine_merge_is_reported_on_both_sides():
    """Two faces merging into one reported both parents DIED, so a merge could not
    be told from an annihilation."""
    ev_prev, ev_curr, p2c, c2p, _ = _track([[0, 1, 2], [3, 4, 5]],
                                           [[0, 1, 2, 3, 4, 5]])
    assert ev_curr[0] == MERGE
    assert list(ev_prev) == [MERGE, MERGE], f"parents reported {list(ev_prev)}"
    assert p2c[0] == 0 and p2c[1] == 0


def test_a_split_is_unchanged():
    ev_prev, ev_curr, _, _, _ = _track([[0, 1, 2, 3, 4, 5]], [[0, 1, 2], [3, 4, 5]])
    assert ev_prev[0] == SPLIT
    assert list(ev_curr) == [SPLIT, SPLIT]


def test_an_exact_match_is_still_persist():
    ev_prev, ev_curr, p2c, c2p, shared = _track([[0, 1, 2]], [[0, 1, 2]])
    assert ev_prev[0] == PERSIST and ev_curr[0] == PERSIST
    assert p2c[0] == 0 and c2p[0] == 0
    assert int(shared[0]) == 3, "an exact match shares its whole boundary"


def test_an_unrelated_face_is_still_born_and_died():
    ev_prev, ev_curr, p2c, c2p, _ = _track([[0, 1, 2]], [[3, 4, 5]])
    assert ev_prev[0] == DIED and ev_curr[0] == BORN
    assert p2c[0] == -1 and c2p[0] == -1


# --- persistence vs intermittency ---------------------------------------------

def test_an_edge_that_flickers_is_not_one_continuous_life():
    """edge_lifecycle reports first_seen and last_seen, so an edge present at t=0,
    absent at t=1 and back at t=2 is indistinguishable from one that never left."""
    snaps = [(np.array([0, 2], np.int32), np.array([1, 3], np.int32)),
             (np.array([2], np.int32), np.array([3], np.int32)),
             (np.array([0, 2], np.int32), np.array([1, 3], np.int32))]
    keys, starts, ends = _temporal.edge_intervals(snaps)
    flick = [(int(s), int(e)) for k, s, e in zip(keys, starts, ends) if k == 1]
    assert flick == [(0, 0), (2, 2)], f"expected two intervals, got {flick}"


def test_a_continuously_present_edge_has_one_interval():
    snaps = [(np.array([0], np.int32), np.array([1], np.int32))] * 3
    keys, starts, ends = _temporal.edge_intervals(snaps)
    assert list(starts) == [0] and list(ends) == [2]


def test_intervals_agree_with_lifecycle_when_nothing_flickers():
    snaps = [(np.array([0], np.int32), np.array([1], np.int32)),
             (np.array([0, 2], np.int32), np.array([1, 3], np.int32))]
    keys_i, starts, ends = _temporal.edge_intervals(snaps)
    keys_l, birth, death = _temporal.edge_lifecycle(snaps)
    assert list(keys_i) == list(keys_l)
    assert list(starts) == list(birth)


# --- correspondence without a similarity score --------------------------------

def test_a_mutating_face_still_counts_as_a_structural_event():
    """Regression in this file's own first pass: introducing FACE_MUTATE gave
    deforming faces an event code that no counter incremented, so they stopped
    contributing to the phase detector entirely."""
    fs = [(np.array([0, 3], np.int32), np.array([0, 1, 2], np.int32)),
          (np.array([0, 3], np.int32), np.array([0, 1, 3], np.int32))]
    es = [(_SRC, _TGT), (_SRC, _TGT)]
    out = _temporal.face_lifecycle(fs, es, False)
    counts = {"persist": out[2], "born": out[3], "died": out[4],
              "split": out[5], "merge": out[6], "mutate": out[7]}
    total = sum(int(c[1]) for c in counts.values())
    assert total > 0, f"the mutation was counted nowhere: {counts}"
    assert int(counts["mutate"][1]) == 1


def test_correspondence_uses_shared_boundary_not_a_similarity_score():
    """Face identity is already exact -- B2 says which cells bound each face, and
    cell keys are canonical. Estimating it with a set-similarity score and a 0.5
    cutoff re-derives, badly, something the complex knows exactly."""
    ev_prev, ev_curr, p2c, c2p, shared = _track([[0, 1, 2]], [[0, 1, 3]])
    assert list(shared) == [2], "shared boundary cells not reported as a count"


def test_shared_boundary_gives_the_magnitude_of_a_mutation():
    """A face that loses one of five boundary cells is barely changed; one that
    loses four is nearly a death. That is a count, reported for the caller's policy,
    not a threshold applied inside the kernel."""
    _, _, _, _, small = _track([[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 5]])
    _, _, _, _, large = _track([[0, 1, 2, 3, 4]], [[0, 1, 5]])
    assert int(small[0]) > int(large[0])


def test_any_shared_boundary_is_a_correspondence():
    """One shared boundary cell is an exact structural fact, not a tuned cutoff.
    Under a 0.5 jaccard these faces were unrelated."""
    ev_prev, ev_curr, p2c, c2p, shared = _track([[0, 1, 2]], [[2, 3, 4]])
    assert int(shared[0]) == 1
    assert ev_curr[0] == MUTATE and ev_prev[0] == MUTATE


def test_no_shared_boundary_is_still_born_and_died():
    ev_prev, ev_curr, p2c, c2p, shared = _track([[0, 1, 2]], [[3, 4, 5]])
    assert ev_prev[0] == DIED and ev_curr[0] == BORN
    assert int(shared[0]) == 0


def test_orientation_is_not_invisible_to_correspondence():
    """A set-similarity score cannot see orientation at all: a face and its reverse
    score identically. Sharing the same boundary cells is still the right
    correspondence, but the shared count must come from the actual boundary."""
    ev_prev, ev_curr, _, _, shared = _track([[0, 1, 2]], [[0, 1, 2]])
    assert ev_curr[0] == PERSIST and int(shared[0]) == 3
