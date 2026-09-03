"""A structural channel is selected by name, never by position.

The character channels are ordered ``['L1_down', 'L_O', 'L_SG', 'L_C']``: topology,
geometry, frustration, coparticipation. Reading frustration from index 0 and
coparticipation from index 1 takes topology and geometry instead, and those two share a
diagonal because the diagonal squares each incidence entry and squaring kills the sign.
That identity is not incidental; it is the reason the F channel exists to carry the
signed/unsigned mismatch that the diagonal cannot.

The consequence is a metric that cannot fail. ``health_ratio`` computed from indices 0 and
1 is identically 1.0 on every complex, so it reported perfect health for two years'
worth of structures without ever being able to say anything else. rexgraph's
``mesh_health.harmonic_health`` resolves by name and documents the trap; the pipeline
carried a stale positional copy of the same computation.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from rexgraph.graph import RexGraph
from rexgraph.mesh_health import harmonic_health


def _chi(rex):
    return (np.asarray(rex.structural_character)
            * np.asarray(rex._rl4_sparse.diagonal())[:, None])


def test_topology_and_geometry_share_a_diagonal_so_their_ratio_is_inert():
    """The identity that makes positional selection silently useless."""
    complexes = [
        RexGraph.from_graph(sources=[0, 1, 2], targets=[1, 2, 0]),
        RexGraph.from_graph(sources=[0, 1, 2, 0, 3, 4], targets=[1, 2, 0, 3, 4, 0]),
        RexGraph.from_hypergraph(np.array([0, 3, 5], dtype=np.int64),
                                 np.array([0, 1, 2, 2, 3], dtype=np.int64)),
    ]
    for rex in complexes:
        chi = _chi(rex)
        names = list(rex.hat_names)
        assert np.array_equal(chi[:, names.index("L1_down")], chi[:, names.index("L_O")]), (
            "diag(G) must equal diag(T); if this ever fails the reasoning below changes"
        )


def test_the_frustration_and_coparticipation_channels_do_not_share_a_diagonal():
    """The correct pair carries information the positional pair cannot."""
    rex = RexGraph.from_graph(sources=[0, 1, 2, 0, 3, 4], targets=[1, 2, 0, 3, 4, 0])
    chi = _chi(rex)
    names = list(rex.hat_names)
    assert not np.array_equal(chi[:, names.index("L_SG")], chi[:, names.index("L_C")])


def test_health_ratio_is_not_identically_one():
    """A metric that cannot move is not measuring anything.

    A 4-cycle with chords gives 1.125 by name and exactly 1.0 positionally, so this fails
    for any implementation that reads indices 0 and 1.
    """
    rex = RexGraph.from_graph(sources=[0, 1, 2, 3, 0, 1], targets=[1, 2, 3, 0, 2, 3])
    health = harmonic_health(rex, np.ones(int(rex.nE)))
    assert health["health_ratio"] is not None
    assert health["health_ratio"] != pytest.approx(1.0), (
        "health_ratio did not move; the channels were probably selected positionally"
    )

    chi = _chi(rex)
    positional = float(chi[:, 0].sum() / chi[:, 1].sum())
    assert positional == pytest.approx(1.0), "the positional reading is inert by construction"


def test_the_pipeline_does_not_select_a_channel_by_index():
    """No call site may reintroduce positional selection.

    Laying all four channels out in order, as the training feature vector does, is fine.
    Selecting one by a bare index is what goes wrong, because the index carries no claim
    about which channel it is.
    """
    source = (pathlib.Path(__file__).resolve().parents[1] / "agent" / "pipeline.py").read_text()
    # code only: the comment above the fix quotes the bad form on purpose, to say what it
    # was and why it was wrong, and a guard that cannot tell prose from code would forbid
    # explaining the very thing it guards against
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    offenders = re.findall(r"chi\[:, ?\d\]", code)
    assert not offenders, (
        f"pipeline.py selects a character channel positionally: {offenders}; "
        "resolve it through hat_names instead"
    )
    assert "hat_names" in source, "the pipeline should resolve channels by name"
