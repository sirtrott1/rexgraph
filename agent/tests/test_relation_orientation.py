"""A relation's head survives every conversion, because sort order is not where it lives.

The head is the participant carrying the -1 coefficient in the boundary column, and the
composite binary puts it first in the stored support. Rebuilding membership with
``np.nonzero`` returns rows in ascending vertex order, which silently reorients any
relation whose head is not its lowest-numbered participant.
"""

from __future__ import annotations

import numpy as np

from rexgraph.graph import RexGraph


def _rex():
    # relation 0 is [0,1,2]; relation 1 is [3,0], whose head 3 is NOT the lower index
    return RexGraph.from_hypergraph(np.array([0, 3, 5], dtype=np.int64),
                                    np.array([0, 1, 2, 3, 0], dtype=np.int64))


def test_the_head_is_declared_not_derived_from_vertex_order():
    """The head comes from the composite binary's head mask, where it is declared."""
    from rexgraph import Cell
    from rexgraph.cells import composite_binary

    rex = _rex()
    cb = composite_binary(Cell(rex, 1, 1))
    head = next(i for i, h in enumerate(cb.head.values) if int(h))
    assert head == 3, "the declared head of relation [3, 0] is vertex 3"
    assert cb.arity == 2

    # The trap, shown on the stored support rather than on a rendering of it: any rebuild
    # that sorts the participants names the lowest index as head, which is a different
    # participant from the declared one whenever the head is not the lowest.
    ptr = np.asarray(rex.boundary_ptr, dtype=np.int64)
    idx = np.asarray(rex.boundary_idx, dtype=np.int64)
    declared = idx[ptr[1]:ptr[2]].tolist()
    assert declared[0] == head, "the stored support puts the declared head first"
    assert sorted(declared)[0] != head, (
        "sorting names vertex 0 as head, which is why the stored order is what is read"
    )


def test_the_bundle_preserves_declared_participant_order():
    from agent.models.store import _bundle_from_rex

    rex = _rex()
    bundle = _bundle_from_rex(rex)
    ptr = np.asarray(bundle.extra["he_ptr"]).tolist()
    idx = np.asarray(bundle.extra["he_idx"]).tolist()

    assert ptr == np.asarray(rex.boundary_ptr).tolist()
    assert idx == np.asarray(rex.boundary_idx).tolist()

    members_of_relation_1 = idx[ptr[1]:ptr[2]]
    assert members_of_relation_1[0] == 3, (
        f"head reversed: relation [3, 0] came back as {members_of_relation_1}"
    )
