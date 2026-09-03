"""A record set becomes relations at their observed arity, never pairwise projections.

One link value observed across k records is one k-ary relation among those records. The
records are its derived grade-zero participants. Emitting k-1 pairwise facts instead
asserts relations the data never contained and erases the arity from the boundary.

Two readings come out of this and they are not interchangeable. The support projection
asks which records are joined by a chain of shared values; H0 asks the algebraic question
about the complex. Four records sharing one value give one support component and a
grade-zero Betti number of 3, and both are correct answers to different questions.
"""

from __future__ import annotations

from agent.data_complex import analyze_rows, rows_to_complex


def _shared(n: int, value: str = "x"):
    """n records that all carry the same link value: one n-ary relation."""
    return [{"id": chr(ord("a") + i), "k": value} for i in range(n)]


def test_a_shared_link_value_is_one_relation_at_its_observed_arity():
    rows = _shared(4)
    rex, meta = rows_to_complex(rows, link_on="k", id_col="id")

    assert int(rex.nE) == 1, "four records sharing a value are ONE relation"
    assert int(rex.nV) == 4
    assert meta["relation_arities"] == [4]


def test_no_pairwise_expansion_of_a_shared_group():
    """The star this replaced would give k-1 relations of arity two."""
    for k in (3, 4, 5, 6):
        rex, meta = rows_to_complex(_shared(k), link_on="k", id_col="id")
        assert int(rex.nE) == 1, f"arity {k} group produced {int(rex.nE)} relations"
        assert meta["relation_arities"] == [k]
        assert 2 not in meta["relation_arities"] or k == 2


def test_the_composite_share_is_preserved_exactly():
    """Read from the composite binary, which is where the exact share lives.

    The structure is binary masks and the share is a Fraction derived from the arity, so
    the exact claim is checked exactly: equality against Fraction(1, k-1), never a
    tolerance against a rendered coefficient.
    """
    from fractions import Fraction

    from rexgraph import Cell
    from rexgraph.cells import composite_binary

    for k in (2, 3, 4, 5):
        rex, _ = rows_to_complex(_shared(k), link_on="k", id_col="id")
        cb = composite_binary(Cell(rex, 1, 0))

        assert cb.arity == k
        assert not cb.witness

        # the structure is binary: existence, head and share support are 0/1 masks
        assert set(cb.existence.values.tolist()) <= {0, 1}
        assert sum(int(x) for x in cb.head.values) == 1
        assert cb.share_support.values.tolist() == [
            0 if int(h) else 1 for h in cb.head.values
        ]

        # the share is exact and derived from the arity, never a float
        shares = [v for v in cb.share.values if v != 0]
        assert all(isinstance(v, Fraction) for v in shares)
        assert shares == [Fraction(1, k - 1)] * (k - 1), (
            f"arity {k}: shares were {shares}, expected {Fraction(1, k - 1)}"
        )

        boundary = list(cb.boundary.values)
        assert boundary.count(Fraction(-1)) == 1
        assert sum(boundary) == 0, "the exact column sums to zero at every arity"

        # and the integer representative is (r-1)b = x - r*h, exactly
        assert list(cb.integer_boundary.values) == [
            -(k - 1) if int(h) else 1 for h in cb.head.values
        ]


def test_support_components_and_h0_are_different_readings():
    """The number art specified: one support component, H0 of 3, both valid."""
    out = analyze_rows(_shared(4), link_on="k", id_col="id")

    assert out["n_support_components"] == 1
    assert out["h0_dimension"] == 3
    assert out["n_clusters"] == out["n_support_components"], "the alias tracks the projection"


def test_the_head_is_deterministic_and_carries_no_claim():
    """Reproducible across builds and across row order, and only canonical.

    The head is the participant at -1. A record source that declares no direction gives no
    basis to choose one, so the construction picks the lowest row index to be reproducible
    and marks the orientation canonical. Nothing may read it as an ordering claim.
    """
    rows = _shared(4)
    first, meta = rows_to_complex(rows, link_on="k", id_col="id")
    second, _ = rows_to_complex(rows, link_on="k", id_col="id")

    from rexgraph import Cell
    from rexgraph.cells import composite_binary

    # the head is the participant the composite binary marks, not an argmin over a float
    # rendering: the mask is the declaration, the float column is a picture of it
    def head_of(rex):
        mask = composite_binary(Cell(rex, 1, 0)).head.values
        return next(i for i, h in enumerate(mask) if int(h))

    assert head_of(first) == head_of(second), "the same input must give the same head"

    # and the head is the first participant listed, which is what makes "lowest row index"
    # a true statement about the construction rather than an assumption about the carrier.
    # from_cells reconstructs canonical C1 from the support it is given, so the order this
    # code passes is the order that decides the head; it does not accept explicit signs.
    from rexgraph.graph import RexGraph

    for members in ([0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]):
        built = RexGraph.from_cells([4, [list(members)]])
        mask = composite_binary(Cell(built, 1, 0)).head.values
        assert next(i for i, h in enumerate(mask) if int(h)) == members[0], (
            f"head should be the first listed participant, got support order {members}"
        )
    assert meta["head_is_canonical"] is True, (
        "the orientation is under-determined by the source and must say so"
    )


def test_an_isolated_record_is_a_participant_with_no_relation():
    """Linking to nothing is not the same as being observed standalone.

    A record that shares no link value is a grade-zero participant that belongs to no
    relation. Giving it an arity-one relation of its own would assert an observation the
    source never made.
    """
    rows = [{"id": "a", "k": "x"}, {"id": "b", "k": "x"}, {"id": "c", "k": "y"}]
    rex, meta = rows_to_complex(rows, link_on="k", id_col="id")

    assert int(rex.nV) == 3, "every record is a declared participant"
    assert int(rex.nE) == 1, "only the shared value is a relation"
    assert meta["unattached_participants"] == ["c"]

    from rexgraph import Cell
    from rexgraph.cells import composite_binary

    participants = composite_binary(Cell(rex, 1, 0)).existence.values
    assert int(participants[2]) == 0, "record c participates in no relation"


def test_no_column_is_manufactured_that_breaks_the_zero_sum_law():
    """An arity-one column carries a single +1 and does not sum to zero.

    Inventing one per unlinked record would break the zero-sum law across the complex in
    order to represent an absence. Leaving the participant unattached keeps every column a
    boundary and still counts the record in H0.
    """
    rows = [{"id": "a", "k": "x"}, {"id": "b", "k": "x"},
            {"id": "c", "k": "y"}, {"id": "d", "k": "z"}]
    rex, _ = rows_to_complex(rows, link_on="k", id_col="id")

    from rexgraph import Cell
    from rexgraph.cells import composite_binary

    # exact, not rounded: the zero-sum law is the definition of a boundary, so checking it
    # through a float rendering would accept a column that only nearly sums to zero
    sums = [sum(composite_binary(Cell(rex, 1, j)).boundary.values)
            for j in range(int(rex.nE))]
    assert sums == [0], f"expected every column to be a boundary, got {sums}"

    assert int(rex.nV) == 4, "the complex stays faithful to the whole record set"
    assert int(rex.betti[0]) == 3, "two unattached participants plus the linked pair"


def test_the_complex_never_drops_a_record():
    """The earlier construction omitted any record that shared no link value."""
    rows = [{"id": "a", "k": "x"}, {"id": "b", "k": "x"}] + \
           [{"id": c, "k": c} for c in "cdef"]
    rex, meta = rows_to_complex(rows, link_on="k", id_col="id")

    assert int(rex.nV) == len(rows) == 6
    assert meta["vertex_labels"] == ["a", "b", "c", "d", "e", "f"]
