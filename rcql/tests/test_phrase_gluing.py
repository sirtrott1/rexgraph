"""Phrase stalk correspondences are primary relations, not inferred joins."""
from fractions import Fraction

import pytest

from rcql import (
    PhraseCorrespondence,
    PhraseGlueResult,
    PhraseMapError,
    PhraseSheaf,
    PhraseStalk,
    SourcePolicy,
    SourceRef,
    UndeclaredRestrictionError,
    bind,
)


def state(name, version, permissions=("read",)):
    return PhraseStalk(
        name,
        bind(
            f"db/record@{version}", object(), SourcePolicy.allow(*permissions),
            source_ref=SourceRef(
                name=f"db/record@{version}", state_digest=f"state-{version}",
                record_id="record", record_version=version,
            ),
        ),
    )


def version_pair():
    return PhraseSheaf(
        (state("v1", 1), state("v3", 3)),
        (PhraseCorrespondence("record-through-time", ("v1", "v3")),),
    )


def test_cross_state_phrase_glue_refuses_to_inherit_identity_without_a_map():
    phrase = version_pair()
    phrase.assign("v1", [Fraction(7)])
    phrase.assign("v3", [Fraction(7)])

    with pytest.raises(UndeclaredRestrictionError) as error:
        phrase.glue()

    assert phrase.missing_restrictions() == (
        ("v1", "record-through-time"), ("v3", "record-through-time"),
    )
    assert "(v1, record-through-time)" in str(error.value)


def test_an_explicit_unit_correspondence_glues_two_selected_states_exactly():
    phrase = version_pair()
    phrase.assign("v1", [Fraction(7)])
    phrase.assign("v3", [Fraction(14)])
    phrase.restrict("v1", "record-through-time", [[2]])
    phrase.identity("v3", "record-through-time")

    result = phrase.glue()

    assert result.glued == result.gluable == 1
    assert result.ratio == Fraction(1)
    assert result.obstructions == ()


def test_correspondence_failure_is_an_exact_localized_strain_not_a_join_miss():
    phrase = version_pair()
    phrase.assign("v1", [Fraction(7)])
    phrase.assign("v3", [Fraction(9)])
    phrase.restrict("v1", "record-through-time", [[2]])
    phrase.identity("v3", "record-through-time")

    result = phrase.glue()

    assert result.obstruction_count == 1
    obstruction = result.obstructions[0]
    assert obstruction.left == (Fraction(14),)
    assert obstruction.right == (Fraction(9),)
    assert obstruction.residual == (Fraction(5),)
    assert obstruction.mediator == 0
    assert isinstance(result, PhraseGlueResult)
    named = result.named_obstructions[0]
    assert (named.left_stalk, named.right_stalk, named.correspondence) == (
        "v1", "v3", "record-through-time",
    )
    assert named.left == (Fraction(14),)
    assert named.right == (Fraction(9),)
    assert named.residual == (Fraction(5),)


def test_a_witness_is_not_accepted_as_a_phrase_correspondence():
    with pytest.raises(PhraseMapError, match="at least two stalks"):
        PhraseCorrespondence("not-between", ("v1",))


def test_wrong_phrase_incidence_uses_the_phrase_error_vocabulary():
    phrase = PhraseSheaf(
        (state("v1", 1), state("v2", 2), state("v3", 3)),
        (
            PhraseCorrespondence("v1-v2", ("v1", "v2")),
            PhraseCorrespondence("v2-v3", ("v2", "v3")),
        ),
    )

    with pytest.raises(PhraseMapError, match="not incident"):
        phrase.restrict("v3", "v1-v2", [[1]])


def test_three_state_correspondence_is_one_oriented_branching_relation():
    phrase = PhraseSheaf(
        (state("baseline", 1), state("observed", 2), state("simulated", 3)),
        (PhraseCorrespondence("same-record", ("baseline", "observed", "simulated")),),
    )
    for name in ("baseline", "observed", "simulated"):
        phrase.assign(name, [Fraction(11)])
        phrase.identity(name, "same-record")

    composite = phrase.composite("same-record")
    result = phrase.glue()

    assert phrase.rex.nE == 1
    assert composite.arity == 3
    assert composite.head.values.tolist() == [1, 0, 0]
    assert composite.share.values.tolist() == [Fraction(0), Fraction(1, 2), Fraction(1, 2)]
    assert result.glued == result.gluable == 3


def test_existing_glue_surface_executes_a_strict_phrase_sheaf():
    from rcql import Executor, parse

    phrase = version_pair()
    phrase.assign("v1", [Fraction(7)])
    phrase.assign("v3", [Fraction(14)])
    phrase.restrict("v1", "record-through-time", [[2]])
    phrase.identity("v3", "record-through-time")

    result = Executor(
        sources={"correspondence": phrase.rex}, params={"section": phrase},
    ).execute(parse("FROM $correspondence RETURN GLUE($section)"))

    assert result.values[0].ratio == Fraction(1)


def test_phrase_source_keeps_all_stalk_provenance_and_executes_glue():
    from rcql import Executor, parse

    phrase = version_pair()
    phrase.assign("v1", [Fraction(7)])
    phrase.assign("v3", [Fraction(14)])
    phrase.restrict("v1", "record-through-time", [[2]])
    phrase.identity("v3", "record-through-time")

    executor = Executor(params={"section": phrase})
    reading = executor.execute(
        parse("FROM PHRASE($section) RETURN GLUE($section)")
    ).values[0]
    explained = executor.execute(
        parse("EXPLAIN FROM PHRASE($section) RETURN GLUE($section)")
    ).values[0]

    assert reading.ratio == Fraction(1)
    assert reading.policy == phrase.policy
    assert [ref.name for ref in reading.contributors] == ["db/record@1", "db/record@3"]
    source_state = explained["source_state"]
    assert source_state["state_digest"] == phrase.source_ref.state_digest
    assert [item["name"] for item in source_state["contributors"]] == [
        "db/record@1", "db/record@3",
    ]


def test_phrase_result_requires_read_from_every_contributing_stalk():
    from rcql import Executor, parse

    phrase = PhraseSheaf(
        (state("readable", 1), state("restricted", 2, permissions=())),
        (PhraseCorrespondence("same-record", ("readable", "restricted")),),
    )

    assert not phrase.policy.permits("read")
    with pytest.raises(PermissionError, match="read"):
        Executor(params={"section": phrase}).execute(
            parse("FROM PHRASE($section) RETURN GLUE($section)")
        )


def test_phrase_result_policy_is_the_intersection_not_a_permission_union():
    phrase = PhraseSheaf(
        (
            state("identity-state", 1, permissions=("read", "identity")),
            state("stats-state", 2, permissions=("read", "stats")),
        ),
        (PhraseCorrespondence("same-record", ("identity-state", "stats-state")),),
    )

    assert phrase.policy.permissions == frozenset({"read"})
    assert phrase.policy.permits("read")
    assert not phrase.policy.permits("identity")
    assert not phrase.policy.permits("stats")


def test_phrase_glue_refuses_a_section_from_a_different_phrase_source():
    from rcql import Executor, parse

    left = version_pair()
    right = version_pair()

    with pytest.raises(TypeError, match="source-bound"):
        Executor(params={"left": left, "right": right}).execute(
            parse("FROM PHRASE($left) RETURN GLUE($right)")
        )
