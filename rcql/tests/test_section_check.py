"""Typed compact sheaf compatibility, not inferred chain maps or cohomology."""
from fractions import Fraction as Q

import pytest
from rexgraph.graph import RexGraph
from rexgraph.sheaf import ExactSectionCheck, ExactSheaf

from rcql import (
    Executor,
    PhraseCorrespondence,
    PhraseMapError,
    PhraseSectionCheck,
    PhraseSheaf,
    PhraseStalk,
    SourcePolicy,
    bind,
    call,
    parse,
    query,
    source,
)


def phrase():
    stalks = tuple(PhraseStalk(name, bind(name, object(), SourcePolicy.allow("read")))
                   for name in ("observed", "predicted", "reference"))
    sh = PhraseSheaf(stalks, (PhraseCorrespondence("compare", tuple(s.name for s in stalks)),),
                     stalk_dims={"observed": 2, "predicted": 1, "reference": 3},
                     correspondence_dims={"compare": 1})
    sh.assign("observed", [Q(1, 3), Q(1, 6)])
    sh.assign("predicted", [Q(1, 2)])
    sh.assign("reference", [1, 2, 3])
    sh.restrict("observed", "compare", [[1, 1]])
    sh.identity("predicted", "compare")
    sh.restrict("reference", "compare", [[Q(1, 2), 0, 0]])
    return sh


def test_phrase_rectangular_maps_are_exact_and_keep_named_provenance():
    sh = phrase()
    result = sh.check_section()
    assert isinstance(result, PhraseSectionCheck)
    assert result.compatible and result.comparison_count == 2
    assert result.policy.permissions == frozenset({"read"})
    assert result.contributors == sh.contributors
    sh.assign("predicted", [Q(2, 3)])
    check = sh.check_section()
    assert not check.compatible
    assert [(o.left_stalk, o.right_stalk, o.correspondence, o.residual)
            for o in check.named_obstructions] == [("observed", "predicted", "compare", (Q(-1, 6),))]
    assert sh.glue().obstruction_count == 2


def test_named_text_builder_and_member_projection_use_the_same_typed_contract():
    sh = phrase()
    executor = Executor(params={"section": sh})
    result = executor.execute(parse(
        'FROM PHRASE($section) LET c=SECTION_CHECK(section=$section) '
        'RETURN c AS certificate, c.compatible AS compatible, c.incidence_count AS incidences, '
        'c.comparison_count AS comparisons, GLUE($section).ratio AS ratio'))
    assert result.values[1:] == (True, 3, 2, Q(1))
    assert result.exactness[1].value == "structural"
    assert [x.value for x in result.exactness[2:]] == ["integer", "integer", "rational"]
    assert result.provenance[0]["result_type"]["kind"] == "ExactSectionCheck"
    node = next(n for n in result.native_plan["nodes"] if n.get("operator") == "SECTION_CHECK")
    assert node["physical"]["method"] == "exact-incidence-section-check"
    built = Executor(sources={"s": sh.rex}).execute(query(source("s"), call("SECTION_CHECK", sh)))
    assert built.values[0] == result.values[0]


def test_explain_never_runs_a_section_check_or_all_pair_glue(monkeypatch):
    sh = phrase()
    for name in ("check_section", "glue", "meets", "_transport"):
        monkeypatch.setattr(sh, name, lambda *a, **k: pytest.fail("EXPLAIN executed a reading"))
    result = Executor(params={"section": sh}).execute(parse(
        'EXPLAIN FROM PHRASE($section) RETURN SECTION_CHECK($section), GLUE($section)'))
    contracts = result.values[0]["returns"]
    assert all(any("no chain-map certificate" in p for p in c["preconditions"]) for c in contracts)


@pytest.mark.parametrize("method", ["SECTION_CHECK", "GLUE"])
def test_same_shape_foreign_sheaf_is_refused_before_reading(method, monkeypatch):
    left, right = phrase(), phrase()
    monkeypatch.setattr(right, "check_section", lambda: pytest.fail("foreign check ran"))
    with pytest.raises(TypeError, match="source-bound"):
        Executor(params={"left": left, "right": right}).execute(parse(
            f'FROM PHRASE($left) RETURN {method}($right)'))


@pytest.mark.parametrize("method", ["SECTION_CHECK", "GLUE"])
@pytest.mark.parametrize("prefix", ["", "EXPLAIN "])
def test_stale_sheaf_incidence_refused_in_planning(method, prefix):
    sh = phrase()
    sh.rex._boundary_idx[0], sh.rex._boundary_idx[1] = sh.rex._boundary_idx[1], sh.rex._boundary_idx[0]
    with pytest.raises(ValueError, match="source incidence or basis changed"):
        Executor(params={"section": sh}).execute(parse(
            prefix + f'FROM PHRASE($section) RETURN {method}($section)'))


def test_identity_cannot_be_guessed_between_unequal_stalk_spaces():
    sh = phrase()
    with pytest.raises(PhraseMapError, match="equal stalk and correspondence dimensions"):
        sh.identity("observed", "compare")


@pytest.mark.parametrize("dimensions", [{"observed": 1}, {"oops": 1}, [1, 2, 3]])
def test_phrase_dimensions_are_complete_named_declarations(dimensions):
    sh = phrase()
    with pytest.raises(PhraseMapError, match="every declared cell"):
        PhraseSheaf(sh.stalks, sh.correspondences, stalk_dims=dimensions)


def test_phrase_source_identity_includes_declared_stalk_coordinate_dimensions():
    sh = phrase()
    uniform = PhraseSheaf(sh.stalks, sh.correspondences)
    assert sh.source_ref.state_digest != uniform.source_ref.state_digest
    assert sh.contributors == uniform.contributors


def test_compact_read_is_not_memoized_across_mutable_sections():
    sh = phrase()
    executor = Executor(params={"section": sh})
    text = parse('FROM PHRASE($section) RETURN SECTION_CHECK($section).compatible')
    assert executor.execute(text).values == (True,)
    sh.assign("predicted", [3])
    assert executor.execute(text).values == (False,)


@pytest.mark.parametrize("member", ["__class__", "policy", "contributors", "source", "h0", "ratio"])
def test_check_members_do_not_expose_arbitrary_carrier_attributes(member):
    sh = phrase()
    with pytest.raises((TypeError, SyntaxError)):
        Executor(params={"section": sh}).execute(parse(
            f'FROM PHRASE($section) RETURN SECTION_CHECK($section).{member}'))


def test_local_sheaf_compact_diagnostics_preserve_all_mediator_requirements():
    rex = RexGraph.from_hypergraph([0, 4, 8], [0, 1, 2, 3, 0, 1, 4, 5])
    sh = ExactSheaf(rex)
    sh.assign(0, [Q(1, 3)])
    sh.assign(1, [Q(1, 2)])
    result = Executor(sources={"s": rex}, params={"section": sh}).execute(parse(
        'FROM $s RETURN SECTION_CHECK($section)'))
    check = result.values[0]
    assert isinstance(check, ExactSectionCheck)
    assert check.incidence_count == 8 and check.comparison_count == 2
    assert [(o.mediator, o.residual) for o in check.obstructions] == [(0, (Q(-1, 6),)), (1, (Q(-1, 6),))]
