"""Exact primary query fields, section aggregation and full structural closure."""
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.partition import document_field, section_response, section_coverage
from rexgraph.sectioning import Sectioning, add_sectioning, add_coarsening, sectionings_of
from rexgraph.tower import semantic_closure


def fixture():
    rex = RexGraph.from_cells([4, [[0, 1, 2], [0, 0], [2], [1, 2, 3], [0, 1]]])
    add_sectioning(rex, "s", {"a": [0, 1], "b": [2, 3, 4]})
    return rex, sectionings_of(rex)["s"]


@pytest.mark.parametrize("weight", ["flat", "invdeg"])
@pytest.mark.parametrize("reading", ["mass", "coverage"])
@pytest.mark.parametrize("exact", [True, False])
def test_readings_match_independent_coalesced_boundary(weight, reading, exact):
    rex, s = fixture()
    # C1 columns: branching, cancelling loop, witness, branching, pair.
    columns = [{0: -Q(1), 1: Q(1, 2), 2: Q(1, 2)}, {}, {2: Q(1)},
               {1: -Q(1), 2: Q(1, 2), 3: Q(1, 2)}, {0: -Q(1), 1: Q(1)}]
    degree = [4, 3, 3, 1]  # Stored participant incidence counts, including loop slots.
    x = [Q(1, degree[i]) if weight == "invdeg" else Q(1) for i in range(4)]
    expected = []
    for c in columns:
        mass = sum((abs(v)*x[i] for i, v in c.items()), Q(0))
        signed = sum((v*x[i] for i, v in c.items()), Q(0))
        expected.append(mass if reading == "mass" else mass-abs(signed))
    field = document_field(rex, [0, 1, 2, 3, 0], reading=reading, seed_weight=weight, exact=exact)
    want = expected if exact else list(map(float, expected))
    np.testing.assert_array_equal(field.numpy(), want)
    assert field.source is rex and field.grade == 1
    got, labels = section_response(rex, s, [0, 1, 2, 3], propagator=reading, seed_weight=weight, exact=exact)
    sums = [sum(expected[:2]), sum(expected[2:])]
    np.testing.assert_array_equal(got, sums if exact else list(map(float, sums)))
    assert labels == ["a", "b"]


def test_flat_weight_is_not_silently_inverse_degree():
    r = RexGraph.from_cells([2, [[0, 1], [0, 1]]])
    s = Sectioning("s", 1, [0, 2], [0, 1], ["all"], n_cells=2)
    assert section_response(r, s, [0], seed_weight="flat", exact=True)[0][0] == 2
    assert section_response(r, s, [0], seed_weight="invdeg", exact=True)[0][0] == 1


@pytest.mark.parametrize("seeds", [[], [0]])
def test_loops_do_not_create_mass_or_coverage(seeds):
    r = RexGraph.from_cells([1, [[0, 0]]])
    s = Sectioning("s", 1, [0, 1], [0], ["loop"], n_cells=1)
    assert section_response(r, s, seeds, exact=True)[0][0] == 0
    assert section_coverage(r, s, seeds, exact=True)[0][0] == 0
    # Public directional degrees still count stored incidence slots.
    np.testing.assert_array_equal(r.in_degree, [1])
    np.testing.assert_array_equal(r.out_degree, [1])


def test_empty_profiles_keep_the_channel_axis():
    r, s = fixture()
    out, labels, channels = section_response(r, s, [], channels=True)
    assert out.shape == (2, len(channels)) and len(channels) == 4
    assert labels == ["a", "b"] and not out.any()


def test_derived_sectioning_resolves_and_sums_the_same_field():
    r, s = fixture()
    add_coarsening(r, "all", "s", [0, 0], ["all"])
    got = section_response(r, sectionings_of(r)["all"], [0, 3], exact=True)[0]
    assert got.tolist() == [sum(document_field(r, [0, 3]).numpy())]


@pytest.mark.parametrize("bad", [[True], [0.5], [-1], [4], [[0]], "0"])
def test_bad_seed_coordinates_are_refused(bad):
    r, s = fixture()
    with pytest.raises((TypeError, ValueError)):
        section_response(r, s, bad)


def test_cover_needs_an_explicit_owner_choice():
    r = RexGraph.from_cells([2, [[0, 1]]])
    s = Sectioning("cover", 1, [0, 1, 2], [0, 0], ["a", "b"], n_cells=1)
    with pytest.raises(ValueError, match="disjoint"):
        section_response(r, s, [0])
    got, _ = section_response(r, s, [0], owner=np.array([1]), n_sections=2, exact=True)
    assert got.tolist() == [0, 1]


@pytest.mark.parametrize("kwargs", [{"seed_weight": "other"}, {"propagator": "other"},
    {"owner": [0.5]*5}, {"owner": [2]*5, "n_sections": 2},
    {"exact": True, "channels": True}, {"exact": True, "propagator": "boundary"}])
def test_unsupported_policies_are_not_substituted(kwargs):
    r, s = fixture()
    with pytest.raises((TypeError, ValueError)):
        section_response(r, s, [0], **kwargs)


def test_closure_preserves_upper_faces_and_isolated_seed_vertices():
    r = RexGraph.from_cells([4, [[0, 1], [1, 2], [0, 2]], [[(0, 1), (1, 1), (2, -1)]]])
    result = semantic_closure(r, 0)
    assert result["converged"] and result["steps"][0]["betti"] == [1, 0, 0]
    isolated = semantic_closure(r, 3)
    assert isolated["vertices"] == [3] and isolated["relations"] == []
    assert isolated["steps"][0]["nV"] == 1 and isolated["steps"][0]["betti"][0] == 1


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": 0.2}, {"seed": 0, "grade": 1},
                                    {"seed": 0, "max_depth": 0}])
def test_closure_refuses_invalid_scope(kwargs):
    r, _ = fixture()
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        semantic_closure(r, **kwargs)
