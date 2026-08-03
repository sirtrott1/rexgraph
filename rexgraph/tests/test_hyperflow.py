"""The flow construction: both grades, then hyperfaces, then three live channels.

The HGNN built its complex with `from_hypergraph(groups)` and nothing else. That is a
FOREST OF STARS: every column is one group, nothing closes, beta_1 = 0. Curl and harmonic
are both empty, so EVERYTHING IS GRADIENT, and a "flow layer" over it is ordinary message
passing with no structural content to offer. It lost to a parameter-matched MLP, which is
the correct outcome for a model whose structure carries nothing.

The construction that has a flow carries BOTH grades in one complex:

    the GROUP        as a branching relation, one column of arity k. Opens the cycle.
    each MEASUREMENT as a 2-ary relation over the same entities. These are the legs.
    then auto_hyperface, which closes the group against the measurements spanning its
    boundary, coefficients solved from B1 c_f = 0.

Neither half works alone: without the group there is no cycle, without the measurements
there is nothing for a hyperface to close. Together the group opens the hole and the face
fills it, which is `curl_dim = cycle_count - dim_H` in action. Only then are the three
orthogonal edge channels all live and separately addressable, and three channels is what
a flow model is for.
"""

import numpy as np

from rexgraph.flow.hyperflow import build_flow_complex

GROUPS = [[0, 1, 2, 3], [4, 1, 5]]          # two entities, sharing entity 1


def _channels(fc):
    return (fc.gradient_dim, fc.curl_dim, fc.harmonic_dim)


#### the construction
def test_groups_alone_are_a_forest_of_stars():
    """What the HGNN did. No cycle, so no curl and no harmonic: all gradient."""
    fc = build_flow_complex(GROUPS, measurements=False, close=False)
    assert fc.harmonic_dim == 0
    assert fc.curl_dim == 0
    assert int(fc.rex.betti[1]) == 0


def test_measurements_alone_do_not_close_either():
    fc = build_flow_complex(GROUPS, include_groups=False, close=False)
    assert fc.curl_dim == 0


def test_both_grades_open_the_cycle():
    """The group is what creates the hole. Before a face it is HARMONIC."""
    fc = build_flow_complex(GROUPS, close=False)
    assert fc.harmonic_dim > 0
    assert fc.curl_dim == 0
    assert fc.harmonic_dim == int(fc.rex.betti[1])


def test_the_hyperface_converts_harmonic_to_curl():
    """curl_dim = cycle_count - dim_H. A face fills a cycle: it stops being a hole and
    starts bounding. The cycle COUNT does not change, only what it is."""
    opened = build_flow_complex(GROUPS, close=False)
    closed = build_flow_complex(GROUPS, close=True)
    assert closed.n_faces == len(GROUPS)
    assert closed.curl_dim == opened.harmonic_dim
    assert closed.harmonic_dim == 0
    assert closed.cycle_count == opened.cycle_count


def test_all_three_channels_are_live_and_orthogonal():
    """The point of the construction. im(B1^T) + im(B2) + ker(L1) = R^nE, as a direct
    sum, so the dimensions add to nE exactly."""
    fc = build_flow_complex(GROUPS + [[6, 7, 8]], close=True)
    g, c, h = _channels(fc)
    assert g > 0 and c > 0
    assert g + c + h == int(fc.rex.nE)


def test_the_chain_condition_holds_exactly():
    fc = build_flow_complex(GROUPS, close=True)
    assert fc.chain_residual == 0.0


def test_nothing_is_invented():
    """No hub vertex (star expansion) and no pairwise fill (clique expansion). The
    entities are exactly those named."""
    fc = build_flow_complex(GROUPS, close=True)
    assert int(fc.rex.nV) == len({v for g in GROUPS for v in g})


#### the flow itself
def test_a_signal_decomposes_into_the_three_channels():
    fc = build_flow_complex(GROUPS, close=True)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(int(fc.rex.nE))
    parts = fc.decompose(f)
    recon = parts["gradient"] + parts["curl"] + parts["harmonic"]
    assert np.allclose(recon, f, atol=1e-8)


def test_the_parts_are_mutually_orthogonal():
    fc = build_flow_complex(GROUPS, close=True)
    rng = np.random.default_rng(1)
    p = fc.decompose(rng.standard_normal(int(fc.rex.nE)))
    keys = ["gradient", "curl", "harmonic"]
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(float(p[keys[i]] @ p[keys[j]])) < 1e-8


def test_the_dirac_moves_signal_between_grades():
    """Equiweight is what guarantees it: Gamma D + D Gamma = 0 means D is ODD with
    respect to the grading, so it can never leave a signal in the grade it started in.
    That is the propagation organ, not a metaphor."""
    fc = build_flow_complex(GROUPS, close=True)
    psi = np.zeros(int(fc.rex.dirac_dimension))
    psi[fc.grade_slice(1)] = 1.0                    # start entirely on edges
    out = fc.step(psi)
    assert np.abs(out[fc.grade_slice(1)]).max() < 1e-12   # nothing stayed
    assert np.abs(out[fc.grade_slice(0)]).max() > 0       # some went down
    assert fc.rex.equiweight_residual == 0


def test_a_constant_vertex_signal_is_annihilated():
    """Level linking: every boundary column sums to zero, so B1^T kills the constant.
    A flow seeded uniformly on vertices has nowhere to go."""
    fc = build_flow_complex(GROUPS, close=True)
    psi = np.zeros(int(fc.rex.dirac_dimension))
    psi[fc.grade_slice(0)] = 1.0
    assert np.abs(fc.step(psi)).max() < 1e-12


#### temporal
def test_propagation_conserves_the_graded_norm():
    """dirac_light is a unitary (wave) propagator, so the total stays put while the
    grades exchange. That is what makes it a flow rather than a diffusion."""
    fc = build_flow_complex(GROUPS, close=True)
    rng = np.random.default_rng(2)
    psi = rng.standard_normal(int(fc.rex.dirac_dimension))
    n0 = float(psi @ psi)
    re, im = fc.propagate(psi, t=0.7)
    assert abs(float(re @ re) + float(im @ im) - n0) < 1e-6


def test_the_propagator_obeys_a_parity_selection_rule():
    """Equiweight, showing up in the dynamics rather than on paper.

    e^{-itD} = cos(tD) - i sin(tD). Equiweight makes D ODD with respect to the grading,
    so any EVEN power of D preserves grade parity and any ODD power flips it. Therefore
    the REAL part (cosine, even) stays in the grade it started in, and the IMAGINARY part
    (sine, odd) is the one that crosses. From a pure edge seed:

        t=0.25   real [0, 4.844, 0]   imag [2.156, 0, 0]
        t=1.0    real [0, 3.570, 0]   imag [3.430, 0, 0]

    Transport is therefore read off the imaginary part, not the real one.
    """
    fc = build_flow_complex(GROUPS, close=True)
    psi = np.zeros(int(fc.rex.dirac_dimension))
    psi[fc.grade_slice(1)] = 1.0                      # a pure edge seed, odd grade
    for t in (0.25, 0.5, 1.0):
        re, im = fc.propagate(psi, t=t)
        e_re, e_im = fc.grade_energy(re), fc.grade_energy(im)
        assert e_re[0] < 1e-12 and e_re[2] < 1e-12    # cosine keeps it on odd grades
        assert e_re[1] > 1e-6
        assert e_im[1] < 1e-12                        # sine takes it entirely off them
        assert e_im[0] > 1e-6


def test_the_grade_profile_moves_over_time():
    """The observable a temporal model reads: where the signal IS, by grade, as t runs.
    Total energy is conserved and the split between grades is what moves."""
    fc = build_flow_complex(GROUPS, close=True)
    psi = np.zeros(int(fc.rex.dirac_dimension))
    psi[fc.grade_slice(1)] = 1.0
    crossed = []
    for t in (0.25, 0.5, 1.0):
        re, im = fc.propagate(psi, t=t)
        crossed.append(fc.grade_energy(im)[0] + fc.grade_energy(im)[2])
    assert crossed[0] < crossed[-1]                   # more has crossed by t=1 than t=0.25
    assert all(c > 1e-6 for c in crossed)


def test_an_unclosed_complex_has_no_curl_to_flow_through():
    """The negative control. Without the hyperface the curl tier is empty, so the same
    signal has one fewer channel to move through."""
    closed = build_flow_complex(GROUPS, close=True)
    opened = build_flow_complex(GROUPS, close=False)
    assert closed.curl_dim > 0
    assert opened.curl_dim == 0
    assert closed.n_faces > opened.n_faces


def test_flow_complex_reports_its_own_shape():
    fc = build_flow_complex(GROUPS, close=True)
    s = fc.summary()
    assert s["n_faces"] == len(GROUPS)
    assert s["gradient"] + s["curl"] + s["harmonic"] == int(fc.rex.nE)
    assert s["chain_residual"] == 0.0


#### the E equation: energy is additive across the towers
def test_energy_is_additive_across_the_towers():
    """E = E_gradient + E_curl + E_harmonic, exactly.

    Additivity is what the chain condition buys: the boundary pairing vanishes,
    <B1^T a, B2 b> = a^T (B1 B2) b = 0, so gradient and curl are orthogonal by
    construction rather than by numerical accident. A nonzero cross term means the
    decomposition is wrong, not that the data is awkward, which is what makes this a
    usable check on the recovery rather than a restatement of it.
    """
    fc = build_flow_complex(GROUPS + [[6, 7, 8], [9, 7, 10]], close=True)
    rng = np.random.default_rng(3)
    for _ in range(5):
        e = fc.energy_split(rng.standard_normal(int(fc.rex.nE)))
        assert abs(e["cross_residual"]) < 1e-8 * max(1.0, e["total"])


def test_the_boundary_pairing_is_what_makes_it_additive():
    """The mechanism, checked directly rather than assumed: <B1^T a, B2 b> = 0."""

    fc = build_flow_complex(GROUPS, close=True)
    B1 = np.asarray(fc.rex.B1, dtype=float)
    B2 = np.asarray(fc.rex.B2, dtype=float)
    rng = np.random.default_rng(4)
    for _ in range(5):
        a = rng.standard_normal(B1.shape[0])
        b = rng.standard_normal(B2.shape[1])
        assert abs(float((B1.T @ a) @ (B2 @ b))) < 1e-9


def test_the_decomposition_uses_the_library_projector():
    """Not dense lstsq. The harmonic part must agree with harmonic_projection applied
    directly, which is the low-rank sparse-Gram path."""
    from rexgraph.harmonic_sparse import harmonic_basis, harmonic_projection

    fc = build_flow_complex(GROUPS, close=True)
    rng = np.random.default_rng(5)
    f = rng.standard_normal(int(fc.rex.nE))
    H = harmonic_basis(fc.rex)
    direct = harmonic_projection(H, f) if H is not None and H.shape[1] else np.zeros(len(f))
    assert np.allclose(fc.decompose(f)["harmonic"], direct, atol=1e-10)
