"""
Tests for rexgraph.types -- typed containers and enumerations.

No heavy dependencies. Tests verify enum values, NamedTuple construction,
field access, and _fields metadata.

Verifies:
    - Enumerations: EdgeType, HodgeComponent, BioesTag, FaceEvent,
      TransitionKind, EnergyRegime, OperatorChannel, PredicateOp, JoinType
    - NamedTuples: construction, field access, _fields, _asdict roundtrip
    - Updated SpectralBundle fields (RL, hats, nhats, chi, K1, L_C)
    - Updated VoidComplex fields (tri_edges, void_indices)
    - RCFBundle with chi field
"""
import numpy as np
import pytest

from rexgraph.types import (
    # Enums
    EdgeType,
    HodgeComponent,
    BioesTag,
    FaceEvent,
    TransitionKind,
    EnergyRegime,
    OperatorChannel,
    PredicateOp,
    JoinType,
    # NamedTuples
    HodgeDecomposition,
    SpectralBundle,
    RCFBundle,
    VertexBundle,
    VoidComplex,
    RCFEResult,
    EnergyKinPot,
    SubComplex,
    MeasurementResult,
    Filtration,
    SchrodingerState,
    WaveState,
    JoinResult,
    ImputeResult,
    PropagationResult,
    StandardMetrics,
)


# Enumerations

class TestEdgeType:

    def test_values(self):
        assert EdgeType.STANDARD == 0
        assert EdgeType.SELF_LOOP == 1
        assert EdgeType.BRANCHING == 2
        assert EdgeType.WITNESS == 3

    def test_is_int(self):
        assert isinstance(EdgeType.STANDARD, int)


class TestHodgeComponent:

    def test_values(self):
        assert HodgeComponent.GRADIENT == 0
        assert HodgeComponent.CURL == 1
        assert HodgeComponent.HARMONIC == 2


class TestBioesTag:

    def test_values(self):
        assert BioesTag.BEGIN == 0
        assert BioesTag.INSIDE == 1
        assert BioesTag.OUTSIDE == 2
        assert BioesTag.END == 3
        assert BioesTag.SINGLE == 4


class TestFaceEvent:

    def test_values(self):
        assert FaceEvent.PERSIST == 0
        assert FaceEvent.BORN == 1
        assert FaceEvent.DIED == 2
        assert FaceEvent.SPLIT == 3
        assert FaceEvent.MERGE == 4


class TestTransitionKind:

    def test_values(self):
        assert TransitionKind.MARKOV == 0
        assert TransitionKind.SCHRODINGER == 1
        assert TransitionKind.DIFFERENTIAL == 2
        assert TransitionKind.REWRITE == 3


class TestEnergyRegime:

    def test_values(self):
        assert EnergyRegime.KINETIC == 0
        assert EnergyRegime.CROSSOVER == 1
        assert EnergyRegime.POTENTIAL == 2


class TestRCFEnums:

    def test_operator_channel(self):
        assert OperatorChannel.TOPOLOGICAL == 0
        assert OperatorChannel.GEOMETRIC == 1
        assert OperatorChannel.FRUSTRATION == 2
        assert OperatorChannel.COPC == 3

    def test_predicate_op(self):
        assert PredicateOp.GT == 0
        assert PredicateOp.BETWEEN == 6

    def test_join_type(self):
        assert JoinType.INNER == 0
        assert JoinType.LEFT == 1
        assert JoinType.OUTER == 2
        assert JoinType.UNION == 3


# NamedTuples - Hodge

class TestHodgeDecomposition:

    def test_construction(self):
        g = np.ones(5)
        c = np.zeros(5)
        h = np.zeros(5)
        hd = HodgeDecomposition(gradient=g, curl=c, harmonic=h)
        assert np.allclose(hd.gradient, g)
        assert np.allclose(hd.curl, c)
        assert np.allclose(hd.harmonic, h)

    def test_fields(self):
        assert HodgeDecomposition._fields == ("gradient", "curl", "harmonic")


# NamedTuples - SpectralBundle (updated)

class TestSpectralBundle:

    def test_has_rl_fields(self):
        """SpectralBundle includes relational Laplacian fields from build_RL."""
        assert "RL" in SpectralBundle._fields
        assert "hats" in SpectralBundle._fields
        assert "nhats" in SpectralBundle._fields
        assert "hat_names" in SpectralBundle._fields
        assert "trace_values" in SpectralBundle._fields
        assert "chi" in SpectralBundle._fields

    def test_has_k1_and_lc(self):
        """SpectralBundle includes K1 overlap matrix and L_C copath Laplacian."""
        assert "K1" in SpectralBundle._fields
        assert "L_C" in SpectralBundle._fields

    def test_has_laplacian_fields(self):
        assert "L0" in SpectralBundle._fields
        assert "L1_down" in SpectralBundle._fields
        assert "L1_up" in SpectralBundle._fields
        assert "L1_full" in SpectralBundle._fields
        assert "L2" in SpectralBundle._fields

    def test_has_betti(self):
        assert "beta0" in SpectralBundle._fields
        assert "beta1" in SpectralBundle._fields
        assert "beta2" in SpectralBundle._fields

    def test_has_coupling(self):
        assert "alpha_G" in SpectralBundle._fields
        assert "alpha_T" in SpectralBundle._fields

    def test_has_legacy_rl1(self):
        """RL_1 = L1 + alpha_G * L_O is still present as legacy 2-term version."""
        assert "RL_1" in SpectralBundle._fields
        assert "evals_RL_1" in SpectralBundle._fields
        assert "evecs_RL_1" in SpectralBundle._fields


# NamedTuples - RCFBundle (updated)

class TestRCFBundle:

    def test_fields(self):
        assert RCFBundle._fields == (
            "RL", "hats", "nhats", "chi", "trace_values", "hat_names")

    def test_construction(self):
        nE = 3
        nhats = 2
        rcf = RCFBundle(
            RL=np.eye(nE),
            hats=[np.eye(nE) * 0.5, np.eye(nE) * 0.5],
            nhats=nhats,
            chi=np.ones((nE, nhats)) / nhats,
            trace_values=np.array([1.0, 1.0]),
            hat_names=["L1_down", "L_O"],
        )
        assert rcf.nhats == 2
        assert rcf.chi.shape == (3, 2)
        assert rcf.hat_names == ["L1_down", "L_O"]


# NamedTuples - VertexBundle

class TestVertexBundle:

    def test_fields(self):
        assert VertexBundle._fields == ("phi", "chi_star", "kappa")

    def test_construction(self):
        vb = VertexBundle(
            phi=np.ones((4, 2)) / 2,
            chi_star=np.ones((4, 2)) / 2,
            kappa=np.ones(4),
        )
        assert vb.phi.shape == (4, 2)
        assert vb.kappa.shape == (4,)


# NamedTuples - VoidComplex (updated)

class TestVoidComplex:

    def test_has_new_fields(self):
        """VoidComplex includes tri_edges and void_indices."""
        assert "tri_edges" in VoidComplex._fields
        assert "void_indices" in VoidComplex._fields

    def test_has_core_fields(self):
        assert "Bvoid" in VoidComplex._fields
        assert "Lvoid" in VoidComplex._fields
        assert "n_voids" in VoidComplex._fields
        assert "n_potential" in VoidComplex._fields
        assert "eta" in VoidComplex._fields
        assert "chi_void" in VoidComplex._fields
        assert "fills_beta" in VoidComplex._fields
        assert "void_strain" in VoidComplex._fields

    def test_field_order(self):
        expected = (
            "Bvoid", "Lvoid", "n_voids", "n_potential",
            "tri_edges", "void_indices",
            "eta", "chi_void", "fills_beta", "void_strain",
        )
        assert VoidComplex._fields == expected


# NamedTuples - RCFEResult

class TestRCFEResult:

    def test_fields(self):
        assert RCFEResult._fields == (
            "curvature", "strain", "bianchi_ok", "bianchi_residual")

    def test_construction(self):
        r = RCFEResult(
            curvature=np.array([0.1, 0.2, 0.3]),
            strain=0.5,
            bianchi_ok=True,
            bianchi_residual=1e-15,
        )
        assert r.bianchi_ok is True
        assert r.strain == 0.5


# NamedTuples - Energy

class TestEnergyKinPot:

    def test_construction(self):
        ekp = EnergyKinPot(E_kin=1.0, E_pot=2.0, ratio=0.5)
        assert ekp.E_kin == 1.0
        assert ekp.E_pot == 2.0
        assert ekp.ratio == 0.5

    def test_asdict(self):
        ekp = EnergyKinPot(E_kin=1.0, E_pot=2.0, ratio=0.5)
        d = ekp._asdict()
        assert d["E_kin"] == 1.0


# NamedTuples - Subcomplex, Filtration, State

class TestSubComplex:

    def test_construction(self):
        sc = SubComplex(
            v_mask=np.ones(3, dtype=np.uint8),
            e_mask=np.ones(3, dtype=np.uint8),
            f_mask=np.zeros(0, dtype=np.uint8),
        )
        assert sc.v_mask.shape == (3,)


class TestFiltration:

    def test_construction(self):
        f = Filtration(
            filt_v=np.zeros(3),
            filt_e=np.ones(3),
            filt_f=np.full(1, 2.0),
        )
        assert f.filt_v.shape == (3,)
        assert f.filt_f[0] == 2.0


class TestSchrodingerState:

    def test_construction(self):
        ss = SchrodingerState(f_re=np.ones(5), f_im=np.zeros(5), t=0.1)
        assert ss.t == 0.1
        assert ss.f_re.shape == (5,)


class TestWaveState:

    def test_construction(self):
        ws = WaveState(F=np.ones(5), dFdt=np.zeros(5), t=0.0)
        assert ws.F.shape == (5,)


# NamedTuples - Results

class TestMeasurementResult:

    def test_construction(self):
        mr = MeasurementResult(outcome=2, collapsed=np.array([0, 0, 1.0]))
        assert mr.outcome == 2
        assert mr.collapsed[2] == 1.0


class TestImputeResult:

    def test_construction(self):
        ir = ImputeResult(
            imputed=np.ones(4), confidence=np.ones(4),
            residual=0.01, n_observed=3, n_imputed=1,
        )
        assert ir.n_observed == 3
        assert ir.n_imputed == 1


class TestJoinResult:

    def test_construction(self):
        jr = JoinResult(
            B1j=np.eye(2), B2j=np.zeros((2, 0)),
            nVj=2, nEj=2, nFj=0,
            beta=(1, 1, 0), chain_residual=0.0,
        )
        assert jr.nVj == 2
        assert jr.beta == (1, 1, 0)


class TestPropagationResult:

    def test_construction(self):
        pr = PropagationResult(
            score=0.85, typed_scores=np.array([0.3, 0.3, 0.25]),
            energy=1.5, coverage=0.9,
        )
        assert pr.score == 0.85


class TestStandardMetrics:

    def test_fields(self):
        assert "pagerank" in StandardMetrics._fields
        assert "clustering" in StandardMetrics._fields
        assert "modularity" in StandardMetrics._fields
        assert "n_communities" in StandardMetrics._fields
