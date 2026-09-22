"""The effective mode count of each Hodge sector is exact, and the harmonic log is its logarithm.

Every count is checked against L_k + w Pi^h assembled densely over Q from the metric
adjoint, with the harmonic projector built from the kernel of the assembled operator,
so the sector traces, the chain law cross term and the completion identity are all
tested by a route that shares none of the trace code.
"""
import itertools
import math
from fractions import Fraction as Q

import numpy as np
import pytest

from rexgraph import RexGraph
from rexgraph.exact_green import ExactSparse
from rexgraph.graded_boundary import solid_octahedron_3rex, square_pyramid_3rex
from rexgraph.harmonic_modes import WEIGHTS, effective_modes, grade_traces, harmonic_log
from rexgraph.native_rank import exact_tower


def _complete(n, faces=False, **kw):
    pairs = list(itertools.combinations(range(n), 2))
    rex = RexGraph(sources=np.array([a for a, _ in pairs], np.int32),
                   targets=np.array([b for _, b in pairs], np.int32), **kw)
    if faces:
        index = {p: i for i, p in enumerate(pairs)}
        rex.add_faces([[index[a, b], index[a, c], index[b, c]]
                       for a, b, c in itertools.combinations(range(n), 3)])
        rex._ensure_clean()
    return rex


def _k4_one_face(**kw):
    rex = _complete(4, **kw)
    rex.add_faces([[0, 1, 3]])
    rex._ensure_clean()
    return rex


def _ring(n, **kw):
    return RexGraph(sources=np.arange(n, dtype=np.int32),
                    targets=np.array([(i + 1) % n for i in range(n)], np.int32), **kw)


def _two_rings():
    return RexGraph(sources=np.array([0, 1, 2, 3, 4, 5, 6], np.int32),
                    targets=np.array([1, 2, 3, 0, 5, 6, 4], np.int32))


def _branching():
    return RexGraph.from_hypergraph(np.array([0, 3, 7, 9, 11, 13, 15], np.int32),
                                    np.array([0, 1, 2, 1, 3, 4, 5, 0, 1, 1, 2, 0, 2, 4, 5], np.int32))


UNWEIGHTED = {
    "K5": lambda: _complete(5),
    "K5 filled": lambda: _complete(5, faces=True),
    "K4 one face": _k4_one_face,
    "two rings": _two_rings,
    "branching": _branching,
    "solid octahedron": lambda: RexGraph.from_cells(solid_octahedron_3rex()),
    "square pyramid": lambda: RexGraph.from_cells(square_pyramid_3rex()),
}
WEIGHTED = {
    "weighted K4 one face": lambda: _k4_one_face(w_E=[Q(1), Q(2), Q(3), Q(1, 2), Q(5, 3), Q(7)]),
    "weighted ring": lambda: _ring(5, w_E=[Q(2), Q(1, 3), Q(4), Q(1), Q(3, 2)]),
    "weighted K5 filled": lambda: _complete(5, faces=True, w_E=[Q(k + 1, 3) for k in range(10)]),
}


def _dense(columns, nrows):
    out = [[Q(0)] * len(columns) for _ in range(nrows)]
    for j, column in enumerate(columns):
        for i, value in column.items():
            out[i][j] = Q(value)
    return out


def _mm(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(len(B))), Q(0)) for j in range(len(B[0]))]
            for i in range(len(A))]


def _t(A, ncols):
    return [[A[i][j] for i in range(len(A))] for j in range(ncols)]


def _scale_rows(values, A):
    return [[values[i] * x for x in row] for i, row in enumerate(A)]


def _metric(rex, grade, size):
    weights = rex.edge_metric_exact if grade == 1 else None
    return [Q(1)] * size if weights is None else [Q(w) for w in weights]


def _assembled(rex, grade):
    """(L_k, Pi^h) as dense rational matrices, the kernel read from L_k itself."""
    shapes, columns = exact_tower(rex)
    sizes = [shapes[0][0]] + [shape[1] for shape in shapes]
    n = sizes[grade]
    m = _metric(rex, grade, n)
    L = [[Q(0)] * n for _ in range(n)]
    if grade > 0:
        B = _dense(columns[grade - 1], sizes[grade - 1])
        lower = _scale_rows([1 / x for x in m],
                            _mm(_t(B, n), _scale_rows(_metric(rex, grade - 1, sizes[grade - 1]), B)))
        L = [[a + b for a, b in zip(r, s, strict=True)] for r, s in zip(L, lower, strict=True)]
    if grade < len(shapes):
        B = _dense(columns[grade], n)
        upper_m = _metric(rex, grade + 1, sizes[grade + 1])
        adjoint = _scale_rows([1 / x for x in upper_m], _t(_scale_rows(m, B), sizes[grade + 1]))
        L = [[a + b for a, b in zip(r, s, strict=True)] for r, s in zip(L, _mm(B, adjoint), strict=True)]
    frame = ExactSparse(n, n, {(i, j): v for i, row in enumerate(L)
                               for j, v in enumerate(row) if v}).kernel_frame()
    H = [[frame.entries.get((i, j), Q(0)) for j in range(frame.ncols)] for i in range(n)]
    P = [[Q(0)] * n for _ in range(n)]
    if frame.ncols:
        HtM = [[H[i][j] * m[i] for i in range(n)] for j in range(frame.ncols)]
        gram = _mm(HtM, H)
        system = ExactSparse(len(gram), len(gram), {(i, j): v for i, row in enumerate(gram)
                                                    for j, v in enumerate(row) if v})
        for col in range(n):
            a = system.solve([HtM[r][col] for r in range(frame.ncols)])
            for i in range(n):
                P[i][col] = sum((H[i][r] * a[r] for r in range(frame.ncols)), Q(0))
    return L, P


def _traces(X):
    n = len(X)
    return (sum((X[i][i] for i in range(n)), Q(0)),
            sum((X[i][j] * X[j][i] for i in range(n) for j in range(n)), Q(0)))


@pytest.mark.parametrize("name", ["K5", "K5 filled", "K4 one face", "two rings", "branching"])
def test_grade_one_sectors_are_the_existing_exact_readers(name):
    rex = UNWEIGHTED[name]()
    traces = grade_traces(rex, 1)
    t2, l2 = rex._second_traces()
    assert traces.down == (rex.trace_T, t2)
    assert traces.up == (rex.trace_L1, l2)
    assert traces.betti == rex.betti_tower[1]


@pytest.mark.parametrize("name", sorted({**UNWEIGHTED, **WEIGHTED}))
def test_every_grade_matches_the_assembled_completion(name):
    rex = {**UNWEIGHTED, **WEIGHTED}[name]()
    for grade in range(len(rex.betti_tower)):
        L, P = _assembled(rex, grade)
        traces = grade_traces(rex, grade)
        assert _traces(P) == (traces.betti, traces.betti)
        assert all(not v for row in _mm(L, P) for v in row)
        assert traces.traces("hodge") == _traces(L)
        for weight in WEIGHTS:
            w = traces.harmonic_weight(weight)
            completed = [[a + w * b for a, b in zip(r, s, strict=True)] for r, s in zip(L, P, strict=True)]
            assert traces.traces("completed", weight) == _traces(completed)
            first, second = _traces(completed)
            assert effective_modes(rex, grade, "completed", weight) == (first * first / second if second else 0)


def _ratios(first, second, rank):
    return first / rank, second / first


@pytest.mark.parametrize("name", sorted({**UNWEIGHTED, **WEIGHTED}))
def test_each_scale_invariant_weight_keeps_its_own_trace_ratio(name):
    """The mean weight keeps tr/rank, the energy weight keeps tr^2/tr and adds beta to the count."""
    rex = {**UNWEIGHTED, **WEIGHTED}[name]()
    for grade in range(len(rex.betti_tower)):
        traces = grade_traces(rex, grade)
        if traces.rank == 0:
            continue
        mean, energy = _ratios(*traces.traces("hodge"), traces.rank)
        assert (traces.harmonic_weight("mean"), traces.harmonic_weight("energy")) == (mean, energy)
        assert _ratios(*traces.traces("completed", "mean"), traces.size)[0] == mean
        assert _ratios(*traces.traces("completed", "energy"), traces.size)[1] == energy
        additive = traces.effective_modes("hodge") + traces.betti
        assert traces.effective_modes("completed", "energy") == additive
        for weight in WEIGHTS:
            assert traces.effective_modes("completed", weight) <= additive
        if traces.betti:
            assert (traces.effective_modes("completed", "unit") == additive) == (energy == 1)
            assert (traces.effective_modes("completed", "mean") == additive) == (mean == energy)


@pytest.mark.parametrize("s", [Q(1, 16), Q(1, 3), Q(5), Q(16)])
def test_the_scale_invariant_weights_ignore_a_uniform_edge_metric_without_faces(s):
    """Without faces M_1 = sI scales L_1 by 1/s; only the unit weight reads that scale."""
    plain, scaled = _complete(5), _complete(5, w_E=[s] * 10)
    for weight in ("mean", "energy"):
        assert effective_modes(scaled, 1, "completed", weight) == effective_modes(plain, 1, "completed", weight) == 10
    assert effective_modes(scaled, 1) != effective_modes(plain, 1)
    assert effective_modes(_complete(5, w_E=[Q(5)] * 10), 1) == 10


def _calculus(rex, metrics=None):
    from rexgraph.native_field import NativeFieldCalculus
    calculus = NativeFieldCalculus.from_rex(rex)
    if metrics is None:
        return calculus
    return NativeFieldCalculus(calculus.complex, tuple(
        metric(space) for metric, space in zip(metrics, calculus.complex.spaces, strict=True)))


def _calculus_traces(calculus, grade, w):
    """tr and tr^2 of L_k + w Pi^h from the calculus's own Hodge and harmonic actions."""
    n = calculus.complex.sizes[grade]
    if n == 0:
        return Q(0), Q(0)
    eye = np.eye(n, dtype=int)
    X = calculus.hodge(grade).apply(eye) + w * calculus.sector(grade, "harmonic").apply(eye)
    return sum(X[i, i] for i in range(n)), sum(X[i, j] * X[j, i] for i in range(n) for j in range(n))


def _diagonal(weights):
    from rexgraph.coordinate_map import CoordinateMetric
    return lambda space: CoordinateMetric.diagonal(space, weights)


def _coupled(entries):
    from rexgraph.coordinate_map import CoordinateMetric
    return lambda space: CoordinateMetric(space, entries)


def _identity(space):
    from rexgraph.coordinate_map import CoordinateMetric
    return CoordinateMetric.identity(space)


def _tridiagonal(n, a, b):
    return tuple([(i, i, a) for i in range(n)] + [(i, i + 1, b) for i in range(n - 1)]
                 + [(i + 1, i, b) for i in range(n - 1)])


@pytest.mark.parametrize("name", sorted({**UNWEIGHTED, **WEIGHTED}))
def test_a_field_calculus_from_the_rex_reads_the_same_traces(name):
    rex = {**UNWEIGHTED, **WEIGHTED}[name]()
    calculus = _calculus(rex)
    for grade in range(len(rex.betti_tower)):
        assert grade_traces(calculus, grade) == grade_traces(rex, grade)


DECLARED = {
    "4 cycle filled, every grade weighted": (
        lambda: RexGraph(sources=np.array([0, 1, 0, 2]), targets=np.array([1, 3, 2, 3]),
                         B2_col_ptr=np.array([0, 4]), B2_row_idx=np.array([0, 1, 2, 3], dtype=np.int32),
                         B2_vals=np.array([1, 1, -1, -1], dtype=np.int32)),
        (_diagonal([Q(2), Q(3), Q(4), Q(5)]), _diagonal([Q(2, 3), Q(3, 5), Q(5, 7), Q(7, 11)]),
         _diagonal([Q(7, 3)]))),
    "K4 one face, every grade weighted": (
        _k4_one_face,
        (_diagonal([Q(1), Q(2), Q(1, 3), Q(5)]), _diagonal([Q(1), Q(2), Q(3), Q(1, 2), Q(5, 3), Q(7)]),
         _diagonal([Q(9, 4)]))),
    "solid octahedron, every grade weighted": (
        lambda: RexGraph.from_cells(solid_octahedron_3rex()),
        (_diagonal([Q(k + 1) for k in range(6)]), _diagonal([Q(k + 2, 3) for k in range(12)]),
         _diagonal([Q(5, k + 1) for k in range(8)]), _diagonal([Q(3, 2)]))),
    "K4 two faces, coupled edge and face metrics": (
        lambda: with_faces_k4([[0, 1, 3], [0, 2, 4]]),
        (_identity, _coupled(_tridiagonal(6, Q(3), Q(1))), _coupled(((0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 3))))),
}


def with_faces_k4(faces):
    rex = _complete(4)
    rex.add_faces(faces)
    rex._ensure_clean()
    return rex


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_declared_metrics_at_every_grade_match_the_calculus_actions(name):
    build, metrics = DECLARED[name]
    calculus = _calculus(build(), metrics)
    for grade in range(len(calculus.metrics)):
        traces = grade_traces(calculus, grade)
        assert traces.traces("hodge") == _calculus_traces(calculus, grade, 0)
        for weight in WEIGHTS:
            w = traces.harmonic_weight(weight)
            assert traces.traces("completed", weight) == _calculus_traces(calculus, grade, w)


def test_the_known_counts():
    assert [effective_modes(_complete(5), 1, s) for s in ("hodge", "completed")] == [4, Q(338, 53)]
    assert [effective_modes(_k4_one_face(), 1, s) for s in ("hodge", "completed")] == [Q(75, 19), Q(289, 59)]
    assert [effective_modes(_branching(), 1, s) for s in ("hodge", "completed")] == [Q(4225, 1423), Q(5929, 1495)]
    assert effective_modes(_complete(5, faces=True), 1) == 10


@pytest.mark.parametrize("n", range(3, 9))
def test_a_ring_reads_its_closed_form(n):
    """C_n has L_1 spectrum 2 - 2cos(2 pi j / n): tr = 2n, tr^2 = 6n, and one hole."""
    rex = _ring(n)
    assert effective_modes(rex, 1, "hodge") == Q(2 * n, 3)
    assert effective_modes(rex, 1) == Q((2 * n + 1) ** 2, 6 * n + 1)


def test_the_completion_separates_equal_spectra_with_different_holes():
    """On a faceless complex L_0 and L_1 share their nonzero spectrum, so only beta differs."""
    rex = _complete(5)
    assert effective_modes(rex, 0, "hodge") == effective_modes(rex, 1, "hodge") == 4
    assert effective_modes(rex, 0) == Q(441, 101)
    assert effective_modes(rex, 1) == Q(338, 53)


@pytest.mark.parametrize("s", [Q(2), Q(1, 3), Q(7, 5)])
def test_a_uniform_edge_metric_scales_each_sector_through_the_adjoint(s):
    """M_1 = sI divides every sector through B_1 by s and multiplies every sector through B_2 by s."""
    plain, scaled = _k4_one_face(), _k4_one_face(w_E=[s] * 6)
    for grade, sector, power in [(0, "up", -1), (1, "down", -1), (1, "up", 1), (2, "down", 1)]:
        first, second = getattr(grade_traces(plain, grade), sector)
        assert getattr(grade_traces(scaled, grade), sector) == (first * s**power, second * s**(2 * power))
        assert grade_traces(scaled, grade).effective_modes(sector) == grade_traces(plain, grade).effective_modes(sector)


@pytest.mark.parametrize("name", ["K5", "K4 one face", "branching"])
def test_the_harmonic_log_is_the_log_of_the_exact_count(name):
    from rexgraph.scale_propagator import malaugh_quantities
    rex = UNWEIGHTED[name]()
    count = effective_modes(rex, 1)
    assert harmonic_log(rex, 1) == pytest.approx(math.log(count), rel=0, abs=1e-15)
    float_reading = malaugh_quantities(rex)
    assert harmonic_log(rex, 1, "down") == pytest.approx(float_reading["H_T"], abs=1e-12)
    if rex.nF_hodge:
        assert harmonic_log(rex, 1, "up") == pytest.approx(float_reading["H_S"], abs=1e-12)


@pytest.mark.parametrize("grade", [True, 1.0, "1"])
def test_a_grade_that_is_not_an_integer_is_refused(grade):
    with pytest.raises(TypeError):
        grade_traces(_ring(4), grade)


@pytest.mark.parametrize("grade", [-1, 2])
def test_a_grade_the_complex_does_not_carry_is_refused(grade):
    with pytest.raises(ValueError, match="not present"):
        effective_modes(_ring(4), grade)


def test_an_unknown_sector_is_refused():
    with pytest.raises(ValueError, match="sector"):
        effective_modes(_ring(4), 1, "curl")


def test_an_unknown_weight_is_refused():
    with pytest.raises(ValueError, match="weight"):
        effective_modes(_ring(4), 1, "completed", "median")


@pytest.mark.parametrize("sector", ["hodge", "down", "up"])
@pytest.mark.parametrize("weight", ["mean", "energy"])
def test_a_weight_off_the_completed_sector_is_refused(sector, weight):
    with pytest.raises(ValueError, match="completed sector"):
        effective_modes(_ring(4), 1, sector, weight)
    with pytest.raises(ValueError, match="completed sector"):
        harmonic_log(_ring(4), 1, sector, weight)


def test_a_sector_with_no_modes_has_no_log():
    rex = _ring(4)
    assert effective_modes(rex, 1, "up") == 0
    with pytest.raises(ValueError, match="no modes"):
        harmonic_log(rex, 1, "up")


def test_a_nonpositive_metric_is_refused():
    with pytest.raises(ValueError, match="positive"):
        grade_traces(_ring(3, w_E=[Q(1), Q(0), Q(2)]), 1)
