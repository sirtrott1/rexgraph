"""Sparse overlapping measurements from a declared ambient cell basis.

An accession is a linear measurement, not automatically a projector, partition
or chain map. Named rectangular coordinates require explicit cross pairings;
equal lengths never identify a type coordinate space with the ambient basis.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from hashlib import sha256
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from rexgraph.cells import cell_count
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graded_metric import DiagonalMetric, _diagonal_contraction, _fraction

__all__ = ["TypeAccession", "AccessionFamily", "TypeView", "TypedFamily",
           "TypedMomentTensor", "CoordinateSpace", "CoordinateField", "CrossMetric",
           "TypeRealization", "FamilyMetric", "co_relate", "moment_tensor"]


def _scalar(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("accession coefficients must be real integers, Fractions or floats")
    if isinstance(value, (Integral, Fraction)):
        return _fraction(value)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("accession coefficients must be finite")
    return value


def _entries(entries, shape):
    coefficients, exact = {}, True
    for row, column, value in entries:
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < n
               for i, n in zip((row, column), shape, strict=True)):
            raise ValueError("sparse coordinate must index its declared output/input basis")
        value = _scalar(value)
        exact &= isinstance(value, Fraction)
        key = (int(row), int(column))
        coefficients[key] = coefficients.get(key, Fraction(0)) + value
        _scalar(coefficients[key])
    return tuple((i, j, v) for (i, j), v in sorted(coefficients.items()) if v), exact


def _digest(entries, prefix):
    parts = [f"{i},{j}:" + (f"q:{v.numerator}/{v.denominator}" if isinstance(v, Fraction)
                           else "f:" + float(v).hex()) for i, j, v in entries]
    return sha256((prefix + "|".join(parts)).encode()).hexdigest()


def _action(entries, shape, array, *, exact=False):
    array = np.asarray(array)
    if array.ndim not in (1, 2) or array.shape[0] != shape[1]:
        raise ValueError("sparse action requires a matching vector or block input axis")
    if exact:
        coefficients = np.array([_fraction(v) for v in array.flat], dtype=object).reshape(array.shape)
        result = np.full((shape[0], *array.shape[1:]), Fraction(0), dtype=object)
        for row, column, weight in entries:
            result[row] += weight * coefficients[column]
        return result
    from rexgraph.core import _sparse
    from rexgraph.native_sparse import NativeSparse
    try:
        coefficients = np.asarray(array, dtype=complex if np.iscomplexobj(array) else float)
        weights = np.array([v for _, _, v in entries], dtype=float)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError("sparse inputs are not representable numerically") from exc
    if (not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(weights))
            or np.any(weights == 0)):
        raise FloatingPointError("sparse inputs are not representable numerically")
    rows = np.array([i for i, _, _ in entries], dtype=np.intp)
    columns = np.array([j for _, j, _ in entries], dtype=np.intp)
    result = NativeSparse(_sparse.dual_from_coo(rows, columns, weights, *shape)).apply(coefficients)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("sparse action is outside numerical range")
    return result


@dataclass(frozen=True)
class CoordinateSpace:
    """Named, ordered coordinates, scoped by the accession's source and grade.

    Equal name and keys explicitly declare the same space. Equal dimension alone
    does not. Keys are unique nonempty strings, including for serialization.
    """

    name: str
    keys: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("coordinate space requires a nonempty name")
        if isinstance(self.keys, (str, bytes)):
            raise TypeError("coordinate keys require an ordered sequence, not a bare string")
        keys = tuple(self.keys)
        if any(not isinstance(k, str) or not k.strip() for k in keys):
            raise ValueError("coordinate keys must be nonempty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("coordinate keys must be unique and ordered")
        object.__setattr__(self, "keys", keys)


@dataclass(frozen=True)
class CoordinateField:
    """Visible type coefficients; deliberately not a cell Chain or Cochain."""

    source: object
    grade: int
    space: CoordinateSpace
    values: np.ndarray
    variance: str

    def __post_init__(self):
        if not isinstance(self.space, CoordinateSpace) or self.variance not in {"chain", "cochain"}:
            raise TypeError("coordinate field requires an explicit space and variance")
        if isinstance(self.grade, (bool, np.bool_)) or not isinstance(self.grade, Integral):
            raise TypeError("coordinate field grade must be an integer")
        values = np.array(self.values, copy=True)
        if values.ndim not in (1, 2) or values.shape[0] != len(self.space.keys):
            raise ValueError("coordinate field axis must match its declared space")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    def with_values(self, values):
        return replace(self, values=values)


@dataclass(frozen=True)
class TypeAccession:
    """A copied sparse measurement supplied as (row, column, value).

    Entries may mix cells and carry signed weights. They are not incidences and
    do not replace the Rex boundary. Duplicate coordinates are summed once.
    Omitted coordinates retain the original ambient endomorphism contract.
    """

    source: object
    grade: int
    name: str
    entries: tuple
    cell_keys: tuple | None = None
    coordinates: CoordinateSpace | None = None
    n_cells: int = field(init=False)
    exact: bool = field(init=False)

    def __post_init__(self):
        if isinstance(self.grade, (bool, np.bool_)) or not isinstance(self.grade, Integral):
            raise TypeError("accession grade must be an integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("accession requires a nonempty type name")
        n = cell_count(self.source, int(self.grade))
        keys = None if self.cell_keys is None else tuple(self.cell_keys)
        if keys is not None and len(keys) != n:
            raise ValueError("accession basis must match its source grade")
        if self.coordinates is not None and not isinstance(self.coordinates, CoordinateSpace):
            raise TypeError("accession coordinates require a CoordinateSpace")
        rows = n if self.coordinates is None else len(self.coordinates.keys)
        entries, exact = _entries(self.entries, (rows, n))
        object.__setattr__(self, "grade", int(self.grade))
        object.__setattr__(self, "cell_keys", keys)
        object.__setattr__(self, "n_cells", n)
        object.__setattr__(self, "entries", entries)
        # Preserve floating source arithmetic even if every entry cancels to zero.
        object.__setattr__(self, "exact", exact)

    @property
    def shape(self):
        return (self.n_cells if self.coordinates is None else len(self.coordinates.keys), self.n_cells)

    @property
    def coefficient_digest(self):
        prefix = (f"ambient-accession-v1|{self.n_cells}|{self.exact}|" if self.coordinates is None else
                  f"coordinate-accession-v1|{self.shape}|{self.exact}|")
        return _digest(self.entries, prefix)

    def check_carrier(self, value):
        value = value.cochain if isinstance(value, Field) else value
        if not isinstance(value, (Chain, Cochain)):
            raise TypeError("ACCESS requires a Chain or Cochain, not an untyped array or type view")
        keys = None if value.cell_keys is None else tuple(value.cell_keys)
        if value.source is not self.source or value.grade != self.grade or keys != self.cell_keys:
            raise ValueError("accession requires the same source, grade and ordered ambient basis")
        if cell_count(self.source, self.grade) != self.n_cells:
            raise ValueError("accession source population changed; bind a fresh map")
        if np.ndim(value.values) not in (1, 2) or value.values.shape[0] != self.n_cells:
            raise ValueError("accession requires a matching vector or block cell axis")
        return value

    def apply(self, value, *, exact=False):
        value = self.check_carrier(value)
        if exact and not self.exact:
            raise TypeError("exact ACCESS requires an integer/rational accession")
        result = _action(self.entries, self.shape, value.values, exact=exact)
        carrier = (value.with_values(result) if self.coordinates is None else
                   CoordinateField(self.source, self.grade, self.coordinates, result,
                                   "chain" if isinstance(value, Chain) else "cochain"))
        return TypeView(self, carrier)

    def check_view(self, value):
        if self.coordinates is None:
            return self.check_carrier(value)
        if (not isinstance(value, CoordinateField) or value.space != self.coordinates
                or value.source is not self.source or value.grade != self.grade):
            raise ValueError("type view requires its declared source, grade and coordinate space")
        if cell_count(self.source, self.grade) != self.n_cells:
            raise ValueError("accession source population changed; bind a fresh map")
        if np.ndim(value.values) not in (1, 2) or value.values.shape[0] != self.shape[0]:
            raise ValueError("type view requires its declared coordinate axis")
        return value


def _same_ambient(left, right):
    if (left.source is not right.source or left.grade != right.grade
            or left.cell_keys != right.cell_keys or left.n_cells != right.n_cells):
        raise ValueError("type family requires the same source, grade and ordered ambient basis")


@dataclass(frozen=True)
class AccessionFamily:
    """An ordered, source bound declaration; never a process global type registry."""

    accessions: tuple[TypeAccession, ...]

    def __post_init__(self):
        maps = tuple(self.accessions)
        if not maps or not all(isinstance(a, TypeAccession) for a in maps):
            raise TypeError("accession family requires one or more explicit TypeAccessions")
        if len({a.name for a in maps}) != len(maps):
            raise ValueError("accession family type names must be unique")
        for a in maps:
            _same_ambient(maps[0], a)
        object.__setattr__(self, "accessions", maps)

    def apply(self, value, *, exact=False):
        return TypedFamily(tuple(a.apply(value, exact=exact) for a in self.accessions))


@dataclass(frozen=True)
class TypeView:
    """Visible coefficients plus the accession that supplied their type axis."""

    accession: TypeAccession
    carrier: Chain | Cochain | CoordinateField

    def __post_init__(self):
        if not isinstance(self.accession, TypeAccession):
            raise TypeError("type view requires an explicit accession")
        carrier = self.accession.check_view(self.carrier)
        values = np.array(carrier.values, copy=True)
        values.flags.writeable = False
        object.__setattr__(self, "carrier", carrier.with_values(values))

    @property
    def values(self):
        return self.carrier.values

    @property
    def name(self):
        return self.accession.name

    @property
    def variance(self):
        return (self.carrier.variance if isinstance(self.carrier, CoordinateField) else
                "chain" if isinstance(self.carrier, Chain) else "cochain")


@dataclass(frozen=True)
class TypedFamily:
    """Ordered views from one ambient space; output coordinates may differ."""

    views: tuple[TypeView, ...]

    def __post_init__(self):
        views = tuple(self.views)
        if not views or not all(isinstance(v, TypeView) for v in views):
            raise TypeError("typed family requires one or more TypeViews")
        AccessionFamily(tuple(v.accession for v in views))
        first = views[0]
        for v in views:
            v.accession.check_view(v.carrier)
            if v.variance != first.variance or v.values.shape[1:] != first.values.shape[1:]:
                raise ValueError("typed family requires matching variance and field shapes")
        object.__setattr__(self, "views", views)

    @property
    def names(self):
        return tuple(v.name for v in self.views)

    def __getitem__(self, name):
        for view in self.views:
            if view.name == name:
                return view
        raise KeyError(name)


def _metric(view, metric):
    a = view.accession
    a.check_view(view.carrier)
    if a.coordinates is not None:
        raise TypeError("type coordinates require an explicit CrossMetric or FamilyMetric, not an ambient metric")
    if metric is None:
        return DiagonalMetric(a.source, a.grade, (Fraction(1),) * a.n_cells, a.cell_keys)
    if not isinstance(metric, DiagonalMetric):
        raise TypeError("common-ambient co relation requires an explicit DiagonalMetric")
    return metric


@dataclass(frozen=True)
class CrossMetric:
    """Explicit real sparse block M[left,right], not a positive metric claim.

    Endpoints declare coordinate spaces through source bound accessions. The
    pairing is sum_columns conjugate(x_left)^T M[left,right] x_right. Neither
    symmetry nor positivity follows from supplying a block. Transpose must be
    requested explicitly when reversing its endpoints.
    """

    left: TypeAccession
    right: TypeAccession
    entries: tuple
    exact: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.left, TypeAccession) or not isinstance(self.right, TypeAccession):
            raise TypeError("cross metric requires explicit accession endpoints")
        _same_ambient(self.left, self.right)
        entries, exact = _entries(self.entries, self.shape)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "exact", exact)

    @property
    def shape(self):
        return (self.left.shape[0], self.right.shape[0])

    @property
    def coefficient_digest(self):
        return _digest(self.entries, f"cross-metric-v1|{self.shape}|{self.exact}|")

    def transpose(self):
        result = CrossMetric(self.right, self.left, tuple((j, i, v) for i, j, v in self.entries))
        # A cancelled floating block is still a floating declaration.
        object.__setattr__(result, "exact", self.exact)
        return result

    def moment(self, left, right, *, exact=False):
        for view, endpoint in ((left, self.left), (right, self.right)):
            if not isinstance(view, TypeView):
                raise TypeError("cross metric requires TypeViews")
            _same_ambient(view.accession, endpoint)
            if view.accession.coordinates != endpoint.coordinates:
                raise ValueError("cross metric endpoint requires its declared output coordinate space")
            view.accession.check_view(view.carrier)
        if left.variance != right.variance or left.values.shape[1:] != right.values.shape[1:]:
            raise ValueError("cross metric requires matching variance and vector/block shapes")
        if exact and not self.exact:
            raise TypeError("exact contraction requires an integer/rational cross metric")
        acted = _action(self.entries, self.shape, right.values, exact=exact)
        return _diagonal_contraction(left.values, acted, (Fraction(1),)*self.shape[0], exact=exact)


@dataclass(frozen=True)
class TypeRealization:
    """Explicit sparse E_tau: V_tau -> C_k, not an inverse or an injection.

    The destination is the accession's declared ambient cell basis. Signed,
    noninjective and zero maps are valid. No chain map property is inferred.
    """

    accession: TypeAccession
    entries: tuple
    exact: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.accession, TypeAccession):
            raise TypeError("realization requires an explicit TypeAccession")
        entries, exact = _entries(self.entries, self.shape)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "exact", exact)

    @property
    def shape(self):
        return (self.accession.n_cells, self.accession.shape[0])

    @property
    def coefficient_digest(self):
        return _digest(self.entries, f"type-realization-v1|{self.shape}|{self.exact}|")

    def check_view(self, view):
        if not isinstance(view, TypeView):
            raise TypeError("realization requires a TypeView")
        _same_ambient(self.accession, view.accession)
        if self.accession.coordinates != view.accession.coordinates:
            raise ValueError("realization requires its declared input coordinate space")
        view.accession.check_view(view.carrier)

    def apply(self, view, *, exact=False):
        self.check_view(view)
        if exact and not self.exact:
            raise TypeError("exact realization requires integer/rational coefficients")
        values = _action(self.entries, self.shape, view.values, exact=exact)
        a = self.accession
        carrier = Chain if view.variance == "chain" else Cochain
        return carrier(a.grade, values, source=a.source, cell_keys=a.cell_keys)


@dataclass(frozen=True)
class FamilyMetric:
    """Coherent block form E_sigma* M E_tau, stored only as sparse factors.

    Realizations are indexed by type name and checked against their coordinate
    spaces. Subfamilies and reordered views are allowed. The direct sum form is
    PSD by factorization, not necessarily positive definite. Exact evaluation
    requires the entire supplied declaration (base and all factors) to be exact.
    """

    realizations: tuple[TypeRealization, ...]
    metric: DiagonalMetric
    _by_name: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        maps = tuple(self.realizations)
        if not maps or not all(isinstance(e, TypeRealization) for e in maps):
            raise TypeError("family metric requires one or more explicit TypeRealizations")
        if not isinstance(self.metric, DiagonalMetric):
            raise TypeError("family metric requires an explicit positive DiagonalMetric")
        AccessionFamily(tuple(e.accession for e in maps))
        a, m = maps[0].accession, self.metric
        if (m.source is not a.source or m.grade != a.grade or m.cell_keys != a.cell_keys
                or len(m.weights) != a.n_cells):
            raise ValueError("family metric requires the same ambient source, grade and ordered basis")
        object.__setattr__(self, "realizations", maps)
        object.__setattr__(self, "_by_name", MappingProxyType({e.accession.name: e for e in maps}))
        self.check_state()

    @property
    def exact(self):
        return self.metric.exact and all(e.exact for e in self.realizations)

    @property
    def shape(self):
        n = sum(e.shape[1] for e in self.realizations)
        return (n, n)

    @property
    def coefficient_digest(self):
        parts = (self.metric.coefficient_digest, *(e.coefficient_digest for e in self.realizations))
        return sha256(("factored-family-metric-v1|" + "|".join(parts)).encode()).hexdigest()

    def check_state(self, *, exact=False):
        if cell_count(self.metric.source, self.metric.grade) != len(self.metric.weights):
            raise ValueError("family metric source population changed; bind a fresh form")
        if exact and not self.exact:
            raise TypeError("exact family contraction requires an integer/rational base metric and all realizations")

    def realization_for(self, view):
        if not isinstance(view, TypeView):
            raise TypeError("family metric requires TypeViews")
        e = self._by_name.get(view.name)
        if e is None:
            raise ValueError(f"family metric has no realization for type {view.name!r}")
        e.check_view(view)
        return e

    def check_family(self, family, *, exact=False):
        if not isinstance(family, TypedFamily):
            raise TypeError("family metric requires a TypedFamily")
        self.check_state(exact=exact)
        # Revalidate visible axes without applying any realization.
        TypedFamily(family.views)
        for v in family.views:
            self.realization_for(v)

    def moment(self, left, right, *, exact=False):
        self.check_state(exact=exact)
        e_left, e_right = self.realization_for(left), self.realization_for(right)
        if left.variance != right.variance or left.values.shape[1:] != right.values.shape[1:]:
            raise ValueError("family contraction requires matching variance and vector/block shapes")
        a = e_left.apply(left, exact=exact)
        b = a if left is right else e_right.apply(right, exact=exact)
        return self.metric.moment(a, b, exact=exact)


def co_relate(left, right, metric=None, *, exact=False):
    """Ordered moment in an ambient, explicit cross or factored family form."""
    if not isinstance(left, TypeView) or not isinstance(right, TypeView):
        raise TypeError("CO_RELATE requires two TypeViews")
    _same_ambient(left.accession, right.accession)
    if isinstance(metric, (CrossMetric, FamilyMetric)):
        return metric.moment(left, right, exact=exact)
    left.accession.check_view(left.carrier)
    right.accession.check_view(right.carrier)
    if right.accession.coordinates is not None:
        raise TypeError("type coordinates require an explicit CrossMetric or FamilyMetric, not an ambient metric")
    return _metric(left, metric).moment(left.carrier, right.carrier, exact=exact)


@dataclass(frozen=True)
class TypedMomentTensor:
    """Requested type by type output, not a cell by cell native operator."""

    values: np.ndarray
    family: TypedFamily
    metric: DiagonalMetric | FamilyMetric

    def __post_init__(self):
        if not isinstance(self.family, TypedFamily) or not isinstance(self.metric, (DiagonalMetric, FamilyMetric)):
            raise TypeError("typed moment tensor requires a declared family and metric")
        n = len(self.family.views)
        values = np.array(self.values, copy=True)
        if values.shape != (n, n):
            raise ValueError("moment tensor axes must match the ordered type family")
        if isinstance(self.metric, FamilyMetric):
            self.metric.check_family(self.family)
        else:
            for view in self.family.views:
                self.metric._carrier(view.carrier)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    @property
    def names(self):
        return self.family.names


def moment_tensor(family, metric=None, *, exact=False):
    """C[s,t] = sum_columns <x_s, x_t>_M; diagonal quadrances, cross moments."""
    if not isinstance(family, TypedFamily):
        raise TypeError("MOMENT_TENSOR requires an explicit TypedFamily")
    if isinstance(metric, FamilyMetric):
        metric.check_family(family, exact=exact)
        # Each selected realization acts once, not once per type pair.
        carriers = tuple(metric.realization_for(v).apply(v, exact=exact) for v in family.views)
        base = metric.metric
    else:
        if any(v.accession.coordinates is not None for v in family.views) or isinstance(metric, CrossMetric):
            raise TypeError("coordinate MOMENT_TENSOR requires a coherent family form; independent cross blocks are not a Gram metric")
        metric = base = _metric(family.views[0], metric)
        carriers = tuple(v.carrier for v in family.views)
    n = len(family.views)
    dtype = object if exact else complex if any(np.iscomplexobj(v.values) for v in family.views) else float
    values = np.empty((n, n), dtype=dtype)
    for i, left in enumerate(carriers):
        for j in range(i, n):
            value = base.moment(left, carriers[j], exact=exact)
            if i == j and not exact:
                value = np.real(value)
            values[i, j] = value
            values[j, i] = value if exact else np.conjugate(value)
    values.flags.writeable = False
    return TypedMomentTensor(values, family, metric)
