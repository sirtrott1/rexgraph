"""RCQL operators over RexGraph values."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Any

import numpy as np
from rexgraph.cells import (
    Cell,
    CellBoundary,
    CellCoboundary,
    CellSet,
    CompositeBinary,
    boundary_of,
    coboundary_of,
    composite_binary,
    corelations,
    enclosure,
    star,
)
from rexgraph.cells import (
    cell as make_cell,
)
from rexgraph.cells import (
    cells as make_cells,
)
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graded_boundary import _sparse_rank, graded_boundaries_from_rex
from rexgraph.green import GreenOperator, vertex_green
from rexgraph.linear_operator import (
    RexOperator,
    boundary_operator,
    coboundary_operator,
    hodge_operator,
)
from rexgraph.metric_field import relation_metric_curvature
from rexgraph.rational_trig import (
    exact_character,
)
from rexgraph.rational_trig import (
    quadrance as rational_quadrance,
)
from rexgraph.rational_trig import (
    spread as rational_spread,
)
from rexgraph.temporal_signal import (
    TemporalSignal,
    TemporalSignalFlow,
    temporal_signal,
)
from rexgraph.temporal_signal import (
    signal_flow as temporal_signal_flow,
)

from .describe import describe_rex
from .types import RCType


@dataclass(frozen=True)
class Operator:
    name: str
    fn: Callable
    result_type: Callable[[tuple[Any, ...]], RCType] | None = None


_REGISTRY: dict[str, Operator] = {}


def register(name: str):
    def wrap(fn):
        _REGISTRY[name.upper()] = Operator(name.upper(), fn)
        return fn
    return wrap


def get_operator(name: str) -> Operator:
    try:
        return _REGISTRY[name.upper()]
    except KeyError as exc:
        raise KeyError(f"unknown RCQL operator {name!r}") from exc


def _cell_count(rex, grade: int) -> int:
    B = graded_boundaries_from_rex(rex)
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B] if B else [0]
    grade = int(grade)
    if grade < 0 or grade >= len(sizes):
        raise ValueError(f"grade {grade} is not present")
    return int(sizes[grade])


def _typed_cells(source, value, *, operator: str):
    """Require a direct primary-cell value from the current source complex."""
    if not isinstance(value, (Cell, CellSet)):
        raise TypeError(f"{operator} expects a Cell or CellSet")
    if value.source is not source:
        raise ValueError(f"{operator} requires a value bound to its source Rex")
    return value


def _typed_composite(source, value, *, operator: str) -> CompositeBinary:
    """Read one exact C1 composite value, retaining source and cell identity."""
    if isinstance(value, CompositeBinary):
        if value.cell.source is not source:
            raise ValueError(f"{operator} requires a value bound to its source Rex")
        return value
    value = _typed_cells(source, value, operator=operator)
    if not isinstance(value, Cell):
        raise TypeError(f"{operator} expects one grade-1 Cell or CompositeBinary")
    return composite_binary(value)


def _typed_value(source, value, *, operator: str, variance: str | None = None,
                 grade: int | None = None):
    """Require one source-bound graded value before performing mathematics.

    Arrays have no grade, variance, source-state, or basis identity.  RCQL's
    adapters therefore accept the core carriers rather than guessing from a
    matching length.  ``Field`` is a cochain reading and unwraps to its
    underlying cochain for geometry.
    """
    if isinstance(value, Field):
        value = value.cochain
    elif isinstance(value, CellBoundary):
        if value.cell.source is not source:
            raise ValueError(f"{operator} requires a value bound to its source Rex")
        if value.chain is None:
            raise ValueError(f"{operator} cannot act on an empty C0 boundary")
        value = value.chain
    elif isinstance(value, CellCoboundary):
        if value.cell.source is not source:
            raise ValueError(f"{operator} requires a value bound to its source Rex")
        value = value.cochain
    expected = {"chain": Chain, "cochain": Cochain}.get(variance)
    if expected is not None and not isinstance(value, expected):
        raise TypeError(f"{operator} expects a {variance}")
    if expected is None and not isinstance(value, (Chain, Cochain)):
        raise TypeError(f"{operator} expects a typed Chain or Cochain")
    if value.source is not source:
        raise ValueError(f"{operator} requires a value bound to its source Rex")
    if grade is not None and value.grade != int(grade):
        raise ValueError(
            f"{operator} expects grade {int(grade)}, got grade {value.grade}"
        )
    expected_cells = _cell_count(source, value.grade)
    if value.n_cells != expected_cells:
        raise ValueError(
            f"{operator} expects {expected_cells} cells at grade {value.grade}, "
            f"got {value.n_cells}"
        )
    return value


def _same_space(source, left, right, *, operator: str):
    """Require two values to name the same typed mathematical space."""
    left = _typed_value(source, left, operator=operator)
    right = _typed_value(source, right, operator=operator)
    if type(left) is not type(right):
        raise TypeError(f"{operator} requires matching chain/cochain variance")
    if left.grade != right.grade:
        raise ValueError(f"{operator} requires matching grades")
    if left.cell_keys != right.cell_keys:
        raise ValueError(f"{operator} requires the same ordered basis")
    if left.values.shape != right.values.shape:
        raise ValueError(f"{operator} requires matching value shapes")
    return left, right


def _typed_temporal_signal(source, value, *, operator: str) -> TemporalSignal:
    """Require one delta field from the same bound TemporalRex source."""
    if not isinstance(value, TemporalSignal):
        raise TypeError(f"{operator} expects a TemporalSignal")
    if value.source is not source:
        raise ValueError(f"{operator} requires a temporal signal bound to its source")
    return value


def _exact_coefficients(values) -> tuple[Fraction, ...] | None:
    """Return a certified exact C1/C0 coefficient vector, or decline the fast path.

    The numerical sparse operator is still the right adapter for measured floats and
    block fields.  An integral or Fraction-valued one-dimensional carrier, however,
    has enough information to apply the declared C1 columns directly and must not be
    rounded through B1 merely because a sparse work matrix happens to be float-backed.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        return None
    exact: list[Fraction] = []
    for value in array:
        if isinstance(value, bool):
            return None
        if isinstance(value, Fraction):
            exact.append(value)
        elif isinstance(value, Integral):
            exact.append(Fraction(int(value)))
        else:
            return None
    return tuple(exact)


def _exact_c1_boundary_action(source, coefficients: tuple[Fraction, ...]) -> Chain:
    """Apply declared C1 relation boundaries without materializing float B1."""
    values = [Fraction(0) for _ in range(_cell_count(source, 0))]
    for coefficient, support in zip(coefficients, source.relation_supports(), strict=True):
        arity = len(support)
        if arity == 1:
            values[int(support[0])] += coefficient
        elif arity:
            values[int(support[0])] -= coefficient
            share = coefficient / (arity - 1)
            for vertex in support[1:]:
                values[int(vertex)] += share
    return Chain(0, np.asarray(values, dtype=object), source=source)


def _exact_c0_coboundary_action(source, coefficients: tuple[Fraction, ...]) -> Cochain:
    """Apply the declared C1 transpose directly to an exact C0 cochain."""
    values: list[Fraction] = []
    for support in source.relation_supports():
        arity = len(support)
        if arity == 1:
            values.append(coefficients[int(support[0])])
        elif arity:
            share = sum((coefficients[int(vertex)] for vertex in support[1:]), Fraction(0))
            values.append(-coefficients[int(support[0])] + share / (arity - 1))
        else:
            values.append(Fraction(0))
    return Cochain(1, np.asarray(values, dtype=object), source=source)


def _quadrance(raw, *, exact: bool):
    """Apply rational quadrance columnwise while preserving a block cell axis."""
    values = np.asarray(raw)
    if values.ndim == 1:
        return rational_quadrance(values, exact=exact)
    if values.ndim == 2:
        if exact:
            return np.asarray(
                [rational_quadrance(values[:, column], exact=True)
                 for column in range(values.shape[1])],
                dtype=object,
            )
        return np.einsum("ij,ij->j", values.conj(), values).real
    raise ValueError("QUADRANCE expects a vector or a two-dimensional block")


@register("REX")
def rex_source(source, name):
    return name


@register("GRADE")
def grade(source, value=None):
    if value is None:
        B = graded_boundaries_from_rex(source)
        return len(B)
    if isinstance(value, CellBoundary) and value.grade is None:
        raise ValueError("the boundary of a C0 cell has no lower grade")
    return int(value.grade)


@register("BOUNDARY")
def boundary(source, grade, values=None):
    if isinstance(grade, (Cell, CellSet)):
        if values is not None:
            raise TypeError("BOUNDARY accepts either a cell value or grade plus Chain")
        return boundary_of(_typed_cells(source, grade, operator="BOUNDARY"))
    op = boundary_operator(source, int(grade))
    if values is None:
        return op
    chain = _typed_value(source, values, operator="BOUNDARY", variance="chain",
                         grade=int(grade))
    exact = _exact_coefficients(chain.values)
    if int(grade) == 1 and exact is not None:
        return _exact_c1_boundary_action(source, exact)
    out = op.apply(chain.values)
    return Chain(int(grade) - 1, out, source=source)


@register("DESCRIBE")
def describe(source):
    return describe_rex(source)


@register("COBOUNDARY")
def coboundary(source, grade, values=None):
    if isinstance(grade, (Cell, CellSet)):
        if values is not None:
            raise TypeError("COBOUNDARY accepts either a cell value or grade plus Cochain")
        return coboundary_of(_typed_cells(source, grade, operator="COBOUNDARY"))
    op = coboundary_operator(source, int(grade))
    if values is None:
        return op
    cochain = _typed_value(source, values, operator="COBOUNDARY", variance="cochain",
                           grade=int(grade))
    exact = _exact_coefficients(cochain.values)
    if int(grade) == 0 and exact is not None:
        return _exact_c0_coboundary_action(source, exact)
    out = op.apply(cochain.values)
    return Cochain(int(grade) + 1, out, source=source)


@register("CELL")
def cell_at(source, grade, index):
    """Address one source-bound carried cell in the graded complex."""
    return make_cell(source, grade, index)


@register("CELLS")
def cells_at(source, grade, indices=None):
    """Address all or selected carried cells at a single grade."""
    return make_cells(source, grade, indices)


@register("INDICATOR")
def indicator(source, value):
    """Materialize the exact 0/1 cochain for one selected primary cell pattern.

    The Cell/CellSet remains an addressing object; this explicit constructor is
    the bridge to a coefficient field for a Green, Hodge, or metric action. A
    lookup of the selected cell is local, while materializing its full graded
    basis necessarily costs O(number of cells at that grade).
    """
    value = _typed_cells(source, value, operator="INDICATOR")
    coefficients = np.zeros(_cell_count(source, value.grade), dtype=np.int64)
    if isinstance(value, Cell):
        coefficients[value.index] = 1
    else:
        coefficients[list(value.indices)] = 1
    return Cochain(value.grade, coefficients, source=source)


@register("COMPOSITE")
def composite(source, value):
    """Read exact C1 existence, orientation/head, and share binary tensors."""
    value = _typed_cells(source, value, operator="COMPOSITE")
    if not isinstance(value, Cell):
        raise TypeError("COMPOSITE expects one Cell")
    return composite_binary(value)


@register("EXISTENCE")
def existence(source, value):
    """Return the exact 0/1 C0 existence mask of one C1 relation."""
    return _typed_composite(source, value, operator="EXISTENCE").existence


@register("HEAD")
def head(source, value):
    """Return the exact 0/1 distinguished-head mask of one C1 relation."""
    return _typed_composite(source, value, operator="HEAD").head


@register("SHARE")
def share(source, value):
    """Return the exact rational C0 share vector of one C1 relation."""
    return _typed_composite(source, value, operator="SHARE").share


@register("SHARE_SUPPORT")
def share_support(source, value):
    """Return the exact 0/1 C0 share-support mask of one C1 relation."""
    return _typed_composite(source, value, operator="SHARE_SUPPORT").share_support


@register("ARITY")
def arity(source, value):
    """Return the declared incidence arity of one C1 relation."""
    return _typed_composite(source, value, operator="ARITY").arity


@register("CORELATIONS")
def co_relations(source, value):
    """Read direct co-relations without a graph projection or clique expansion."""
    return corelations(_typed_cells(source, value, operator="CORELATIONS"))


@register("STAR")
def graded_star(source, value):
    """Return the upward graded closure of cells under direct co-relation."""
    return star(_typed_cells(source, value, operator="STAR"))


@register("ENCLOSURE")
def graded_enclosure(source, value):
    """Return the full source-bound graded enclosure of a cell pattern."""
    return enclosure(_typed_cells(source, value, operator="ENCLOSURE"))


@register("HODGE_OPERATOR")
def hodge_op(source, grade, alpha=1):
    return hodge_operator(source, int(grade), alpha=alpha)


@register("RANK")
def rank(source, value):
    if isinstance(value, RexOperator):
        if value.source is not source:
            raise ValueError("RANK requires an operator bound to its source Rex")
        return int(_sparse_rank(value.as_scipy()))
    if isinstance(value, (int, np.integer)):
        return int(_sparse_rank(boundary_operator(source, int(value)).as_scipy()))
    raise TypeError("RANK expects a boundary operator or grade")


@register("NULLITY")
def nullity(source, value):
    if isinstance(value, RexOperator):
        if value.source is not source:
            raise ValueError("NULLITY requires an operator bound to its source Rex")
        return int(value.shape[1] - _sparse_rank(value.as_scipy()))
    grade = int(value)
    op = boundary_operator(source, grade)
    return int(op.shape[1] - _sparse_rank(op.as_scipy()))


@register("BETTI")
def betti(source, grade):
    return int(source.betti[int(grade)])


@register("HODGE")
def hodge(source, flow):
    cochain = _typed_value(source, flow, operator="HODGE", variance="cochain", grade=1)
    grad, curl, harm = source.hodge(np.ascontiguousarray(cochain.values, dtype=np.float64))
    return {
        "gradient": Cochain(1, grad, source=source),
        "curl": Cochain(1, curl, source=source),
        "harmonic": Cochain(1, harm, source=source),
    }


@register("HARMONIC")
def harmonic(source, flow):
    return hodge(source, flow)["harmonic"]


@register("GREEN")
def green(source, values=None):
    action = vertex_green(source)
    if values is None:
        return action
    return apply(source, action, values)


@register("APPLY")
def apply(source, action, values):
    """Apply a declared Green or grade-preserving Rex action to a typed cochain."""
    if isinstance(action, GreenOperator):
        operator = action.operator
        solve = action.solve
        kind = action.kind
    elif isinstance(action, RexOperator):
        operator = action
        solve = action.apply
        kind = action.name
    else:
        raise TypeError("APPLY expects a GreenOperator or RexOperator action")
    if operator.source is not source:
        raise ValueError("APPLY requires an action bound to its source Rex")
    if operator.domain_grade != operator.codomain_grade:
        raise TypeError(
            "APPLY accepts only grade-preserving RexOperator actions; use BOUNDARY or "
            "COBOUNDARY for a graded map"
        )
    cochain = _typed_value(
        source,
        values,
        operator="APPLY",
        variance="cochain",
        grade=operator.domain_grade,
    )
    out = solve(np.asarray(cochain.values, dtype=np.float64))
    field = Cochain(operator.codomain_grade, out, source=source)
    return Field(field, action, kind=kind)


@register("QUADRANCE")
def quadrance(source, values, exact=False):
    value = _typed_value(source, values, operator="QUADRANCE")
    return _quadrance(value.values, exact=bool(exact))


@register("SPREAD")
def spread(source, left, right, exact=False):
    left, right = _same_space(source, left, right, operator="SPREAD")
    a, b = np.asarray(left.values), np.asarray(right.values)
    if a.ndim != 1:
        raise ValueError("SPREAD currently expects one-dimensional typed values")
    return rational_spread(a, b, exact=bool(exact))


@register("ACCUMULATE")
def accumulate(source, left, right):
    """Add two aligned graded coefficient fields without discarding their carrier.

    This is a tensor accumulation, not a path count. The values must already
    inhabit the same source-bound ordered basis and temporal state; time or
    basis transport has to be a separately declared action rather than an
    implicit length-based merge.
    """
    left, right = _same_space(source, left, right, operator="ACCUMULATE")
    return left.with_values(np.asarray(left.values) + np.asarray(right.values))


@register("HODGE_COORDS")
def hodge_coordinates(source, flow):
    from rexgraph.hodge_coords import hodge_coords

    cochain = _typed_value(
        source, flow, operator="HODGE_COORDS", variance="cochain", grade=1
    )
    return hodge_coords(source, cochain.values)


@register("WINDING")
def winding(source, flow):
    from rexgraph.harmonic_sparse import harmonic_winding
    from rexgraph.hodge_coords import harmonic_frame

    cochain = _typed_value(source, flow, operator="WINDING", variance="cochain", grade=1)
    return harmonic_winding(harmonic_frame(source), cochain.values)


@register("CLOSURE")
def closure(source, seed, max_depth=8, grade=0):
    from rexgraph.tower import semantic_closure

    if int(grade) != 0:
        raise NotImplementedError(
            "CLOSURE currently implements only grade-0 vertex seeds; "
            "other grades must refuse until their incidence semantics exist"
    )
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("CLOSURE currently expects an integer grade-0 cell index")
    seed = int(seed)
    if seed < 0 or seed >= _cell_count(source, 0):
        raise ValueError(f"CLOSURE seed index {seed} is not present at grade 0")
    return semantic_closure(source, seed, max_depth=int(max_depth), grade=0)


@register("SIGNIFICANCE")
def significance(source, edge):
    if isinstance(edge, (bool, np.bool_)) or not isinstance(edge, (int, np.integer)):
        raise TypeError("SIGNIFICANCE currently expects an integer grade-1 cell index")
    idx = int(edge)
    if idx < 0 or idx >= _cell_count(source, 1):
        raise ValueError(f"SIGNIFICANCE edge index {idx} is not present")
    from rexgraph.semantic import significance as _significance
    return float(_significance(source, [idx])[0])


@register("CHARACTER")
def character(source, exact=False):
    if not exact:
        return np.asarray(source.structural_character)
    values, channels = exact_character(source)
    return {
        "values": np.asarray(values if values is not None else (), dtype=object),
        "channels": tuple(channels),
        "exactness": "rational",
    }


@register("ZERO")
def zero(source, grade, kind="cochain"):
    grade = int(grade)
    size = _cell_count(source, grade)
    if str(kind).lower() == "cochain":
        return Cochain(grade, np.zeros(size, dtype=np.int64), source=source)
    if str(kind).lower() == "chain":
        return Chain(grade, np.zeros(size, dtype=np.int64), source=source)
    raise ValueError("ZERO kind must be 'chain' or 'cochain'")


@register("TEMPORAL_DELTA")
def temporal_delta(source, step):
    """Read one exact C1 temporal delta field from a TemporalRex source.

    The returned value keeps existence, orientation, signing, exact head identity,
    and separately declared amplitude rather than collapsing the transition to a
    table diff or vertex-path request.
    """
    return temporal_signal(source, int(step))


@register("SIGNAL_AT")
def signal_at(source, signal, key):
    """Read one changed C1 relation by its exact support identity."""
    return _typed_temporal_signal(source, signal, operator="SIGNAL_AT").event(key)


@register("SIGNAL_SOURCE")
def signal_source(source, signal, channel="structural"):
    """Materialize a typed C0 source field from one temporal delta channel."""
    return _typed_temporal_signal(source, signal, operator="SIGNAL_SOURCE").source_field(channel)


@register("RELATION_SIGNAL")
def relation_signal(source, signal, channel="amplitude"):
    """Read the direct current-C1 temporal field for a named delta channel."""
    return _typed_temporal_signal(source, signal, operator="RELATION_SIGNAL").relation_field(channel)


@register("SIGNAL_FLOW")
def signal_field_flow(source, signal, channel="structural") -> TemporalSignalFlow:
    """Apply the local graded B1* then B1 response to a temporal source field."""
    return temporal_signal_flow(
        _typed_temporal_signal(source, signal, operator="SIGNAL_FLOW"), channel
    )


@register("SIGNAL_HODGE")
def signal_hodge(source, signal, channel="amplitude"):
    """Split a direct temporal C1 field on its current relational-complex basis.

    The action is numerical because the current Hodge adapter is numerical.  It
    is deliberately separate from the exact delta carrier and from SIGNAL_FLOW:
    the latter starts from a C0 boundary source and is therefore gradient by
    construction, whereas this operation may expose curl or harmonic content.
    """
    signal = _typed_temporal_signal(source, signal, operator="SIGNAL_HODGE")
    field = signal.relation_field(channel)
    gradient, curl, harmonic = signal.current.hodge(
        np.ascontiguousarray(field.values, dtype=np.float64)
    )
    return {
        "gradient": Cochain(1, gradient, cell_keys=field.cell_keys, source=signal.current),
        "curl": Cochain(1, curl, cell_keys=field.cell_keys, source=signal.current),
        "harmonic": Cochain(1, harmonic, cell_keys=field.cell_keys, source=signal.current),
    }


@register("METRIC_CURVATURE")
def metric_curvature(source, metric):
    """Read C0 strain and C1 contributions from a direct C1 metric field.

    This is metric curvature over declared relation boundaries.  It retains
    branching shares and repeated incidence, rather than delegating to a
    pairwise source/target projection.
    """
    metric = _typed_value(source, metric, operator="METRIC_CURVATURE",
                          variance="cochain", grade=1)
    return relation_metric_curvature(source, metric)


@register("FILES")
def files(source, limit=100, offset=0):
    """Return a bounded slice of one file catalog."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("FILES expects a FileCatalog source")
    return source.list(limit=int(limit), offset=int(offset))


@register("SEARCH")
def search(source, text, limit=100):
    """Search one file catalog using literal terms."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("SEARCH expects a FileCatalog source")
    return source.search(str(text), limit=int(limit))


@register("FILE_INFO")
def file_info(source, name):
    """Return bounded metadata for one catalog entry."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("FILE_INFO expects a FileCatalog source")
    return source.info(str(name))


@register("FILE_HASH")
def file_hash(source, name):
    """Return the current sha256 for one catalog entry."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("FILE_HASH expects a FileCatalog source")
    return source.hash(str(name))


@register("HASH_FILES")
def hash_files(source):
    """Hash every entry in one file catalog."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("HASH_FILES expects a FileCatalog source")
    return source.hash_all()


def _record_view(record):
    """Return the structural part of one RCDB record."""
    return {
        "id": str(record.id),
        "version": int(record.version),
        "created": float(record.created),
        "tx_from": float(record.tx_from),
        "tx_to": None if record.tx_to is None else float(record.tx_to),
        "valid_from": None if record.valid_from is None else float(record.valid_from),
        "valid_to": None if record.valid_to is None else float(record.valid_to),
        "signature": dict(record.signature),
    }


@register("RCDB_LIST")
def rcdb_list(source, limit=100, offset=0):
    """Return bounded structural record summaries from an RCDB source."""
    if not hasattr(source, "list") or not hasattr(source, "get"):
        raise TypeError("RCDB_LIST expects an RCDB store")
    rows = source.list(limit=min(1000, max(1, int(limit))), offset=max(0, int(offset)))
    return [_record_view(row) for row in rows]


@register("RCDB_SEARCH")
def rcdb_search(source, text, limit=100):
    """Search an RCDB vocabulary using literal terms."""
    if not hasattr(source, "query"):
        raise TypeError("RCDB_SEARCH expects an RCDB store")
    labels = [term for term in str(text).split() if term]
    if not labels:
        return rcdb_list(source, limit=limit)
    rows = source.query(limit=min(1000, max(1, int(limit))), labels_any=labels)
    return [_record_view(row) for row in rows]


@register("RCDB_GET")
def rcdb_get(source, record_id):
    """Load one Rex payload from an RCDB source by exact id."""
    if not hasattr(source, "get"):
        raise TypeError("RCDB_GET expects an RCDB store")
    return source.get(str(record_id))


@register("RCDB_HISTORY")
def rcdb_history(source, record_id):
    """Return bounded structural version summaries for one RCDB record."""
    if not hasattr(source, "history"):
        raise TypeError("RCDB_HISTORY expects an RCDB store")
    return [_record_view(row) for row in source.history(str(record_id))[:1000]]


@register("RCDB_STATS")
def rcdb_stats(source):
    """Return store statistics without backend paths."""
    if not hasattr(source, "stats"):
        raise TypeError("RCDB_STATS expects an RCDB store")
    stats = dict(source.stats())
    stats.pop("root", None)
    stats.pop("path", None)
    stats.pop("uri", None)
    return stats


@register("TENSORS")
def tensors(source, name, limit=1000):
    """Return bounded tensor metadata for one cataloged safetensors file."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("TENSORS expects a FileCatalog source")
    return source.tensors(str(name), limit=int(limit))


@register("SEARCH_TENSORS")
def search_tensors(source, name, text, limit=100):
    """Search tensor names inside one cataloged safetensors file."""
    from rexgraph.io.catalog import FileCatalog
    if not isinstance(source, FileCatalog):
        raise TypeError("SEARCH_TENSORS expects a FileCatalog source")
    return source.search_tensors(str(name), str(text), limit=int(limit))


@register("STATE_HASH")
def state_hash(source):
    """Return the canonical tensor state digest of a Rex source."""
    from rexgraph.io.catalog import object_digest
    return object_digest(source)


@register("RCDB_HASH")
def rcdb_hash(source, record_id):
    """Return the canonical state digest of one RCDB record."""
    if not hasattr(source, "get"):
        raise TypeError("RCDB_HASH expects an RCDB store")
    from rexgraph.io.catalog import object_digest
    return object_digest(source.get(str(record_id)))

@register("RCDB_COMMITS")
def rcdb_commits(source, record_id, limit=1000):
    """Return bounded structural mutation lineage without raw delta tensors."""
    if not hasattr(source, "commit_history"):
        raise TypeError("RCDB_COMMITS expects an RCDB store")
    out = []
    for package in source.commit_history(str(record_id))[:min(1000, max(1, int(limit)))]:
        out.append({
            "digest": package.digest,
            "link": package.link.digest,
            "parent": package.link.parent_digest,
            "transition": package.transition.digest,
            "previous_state": package.transition.previous_state,
            "resulting_state": package.transition.resulting_state,
            "delta_state": package.transition.delta_state,
            "tx_time": float(package.transition.tx_time),
            "actor": package.transition.actor,
            "policy": package.transition.policy,
            "transition_signer": package.transition.signer_id,
            "lineage_signer": package.link.signer_id,
        })
    return out


@register("RCDB_VERIFY")
def rcdb_verify(source, record_id):
    """Verify the persisted mutation lineage for one RCDB record."""
    if not hasattr(source, "verify_commits"):
        raise TypeError("RCDB_VERIFY expects an RCDB store")
    return bool(source.verify_commits(str(record_id)))

@register("RCDB_STATE_HASH")
def rcdb_state_hash(source):
    """Return the canonical logical state digest of one RCDB store."""
    if not hasattr(source, "state_digest"):
        raise TypeError("RCDB_STATE_HASH expects an RCDB store")
    return source.state_digest()


@register("RCDB_SECURITY")
def rcdb_security(source):
    """Return bounded RCDB security configuration without secrets or paths."""
    if not hasattr(source, "security_status"):
        raise TypeError("RCDB_SECURITY expects an RCDB store")
    return dict(source.security_status())
