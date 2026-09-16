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
    cell_count,
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
from rexgraph.channel_operator import ChannelOperator
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graded_boundary import _sparse_rank
from rexgraph.green import GreenOperator, vertex_green
from rexgraph.linear_operator import (
    MetricAdjointOperator,
    RexOperator,
    boundary_operator,
    coboundary_operator,
    hodge_operator,
    metric_adjoint,
)
from rexgraph.metric_field import relation_metric_curvature
from rexgraph.operator_bracket import GradedOperatorBracket, OperatorBracket, operator_bracket
from rexgraph.rational_trig import (
    exact_character,
)
from rexgraph.rational_trig import (
    quadrance as rational_quadrance,
)
from rexgraph.rational_trig import (
    spread as rational_spread,
)
from rexgraph.sheaf import ExactSheaf
from rexgraph.temporal_signal import (
    TemporalSignal,
    TemporalSignalFlow,
    temporal_signal,
)
from rexgraph.temporal_signal import (
    signal_flow as temporal_signal_flow,
)
from rexgraph.weighted_dirac import GradedChain, WeightedDiracOperator, weighted_dirac
from rexgraph.weighted_hodge import WeightedHodgeOperator, weighted_hodge

from .describe import describe_rex
from .execution_trace import record_method
from .types import RCType


@dataclass(frozen=True)
class Operator:
    name: str
    fn: Callable
    result_type: Callable[[tuple[Any, ...]], RCType] | None = None


_REGISTRY: dict[str, Operator] = {}


def register(name: str):
    from .names import canonical_name, insert_unique
    name = canonical_name(name)
    def wrap(fn):
        insert_unique(_REGISTRY, name, Operator(name, fn))
        return fn
    return wrap


def get_operator(name: str) -> Operator:
    from .names import canonical_name
    name = canonical_name(name)
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown RCQL operator {name!r}") from exc


@register("SHOW_OPERATORS")
def show_operators(source, limit=1000, offset=0):
    """Bounded, static global contracts; never inspect a source or invoke a kernel."""
    from .inventory import operator_inventory

    return operator_inventory(limit=limit, offset=offset)


@register("COUNT")
def count_values(source, values):
    if isinstance(values, CellSet):
        values = _typed_cells(source, values, operator="COUNT").indices
    if not isinstance(values, (tuple, list)):
        raise TypeError("COUNT requires a finite sequence or native cell selection")
    record_method("sequence-count", count=len(values))
    return len(values)


@register("SUM")
def sum_values(source, values):
    from .aggregates import reduce_scalars
    result = reduce_scalars(values)
    record_method("scalar-sequence-sum", count=len(values))
    return result


@register("MEAN")
def mean_values(source, values):
    from .aggregates import reduce_scalars
    result = reduce_scalars(values, mean=True)
    record_method("scalar-sequence-mean", count=len(values))
    return result


def _cell_count(rex, grade: int) -> int:
    from rexgraph.cells import cell_count
    return cell_count(rex, grade)


def _typed_cells(source, value, *, operator: str):
    """Require a direct primary cell value from the current source complex."""
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
                 grade: int | None = None, allow_empty_upper: bool = False):
    """Require one source bound graded value before performing mathematics.

    Arrays have no grade, variance, source state, or basis identity.  RCQL's
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
    expected_cells = (cell_count(source, value.grade, allow_empty_upper=True) if allow_empty_upper
                      else _cell_count(source, value.grade))
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
    block fields.  An integral or Fraction valued one dimensional carrier, however,
    has enough information to apply the declared C1 columns directly and must not be
    rounded through B1 merely because a sparse work matrix happens to be float backed.
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


def _require_exact_coefficients(values):
    if _exact_coefficients(np.asarray(values).ravel()) is None:
        raise TypeError("exact=True requires an integer or rational carrier")


def _exact_boundary_action(source, op, values, *, transpose=False):
    """Use the core's certified incidence action, also shared by metric adjoints."""
    array = np.asarray(values)
    if array.ndim not in (1, 2) or _exact_coefficients(array.ravel()) is None:
        return None
    if op.exact_matvec is None:
        raise ValueError("exact boundary action requires integer higher boundary coefficients")
    return op.apply(array, exact=True)


def _grade_argument(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError('grade must be an integer')
    return int(value)


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
        return np.asarray([rational_quadrance(values[:, j]) for j in range(values.shape[1])])
    raise ValueError("QUADRANCE expects a vector or a two-dimensional block")


@register("REX")
def rex_source(source, name):
    return name


@register("GRADE")
def grade(source, value=None):
    if value is None:
        from rexgraph.native_sparse import native_boundaries
        B = native_boundaries(source)
        return len(B)
    if isinstance(value, CellBoundary) and value.grade is None:
        raise ValueError("the boundary of a C0 cell has no lower grade")
    if isinstance(value, (GradedChain, WeightedDiracOperator)):
        raise TypeError("a direct-sum value has no single grade; select GRADE_COMPONENT first")
    return int(value.grade)


@register("BOUNDARY")
def boundary(source, grade, values=None):
    if isinstance(grade, (Cell, CellSet)):
        if values is not None:
            raise TypeError("BOUNDARY accepts either a cell value or grade plus Chain")
        return boundary_of(_typed_cells(source, grade, operator="BOUNDARY"))
    grade = _grade_argument(grade)
    op = boundary_operator(source, grade)
    if values is None:
        record_method("sparse-operator-handle", construction="boundary", grade=int(grade))
        return op
    chain = _typed_value(source, values, operator="BOUNDARY", variance="chain",
                         grade=int(grade))
    if chain.cell_keys is not None:
        raise TypeError('BOUNDARY requires the canonical ordered basis (cell_keys=None)')
    exact = _exact_boundary_action(source, op, chain.values)
    if exact is not None:
        record_method("exact-incidence-action", direction="boundary", grade=int(grade))
        return Chain(int(grade) - 1, exact, source=source)
    out = op.apply(chain.values)
    record_method("sparse-boundary-action", direction="boundary", grade=int(grade))
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
    grade = _grade_argument(grade)
    op = coboundary_operator(source, grade)
    if values is None:
        record_method("sparse-operator-handle", construction="coboundary", grade=int(grade))
        return op
    cochain = _typed_value(source, values, operator="COBOUNDARY", variance="cochain",
                           grade=int(grade))
    if cochain.cell_keys is not None:
        raise TypeError('COBOUNDARY requires the canonical ordered basis (cell_keys=None)')
    exact = _exact_boundary_action(source, op, cochain.values, transpose=True)
    if exact is not None:
        record_method("exact-incidence-action", direction="coboundary", grade=int(grade))
        return Cochain(int(grade) + 1, exact, source=source)
    out = op.apply(cochain.values)
    record_method("sparse-boundary-action", direction="coboundary", grade=int(grade))
    return Cochain(int(grade) + 1, out, source=source)


@register("CELL")
def cell_at(source, grade, index):
    """Address one source bound carried cell in the graded complex."""
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
    """Return the exact 0/1 distinguished head mask of one C1 relation."""
    return _typed_composite(source, value, operator="HEAD").head


@register("SHARE")
def share(source, value):
    """Return the exact rational C0 share vector of one C1 relation."""
    return _typed_composite(source, value, operator="SHARE").share


@register("SHARE_SUPPORT")
def share_support(source, value):
    """Return the exact 0/1 C0 share support mask of one C1 relation."""
    return _typed_composite(source, value, operator="SHARE_SUPPORT").share_support


@register("ARITY")
def arity(source, value):
    """Return the declared incidence arity of one C1 relation."""
    return _typed_composite(source, value, operator="ARITY").arity


@register("CORELATIONS")
def co_relations(source, value):
    """Read direct co relations without a graph projection or clique expansion."""
    return corelations(_typed_cells(source, value, operator="CORELATIONS"))


@register("STAR")
def graded_star(source, value):
    """Return the upward graded closure of cells under direct co relation."""
    return star(_typed_cells(source, value, operator="STAR"))


@register("ENCLOSURE")
def graded_enclosure(source, value):
    """Return the full source bound graded enclosure of a cell pattern."""
    return enclosure(_typed_cells(source, value, operator="ENCLOSURE"))


@register("GLUE")
def glue(source, section):
    """Evaluate a declared exact local section over this phrase's relational complex.

    The sheaf already contains the selected stalks and their incidence restrictions.
    This adapter only establishes that they are sections of the phrase source, then
    compares transported values at every shared mediator. Cross state phrase
    restrictions must be explicit, but they are maps of section coordinates:
    this operation does NOT verify a full graded chain map or merge complexes.
    """
    if not isinstance(section, ExactSheaf):
        raise TypeError("GLUE expects an ExactSheaf local-section carrier")
    if section.rex is not source:
        raise ValueError("GLUE requires an ExactSheaf bound to its phrase source Rex")
    record_method("exact-all-pair-section-gluing")
    return section.glue()


@register("SECTION_CHECK")
def section_check(source, section):
    """Exact incidence compatibility without constructing the meeting pair graph."""
    if not isinstance(section, ExactSheaf):
        raise TypeError("SECTION_CHECK expects an ExactSheaf local-section carrier")
    if section.rex is not source:
        raise ValueError("SECTION_CHECK requires an ExactSheaf bound to its phrase source Rex")
    record_method("exact-incidence-section-check")
    return section.check_section()


@register("HODGE_OPERATOR")
def hodge_op(source, grade, alpha=1):
    record_method("factored-hodge-handle", grade=int(grade), alpha=alpha)
    return hodge_operator(source, int(grade), alpha=alpha)


def _weighted_hodge(source, grade, sector, metric, lower_metric=None, upper_metric=None):
    result = weighted_hodge(source, grade, sector=sector, metric=metric,
                            lower_metric=lower_metric, upper_metric=upper_metric)
    record_method("factored-weighted-hodge-handle", grade=result.domain_grade,
                  sector=sector, active_sectors=result.active_sectors,
                  exact_action=result.exact_matvec is not None,
                  metric_self_adjoint=True, metric_psd=result.metric_psd)
    return result


@register("HODGE_DOWN")
def hodge_down(source, grade, metric=None, lower_metric=None):
    return _weighted_hodge(source, grade, "down", metric, lower_metric)


@register("HODGE_UP")
def hodge_up(source, grade, metric=None, upper_metric=None):
    return _weighted_hodge(source, grade, "up", metric, upper_metric=upper_metric)


@register("HODGE_SUM")
def hodge_sum(source, grade, metric=None, lower_metric=None, upper_metric=None):
    return _weighted_hodge(source, grade, "sum", metric, lower_metric, upper_metric)


@register("HODGE_DIFFERENCE")
def hodge_difference(source, grade, metric=None, lower_metric=None, upper_metric=None):
    return _weighted_hodge(source, grade, "difference", metric, lower_metric, upper_metric)


@register("DIRAC")
def dirac(source, metrics=None):
    result = weighted_dirac(source, metrics=metrics)
    record_method("factored-weighted-dirac-handle", anti=False)
    return result


@register("ANTI_DIRAC")
def anti_dirac(source, metrics=None):
    result = weighted_dirac(source, metrics=metrics, anti=True)
    record_method("factored-weighted-dirac-handle", anti=True)
    return result


@register("GRADED_CHAIN")
def graded_chain(source, components):
    result = GradedChain(source, components)
    record_method("graded-chain-construction")
    return result


@register("GRADE_COMPONENT")
def grade_component(source, state, grade):
    if not isinstance(state, GradedChain) or state.source is not source:
        raise TypeError("GRADE_COMPONENT requires a GradedChain bound to the source")
    record_method("graded-chain-component")
    return state.component(grade)


@register("RANK")
def rank(source, value):
    return _rank_reading(source, value)[0]


def _rank_reading(source, value):
    from rexgraph.graded_boundary import _rank_integer_columns
    from rexgraph.native_rank import boundary_columns
    if isinstance(value, RexOperator):
        if value.source is not source:
            raise ValueError("rank/nullity requires an operator bound to its source Rex")
        if value.exact_rank_factory is not None:
            result, method = value.exact_rank_factory()
            record_method("native-exact-rank", algorithm=method)
        else:
            # Compatibility for explicitly materializable external operators. No
            # basis probing or rank inference from the handle's name/source.
            result = _sparse_rank(value.as_scipy(), exact=True)
            record_method("explicit-matrix-exact-rank", materializes_sparse=True)
        return int(result), value.shape[1]
    elif isinstance(value, (int, np.integer)):
        grade = _grade_argument(value)
        shape, columns = boundary_columns(source, grade, integer=True)
        result, method = _rank_integer_columns(shape, columns)
        record_method("native-exact-rank", algorithm=method, grade=grade)
        return result, shape[1]
    raise TypeError("RANK expects a boundary operator or grade")


@register("NULLITY")
def nullity(source, value):
    result, domain_size = _rank_reading(source, value)
    return domain_size - result


@register("BETTI")
def betti(source, grade):
    grade = _grade_argument(grade)
    cached = "_betti_tower_reading" in source.__dict__
    values, methods = source._betti_tower_reading
    if not 0 <= grade < len(values):
        raise ValueError(f"grade {grade} is not present")
    record_method("native-exact-betti", cache_hit=cached,
                  algorithms=() if cached else methods, chain_condition="exact-zero")
    return values[grade]


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
def apply(source, action, values, exact=False):
    """Legacy cochain actions or explicitly typed adjoint/weighted Hodge actions."""
    if not isinstance(exact, (bool, np.bool_)):
        raise TypeError("exact must be a boolean")
    from .calculus_operators import ComposedAction, apply_composition
    if isinstance(action, ComposedAction):
        return apply_composition(source, action, values, exact)
    from rexgraph.column_expansion import PrimaryBoundary
    if isinstance(action, PrimaryBoundary):
        if action.source is not source:
            raise ValueError("primary boundary requires its bound source")
        carrier = _typed_value(source, values, operator="PRIMARY_LIFT", variance="chain", grade=action.domain_grade)
        if carrier.cell_keys is not None:
            raise ValueError("primary boundary requires the canonical ordered basis")
        out = action.apply(np.asarray(carrier.values), exact=exact)
        record_method("exact-primary-boundary-action" if exact else "native-primary-boundary-action")
        return Chain(action.codomain_grade, out, source=source)
    from rexgraph.adjugate_operator import AdjugateOperator
    from rexgraph.rational_operator import RationalOperator
    from rexgraph.markov import MarkovView
    from rexgraph.text_overlap import TextOverlapView
    if isinstance(action, (RationalOperator, MarkovView, TextOverlapView)):
        if action.source is not source:
            raise ValueError("typed action requires its bound source")
        carrier = _typed_value(source, values, operator=action.name, variance=action.variance,
                               grade=action.domain_grade, allow_empty_upper=True)
        if carrier.cell_keys is not None:
            raise ValueError("typed action requires the canonical ordered basis")
        out = action.apply(np.asarray(carrier.values), exact=exact)
        family = ("markov" if isinstance(action, MarkovView) else
                  "text-overlap" if isinstance(action, TextOverlapView) else "factored-rational")
        record_method(("exact-" if exact else "native-") + family + "-action", kind=action.name)
        return type(carrier)(carrier.grade, out, source=source)
    if isinstance(action, AdjugateOperator):
        if action.source is not source:
            raise ValueError("adjugate requires its bound source")
        carrier = _typed_value(source, values, operator="ADJUGATE", variance=action.variance,
                               grade=action.domain_grade, allow_empty_upper=True)
        if carrier.cell_keys is not None:
            raise ValueError("adjugate requires the canonical ordered basis")
        out = action.apply(np.asarray(carrier.values), exact=exact)
        record_method("exact-adjugate-action" if exact else "native-adjugate-action",
                      coefficient_actions=dict(action.parameters)["coefficient_actions"])
        return type(carrier)(carrier.grade, out, source=source)
    if isinstance(action, ChannelOperator):
        if action.source is not source:
            raise ValueError("APPLY requires a channel bound to its source Rex")
        carrier = _typed_value(source, values, operator="APPLY", variance="cochain", grade=1)
        if carrier.cell_keys is not None:
            raise ValueError("channel APPLY requires the canonical ordered basis")
        if not exact and np.iscomplexobj(carrier.values):
            raise TypeError("channel APPLY currently requires real coefficients")
        coefficients = np.asarray(carrier.values) if exact else np.asarray(carrier.values, dtype=np.float64)
        out = action.apply(coefficients, exact=exact)
        record_method("rational-channel-action" if exact else "factored-channel-action", channel=action.channel,
                      g_channel=action.g_channel, c_channel=action.c_channel,
                      trace_normalized=False, frustration_reference="raw-G")
        return Field(Cochain(1, out, source=source), action, kind=action.name)
    if isinstance(action, OperatorBracket):
        if action.source is not source:
            raise TypeError("APPLY requires a bracket bound to its source")
        carrier = _typed_value(source, values, operator="APPLY", variance=action.variance,
                               grade=action.domain_grade, allow_empty_upper=True)
        if carrier.cell_keys is not None:
            raise ValueError("operator bracket requires the canonical ordered basis")
        out = action.apply(np.asarray(carrier.values), exact=exact)
        record_method(("rational-" if exact else "factored-") + "operator-bracket-action", bracket=action.name, graded=False)
        return type(carrier)(carrier.grade, out, source=source)
    if isinstance(action, GradedOperatorBracket):
        if action.source is not source:
            raise TypeError("APPLY requires a bracket bound to its source")
        result = action.apply(values, exact=exact)
        record_method(("rational-" if exact else "factored-") + "operator-bracket-action", bracket=action.name, graded=True)
        return result
    if isinstance(action, WeightedDiracOperator):
        if action.source is not source:
            raise TypeError("APPLY requires a Dirac bound to its source")
        result = action.apply(values, exact=exact)
        record_method(("rational-" if exact else "factored-") + "weighted-dirac-action", anti=action.anti)
        return result
    if isinstance(action, (MetricAdjointOperator, WeightedHodgeOperator)):
        if action.source is not source:
            raise ValueError("APPLY requires an action bound to its source Rex")
        carrier = values.cochain if isinstance(values, Field) else values
        if isinstance(carrier, CellBoundary):
            carrier = carrier.chain
        if isinstance(carrier, CellCoboundary):
            carrier = carrier.cochain
        metric = action.grade_metric if isinstance(action, WeightedHodgeOperator) else action.codomain_metric
        metric._carrier(carrier)
        expected = Chain if action.variance == "chain" else Cochain
        if not isinstance(carrier, expected):
            raise TypeError("APPLY typed action requires the operator's chain/cochain variance")
        out = action.apply(np.asarray(carrier.values), exact=exact)
        family = "weighted-hodge" if isinstance(action, WeightedHodgeOperator) else "metric-adjoint"
        record_method(("rational-" if exact else "factored-") + family + "-action",
                      domain_grade=action.domain_grade, codomain_grade=action.codomain_grade)
        return expected(action.codomain_grade, out, source=source)
    if exact:
        raise TypeError("APPLY exact=True requires an explicitly certified adjoint or weighted Hodge action")
    if isinstance(action, GreenOperator) and action.metric is not None:
        if action.operator.source is not source:
            raise ValueError("APPLY requires an action bound to its source Rex")
        carrier = values.chain if isinstance(values, CellBoundary) else values
        if not isinstance(carrier, Chain):
            raise TypeError("metric Green solve requires Chain variance")
        action.metric._carrier(carrier)
        out, info = action.solve_with_info(np.asarray(carrier.values))
        record_method("green-solve", **info)
        return Chain(carrier.grade, out, source=source, cell_keys=carrier.cell_keys)
    if isinstance(action, GreenOperator):
        operator = action.operator
        solve = action.solve_with_info
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
    if np.iscomplexobj(cochain.values):
        raise TypeError("APPLY currently requires real coefficients; complex values cannot be discarded")
    if isinstance(action, GreenOperator):
        out, info = solve(np.asarray(cochain.values, dtype=np.float64))
        record_method("green-solve", **info)
    else:
        out = solve(np.asarray(cochain.values, dtype=np.float64))
        record_method("rex-operator-action", construction=action.name,
                      representation="factored-or-custom-action")
    field = Cochain(operator.codomain_grade, out, source=source)
    return Field(field, action, kind=kind)


@register("ADJOINT")
def adjoint(source, operator, domain_metric=None, codomain_metric=None):
    if not isinstance(operator, RexOperator) or operator.source is not source:
        raise TypeError("ADJOINT requires a RexOperator bound to its source Rex")
    result = metric_adjoint(operator, domain_metric, codomain_metric)
    record_method("factored-metric-adjoint-handle", variance=result.variance,
                  domain_metric=result.domain_metric.coefficient_digest,
                  codomain_metric=result.codomain_metric.coefficient_digest,
                  exact_action=result.exact_matvec is not None)
    return result


def _bracket(source, left, right, anti):
    if getattr(left, "source", None) is not source or getattr(right, "source", None) is not source:
        raise TypeError("bracket operands must be bound to the source")
    result = operator_bracket(left, right, anti=anti)
    record_method("factored-operator-bracket-handle", bracket=result.name,
                  graded=isinstance(result, GradedOperatorBracket), matrix_free=True)
    return result


@register("COMMUTATOR")
def commutator(source, left, right):
    return _bracket(source, left, right, False)


@register("ANTICOMMUTATOR")
def anticommutator(source, left, right):
    return _bracket(source, left, right, True)


@register("RESOLVENT")
def resolvent(source, action, alpha=1.0, tol=1e-10, maxiter=1000):
    if not isinstance(action, RexOperator) or action.source is not source:
        raise TypeError("RESOLVENT requires a RexOperator bound to its source")
    if action.shape[0] != _cell_count(source, action.domain_grade):
        raise ValueError("RESOLVENT operator population differs from its source grade")
    # The declared input stays rational in the AST/plan. This solver's contract
    # is numerical, and NumPy's finite check does not accept Fraction objects.
    tolerance = float(tol) if isinstance(tol, Fraction) else tol
    result = GreenOperator.resolvent(action, alpha, tol=tolerance, maxiter=maxiter)
    record_method("identity-plus-operator-resolvent", alpha=float(alpha), tol=float(tol),
                  maxiter=int(maxiter), matrix_free=True,
                  solve_form="positive-diagonal-metric" if result.metric is not None else "euclidean")
    return result


@register("GREEN_SOLVE")
def green_solve(source, action, values):
    if not isinstance(action, GreenOperator):
        raise TypeError("GREEN_SOLVE requires an explicit Green action")
    return apply(source, action, values)


@register("METRIC")
def metric(source, grade, weights=None):
    from rexgraph.graded_metric import diagonal_metric
    result = diagonal_metric(source, grade, weights)
    record_method("positive-diagonal-metric" if weights is not None else "identity-metric",
                  grade=result.grade, coefficient_digest=result.coefficient_digest,
                  coefficient_arithmetic="rational" if result.exact else "approximate")
    return result


def _contraction_metric(source, value, selected):
    from rexgraph.graded_metric import DiagonalMetric
    if selected is None:
        return DiagonalMetric(source, value.grade, (Fraction(1),) * value.n_cells, value.cell_keys)
    if not isinstance(selected, DiagonalMetric):
        raise TypeError("contraction requires a declared DiagonalMetric")
    selected._carrier(value)
    return selected


@register("MOMENT")
def moment(source, left, right, metric=None, exact=False):
    left, right = _same_space(source, left, right, operator="MOMENT")
    selected = _contraction_metric(source, left, metric)
    result = selected.moment(left, right, exact=exact)
    record_method("rational-metric-contraction" if exact else "hermitian-metric-contraction",
                  functional="moment", metric_digest=selected.coefficient_digest)
    return result


@register("INTEGRATE")
def integrate(source, cochain, chain, exact=False):
    from rexgraph.graded_metric import integrate as pair
    cochain = _typed_value(source, cochain, operator="INTEGRATE", variance="cochain")
    chain = _typed_value(source, chain, operator="INTEGRATE", variance="chain")
    result = pair(cochain, chain, exact=exact)
    record_method("rational-dual-pairing" if exact else "numeric-dual-pairing",
                  functional="integrate", pairing="bilinear-cochain-chain",
                  grade=cochain.grade, materializes_sparse=False)
    return result


@register("QUADRANCE")
def quadrance(source, values, exact=False, metric=None):
    value = _typed_value(source, values, operator="QUADRANCE")
    if exact:
        _require_exact_coefficients(value.values)
    if metric is not None:
        selected = _contraction_metric(source, value, metric)
        result = selected.moment(value, value, exact=exact)
        result = result if exact else float(np.real(result))
        record_method("rational-metric-contraction" if exact else "hermitian-metric-contraction",
                      functional="quadrance", metric_digest=selected.coefficient_digest)
        return result
    result = _quadrance(value.values, exact=bool(exact))
    record_method("rational-contraction" if exact else "hermitian-contraction", functional="quadrance")
    return result


@register("ACCESS")
def access(source, value, accession, exact=False):
    from rexgraph.type_accession import TypeAccession
    if not isinstance(accession, TypeAccession) or accession.source is not source:
        raise TypeError("ACCESS requires a source-bound TypeAccession")
    result = accession.apply(_typed_value(source, value, operator="ACCESS"), exact=exact)
    record_method("rational-sparse-accession" if exact else "csr-accession",
                  names=(accession.name,), coefficient_digests=(accession.coefficient_digest,),
                  nnz=len(accession.entries), shape=accession.shape,
                  output_space="ambient" if accession.coordinates is None else accession.coordinates.name,
                  chain_preserving=None)
    return result


@register("CHAIN_MAP")
def chain_map(source, declaration):
    from rexgraph.chain_map import ChainMap, GradedMap
    declaration = declaration.declaration if isinstance(declaration, ChainMap) else declaration
    if not isinstance(declaration, GradedMap) or declaration.domain.source is not source:
        raise TypeError("CHAIN_MAP requires a source-bound GradedMap")
    result = declaration.verify()
    record_method("exact-sparse-chain-map-verification",
                  coefficient_digest=declaration.coefficient_digest,
                  domain_digest=declaration.domain.coefficient_digest,
                  codomain_digest=declaration.codomain.coefficient_digest,
                  shapes=declaration.shapes, nnz=tuple(len(p) for p in declaration.components),
                  source_residual=str(result.source_residual), target_residual=str(result.target_residual),
                  commutation_residuals=tuple(str(r) for r in result.commutation_residuals),
                  chain_preserving=True)
    return result


@register("ACCESS_TYPES")
def access_types(source, value, accessions, exact=False):
    from rexgraph.type_accession import AccessionFamily
    if not isinstance(accessions, AccessionFamily) or accessions.accessions[0].source is not source:
        raise TypeError("ACCESS_TYPES requires a source-bound AccessionFamily")
    result = accessions.apply(_typed_value(source, value, operator="ACCESS_TYPES"), exact=exact)
    record_method("rational-sparse-accession" if exact else "csr-accession",
                  names=result.names, coefficient_digests=tuple(a.coefficient_digest for a in accessions.accessions),
                  nnz=sum(len(a.entries) for a in accessions.accessions),
                  shapes=tuple(a.shape for a in accessions.accessions),
                  member_shapes=tuple(v.values.shape for v in result.views), chain_preserving=None)
    return result


@register("CO_RELATE")
def co_relate(source, left, right, metric=None, exact=False):
    from rexgraph.type_accession import CrossMetric, FamilyMetric, TypeView
    from rexgraph.type_accession import co_relate as contract
    if (not isinstance(left, TypeView) or not isinstance(right, TypeView)
            or left.accession.source is not source or right.accession.source is not source):
        raise TypeError("CO_RELATE requires source-bound TypeViews")
    if isinstance(metric, FamilyMetric):
        result = contract(left, right, metric, exact=exact)
        record_method("rational-factored-family-contraction" if exact else "csr-hermitian-family-contraction",
                      names=(left.name, right.name), metric_digest=metric.coefficient_digest,
                      base_metric_digest=metric.metric.coefficient_digest,
                      realization_digests=tuple(metric.realization_for(v).coefficient_digest for v in (left, right)),
                      family_psd=True, positive_definite=None, functional="co-relation")
        return result
    if isinstance(metric, CrossMetric):
        result = contract(left, right, metric, exact=exact)
        record_method("rational-sparse-cross-contraction" if exact else "csr-sesquilinear-cross-contraction",
                      names=(left.name, right.name), metric_digest=metric.coefficient_digest,
                      shape=metric.shape, nnz=len(metric.entries), positivity="not-asserted",
                      functional="co-relation")
        return result
    if left.accession.coordinates is not None or right.accession.coordinates is not None:
        raise TypeError("type coordinates require an explicit CrossMetric or FamilyMetric, not an ambient metric")
    selected = _contraction_metric(source, left.carrier, metric)
    result = contract(left, right, selected, exact=exact)
    record_method("rational-type-contraction" if exact else "hermitian-type-contraction",
                  names=(left.name, right.name), metric_digest=selected.coefficient_digest,
                  functional="co-relation")
    return result


@register("MOMENT_TENSOR")
def moment_tensor(source, family, metric=None, exact=False):
    from rexgraph.type_accession import FamilyMetric, TypedFamily
    from rexgraph.type_accession import moment_tensor as contract
    if not isinstance(family, TypedFamily) or family.views[0].accession.source is not source:
        raise TypeError("MOMENT_TENSOR requires a source-bound TypedFamily")
    result = contract(family, metric, exact=exact)
    if isinstance(result.metric, FamilyMetric):
        record_method("rational-factored-family-contraction" if exact else "csr-hermitian-family-contraction",
                      names=result.names, metric_digest=metric.coefficient_digest,
                      base_metric_digest=metric.metric.coefficient_digest,
                      realization_digests=tuple(metric.realization_for(v).coefficient_digest for v in family.views),
                      realization_actions=len(family.views), family_psd=True, positive_definite=None,
                      functional="typed-moment-tensor", output_axes="type-by-type")
        return result
    record_method("rational-type-contraction" if exact else "hermitian-type-contraction",
                  names=result.names, metric_digest=result.metric.coefficient_digest,
                  functional="typed-moment-tensor", output_axes="type-by-type")
    return result


@register("SPREAD")
def spread(source, left, right, exact=False, metric=None):
    left, right = _same_space(source, left, right, operator="SPREAD")
    a, b = np.asarray(left.values), np.asarray(right.values)
    if exact:
        _require_exact_coefficients(a)
        _require_exact_coefficients(b)
    if a.ndim != 1:
        raise ValueError("SPREAD currently expects one-dimensional typed values")
    if metric is not None:
        selected = _contraction_metric(source, left, metric)
        ip = selected.moment(left, right, exact=exact)
        qa = selected.moment(left, left, exact=exact)
        qb = selected.moment(right, right, exact=exact)
        if not exact:
            qa, qb = float(np.real(qa)), float(np.real(qb))
        if qa == 0 or qb == 0:
            result = None
        elif exact:
            result = Fraction(1) - ip * ip / (qa * qb)
        else:
            # Divide by the larger quadrance before multiplying: neither the
            # squared moment nor qa*qb needs to be representable as a double.
            magnitude = abs(ip)
            result = 1.0 - (magnitude / max(qa, qb)) * magnitude / min(qa, qb)
        if result is not None and not exact and not np.isfinite(result):
            raise FloatingPointError("metric spread is outside numerical range")
        record_method("rational-metric-contraction" if exact else "hermitian-metric-contraction",
                      functional="spread", metric_digest=selected.coefficient_digest, defined=result is not None)
        return result
    result = rational_spread(a, b, exact=bool(exact))
    record_method("rational-contraction" if exact else "hermitian-contraction", functional="spread", defined=result is not None)
    return result


@register("ACCUMULATE")
def accumulate(source, left, right):
    """Add two aligned graded coefficient fields without discarding their carrier.

    This is a tensor accumulation, not a path count. The values must already
    inhabit the same source bound ordered basis and temporal state; time or
    basis transport has to be a separately declared action rather than an
    implicit length based merge.
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
    values = np.asarray(cochain.values)
    output = harmonic_winding(harmonic_frame(source, native=True), values)
    # Numerical inputs retain a numerical contract even when their represented
    # coefficients happen to be integral. Exact inputs keep the core pairing.
    return np.asarray(output, dtype=float) if values.dtype.kind == 'f' else output


@register("CLOSURE")
def closure(source, seed, max_depth=8, grade=0):
    from rexgraph.tower import semantic_closure
    result = semantic_closure(source, seed, max_depth=max_depth, grade=grade)
    record_method("native-induced-structural-closure", steps=len(result["steps"]), converged=result["converged"])
    return result


@register("SIGNIFICANCE")
def significance(source, edge):
    if isinstance(edge, (bool, np.bool_)) or not isinstance(edge, (int, np.integer)):
        raise TypeError("SIGNIFICANCE currently expects an integer grade-1 cell index")
    idx = int(edge)
    if idx < 0 or idx >= _cell_count(source, 1):
        raise ValueError(f"SIGNIFICANCE edge index {idx} is not present")
    from rexgraph.semantic import significance as _significance
    return float(_significance(source, [idx])[0])


class CharacterResult(dict):
    """Character payload with structural labels beside its rational coefficient array."""


@register("CHANNEL")
def channel(source, name):
    from rexgraph.channel_operator import channel_operator
    action = channel_operator(source, name)
    record_method("factored-channel-handle", channel=action.channel,
                  g_channel=action.g_channel, c_channel=action.c_channel,
                  trace_normalized=False, frustration_reference="raw-G")
    return action


@register("STAR_CHARACTER")
def star_character(source, cell, exact=False):
    from rexgraph.channel_operator import _require_channel_source
    from rexgraph.rational_trig import exact_star_character
    from rexgraph.sparse_character import build_sparse_character_cheap
    cell = _typed_cells(source, cell, operator="STAR_CHARACTER")
    if not isinstance(cell, Cell) or cell.grade != 0:
        raise TypeError("STAR_CHARACTER requires one C0 cell")
    _require_channel_source(source)
    if exact:
        values, names = exact_star_character(source)
    else:
        bundle = build_sparse_character_cheap(source)
        values, names = bundle["chi_star"], bundle["hat_names"]
    record_method("exact-star-mean" if exact else "incidence-star-mean",
                  cell=cell.index, g_channel=source.g_channel, c_channel=source.c_channel)
    return CharacterResult({"values": np.asarray(values[cell.index], dtype=object if exact else float),
                            "channels": tuple(names), "grade": 0, "cell": cell.index,
                            "exactness": "rational" if exact else "approximate"})


def _moment_operator(source, action):
    if not isinstance(action, RexOperator) or action.source is not source:
        raise TypeError("scale reading requires a RexOperator bound to its source")
    if not action.symmetric or action.domain_grade != action.codomain_grade or action.shape[0] != action.shape[1]:
        raise TypeError("scale reading requires a symmetric square operator")
    if action.shape[0] != _cell_count(source, action.domain_grade):
        raise ValueError("scale operator population differs from its source grade")
    return action


def _moment_matrix(action):
    matrix = action.as_native()
    if matrix.shape != action.shape or np.iscomplexobj(matrix.data):
        raise ValueError("scale moments require a real matrix matching the operator spaces")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("scale moments require finite coefficients")
    residual = matrix.add(matrix.T, -1)
    if residual.nnz and np.any(residual.data != 0):
        raise ValueError("scale moments require a symmetric matrix, not only a declared flag")
    return matrix


@register("SCALE_MOMENT")
def scale_moment(source, action, order, local=False, exact=False):
    from rexgraph.channel_operator import ChannelOperator
    from rexgraph.scale_propagator import trace_moments
    action = _moment_operator(source, action)
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("SCALE_MOMENT order must be a nonnegative integer")
    if not isinstance(local, bool) or not isinstance(exact, bool):
        raise TypeError("local and exact must be booleans")
    if exact and order != 0 and (order != 1 or not isinstance(action, ChannelOperator)):
        raise ValueError("exact SCALE_MOMENT supports order 0, or channel order 1 only")
    n = action.shape[0]
    if order == 0:
        values = np.full(n, Fraction(1), dtype=object) if exact else np.ones(n)
        method = "identity-moment"
    elif order == 1 and isinstance(action, ChannelOperator):
        values = action.diagonal(exact=exact)
        method = "exact-channel-diagonal" if exact else "channel-diagonal"
    else:
        matrix = _moment_matrix(action)
        values = trace_moments(matrix, int(order), local=local)[-1]
        method = "sparse-halved-moment"
    if not exact and not np.all(np.isfinite(values)):
        raise FloatingPointError("scale moment is outside float64")
    record_method(method, order=int(order), local=local, exact=exact,
                  materializes_sparse=method == "sparse-halved-moment")
    if local:
        return Cochain(action.codomain_grade, values, source=source)
    return sum(values, Fraction(0)) if exact else float(np.sum(values))


@register("CHARACTER_ENERGY")
def character_energy(source, action):
    from rexgraph.scale_propagator import energy_character
    action = _moment_operator(source, action)
    values = energy_character(_moment_matrix(action))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("character energy is outside float64")
    record_method("sparse-row-quadrance", materializes_sparse=True, order=2, local=True)
    return Cochain(action.codomain_grade, values, source=source)


@register("CHARACTER")
def character(source, exact=False):
    if not exact:
        return np.asarray(source.structural_character)
    values, channels = exact_character(source)
    record_method("exact-incidence-diagonal-character", g_channel=source.g_channel, c_channel=source.c_channel)
    return CharacterResult({
        "values": np.asarray(values, dtype=object).reshape((int(source.nE), len(channels))),
        "channels": tuple(channels),
        "exactness": "rational",
    })


@register("ZERO")
def zero(source, grade, kind="cochain"):
    grade = int(grade)
    size = _cell_count(source, grade)
    record_method("integer-zero", grade=grade, size=size)
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
    table diff or vertex path request.
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
    """Split a direct temporal C1 field on its current relational complex basis.

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
    return _read_rcdb_record(source, record_id).value


def _read_rcdb_record(source, record_id):
    """Use RCDB's own version/payload/digest contract, never a second lookup."""
    snapshot = source.read_record(record_id)
    if snapshot is None:
        raise KeyError(f"RCDB record {record_id!r} is not present")
    record_method("rcdb-record-read", record_id=snapshot.record.id,
                  record_version=snapshot.record.version, state_digest=snapshot.state_digest)
    return snapshot


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
@register("TENSOR_MANIFEST")
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
    return _read_rcdb_record(source, record_id).state_digest

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
    digest = source.state_digest()
    record_method("rcdb-logical-state-digest", manifest_version=1, state_digest=digest)
    return digest


@register("RCDB_SECURITY")
def rcdb_security(source):
    """Return bounded RCDB security configuration without secrets or paths."""
    if not hasattr(source, "security_status"):
        raise TypeError("RCDB_SECURITY expects an RCDB store")
    return dict(source.security_status())


from .value_operators import install as _install_value_operators

_install_value_operators(register)

from .calculus_operators import install as _install_calculus_operators

_install_calculus_operators(register)

from .critical_operators import install as _install_critical_operators

_install_critical_operators(register)

from .certificate_operators import install as _install_certificate_operators

_install_certificate_operators(register)

from .temporal_operators import install as _install_temporal_operators

_install_temporal_operators(register)

from .structure_operators import install as _install_structure_operators
from .homology_operators import install as _install_homology_operators

_install_homology_operators(register)

_install_structure_operators(register)

from .partition_operators import install as _install_partition_operators

_install_partition_operators(register)

from .artifact_operators import install as _install_artifact_operators
from .filling_operators import install as _install_filling_operators

_install_filling_operators(register)
from .difference_operators import install as _install_difference_operators
from .replay_operators import install as _install_replay_operators

_install_replay_operators(register)

_install_difference_operators(register)

from .document_operators import install as _install_document_operators

_install_document_operators(register)

from .rational_operators import install as _install_rational_operators

_install_rational_operators(register)

from .markov_operators import install as _install_markov_operators

_install_markov_operators(register)

_install_artifact_operators(register)

from .symmetry_operators import install as _install_symmetry_operators

_install_symmetry_operators(register)

from .corpus_contracts import corpus_field

register("CORPUS_FIELD")(corpus_field)

from .turn_contracts import turn_field, path_change

register("TURN_FIELD")(turn_field)
register("PATH_CHANGE")(path_change)
