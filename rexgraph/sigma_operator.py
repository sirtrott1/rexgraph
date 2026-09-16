"""Explicit exponential vertex families and incidence factored C1 sigma actions.

The family is w_v(s) = a_v exp(r_v s), with a_v positive. No prime sequence,
channel selection or coparticipation convention is inferred. Each weighted
boundary column has unit quadrance before its fixed relation metric is applied.
G means the raw unsigned channel here, never the upper Hodge sector.
"""
from __future__ import annotations

from fractions import Fraction
from math import isfinite
from numbers import Real

import numpy as np

from rexgraph.linear_operator import RexOperator, _numeric_array
from rexgraph.native_sparse import NativeSparse
from rexgraph.operator_bracket import operator_bracket


def family_parameters(sigma, amplitudes, rates, channels, c_channel, grade=1):
    """Validate declarations without applying any operator or constructing a Gram."""
    def scalar(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Fraction)):
            raise TypeError("sigma family coefficients must be real scalars")
        try:
            out = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("sigma family coefficient is outside numerical range") from exc
        if not isfinite(out):
            raise ValueError("sigma family coefficients must be finite")
        return out

    sigma = scalar(sigma)
    if not isinstance(grade, int) or isinstance(grade, bool) or grade != 1:
        raise ValueError("this sigma family acts on C1; a higher tower family is a separate declaration")
    if not all(isinstance(v, (list, tuple)) for v in (amplitudes, rates, channels)):
        raise TypeError("sigma family requires explicit amplitude, rate and channel sequences")
    amplitudes, rates = tuple(map(scalar, amplitudes)), tuple(map(scalar, rates))
    if len(amplitudes) != len(rates) or any(a <= 0 for a in amplitudes):
        raise ValueError("sigma amplitudes must be positive and match the rate axis")
    channels = tuple(channels)
    if (not channels or any(not isinstance(c, str) or c not in {"T", "G", "F", "C"} for c in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError("sigma channels must be a nonempty distinct sequence drawn from T, G, F, C")
    if c_channel not in ("share", "count"):
        raise ValueError("sigma C channel must explicitly select share or count")
    return sigma, amplitudes, rates, channels, c_channel


def _source_state(source):
    from rexgraph.sparse_character import _require_distinct_channel_participants
    source._ensure_clean()
    _require_distinct_channel_participants(source)
    for name in ("w_V", "vertex_weights"):
        weights = getattr(source, name, None)
        if weights is not None and np.any(np.asarray(weights) != 1):
            raise ValueError("declare vertex weighting in the sigma family, not both source and family")
    b = NativeSparse(source._B1_dual)
    weights = source.edge_metric_exact
    weights = tuple([Fraction(1)] * b.shape[1] if weights is None else weights)
    state = (b.shape, b.dual.row_ptr.tobytes(), b.dual.col_idx.tobytes(), b.data.tobytes(), weights)
    return b, weights, state


def _scale(diagonal, values):
    return diagonal.reshape((len(diagonal),) + (1,) * (values.ndim - 1)) * values


def _handle(source, state, parameters, action, derivative, traces):
    def checked(values, tangent=False):
        if _source_state(source)[2] != state:
            raise ValueError("sigma source boundary or metric changed; bind a fresh family")
        values = _numeric_array(values, operation="sigma action")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out = derivative(values) if tangent else action(values)
        if not np.isfinite(out).all():
            raise FloatingPointError("sigma action exceeds numerical range")
        return out

    n = state[0][1]
    result = RexOperator("SIGMA_OPERATOR", (n, n), 1, 1, checked, source=source, symmetric=True,
                         psd=True, construction="sigma-family", parameters=parameters)
    tangent = RexOperator("SIGMA_DERIVATIVE", (n, n), 1, 1, lambda x: checked(x, True), source=source,
                          symmetric=True, construction="sigma-derivative", parameters=parameters)
    object.__setattr__(result, "derivative", tangent)
    object.__setattr__(result, "scaled_channel_traces", tuple(traces.items()))
    return result


def sigma_operator(source, sigma, amplitudes, rates, channels, c_channel, grade=1):
    """Sum selected trace normalized channels and retain its analytic derivative.

    For nonzero columns, D[v,e] = sqrt(w_v/q_e) B1[v,e] m_e, where
    q_e = sum_v w_v B1[v,e]^2 and m_e is the fixed stored relation metric.
    T = D^T D; raw G replaces the unweighted D factor by its absolute value
    before the fixed relation metric. F uses orientation disagreement
    before the relation metric, including signed metrics. C is the independent
    unweighted share or count channel. Zero columns and zero channels stay zero.

    Exponential weights and square roots make the full action numerical. There
    is no certified rational action, eigenbasis, finite difference or Gram cache.
    All vector products use NativeSparse and the compiled DualCSR kernels.
    """
    sigma, amplitudes, rates, channels, c_channel = family_parameters(
        sigma, amplitudes, rates, channels, c_channel, grade)
    b, metric, state = _source_state(source)
    nv, ne = b.shape
    if len(amplitudes) != nv:
        raise ValueError("sigma family must name every vertex in the canonical C0 order")
    parameters = (("family", "explicit-exponential-vertex"), ("sigma", sigma),
                  ("amplitudes", amplitudes), ("rates", rates),
                  ("channels", channels), ("g_channel", "raw"), ("c_channel", c_channel),
                  ("column_normalization", "unit-before-relation-metric"), ("channel_normalization", "trace"),
                  ("relation_metric_rescaling", "common-absolute-maximum-cancels-in-hats"))
    rows = np.repeat(np.arange(nv), np.diff(b.dual.row_ptr))
    cols = b.dual.col_idx
    c = b.with_data((b.data != 0).astype(float) if c_channel == "count" else abs(b.data))
    row_mass = c.apply(np.ones(ne))
    cd = np.bincount(cols, weights=c.data * (row_mass[rows] - c.data), minlength=ne)
    ctotal = cd + c.column_quadrances()
    if channels == ("C",):
        trace = float(cd.sum())
        if not isfinite(trace) or trace < 0:
            raise FloatingPointError("sigma C trace exceeds numerical range")
        def c_action(x):
            return (_scale(ctotal, x) - c.transpose_apply(c.apply(x))) / trace if trace else np.zeros_like(x)
        return _handle(source, state, parameters, c_action, np.zeros_like, {"C": trace})
    # All selected T/G/F hats are homogeneous of degree zero in a common
    # relation metric scale. Divide over Q before conversion, retaining ratios
    # even when the original integers or Fractions cannot fit in float64.
    scale = max(map(abs, metric), default=Fraction(0))
    normalized_metric = tuple(w / scale for w in metric) if scale else metric
    m = _numeric_array(np.asarray(normalized_metric, dtype=object), operation="sigma relation metric")
    if any(a != 0 and f == 0 for a, f in zip(metric, m, strict=True)):
        raise FloatingPointError("sigma relation metric underflow")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        squared = m * m
    if not np.isfinite(squared).all() or np.any((m != 0) & (squared == 0)):
        raise FloatingPointError("sigma squared relation metric exceeds numerical range")
    rates = np.asarray(rates)
    # Common scaling cancels from each column. Subtracting the largest log
    # weight avoids overflow without replacing a small positive value by zero.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        logw = np.log(amplitudes) + sigma * rates
        w = np.exp(logw - np.max(logw)) if nv else np.empty(0)
    if not np.isfinite(logw).all() or not np.isfinite(w).all() or np.any(w == 0):
        raise FloatingPointError("sigma vertex family exceeds numerical dynamic range")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        quadratic = b.data * b.data * w[rows]
        q = np.bincount(cols, weights=quadratic, minlength=ne)
        active = np.bincount(cols, weights=(b.data != 0), minlength=ne) != 0
        if not np.isfinite(q).all() or np.any(q[active] <= 0):
            raise FloatingPointError("sigma column quadrance exceeds numerical range")
        rate_mean = np.zeros(ne)
        rate_mean[active] = np.bincount(cols, weights=quadratic * rates[rows], minlength=ne)[active] / q[active]
        base = np.zeros_like(b.data)
        present = b.data != 0
        base[present] = b.data[present] * np.sqrt(w[rows[present]] / q[cols[present]])
        tangent = 0.5 * (rates[rows] - rate_mean[cols])
    if not np.isfinite(base).all() or not np.isfinite(tangent).all() or np.any((base == 0) & present):
        raise FloatingPointError("sigma normalized boundary exceeds numerical range")
    d = b.with_data(base * m[cols])
    a = b.with_data(abs(base) * m[cols])
    dp, ap = d.with_data(d.data * tangent), a.with_data(a.data * tangent)
    pos, neg = b.with_data(np.maximum(base, 0)), b.with_data(np.maximum(-base, 0))
    pp, np_ = pos.with_data(pos.data * tangent), neg.with_data(neg.data * tangent)
    magnitude = abs(m)
    fd = 2 * magnitude * (pos.transpose_apply(neg.apply(magnitude)) + neg.transpose_apply(pos.apply(magnitude)))
    fdp = 2 * magnitude * (pp.transpose_apply(neg.apply(magnitude)) + pos.transpose_apply(np_.apply(magnitude))
                          + np_.transpose_apply(pos.apply(magnitude)) + neg.transpose_apply(pp.apply(magnitude)))
    if "F" in channels:
        positive = (base > 0) & (m[cols] != 0)
        negative = (base < 0) & (m[cols] != 0)
        positive_rows = np.bincount(rows, weights=positive, minlength=nv) > 0
        negative_rows = np.bincount(rows, weights=negative, minlength=nv) > 0
        active_f = np.bincount(cols, weights=(positive & negative_rows[rows]) | (negative & positive_rows[rows]),
                               minlength=ne) > 0
        if np.any(active_f & (fd == 0)):
            raise FloatingPointError("sigma lost a nonzero frustration diagonal to underflow")
    tdiag = d.column_quadrances()
    tprime = 2 * np.bincount(cols, weights=d.data * dp.data, minlength=ne)
    diagonals = {"T": tdiag, "G": tdiag, "F": fd, "C": cd}
    derivatives = {"T": tprime, "G": tprime, "F": fdp, "C": np.zeros(ne)}
    traces = {key: float(diagonals[key].sum()) for key in channels}
    slopes = {key: float(derivatives[key].sum()) for key in channels}
    if any(not np.isfinite(v).all() for v in (*diagonals.values(), *derivatives.values())):
        raise FloatingPointError("sigma channel diagonal exceeds numerical range")
    if any(not isfinite(v) or v < 0 for v in traces.values()) or any(not isfinite(v) for v in slopes.values()):
        raise FloatingPointError("sigma channel trace exceeds numerical range")

    def gram(matrix, value):
        return matrix.transpose_apply(matrix.apply(value))

    def gram_prime(matrix, derivative, value):
        return derivative.transpose_apply(matrix.apply(value)) + matrix.transpose_apply(derivative.apply(value))

    def raw(key, x, derivative):
        if key == "T":
            return gram_prime(d, dp, x) if derivative else gram(d, x)
        if key == "G":
            return gram_prime(a, ap, x) if derivative else gram(a, x)
        if key == "F":
            weighted = _scale(m, x)
            if derivative:
                overlap = (pp.transpose_apply(neg.apply(weighted)) + pos.transpose_apply(np_.apply(weighted))
                           + np_.transpose_apply(pos.apply(weighted)) + neg.transpose_apply(pp.apply(weighted)))
            else:
                overlap = pos.transpose_apply(neg.apply(weighted)) + neg.transpose_apply(pos.apply(weighted))
            return _scale(fdp if derivative else fd, x) - 2 * _scale(m, overlap)
        return np.zeros_like(x) if derivative else _scale(ctotal, x) - gram(c, x)

    def action(values, derivative=False):
        out = np.zeros_like(values)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for key in channels:
                tr = traces[key]
                if tr == 0:
                    continue
                term = raw(key, values, False) / tr
                out += raw(key, values, True) / tr - term * (slopes[key] / tr) if derivative else term
        return out
    return _handle(source, state, parameters, action, lambda x: action(x, True), traces)


def critical_commutator(source, sigma, amplitudes, rates, channels, c_channel, grade=1):
    """[R(s),R(1-s)]; the center is automatic, not evidence of criticality."""
    left = sigma_operator(source, sigma, amplitudes, rates, channels, c_channel, grade)
    right = left if float(sigma) == 0.5 else sigma_operator(
        source, 1 - float(sigma), amplitudes, rates, channels, c_channel, grade)
    return operator_bracket(left, right)


def critical_rate(source, amplitudes, rates, channels, c_channel, reading="generator", grade=1):
    """[R,R'] at s=1/2, or the involution commutator slope, which is -2[R,R']."""
    if reading not in ("generator", "slope"):
        raise ValueError("critical rate reading must be generator or slope")
    center = sigma_operator(source, 0.5, amplitudes, rates, channels, c_channel, grade)
    bracket = operator_bracket(center, center.derivative)
    if reading == "generator":
        return bracket
    return RexOperator("CRITICAL_RATE", bracket.shape, 1, 1, lambda x: -2 * bracket.apply(x),
                       source=source, construction="sigma-critical-slope",
                       transpose_matvec=lambda x: -2 * bracket.transpose_apply(x),
                       parameters=center.parameters + (("reading", reading),))
