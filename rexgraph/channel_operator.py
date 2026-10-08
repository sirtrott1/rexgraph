"""Native incidence factored structural channels.

G follows the source's raw/normalized selection; F always uses raw unsigned G.
T/G/F carry relation weights, C retains the independent share/count reading.
No channel Gram is formed for an action or first moment diagonal. Sparse
materialization is explicit and can incur sum_v deg(v)^2 fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from rexgraph.linear_operator import RexOperator
from rexgraph.native_sparse import NativeSparse
from rexgraph.rational_trig import CHANNEL_ORDER, exact_channel_diagonals
from rexgraph.sparse_character import (
    _f64,
    _require_distinct_channel_participants,
    build_sparse_channels,
    channel_diagonals,
)

CHANNEL_KEYS = dict(zip(("T", "G", "F", "C"), CHANNEL_ORDER, strict=True))


def _require_channel_source(rex):
    _require_numerical_channel_source(rex)
    _require_distinct_channel_participants(rex)


def _require_numerical_channel_source(rex):
    rex._ensure_clean()
    for attr in ("w_V", "vertex_weights"):
        weights = getattr(rex, attr, None)
        if weights is not None and np.any(np.asarray(weights) != 1):
            raise ValueError("factored channel readings do not support vertex weighting")
    if rex.g_channel == "normalized":
        weights = rex.edge_metric_exact
        if weights is not None and any(w < 0 for w in weights):
            raise ValueError("normalized G requires nonnegative relation weights")


@dataclass(frozen=True)
class ChannelOperator(RexOperator):
    """A selected C1 channel before trace normalization, not a hat.

    Like other Rex handles it is source bound, not a transaction snapshot.
    Keep the source unchanged while reusing the handle.
    """

    channel: str = "T"
    g_channel: str = "raw"
    c_channel: str = "share"

    def diagonal(self, *, exact=False):
        _require_channel_source(self.source)
        if ((self.source.g_channel, self.source.c_channel) != (self.g_channel, self.c_channel)
                or int(self.source.nE) != self.shape[0]):
            raise ValueError("channel source or selection changed; bind a fresh operator")
        if exact:
            diags, _ = exact_channel_diagonals(self.source)
            return np.asarray(diags[CHANNEL_KEYS[self.channel]], dtype=object)
        diags = channel_diagonals(self.source)
        if diags is None:
            raise ValueError("this source has no incidence-only channel diagonals")
        return np.asarray(diags[CHANNEL_KEYS[self.channel]], dtype=float)


def channel_operator(rex, name):
    """Return T, selected G, raw-G frustration F, or selected C as a factored action.

    T, raw G, F and C have exact rational actions and transposes. Normalized G
    has an exact diagonal but no general rational full action. Construction does
    not cast relation weights to floats, even outside float64's range.
    """
    if not isinstance(name, str) or name.upper() not in CHANNEL_KEYS:
        raise ValueError("channel must be T, G, F or C")
    name = name.upper()
    _require_channel_source(rex)
    n = int(rex.nE)
    selection = rex.g_channel, rex.c_channel
    cached = {}

    def check():
        _require_channel_source(rex)
        if (rex.g_channel, rex.c_channel) != selection or int(rex.nE) != n:
            raise ValueError("channel source changed; bind a fresh operator")

    def apply(values):
        check()
        if "apply" not in cached:
            _, cached["apply"], _ = build_factored_operator(
                rex, None, (), (), trace_normalized=False)
        result = cached["apply"](CHANNEL_KEYS[name], values)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("channel action is not representable in float64")
        return result

    def materialize():
        check()
        return dict(build_sparse_channels(rex, native=True))[CHANNEL_KEYS[name]]

    def exact_apply(values):
        check()
        if "exact_apply" not in cached:
            cached["exact_apply"] = _exact_channel_action(rex, name)
        return cached["exact_apply"](values)

    rational = name != "G" or selection[0] == "raw"
    return ChannelOperator(
        "CHANNEL_" + name, (n, n), 1, 1, apply, matrix_factory=materialize,
        source=rex, symmetric=True, psd=True, channel=name,
        g_channel=selection[0], c_channel=selection[1],
        exact_matvec=exact_apply if rational else None,
        exact_transpose_matvec=exact_apply if rational else None,
    )


def _exact_gram_action(columns, values):
    """Apply one exact primary Gram through incidence, without pair enumeration."""
    zero = Fraction(0)
    rows = {}
    for e, col in enumerate(columns):
        for v, coefficient in col.items():
            rows[v] = rows.get(v, zero) + coefficient * values[e]
    out = np.full(values.shape, zero, dtype=object)
    for e, col in enumerate(columns):
        for v, coefficient in col.items():
            out[e] += coefficient * rows[v]
    return out


def _exact_channel_action(rex, name):
    """Sparse Q incidence passes, never a relation square Gram or float recovery.

    Primary shares come from the existing exact boundary reader. Relation weights
    use edge_metric_exact: a stored float means its exact binary value, not a
    guessed decimal. F's diagonal uses orientation and absolute weight masses;
    its off diagonal uses signed weights. C ignores relation weights entirely.
    """
    from rexgraph.faces import _exact_b1_block
    columns = _exact_b1_block(rex, range(int(rex.nE)))
    n = len(columns)
    zero = Fraction(0)

    if name == "C":
        support_count = rex.c_channel == "count"
        unsigned = [{v: Fraction(1) if support_count else abs(c) for v, c in col.items()} for col in columns]
        # C = diag(K_C 1) - K_C, including the self overlap in both terms.
        row_mass = _exact_gram_action(unsigned, np.full(n, Fraction(1), dtype=object))
        return lambda values: row_mass.reshape((n,) + (1,)*(values.ndim-1))*values - _exact_gram_action(unsigned, values)

    weights = rex.edge_metric_exact
    weights = [Fraction(1)]*n if weights is None else weights
    if name == "T":
        signed = [{v: weights[e]*c for v, c in col.items()} if weights[e] else {} for e, col in enumerate(columns)]
        return lambda values: _exact_gram_action(signed, values)
    if name == "G" and rex.g_channel == "raw":
        unsigned = [{v: weights[e]*abs(c) for v, c in col.items()} if weights[e] else {} for e, col in enumerate(columns)]
        return lambda values: _exact_gram_action(unsigned, values)
    if name == "F":
        diagonal = np.asarray(exact_channel_diagonals(rex)[0]["L_SG"], dtype=object)
        entries = tuple((e, v, c > 0, weights[e]*abs(c)) for e, col in enumerate(columns)
                        for v, c in col.items() if weights[e] and c)

        def frustration(values):
            # T-G = -2 times opposite orientation products. Separate orientation
            # BEFORE weighting: a negative metric is not a reversed boundary.
            positive, negative = {}, {}
            for e, v, orientation, coefficient in entries:
                rows = positive if orientation else negative
                rows[v] = rows.get(v, zero) + coefficient*values[e]
            out = diagonal.reshape((n,) + (1,)*(values.ndim-1))*values
            for e, v, orientation, coefficient in entries:
                opposite = negative if orientation else positive
                out[e] -= 2*coefficient*opposite.get(v, zero)
            return out
        return frustration
    raise TypeError("normalized G has no certified rational full action")


def build_factored_operator(rex, chan, active_names, traces, *, trace_normalized=True):
    """Return (apply_rl, apply_hat, Bs) through signed and unsigned incidence.

    Each action takes O(nnz(B1)) arithmetic operations per field column. The
    summed action reuses the signed and unsigned passes required by T, raw G
    and F. No relation pair matrix is formed.

    Reproduces the assembled trace normalized channels (T, G, F, C).
    With trace_normalized=False, apply_hat returns the channel before trace
    normalization. The formulas below display unit relation weights; the
    implementation folds relation weights into the T/G factors:
      T·x     = B1ᵀ(B1 x)
      G/L_O·x = |B1|ᵀ(|B1| x)  (raw), or  x - D^-½⊙(|B1|ᵀ(|B1|(D^-½⊙x)))  (normalized)
      F/L_SG·x= diag(F)⊙x + B1ᵀ(B1 x) - |B1|ᵀ(|B1| x)      (F_off = T - G, diag zero)
      C/L_C·x = (diag(C)+diag(G))⊙x - |B1|ᵀ(|B1| x)         (C_off = -G_off counts)
    Diagonals and actions come from incidence passes, O(nnz(B1)) per block column.
    ``chan`` is retained for call compatibility; no assembled channel is required.
    Relation weights enter T/G/F, never C. C respects the selected share/count view.
    """
    _require_channel_source(rex)
    return _build_numerical_factored_operator(
        rex, active_names, traces, trace_normalized=trace_normalized)


def _numerical_channel_factors(rex):
    """Prepare weighted signed, unsigned and unweighted C slot factors.

    Signed slots accumulate with their signs; unsigned slots accumulate their
    magnitudes. Repeated participants remain in the source. C's count reading
    uses one entry per realized unsigned support address.
    """
    from rexgraph.core import _sparse

    _require_numerical_channel_source(rex)
    Bs = NativeSparse(rex._B1_dual)
    Ba = Bs.with_data(abs(Bs.data))
    nE = Bs.shape[1]
    from rexgraph.sparse_character import _channel_metric
    w = _channel_metric(rex)
    w = np.ones(nE) if w is None else np.asarray(w)
    Bw = Bs.with_data(Bs.data * w[Bs.dual.col_idx])
    Aw = Ba.with_data(Ba.data * w[Ba.dual.col_idx])
    Ac = Ba
    if rex.c_channel == 'count':
        support = NativeSparse(_sparse.canonical_dual(Ba.dual))
        Ac = support.with_data(np.ones(support.nnz))
    c_mass = Ac.transpose_apply(Ac.apply(np.ones(nE)))
    # Opposite ORIENTATION mass, regardless of the signs of the metric weights.
    # Self mass cancels the T-G diagonal for repeated signed slots.
    pos = Bs.with_data(np.maximum(Bs.data, 0))
    neg = Bs.with_data(np.maximum(-Bs.data, 0))
    aw = abs(w)
    dLSG = 2 * aw * (pos.transpose_apply(neg.apply(aw)) + neg.transpose_apply(pos.apply(aw)))
    degree = None
    if rex.g_channel != 'raw':
        degree = Aw.transpose_apply(Aw.apply(np.ones(nE, dtype=_f64)))
        if np.any(w < 0):
            raise ValueError('normalized G requires nonnegative relation weights')
        if np.any(~np.isfinite(degree)) or np.any(degree < 0):
            raise ValueError('normalized G requires finite nonnegative row mass')
    return Bs, Bw, Aw, Ac, c_mass, dLSG, degree


def _numerical_channel_diagonals(rex):
    """Read numerical channel diagonals after each factor accumulates its slots."""
    from rexgraph.core import _sparse

    _, Bw, Aw, Ac, c_mass, frustration_mass, degree = _numerical_channel_factors(rex)

    def quadrances(factor):
        return NativeSparse(_sparse.canonical_dual(factor.dual)).column_quadrances()

    t, g = quadrances(Bw), quadrances(Aw)
    f = np.maximum(0, frustration_mass + t - g)
    c = np.maximum(0, c_mass - quadrances(Ac))
    if degree is not None:
        selected_g = np.ones_like(g)
        nz = degree > 0
        selected_g[nz] = np.maximum(0, degree[nz] - g[nz]) / degree[nz]
        g = selected_g
    return dict(zip(CHANNEL_ORDER, (t, g, f, c), strict=True))


def _build_numerical_factored_operator(rex, active_names, traces, *, trace_normalized=True):
    """Apply the assembled numerical channels through stored slot incidences."""
    Bs, Bw, Aw, Ac, c_mass, dLSG, degree = _numerical_channel_factors(rex)
    nE = Bs.shape[1]
    raw = degree is None
    dh = np.zeros(nE, dtype=_f64)
    if not raw:
        nz = degree > 0
        dh[nz] = 1.0 / np.sqrt(degree[nz])
    trmap = dict(zip(active_names, traces, strict=True))
    if any(not np.isfinite(t) or t < 0 for t in traces):
        raise ValueError('channel traces must be finite and nonnegative')

    def chan_mv(name, P):
        if name == 'L1_down':                       # T
            return Bw.transpose_apply(Bw.apply(P))
        if name == 'L_O':                           # G (raw) or normalized L_O
            if raw:
                return Aw.transpose_apply(Aw.apply(P))
            return P - dh[:, None] * Aw.transpose_apply(Aw.apply(dh[:, None] * P))
        if name == 'L_SG':                          # F = T - G (diag zero off), diag(F)
            return dLSG[:, None] * P + Bw.transpose_apply(Bw.apply(P)) - Aw.transpose_apply(Aw.apply(P))
        if name == 'L_C':                           # C = D_C - G_off (counts)
            return c_mass[:, None] * P - Ac.transpose_apply(Ac.apply(P))
        raise ValueError(name)

    def apply_rl(P):
        P = relation_block(P)
        flat = P.ndim == 1
        block = P[:, None] if flat else P
        out = np.zeros_like(block, dtype=np.result_type(P, float))
        down, unsigned = None, None
        for name in active_names:
            tr = trmap[name] if trace_normalized else 1.0
            if not tr:
                continue
            if name in ('L1_down', 'L_SG') and down is None:
                down = Bw.transpose_apply(Bw.apply(block))
            if (name == 'L_SG' or (name == 'L_O' and raw)) and unsigned is None:
                unsigned = Aw.transpose_apply(Aw.apply(block))
            if name == 'L1_down':
                value = down
            elif name == 'L_O' and raw:
                value = unsigned
            elif name == 'L_SG':
                value = dLSG[:, None] * block + down - unsigned
            else:
                value = chan_mv(name, block)
            out += value / tr
        return out[:, 0] if flat else out

    def relation_block(P):
        P = np.asarray(P)
        if P.ndim not in (1, 2) or P.shape[0] != nE:
            raise ValueError('channel action expects a relation vector or block')
        return P

    def apply_hat(name, P):
        P = relation_block(P)
        block = P[:, None] if P.ndim == 1 else P
        tr = trmap[name] if trace_normalized else 1.0
        out = chan_mv(name, block) / tr if tr else np.zeros_like(block, dtype=np.result_type(P, float))
        return out[:, 0] if P.ndim == 1 else out

    return apply_rl, apply_hat, Bs
