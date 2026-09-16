"""Sparse C1 value alignment on a native timeline's declared relation identities."""
from fractions import Fraction
from numbers import Integral

import numpy as np

from rexgraph.graph import TemporalRex
from rexgraph.temporal_signal import relation_identity
from rexgraph.type_accession import _scalar

__all__ = ["align_by_lineage", "existence_history", "lifetime_labels"]


def history_key(key):
    if key is None:
        return None
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    if isinstance(key, (list, tuple)) and key and all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in key):
        return tuple(sorted(key))
    raise TypeError("history key must be an integer relation ID or a nonempty support tuple")


def step_interval(start, stop, count):
    stop = count if stop is None else stop
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, stop)) or not 0 <= start <= stop <= count:
        raise ValueError("history interval requires 0 <= start <= stop <= snapshot count")
    return start, stop


def _frames(source, start, stop):
    identity_mode = None
    for step in range(start, stop):
        rex = source.reconstruct_at(step)
        keys = tuple(relation_identity(rex, i) for i in range(rex.nE))
        if len(set(keys)) != len(keys):
            raise ValueError("lineage reading requires unique C1 identities; anonymous parallel cells are ambiguous")
        mode = "relation_ids" if rex.relation_ids is not None else "support"
        if keys and identity_mode is not None and mode != identity_mode:
            raise ValueError("alignment cannot mix persisted IDs and anonymous support identities")
        if keys:
            identity_mode = mode
        yield step, keys, identity_mode


def alignment_values(source, values, start=0, stop=None):
    """Check declared values and bounds without reconstructing any snapshot."""
    if not isinstance(source, TemporalRex):
        raise TypeError("lineage alignment requires a native TemporalRex")
    stop = source.T if stop is None else stop
    if (any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in (start, stop))
            or not 0 <= start <= stop <= source.T):
        raise ValueError("alignment requires 0 <= start <= stop <= snapshot count")
    if not isinstance(values, (list, tuple)) or len(values) != stop - start:
        raise ValueError("alignment requires one C1 value vector per selected snapshot")
    vectors = []
    for vector in values:
        if not isinstance(vector, (list, tuple, np.ndarray)):
            raise TypeError("alignment values must be explicit canonical C1 vectors")
        array = np.asarray(vector, dtype=object)
        if array.ndim != 1:
            raise ValueError("alignment requires scalar C1 vectors, not feature blocks")
        items = []
        for raw in array:
            value = _scalar(raw)
            items.append(int(raw) if isinstance(raw, Integral) else value)
        vectors.append(tuple(items))
    return tuple(vectors), int(start), int(stop)


def align_by_lineage(source, values, *, start=0, stop=None):
    """Align scalar C1 readings over [start, stop) without a dense union array.

    Input vectors name the canonical C1 order returned by reconstruct_at(step).
    Persisted relation IDs take precedence; anonymous sources use Core's exact
    support keys and refuse ambiguous parallel cells. Coefficients are not
    transported or sign adjusted. This is identity alignment, not a chain map.

    ``presence`` carries every existing cell, including measured zeros. Sparse
    ``entries`` carries only nonzero readings. Outside presence the value is
    missing, not a measured zero. Union keys follow first appearance.
    """
    vectors, start, stop = alignment_values(source, values, start, stop)
    positions, presence, entries, maps = {}, [], [], []
    identity_mode = None
    for row, ((step, keys, identity_mode), vector) in enumerate(zip(_frames(source, start, stop), vectors, strict=True)):
        if len(vector) != len(keys):
            raise ValueError(f"alignment step {step} requires {len(keys)} canonical C1 values")
        local = []
        for key, value in zip(keys, vector, strict=True):
            index = positions.setdefault(key, len(positions))
            local.append(index)
            presence.append((row, index))
            if value:
                entries.append((row, index, value))
        maps.append(tuple(local))
    domain = "real" if any(isinstance(v, float) for vector in vectors for v in vector) else (
        "Q" if any(isinstance(v, Fraction) for vector in vectors for v in vector) else "Z")
    return {"keys": tuple(positions), "entries": tuple(entries), "presence": tuple(presence),
            "cell_maps": tuple(maps), "shape": (stop-start, len(positions)),
            "steps": tuple(range(start, stop)), "times": tuple(source.time_at(t) for t in range(start, stop)),
            "identity": identity_mode or "empty", "coefficient_domain": domain,
            "implicit_zero": 0, "missing": "outside-presence"}


def _history(source, key, start, stop, *, neighbors):
    if not isinstance(source, TemporalRex):
        raise TypeError("temporal reading requires a native TemporalRex")
    key = history_key(key)
    start, stop = step_interval(start, stop, source.T)
    lo, hi = (max(0, start-1), min(source.T, stop+1)) if neighbors and start < stop else (start, stop)
    by_step = {step: frozenset(keys) for step, keys, _ in _frames(source, lo, hi)}
    keys = (key,) if key is not None else tuple(sorted({k for t in range(start, stop) for k in by_step[t]}))
    index = {key: j for j, key in enumerate(keys)}
    entries = tuple((t-start, index[k], 1) for t in range(start, stop) for k in sorted(by_step[t]) if k in index)
    return {"keys": keys, "entries": entries, "shape": (stop-start, len(keys)),
            "steps": tuple(range(start, stop)), "times": tuple(source.time_at(t) for t in range(start, stop)),
            "basis": "C1-identities-by-step", "implicit_zero": 0}, by_step


def existence_history(source, key=None, start=0, stop=None):
    """Sparse existence on [start, stop); no neighboring frame is reconstructed."""
    return _history(source, key, start, stop, neighbors=False)[0]


def lifetime_labels(source, key=None, start=0, stop=None):
    """BIOES labels for existing cells, preserving lifetimes across window bounds."""
    history, by_step = _history(source, key, start, stop, neighbors=True)
    labels = []
    for row, column, _ in history["entries"]:
        step, identity = history["steps"][row], history["keys"][column]
        previous = identity in by_step.get(step-1, ())
        following = identity in by_step.get(step+1, ())
        label = "I" if previous and following else "E" if previous else "B" if following else "S"
        labels.append((row, column, label))
    return {**history, "entries": tuple(labels), "implicit_zero": "O"}
