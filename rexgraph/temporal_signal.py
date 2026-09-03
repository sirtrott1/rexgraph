"""Temporal delta fields and exact local signal actions.

The temporal store already retains successive relational-complex states.  This
module turns one state transition into a source-bound *field*, without reducing
the transition to a vertex walk or a count of changed records.

``TemporalSignal`` keeps the independent C1 conditions apart:

* existence (birth / death),
* orientation and the exact distinguished head,
* signing (the independent gauge channel), and
* a separately declared numerical relation amplitude.

The structural source is the exact C0 field ``sum(B_current - B_previous)``
over changed relations.  Thus a birth, a death, and a moved head are different
events that nevertheless enter the appropriate boundary tensor directly.
Signing is retained as an event but intentionally has no B1 source: a gauge
change is not a change of boundary geometry.  Amplitude is also separate from
the structural channels; when weights are present it is an approximate measured
field, never relabelled as exact topology.

The first action supplied here is local and exact: ``B1*`` followed by ``B1``.
It is the C0 block of ``D^2`` and gives a graded response through actual
relations, not a search for vertex paths.  Higher-grade temporal identities
need their own stable cell identity carrier before a corresponding temporal
source can be represented honestly, so this module deliberately names its C1
scope rather than fabricating a general-grade claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from rexgraph.cochain import Chain, Cochain

__all__ = [
    "RelationKey",
    "TemporalSignalEvent",
    "TemporalSignal",
    "TemporalSignalFlow",
    "relation_key",
    "relation_identity",
    "temporal_signal",
    "signal_flow",
]


# Anonymous timelines use an exact support tuple rather than the temporal
# kernel's 64-bit hash.  An identified timeline uses its persisted int64 C1
# relation ID instead: equal-support relations are distinct primary cells and
# support is not their temporal identity.
RelationKey = int | tuple[int, ...]
_Channel = Literal["structural", "existence", "geometry", "amplitude", "signing"]
_RelationChannel = Literal["amplitude", "existence", "orientation", "signing"]


@dataclass(frozen=True)
class _BoundaryColumn:
    """One sparse exact C1 boundary, carried as vertex/coefficient pairs."""

    entries: tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True)
class TemporalSignalEvent:
    """One independently-addressable C1 state transition.

    ``key`` is a persisted relation ID when the source timeline carries one;
    otherwise it is the canonical support tuple.  In either mode the head is
    excluded, so orientation changes remain one relation lineage.
    """

    key: RelationKey
    existence: int
    orientation: int
    signing: int
    previous_head: int | None
    head: int | None
    previous_amplitude: float | None
    amplitude: float | None
    boundary_changed: bool

    @property
    def amplitude_delta(self) -> float:
        """Numerical amplitude change on the relation's actual lifetime.

        A missing relation contributes zero, while a present but unweighted
        relation carries unit amplitude.  This matches the amplitude source
        field instead of pretending that a birth replaced an unseen unit cell.
        """
        before = 0.0 if self.previous_head is None else _effective_amplitude(self.previous_amplitude)
        after = 0.0 if self.head is None else _effective_amplitude(self.amplitude)
        return after - before

    @property
    def head_changed(self) -> bool:
        """Whether the exact C1 boundary changed through its distinguished head."""
        return self.previous_head is not None and self.head is not None and self.head != self.previous_head


@dataclass(frozen=True)
class TemporalSignal:
    """The direct C1 signal emitted by one TemporalRex transition.

    ``event(key)`` is an average O(1) exact relation-identity lookup after the
    transition index has been built.  Materializing a boundary is necessarily proportional
    to that relation's arity; applying a field action is proportional to the
    participating boundary entries.  Those costs are stated rather than hidden
    behind an O(1) label.
    """

    source: Any
    step: int
    when: float
    previous: Any
    current: Any
    events: tuple[TemporalSignalEvent, ...]
    vertex_count: int
    _positions: Mapping[RelationKey, int] = field(repr=False, compare=False)
    _previous_columns: Mapping[RelationKey, _BoundaryColumn] = field(repr=False, compare=False)
    _current_columns: Mapping[RelationKey, _BoundaryColumn] = field(repr=False, compare=False)
    _current_keys: tuple[RelationKey, ...] = field(repr=False, compare=False)

    @property
    def keys(self) -> tuple[RelationKey, ...]:
        """Changed relation identities in canonical support order."""
        return tuple(event.key for event in self.events)

    @property
    def current_keys(self) -> tuple[RelationKey, ...]:
        """The current C1 basis, in its stored order, for the local response."""
        return self._current_keys

    def event(self, key: RelationKey | list[int] | np.ndarray) -> TemporalSignalEvent:
        """Look up one changed relation by exact C1 identity in average O(1)."""
        canonical = _canonical_key(key)
        try:
            return self.events[self._positions[canonical]]
        except KeyError as exc:
            raise KeyError(f"no temporal signal event for relation support {canonical!r}") from exc

    def source_field(self, channel: _Channel = "structural") -> Chain:
        """Materialize one C0 source field from the chosen temporal channel.

        ``structural`` is exact and includes all changed boundary columns;
        ``existence`` contains only births and deaths; ``geometry`` contains
        persistent boundary changes, including a head move whose coarse
        orientation polarity stayed unchanged.  ``signing`` is an exact zero
        B1 source while retaining its nonzero event channel in ``events``.
        ``amplitude`` preserves declared relation weights as a numerical field.
        """
        channel = _check_channel(channel)
        exact = self.is_exact(channel)
        if exact:
            out: list[Fraction] | np.ndarray = [Fraction(0) for _ in range(self.vertex_count)]
        else:
            out = np.zeros(self.vertex_count, dtype=np.float64)

        for event in self.events:
            previous = self._previous_columns.get(event.key)
            current = self._current_columns.get(event.key)
            if channel == "structural":
                _accumulate(out, current, 1, exact=exact)
                _accumulate(out, previous, -1, exact=exact)
            elif channel == "existence":
                if event.existence > 0:
                    _accumulate(out, current, 1, exact=exact)
                elif event.existence < 0:
                    _accumulate(out, previous, -1, exact=exact)
            elif channel == "geometry":
                if event.existence == 0 and event.boundary_changed:
                    _accumulate(out, current, 1, exact=exact)
                    _accumulate(out, previous, -1, exact=exact)
            elif channel == "amplitude":
                _accumulate(out, current, _effective_amplitude(event.amplitude), exact=exact)
                _accumulate(out, previous, -_effective_amplitude(event.previous_amplitude), exact=exact)
            # signing has no B1 source.  Its exact event values remain queryable
            # above; treating a gauge update as altered boundary geometry would
            # conflate the two channels the temporal tensor separates.

        values = np.asarray(out, dtype=object if exact else np.float64)
        return Chain(0, values, cell_keys=tuple(range(self.vertex_count)), source=self.source)

    def is_exact(self, channel: _Channel = "structural") -> bool:
        """Whether the requested source was constructed from exact structure only."""
        channel = _check_channel(channel)
        if channel != "amplitude":
            return True
        return all(event.previous_amplitude is None and event.amplitude is None for event in self.events)

    def relation_field(self, channel: _RelationChannel = "amplitude") -> Cochain:
        """Return the direct current-C1 temporal field, without taking a boundary.

        This is distinct from :meth:`source_field`.  A C0 source followed by
        ``B1*`` is necessarily a gradient response; the direct C1 delta is the
        object that can legitimately have gradient, curl, and harmonic sectors.
        Its ordered basis is the current snapshot's actual relation basis.  A
        disappeared relation has no coordinate in that basis and remains in the
        full C0 temporal source instead of being silently assigned to another
        current cell.

        ``amplitude`` is the current relation amplitude minus the previous one,
        with absence treated as zero.  The three binary channels retain their
        independently recorded values and do not invent a scalar for exact head
        motion; the head itself remains available from ``event(key)``.
        """
        channel = _check_relation_channel(channel)
        exact = channel != "amplitude" or self.is_exact("amplitude")
        by_key = {event.key: event for event in self.events}
        values: list[Fraction | float | int] = []
        for key in self.current_keys:
            event = by_key.get(key)
            if event is None:
                values.append(Fraction(0) if exact else 0.0)
                continue
            if channel == "amplitude":
                if exact:
                    before = Fraction(0) if event.previous_head is None else Fraction(1)
                    after = Fraction(0) if event.head is None else Fraction(1)
                    values.append(after - before)
                else:
                    before = 0.0 if event.previous_head is None else _effective_amplitude(event.previous_amplitude)
                    after = 0.0 if event.head is None else _effective_amplitude(event.amplitude)
                    values.append(after - before)
            else:
                values.append(int(getattr(event, channel)))
        dtype = object if exact and channel == "amplitude" else (np.int64 if exact else np.float64)
        return Cochain(1, np.asarray(values, dtype=dtype),
                       cell_keys=self.current_keys, source=self.current)


@dataclass(frozen=True)
class TemporalSignalFlow:
    """One local graded response to a temporal source field.

    ``relation_response`` is ``B1* source`` on the current C1 basis and
    ``returned_boundary`` is ``B1 relation_response``.  Together they form the
    C0 ``D^2`` response.  The carrier keeps each grade and basis attached, so
    equal-length arrays cannot be mistaken for the same field space.
    """

    signal: TemporalSignal
    channel: str
    source_field: Chain
    relation_response: Cochain
    returned_boundary: Chain
    exact: bool


def relation_key(rex: Any, index: int) -> RelationKey:
    """Return the exact orientation-independent C1 support identity for one cell."""
    rex._ensure_clean()
    index = int(index)
    ptr = np.asarray(rex._boundary_ptr)
    idx = np.asarray(rex._boundary_idx)
    if index < 0 or index + 1 >= ptr.size:
        raise ValueError(f"C1 cell index {index} is not present")
    return _canonical_key(idx[int(ptr[index]):int(ptr[index + 1])])


def relation_identity(rex: Any, index: int) -> RelationKey:
    """Return the persisted C1 ID when present, otherwise the exact support key."""
    rex._ensure_clean()
    index = int(index)
    identities = getattr(rex, "relation_ids", None)
    if identities is not None:
        if index < 0 or index >= len(identities):
            raise ValueError(f"C1 cell index {index} is not present")
        return int(identities[index])
    return relation_key(rex, index)


def temporal_signal(temporal: Any, step: int) -> TemporalSignal:
    """Build the exact C1 delta field from ``step - 1`` to ``step``.

    This reads snapshots through TemporalRex's public reconstruction method,
    so checkpoint versus delta-backed storage does not change the signal.  The
    function requires a continuous C0 index space, as TemporalRex itself does;
    a newly visible vertex simply enlarges the direct source basis to the union
    size for this one transition.
    """
    if not hasattr(temporal, "reconstruct_at") or not hasattr(temporal, "T"):
        raise TypeError("TEMPORAL_DELTA expects a TemporalRex-like source")
    step = int(step)
    if step <= 0 or step >= int(temporal.T):
        raise ValueError(f"temporal signal step must lie in [1, {int(temporal.T) - 1}], got {step}")

    previous = temporal.reconstruct_at(step - 1)
    current = temporal.reconstruct_at(step)
    previous_cells = _snapshot_cells(previous)
    current_cells = _snapshot_cells(current)
    all_keys = tuple(sorted(set(previous_cells) | set(current_cells)))
    events: list[TemporalSignalEvent] = []
    previous_columns: dict[RelationKey, _BoundaryColumn] = {}
    current_columns: dict[RelationKey, _BoundaryColumn] = {}

    for key in all_keys:
        before = previous_cells.get(key)
        after = current_cells.get(key)
        if before is not None:
            previous_columns[key] = before.column
        if after is not None:
            current_columns[key] = after.column

        existence = 1 if before is None else (-1 if after is None else 0)
        orientation = 0
        signing = 0
        if before is not None and after is not None:
            orientation = (after.polarity - before.polarity) // 2
            signing = (after.sign - before.sign) // 2
        before_amplitude = None if before is None else before.amplitude
        after_amplitude = None if after is None else after.amplitude
        amplitude_changed = before is not None and after is not None and (
            _effective_amplitude(before_amplitude) != _effective_amplitude(after_amplitude)
        )
        boundary_changed = before is not None and after is not None and before.column != after.column
        if existence or orientation or signing or boundary_changed or amplitude_changed:
            events.append(TemporalSignalEvent(
                key=key,
                existence=existence,
                orientation=orientation,
                signing=signing,
                previous_head=None if before is None else before.head,
                head=None if after is None else after.head,
                previous_amplitude=before_amplitude,
                amplitude=after_amplitude,
                boundary_changed=boundary_changed,
            ))

    positions = {event.key: i for i, event in enumerate(events)}
    when = float(temporal.time_at(step)) if hasattr(temporal, "time_at") else float(step)
    return TemporalSignal(
        source=temporal,
        step=step,
        when=when,
        previous=previous,
        current=current,
        events=tuple(events),
        vertex_count=max(int(previous.nV), int(current.nV)),
        _positions=positions,
        _previous_columns=previous_columns,
        _current_columns=current_columns,
        _current_keys=tuple(current_cells),
    )


def signal_flow(signal: TemporalSignal, channel: _Channel = "structural") -> TemporalSignalFlow:
    """Apply the exact current ``B1*`` then ``B1`` action to a delta source.

    No adjacency matrix, path enumeration, damping factor, threshold, dense
    factorization, or eigensolve appears here.  This is a direct sparse
    contraction over the current C1 boundary entries.  Its cost is linear in
    the current C1 incidence count, which is the actual work being requested.
    """
    if not isinstance(signal, TemporalSignal):
        raise TypeError("SIGNAL_FLOW expects a TemporalSignal")
    channel = _check_channel(channel)
    source = signal.source_field(channel)
    exact = signal.is_exact(channel)
    source_values = source.values
    coefficients: list[Fraction | float] = []
    for key in signal.current_keys:
        column = signal._current_columns[key]
        if exact:
            coefficients.append(sum(
                (coefficient * source_values[vertex] for vertex, coefficient in column.entries),
                Fraction(0),
            ))
        else:
            coefficients.append(sum(
                float(coefficient) * float(source_values[vertex])
                for vertex, coefficient in column.entries
            ))
    response_values = np.asarray(coefficients, dtype=object if exact else np.float64)
    response = Cochain(1, response_values, cell_keys=signal.current_keys, source=signal.source)

    if exact:
        returned: list[Fraction] | np.ndarray = [Fraction(0) for _ in range(signal.vertex_count)]
    else:
        returned = np.zeros(signal.vertex_count, dtype=np.float64)
    for coefficient, key in zip(response_values, signal.current_keys, strict=True):
        _accumulate(returned, signal._current_columns[key], coefficient, exact=exact)
    returned_values = np.asarray(returned, dtype=object if exact else np.float64)
    return TemporalSignalFlow(
        signal=signal,
        channel=channel,
        source_field=source,
        relation_response=response,
        returned_boundary=Chain(
            0,
            returned_values,
            cell_keys=tuple(range(signal.vertex_count)),
            source=signal.source,
        ),
        exact=exact,
    )


@dataclass(frozen=True)
class _SnapshotCell:
    head: int
    polarity: int
    sign: int
    amplitude: float | None
    column: _BoundaryColumn


def _snapshot_cells(rex: Any) -> dict[RelationKey, _SnapshotCell]:
    """Index a snapshot by exact persisted ID or exact support when anonymous."""
    rex._ensure_clean()
    ptr = np.asarray(rex._boundary_ptr)
    idx = np.asarray(rex._boundary_idx)
    signs = getattr(rex, "_signs", None)
    amplitudes = getattr(rex, "_w_E", None)
    identities = getattr(rex, "relation_ids", None)
    out: dict[RelationKey, _SnapshotCell] = {}
    for index in range(int(ptr.size - 1)):
        support = tuple(int(vertex) for vertex in idx[int(ptr[index]):int(ptr[index + 1])])
        key = int(identities[index]) if identities is not None else _canonical_key(support)
        if key in out:
            if identities is None:
                raise ValueError(
                    "temporal signal refuses parallel C1 cells with equal support: "
                    "TemporalRex needs a stable relation identity beyond support before "
                    "their independent deltas can be represented"
                )
            raise ValueError(
                "temporal signal found a repeated C1 identity in one snapshot; "
                "each live relation must have exactly one stable ID"
            )
        if not support:
            raise ValueError("temporal signal cannot form a C1 boundary for an empty relation")
        head = support[0]
        base = min(support)
        sign = 1 if signs is None else int(np.asarray(signs).ravel()[index])
        amplitude = None if amplitudes is None else float(np.asarray(amplitudes).ravel()[index])
        out[key] = _SnapshotCell(
            head=head,
            polarity=1 if head == base else -1,
            sign=sign,
            amplitude=amplitude,
            column=_column(support),
        )
    return out


def _column(support: tuple[int, ...]) -> _BoundaryColumn:
    """Construct one exact C1 boundary directly from declared incidence."""
    coefficients: dict[int, Fraction] = {}
    arity = len(support)
    if arity == 1:
        coefficients[support[0]] = Fraction(1)
    else:
        coefficients[support[0]] = Fraction(-1)
        share = Fraction(1, arity - 1)
        for vertex in support[1:]:
            coefficients[vertex] = coefficients.get(vertex, Fraction(0)) + share
    return _BoundaryColumn(tuple(
        (vertex, coefficient)
        for vertex, coefficient in sorted(coefficients.items())
        if coefficient
    ))


def _canonical_key(values: RelationKey | list[int] | np.ndarray) -> RelationKey:
    # Repetition is retained in the identity.  It is not a binary mask: a
    # relation whose declared boundary lists a vertex twice remains a distinct
    # C1 cell even when some coefficients cancel in its flattened B1 column.
    if isinstance(values, (int, np.integer)):
        return int(values)
    return tuple(sorted(int(value) for value in values))


def _effective_amplitude(value: float | None) -> float:
    """The unweighted structural field carries unit relation amplitude."""
    return 1.0 if value is None else float(value)


def _check_channel(channel: str) -> _Channel:
    channel = str(channel).lower()
    if channel not in {"structural", "existence", "geometry", "amplitude", "signing"}:
        raise ValueError(
            "signal channel must be structural, existence, geometry, amplitude, or signing"
        )
    return channel  # type: ignore[return-value]


def _check_relation_channel(channel: str) -> _RelationChannel:
    channel = str(channel).lower()
    if channel not in {"amplitude", "existence", "orientation", "signing"}:
        raise ValueError("relation signal channel must be amplitude, existence, orientation, or signing")
    return channel  # type: ignore[return-value]


def _accumulate(out, column: _BoundaryColumn | None, multiplier, *, exact: bool) -> None:
    """Accumulate one sparse boundary column into an exact or numerical field."""
    if column is None:
        return
    if exact:
        factor = multiplier if isinstance(multiplier, Fraction) else Fraction(multiplier)
        for vertex, coefficient in column.entries:
            out[vertex] += factor * coefficient
    else:
        factor = float(multiplier)
        for vertex, coefficient in column.entries:
            out[vertex] += factor * float(coefficient)
