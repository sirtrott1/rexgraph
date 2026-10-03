"""Source bound selections and checked lineage for the complete cell tower.

Restrictions delegate to the registered Core partition/transport path. Maps
retain original basis addresses; neither identities nor boundary coefficients
are inferred from a graph projection. Lineage digests are content identities,
not authorization or signatures.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hmac

import numpy as np

from .components.transport import CellMaps
from .cells import _integer_coordinate
from .identity import manifest_digest
from .object_identity import object_digest

__all__ = ["Selection", "CellMaps", "Lineage", "restrict", "glue"]


def _sizes(source):
    from .partition_state import partition_tower
    boundaries, _ = partition_tower(source)
    return (int(boundaries[0].shape[0]), *(int(matrix.shape[1]) for matrix in boundaries))


def _index(value, name):
    return _integer_coordinate(value, name)


def _indices(values, size):
    array = tuple(_index(value, "cell index") for value in values)
    if any(index < 0 or index >= size for index in array):
        raise ValueError("selection index is outside its source basis")
    return tuple(sorted(set(array)))


def _masks(sizes, indices):
    output = []
    for size, selected in zip(sizes, indices, strict=True):
        mask = np.zeros(size, dtype=np.uint8)
        mask[list(selected)] = 1
        output.append(mask)
    return tuple(output)


def _digest(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {name} digest")


@dataclass(frozen=True, init=False)
class Selection:
    """A canonical multi grade request bound to one unchanged source state.

    ``Selection(rex, {0: [isolated_vertex], 3: [process]})`` selects original
    cell addresses. ``from_masks`` accepts explicitly binary vectors, checked
    before conversion. Missing grades request no cells; restriction supplies
    the complete downward closure. The source remains a live capability.
    """
    source: object = field(compare=False, repr=False)
    source_state: str
    source_sizes: tuple[int, ...]
    indices: tuple[tuple[int, ...], ...]

    def __init__(self, source, cells=None):
        if cells is not None and not isinstance(cells, Mapping):
            raise TypeError("selection cells must map grades to original indices")
        sizes = _sizes(source)
        selected = [()] * len(sizes)
        for raw_grade, values in (cells or {}).items():
            grade = _index(raw_grade, "selection grade")
            if not 0 <= grade < len(sizes):
                raise ValueError("selection grade is outside the carried source tower")
            selected[grade] = _indices(values, sizes[grade])
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_state", object_digest(source))
        object.__setattr__(self, "source_sizes", sizes)
        object.__setattr__(self, "indices", tuple(selected))

    @classmethod
    def from_masks(cls, source, masks):
        from .partition_state import _mask
        if not isinstance(masks, Mapping):
            raise TypeError("selection masks must map grades to binary vectors")
        sizes = _sizes(source)
        cells = {}
        for raw_grade, values in masks.items():
            grade = _index(raw_grade, "selection grade")
            if not 0 <= grade < len(sizes):
                raise ValueError("selection grade is outside the carried source tower")
            cells[grade] = tuple(map(int, np.flatnonzero(_mask(values, sizes[grade], grade))))
        return cls(source, cells)

    @classmethod
    def from_cells(cls, value):
        from .cells import Cell, CellSet, GradedCellPattern
        if isinstance(value, Cell):
            return cls(value.source, {value.grade: (value.index,)})
        if isinstance(value, CellSet):
            return cls(value.source, {value.grade: value.indices})
        if isinstance(value, GradedCellPattern):
            return cls(value.source, {cells.grade: cells.indices for cells in value.cell_sets})
        raise TypeError("selection requires a Cell, CellSet or GradedCellPattern")

    def check_state(self, source=None):
        if source is not None and source is not self.source:
            raise ValueError("selection belongs to another source; bind it explicitly")
        if object_digest(self.source) != self.source_state:
            raise ValueError("selection source changed after its basis was captured")

    def bind(self, source):
        """Explicitly rebind to a source carrying the identical sealed state."""
        if object_digest(source) != self.source_state:
            raise ValueError("selection cannot bind to a different source state")
        return Selection(source, dict(enumerate(self.indices)))

    def as_pattern(self):
        """Adapt the same request to existing relative coordinate operations."""
        from .cells import CellSet, GradedCellPattern
        self.check_state()
        return GradedCellPattern(self.source, tuple(CellSet(self.source, grade, indices)
            for grade, indices in enumerate(self.indices)))

    @property
    def masks(self):
        self.check_state()
        return _masks(self.source_sizes, self.indices)

    @property
    def request_digest(self):
        from .partition_state import _selection_digest_indices
        self.check_state()
        return _selection_digest_indices(self.source_sizes, self.indices)

    def as_record(self):
        self.check_state()
        return {"object_type": "Selection", "version": 1, "source_state": self.source_state,
                "source_sizes": self.source_sizes, "indices": self.indices}

    @property
    def digest(self):
        return manifest_digest(self.as_record())

    @classmethod
    def from_record(cls, record, source):
        if (not isinstance(record, dict) or set(record) != {"object_type", "version", "source_state", "source_sizes", "indices"}
                or record["object_type"] != "Selection" or type(record["version"]) is not int or record["version"] != 1
                or not isinstance(record["indices"], tuple)):
            raise ValueError("invalid selection record")
        _digest(record["source_state"], "selection source")
        value = cls(source, dict(enumerate(record["indices"])))
        if value.as_record() != record:
            raise ValueError("selection record does not match its canonical source basis")
        return value


@dataclass(frozen=True)
class Lineage:
    """A sealed restriction declaration with both requested and retained cells.

    ``cell_maps[k][i]`` is result cell i's original source address.
    ``old_to_new[k][j]`` is source cell j's result address, or -1 if removed.
    ``verify(source, result)`` reapplies the declaration through registered
    transport and checks exact state identity, including carried state.
    """
    source_state: str
    result_state: str
    source_sizes: tuple[int, ...]
    result_sizes: tuple[int, ...]
    requested: tuple[tuple[int, ...], ...]
    cell_maps: tuple[tuple[int, ...], ...]
    policy_digest: str = ""
    carried_state: str = "structural"
    parents: tuple[str, ...] = ()
    selection_digest: str = ""

    def __post_init__(self):
        _digest(self.source_state, "source state")
        _digest(self.result_state, "result state")
        _digest(self.selection_digest, "requested selection")
        for sizes in (self.source_sizes, self.result_sizes):
            if (not isinstance(sizes, tuple) or len(sizes) < 2
                    or any(type(size) is not int or not 0 <= size < 2**31 for size in sizes)):
                raise ValueError("lineage requires exact native grade sizes")
        if len(self.source_sizes) != len(self.result_sizes):
            raise ValueError("lineage source and result towers must have the same grades")
        for field_name in ("requested", "cell_maps"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or len(value) != len(self.source_sizes):
                raise ValueError("lineage cell maps must cover the complete source tower")
            for selected, size in zip(value, self.source_sizes, strict=True):
                if (not isinstance(selected, tuple) or any(type(i) is not int for i in selected)
                        or _indices(selected, size) != selected):
                    raise ValueError("lineage cell addresses must be canonical original indices")
        if tuple(map(len, self.cell_maps)) != self.result_sizes:
            raise ValueError("lineage cell maps do not cover the result basis")
        if any(not set(requested) <= set(retained) for requested, retained in zip(self.requested, self.cell_maps, strict=True)):
            raise ValueError("lineage omits a requested cell")
        if self.carried_state not in {"structural", "all"} or type(self.policy_digest) is not str:
            raise ValueError("invalid lineage transport policy")
        if not isinstance(self.parents, tuple):
            raise ValueError("invalid lineage parent inventory")
        for parent in self.parents:
            _digest(parent, "parent lineage")

    @property
    def old_to_new(self):
        maps = {}
        for grade, (size, selected) in enumerate(zip(self.source_sizes, self.cell_maps, strict=True)):
            mapping = np.full(size, -1, dtype=np.int64)
            mapping[list(selected)] = np.arange(len(selected))
            maps[grade] = mapping
        return CellMaps(maps, required_grades=tuple(range(len(self.source_sizes))))

    def _content(self):
        return {"object_type": "Lineage", "version": 1, "source_state": self.source_state,
                "result_state": self.result_state, "source_sizes": self.source_sizes,
                "result_sizes": self.result_sizes, "requested": self.requested,
                "cell_maps": self.cell_maps, "policy_digest": self.policy_digest,
                "carried_state": self.carried_state, "parents": self.parents,
                "selection_digest": self.selection_digest}

    @property
    def digest(self):
        return manifest_digest(self._content())

    def as_record(self):
        return {**self._content(), "digest": self.digest}

    @classmethod
    def from_record(cls, record):
        expected = {"object_type", "version", "digest", "source_state", "result_state", "source_sizes",
                    "result_sizes", "requested", "cell_maps", "policy_digest", "carried_state", "parents", "selection_digest"}
        if (not isinstance(record, dict) or set(record) != expected or record["object_type"] != "Lineage"
                or type(record["version"]) is not int or record["version"] != 1):
            raise ValueError("invalid lineage record")
        value = cls(**{key: record[key] for key in expected - {"object_type", "version", "digest"}})
        _digest(record["digest"], "lineage")
        if not hmac.compare_digest(value.digest, record["digest"]):
            raise ValueError("lineage content digest mismatch")
        return value

    def to_bytes(self):
        from .value_codec import pack_value
        return pack_value(self.as_record())

    @classmethod
    def from_bytes(cls, payload):
        from .value_codec import unpack_value
        if len(payload) > 16*1024*1024:
            raise ValueError("lineage artifact exceeds its byte limit")
        return cls.from_record(unpack_value(payload))

    def verify(self, source, result):
        if object_digest(source) != self.source_state or object_digest(result) != self.result_state:
            raise ValueError("lineage source or result state changed")
        if _sizes(source) != self.source_sizes or _sizes(result) != self.result_sizes:
            raise ValueError("lineage grade sizes do not match their states")
        rebuilt = restrict(source, Selection(source, dict(enumerate(self.requested))),
                           carried_state=self.carried_state, policy_digest=self.policy_digest)
        if (rebuilt.cell_maps != self.cell_maps or rebuilt.state.result_state != self.result_state
                or rebuilt.state.selection_digest != self.selection_digest):
            raise ValueError("lineage does not reproduce its declared exact restriction")
        return True

    def compose(self, following):
        """Compose two basis maps only when their intermediate states agree."""
        if (not isinstance(following, Lineage) or self.result_state != following.source_state
                or self.result_sizes != following.source_sizes):
            raise ValueError("lineage composition requires the identical intermediate state")
        maps = tuple(tuple(parent[index] for index in child)
                     for parent, child in zip(self.cell_maps, following.cell_maps, strict=True))
        mode = "all" if self.carried_state == following.carried_state == "all" else "structural"
        from .partition_state import _selection_digest_indices
        return Lineage(self.source_state, following.result_state, self.source_sizes,
                       following.result_sizes, maps, maps, following.policy_digest, mode,
                       (self.digest, following.digest), _selection_digest_indices(self.source_sizes, maps))


def restrict(source, selection, *, carried_state="structural", policy_digest="", closure="subcomplex"):
    """Return RexPartition with its owned graph, both cell maps and sealed lineage."""
    from .partition_state import build_rex_partition
    if not isinstance(selection, Selection):
        selection = Selection.from_cells(selection)
    selection.check_state(source)
    masks = selection.masks
    return build_rex_partition(source, masks[1], v_mask=masks[0],
        grade_masks=dict(enumerate(masks[2:], 2)), closure=closure,
        carried_state=carried_state, policy_digest=policy_digest)


def glue(parts, *, source, carried_state=None, policy_digest=""):
    """Reconstruct the union of verified restrictions in their original basis.

    The original source is explicit authority for overlapping state. Every part
    must reproduce from it; modified or foreign parts are refused. No original
    source payload is hidden inside a portable partition. Arbitrary edited
    complexes require a separate, explicitly declared merge contract.
    """
    from .partition_state import RexPartition
    parts = tuple(parts)
    if not parts or any(not isinstance(part, RexPartition) for part in parts):
        raise TypeError("glue requires a nonempty family of RexPartition values")
    modes = {part.lineage.carried_state for part in parts}
    if carried_state is None:
        if len(modes) != 1:
            raise ValueError("mixed carried-state parts require an explicit glue policy")
        carried_state = next(iter(modes))
    if carried_state not in {"structural", "all"}:
        raise ValueError("glue carried_state must be structural or all")
    if carried_state == "all" and "structural" in modes:
        raise ValueError("glue cannot restore omitted application state without an explicit derivation")
    cells = {}
    parents = []
    for part in parts:
        lineage = part.lineage
        lineage.verify(source, part.rex)
        for grade, indices in enumerate(lineage.cell_maps):
            cells.setdefault(grade, set()).update(indices)
        parents.append(lineage.digest)
    result = restrict(source, Selection(source, cells), carried_state=carried_state, policy_digest=policy_digest)
    return replace(result, lineage_parents=tuple(sorted(set(parents))))
