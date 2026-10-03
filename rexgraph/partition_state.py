"""Canonical lineage and arbitrary grade closure for derived Rex partitions."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .identity import manifest_digest

PARTITION_VERSION = 1

__all__ = [
    "PARTITION_VERSION",
    "PartitionState",
    "RexPartition",
    "build_rex_partition",
    "faces_in_support",
    "partition_tower",
    "partition_policy",
    "partition_from_policy",
]


@dataclass(frozen=True)
class PartitionState:
    """Identify a derived partition without embedding source data."""

    source_state: str
    result_state: str
    selection_digest: str
    policy_digest: str = ""
    closure: str = "projection"

    def manifest(self) -> dict[str, Any]:
        """Return canonical lineage fields."""
        return {
            "closure": self.closure,
            "policy_digest": self.policy_digest,
            "result_state": self.result_state,
            "selection_digest": self.selection_digest,
            "source_state": self.source_state,
            "version": PARTITION_VERSION,
        }

    @property
    def digest(self) -> str:
        """Return the stable partition lineage identity."""
        return manifest_digest({"object_type": "PartitionState", **self.manifest()})


@dataclass(frozen=True)
class RexPartition:
    """One derived RexGraph together with canonical source and policy lineage."""

    rex: object
    state: PartitionState
    cell_maps: tuple[tuple[int, ...], ...] = ()
    source_sizes: tuple[int, ...] = ()
    requested: tuple[tuple[int, ...], ...] = ()
    carried_state: str = "structural"
    lineage_parents: tuple[str, ...] = ()

    @property
    def lineage(self):
        from .selection import Lineage, _sizes
        self.check_state()
        if not self.source_sizes:
            raise ValueError("legacy partition has no certified original source basis")
        return Lineage(self.state.source_state, self.state.result_state, self.source_sizes,
                       _sizes(self.rex), self.requested, self.cell_maps,
                       self.state.policy_digest, self.carried_state, self.lineage_parents, self.state.selection_digest)

    @property
    def old_to_new(self):
        return self.lineage.old_to_new

    def check_state(self):
        """Refuse lineage for a result modified after its construction."""
        from .object_identity import object_digest
        if object_digest(self.rex) != self.state.result_state:
            raise ValueError("partition result changed after its lineage was captured")

    @property
    def manifest(self):
        self.check_state()
        return self.state.manifest()

    @property
    def digest(self):
        self.check_state()
        return self.state.digest


def _selection_field(digest, grade, size, chunks):
    if grade == 0:
        digest.update(b"vertices\x00")
    else:
        name = "edges" if grade == 1 else "faces" if grade == 2 else f"grade{grade}"
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
    digest.update(size.to_bytes(8, "big"))
    for chunk in chunks:
        digest.update(chunk)


def _selection_digest(masks: list[np.ndarray]) -> str:
    """Bind requested masks without conflating closure added lower cells."""
    digest = hashlib.sha256()
    digest.update(b"rexgraph-partition-selection\x00")
    # The archived grade two framing always included an empty face mask even for a
    # 1 rex. Preserve that identity, then extend it monotonically with grade3+ fields.
    selection_masks = list(masks[1:])
    if len(selection_masks) == 1:
        selection_masks.append(np.zeros(0, dtype=np.uint8))
    for grade, value in enumerate(selection_masks, start=1):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
        _selection_field(digest, grade, array.size, (memoryview(array),))
    # Extend legacy identities only when the caller explicitly selects vertices.
    if np.any(masks[0]):
        array = np.ascontiguousarray(masks[0], dtype=np.uint8)
        _selection_field(digest, 0, array.size, (memoryview(array),))
    return digest.hexdigest()


def _selection_digest_indices(sizes, indices):
    """Hash canonical sparse addresses in legacy mask framing with bounded memory."""
    def chunks(size, selected):
        position = 0
        for start in range(0, size, 65536):
            mask = np.zeros(min(65536, size-start), dtype=np.uint8)
            end = start+mask.size
            while position < len(selected) and selected[position] < end:
                mask[selected[position]-start] = 1
                position += 1
            yield memoryview(mask)

    digest = hashlib.sha256(b"rexgraph-partition-selection\x00")
    for grade, (size, selected) in enumerate(zip(sizes[1:], indices[1:], strict=True), 1):
        _selection_field(digest, grade, size, chunks(size, selected))
    if len(sizes) == 2:
        _selection_field(digest, 2, 0, ())
    if indices[0]:
        _selection_field(digest, 0, sizes[0], chunks(sizes[0], indices[0]))
    return digest.hexdigest()


def _mask(value: Any, size: int, grade: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != size:
        raise ValueError(
            f"grade {grade} mask must be a vector of length {size}"
        )
    if array.dtype.kind not in "biuf" or np.any((array != 0) & (array != 1)):
        raise ValueError("partition masks require boolean or binary numeric values")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _requested_masks(
    sizes: list[int],
    e_mask: Any,
    f_mask: Any,
    grade_masks: Mapping[int, Any] | None,
    v_mask: Any = None,
) -> list[np.ndarray]:
    top_grade = len(sizes) - 1
    requested = [np.zeros(size, dtype=np.uint8) for size in sizes]
    requested[1] = _mask(e_mask, sizes[1], 1)
    if v_mask is not None:
        requested[0] = _mask(v_mask, sizes[0], 0)
    if grade_masks is not None and not isinstance(grade_masks, Mapping):
        raise TypeError("grade_masks must map integer grades to masks")
    normalized = dict(grade_masks or {})
    for grade in normalized:
        if not isinstance(grade, int) or isinstance(grade, bool):
            raise TypeError("partition grade mask keys must be integer grades")
        if grade < 2 or grade > top_grade:
            raise ValueError(
                f"partition grade {grade} is outside the carried tower 2..{top_grade}"
            )
    if f_mask is not None and 2 in normalized:
        raise ValueError("f_mask and grade_masks[2] may not both select grade two")
    if top_grade >= 2:
        face_value = normalized.pop(2, f_mask)
        if face_value is not None:
            requested[2] = _mask(face_value, sizes[2], 2)
    elif f_mask is not None:
        _mask(f_mask, 0, 2)
    for grade, value in normalized.items():
        requested[grade] = _mask(value, sizes[grade], grade)
    return requested


def _downward_closure(rex, columns, requested: list[np.ndarray]) -> list[np.ndarray]:
    closed = [mask.astype(bool, copy=True) for mask in requested]
    for grade in range(len(columns), 1, -1):
        selected = np.flatnonzero(closed[grade])
        for index in selected:
            for lower in columns[grade - 1][index]:
                closed[grade - 1][lower] = True
    # Vertex existence follows stored relation support, not only nonzero B1 entries.
    # A self loop stores the same vertex twice and its signed B1 column cancels to zero,
    # but selecting that relation must still retain the vertex it contains.
    source_ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
    source_idx = np.asarray(rex._boundary_idx, dtype=np.int64)
    for edge in np.flatnonzero(closed[1]):
        lo, hi = int(source_ptr[edge]), int(source_ptr[edge + 1])
        closed[0][source_idx[lo:hi]] = True
    return closed


def _boundary_tower(rex):
    """Read the stored boundary tower without silently filtering invalid faces."""
    from rexgraph.native_sparse import raw_boundary_carriers
    return raw_boundary_carriers(rex)


def partition_tower(rex):
    """Certify the original tower over Q, without filtering stored faces.

    Primary shares retain their exact arity denominators. Higher maps must
    carry integers. No tolerance, rank reduction or chosen basis is used.
    """
    from rexgraph.graph import RexGraph
    from rexgraph.native_rank import boundary_columns, tower_chain_residual
    if not isinstance(rex, RexGraph):
        raise TypeError("partition requires a native RexGraph")
    rex._ensure_clean()
    boundaries = _boundary_tower(rex)
    shapes, columns = [], []
    for grade in range(1, len(boundaries)+1):
        shape, col = boundary_columns(rex, grade, carriers=boundaries)
        if shapes and shapes[-1][1] != shape[0]:
            raise ValueError("partition tower has incompatible grade axes")
        shapes.append(shape)
        columns.append(col)
    residual = tower_chain_residual(shapes, columns)
    if residual:
        raise ValueError(f"source RexGraph does not satisfy the graded chain condition: {residual}")
    return boundaries, columns


def faces_in_support(selection):
    """Stored C2 cells whose nonempty exact boundary lies wholly in a C1 set.

    This certifies the raw source chain law, not a new filling. Empty boundary
    columns are excluded: containment alone would associate them with every set.
    """
    from rexgraph.cells import Cell, CellSet
    if not isinstance(selection, (Cell, CellSet)) or selection.grade != 1:
        raise TypeError("faces_in_support requires a C1 Cell or CellSet")
    rex = selection.source
    indices = (selection.index,) if isinstance(selection, Cell) else selection.indices
    selected = CellSet(rex, 1, indices)
    _, columns = partition_tower(rex)
    support = set(selected.indices)
    faces = () if len(columns) < 2 else tuple(
        j for j, col in enumerate(columns[1]) if col and set(col) <= support)
    return CellSet(rex, 2, faces)


def _result_rex(rex, boundaries, closed, carried_state):
    from rexgraph.graph import RexGraph
    from rexgraph.native_sparse import as_native, restrict_carrier
    from rexgraph.components import component_registry
    registry = component_registry()
    owners = None if carried_state == "all" else {"support", "declaration", "weights", "identity", "relations"}
    captured = registry.capture(rex, components=owners)

    edges = np.flatnonzero(closed[1]).astype(np.int64)
    vertices = np.flatnonzero(closed[0]).astype(np.int64)
    vertex_remap = np.full(int(rex.nV), -1, dtype=np.int64)
    vertex_remap[vertices] = np.arange(vertices.size, dtype=np.int64)

    boundary_ptr = [0]
    boundary_idx = []
    source_ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
    source_idx = np.asarray(rex._boundary_idx, dtype=np.int64)
    for edge in edges:
        lo, hi = int(source_ptr[edge]), int(source_ptr[edge + 1])
        boundary_idx.extend(int(vertex_remap[value]) for value in source_idx[lo:hi])
        boundary_ptr.append(len(boundary_idx))

    kwargs: dict[str, Any] = {}
    faces = np.zeros(0, dtype=np.int64)
    if len(boundaries) >= 2:
        faces = np.flatnonzero(closed[2]).astype(np.int64)
        restricted_b2 = as_native(restrict_carrier(boundaries[1], edges, faces)).dual
        kwargs.update(
            B2_col_ptr=np.asarray(restricted_b2.col_ptr, dtype=np.int32),
            B2_row_idx=np.asarray(restricted_b2.row_idx, dtype=np.int32),
            B2_vals=np.asarray(restricted_b2.vals_csc, dtype=np.float64),
        )

    result = RexGraph(
        boundary_ptr=np.asarray(boundary_ptr, dtype=np.int32),
        boundary_idx=np.asarray(boundary_idx, dtype=np.int32),
        directed=bool(getattr(rex, "_directed", False)),
        g_channel=str(getattr(rex, "_g_channel", "raw")),
        c_channel=str(getattr(rex, "_c_channel", "share")),
        **kwargs,
    )
    result._nV = int(vertices.size)
    maps = {}
    for grade, mask in enumerate(closed):
        mapping = np.full(len(mask), -1, dtype=np.int64)
        mapping[mask] = np.arange(np.count_nonzero(mask))
        maps[grade] = mapping
    maps.setdefault(2, np.empty(0, np.int64))
    registry.remap(captured, result, maps, components=owners,
                   mode="restrict" if carried_state == "all" else "structural")
    return result


def build_rex_partition(
    rex,
    e_mask,
    *,
    v_mask=None,
    f_mask=None,
    grade_masks: Mapping[int, Any] | None = None,
    policy_digest: str = "",
    closure: str = "subcomplex",
    carried_state: str = "structural",
) -> RexPartition:
    """Extract a downward closed partition across the complete carried grade tower.

    ``e_mask`` preserves the reference grade one API. ``f_mask`` selects grade two,
    ``v_mask`` retains explicit vertices, including isolated ones, while
    ``grade_masks`` names any grade from two through the source top grade. A
    selected cell brings every nonzero boundary cell below it into the result. The
    selection digest binds the requested masks; closure added cells do not rewrite the
    caller's selection identity.

    ``carried_state='structural'`` omits application metadata, signals and sections.
    ``'all'`` transports all registered carriers on the selected bases. Bind the policy that
    authorized a structural projection through ``policy_digest``. That digest is
    lineage, not an authorization grant. C1 identities and metrics are retained;
    ``cell_maps[k][i]`` is result cell i's source index at grade k.
    """
    from rexgraph.graph import RexGraph

    from .object_identity import object_digest

    if not isinstance(rex, RexGraph):
        raise TypeError("rex must be a RexGraph")
    if closure != "subcomplex":
        raise ValueError("native Rex partitions require subcomplex closure")
    if not isinstance(policy_digest, str):
        raise TypeError("policy_digest must be a string")
    if carried_state not in {"structural", "all"}:
        raise ValueError("carried_state must be 'structural' or 'all'")
    boundaries, columns = partition_tower(rex)
    sizes = [int(boundaries[0].shape[0])] + [int(matrix.shape[1]) for matrix in boundaries]
    requested = _requested_masks(sizes, e_mask, f_mask, grade_masks, v_mask)
    selection = _selection_digest(requested)
    closed = _downward_closure(rex, columns, requested)
    result = _result_rex(rex, boundaries, closed, carried_state)
    partition_tower(result)
    state = PartitionState(
        object_digest(rex),
        object_digest(result),
        selection,
        policy_digest,
        closure,
    )
    return RexPartition(result, state, tuple(tuple(map(int, np.flatnonzero(m))) for m in closed),
                        tuple(sizes), tuple(tuple(map(int, np.flatnonzero(m))) for m in requested), carried_state)


def partition_policy(rex, policy):
    """Validate an explicit state bound selection, without constructing a result.

    The mapping has ``source_state``, ``cells`` and optional ``closure`` and
    ``carried_state`` fields. Application state is omitted by default; ``all``
    transports it and becomes part of the normalized policy identity.
    ``cells`` lists pairs of a grade and its selected indices. Indices and grades
    are canonicalized in ascending order; repeats are refused. Closure always
    includes full boundaries, so this is not a guarantee of disjoint data splits.
    A policy describes selection, not authority or removal of identity.
    """
    from operator import index
    from .object_identity import object_digest

    if not isinstance(policy, Mapping):
        raise TypeError("partition policy requires an explicit mapping")
    if set(policy) - {"source_state", "cells", "closure", "carried_state"} or not {"source_state", "cells"} <= set(policy):
        raise ValueError("partition policy requires source_state and cells; closure and carried_state are optional")
    if policy.get("carried_state", "structural") not in {"structural", "all"}:
        raise ValueError("partition carried_state must be 'structural' or 'all'")
    if policy.get("closure", "subcomplex") != "subcomplex":
        raise ValueError("partition policy closure must be subcomplex")
    if not isinstance(policy["source_state"], str) or policy["source_state"] != object_digest(rex):
        raise ValueError("partition policy source_state differs from the bound source")
    boundaries, _ = partition_tower(rex)
    sizes = [int(boundaries[0].shape[0])] + [int(b.shape[1]) for b in boundaries]

    def integer(value):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("partition policy coordinates must be integers, not booleans")
        return int(index(value))

    cells = policy["cells"]
    if not isinstance(cells, (list, tuple)):
        raise TypeError("partition policy cells must be a list of grade and index list pairs")
    selected = {}
    for entry in cells:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError("partition policy cells require grade and index list pairs")
        grade = integer(entry[0])
        if grade < 0 or grade >= len(sizes) or grade in selected:
            raise ValueError("partition policy grade is repeated or outside the source tower")
        if not isinstance(entry[1], (list, tuple)):
            raise TypeError("partition policy indices must be a list")
        indices = tuple(integer(i) for i in entry[1])
        if len(set(indices)) != len(indices) or any(i < 0 or i >= sizes[grade] for i in indices):
            raise ValueError("partition policy index is repeated or outside its grade")
        selected[grade] = sorted(indices)
    normalized = {"source_state": policy["source_state"], "closure": "subcomplex",
                  "cells": [[g, indices] for g, indices in sorted(selected.items()) if indices]}
    if policy.get("carried_state") == "all":
        normalized["carried_state"] = "all"
    return normalized


def partition_from_policy(rex, policy, *, authority_digest=""):
    """Build a declared selection using the existing exact partition algorithm.

    The lineage policy digest binds both the normalized selection policy and the
    caller's authority digest. Neither is a permission grant. Source identities,
    weights, signs and complete boundary slots remain visible in the result.
    """
    from rexgraph.cells import cell_count

    if not isinstance(authority_digest, str):
        raise TypeError("partition authority_digest must be a string")
    policy = partition_policy(rex, policy)
    masks = {}
    for grade, indices in policy["cells"]:
        mask = np.zeros(cell_count(rex, grade), dtype=np.uint8)
        mask[indices] = 1
        masks[grade] = mask
    digest = manifest_digest({"object_type": "AuthorizedPartitionSelection", "version": 1,
                              "selection": policy, "authority_digest": authority_digest})
    return build_rex_partition(rex, masks.pop(1, np.zeros(int(rex.nE), dtype=np.uint8)),
                              v_mask=masks.pop(0, None), grade_masks=masks,
                              closure=policy["closure"], policy_digest=digest,
                              carried_state=policy.get("carried_state", "structural"))
