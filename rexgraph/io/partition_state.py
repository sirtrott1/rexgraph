"""Canonical lineage and arbitrary-grade closure for derived Rex partitions."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import manifest_digest

PARTITION_VERSION = 1

__all__ = [
    "PARTITION_VERSION",
    "PartitionState",
    "RexPartition",
    "build_rex_partition",
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
        """Return the stable partition-lineage identity."""
        return manifest_digest({"object_type": "PartitionState", **self.manifest()})


@dataclass(frozen=True)
class RexPartition:
    """One derived RexGraph together with canonical source and policy lineage."""

    rex: object
    state: PartitionState


def _selection_digest(masks: list[np.ndarray]) -> str:
    """Bind requested masks without conflating closure-added lower cells."""
    digest = hashlib.sha256()
    digest.update(b"rexgraph-partition-selection\x00")
    # The archived grade-two framing always included an empty face mask even for a
    # 1-rex. Preserve that identity, then extend it monotonically with grade3+ fields.
    selection_masks = list(masks[1:])
    if len(selection_masks) == 1:
        selection_masks.append(np.zeros(0, dtype=np.uint8))
    for grade, value in enumerate(selection_masks, start=1):
        name = "edges" if grade == 1 else "faces" if grade == 2 else f"grade{grade}"
        array = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(array.size.to_bytes(8, "big"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mask(value: Any, size: int, grade: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint8).reshape(-1)
    if array.size != size:
        raise ValueError(
            f"grade {grade} mask length {array.size} does not match its {size}-cell basis"
        )
    return np.ascontiguousarray(array)


def _requested_masks(
    sizes: list[int],
    e_mask: Any,
    f_mask: Any,
    grade_masks: Mapping[int, Any] | None,
) -> list[np.ndarray]:
    top_grade = len(sizes) - 1
    requested = [np.zeros(size, dtype=np.uint8) for size in sizes]
    requested[1] = _mask(e_mask, sizes[1], 1)
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


def _downward_closure(rex, boundaries, requested: list[np.ndarray]) -> list[np.ndarray]:
    closed = [mask.astype(bool, copy=True) for mask in requested]
    for grade in range(len(boundaries), 1, -1):
        selected = np.flatnonzero(closed[grade])
        if selected.size:
            lower = boundaries[grade - 1][:, selected].nonzero()[0]
            closed[grade - 1][np.asarray(lower, dtype=np.int64)] = True
    # Vertex existence follows stored relation support, not only nonzero B1 entries.
    # A self-loop stores the same vertex twice and its signed B1 column cancels to zero,
    # but selecting that relation must still retain the vertex it contains.
    source_ptr = np.asarray(rex._boundary_ptr, dtype=np.int64)
    source_idx = np.asarray(rex._boundary_idx, dtype=np.int64)
    for edge in np.flatnonzero(closed[1]):
        lo, hi = int(source_ptr[edge]), int(source_ptr[edge + 1])
        closed[0][source_idx[lo:hi]] = True
    return closed


def _boundary_tower(rex):
    """Read the stored boundary tower without silently filtering invalid faces."""
    import scipy.sparse as sp

    from rexgraph.core._sparse import to_scipy_csr

    boundaries = [to_scipy_csr(rex._B1_dual).tocsr()]
    duals = list(getattr(rex, "_graded_duals", None) or ())
    if int(rex.nF) > 0:
        if rex._B2_dual is None:  # pragma: no cover - constructor invariant
            raise ValueError("source RexGraph has faces but no stored B2")
        boundaries.append(to_scipy_csr(rex._B2_dual).tocsr())
    elif duals:
        # Preserve an empty grade-two slot before B3 rather than shifting every higher
        # operator down by one grade.
        boundaries.append(sp.csr_matrix((int(rex.nE), int(duals[0].shape[0]))))
    boundaries.extend(sp.csr_matrix(matrix) for matrix in duals)
    return boundaries


def _result_rex(rex, boundaries, closed):
    from rexgraph.graph import RexGraph

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
        restricted_b2 = boundaries[1][edges, :][:, faces].tocsc()
        kwargs.update(
            B2_col_ptr=np.asarray(restricted_b2.indptr, dtype=np.int32),
            B2_row_idx=np.asarray(restricted_b2.indices, dtype=np.int32),
            B2_vals=np.asarray(restricted_b2.data, dtype=np.float64),
        )

    edge_remap = {int(old): new for new, old in enumerate(edges)}
    vertex_positions = {int(old): new for new, old in enumerate(vertices)}
    boundary_weights = {}
    for key, value in (getattr(rex, "_w_boundary", {}) or {}).items():
        try:
            old_edge, old_vertex = int(key[0]), int(key[1])
        except (IndexError, TypeError, ValueError):
            continue
        if old_edge in edge_remap and old_vertex in vertex_positions:
            boundary_weights[(edge_remap[old_edge], vertex_positions[old_vertex])] = deepcopy(
                value
            )

    weights = getattr(rex, "_w_E", None)
    signs = getattr(rex, "_signs", None)
    result = RexGraph(
        boundary_ptr=np.asarray(boundary_ptr, dtype=np.int32),
        boundary_idx=np.asarray(boundary_idx, dtype=np.int32),
        w_E=None if weights is None else np.ascontiguousarray(np.asarray(weights)[edges]),
        w_boundary=boundary_weights,
        directed=bool(getattr(rex, "_directed", False)),
        signs=None if signs is None else np.ascontiguousarray(np.asarray(signs)[edges]),
        g_channel=str(getattr(rex, "_g_channel", "raw")),
        c_channel=str(getattr(rex, "_c_channel", "share")),
        **kwargs,
    )
    if len(boundaries) >= 3:
        result._graded_duals = [
            boundaries[grade - 1][
                np.flatnonzero(closed[grade - 1]), :
            ][:, np.flatnonzero(closed[grade])].tocsr()
            for grade in range(3, len(boundaries) + 1)
        ]
    return result


def build_rex_partition(
    rex,
    e_mask,
    *,
    f_mask=None,
    grade_masks: Mapping[int, Any] | None = None,
    policy_digest: str = "",
    closure: str = "subcomplex",
) -> RexPartition:
    """Extract a downward-closed partition across the complete carried grade tower.

    ``e_mask`` preserves the reference grade-one API. ``f_mask`` selects grade two,
    while ``grade_masks`` names any grade from two through the source top grade. A
    selected cell brings every nonzero boundary cell below it into the result. The
    selection digest binds the requested masks; closure-added cells do not rewrite the
    caller's selection identity.

    Application metadata is deliberately absent from the result. Bind the policy that
    authorized a structural projection through ``policy_digest``.
    """
    from rexgraph.graded_boundary import verify_chain
    from rexgraph.graph import RexGraph

    from .catalog import object_digest

    if not isinstance(rex, RexGraph):
        raise TypeError("rex must be a RexGraph")
    if closure != "subcomplex":
        raise ValueError("native Rex partitions require subcomplex closure")
    if not isinstance(policy_digest, str):
        raise TypeError("policy_digest must be a string")
    rex._ensure_clean()
    boundaries = _boundary_tower(rex)
    valid_source, _source_residual = verify_chain(boundaries)
    if not valid_source:
        raise ValueError("source RexGraph does not satisfy the graded chain condition")
    sizes = [int(boundaries[0].shape[0])] + [int(matrix.shape[1]) for matrix in boundaries]
    requested = _requested_masks(sizes, e_mask, f_mask, grade_masks)
    selection = _selection_digest(requested)
    closed = _downward_closure(rex, boundaries, requested)
    result = _result_rex(rex, boundaries, closed)
    valid_result, _result_residual = verify_chain(_boundary_tower(result))
    if not valid_result:  # pragma: no cover - closure invariant
        raise ValueError("partition restriction broke the graded chain condition")
    state = PartitionState(
        object_digest(rex),
        object_digest(result),
        selection,
        policy_digest,
        closure,
    )
    return RexPartition(result, state)
