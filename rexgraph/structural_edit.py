"""Native structural edits and transport of carried state through compaction.

No graph expansion: a C1 column is always one relation, whatever its arity.
Deletion drops cofaces, rather than truncating a boundary into a non cycle.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

import numpy as np


def remap_carried_state(rex, maps):
    """Transport metadata and all upper boundaries through native C1/C2 maps."""
    from rexgraph.native_sparse import csr_carrier, sparse_arrays
    upper = []
    for grade, matrix in enumerate(rex._graded_duals or (), 3):
        ptr, columns, data, shape = sparse_arrays(matrix)
        lower = maps[grade - 1]
        if shape[0] != len(lower):
            raise ValueError("upper boundary shape does not match the lower cell basis")
        alive = np.ones(shape[1], dtype=bool)
        for i in np.flatnonzero(lower < 0):
            start, stop = int(ptr[i]), int(ptr[i + 1])
            alive[columns[start:stop][data[start:stop] != 0]] = False
        current = np.full(shape[1], -1, dtype=np.int32)
        current[alive] = np.arange(np.count_nonzero(alive), dtype=np.int32)
        maps[grade] = current
        out_ptr, out_idx, out_data = [0], [], []
        for i in np.flatnonzero(lower >= 0):
            start, stop = int(ptr[i]), int(ptr[i + 1])
            cols, vals = columns[start:stop], data[start:stop]
            keep = alive[cols]
            out_idx.extend(current[cols[keep]])
            out_data.extend(vals[keep])
            out_ptr.append(len(out_idx))
        upper.append(csr_carrier(np.asarray(out_ptr, dtype=np.int64),
                     np.asarray(out_idx, dtype=np.int64), np.asarray(out_data, dtype=data.dtype),
                     (int(np.count_nonzero(lower >= 0)), int(np.count_nonzero(alive)))))
    if rex._graded_duals is not None:
        rex._graded_duals = upper
    metadata = getattr(rex, "_cell_metadata", None)
    if metadata:
        rex._cell_metadata = {grade: {int(maps[grade][i]): value for i, value in entries.items()
                                     if maps[grade][i] >= 0}
                              for grade, entries in metadata.items() if grade in maps}
    from rexgraph.sectioning import sectionings_of
    for section in sectionings_of(rex).values():
        mapping = maps[section.grade]
        section.n_cells = int(np.count_nonzero(mapping >= 0))
        if section.refines:
            continue  # section IDs/parent hierarchy have not changed
        ptr, ids = [0], []
        for i in range(section.n_sections):
            members = mapping[section.cells(i)]
            ids.extend(members[members >= 0])
            ptr.append(len(ids))
        section.indptr = np.asarray(ptr, dtype=np.int64)
        section.indices = np.asarray(ids, dtype=np.int64)
    meta = getattr(rex, "_agent_meta", None)
    if meta and "vertex_labels" in meta:
        meta["vertex_labels"] = [label for i, label in enumerate(meta["vertex_labels"])
                                 if i < len(maps[0]) and maps[0][i] >= 0]
    signals = getattr(rex, "_signals", None)
    if isinstance(signals, np.ndarray) and signals.ndim and signals.shape[0] == len(maps[1]):
        rex._signals = signals[maps[1] >= 0].copy()


def edit_relations(state, operation, value):
    """Return an independently owned Rex after ADD columns or REMOVE C1 indices.

    ADD accepts a sequence of full boundary supports, or a mapping with
    ``columns``, optional ``weights``, ``signs``, and ``relation_ids``. REMOVE
    accepts current ordered C1 indices. Each clause addresses the state produced
    by the preceding clause; it never silently reuses a stale cell basis.
    """
    from rexgraph.graph import RexGraph
    from rexgraph.io.rex_state import RexState, from_state, to_state
    if not isinstance(state, RexGraph):
        raise TypeError("structural editing requires a static native RexGraph")
    validate_edit(operation, value)
    payload = to_state(state)
    rex = from_state(RexState({name: tensor.copy() for name, tensor in payload.tensors.items()},
                             deepcopy(payload.header)))
    if operation == "ADD":
        spec = value if isinstance(value, Mapping) else {"columns": value}
        columns = spec["columns"]
        rex.add_hyperedges(columns, w_E=spec.get("weights"), signs=spec.get("signs"),
                           relation_ids=spec.get("relation_ids"))
        rex.compact()
    else:
        indices = list(value)
        if any(i < 0 or i >= rex.nE for i in indices):
            raise ValueError("REMOVE index is absent from the current relation basis")
        mask = np.zeros(rex.nE, dtype=np.int32)
        mask[indices] = 1
        rex.remove_edges(mask)
        rex.compact()
    # Existing section memberships remain explicit; appending does not silently
    # assign new relations to a section. Population is the new carrier's size.
    from rexgraph.sectioning import sectionings_of
    from rexgraph.cells import cell_count
    for section in sectionings_of(rex).values():
        section.n_cells = cell_count(rex, section.grade)
    return rex


def validate_edit(operation, value):
    """Validate the structural input contract without constructing or editing."""
    if operation not in {"ADD", "REMOVE"}:
        raise ValueError("structural operation must be ADD or REMOVE")
    if operation == "REMOVE":
        if not isinstance(value, (list, tuple, np.ndarray)) or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in value
        ):
            raise TypeError("REMOVE requires exact current C1 indices")
        return
    spec = value if isinstance(value, Mapping) else {"columns": value}
    if "columns" not in spec or set(spec) - {"columns", "weights", "signs", "relation_ids"}:
        raise ValueError("ADD accepts columns, weights, signs and relation_ids only")
    if not isinstance(spec["columns"], (list, tuple, np.ndarray)):
        raise TypeError("ADD requires a sequence of full boundary columns")
    from rexgraph.graph import _as_exact_i32_vector, _validate_c1_support
    for col in spec["columns"]:
        _validate_c1_support(_as_exact_i32_vector(col, context="ADD"), context="ADD")
    for key in ("weights", "signs", "relation_ids"):
        if key in spec and (np.asarray(spec[key]).ndim != 1 or len(spec[key]) != len(spec["columns"])):
            raise ValueError(f"ADD {key} must have one entry per relation")
