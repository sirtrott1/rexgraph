"""Native structural edits and transport of carried state through compaction.

No graph expansion: a C1 column is always one relation, whatever its arity.
Deletion drops cofaces, rather than truncating a boundary into a non cycle.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

import numpy as np


def remap_carried_state(rex, maps, *, captured):
    """Apply components captured before replacing the primary cell support.

    Native edits and functional restrictions share the registered owners. Requiring
    the pre edit capture prevents reconstructing lost declarations from new arrays.
    """
    from rexgraph.components import component_registry
    return component_registry().remap(captured, rex, maps)


def edit_relations(state, operation, value):
    """Return an independently owned Rex after ADD columns or REMOVE C1 indices.

    ADD accepts a sequence of full boundary supports, or a mapping with
    ``columns``, optional ``weights``, ``signs``, and ``relation_ids``. REMOVE
    accepts current ordered C1 indices. Each clause addresses the state produced
    by the preceding clause; it never silently reuses a stale cell basis.
    """
    from rexgraph.graph import RexGraph
    from rexgraph.state import RexState, from_state, to_state
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
