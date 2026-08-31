"""RCQL operators over RexGraph values."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from rexgraph.cochain import Cochain, Field
from rexgraph.graded_boundary import _sparse_rank, graded_boundaries_from_rex
from rexgraph.green import vertex_green
from rexgraph.linear_operator import (
    RexOperator,
    boundary_operator,
    coboundary_operator,
    hodge_operator,
)

from .describe import describe_rex
from .types import RCType


@dataclass(frozen=True)
class Operator:
    name: str
    fn: Callable
    result_type: Callable[[tuple[Any, ...]], RCType] | None = None


_REGISTRY: dict[str, Operator] = {}


def register(name: str):
    def wrap(fn):
        _REGISTRY[name.upper()] = Operator(name.upper(), fn)
        return fn
    return wrap


def get_operator(name: str) -> Operator:
    try:
        return _REGISTRY[name.upper()]
    except KeyError as exc:
        raise KeyError(f"unknown RCQL operator {name!r}") from exc


def _cell_count(rex, grade: int) -> int:
    B = graded_boundaries_from_rex(rex)
    sizes = [B[0].shape[0]] + [b.shape[1] for b in B] if B else [0]
    grade = int(grade)
    if grade < 0 or grade >= len(sizes):
        raise ValueError(f"grade {grade} is not present")
    return int(sizes[grade])


@register("REX")
def rex_source(source, name):
    return name


@register("GRADE")
def grade(source, value=None):
    if value is None:
        B = graded_boundaries_from_rex(source)
        return len(B)
    return int(value.grade)


@register("BOUNDARY")
def boundary(source, grade, values=None):
    op = boundary_operator(source, int(grade))
    if values is None:
        return op
    raw = values.values if isinstance(values, Cochain) else values
    out = op.apply(raw)
    return Cochain(int(grade) - 1, out, source=source)


@register("DESCRIBE")
def describe(source):
    return describe_rex(source)


@register("COBOUNDARY")
def coboundary(source, grade, values=None):
    op = coboundary_operator(source, int(grade))
    if values is None:
        return op
    raw = values.values if isinstance(values, Cochain) else values
    out = op.apply(raw)
    return Cochain(int(grade) + 1, out, source=source)


@register("HODGE_OPERATOR")
def hodge_op(source, grade, alpha=1):
    return hodge_operator(source, int(grade), alpha=alpha)


@register("RANK")
def rank(source, value):
    if isinstance(value, RexOperator):
        return int(_sparse_rank(value.as_scipy()))
    if isinstance(value, (int, np.integer)):
        return int(_sparse_rank(boundary_operator(source, int(value)).as_scipy()))
    raise TypeError("RANK expects a boundary operator or grade")


@register("NULLITY")
def nullity(source, value):
    if isinstance(value, RexOperator):
        return int(value.shape[1] - _sparse_rank(value.as_scipy()))
    grade = int(value)
    op = boundary_operator(source, grade)
    return int(op.shape[1] - _sparse_rank(op.as_scipy()))


@register("BETTI")
def betti(source, grade):
    return int(source.betti[int(grade)])


@register("HODGE")
def hodge(source, flow):
    values = flow.values if isinstance(flow, Cochain) else flow
    grad, curl, harm = source.hodge(np.ascontiguousarray(values, dtype=np.float64))
    return {
        "gradient": Cochain(1, grad, source=source),
        "curl": Cochain(1, curl, source=source),
        "harmonic": Cochain(1, harm, source=source),
    }


@register("HARMONIC")
def harmonic(source, flow):
    return hodge(source, flow)["harmonic"]


@register("GREEN")
def green(source, values):
    grade = values.grade if isinstance(values, Cochain) else 0
    if grade != 0:
        raise ValueError("GREEN currently expects a grade 0 cochain")
    raw = values.values if isinstance(values, Cochain) else values
    op = vertex_green(source)
    field = Cochain(0, op.solve(np.asarray(raw, dtype=np.float64)), source=source)
    return Field(field, op, kind="green")


@register("QUADRANCE")
def quadrance(source, values):
    raw = values.values if isinstance(values, (Cochain, Field)) else values
    x = np.asarray(raw)
    if x.ndim == 1:
        return np.vdot(x, x).real.item()
    return np.einsum("ij,ij->j", x.conj(), x).real


@register("SPREAD")
def spread(source, left, right):
    a = left.values if isinstance(left, (Cochain, Field)) else left
    b = right.values if isinstance(right, (Cochain, Field)) else right
    a = np.asarray(a)
    b = np.asarray(b)
    qa = np.vdot(a, a).real
    qb = np.vdot(b, b).real
    if qa == 0 or qb == 0:
        return 0
    ab = np.vdot(a, b)
    return (1 - (ab.conjugate() * ab).real / (qa * qb)).item()


@register("HODGE_COORDS")
def hodge_coordinates(source, flow):
    from rexgraph.hodge_coords import hodge_coords

    raw = flow.values if isinstance(flow, Cochain) else flow
    return hodge_coords(source, raw)


@register("WINDING")
def winding(source, flow):
    from rexgraph.harmonic_sparse import harmonic_winding
    from rexgraph.hodge_coords import harmonic_frame

    raw = flow.values if isinstance(flow, Cochain) else flow
    return harmonic_winding(harmonic_frame(source), raw)


@register("CLOSURE")
def closure(source, seed, max_depth=8, grade=0):
    from rexgraph.tower import semantic_closure

    return semantic_closure(source, int(seed), max_depth=int(max_depth), grade=int(grade))


@register("SIGNIFICANCE")
def significance(source, edge):
    idx = int(edge)
    from rexgraph.semantic import significance as _significance
    return float(_significance(source, [idx])[0])


@register("CHARACTER")
def character(source):
    return np.asarray(source.structural_character)


@register("ZERO")
def zero(source, grade):
    grade = int(grade)
    return Cochain(grade, np.zeros(_cell_count(source, grade), dtype=np.float64), source=source)


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
    if not hasattr(source, "get"):
        raise TypeError("RCDB_GET expects an RCDB store")
    return source.get(str(record_id))


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
    if not hasattr(source, "get"):
        raise TypeError("RCDB_HASH expects an RCDB store")
    from rexgraph.io.catalog import object_digest
    return object_digest(source.get(str(record_id)))

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
    return source.state_digest()


@register("RCDB_SECURITY")
def rcdb_security(source):
    """Return bounded RCDB security configuration without secrets or paths."""
    if not hasattr(source, "security_status"):
        raise TypeError("RCDB_SECURITY expects an RCDB store")
    return dict(source.security_status())
