"""JSON values returned by the System API."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from fractions import Fraction
from math import isfinite
from typing import Any

import numpy as np
from rcql.capabilities import SourcePolicy
from rexgraph.channel_operator import ChannelOperator
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator
from rexgraph.operator_bracket import GradedOperatorBracket, OperatorBracket
from rexgraph.sheaf import ExactGlueResult, ExactSectionCheck
from rexgraph.weighted_dirac import GradedChain, WeightedDiracOperator


def _array(value: Any, max_values: int) -> dict[str, Any]:
    arr = np.asarray(value)
    out = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size <= max_values:
        out["values"] = json_value(arr.tolist(), max_values=max_values)
    elif arr.size:
        flat = arr.ravel()
        out["sample"] = json_value(flat[:max_values].tolist(), max_values=max_values)
        if arr.dtype.kind in "fiu" and np.any(np.isfinite(arr)):
            finite = arr[np.isfinite(arr)]
            out["min"] = json_value(np.min(finite).item(), max_values=max_values)
            out["max"] = json_value(np.max(finite).item(), max_values=max_values)
    return out


def json_value(value: Any, *, max_values: int = 256) -> Any:
    """Render one RCQL result as JSON, bounding array previews by max_values."""
    if isinstance(max_values, bool) or not isinstance(max_values, int) or max_values < 0:
        raise ValueError("max_values must be a nonnegative integer")
    if isinstance(value, float) and not isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        import hashlib
        return {"kind": "ArtifactBytes", "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(), "payload": "omitted"}
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.generic):
        return json_value(value.item(), max_values=max_values)
    if isinstance(value, complex):
        return {"real": json_value(value.real, max_values=max_values),
                "imaginary": json_value(value.imag, max_values=max_values)}
    if isinstance(value, np.ndarray):
        return _array(value, max_values)
    from rexgraph.model_state import ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput
    if isinstance(value, (ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput)):
        value.check_state()
        out = {"kind": type(value).__name__, "digest": value.coefficient_digest}
        if isinstance(value, ModelTimeline):
            out.update(count=len(value.times), times=json_value(value.times[:max_values], max_values=max_values),
                       model_digests=list(value.model_digests[:max_values]),
                       source_digests=list(value.source_digests[:max_values]),
                       truncated=len(value.times) > max_values, temporal_index="ordinal",
                       time_axis=value.time_axis, time_unit=value.time_unit)
            return out
        out.update(source=value.source.as_record(),
                   coordinates={"name": value.space.name, "count": len(value.space.keys),
                                "keys": list(value.space.keys[:max_values]),
                                "truncated": len(value.space.keys) > max_values},
                   dependencies=[v.as_record() for v in value.dependencies[:max_values]],
                   dependencies_truncated=len(value.dependencies) > max_values)
        if isinstance(value, ModelState):
            parameters = value.payload.get("weights", {})
            out.update(adapter=value.adapter, adapter_version=value.adapter_version,
                       arithmetic=value.arithmetic, step=value.step, parent=value.parent,
                       resumable="optimizer" in value.payload,
                       axes=[{"name": a.name, "size": len(a.keys)} for a in value.output_axes],
                       parameter_count=len(parameters),
                       parameters=[{"name": name, "shape": list(v.shape), "dtype": str(v.dtype)}
                                   for name, v in list(parameters.items())[:max_values]
                                   if isinstance(v, np.ndarray)],
                       parameters_truncated=len(parameters) > max_values)
        elif isinstance(value, ModelInput):
            out.update(arithmetic=value.arithmetic, origin_digest=value.origin_digest,
                       original_arithmetic=value.origin_arithmetic,
                       axes=[{"name": a.name, "size": len(a.keys)} for a in value.axes],
                       values=_array(value.values, max_values), original="retained in the model input")
        elif isinstance(value, ModelOutput):
            out.update(arithmetic=value.arithmetic, model_digest=value.model_digest, method=value.method,
                       grade=value.grade, variance=value.variance,
                       axes=[{"name": a.name, "size": len(a.keys)} for a in value.axes],
                       values=_array(value.values, max_values))
        else:
            out.update(targets=_array(value.targets, max_values), observed=_array(value.observed, max_values),
                       observed_count=int(value.observed.sum()), inputs="query explicitly")
        return out
    from rexgraph.chain_map import SymmetryGroup
    if isinstance(value, SymmetryGroup):
        value.check_state()
        return {"kind": "SymmetryGroup", "metric": "euclidean", "product_order": "rightmost-first",
                "generator_count": value.generator_count, "word": list(value.word[:max_values]),
                "word_length": len(value.word), "sizes": list(value.sizes[:max_values]),
                "grades": len(value.sizes), "map": "lazy",
                "truncated": len(value.word) > max_values or len(value.sizes) > max_values}
    from rexgraph.void_state import VoidState
    if isinstance(value, VoidState):
        value.check_state()
        return {"kind": "VoidState", "source_state": value.source_state, "shape": list(value.shape),
                "n_voids": value.n_voids, "n_potential": value.n_potential, "strain": value.strain,
                "region": list(value.region[:max_values]), "region_truncated": len(value.region) > max_values,
                "potential": json_value(value.potential[:max_values], max_values=max_values),
                "potential_truncated": value.n_potential > max_values,
                "void_indices": list(value.void_indices[:max_values]),
                "columns": json_value(value.columns[:max_values], max_values=max_values),
                "columns_truncated": value.n_voids > max_values, "homology": "query explicitly"}
    from rexgraph.relative_quotient import RelativeQuotient
    if isinstance(value, RelativeQuotient):
        value.check_state()
        return {"kind": "RelativeQuotient", "sizes": list(value.sizes),
                "source_boundary_digest": value.source_boundary_digest,
                "coefficient_digest": value.coefficient_digest,
                "residuals": json_value(value.residuals, max_values=max_values),
                "boundaries": [{"shape": [value.sizes[k], value.sizes[k+1]], "nnz": len(entries),
                    "entries": json_value(entries[:max_values], max_values=max_values),
                    "entries_truncated": len(entries) > max_values}
                    for k, entries in enumerate(value.boundaries)],
                "coordinates": [{"kept_count": len(kept), "removed_count": len(removed),
                    "kept": list(kept[:max_values]), "removed": list(removed[:max_values]),
                    "truncated": max(len(kept), len(removed)) > max_values}
                    for kept, removed in zip(value.cell_maps, value.removed_cells, strict=True)],
                "homology": "query readings explicitly"}
    from rexgraph.boundary_difference import BoundaryDifference
    if isinstance(value, BoundaryDifference):
        value.check_state()
        return {"kind": "BoundaryDifference", "shape": list(value.shape), "nnz": value.nnz,
                "matching": value.matching, "source_digests": list(value.source_digests),
                "entries": json_value(value.entries[:max_values], max_values=max_values),
                "entries_truncated": value.nnz > max_values}
    if isinstance(value, RexGraph):
        return {"kind": "RexGraph", "nV": value.nV, "nE": value.nE, "nF": value.nF,
                "g_channel": value.g_channel, "c_channel": value.c_channel}
    if isinstance(value, TemporalRex):
        return {"kind": "TemporalRex", "T": value.T}
    if isinstance(value, GradedChain):
        return {"kind": "GradedChain", "sizes": list(value.sizes),
                "components": [json_value(c, max_values=max_values) for c in value.components]}
    if isinstance(value, WeightedDiracOperator):
        return {"kind": "GradedOperator", "name": value.name, "shape": list(value.shape),
                "sizes": list(value.sizes), "exact_action": value.exact,
                "metric_self_adjoint": value.metric_self_adjoint,
                "metric_skew_adjoint": value.metric_skew_adjoint}
    if isinstance(value, (OperatorBracket, GradedOperatorBracket)):
        graded = isinstance(value, GradedOperatorBracket)
        return {"kind": "GradedOperatorBracket" if graded else "OperatorBracket",
                "name": value.name, "shape": list(value.shape),
                "spaces": {"sizes": list(value.sizes), "variance": "chain"} if graded else {
                    "domain_grade": value.domain_grade, "codomain_grade": value.codomain_grade,
                    "variance": value.variance},
                "operands": [json_value(op, max_values=max_values) for op in (value.left, value.right)],
                "metric_self_adjoint": value.metric_self_adjoint,
                "metric_skew_adjoint": value.metric_skew_adjoint,
                "euclidean_skew_adjoint": None if graded else value.euclidean_skew_adjoint,
                "exact_action": value.exact if graded else value.exact_matvec is not None,
                "psd": False}
    if isinstance(value, (Chain, Cochain)):
        return {
            "kind": "Chain" if isinstance(value, Chain) else "Cochain",
            "grade": value.grade,
            "cells": value.n_cells,
            "cell_keys": json_value(value.cell_keys, max_values=max_values),
            "values": _array(value.values, max_values),
        }
    if isinstance(value, Field):
        return {
            "kind": "Field",
            "field": value.kind,
            "grade": value.grade,
            "values": _array(value.values, max_values),
            "operator": json_value(value.operator, max_values=max_values),
        }
    if isinstance(value, RexOperator):
        out = {
            "kind": "RexOperator",
            "name": value.name,
            "shape": list(value.shape),
            "domain_grade": value.domain_grade,
            "codomain_grade": value.codomain_grade,
            "symmetric": value.symmetric,
            "psd": value.psd,
            "arithmetic": value.arithmetic,
        }
        if isinstance(value, ChannelOperator):
            out.update(channel=value.channel, g_channel=value.g_channel, c_channel=value.c_channel,
                       exact_action=value.exact_matvec is not None,
                       exact_transpose=value.exact_transpose_matvec is not None,
                       exact_diagonal=True, trace_normalized=False, frustration_reference="raw-G")
        from rexgraph.rational_operator import ResolventGroup
        if isinstance(value, ResolventGroup):
            value.check_state()
            out.update(kind="ResolventGroup", word=list(value.word[:max_values]),
                       scales=json_value(value.scales[:max_values], max_values=max_values),
                       word_length=len(value.word), generators=len(value.operators),
                       truncated=len(value.word) > max_values or len(value.scales) > max_values,
                       exact_action=True, product_order="rightmost-first")
        return out
    if isinstance(value, GreenOperator):
        return {
            "kind": "GreenOperator",
            "green": value.kind,
            "operator": json_value(value.operator, max_values=max_values),
            "parameters": json_value(value.parameters, max_values=max_values),
            "solve_form": "positive-diagonal-metric" if value.metric is not None else "euclidean",
            "metric_digest": None if value.metric is None else value.metric.coefficient_digest,
            "action_variance": value.operator.variance,
        }
    if isinstance(value, dict):
        return {str(k): json_value(v, max_values=max_values) for k, v in value.items()}
    if isinstance(value, SourcePolicy):
        return {"permissions": sorted(value.permissions),
                "record_fields": None if value.record_fields is None else sorted(value.record_fields),
                "digest": value.digest}
    if isinstance(value, (tuple, list)):
        return [json_value(v, max_values=max_values) for v in value]
    if is_dataclass(value):
        # Do not deep copy a dataclass's live source graph just to render it.
        out = {field.name: json_value(getattr(value, field.name), max_values=max_values)
               for field in fields(value) if not field.name.startswith("_") and field.name != "source"}
        if isinstance(value, ExactSectionCheck):
            out.update(kind="ExactSectionCheck", compatible=value.compatible,
                       diagnostic="anchor-incidence-residuals")
        elif isinstance(value, ExactGlueResult):
            out.update(kind="ExactGlueResult", ratio=json_value(value.ratio),
                       obstruction_count=value.obstruction_count,
                       agreement_components=value.h0, diagnostic="all-pair-incidence-residuals")
        return out
    if hasattr(value, "_asdict"):
        return json_value(value._asdict(), max_values=max_values)
    # Opaque handles are not an invitation to expose their repr, paths or state.
    return {"python_type": type(value).__name__}
