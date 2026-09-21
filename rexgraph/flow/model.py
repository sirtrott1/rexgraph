"""Persistent observations for the existing online Green field."""
from __future__ import annotations

import numpy as np
from rexgraph.model_state import ModelState, ModelOutput
from rexgraph.model_runtime import model_coordinates
from rexgraph.tensor_field import FieldSource
from rexgraph.coordinate_map import CoordinateMap

__all__ = ["online_model", "infer_online", "correct_online", "transport_online"]


def _ids(source):
    values = np.asarray(source.relation_ids)
    if values.ndim != 1 or values.dtype.kind not in "iu" or len(values) != source.nE or len(set(values.tolist())) != len(values):
        raise ValueError("online local models require distinct primary relation identities")
    return values.copy()


def online_model(source, *, green_lam=4.0, green_iters=20, values=None, reference=None):
    """Declare a numerical flow field on primary identities rather than support keys."""
    if not np.isfinite(green_lam) or green_lam < 0:
        raise ValueError("flow parameter must be finite and nonnegative")
    if isinstance(green_iters, bool) or not isinstance(green_iters, int) or green_iters < 1:
        raise ValueError("flow iterations must be positive")
    ids = _ids(source)
    data = np.zeros(len(ids), dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if data.shape != (len(ids),) or not np.isfinite(data).all(): raise ValueError("invalid local flow values")
    ref = FieldSource(source) if reference is None else reference.bind(source)
    return ModelState("online_flow", "1", ref, model_coordinates(source), (),
        {"green_lam": float(green_lam), "green_iters": green_iters, "identity": "lineage"},
        {"values": data, "relation_ids": ids, "observed": np.zeros(len(ids), dtype=bool)}, "approximate")


def _restore(state, source):
    from rexgraph.flow.online import GreensCochainField
    state.bind(source)
    if state.adapter_version != "1" or not np.array_equal(_ids(source), state.payload["relation_ids"]):
        raise ValueError("online model primary identities differ")
    model = GreensCochainField(green_lam=state.configuration["green_lam"],
                              green_iters=state.configuration["green_iters"], identity="lineage")
    model.phi = {int(k): float(v) for k, v in zip(state.payload["relation_ids"], state.payload["values"], strict=True)}
    return model


def infer_online(state, source, inputs=None):
    if inputs is not None: raise ValueError("an online flow field takes no feature input")
    model = _restore(state, source)
    prediction = model.predict(source, np.arange(source.nE))
    return ModelOutput(prediction, state.space, (), state.source, state.coefficient_digest,
                       "approximate", "native-online-green-prediction", state.dependencies, 1, "cochain")


def correct_online(state, source, batch):
    model = _restore(state, source)
    if batch.space != state.space or not batch.source.matches(state.source):
        raise ValueError("online observation source or coordinates differ")
    if batch.inputs is not None: raise ValueError("online field does not take features")
    if batch.targets.shape != (source.nE,) or batch.targets.dtype.kind not in "iuf":
        raise ValueError("online targets require one real value per primary relation")
    if not batch.observed.any(): raise ValueError("online correction requires observed targets")
    prediction = model.predict(source, np.arange(source.nE))
    region = np.flatnonzero(batch.observed)
    correction = model.correct(source, region, np.asarray(batch.targets[region], dtype=np.float64))
    values = np.array([model.phi[int(k)] for k in state.payload["relation_ids"]], dtype=np.float64)
    payload = {"values": values, "relation_ids": state.payload["relation_ids"],
               "observed": np.logical_or(state.payload["observed"], batch.observed),
               "last_batch": batch.coefficient_digest, "prediction": prediction,
               "target": batch.targets, "target_input": batch.target_input, "target_mask": batch.observed, "correction": correction}
    extra = () if batch.target_input is None else (batch.target_input.source, *batch.target_input.dependencies)
    dependencies = tuple({s.coefficient_digest: s for s in (*state.dependencies, *batch.dependencies, batch.source, *extra)}.values())
    return state.child(payload=payload, step=state.step + 1, dependencies=dependencies)


def transport_online(state, source, destination, mapping):
    _restore(state, source); ids = _ids(destination.source)
    if not isinstance(mapping, CoordinateMap) or mapping.domain != state.space or mapping.codomain != model_coordinates(destination.source):
        raise ValueError("online transport coordinates differ")
    values = np.zeros(len(ids), dtype=np.float64)
    for i, j, v in mapping.entries: values[i] += float(v) * state.payload["values"][j]
    payload = {"values": values, "relation_ids": ids, "observed": np.zeros(len(ids), dtype=bool),
               "transport_digest": mapping.coefficient_digest, "transport_entries": mapping.entries,
               "transport_domain": (mapping.domain.name, mapping.domain.keys), "observation_policy": "new observations are explicit"}
    return state.child(payload=payload, source=destination, space=mapping.codomain,
                       dependencies=(*state.dependencies, state.source))
