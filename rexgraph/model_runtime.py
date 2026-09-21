"""Native exact models and explicit local learning dispatch."""
from __future__ import annotations

from fractions import Fraction

from rexgraph.model_state import ModelState, ModelOutput, ModelBatch
from rexgraph.tensor_field import FieldSource, TensorField, apply_tensor
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMetric

__all__ = ["native_model", "infer_model", "train_model", "transport_model", "model_coordinates", "certify_native_response"]


def model_coordinates(source, grade=1):
    from rexgraph.chain_map import CoordinateComplex
    complex_ = CoordinateComplex.from_rex(source)
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade < len(complex_.spaces):
        raise ValueError("model grade is outside the native tower")
    return complex_.spaces[grade]


def native_model(source, *, grade=1, operation="green", parameter=Fraction(1), metrics=None, reference=None):
    """Capture an exact field action and its declared metric tower."""
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.graded_metric import _fraction
    calculus = NativeFieldCalculus.from_rex(source, metrics)
    parameter = _fraction(parameter)
    if operation not in {"green", "hodge", "gradient", "curl", "harmonic"}:
        raise ValueError("unsupported native model operation")
    action = (calculus.green(grade, parameter) if operation == "green" else
              calculus.hodge(grade) if operation == "hodge" else calculus.sector(grade, operation))
    ref = FieldSource(source) if reference is None else reference.bind(source)
    configuration = {"grade": grade, "operation": operation, "parameter": parameter,
        "metric_spaces": tuple((m.space.name, m.space.keys) for m in calculus.metrics),
        "metrics": tuple(m.entries for m in calculus.metrics), "action_digest": action.coefficient_digest}
    return ModelState("native_field", "1", ref, action.domain, (), configuration, {}, "rational")


def _calculus(state, source):
    from rexgraph.native_field import NativeFieldCalculus
    c = state.configuration
    metrics = tuple(CoordinateMetric(CoordinateSpace(s[0], tuple(s[1])), tuple(v))
                    for s, v in zip(c["metric_spaces"], c["metrics"], strict=True))
    calculus = NativeFieldCalculus.from_rex(source, metrics)
    grade, operation = c["grade"], c["operation"]
    action = (calculus.green(grade, c["parameter"]) if operation == "green" else
              calculus.hodge(grade) if operation == "hodge" else calculus.sector(grade, operation))
    if action.coefficient_digest != c["action_digest"] or action.domain != state.space:
        raise ValueError("native model action changed")
    return action


def infer_model(state, source, inputs=None):
    """Evaluate the declared model without fitting a surrogate for an exact action."""
    if not isinstance(state, ModelState):
        raise TypeError("model inference requires ModelState")
    reference = state.bind(source)
    if state.adapter == "native_field":
        if state.adapter_version != "1": raise ValueError("unsupported native model version")
        if not isinstance(inputs, TensorField): raise TypeError("native model input requires an exact TensorField")
        if inputs.source is None or not inputs.source.matches(state.source):
            raise ValueError("model input source differs from the declared state")
        bound = inputs.bind(source)
        out = apply_tensor(_calculus(state, source), bound)
        return ModelOutput(out.values, out.space, out.axes, reference, state.coefficient_digest,
                           "rational", "native-rational-field-action", out.dependencies, out.grade, out.variance)
    if state.adapter == "online_flow":
        from rexgraph.flow.model import infer_online
        return infer_online(state, source, inputs)
    from rexgraph.nn.lifecycle import infer_checkpoint
    return infer_checkpoint(state, source, inputs)


def train_model(state, source, batch, *, steps=1):
    """Return a new immutable state; publication is a separate operation."""
    state.bind(source)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("learning steps must be a positive integer")
    if not isinstance(batch, ModelBatch): raise TypeError("learning requires ModelBatch")
    batch.check_state()
    if batch.space != state.space or not batch.source.matches(state.source):
        raise ValueError("training observations use another source or coordinate order")
    if state.adapter == "native_field":
        raise TypeError("native exact field actions do not require training")
    if state.adapter == "online_flow":
        if steps != 1: raise ValueError("online correction is one declared observation step")
        from rexgraph.flow.model import correct_online
        return correct_online(state, source, batch)
    from rexgraph.nn.lifecycle import train_checkpoint
    return train_checkpoint(state, source, batch, steps=steps)


def transport_model(state, source, destination, mapping, *, optimizer="reset"):
    """Transport declared local parameters without guessing optimizer covariances."""
    state.bind(source)
    if not isinstance(destination, FieldSource) or destination.source is None:
        raise TypeError("model transport requires a live destination FieldSource")
    destination.check()
    if state.adapter == "online_flow":
        if optimizer != "reset":
            raise ValueError("online flow has no optimizer state to carry")
        from rexgraph.flow.model import transport_online
        return transport_online(state, source, destination, mapping)
    if state.adapter == "native_field":
        raise TypeError("declare the new native action and metrics, then use NativeFieldEvolution")
    from rexgraph.nn.lifecycle import transport_checkpoint
    return transport_checkpoint(state, source, destination, mapping, optimizer=optimizer)


def certify_native_response(state, source, inputs, candidate, *, recorded_binary=False):
    """Retain an exact residual and a quadrance bound for the declared Green response."""
    from rexgraph.tensor_moment import MomentSpan, CoordinatePairing
    if not isinstance(state, ModelState) or state.adapter != "native_field" or state.configuration["operation"] != "green":
        raise TypeError("response certification requires a native positive Green model")
    state.bind(source)
    if not isinstance(recorded_binary, bool):
        raise TypeError("recorded binary interpretation must be explicitly boolean")
    interpretation = "rational candidate"
    if isinstance(candidate, ModelOutput):
        contract = candidate.arithmetic
        if contract == "approximate" and not recorded_binary:
            raise TypeError("numerical candidates require explicit recorded_binary=True")
        interpretation = "recorded binary candidate" if contract == "approximate" else interpretation
        original_digest = candidate.coefficient_digest
        candidate = candidate.recorded_binary() if contract == "approximate" else candidate.tensor()
    elif isinstance(candidate, TensorField):
        contract = "rational"
        original_digest = candidate.coefficient_digest
    else:
        raise TypeError("candidate requires a retained model output or tensor field")
    if not isinstance(inputs, TensorField):
        raise TypeError("native response source requires an exact tensor field")
    for field in (candidate, inputs):
        field.check_state()
        if field.source is None or not field.source.matches(state.source):
            raise ValueError("response certificate source identity differs")
        if field.space != state.space or field.grade != state.configuration["grade"] or field.variance != "chain":
            raise ValueError("response certificate coordinates or variance differ")
    if candidate.axes != inputs.axes:
        raise ValueError("response certificate must retain matching field axes")
    action = _calculus(state, source)
    transported = apply_tensor(action.owner.hodge(action.grade), candidate.bind(source))
    residual = TensorField(inputs.space, inputs.values-candidate.values-action.parameter*transported.values,
        inputs.axes, state.source, inputs.grade, inputs.variance,
        (state.coefficient_digest, inputs.coefficient_digest, original_digest),
        (*inputs.dependencies, *candidate.dependencies))
    bound = MomentSpan(residual, residual, CoordinatePairing.metric(action.owner.metrics[action.grade]),
                       ("residual", "residual"), state.coefficient_digest, (state.source,))
    return {"residual": residual, "quadrance_bound": bound,
            "exact_solution": not any(v != 0 for v in residual.values.flat),
            "candidate_arithmetic": contract, "interpretation": interpretation,
            "candidate_digest": original_digest, "model_digest": state.coefficient_digest,
            "bound_scope": "error quadrance for each linear combination of retained field columns"}
