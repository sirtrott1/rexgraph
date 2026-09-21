"""Local model queries over selected native data and immutable checkpoints."""
from __future__ import annotations
from .execution_trace import current_binding, current_policy, record_method
from .binding import Binding
from .capabilities import SourcePolicy


def source_for(source, state=None):
    from rexgraph.model_state import ModelState
    from rexgraph.io.model_record import read_model_record, read_model_history
    first = source.get_metadata(1, 0, "model") if source.nE else None
    if isinstance(first, ModelState):
        if source.get_metadata(1, 0, "model_timeline") is not None:
            _, snapshots = read_model_history(source)
            if state is None: return snapshots[-1][1]
            for saved, data in snapshots:
                if saved.source.matches(state.source):
                    state.bind(data)
                    return data
            raise ValueError("model source is not present in the selected history")
        _, data = read_model_record(source)
    else:
        data = source
    if state is not None: state.bind(data)
    return data


def authorize(source, state, binding=None):
    data = source_for(source, state)
    if binding is not None:
        binding.source.require("read")
        if data is source:
            if state.source.record_id is not None and state.source.record_id != binding.ref.record_id:
                raise ValueError("model source record identity differs")
            if state.source.version is not None and state.source.version != binding.ref.record_version:
                raise ValueError("model source version differs")
    return data


def _reference(source):
    from rexgraph.tensor_field import FieldSource
    binding = current_binding()
    if binding is None: return FieldSource(source)
    return FieldSource(source, binding.ref.record_id, binding.ref.record_version)


def model_init(source, adapter="native_field", configuration=None, axes=None, optimizer=None, seed=0):
    from rexgraph.model_runtime import native_model
    config = dict(configuration or {})
    if adapter == "native_field":
        if optimizer is not None or axes is not None:
            raise ValueError("native field action has no neural optimizer or fixed output feature axes")
        state = native_model(source, reference=_reference(source), **config)
    elif adapter == "online_flow":
        if optimizer is not None or axes is not None:
            raise ValueError("online correction declares no neural optimizer")
        from rexgraph.flow.model import online_model
        state = online_model(source, reference=_reference(source), **config)
    else:
        from rexgraph.nn.lifecycle import create_checkpoint
        state = create_checkpoint(source, adapter=adapter, configuration=config, axes=axes,
                                  optimizer=optimizer, reference=_reference(source), seed=seed)
    record_method("native-local-model-declaration", adapter=state.adapter, model_digest=state.coefficient_digest,
                  arithmetic=state.arithmetic)
    return state


def model_state(source, index=0):
    from rexgraph.io.model_record import read_model_record
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("model cell index must be nonnegative")
    state, _ = read_model_record(source, index)
    record_method("selected-native-model-state", model_digest=state.coefficient_digest)
    return state


def model_at(source, time):
    from rexgraph.io.model_record import read_model_history
    state, _ = read_model_history(source, time)
    record_method("exact-model-history-selection", model_digest=state.coefficient_digest)
    return state


def model_batch(source, state, targets, observed, inputs=None, contributors=()):
    from rexgraph.model_state import ModelBatch
    authorize(source, state, current_binding())
    refs = []
    for binding in contributors:
        if not isinstance(binding, Binding): raise TypeError("model evidence requires explicit source Bindings")
        binding.source.require("read")
        if SourcePolicy.intersection(current_policy(), binding.source.policy).digest != current_policy().digest:
            raise PermissionError("model query policy exceeds an evidence contributor policy")
        from rexgraph.tensor_field import FieldSource
        refs.append(FieldSource(binding.value, binding.ref.record_id, binding.ref.record_version))
    record_method("declared-model-observations")
    return ModelBatch(state.source, state.space, targets, observed, inputs, tuple(refs))


def model_infer(source, state, inputs=None):
    from rexgraph.model_runtime import infer_model
    data = authorize(source, state, current_binding())
    output = infer_model(state, data, inputs)
    record_method(output.method, arithmetic=output.arithmetic, model_digest=output.model_digest,
                  retained_axes=tuple((a.name, a.keys) for a in output.axes), source=state.source.as_record())
    return output


def model_train(source, state, batch, steps=1):
    from rexgraph.model_runtime import train_model
    if not current_policy().permits("train"): raise PermissionError("model learning requires train permission")
    data = authorize(source, state, current_binding())
    result = train_model(state, data, batch, steps=steps)
    record_method("native-model-observation-update", arithmetic=result.arithmetic,
                  parent=state.coefficient_digest, model_digest=result.coefficient_digest, step=result.step)
    return result


def model_transport(source, state, destination, mapping, optimizer="reset"):
    from rexgraph.model_runtime import transport_model
    from rexgraph.tensor_field import FieldSource
    data = authorize(source, state, current_binding())
    if not isinstance(destination, Binding): raise TypeError("model transport requires a destination Binding")
    destination.source.require("read")
    if SourcePolicy.intersection(current_policy(), destination.source.policy).digest != current_policy().digest:
        raise PermissionError("model transport exceeds the destination policy")
    reference = FieldSource(destination.value, destination.ref.record_id, destination.ref.record_version)
    result = transport_model(state, data, reference, mapping, optimizer=optimizer)
    record_method("declared-model-parameter-transport", arithmetic=result.arithmetic,
                  model_digest=result.coefficient_digest, optimizer_policy=optimizer)
    return result


def model_record(source, state):
    from rexgraph.io.model_record import model_record as make_record
    data = authorize(source, state, current_binding())
    record_method("native-model-record-construction", model_digest=state.coefficient_digest)
    return make_record(data, state)


def model_history(source, states, sources, times, time_axis="model_observation", time_unit="step"):
    from rexgraph.io.model_record import model_history as make_history
    states, sources = tuple(states), tuple(sources)
    if not states or len(states) != len(sources):
        raise ValueError("model history requires one binding per checkpoint")
    data = []
    for state, binding in zip(states, sources, strict=True):
        if not isinstance(binding, Binding):
            raise TypeError("model history sources require explicit Bindings")
        binding.source.require("read")
        if SourcePolicy.intersection(current_policy(), binding.source.policy).digest != current_policy().digest:
            raise PermissionError("model history query exceeds a contributor policy")
        data.append(authorize(binding.value, state, binding))
    result = make_history(states, data, times, time_axis=time_axis, time_unit=time_unit)
    record_method("native-model-history-construction", snapshots=len(states))
    return result


def model_field(source, output):
    from rexgraph.model_state import ModelOutput
    if not isinstance(output, ModelOutput): raise TypeError("model field requires ModelOutput")
    field = output.tensor()
    data = source_for(source)
    field = field.bind(data)
    record_method("exact-model-field", model_digest=output.model_digest, arithmetic="rational")
    return field


def model_values(source, output):
    output.check_state()
    record_method("model-value-observation", arithmetic=output.arithmetic, model_digest=output.model_digest)
    return output.values.copy()


def model_input(source, value, dtype="float64"):
    from rexgraph.model_state import model_input as convert
    from rexgraph.tensor_field import TensorField
    if not isinstance(value, TensorField):
        value.check_state()
    result = convert(value, dtype)
    record_method("declared-model-input-conversion", arithmetic=result.arithmetic,
                  origin=result.origin_digest, conversion_digest=result.coefficient_digest,
                  dtype=str(result.values.dtype))
    return result


def model_certify(source, state, inputs, candidate, recorded_binary=False):
    from rexgraph.model_runtime import certify_native_response
    data = authorize(source, state, current_binding())
    result = certify_native_response(state, data, inputs, candidate, recorded_binary=recorded_binary)
    record_method("exact-native-response-certificate", candidate_arithmetic=result["candidate_arithmetic"],
                  interpretation=result["interpretation"], model_digest=state.coefficient_digest)
    return result


def model_info(source, state):
    authorize(source, state, current_binding())
    record_method("native-model-descriptor")
    return {"adapter": state.adapter, "adapter_version": state.adapter_version, "step": state.step,
            "digest": state.coefficient_digest, "parent": state.parent, "arithmetic": state.arithmetic,
            "source": state.source.as_record(), "resumable": "optimizer" in state.payload,
            "axes": tuple((a.name, a.keys) for a in (state.space, *state.output_axes))}


def install(register):
    for name, fn in (("MODEL_INIT", model_init), ("MODEL_STATE", model_state), ("MODEL_AT", model_at),
                     ("MODEL_BATCH", model_batch), ("MODEL_INFER", model_infer), ("MODEL_TRAIN", model_train),
                     ("MODEL_TRANSPORT", model_transport), ("MODEL_RECORD", model_record), ("MODEL_HISTORY", model_history),
                     ("MODEL_FIELD", model_field), ("MODEL_VALUES", model_values), ("MODEL_INPUT", model_input), ("MODEL_INFO", model_info), ("MODEL_CERTIFY", model_certify)):
        register(name)(fn)
