"""Source bound training and inference with explicit checkpoint continuation."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Mapping
from threading import RLock
import platform
import random
import numpy as np

from rexgraph.model_state import ModelState, ModelOutput, ModelInput, thaw_tree
from rexgraph.tensor_field import FieldSource
from rexgraph.type_accession import CoordinateSpace
from rexgraph.model_runtime import model_coordinates

__all__ = ["ModelAdapter", "register_model_adapter", "create_checkpoint", "capture_checkpoint",
           "infer_checkpoint", "train_checkpoint", "restore_checkpoint", "transport_checkpoint"]
_LOCK = RLock()


@dataclass(frozen=True)
class ModelAdapter:
    """Trusted callbacks use declared inputs and captured state, not hidden mutable data."""
    version: str
    build: object
    forward: object
    loss: object
    parameter_axes: object = None
    coordinates: object = None
    grade: int | None = 1
    variance: str = "cochain"
    transport_static: bool = False
    reduction: str = "mean"
    implementation_modules: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("model adapter requires an implementation version")
        if any(not callable(v) for v in (self.build, self.forward, self.loss)):
            raise TypeError("model adapter requires construction, forward and per row loss functions")
        if self.parameter_axes is not None and not callable(self.parameter_axes):
            raise TypeError("parameter axis declaration must be callable")
        if self.coordinates is not None and not callable(self.coordinates):
            raise TypeError("model coordinate declaration must be callable")
        if self.grade is not None and (isinstance(self.grade, bool) or not isinstance(self.grade, int) or self.grade < 0):
            raise ValueError("invalid model coordinate grade")
        if any(not isinstance(name, str) or not name for name in self.implementation_modules):
            raise TypeError("implementation module names must be declared strings")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("declare mean or sum for the observed numerical objective")
        if self.variance not in {"coordinate", "chain", "cochain"}:
            raise ValueError("invalid model coordinate variance")


def register_model_adapter(name, adapter):
    """Use the existing component registry with an explicit lifecycle contract."""
    from rexgraph.nn.factory import register
    if not isinstance(name, str) or not name or not isinstance(adapter, ModelAdapter):
        raise TypeError("register a named ModelAdapter")
    register("lifecycle", name, adapter, native=True, available_fn=lambda: True)


def _builtins():
    from rexgraph.nn.factory import _REG
    if "coparticipation" in _REG.get("lifecycle", {}): return
    def build(source, config):
        import torch
        from rexgraph.flow.cochain import CoParticipationCochain
        return CoParticipationCochain(source, config["n_classes"],
            green_lam=config.get("green_lam", 4.0), green_iters=config.get("green_iters", 20),
            green_channel=config.get("green_channel", "low"), restrict_vertices=config.get("restrict_vertices"),
            dtype=getattr(torch, config.get("dtype", "float64")))
    def forward(model, inputs):
        if inputs is not None: raise ValueError("a bare cochain model has no feature input")
        return model()
    def loss(pred, target):
        import torch
        if target.dtype != torch.int64 or target.ndim != 1:
            raise TypeError("cochain classification targets must be integer class indices")
        return torch.nn.functional.cross_entropy(pred, target, reduction="none")
    register_model_adapter("coparticipation", ModelAdapter("1", build, forward, loss, lambda m: {"Z": 0}, transport_static=True,
        implementation_modules=("rexgraph.flow.cochain", "rexgraph.nn.optim", "rexgraph.nn.factory")))


def _adapter(name, version=None):
    from rexgraph.nn.factory import _REG, _avail
    _builtins()
    item = _REG.get("lifecycle", {}).get(name)
    if item is None or not _avail(item): raise KeyError(f"unavailable model lifecycle adapter {name!r}")
    adapter = item["factory"]
    if not isinstance(adapter, ModelAdapter): raise TypeError("invalid registered model adapter")
    if version is not None and version != adapter.version: raise ValueError("model implementation version differs")
    return adapter


def _implementation(adapter):
    import ast
    import hashlib
    import importlib
    import inspect
    import textwrap
    code = {}
    for name in ("build", "forward", "loss", "parameter_axes", "coordinates"):
        callback = getattr(adapter, name)
        if callback is None:
            code[name] = None
            continue
        try:
            normalized = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(callback))), include_attributes=False)
            code[name] = hashlib.sha256(normalized.encode()).hexdigest()
        except (OSError, TypeError, IndentationError, SyntaxError):
            code[name] = "declared version " + adapter.version
    modules = {}
    for name in adapter.implementation_modules:
        module = importlib.import_module(name)
        try:
            normalized = ast.dump(ast.parse(inspect.getsource(module)), include_attributes=False)
        except (OSError, TypeError, SyntaxError) as exc:
            raise ValueError("declared implementation module has no inspectable Python source") from exc
        modules[name] = hashlib.sha256(normalized.encode()).hexdigest()
    return {"callbacks": code, "modules": modules, "adapter_version": adapter.version}


def _coordinates(adapter, source, configuration):
    value = (model_coordinates(source, 1 if adapter.grade is None else adapter.grade)
             if adapter.coordinates is None else adapter.coordinates(source, configuration))
    if not isinstance(value, CoordinateSpace):
        raise TypeError("model coordinate function must return CoordinateSpace")
    return value


def _tree_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided: raise TypeError("checkpoint state uses dense parameter tensors; rebuild sparse operators")
        if value.dtype == torch.bfloat16:
            return {"__torch_bfloat16__": value.detach().cpu().contiguous().view(torch.uint16).numpy().copy()}
        return value.detach().cpu().numpy().copy()
    if isinstance(value, Mapping): return {k: _tree_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return tuple(_tree_cpu(v) for v in value)
    return value


def _tree_torch(value, device):
    import torch
    if isinstance(value, np.ndarray): return torch.from_numpy(value.copy()).to(device)
    if isinstance(value, Mapping):
        if set(value) == {"__torch_bfloat16__"}:
            return torch.from_numpy(value["__torch_bfloat16__"].copy()).view(torch.bfloat16).to(device)
        return {k: _tree_torch(v, device) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(_tree_torch(v, device) for v in value)
    return value


def _rng():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state().cpu().numpy().copy(),
            "cuda": tuple(v.cpu().numpy().copy() for v in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()}


def _set_rng(state, *, require_cuda=True):
    import torch
    random.setstate(tuple(thaw_tree(state["python"])))
    np.random.set_state(tuple(thaw_tree(state["numpy"])))
    torch.random.set_rng_state(torch.from_numpy(state["torch"].copy()))
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            if require_cuda:
                raise ValueError("checkpoint CUDA RNG topology is unavailable")
            return
        torch.cuda.set_rng_state_all([torch.from_numpy(v.copy()) for v in state["cuda"]])


@contextmanager
def _isolated_rng(state=None, seed=None, *, require_cuda=True):
    import torch
    with _LOCK:
        previous = _rng()
        try:
            if state is not None: _set_rng(state, require_cuda=require_cuda)
            elif seed is not None:
                random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
            yield
        finally:
            _set_rng(previous)


def _environment():
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__)}


def _operator_digest(value):
    from rexgraph.io.rex_state import state_digest
    import torch
    if value is None: return None
    if not isinstance(value, torch.Tensor): raise TypeError("optimizer operator must be a tensor")
    if value.layout == torch.sparse_coo:
        value = value.coalesce()
        tensors = {"indices": value.indices().detach().cpu().numpy(),
                   "values": value.values().detach().cpu().numpy(), "shape": np.array(value.shape, dtype=np.int64)}
    elif value.layout == torch.strided:
        tensors = {"values": value.detach().cpu().numpy()}
    else: raise TypeError("unsupported optimizer operator layout")
    return state_digest(tensors)


def _optimizer(model, specification):
    from rexgraph.nn.factory import make_optimizer
    spec = thaw_tree(specification)
    name = spec.pop("name", "auto")
    expected = spec.pop("resolved", None)
    opt, label = make_optimizer(name, model, model.parameters(), **spec)
    for group in opt.param_groups:
        operator = group.get("green_adj")
        if operator is not None:
            placements = {(p.device, p.dtype) for p in group["params"]}
            if len(placements) != 1:
                raise ValueError("one Green parameter group requires one dtype and device")
            device, dtype = next(iter(placements))
            group["green_adj"] = operator.to(device=device, dtype=dtype).coalesce()
            group.pop("_op_cache", None)
    identity = type(opt).__module__ + "." + type(opt).__qualname__
    if expected is not None and expected != identity:
        raise ValueError("optimizer dispatch differs from the recorded implementation")
    return opt, {**dict(specification), "name": name, "resolved": identity}, label


def _capture_optimizer(model, opt):
    state = opt.state_dict()
    groups = []
    operators = []
    names = {id(p): n for n, p in model.named_parameters()}
    parameters = []
    for live, saved in zip(opt.param_groups, state["param_groups"], strict=True):
        groups.append({k: v for k, v in saved.items() if k not in {"green_adj", "_op_cache"}})
        operators.append(_operator_digest(live.get("green_adj")))
        parameters.append(tuple(names[id(p)] for p in live["params"]))
    return _tree_cpu({"state": state["state"], "param_groups": groups}), tuple(operators), tuple(parameters)


def capture_checkpoint(model, source, *, adapter="coparticipation", configuration=None, optimizer=None,
                       optimizer_spec=None, axes=None, reference=None, step=0, parent=None,
                       observations=None, rng=None, dependencies=()):
    """Capture parameters and optional training continuation without modifying the model."""
    impl = _adapter(adapter)
    configuration = dict(configuration or {})
    if getattr(model, "_rex", source) is not source:
        raise ValueError("model is attached to another native source")
    reference = FieldSource(source) if reference is None else reference.bind(source)
    if axes is None:
        count = int(configuration.get("n_classes", getattr(getattr(model, "Z", None), "shape", (0, 0))[1]))
        if count < 1:
            raise ValueError("declare the model output axes explicitly")
        axes = (CoordinateSpace("model_classes", tuple(str(i) for i in range(count))),)
    axes = tuple(axes)
    space = _coordinates(impl, source, configuration)
    parameter_axes = {} if impl.parameter_axes is None else dict(impl.parameter_axes(model))
    names = dict(model.named_parameters())
    for name, axis in parameter_axes.items():
        if name not in names or not isinstance(axis, int) or not 0 <= axis < names[name].ndim:
            raise ValueError("invalid model parameter axis declaration")
        if names[name].shape[axis] != len(space.keys):
            raise ValueError("model parameter axis differs from primary cells")
    payload = {"weights": _tree_cpu(model.state_dict()), "parameter_axes": parameter_axes,
               "observations": observations, "environment": _environment(), "implementation": _implementation(impl),
               "output_grade": impl.grade, "output_variance": impl.variance,
               "transport_static": impl.transport_static, "parameter_axes_declared": impl.parameter_axes is not None,
               "objective_reduction": impl.reduction,
               "rng": _rng() if rng is None else rng, "module_modes": {n: m.training for n, m in model.named_modules()}}
    if optimizer is not None:
        saved, operators, names = _capture_optimizer(model, optimizer)
        payload.update(optimizer=saved, operator_digests=operators, optimizer_parameters=names)
        if optimizer_spec is None: raise ValueError("resumable checkpoint requires the optimizer specification")
        configuration["optimizer"] = dict(optimizer_spec)
        configuration["optimizer"]["resolved"] = type(optimizer).__module__ + "." + type(optimizer).__qualname__
    return ModelState(adapter, impl.version, reference, space, axes,
                      configuration, payload, "approximate", step, parent, tuple(dependencies))


def create_checkpoint(source, *, adapter="coparticipation", configuration=None, axes=None,
                      optimizer=None, reference=None, seed=0):
    """Construct through the registered model and existing optimizer factory."""
    if isinstance(seed, bool) or not isinstance(seed, int): raise TypeError("model seed must be an integer")
    impl = _adapter(adapter)
    config = dict(configuration or {})
    if "optimizer" in config: raise ValueError("supply optimizer as its own declared specification")
    with _isolated_rng(seed=seed):
        model = impl.build(source, config)
        opt, spec, label = _optimizer(model, {"name": "auto"} if optimizer is None else optimizer)
        result = capture_checkpoint(model, source, adapter=adapter, configuration=config, optimizer=opt,
            optimizer_spec=spec, axes=axes, reference=reference, rng=_rng())
    return result


def _restore(state, source, *, resume, device, strict_environment=True):
    import torch
    state.bind(source)
    impl = _adapter(state.adapter, state.adapter_version)
    if state.arithmetic != "approximate": raise TypeError("neural checkpoint requires its numerical contract")
    if thaw_tree(state.payload["implementation"]) != _implementation(impl):
        raise ValueError("registered model implementation differs from its checkpoint")
    if state.payload["objective_reduction"] != impl.reduction:
        raise ValueError("registered objective reduction differs from the checkpoint")
    if resume and strict_environment and dict(state.payload["environment"]) != _environment():
        raise ValueError("training continuation environment differs; explicitly opt into migration")
    config = thaw_tree(state.configuration)
    spec = config.pop("optimizer", None)
    model = impl.build(source, config).to(device)
    recorded = _tree_torch(state.payload["weights"], device)
    current = model.state_dict()
    if set(recorded) != set(current):
        raise ValueError("rebuilt model state names differ")
    for name, value in recorded.items():
        if isinstance(value, torch.Tensor):
            candidate = current[name]
            if not isinstance(candidate, torch.Tensor) or candidate.shape != value.shape or candidate.dtype != value.dtype:
                raise ValueError("rebuilt model tensor shape or dtype differs")
    model.load_state_dict(recorded, strict=True)
    if _coordinates(impl, source, config) != state.space:
        raise ValueError("model primary coordinate order changed")
    modes = state.payload["module_modes"]
    for name, module in model.named_modules():
        if name not in modes: raise ValueError("model module structure differs")
        module.training = bool(modes[name])
    opt = None
    if resume:
        if "optimizer" not in state.payload or spec is None:
            raise ValueError("inference checkpoint does not contain training continuation")
        opt, _, _ = _optimizer(model, spec)
        native_operators = [g.get("green_adj") for g in opt.param_groups]
        for i, (op, digest) in enumerate(zip(native_operators, state.payload["operator_digests"], strict=True)):
            if _operator_digest(op) != digest:
                raise ValueError("rebuilt optimizer operator differs from its checkpoint")
        names = {id(p): n for n, p in model.named_parameters()}
        current = tuple(tuple(names[id(p)] for p in group["params"]) for group in opt.param_groups)
        if current != tuple(state.payload["optimizer_parameters"]):
            raise ValueError("optimizer parameter order differs")
        saved = _tree_torch(state.payload["optimizer"], device)
        saved["param_groups"] = list(saved["param_groups"])
        opt.load_state_dict(saved)
        for group, operator in zip(opt.param_groups, native_operators, strict=True):
            if operator is not None: group["green_adj"] = operator
            group.pop("_op_cache", None)
    return model, opt, impl


def restore_checkpoint(state, source, *, resume=False, device="cpu", strict_environment=True):
    """Restore executable state while leaving global random generators unchanged."""
    with _isolated_rng(state.payload["rng"], require_cuda=resume):
        model, optimizer, _ = _restore(state, source, resume=resume, device=device,
                                       strict_environment=strict_environment)
    return model, optimizer


def _inputs(value, device):
    import torch
    if isinstance(value, ModelInput):
        value.check_state()
        return torch.from_numpy(value.values.copy()).to(device)
    if value is None: return None
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf": raise TypeError("model input must use a declared numerical dtype")
        if value.dtype.kind == "f" and not np.isfinite(value).all(): raise ValueError("nonfinite model input")
        return torch.from_numpy(value.copy()).to(device)
    if isinstance(value, Mapping): return {k: _inputs(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return tuple(_inputs(v, device) for v in value)
    raise TypeError("model inputs must be explicit tensor data")


def _input_dependencies(state, inputs):
    found = []
    def visit(value):
        if isinstance(value, ModelInput):
            value.check_state()
            if value.space != state.space or not value.source.matches(state.source):
                raise ValueError("model input source or coordinates differ")
            found.extend((value.source, *value.dependencies))
        elif isinstance(value, Mapping):
            for v in value.values(): visit(v)
        elif isinstance(value, (tuple, list)):
            for v in value: visit(v)
    visit(inputs)
    return tuple({v.coefficient_digest: v for v in found}.values())


def infer_checkpoint(state, source, inputs=None, *, device="cpu"):
    import torch
    dependencies = _input_dependencies(state, inputs)
    with _isolated_rng(state.payload["rng"], require_cuda=False), torch.no_grad():
        model, _, adapter = _restore(state, source, resume=False, device=device)
        model.eval()
        values = adapter.forward(model, _inputs(inputs, device))
        if not isinstance(values, torch.Tensor): raise TypeError("model adapter must return one retained tensor")
        if values.dtype == torch.bfloat16:
            raise TypeError("declare a float output conversion in the adapter for bfloat16 inference")
        values = values.detach().cpu().numpy().copy()
    return ModelOutput(values, state.space, state.output_axes, state.source, state.coefficient_digest,
                       "approximate", "registered-native-model-inference", (*state.dependencies, *dependencies),
                       state.payload["output_grade"], state.payload["output_variance"])


def train_checkpoint(state, source, batch, *, steps=1, device="cpu", strict_environment=True):
    import torch
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("training steps must be a positive integer")
    state.bind(source); batch.check_state()
    if batch.space != state.space or not batch.source.matches(state.source):
        raise ValueError("training batch source or coordinates differ")
    if not batch.observed.any(): raise ValueError("training requires an explicitly observed target")
    input_dependencies = _input_dependencies(state, (batch.inputs, batch.target_input))
    with _isolated_rng(state.payload["rng"]):
        model, optimizer, adapter = _restore(state, source, resume=True, device=device,
                                             strict_environment=strict_environment)
        # Construction can consume random numbers. Continuation starts at the saved state.
        _set_rng(state.payload["rng"])
        model.train()
        mask = torch.from_numpy(batch.observed.copy()).to(device)
        targets = _inputs(batch.targets, device)
        inputs = _inputs(batch.inputs, device)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = adapter.forward(model, inputs)
            expected = (len(state.space.keys), *(len(a.keys) for a in state.output_axes))
            if tuple(prediction.shape) != expected: raise ValueError("model output coordinates changed")
            # Unobserved labels never enter the objective, even as placeholders.
            terms = adapter.loss(prediction[mask], targets[mask])
            if not isinstance(terms, torch.Tensor) or terms.ndim != 1 or terms.shape[0] != int(mask.sum()):
                raise ValueError("model objective must return one loss per observed row")
            loss = terms.mean() if adapter.reduction == "mean" else terms.sum()
            if not torch.isfinite(loss): raise ArithmeticError("nonfinite training objective")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ArithmeticError("nonfinite training gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        config = thaw_tree(state.configuration)
        spec = config.pop("optimizer")
        dependencies = tuple({v.coefficient_digest: v for v in (*state.dependencies, *batch.dependencies, *input_dependencies, batch.source)}.values())
        observations = {"batch_digest": batch.coefficient_digest, "targets": batch.targets,
                        "observed": batch.observed, "inputs": batch.inputs, "target_input": batch.target_input, "losses": tuple(losses),
                        "loss_rows": np.flatnonzero(batch.observed), "loss_values": _tree_cpu(terms),
                        "objective_reduction": adapter.reduction,
                        "target_source": batch.source.as_record()}
        return capture_checkpoint(model, source, adapter=state.adapter, configuration=config,
            optimizer=optimizer, optimizer_spec=spec, axes=state.output_axes, reference=state.source,
            step=state.step + steps, parent=state.coefficient_digest, observations=observations,
            rng=_rng(), dependencies=dependencies)


def transport_checkpoint(state, source, destination, mapping, *, optimizer="reset"):
    """Transport declared primary parameter axes with an explicit continuation policy."""
    import torch  # noqa: F401  transport requires the optional torch runtime
    from rexgraph.coordinate_map import CoordinateMap
    state.bind(source); destination.check()
    adapter = _adapter(state.adapter, state.adapter_version)
    if thaw_tree(state.payload["implementation"]) != _implementation(adapter):
        raise ValueError("registered model implementation differs from its checkpoint")
    target_space = _coordinates(adapter, destination.source, state.configuration)
    if not isinstance(mapping, CoordinateMap) or mapping.domain != state.space or mapping.codomain != target_space:
        raise ValueError("parameter transport requires declared endpoint coordinates")
    if optimizer not in {"reset", "carry"}: raise ValueError("unknown optimizer transport policy")
    if not adapter.transport_static or not state.payload["transport_static"]:
        raise ValueError("adapter must declare remaining parameters and buffers independent of source coordinates before transport")
    axes = state.payload["parameter_axes"]
    if not state.payload["parameter_axes_declared"]:
        raise ValueError("model adapter has not declared its parameter axis policy")
    rows = {}; columns = set()
    permutation = len(mapping.domain.keys) == len(mapping.codomain.keys)
    for i, j, value in mapping.entries:
        if value != 1 or i in rows or j in columns: permutation = False
        rows[i] = j; columns.add(j)
    permutation = permutation and len(rows) == len(mapping.domain.keys)
    if optimizer == "carry" and "optimizer" not in state.payload:
        raise ValueError("checkpoint has no optimizer state to carry")
    if optimizer == "carry" and not permutation:
        raise ValueError("optimizer moments can be carried only through a bijective coordinate permutation")
    config = thaw_tree(state.configuration)
    spec = config.pop("optimizer", {"name": "auto"})
    with _isolated_rng(state.payload["rng"]):
        adapter = _adapter(state.adapter, state.adapter_version)
        model = adapter.build(destination.source, config)
        weights = thaw_tree(state.payload["weights"])
        for name, axis in axes.items():
            old = weights[name]
            a = np.moveaxis(old, axis, 0)
            out = np.zeros((len(mapping.codomain.keys), *a.shape[1:]), dtype=old.dtype)
            for i, j, value in mapping.entries: out[i] += float(value) * a[j]
            if not np.isfinite(out).all(): raise ArithmeticError("nonfinite transported parameter")
            weights[name] = np.moveaxis(out, 0, axis)
        model.load_state_dict(_tree_torch(weights, "cpu"), strict=True)
        opt, spec, _ = _optimizer(model, spec)
        if optimizer == "carry":
            saved = thaw_tree(state.payload["optimizer"])
            for group, names in zip(saved["param_groups"], state.payload["optimizer_parameters"], strict=True):
                for identifier, name in zip(group["params"], names, strict=True):
                    axis = axes.get(name)
                    if axis is None: continue
                    for key, value in saved["state"].get(identifier, {}).items():
                        if isinstance(value, np.ndarray) and value.shape == state.payload["weights"][name].shape:
                            saved["state"][identifier][key] = np.take(value, [rows[i] for i in range(len(rows))], axis=axis)
            operators = [g.get("green_adj") for g in opt.param_groups]
            saved["param_groups"] = list(saved["param_groups"])
            opt.load_state_dict(_tree_torch(saved, "cpu"))
            for group, operator in zip(opt.param_groups, operators, strict=True):
                if operator is not None: group["green_adj"] = operator
                group.pop("_op_cache", None)
        result = capture_checkpoint(model, destination.source, adapter=state.adapter, configuration=config,
            optimizer=opt, optimizer_spec=spec, axes=state.output_axes, reference=destination,
            step=state.step, parent=state.coefficient_digest, observations=None, rng=state.payload["rng"],
            dependencies=(*state.dependencies, state.source))
    payload = thaw_tree(result.payload)
    payload["transport"] = {"map_digest": mapping.coefficient_digest, "map_entries": mapping.entries,
                            "domain": (mapping.domain.name, mapping.domain.keys),
                            "codomain": (mapping.codomain.name, mapping.codomain.keys), "optimizer": optimizer,
                            "observation_policy": "new targets must be supplied explicitly"}
    return ModelState(result.adapter, result.adapter_version, result.source, result.space, result.output_axes,
                      result.configuration, payload, result.arithmetic, result.step, result.parent, result.dependencies)
