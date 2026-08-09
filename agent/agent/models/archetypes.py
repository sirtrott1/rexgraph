"""
archetypes: the model types, each built on the rexgraph.nn substrate.

Every archetype exposes: a `use_case`, a `defaults` param dict, the `data_kind` it consumes, a
`synth` generator, and `build(cfg, bundle)` returning an nn.Module. Select an archetype by name and
override its defaults. Register new archetypes with `register_archetype(...)`.

Components come from rexgraph.nn (PropagatorAttention, build_attention, the rcf_torch propagators).
These archetypes assemble models from that substrate; they are not part of the library.

No archetype builds an optimizer: `build(cfg, bundle)` returns the module and `train.train_one`
routes it through `make_optimizer("auto")`. All four are feature-space models (they consume a
feature matrix; none exposes `greens_groups()`), so the router gives them plain Adam. `hgnn` uses
the complex as a fixed operator (B1/L0/L1 buffers) rather than as its parameter space, so it is
feature-space too: Green's/Hodge preconditioning has nothing to precondition there. For the
relational-native path, where the parameters ARE a cochain on the complex, see
`rexgraph.flow.cochain`.
"""
from __future__ import annotations

from . import data as _data

ARCHETYPES: dict = {}


def register_archetype(name, *, use_case, data_kind, defaults, build, synth):
    ARCHETYPES[name] = {"name": name, "use_case": use_case, "data_kind": data_kind,
                        "defaults": dict(defaults), "build": build, "synth": synth}


def get(name):
    if name not in ARCHETYPES:
        raise KeyError(f"unknown archetype {name!r} (have: {', '.join(sorted(ARCHETYPES))})")
    return ARCHETYPES[name]


def merged_cfg(name, overrides=None):
    cfg = dict(get(name)["defaults"])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def _model_build(fn_name):
    """A build callable that imports the torch half only when it is actually called.

    The archetypes are DATA: a name, a use-case, the kind of data it consumes and its
    default params. None of that needs torch, and listing them is what the CLI, the
    /ml/archetypes route and list_archetypes() do. The models are torch modules, so
    importing them pulls the whole ml extra in, and doing that at module scope meant
    `rexgraph-models --help` could not run without it.
    """
    def build(cfg, bundle):
        try:
            from . import _torch_models
        except ModuleNotFoundError as exc:                     # torch is the ml extra
            raise ModuleNotFoundError(
                f"building an archetype needs the ml extra: "
                f"pip install 'rexgraph-agent[ml]' ({exc})") from exc
        return getattr(_torch_models, fn_name)(cfg, bundle)

    build.__name__ = fn_name
    build.__qualname__ = fn_name
    return build


register_archetype(
    "mlp", use_case="Tabular / vector data - classification or regression.",
    data_kind="vector",
    defaults={"feat_dim": 16, "n_classes": 4, "d_hid": 128, "n_layers": 2, "task": "classification"},
    build=_model_build("_build_mlp"), synth=lambda cfg, seed: _data.synth_vectors(
        feat_dim=cfg["feat_dim"], n_classes=cfg["n_classes"], seed=seed))


# CNN, for images

register_archetype(
    "cnn", use_case="Image classification. norm=False drops the batch norm that fixes conditioning, "
                    "which is the ill-conditioned setting an optimizer A/B needs.",
    data_kind="image",
    defaults={"in_channels": 3, "n_classes": 4, "depth": 2, "width": 32, "norm": True},
    build=_model_build("_build_cnn"), synth=lambda cfg, seed: _data.synth_images(
        c=cfg["in_channels"], n_classes=cfg["n_classes"], seed=seed))


# LM, for sequences (relational or standard attention)

register_archetype(
    "lm", use_case="Sequence / language modeling (next-token). attention: 'relational' (propagator) "
                   "or 'standard'.",
    data_kind="sequence",
    defaults={"vocab": 24, "d": 64, "n_head": 4, "n_layer": 2, "seq_len": 24, "attention": "relational"},
    build=_model_build("_build_lm"), synth=lambda cfg, seed: _data.synth_sequences(
        vocab=cfg["vocab"], seq_len=cfg["seq_len"], seed=seed))


# HGNN - relational-complex hypergraph net (advection + diffusion)

register_archetype(
    "hgnn", use_case="Node classification on hypergraphs / higher-order relational data. "
                     "Fiber-bundle advection+diffusion; uses the complex's signed orientation.",
    data_kind="hypergraph",
    defaults={"feat_dim": 16, "n_classes": 4, "d_hid": 32, "n_layers": 2, "flow": True,
              "oriented": False},
    build=_model_build("_build_hgnn"), synth=lambda cfg, seed: _data.synth_hypergraph(
        feat_dim=cfg["feat_dim"], n_classes=cfg["n_classes"], oriented=cfg.get("oriented", False),
        seed=seed))
