# rexgraph.flow: relational-native models (the complex IS the model)

This directory is the one to read before building any model on relational data with
RexGraph. It holds the relational-native learners: models whose parameters live ON the
complex as cochains, trained by preconditioning the gradient with the complex's own
Green's function. They are not feature extractors feeding a downstream network. The
structure is the model.

## The one idea that has to land first

RexGraph computes exactly, from structure, the quantities that mainstream ML burns
billions of parameters and gradient descent trying to approximate: per-edge character
`chi`, coherence `kappa`, the Green's character `phi`, the four typed channels. These are
proper tensor fields over the complex, not flat weight arrays.

The wrong instinct, and the one every prior model has taken, is to pull those fields out,
flatten them with `np.mean` or `np.quantile` into a handful of scalars, and feed them to
LightGBM or an MLP. That throws away the exact structure and then compares the flattened
version against a flat baseline, which of course ties. Do not do this.

The right pattern is the opposite: keep the structure, and let a model that lives on the
complex carry the signal. That is what the models here are.

## The models

- `cochain.py` : `CoParticipationCochain`. A bare cochain `Z[nE, n_classes]` over the
  edges, no features and no embeddings. It exposes `greens_groups()`, so
  `make_optimizer("auto")` routes it to `GreensCochain` (Green's-preconditioned Adam)
  automatically. The optimizer propagates class through the co-participation structure to
  edges that carry no gradient of their own.
- `attention.py` : `CoParticipationAttention`. Relational attention as a settle over the
  co-participation adjacency, fit self-supervised. A handful of parameters, not a network.
- `gate.py` / `navigator.py` / `online.py` : the flow loop. `MalaughGate` wakes only when
  a structural-entropy scalar MOVES; `FieldNavigator.step` runs the matrix-free Hodge flow
  on the disturbed region; `GreensCochainField` is the online predict-then-observe field
  (one Green's solve plus one relational correction per event, no epochs, no learning
  rate). See the temporal system for how this closes a live-change loop.

## The correct construction, end to end

The canonical shape for relational or scientific data: entities are VERTICES, the
measurements or relations you want to predict are EDGES (edge-primary), and an entity that
participates in K relations is a vertex of arity K, so those K edges become mutual
co-participants. That co-participation structure is what lets the cochain reach an edge
the loss never touches.

Note what this construction does and does not give you. The edges it builds are 2-ary; the
arity lives on the VERTEX, and the operator the cochain trains through is the resulting
co-participation adjacency. For a signed boundary COLUMN of arity K (a branching hyperedge
rather than a high-arity vertex) build the complex with `RexGraph.from_hypergraph`.

```python
import numpy as np
from rexgraph.graph import RexGraph
from rexgraph.flow.cochain import CoParticipationCochain

# entities -> vertex ids; each measurement is one edge between two entities.
# an entity referenced by K measurements is automatically an arity-K branching vertex,
# so its K edges are mutual co-participants. Nothing extra to declare.
src = np.asarray(edge_left_entity,  np.int32)   # one endpoint per measurement
tgt = np.asarray(edge_right_entity, np.int32)   # the other endpoint
rex = RexGraph(sources=src, targets=tgt)         # each measurement IS an edge

labels = np.asarray(class_per_edge, np.int64)    # the thing to predict, per edge
obs    = np.asarray(observed_mask,  bool)         # True where the label is known

# the model IS a cochain on the complex. no features, no embeddings.
model = CoParticipationCochain(rex, n_classes)
model.fit(labels, obs, epochs=300, lr=0.3)        # routes through make_optimizer("auto")
pred = model.predict()                            # class for EVERY edge, incl. masked ones
acc  = float((pred[~obs] == labels[~obs]).mean())
```

The masked edges receive no gradient from the loss. A structure-blind optimizer (plain
Adam on the same cochain) has no way to reach a parameter the loss does not touch, so it
leaves every masked row at its initial value and emits an arbitrary constant prediction.
Under the shipped zeros init that constant is whatever class index 0 happens to be (argmax
of an all-zero row ties to 0), measured anywhere from 0.13 to 0.46 on a 4-class complex,
so it is sometimes well BELOW the majority-class baseline; with the tie broken by a tiny
random init it is chance. `GreensCochain` propagates class along the co-participation
structure and classifies them.

Measured, zero features, 3 classes, 3 mask seeds, maximum vertex arity 424: majority-class
baseline 0.332 +/- 0.011 -> 0.650 +/- 0.015 held-out accuracy. Both the Adam arm and the
GreensCochain arm fit the OBSERVED edges to 1.000, so the difference is entirely in what
reaches a masked row. On a synthetic 4-class complex at 70/30 masking the same ablation
reads 0.251 +/- 0.006 (plain Adam, random init) -> 0.896 +/- 0.004, against a majority
baseline of 0.423.

That jump is the co-participation STRUCTURE, not the optimizer as such. A clamped label
propagation over the same operator scores 0.658 +/- 0.007 on the first complex (a tie, and
121x cheaper) and 0.911 +/- 0.005 on the second (ahead of the optimizer). What the
optimizer buys is that it composes with any loss and any model exposing `greens_groups()`,
not that it is the strongest structural predictor available. And where real per-entity
features exist, a gradient-boosted model on them reached 0.755 +/- 0.008 on the first
complex: this path is what you get with NO features, not a ceiling on what the data
supports.

The relational-attention settle in `attention.py` reaches R^2 in the 0.92 to 0.95 range on
the same construction. That figure was not re-measured in the v1.0.5 benchmark re-run.

## Do not

- Do not flatten `chi` / `phi` / the channels into scalars for a downstream learner. Feed
  the structure to a model on the complex (the cochain), not averaged features to a flat
  model.
- Build the optimizer with `make_optimizer("auto", model, params)`. It returns
  `GreensCochain` for a cochain model (one exposing `greens_groups()`) and plain Adam for a
  feature-space model. Do not pick an optimizer by name; the router is the one that gets it
  right, and the optimizer only helps when the model carries relational structure, which
  `cochain.py` states in its docstring.
- Do not reach for a dense kernel in `rexgraph/core/*.pyx` because it looks like familiar
  linear algebra. Those files carry both the current sparse/matrix-free path and older
  dense forms kept as exact reference oracles. Use the `RexGraph` object model and its
  sparse properties, which are the live path.
- Do not hand-roll retrieval, similarity, or basis solves. The library has demand-driven,
  O(seed) versions: `coherence_response(seed)`, `agentic_reading`, `character_response(seed)`,
  `local_context(seed)`, `phi_similarity_score`, `chi_cosine`, `similarity_complex`.

## Why the cochain instead of flatten-and-boost

A gradient learner over flat features has to discover, from data and with many parameters,
an approximation of the co-participation geometry that the complex already holds exactly.
The cochain skips that: the geometry is the operator, the label field is the only
parameter, and one Green's solve carries a known label to an unknown one along the real
structure. Fewer parameters, no feature engineering, and it classifies edges a featureless
flat model cannot address at all.

The measured boundary on that: where real per-entity features exist, a booster over them
can still win outright, and even a booster given nothing but the two endpoint vertex ids
as categoricals scored 0.709 +/- 0.007 against the cochain's 0.650. Entity identity is
structure too. The cochain's claim is what it does with zero features and what it does to
an edge the loss never touches, not that it dominates a feature-rich model.
