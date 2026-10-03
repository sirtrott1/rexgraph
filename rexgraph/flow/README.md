# rexgraph.flow: relational native models

This directory holds the relational native learners: models whose parameters live on
the complex as cochains, trained by preconditioning the gradient with the complex's own
Green's function.

Install the Core neural components from the repository root:

```bash
pip install ".[nn]"
```

The cochain's sparse operator follows its parameter dtype and device through normal
PyTorch module placement. Safetensors reload preserves the stored cochain dtype.
The online field caches operators against Core's current native boundary, so public
graph edits invalidate a previously cached flow operator. Retained model records bind
the source, coordinates and implementation; Agent's assembled feature model checkpoint
directories have a separate warm start contract documented in `agent/agent/models/README.md`.

## The models

- `cochain.py` : `CoParticipationCochain`. A bare cochain `Z[nE, n_classes]` over the
  edges, no features and no embeddings. It exposes `greens_groups()`, so
  `make_optimizer("auto")` routes it to `GreensCochain` (Green's-preconditioned Adam)
  automatically. The optimizer propagates class through the co participation structure to
  edges that carry no gradient of their own.
- `attention.py` : `CoParticipationAttention`. Relational attention as a settle over the
  co participation adjacency, fit self supervised. It learns a parameterized settle over that adjacency.
- `gate.py` / `navigator.py` / `online.py` : the flow loop. `MalaughGate` wakes only when
  a structural entropy scalar MOVES; `FieldNavigator.step` runs the matrix free Hodge flow
  on the disturbed region; `GreensCochainField` is the online predict then observe field
  (one Green's solve plus one relational correction per event, no epochs, no learning
  rate). See the temporal system for how this closes a live change loop.

## The correct construction, end to end

The canonical shape for relational or scientific data: entities are VERTICES, the
measurements or relations you want to predict are EDGES (edge primary), and an entity that
participates in K relations is a vertex of arity K, so those K edges become mutual
co participants. That co participation structure is what lets the cochain reach an edge
the loss never touches.

Note what this construction does and does not give you. The edges it builds are 2 ary; the
arity lives on the VERTEX, and the operator the cochain trains through is the resulting
co participation adjacency. For a signed boundary COLUMN of arity K (a branching hyperedge
rather than a high arity vertex) build the complex with `RexGraph.from_hypergraph`.

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

The masked edges receive no gradient from the loss. `GreensCochain` propagates
updates through the co participation operator to rows without observed labels.
Performance depends on the structure and the available observations.

## Using the models

Build the optimizer with `make_optimizer("auto", model, params)`. It selects
`GreensCochain` for models exposing `greens_groups()` and Adam for feature models.
Use `RexGraph` sparse properties for structure and the dense kernels for explicit
reference comparisons. Core also supplies seeded response and similarity methods:
`coherence_response(seed)`, `character_response(seed)`, `local_context(seed)`,
`phi_similarity_score`, `chi_cosine` and `similarity_complex`.
