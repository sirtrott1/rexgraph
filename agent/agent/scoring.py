"""
agent.scoring: one ranking, using the reads the library provides for exactly this.

An earlier version of this module called `interfacing_vector` with `target=None`.
That was a misreading. In `_interfacing` the target is a TARGET EDGE VECTOR and the
channel score is a bilinear form I_X = target^T S_X psi between the source's induced
flow and a target pattern; passing None scores psi against itself, which is a self
energy and interfaces with nothing. It also built the whole bundle per document,
paying O(nV . solve) for an answer needed at a handful of vertices.

RexGraph already answers "what does this query touch in this complex" directly, and
demand-driven:

    coherence_response(seed)  kappa at just the query's vertices, by diffusion.
                              O(|seed| . nhats . diffusion), and identical to
                              coherence[seed] rather than an approximation of it
    agentic_reading(seed)     the decision-ready reading the agent layer is meant to
                              consume: the bounded neighborhood, relations ranked by
                              effective resistance (the bridges), entities whose
                              coherence is a low outlier under a data-adaptive Tukey
                              fence, and context_size (what a correct answer costs)

So relevance here is the query's footprint measured by the DOCUMENT's own coherence
field: sum of kappa over the matched vertices. It grows with how much of the query
the document carries and with how coherent that footprint is inside it, and it needs
no mixing constant, because it is one field summed over one seed.

Lexical overlap remains a candidate prefilter only. It decides what to look at.
"""

from __future__ import annotations

# In the sibling rcdb package for the same reason the coherence measurements are:
# a store scores a candidate set without the application present.
from rcdb.analytics import (  # noqa: E402,F401
    MIN_SHARED,
    interfacing_score,
    shared_indices,
)
