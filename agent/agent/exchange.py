"""
agent.exchange: input/output exchange as a relational complex.

Each model exchange (input context + query + output) is a single
relational complex. The exchange edges bridge shared entities
between input and output. Structural consistency of the exchange
is measured through the chain condition on the combined complex.

The exchange complex has three edge types:
    0 = input  (edges from input text)
    1 = output (edges from output text)
    2 = exchange (edges bridging shared entities)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExchangeResult:
    """Analysis of a single input/output exchange."""
    n_input_vertices: int = 0
    n_output_vertices: int = 0
    n_shared: int = 0
    n_exchange_edges: int = 0
    betti: tuple = ()
    kappa_mean: float = 0.0
    hodge_gradient: float = 0.0
    hodge_curl: float = 0.0
    hodge_harmonic: float = 0.0
    chi_mean: list = field(default_factory=list)
    chain_residual: float = 0.0
    exchange_kappa: float = 0.0  # kappa restricted to exchange edges
    input_type: str = ""
    output_type: str = ""


def build_exchange_complex(
    input_text: str,
    output_text: str,
    min_count: int = 1,
    max_vocab: int = 400,
):
    """Build a single relational complex from an input/output exchange.

    The combined complex contains:
        - Input edges (word co-occurrence in input)
        - Output edges (word co-occurrence in output)
        - Exchange edges (shared entities bridging input to output)

    Returns (RexGraph, ExchangeResult, EdgeConstruction).
    """
    from agent.adapters.text import TextAdapter
    from rexgraph.graph import RexGraph

    ta = TextAdapter()
    in_ec = ta.build(input_text, min_count=min_count, max_vocab=max_vocab)
    out_ec = ta.build(output_text, min_count=min_count, max_vocab=max_vocab)

    if in_ec.nE == 0 or out_ec.nE == 0:
        return None, ExchangeResult(), None

    # Build unified vertex set
    in_labels = in_ec.vertex_labels
    out_labels = out_ec.vertex_labels

    unified_labels = list(in_labels)
    out_to_unified = {}

    for i, label in enumerate(out_labels):
        if label in in_labels:
            out_to_unified[i] = in_labels.index(label)
        else:
            out_to_unified[i] = len(unified_labels)
            unified_labels.append(label)

    len(unified_labels)

    # Remap output edges to unified vertex indices
    out_sources_remapped = np.array(
        [out_to_unified[int(s)] for s in out_ec.sources], dtype=np.int32,
    )
    out_targets_remapped = np.array(
        [out_to_unified[int(t)] for t in out_ec.targets], dtype=np.int32,
    )

    # Find shared entities (vertices present in both)
    in_set = set(range(len(in_labels)))
    out_set = set(out_to_unified.values())
    shared = sorted(in_set & out_set)

    # Build exchange edges: connect each shared vertex to itself
    # across the input/output boundary via its neighbors
    exchange_src = []
    exchange_tgt = []
    exchange_wt = []
    exchange_signs = []

    # For each pair of shared vertices, create an exchange edge if the (undirected)
    # pair co-occurs as an edge in BOTH input and output. Build the two edge-pair
    # sets once (O(nE)) so each pair test is an O(1) membership - total O(n_shared²)
    # instead of the old O(n_shared² · nE) triple loop. Self-loops (endpoints equal)
    # collapse to a singleton frozenset and never match a 2-vertex pair, as intended.
    in_pairs = {frozenset((int(in_ec.sources[k]), int(in_ec.targets[k])))
                for k in range(in_ec.nE)}
    out_pairs = {frozenset((int(out_sources_remapped[k]), int(out_targets_remapped[k])))
                 for k in range(out_ec.nE)}
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            vi, vj = shared[i], shared[j]
            pair = frozenset((vi, vj))
            # Exchange edge: entities that appear in BOTH
            if pair in in_pairs and pair in out_pairs:
                exchange_src.append(vi)
                exchange_tgt.append(vj)
                exchange_wt.append(1.0)
                exchange_signs.append(1.0)

    # Combine all edges
    all_sources = np.concatenate([
        in_ec.sources,
        out_sources_remapped,
        np.array(exchange_src, dtype=np.int32) if exchange_src else np.array([], dtype=np.int32),
    ])
    all_targets = np.concatenate([
        in_ec.targets,
        out_targets_remapped,
        np.array(exchange_tgt, dtype=np.int32) if exchange_tgt else np.array([], dtype=np.int32),
    ])
    all_weights = np.concatenate([
        in_ec.weights,
        out_ec.weights,
        np.array(exchange_wt, dtype=np.float64) if exchange_wt else np.array([], dtype=np.float64),
    ])
    all_signs = np.concatenate([
        in_ec.signs,
        out_ec.signs,
        np.array(exchange_signs, dtype=np.float64) if exchange_signs else np.array([], dtype=np.float64),
    ])
    all_types = np.concatenate([
        np.zeros(in_ec.nE, dtype=np.int32),          # type 0 = input
        np.ones(out_ec.nE, dtype=np.int32),           # type 1 = output
        np.full(len(exchange_src), 2, dtype=np.int32), # type 2 = exchange
    ])

    nE = len(all_sources)
    if nE == 0:
        return None, ExchangeResult(), None

    # Build the combined EdgeConstruction
    from agent.adapters import EdgeConstruction
    combined_ec = EdgeConstruction(
        sources=all_sources,
        targets=all_targets,
        weights=all_weights,
        signs=all_signs,
        type_labels=all_types,
        vertex_labels=unified_labels,
        n_types=3,
        type_names=["input", "output", "exchange"],
        source_text=input_text + "\n\n---\n\n" + output_text,
    )

    # Build RexGraph
    rex = RexGraph(sources=all_sources, targets=all_targets)
    from agent.auto import attach_faces
    rex = attach_faces(rex, type_labels=all_types)

    # Compute exchange result
    result = ExchangeResult(
        n_input_vertices=len(in_labels),
        n_output_vertices=len(out_labels),
        n_shared=len(shared),
        n_exchange_edges=len(exchange_src),
    )

    try:
        result.betti = rex.betti
        kappa = rex.coherence
        if kappa is not None and len(kappa) > 0:
            km = float(kappa.mean())
            result.kappa_mean = 0.0 if np.isnan(km) else km
        else:
            result.kappa_mean = 0.0

        flow = np.ones(rex.nE, dtype=np.float64)
        h = rex.hodge_full(flow)
        result.hodge_gradient = float(h.get("pct_grad", 0))
        result.hodge_curl = float(h.get("pct_curl", 0))
        result.hodge_harmonic = float(h.get("pct_harm", 0))

        chi = rex.structural_character
        if chi is not None:
            result.chi_mean = chi.mean(axis=0).tolist()
            chan = ["T", "G", "F", "C"]
            n = min(4, len(result.chi_mean))
            result.input_type = chan[int(np.argmax(result.chi_mean[:n]))]

        # Kappa restricted to exchange edges
        kappa = rex.coherence
        n_in = in_ec.nE
        n_out = out_ec.nE
        exchange_indices = list(range(n_in + n_out, nE))
        if exchange_indices and kappa is not None and len(kappa) >= nE:
            ex_kappas = [float(kappa[i]) for i in exchange_indices if i < len(kappa)]
            if ex_kappas:
                result.exchange_kappa = float(np.mean(ex_kappas))
        elif kappa is not None and len(exchange_src) > 0:
            # Fall back to mean kappa if indexing fails
            km = float(kappa.mean())
            result.exchange_kappa = 0.0 if np.isnan(km) else km
    except Exception as e:
        logger.warning("Exchange analysis failed: %s", e)

    return rex, result, combined_ec


def analyze_exchange_sequence(exchanges: list) -> dict:
    """Analyze a sequence of exchanges for drift, consistency, and memory.

    Each exchange is an ExchangeResult. The sequence represents a
    conversation. BIOES tags track which relational edges persist,
    are born, or die across exchanges.
    """
    if len(exchanges) < 2:
        return {"n_exchanges": len(exchanges)}

    # Track kappa drift
    kappas = [e.kappa_mean for e in exchanges]
    kappa_drift = max(kappas) - min(kappas) if kappas else 0

    # Track Hodge drift
    grads = [e.hodge_gradient for e in exchanges]
    grad_drift = max(grads) - min(grads) if grads else 0

    # Track exchange kappa (structural consistency per exchange)
    ex_kappas = [e.exchange_kappa for e in exchanges if e.exchange_kappa > 0]
    ex_kappa_mean = float(np.mean(ex_kappas)) if ex_kappas else 0

    # Track shared entity count over time
    shared_counts = [e.n_shared for e in exchanges]
    shared_trend = shared_counts[-1] - shared_counts[0] if len(shared_counts) >= 2 else 0

    return {
        "n_exchanges": len(exchanges),
        "kappa_drift": kappa_drift,
        "grad_drift": grad_drift,
        "exchange_kappa_mean": ex_kappa_mean,
        "shared_trend": shared_trend,
        "shared_counts": shared_counts,
        "kappas": kappas,
    }
