"""
agent.hallucination: structural hallucination detection.

Builds a relational complex from a model response and compares
its topology against the source document. Structural divergence
indicates the model invented relationships, contradicted the
document's structure, or introduced entities that create voids
not present in the source.

Uses compiled kernels:
    _cross_complex.align_by_labels
    _cross_complex.cross_complex_kappa
    _cross_complex.cross_complex_void_fraction
    _cross_complex.cross_complex_bridge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FlaggedClaim:
    """A specific structural inconsistency in the response."""
    source_label: str
    target_label: str
    reason: str          # void_creation, channel_mismatch, kappa_drop
    severity: float      # 0-1, higher = more likely hallucination
    detail: str = ""
    criticality: float = 1.0   # the relation's effective resistance (1 = bridge, low = redundant)


@dataclass
class HallucinationReport:
    """Structural comparison between a model response and source."""
    n_response_entities: int = 0
    n_source_entities: int = 0
    n_shared_entities: int = 0
    kappa_correlation: float = 0.0
    chi_divergence: float = 0.0
    hodge_divergence: tuple = (0.0, 0.0, 0.0)
    new_voids: int = 0
    flagged_claims: List[FlaggedClaim] = field(default_factory=list)
    overall_score: float = 0.0    # 0 = trustworthy, 1 = hallucinated

    @property
    def n_flags(self):
        return len(self.flagged_claims)

    def summary(self):
        if self.overall_score < 0.2:
            verdict = "structurally consistent"
        elif self.overall_score < 0.5:
            verdict = "minor divergences"
        else:
            verdict = "significant structural divergence"
        return (
            f"{verdict} (score={self.overall_score:.2f}, "
            f"shared={self.n_shared_entities}, "
            f"kappa_corr={self.kappa_correlation:.3f}, "
            f"flags={self.n_flags})"
        )


def detect_hallucinations(
    source_rex,
    response_text: str,
    source_labels: List[str],
    threshold: float = 0.3,
) -> HallucinationReport:
    """Compare a model response against the source document.

    Builds a rex from the response text, aligns entities,
    and checks for structural inconsistencies.
    """
    from agent.auto import auto_rex
    from agent.adapters.text import TextAdapter

    report = HallucinationReport()

    # Build response rex
    ta = TextAdapter()
    resp_ec = ta.build(response_text, min_count=1, max_vocab=300)
    if resp_ec.nE == 0:
        return report

    try:
        from rexgraph.graph import RexGraph
        resp_rex = RexGraph(sources=resp_ec.sources, targets=resp_ec.targets)
        if resp_ec.n_types > 1:
            resp_rex = resp_rex.typed_face_selection(resp_ec.type_labels)
    except Exception as e:
        logger.warning("Failed to build response rex: %s", e)
        return report

    resp_labels = resp_ec.vertex_labels
    report.n_response_entities = len(resp_labels)
    report.n_source_entities = len(source_labels)

    # Align entities
    try:
        from rexgraph.core._cross_complex import (
            align_by_labels,
            cross_complex_kappa,
            cross_complex_void_fraction,
            cross_complex_bridge,
        )

        shared, idx_a, idx_b = align_by_labels(source_labels, resp_labels)
        report.n_shared_entities = len(shared)

        if len(shared) < 2:
            report.overall_score = 0.5  # can't compare, assume moderate risk
            return report

        # Kappa correlation
        source_kappa = source_rex.coherence
        resp_kappa = resp_rex.coherence
        if source_kappa is not None and resp_kappa is not None:
            result = cross_complex_kappa(source_kappa, resp_kappa, idx_a, idx_b)
            report.kappa_correlation = result.get("correlation", 0.0)

        # Void comparison
        source_void = getattr(source_rex, "_void_data", None)
        resp_void = getattr(resp_rex, "_void_data", None)
        if source_void and resp_void:
            vf = cross_complex_void_fraction(
                source_void.get("n_voids", 0), source_void.get("n_potential", 1),
                resp_void.get("n_voids", 0), resp_void.get("n_potential", 1),
            )
            # Voids in response that don't exist in source
            report.new_voids = max(0,
                resp_void.get("n_voids", 0) - source_void.get("n_voids", 0))

        # Structural character comparison
        source_chi = None
        resp_chi = None
        try:
            source_chi = source_rex.structural_character
            resp_chi = resp_rex.structural_character
        except Exception:
            pass

        if source_chi is not None and resp_chi is not None:
            s_mean = source_chi.mean(axis=0)
            r_mean = resp_chi.mean(axis=0)
            n = min(len(s_mean), len(r_mean))
            report.chi_divergence = float(np.linalg.norm(s_mean[:n] - r_mean[:n]))

        # Hodge divergence
        try:
            s_flow = np.ones(source_rex.nE, dtype=np.float64)
            r_flow = np.ones(resp_rex.nE, dtype=np.float64)
            s_hodge = source_rex.hodge_full(s_flow)
            r_hodge = resp_rex.hodge_full(r_flow)
            if s_hodge and r_hodge:
                dg = abs(s_hodge.get("pct_grad", 0) - r_hodge.get("pct_grad", 0))
                dc = abs(s_hodge.get("pct_curl", 0) - r_hodge.get("pct_curl", 0))
                dh = abs(s_hodge.get("pct_harm", 0) - r_hodge.get("pct_harm", 0))
                report.hodge_divergence = (dg, dc, dh)
        except Exception:
            pass

        # Flag specific claims
        report.flagged_claims = _flag_claims(
            source_rex, resp_rex, source_labels, resp_labels,
            source_chi, resp_chi, shared, idx_a, idx_b, threshold,
        )

        # Overall score
        scores = []
        if report.kappa_correlation != 0:
            scores.append(max(0, 1.0 - report.kappa_correlation))
        if report.chi_divergence > 0:
            scores.append(min(1.0, report.chi_divergence))
        if report.new_voids > 0:
            scores.append(min(1.0, report.new_voids / 10.0))
        if report.hodge_divergence != (0, 0, 0):
            scores.append(sum(report.hodge_divergence) / 3.0)
        # resistance-weighted flags: a mismatch on a bridge relation counts more
        if report.flagged_claims:
            scores.append(float(np.mean([f.severity for f in report.flagged_claims])))

        report.overall_score = float(np.mean(scores)) if scores else 0.0

    except ImportError:
        logger.warning("cross_complex kernels not available")
    except Exception as e:
        logger.warning("Hallucination detection failed: %s", e)

    return report


def _flag_claims(
    source_rex, resp_rex, source_labels, resp_labels,
    source_chi, resp_chi, shared, idx_a, idx_b, threshold,
):
    """Identify specific edges in the response that diverge from source."""
    flags = []
    chan_names = ["T", "G", "F", "C"]

    if source_chi is None or resp_chi is None:
        return flags

    # Check each shared edge for channel mismatch
    for k in range(len(shared)):
        ea, eb = idx_a[k], idx_b[k]
        if ea >= source_chi.shape[0] or eb >= resp_chi.shape[0]:
            continue

        s_chi = source_chi[ea]
        r_chi = resp_chi[eb]
        n = min(len(s_chi), len(r_chi), 4)

        # Dominant channel mismatch
        s_dom = int(np.argmax(s_chi[:n]))
        r_dom = int(np.argmax(r_chi[:n]))
        if s_dom != r_dom:
            div = float(np.linalg.norm(s_chi[:n] - r_chi[:n]))
            if div > threshold:
                flags.append(FlaggedClaim(
                    source_label=source_labels[ea] if ea < len(source_labels) else f"e{ea}",
                    target_label=resp_labels[eb] if eb < len(resp_labels) else f"e{eb}",
                    reason="channel_mismatch",
                    severity=min(1.0, div),
                    detail=f"source={chan_names[s_dom]}, response={chan_names[r_dom]}",
                ))
                flags[-1]._src_edge = int(ea)   # stash for resistance weighting below

    # Weight each flag by how LOAD-BEARING the diverging relation is in the SOURCE:
    # a channel mismatch on a bridge (effective_resistance->1) is far more serious than
    # on a redundant relation with alternate paths. Batch one L0 solve, no per-edge loop.
    if flags:
        try:
            src_edges = np.array([getattr(f, "_src_edge", 0) for f in flags], dtype=int)
            r_eff = source_rex._effective_resistance_batch(src_edges)
            for f, r in zip(flags, r_eff):
                f.criticality = round(float(r), 4)
                f.severity = round(min(1.0, f.severity * float(r)), 4)  # resistance-weighted
        except Exception:
            pass
    return flags


def iterative_rechunk(chunks, report, source_rex, edge_spans, text, max_iters=2):
    """Adjust chunks based on hallucination feedback.

    Low kappa chunks get expanded (need more context).
    Void-creating chunks get merged (missing relationships).
    Channel-mismatch chunks get split (confused structural types).
    """
    from agent.chunking import hodge_chunk, Chunk

    if report.overall_score < 0.2:
        return chunks  # no significant issues

    adjusted = list(chunks)

    for iteration in range(max_iters):
        changes = 0

        for i, chunk in enumerate(adjusted):
            if chunk.kappa < 0.3 and i + 1 < len(adjusted):
                # Low kappa: expand by merging with neighbor
                nxt = adjusted[i + 1]
                adjusted[i] = Chunk(
                    idx=i,
                    text=text[chunk.char_start:nxt.char_end],
                    char_start=chunk.char_start,
                    char_end=nxt.char_end,
                    sentence_start=chunk.sentence_start,
                    sentence_end=nxt.sentence_end,
                    edge_indices=chunk.edge_indices + nxt.edge_indices,
                    n_edges=chunk.n_edges + nxt.n_edges,
                )
                adjusted.pop(i + 1)
                changes += 1
                break

        if changes == 0:
            break

    # Re-index
    for i, c in enumerate(adjusted):
        c.idx = i

    return adjusted


def detect_hallucinations_exchange(
    source_text: str,
    response_text: str,
    threshold: float = 0.3,
) -> HallucinationReport:
    """Detect hallucinations by building the exchange as a single relational complex.

    Instead of comparing two separate complexes, builds one complex
    containing both input and output with exchange edges bridging
    shared entities. The exchange edge count, exchange kappa, and
    Hodge profile of the combined complex measure structural consistency.
    """
    from agent.exchange import build_exchange_complex

    rex, ex_result, ec = build_exchange_complex(source_text, response_text)

    report = HallucinationReport()
    if rex is None:
        report.overall_score = 0.5
        return report

    report.n_source_entities = ex_result.n_input_vertices
    report.n_response_entities = ex_result.n_output_vertices
    report.n_shared_entities = ex_result.n_shared
    report.kappa_correlation = ex_result.exchange_kappa if not np.isnan(ex_result.exchange_kappa) else 0.0
    report.hodge_divergence = (
        ex_result.hodge_gradient,
        ex_result.hodge_curl,
        ex_result.hodge_harmonic,
    )

    # Score based on exchange edge density and kappa
    if ex_result.n_shared > 0:
        # Exchange edge density: how many shared entities have
        # consistent relational structure in both input and output
        max_possible = ex_result.n_shared * (ex_result.n_shared - 1) // 2
        edge_density = ex_result.n_exchange_edges / max(max_possible, 1)

        scores = []
        scores.append(1.0 - min(1.0, edge_density * 2))  # low density = high score
        if not np.isnan(ex_result.exchange_kappa):
            scores.append(max(0, 1.0 - ex_result.exchange_kappa))
        # Low shared entity ratio = suspicious
        entity_ratio = ex_result.n_shared / max(ex_result.n_output_vertices, 1)
        scores.append(max(0, 1.0 - entity_ratio))

        # Typed void analysis - characterize WHAT kind of structure is missing
        if rex is not None:
            try:
                vc = rex.void_complex
                if vc and vc.get('n_voids', 0) > 0 and vc.get('Bvoid') is not None:
                    # Matrix-free on the universal scale-free path (dense sb['RL']/hats
                    # are None): per-void character via LSQR pinv quadratic forms.
                    if rex._use_sparse_character:
                        from rexgraph.sparse_character import void_character_sparse
                        void_chi = void_character_sparse(rex, vc['Bvoid'])
                    else:
                        from rexgraph.core._void import void_character_all
                        sb = rex.spectral_bundle
                        void_chi = void_character_all(
                            vc['Bvoid'], sb['RL'], sb['hats'], sb['nhats'],
                            vc['n_voids'], rex.nE,
                        )
                    if isinstance(void_chi, np.ndarray) and void_chi.size > 0:
                        chan_names = ["T", "G", "F", "C"]
                        void_mean = void_chi.mean(axis=0) if void_chi.ndim > 1 else void_chi
                        n_chan = min(4, len(void_mean))
                        dom = chan_names[int(np.argmax(void_mean[:n_chan]))]
                        report.flagged_claims.append(FlaggedClaim(
                            source_label="void_analysis", target_label=dom,
                            reason="void_type",
                            severity=float(vc['n_voids']) / max(vc.get('n_potential', 1), 1),
                            detail="%d %s-dominant voids in exchange" % (vc['n_voids'], dom),
                        ))
            except Exception:
                pass

        report.overall_score = float(np.mean(scores))
    else:
        report.overall_score = 0.8  # no shared entities = very suspicious

    return report
