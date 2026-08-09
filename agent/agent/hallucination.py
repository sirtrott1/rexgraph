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
    flagged_claims: list[FlaggedClaim] = field(default_factory=list)
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
