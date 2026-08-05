"""
Multi-backend OCR fusion: structural comparison via rexgraph.

Runs the same document through multiple OCR backends, builds a
relational complex from each output, and uses Hodge decomposition,
void analysis, and structural character to measure where the
backends agree and disagree.

This is novel: nobody else has the mathematics to structurally
compare OCR outputs.  Traditional comparison is character-level
diff.  This compares the *relational topology* of the extracted
content: gradient vs curl vs harmonic structure, void patterns,
coherence distributions.

Usage:

    from agent.integrations.ocr_fusion import OCRFusion

    fusion = OCRFusion(backends=["paddleocr", "unlimited-ocr"])
    report = fusion.compare("document.pdf")

    # Which backend produced more structural coherence?
    print(report.kappa_comparison)

    # Where do the backends structurally disagree?
    print(report.hodge_divergence)

    # Merge the best structural features
    merged_rex = fusion.merge("document.pdf")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    """Analysis results from a single OCR backend."""

    backend_name: str
    text: str = ""
    n_chars: int = 0
    n_words: int = 0

    # RexGraph metrics
    nV: int = 0
    nE: int = 0
    nF: int = 0
    chain_valid: bool = False
    betti: tuple = ()

    # Hodge decomposition
    pct_gradient: float = 0.0
    pct_curl: float = 0.0
    pct_harmonic: float = 0.0

    # Relational
    kappa_mean: float = 0.0
    kappa_std: float = 0.0
    dominant_channel: str = ""
    channel_distribution: dict[str, int] = field(default_factory=dict)

    # Void analysis
    n_voids: int = 0
    void_fraction: float = 0.0

    # RCFE
    strain_norm: float = 0.0
    bianchi_ok: bool = True

    # Raw objects (for downstream use)
    rex: Any = None
    analysis: dict[str, Any] = field(default_factory=dict)
    edge_construction: Any = None

    elapsed: float = 0.0


@dataclass
class FusionReport:
    """Structural comparison of multiple OCR backends on one document."""

    source: str
    backends: list[BackendResult] = field(default_factory=list)

    # Comparison metrics
    @property
    def n_backends(self) -> int:
        return len(self.backends)

    @property
    def hodge_divergence(self) -> dict[str, float]:
        """Measure how much backends disagree on Hodge structure.

        Returns the standard deviation of each Hodge component
        across backends.  High divergence = backends see
        fundamentally different document structure.
        """
        if len(self.backends) < 2:
            return {}
        grads = [b.pct_gradient for b in self.backends]
        curls = [b.pct_curl for b in self.backends]
        harms = [b.pct_harmonic for b in self.backends]
        return {
            "gradient_std": round(float(np.std(grads)), 4),
            "curl_std": round(float(np.std(curls)), 4),
            "harmonic_std": round(float(np.std(harms)), 4),
            "total_divergence": round(
                float(np.std(grads) + np.std(curls) + np.std(harms)), 4,
            ),
        }

    @property
    def kappa_comparison(self) -> dict[str, Any]:
        """Compare coherence across backends."""
        if not self.backends:
            return {}
        return {
            b.backend_name: {
                "kappa_mean": b.kappa_mean,
                "kappa_std": b.kappa_std,
            }
            for b in self.backends
        }

    @property
    def void_comparison(self) -> dict[str, Any]:
        """Compare void structure across backends.

        Higher void fraction means more expected relationships
        are missing, which can indicate OCR quality issues or
        genuine structural gaps in the document.
        """
        if not self.backends:
            return {}
        return {
            b.backend_name: {
                "n_voids": b.n_voids,
                "void_fraction": b.void_fraction,
            }
            for b in self.backends
        }

    @property
    def complexity_comparison(self) -> dict[str, Any]:
        """Compare topological complexity across backends."""
        if not self.backends:
            return {}
        return {
            b.backend_name: {
                "nV": b.nV, "nE": b.nE, "nF": b.nF,
                "betti": b.betti,
            }
            for b in self.backends
        }

    @property
    def best_coherence(self) -> str | None:
        """Which backend produced the highest mean coherence?"""
        if not self.backends:
            return None
        return max(self.backends, key=lambda b: b.kappa_mean).backend_name

    @property
    def best_structure(self) -> str | None:
        """Which backend extracted the richest relational structure?

        Measured by number of faces (higher-order relationships).
        """
        if not self.backends:
            return None
        return max(self.backends, key=lambda b: b.nF).backend_name

    @property
    def lowest_void_fraction(self) -> str | None:
        """Which backend had the fewest structural gaps?"""
        if not self.backends:
            return None
        return min(self.backends, key=lambda b: b.void_fraction).backend_name

    def best_result(self, criterion: str = "coherence"):
        """Return the highest-confidence BackendResult.

        criterion: 'coherence' (kappa), 'structure' (nF), or 'chars'.
        Only considers backends that actually produced text.
        """
        usable = [b for b in self.backends if (b.text or "").strip()]
        if not usable:
            return None
        if criterion == "structure":
            key = lambda b: (b.nF, b.kappa_mean)
        elif criterion == "chars":
            key = lambda b: b.n_chars
        else:
            key = lambda b: (b.kappa_mean, b.nF)
        return max(usable, key=key)

    def best_text(self, criterion: str = "coherence") -> str:
        """Return the text from the highest-confidence backend."""
        br = self.best_result(criterion)
        return br.text if br is not None else ""

    def summary(self) -> str:
        """Human-readable comparison summary."""
        lines = [f"OCR Fusion Report: {self.source}"]
        lines.append(f"Backends compared: {self.n_backends}")
        lines.append("")

        for b in self.backends:
            lines.append(f"  {b.backend_name}:")
            lines.append(f"    Text: {b.n_chars} chars, {b.n_words} words")
            lines.append(f"    Rex: {b.nV}V {b.nE}E {b.nF}F")
            lines.append(
                f"    Hodge: grad={b.pct_gradient:.1%} "
                f"curl={b.pct_curl:.1%} harm={b.pct_harmonic:.1%}"
            )
            lines.append(f"    Kappa: {b.kappa_mean:.4f} ± {b.kappa_std:.4f}")
            lines.append(f"    Voids: {b.n_voids} ({b.void_fraction:.1%})")
            lines.append("")

        div = self.hodge_divergence
        if div:
            lines.append(
                f"Hodge divergence: {div.get('total_divergence', 0):.4f}"
            )
        if self.best_coherence:
            lines.append(f"Best coherence: {self.best_coherence}")
        if self.lowest_void_fraction:
            lines.append(f"Fewest gaps: {self.lowest_void_fraction}")

        return "\n".join(lines)


class OCRFusion:
    """Multi-backend OCR comparison and fusion.

    Parameters
    ----------
    backends : list of str or client objects
        Backend names (``'paddleocr'``, ``'unlimited-ocr'``,
        ``'deepseek-ocr-2'``, ``'mistral'``, ``'got-ocr'``)
        or pre-configured client objects.
    strategy : str
        Adapter strategy for relational complex construction:
        ``'text'`` or ``'layout'``.
    """

    def __init__(
        self,
        backends: list | None = None,
        strategy: str = "text",
        **adapter_kwargs,
    ):
        self.backend_specs = backends or ["paddleocr"]
        self.strategy = strategy
        self.adapter_kwargs = adapter_kwargs
        self._clients = None

    def _resolve_clients(self) -> list:
        """Resolve backend names to client objects."""
        if self._clients is not None:
            return self._clients

        from agent.integrations.unlimited_ocr import (
            GOTOCRClient,
            MistralOCRClient,
            OfflineOCRClient,
            PaddleOCRClient,
            UnlimitedOCRClient,
        )

        clients = []
        for spec in self.backend_specs:
            if hasattr(spec, "ocr_image"):
                # Already a client object
                clients.append(spec)
            elif spec == "paddleocr":
                clients.append(PaddleOCRClient())
            elif spec == "unlimited-ocr":
                clients.append(UnlimitedOCRClient())
            elif spec == "deepseek-ocr-2":
                clients.append(UnlimitedOCRClient.deepseek_ocr2())
            elif spec == "deepseek-ocr":
                clients.append(UnlimitedOCRClient.deepseek_ocr())
            elif spec == "mistral":
                clients.append(MistralOCRClient())
            elif spec == "got-ocr":
                clients.append(GOTOCRClient())
            elif spec == "offline":
                clients.append(OfflineOCRClient())
            else:
                logger.warning("Unknown backend: %s, skipping", spec)

        self._clients = clients
        return clients

    def compare(
        self,
        source: str,
        depth: str = "standard",
    ) -> FusionReport:
        """Run OCR on a source through all backends and compare.

        Parameters
        ----------
        source : str
            Path to image, PDF, or directory.
        depth : str
            Analysis depth: ``'quick'``, ``'standard'``, ``'full'``.

        Returns
        -------
        FusionReport
        """

        clients = self._resolve_clients()
        report = FusionReport(source=source)

        for client in clients:
            name = getattr(client, "backend_name", type(client).__name__)
            logger.info("Running OCR with %s on %s", name, source)

            try:
                result = self._run_single(
                    client, source, name, depth,
                )
                report.backends.append(result)
            except Exception as e:
                logger.error("Backend %s failed: %s", name, e)
                report.backends.append(BackendResult(
                    backend_name=name,
                ))

        return report

    def _run_single(
        self,
        client,
        source: str,
        name: str,
        depth: str,
    ) -> BackendResult:
        """Run a single backend and analyze the result."""
        import time

        from agent.adapters.ocr import OCRAdapter
        from agent.pipeline import AnalysisPipeline
        from rexgraph.graph import RexGraph

        start = time.time()
        adapter = OCRAdapter(client=client)
        ec = adapter.build(source, strategy=self.strategy, **self.adapter_kwargs)

        if ec.nE == 0:
            return BackendResult(
                backend_name=name, elapsed=time.time() - start,
            )

        # Build RexGraph
        w_mag = ec.weights
        rex = RexGraph(
            sources=ec.sources,
            targets=ec.targets,
            w_E=w_mag if not np.allclose(w_mag, 1.0) else None,
        )
        if ec.n_types > 1:
            from agent.auto import attach_faces
            rex = attach_faces(rex, type_labels=ec.type_labels)

        # Run analysis pipeline
        pipe = AnalysisPipeline(rex)
        analysis = pipe.run(depth=depth)

        # Extract text for word count
        text = ""
        if hasattr(client, "_last_text"):
            text = client._last_text

        # Build result
        hodge = analysis.get("hodge", {})
        rel = analysis.get("relational", {})
        void = analysis.get("void", {})
        rcfe = analysis.get("rcfe", {})

        elapsed = time.time() - start

        return BackendResult(
            backend_name=name,
            text=text,
            n_chars=len(text),
            n_words=len(text.split()),
            nV=rex.nV,
            nE=rex.nE,
            nF=rex.nF,
            chain_valid=rex.chain_valid,
            betti=tuple(rex.betti),
            pct_gradient=hodge.get("pct_gradient", 0),
            pct_curl=hodge.get("pct_curl", 0),
            pct_harmonic=hodge.get("pct_harmonic", 0),
            kappa_mean=rel.get("kappa_mean", 0),
            kappa_std=rel.get("kappa_std", 0),
            dominant_channel=max(
                rel.get("dominant_channel_distribution", {"T": 0}),
                key=rel.get("dominant_channel_distribution", {"T": 0}).get,
            ) if rel.get("dominant_channel_distribution") else "",
            channel_distribution=rel.get("dominant_channel_distribution", {}),
            n_voids=void.get("n_voids", 0),
            void_fraction=void.get("void_fraction", 0),
            strain_norm=rcfe.get("strain_norm", 0),
            bianchi_ok=rcfe.get("bianchi_ok", True),
            rex=rex,
            analysis=analysis,
            edge_construction=ec,
            elapsed=elapsed,
        )
