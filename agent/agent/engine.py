"""
Agent decision engine: inspect data, plan analysis, execute.

The engine determines what kind of relational structure the data
carries, selects the construction strategy, runs the analysis,
and produces a domain-specific interpretation.  Every decision
is recorded with its rationale.

No fitted parameters.  No domain-specific thresholds.  The
algebraic structure of the data itself determines the plan.

Usage:

    engine = DecisionEngine()

    # From any data source
    result = engine.run(feature_matrix)
    result = engine.run(triples, contexts=document_contexts)
    result = engine.run(text_corpus)

    # Plan without executing
    plan = engine.plan(data)
    print(plan.decisions)

    # With a domain-specific signal
    result = engine.run(feature_matrix, signal=survival_correlation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Decision:
    """One decision the engine made, with its rationale."""
    stage: str           # which stage: input, edges, faces, depth, domain
    key: str             # what was decided: threshold, typing, face_selection
    value: Any           # the decision value
    rationale: str       # why this value was chosen
    alternatives: list   # what other values were considered


@dataclass
class AnalysisPlan:
    """Complete plan for analyzing a dataset."""
    input_type: str = ""
    adapter: str = ""
    n_entities_est: int = 0
    n_edges_est: int = 0

    # Edge construction
    metric: str = ""
    threshold: Any = "auto"
    sign_strategy: str = "auto"
    typing_strategy: str = "auto"

    # Face selection
    face_selection: str = "all"
    has_context: bool = False
    context_source: str = ""
    n_contexts: int = 0

    # Analysis depth
    depth: str = "standard"
    depth_reason: str = ""

    # Signal
    has_signal: bool = False
    signal_source: str = ""

    # Domain
    domain: str = "exploratory"
    interpretation_strategy: str = "structural"

    # Decision log
    decisions: List[Decision] = field(default_factory=list)

    def log(self, stage, key, value, rationale, alternatives=None):
        self.decisions.append(Decision(
            stage=stage, key=key, value=value,
            rationale=rationale,
            alternatives=alternatives or [],
        ))


@dataclass
class EngineResult:
    """Complete output of the engine."""
    plan: AnalysisPlan
    rex: Any                           # RexGraph
    meta: Dict[str, Any]
    analysis: Dict[str, Any]
    signal_decomposition: Optional[Dict] = None
    interpretation: Optional[Dict] = None
    enrichment: Optional[List] = None
    session: Any = None

    def save(self, path: str, format: str = None, cache: str = "all"):
        """Save the complete result to disk.

        Writes the RexGraph via rexgraph.io (with all cached
        properties), plus the analysis results, plan, interpretation,
        and enrichment as JSON sidecars.

        Supported formats:
            .rex    NumPy bundle (zero dependencies)
            .zarr   Chunked compressed (requires zarr)
            .h5     Single HDF5 file (requires h5py)

        The sidecar files are always JSON, stored alongside the
        complex in the same directory (.rex, .zarr) or as a
        companion file (.h5).

        Parameters
        ----------
        path : str
            Output path (e.g., 'results.rex', 'results.zarr').
        format : str, optional
            Override format detection.
        cache : str
            Cache level for the RexGraph: 'all', 'standard', or None.
        """
        import json
        import os

        from rexgraph.io import save as _io_save
        from rexgraph.io._compat import dumps

        # Save the RexGraph
        _io_save(path, self.rex, cache=cache, format=format)

        # Determine sidecar directory
        if os.path.isdir(path):
            sidecar_dir = path
        else:
            sidecar_dir = os.path.dirname(path) or "."
            base = os.path.splitext(os.path.basename(path))[0]
            sidecar_dir = os.path.join(sidecar_dir, base + "_meta")
            os.makedirs(sidecar_dir, exist_ok=True)

        # Save analysis results
        if self.analysis:
            with open(os.path.join(sidecar_dir, "analysis.json"), "w") as f:
                f.write(dumps(self.analysis, indent=2))

        # Save plan
        plan_dict = {
            "input_type": self.plan.input_type,
            "adapter": self.plan.adapter,
            "domain": self.plan.domain,
            "depth": self.plan.depth,
            "depth_reason": self.plan.depth_reason,
            "face_selection": self.plan.face_selection,
            "has_context": self.plan.has_context,
            "context_source": self.plan.context_source,
            "n_contexts": self.plan.n_contexts,
            "metric": self.plan.metric,
            "threshold": str(self.plan.threshold),
            "sign_strategy": self.plan.sign_strategy,
            "typing_strategy": self.plan.typing_strategy,
            "has_signal": self.plan.has_signal,
            "signal_source": self.plan.signal_source,
            "interpretation_strategy": self.plan.interpretation_strategy,
            "decisions": [
                {
                    "stage": d.stage,
                    "key": d.key,
                    "value": str(d.value),
                    "rationale": d.rationale,
                    "alternatives": [str(a) for a in d.alternatives],
                }
                for d in self.plan.decisions
            ],
        }
        with open(os.path.join(sidecar_dir, "plan.json"), "w") as f:
            json.dump(plan_dict, f, indent=2)

        # Save metadata
        if self.meta:
            with open(os.path.join(sidecar_dir, "meta.json"), "w") as f:
                f.write(dumps(self.meta, indent=2))

        # Save interpretation
        if self.interpretation:
            with open(os.path.join(sidecar_dir, "interpretation.json"), "w") as f:
                f.write(dumps(self.interpretation, indent=2))

        # Save signal decomposition
        if self.signal_decomposition:
            with open(os.path.join(sidecar_dir, "signal.json"), "w") as f:
                f.write(dumps(self.signal_decomposition, indent=2))

        # Save enrichment count and sample
        if self.enrichment:
            enrichment_data = {
                "n_triples": len(self.enrichment),
                "sample": [
                    {"s": t.s, "p": t.p, "o": t.o}
                    for t in self.enrichment[:20]
                ],
            }
            with open(os.path.join(sidecar_dir, "enrichment.json"), "w") as f:
                json.dump(enrichment_data, f, indent=2)

    @classmethod
    def load(cls, path: str, format: str = None) -> "EngineResult":
        """Load a saved EngineResult from disk.

        Parameters
        ----------
        path : str
            Path to the saved result.
        format : str, optional
            Override format detection.

        Returns
        -------
        EngineResult
        """
        import json
        import os

        from rexgraph.io import load as _io_load

        rex = _io_load(path, format=format)

        # Find sidecar directory
        if os.path.isdir(path):
            sidecar_dir = path
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            sidecar_dir = os.path.join(
                os.path.dirname(path) or ".", base + "_meta"
            )

        def _load_json(name):
            p = os.path.join(sidecar_dir, name)
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
            return None

        analysis = _load_json("analysis.json") or {}
        plan_dict = _load_json("plan.json") or {}
        meta = _load_json("meta.json") or {}
        interpretation = _load_json("interpretation.json")
        signal_decomp = _load_json("signal.json")

        # Reconstruct plan
        plan = AnalysisPlan()
        for k, v in plan_dict.items():
            if k != "decisions" and hasattr(plan, k):
                setattr(plan, k, v)
        for d in plan_dict.get("decisions", []):
            plan.decisions.append(Decision(
                stage=d["stage"],
                key=d["key"],
                value=d["value"],
                rationale=d["rationale"],
                alternatives=d.get("alternatives", []),
            ))

        return cls(
            plan=plan,
            rex=rex,
            meta=meta,
            analysis=analysis,
            signal_decomposition=signal_decomp,
            interpretation=interpretation,
            enrichment=None,
        )


class DecisionEngine:
    """Inspect data, plan the analysis, execute it.

    The engine makes decisions based on the algebraic structure of
    the data itself.  No fitted parameters.  Every decision is
    recorded with its rationale so the user can understand and
    override any choice.
    """

    def plan(self, data, *, contexts=None, context_matrix=None,
             signal=None, **kwargs) -> AnalysisPlan:
        """Inspect data and produce an analysis plan without executing.

        Parameters
        ----------
        data : any supported input
        contexts : dict, optional
            Context label -> entity list mapping for face selection.
        context_matrix : ndarray, optional
            Pre-built binary context matrix.
        signal : ndarray, optional
            Domain-specific edge signal to decompose.

        Returns
        -------
        AnalysisPlan
        """
        plan = AnalysisPlan()

        # Stage 1: Input type detection
        self._decide_input_type(plan, data)

        # Stage 2: Edge construction parameters
        self._decide_edge_construction(plan, data, **kwargs)

        # Stage 3: Face selection
        self._decide_face_selection(plan, data, contexts,
                                    context_matrix, **kwargs)

        # Stage 4: Analysis depth
        self._decide_depth(plan, **kwargs)

        # Stage 5: Signal decomposition
        self._decide_signal(plan, signal)

        # Stage 6: Domain and interpretation
        self._decide_domain(plan, data, signal)

        return plan

    def run(self, data, *, contexts=None, context_matrix=None,
            signal=None, depth=None, session=None,
            **kwargs) -> EngineResult:
        """Full pipeline: detect, plan, build, analyze, interpret.

        Parameters
        ----------
        data : any supported input
        contexts : dict, optional
        context_matrix : ndarray, optional
        signal : ndarray, optional
        depth : str, optional
            Override the auto-selected depth.
        session : Session, optional
            Session to record snapshots in.
        **kwargs
            Forwarded to the adapter.

        Returns
        -------
        EngineResult
        """
        plan = self.plan(
            data, contexts=contexts,
            context_matrix=context_matrix,
            signal=signal, **kwargs,
        )

        if depth is not None:
            plan.depth = depth
            plan.depth_reason = "user override"

        # Build the complex
        rex, meta = self._build(plan, data, contexts,
                                context_matrix, **kwargs)

        # Run analysis
        analysis = self._analyze(plan, rex)

        # Signal decomposition
        signal_decomp = None
        if signal is not None and len(signal) == rex.nE:
            signal_decomp = self._decompose_signal(
                plan, rex, signal
            )

        # Domain interpretation
        interpretation = self._interpret(plan, rex, meta, analysis,
                                         signal_decomp)

        # Enrichment triples (for KG domain)
        enrichment = None
        if plan.domain == "knowledge_graph":
            enrichment = self._generate_enrichment(rex, analysis)

        # Session recording
        if session is not None:
            session.add_snapshot(
                rex, action="engine_run",
                params={"plan": plan.__dict__},
                results=analysis,
                summary=(
                    f"{plan.input_type} -> {rex.nV}V {rex.nE}E "
                    f"{rex.nF}F ({plan.face_selection})"
                ),
            )

        return EngineResult(
            plan=plan,
            rex=rex,
            meta=meta,
            analysis=analysis,
            signal_decomposition=signal_decomp,
            interpretation=interpretation,
            enrichment=enrichment,
            session=session,
        )

    # Decision methods

    def _decide_input_type(self, plan, data):
        """Detect what kind of data this is."""
        from agent.auto import detect_input_type

        # Check for TrustGraph triples first
        if self._is_triple_list(data):
            plan.input_type = "triples"
            plan.adapter = "TrustGraphAdapter"
            plan.log("input", "type", "triples",
                     "Data is a list of objects with .s, .p, .o attributes "
                     "or (s, p, o) tuples",
                     ["feature_matrix", "edge_csv", "text"])
            return

        input_type = detect_input_type(data)
        plan.input_type = input_type

        adapter_map = {
            "feature_matrix": "FeatureMatrixAdapter",
            "feature_csv": "FeatureMatrixAdapter",
            "correlation": "CorrelationAdapter",
            "adjacency": "AdjacencyAdapter",
            "edge_csv": "EdgeListAdapter",
            "text": "TextAdapter",
        }
        plan.adapter = adapter_map.get(input_type, "auto")

        plan.log("input", "type", input_type,
                 f"Detected from data shape/content",
                 list(adapter_map.keys()))

    def _decide_edge_construction(self, plan, data, **kwargs):
        """Decide metric, threshold, signs, and typing."""

        if plan.input_type == "triples":
            plan.metric = "predicate"
            plan.threshold = "none"
            plan.sign_strategy = "positive"
            plan.typing_strategy = "predicate_local_name"
            plan.log("edges", "metric", "predicate",
                     "KG triples: each triple is an edge, "
                     "predicate is the edge type")
            return

        if plan.input_type in ("feature_matrix", "feature_csv"):
            plan.metric = "pearson_correlation"
            plan.threshold = kwargs.get("threshold", "auto")
            plan.sign_strategy = "correlation"

            # Typing strategy
            feature_names = kwargs.get("feature_names")
            if feature_names and self._has_family_prefixes(feature_names):
                plan.typing_strategy = "column_family"
                plan.log("edges", "typing", "column_family",
                         "Feature names have common prefixes "
                         "(e.g., vital_0, lab_1)",
                         ["spectral", "none"])
            else:
                plan.typing_strategy = kwargs.get("typing", "spectral")

            plan.log("edges", "metric", "pearson_correlation",
                     "Feature matrix: edges from correlation")
            return

        if plan.input_type == "correlation":
            plan.metric = "given_correlation"
            plan.threshold = kwargs.get("threshold", "auto")
            plan.sign_strategy = "correlation"
            plan.typing_strategy = kwargs.get("typing", "spectral")
            plan.log("edges", "metric", "given_correlation",
                     "Pre-computed correlation matrix")
            return

        if plan.input_type == "text":
            plan.metric = "co_occurrence"
            plan.threshold = "none"
            plan.sign_strategy = "positive"
            plan.typing_strategy = "granularity"
            plan.log("edges", "metric", "co_occurrence",
                     "Text: edges from word co-occurrence "
                     "within EDUs, sentences, paragraphs")
            return

        # Default
        plan.metric = "given"
        plan.threshold = "none"
        plan.sign_strategy = kwargs.get("sign", "positive")
        plan.typing_strategy = kwargs.get("typing", "none")
        plan.log("edges", "metric", "given",
                 "Edge list or adjacency: edges as provided")

    def _decide_face_selection(self, plan, data, contexts,
                               context_matrix, **kwargs):
        """Decide how faces are selected."""
        has_ctx = (
            context_matrix is not None
            or (contexts is not None and len(contexts) > 1)
        )

        if has_ctx:
            plan.has_context = True
            plan.face_selection = "all"
            if contexts is not None:
                plan.n_contexts = len(contexts)
                plan.context_source = "explicit"
            elif context_matrix is not None:
                plan.n_contexts = context_matrix.shape[0]
                plan.context_source = "matrix"
            plan.log("faces", "selection", "context (primary) + typed + promote + none",
                     f"Context matrix available with {plan.n_contexts} "
                     f"contexts: algebraic face selection E = C^T|B1| > 0",
                     ["typed", "promote", "none"])
            return

        # For triples, check if document provenance is available
        if plan.input_type == "triples" and self._triples_have_provenance(data):
            plan.has_context = True
            plan.face_selection = "all"
            plan.context_source = "document_provenance"
            plan.log("faces", "selection", "context (from triple provenance)",
                     "Triples carry named graph or document IDs: "
                     "using document membership as context matrix",
                     ["typed", "promote"])
            return

        # No context: use typed if multiple edge types exist
        plan.face_selection = "all"
        plan.log("faces", "selection", "typed (primary) + promote + none",
                 "No context matrix: typed face selection from "
                 "edge types (same-type triangles become faces)",
                 ["promote", "none"])

    def _decide_depth(self, plan, **kwargs):
        """Decide analysis depth based on data size."""
        user_depth = kwargs.get("depth")
        if user_depth:
            plan.depth = user_depth
            plan.depth_reason = "user specified"
            plan.log("depth", "level", user_depth,
                     "User specified depth")
            return

        nE_est = plan.n_edges_est

        if nE_est > 50000:
            plan.depth = "quick"
            plan.depth_reason = (
                f"Estimated {nE_est} edges: spectral computation "
                f"scales as O(n^3), using quick depth"
            )
        elif nE_est > 1000:
            plan.depth = "standard"
            plan.depth_reason = (
                f"Estimated {nE_est} edges: full structural "
                f"analysis feasible"
            )
        else:
            plan.depth = "full"
            plan.depth_reason = (
                f"Estimated {nE_est} edges: small complex, "
                f"including RCFE strain and Dirac"
            )

        plan.log("depth", "level", plan.depth, plan.depth_reason,
                 ["quick", "standard", "full"])

    def _decide_signal(self, plan, signal):
        """Decide whether to decompose a signal."""
        if signal is not None:
            plan.has_signal = True
            plan.signal_source = "user_provided"
            plan.log("signal", "decompose", True,
                     "User provided an edge signal: "
                     "will run Hodge decomposition")
        else:
            plan.has_signal = False

    def _decide_domain(self, plan, data, signal):
        """Decide the interpretation domain."""
        if plan.input_type == "triples":
            plan.domain = "knowledge_graph"
            plan.interpretation_strategy = "confidence_and_enrichment"
            plan.log("domain", "type", "knowledge_graph",
                     "Input is KG triples: confidence scoring, "
                     "per-entity structural character, enrichment")
            return

        if plan.input_type == "text":
            plan.domain = "language"
            plan.interpretation_strategy = "fingerprint"
            plan.log("domain", "type", "language",
                     "Input is text: structural fingerprint, "
                     "section detection")
            return

        if plan.input_type in ("feature_matrix", "feature_csv"):
            if signal is not None:
                plan.domain = "clinical"
                plan.interpretation_strategy = "signal_decomposition"
                plan.log("domain", "type", "clinical",
                         "Feature matrix with signal: per-patient "
                         "face/void realization, Hodge decomposition")
            else:
                plan.domain = "exploratory"
                plan.interpretation_strategy = "structural"
                plan.log("domain", "type", "exploratory",
                         "Feature matrix without signal: structural "
                         "character, void analysis")
            return

        plan.domain = "exploratory"
        plan.interpretation_strategy = "structural"
        plan.log("domain", "type", "exploratory",
                 "Default: structural analysis")

    # Build and analyze

    def _build(self, plan, data, contexts, context_matrix, **kwargs):
        """Build the RexGraph according to the plan."""
        if plan.input_type == "triples":
            from agent.integrations.trustgraph_adapter import (
                TrustGraphAdapter,
            )
            adapter = TrustGraphAdapter()
            return adapter.from_triples(
                data,
                face_selection=plan.face_selection,
                contexts=contexts,
                context_matrix=context_matrix,
            )
        else:
            from agent.auto import auto_rex
            build_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ("threshold", "typing", "sign",
                         "feature_names", "vertex_labels",
                         "face_selection")
            }
            if "face_selection" not in build_kwargs:
                # Map plan face selection to auto_rex format
                fs = plan.face_selection
                if fs == "all":
                    fs = "typed"  # auto_rex doesn't have "all"
                build_kwargs["face_selection"] = fs

            rex = auto_rex(data, **build_kwargs)
            meta = getattr(rex, "_agent_meta", {})
            return rex, meta

    def _analyze(self, plan, rex):
        """Run the analysis pipeline."""
        if plan.input_type == "triples":
            from agent.integrations.trustgraph_adapter import (
                TrustGraphAdapter,
            )
            adapter = TrustGraphAdapter()
            return adapter.analyze(rex, depth=plan.depth)
        else:
            from agent.pipeline import AnalysisPipeline
            pipe = AnalysisPipeline(rex)
            return pipe.run(depth=plan.depth)

    def _decompose_signal(self, plan, rex, signal):
        """Decompose an edge signal."""
        if plan.input_type == "triples":
            from agent.integrations.trustgraph_adapter import (
                TrustGraphAdapter,
            )
            adapter = TrustGraphAdapter()
            return adapter.decompose_signal(
                rex, signal, plan.signal_source
            )
        else:
            from agent.pipeline import AnalysisPipeline
            pipe = AnalysisPipeline(rex)
            return pipe.decompose_signal(
                signal, plan.signal_source
            )

    def _interpret(self, plan, rex, meta, analysis, signal_decomp):
        """Produce domain-specific interpretation."""
        interp = {
            "domain": plan.domain,
            "strategy": plan.interpretation_strategy,
        }

        if plan.domain == "knowledge_graph":
            interp.update(self._interpret_kg(rex, meta, analysis))
        elif plan.domain == "clinical":
            interp.update(self._interpret_clinical(
                rex, analysis, signal_decomp
            ))

        return interp

    def _interpret_kg(self, rex, meta, analysis):
        """Knowledge graph interpretation."""
        result = {}

        # Per-entity structural summary
        rel = analysis.get("relational", {})
        kappa = rel.get("kappa_per_vertex")
        phi = rel.get("phi_per_vertex")
        labels = meta.get("vertex_labels", [])

        if kappa and labels:
            # Entities needing review = low-coherence OUTLIERS, flagged by a
            # data-adaptive lower Tukey fence (q1 - 1.5·IQR), not a fixed magic
            # cutoff. Coherence κ is continuous (no integer invariant applies), so
            # the threshold is derived from the κ distribution itself - matching the
            # project's outlier-detection convention (schema_complex relation_lint).
            n = min(len(kappa), len(labels))
            k_arr = np.asarray(kappa[:n], dtype=np.float64)
            q1, q3 = np.percentile(k_arr, [25.0, 75.0])
            fence = float(q1 - 1.5 * (q3 - q1))
            review_entities = [
                {"entity": labels[i], "kappa": kappa[i]}
                for i in range(n)
                if kappa[i] < fence
            ]
            result["entities_needing_review"] = review_entities
            result["n_review"] = len(review_entities)

            # What's LOAD-BEARING around the flagged entities: seed the incoherent
            # entities, diffuse (demand-driven), and read which reached relations are
            # BRIDGES - critical links with no backup path - plus how far the flagged
            # incoherence reaches (blast radius). The "what's load-bearing / what's
            # frustrated" verdict the narrative otherwise lacks.
            try:
                seeds = [i for i in range(n) if kappa[i] < fence]
                if not seeds:
                    seeds = [int(x) for x in np.argsort(k_arr)[:min(3, n)]]
                if seeds:
                    ar = rex.agentic_reading(vertices=seeds, t=1.0)
                    n_bridges = sum(1 for lb in ar["load_bearing"]
                                    if lb["effective_resistance"] > 0.9)
                    result["blast_radius"] = ar["context_size"]
                    result["load_bearing_relation_count"] = n_bridges
                    if n_bridges:
                        result["structural_risk"] = (
                            f"{n_bridges} load-bearing relation(s) reached from the "
                            "flagged entities - critical links with no backup path; the "
                            "answer here depends on structure that would fragment if any "
                            "were wrong")
            except Exception:
                pass

        # Void summary
        void = analysis.get("void", {})
        if void.get("n_voids", 0) > 0:
            vf = void.get("void_fraction", 0)
            result["void_assessment"] = (
                f"{void['n_voids']} voids out of "
                f"{void.get('n_potential', 0)} potential triangles "
                f"(void fraction {vf:.1%})"
            )
            vchi = void.get("void_chi_mean")
            if vchi:
                dom = max(vchi, key=vchi.get)
                result["void_dominant_channel"] = dom
        else:
            result["void_assessment"] = "No voids: all triangles realized"

        # Hodge assessment
        hodge = analysis.get("hodge", {})
        g = hodge.get("pct_gradient", 0)
        c = hodge.get("pct_curl", 0)
        h = hodge.get("pct_harmonic", 0)
        beyond = c + h
        if beyond > 0.5:
            result["hodge_assessment"] = (
                f"{beyond:.0%} of structural information is beyond "
                f"pairwise methods (curl={c:.0%}, harmonic={h:.0%})"
            )
        else:
            result["hodge_assessment"] = (
                f"Gradient-dominant ({g:.0%}): most structure "
                f"accessible to standard graph methods"
            )

        # Harmonic mode analysis (from compiled _harmonic module)
        dim_H = hodge.get("dim_H", 0)
        if dim_H > 0:
            result["oscillatory_modes"] = dim_H
            modes = hodge.get("harmonic_modes", [])
            if modes:
                result["harmonic_mode_count"] = len(modes)

            # Frustration and coparticipation
            frust = hodge.get("frustration_total", 0)
            copart = hodge.get("coparticipation_total", 0)
            health = hodge.get("health_ratio")

            if health is not None:
                if health > 1.1:
                    result["stability_assessment"] = (
                        f"Unstable: frustration ({frust:.2f}) "
                        f"exceeds coparticipation ({copart:.2f}), "
                        f"health ratio {health:.3f}"
                    )
                elif health < 0.9:
                    result["stability_assessment"] = (
                        f"Stable: coparticipation ({copart:.2f}) "
                        f"exceeds frustration ({frust:.2f}), "
                        f"health ratio {health:.3f}"
                    )
                else:
                    result["stability_assessment"] = (
                        f"Near equilibrium: frustration ({frust:.2f}) and "
                        f"coparticipation ({copart:.2f}) are balanced, "
                        f"health ratio {health:.3f}"
                    )

        return result

    def _interpret_clinical(self, rex, analysis, signal_decomp):
        """Clinical data interpretation."""
        result = {}

        if signal_decomp:
            g = signal_decomp.get("pct_gradient", 0)
            c = signal_decomp.get("pct_curl", 0)
            h = signal_decomp.get("pct_harmonic", 0)
            beyond = c + h
            result["signal_beyond_pairwise"] = f"{beyond:.0%}"
            result["signal_gradient"] = f"{g:.0%}"

        return result

    def _generate_enrichment(self, rex, analysis):
        """Generate enrichment triples for knowledge graphs."""
        from agent.integrations.trustgraph_adapter import (
            TrustGraphAdapter,
        )
        adapter = TrustGraphAdapter()
        return adapter.to_enrichment_triples(rex, analysis)

    # Helpers

    def _is_triple_list(self, data) -> bool:
        """Check if data is a list of triples."""
        if not isinstance(data, list) or len(data) == 0:
            return False
        first = data[0]
        # Has .s, .p, .o attributes
        if hasattr(first, "s") and hasattr(first, "p") and hasattr(first, "o"):
            return True
        # Is a tuple/list of length 3+
        if isinstance(first, (tuple, list)) and len(first) >= 3:
            # Check it's not numeric (would be a feature matrix row)
            if all(isinstance(x, str) for x in first[:3]):
                return True
        return False

    def _has_family_prefixes(self, names) -> bool:
        """Check if feature names have common family prefixes."""
        if not names or len(names) < 4:
            return False
        prefixes = set()
        for name in names:
            parts = name.split("_")
            if len(parts) >= 2:
                prefixes.add(parts[0])
        # At least 2 distinct prefixes, covering most features
        return len(prefixes) >= 2 and len(prefixes) < len(names) * 0.5

    def _triples_have_provenance(self, triples) -> bool:
        """Check if triples carry document provenance."""
        if not triples:
            return False
        first = triples[0]
        if hasattr(first, "g") and first.g is not None:
            return True
        if isinstance(first, (tuple, list)) and len(first) >= 4:
            return True
        return False
