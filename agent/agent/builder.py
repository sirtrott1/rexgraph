"""
agent.builder: custom agent composition from config.

Define pipelines as YAML/JSON, execute without writing Python.
Each step is a named building block with configurable parameters.

Usage:

    # From YAML
    agent = AgentBuilder.from_yaml("my_agent.yaml")
    result = agent.run(files=["contract.pdf"], query="liability clause")

    # From dict
    agent = AgentBuilder({
        "name": "contract-reviewer",
        "steps": [
            {"type": "ocr", "params": {"strategy": "layout"}},
            {"type": "chunk", "params": {"min_chars": 200}},
            {"type": "query", "params": {"mode": "spectral"}},
            {"type": "model", "params": {"prompt_template": "Analyze: {context}"}},
            {"type": "hallucination_check"},
            {"type": "export", "params": {"format": "safetensors"}},
        ],
    })
    result = agent.run(files=["contract.pdf"])

    # Save/share agent configs
    agent.save("contract-reviewer.yaml")
    agent2 = AgentBuilder.load("contract-reviewer.yaml")

Example YAML:

    name: contract-reviewer
    description: Structural analysis of legal contracts
    version: 1

    defaults:
      min_count: 1
      max_vocab: 400
      depth: standard

    steps:
      - type: ocr
        params:
          strategy: layout
          dpi: 300

      - type: corpus
        params:
          depth: standard

      - type: chunk
        params:
          min_chars: 200

      - type: query
        params:
          mode: spectral
          top_k: 3

      - type: model
        params:
          prompt_template: |
            You are a contract analyst. Given the following structural
            context (kappa={kappa}, channel={channel}, hodge={hodge}),
            analyze this section:

            {context}
          max_tokens: 2048

      - type: hallucination_check

      - type: export
        params:
          format: safetensors
          include_features: true
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    step_type: str
    status: str = "ok"  # ok, error, skipped
    data: dict = field(default_factory=dict)
    elapsed: float = 0.0
    error: str = ""


@dataclass
class AgentResult:
    """Result of a full agent run."""
    name: str = ""
    steps: list[StepResult] = field(default_factory=list)
    elapsed: float = 0.0

    # Accumulated state across steps
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    query_results: list = field(default_factory=list)
    model_response: str = ""
    hallucination_report: dict = field(default_factory=dict)
    export_path: str = ""


# Step registry
_STEPS = {}

def register_step(name: str):
    """Decorator to register a step type."""
    def wrapper(fn):
        _STEPS[name] = fn
        return fn
    return wrapper


@register_step("ocr")
def _step_ocr(files, state, params):
    """OCR processing of input files."""
    from agent.adapters.ocr import OCRAdapter
    from agent.adapters.text import TextAdapter

    # Canonical OCR extension set (kept in sync with the pipeline).
    ocr_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp",
                ".bmp", ".tiff", ".tif"}
    strategy = params.get("strategy", "text")
    results = []

    for filepath in files:
        try:
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ocr_exts:
                adapter = OCRAdapter()
                ec = adapter.build(filepath, strategy=strategy,
                                   dpi=params.get("dpi", 200))
            else:
                adapter = TextAdapter()
                with open(filepath, encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
                ec = adapter.build(text, min_count=params.get("min_count", 1),
                                   max_vocab=params.get("max_vocab", 400))
            results.append({"file": filepath, "nV": ec.nV, "nE": ec.nE, "ec": ec})
        except Exception as e:
            results.append({"file": filepath, "error": str(e)})

    state["edge_constructions"] = results
    return {"n_processed": len(results)}


@register_step("corpus")
def _step_corpus(files, state, params):
    """Build a corpus from processed documents."""
    from agent.corpus import CorpusBuilder

    corpus = CorpusBuilder()
    ecs = state.get("edge_constructions", [])

    for item in ecs:
        ec = item.get("ec")
        if ec is None or ec.nE == 0:
            continue
        # Pass the already-built construction straight through instead of
        # re-deriving from ec.source_text, which is empty for CSV/feature
        # adapters and would add blank documents (audit B2).
        doc_id = item.get("file", "doc")
        corpus.add_document(
            source=doc_id,
            doc_id=doc_id,
            text=ec.source_text or "",
            edge_construction=ec,
        )

    if not ecs:
        # Direct from files - route each through auto_rex by detected type
        # (CSV -> edge/feature, JSON -> loader, text -> co-occurrence) rather
        # than force-reading every file as prose (audit B4).
        for filepath in files:
            corpus.add_document(source=filepath, doc_id=filepath)

    depth = params.get("depth", "standard")
    corpus.build(depth=depth)
    state["corpus"] = corpus

    docs = []
    for doc in corpus.documents:
        d = {"doc_id": doc.doc_id}
        if doc.rex:
            d["nV"] = doc.rex.nV
            d["nE"] = doc.rex.nE
            d["betti"] = doc.rex.betti
            kappa = doc.rex.coherence
            d["kappa"] = round(float(kappa.mean()), 4) if kappa is not None and len(kappa) > 0 and not np.isnan(kappa.mean()) else 0.0
        docs.append(d)

    return {"n_documents": corpus.n_documents, "documents": docs}


@register_step("chunk")
def _step_chunk(files, state, params):
    """Hodge-based chunking."""
    from agent.adapters.text import TextAdapter
    from agent.chunking import hodge_chunk
    from rexgraph.graph import RexGraph

    corpus = state.get("corpus")
    min_chars = params.get("min_chars", 100)
    ta = TextAdapter()
    all_chunks = []

    if corpus:
        for doc in corpus.documents:
            if doc.rex is None:
                continue
            source = doc.text or ''
            if not source:
                continue
            ec = ta.build(source, min_count=1, max_vocab=400)
            if ec.nE == 0:
                continue
            rex = RexGraph(sources=ec.sources, targets=ec.targets)
            if ec.n_types > 1:
                rex = rex.typed_face_selection(ec.type_labels)
            chunks = hodge_chunk(rex, ec.edge_spans, ec.sentence_spans,
                                 source, min_chunk_chars=min_chars)
            all_chunks.append({"doc_id": doc.doc_id, "chunks": chunks})

    state["chunks"] = all_chunks
    total = sum(len(dc["chunks"]) for dc in all_chunks)
    return {"n_chunk_groups": len(all_chunks), "total_chunks": total}


@register_step("query")
def _step_query(files, state, params):
    """Spectral/chi/hybrid query."""
    corpus = state.get("corpus")
    query = state.get("query", "")
    if not corpus or not query:
        return {"skipped": "no corpus or query"}

    mode = params.get("mode", "hybrid")
    top_k = params.get("top_k", 5)
    result = corpus.query(query, mode=mode, top_k=top_k)
    state["query_result"] = result
    return {"ranked": result.ranked_sections}


@register_step("model")
def _step_model(files, state, params):
    """Send context to a model."""
    chunks = state.get("chunks", [])
    query = state.get("query", "")
    template = params.get("prompt_template", "Analyze:\n{context}")
    max_tokens = params.get("max_tokens", 1024)

    # Build context from top chunks
    context_parts = []
    for dc in chunks:
        for chunk in dc.get("chunks", [])[:3]:
            context_parts.append(chunk.text)
    context = "\n\n".join(context_parts)

    # Format prompt
    # Get structural info from first chunk for template
    first_chunk = None
    if chunks and chunks[0].get("chunks"):
        first_chunk = chunks[0]["chunks"][0]

    prompt = template.format(
        context=context,
        query=query,
        kappa=f"{first_chunk.kappa:.3f}" if first_chunk else "N/A",
        channel=first_chunk.dominant_channel if first_chunk else "N/A",
        hodge=f"{first_chunk.hodge_gradient:.2f}/{first_chunk.hodge_curl:.2f}/{first_chunk.hodge_harmonic:.2f}" if first_chunk else "N/A",
    )

    state["prompt"] = prompt
    state["model_params"] = {"max_tokens": max_tokens}

    # Try calling the model
    try:
        import httpx
        model_url = os.environ.get("CHAT_MODEL_URL", "http://localhost:10000")
        r = httpx.post(
            model_url + "/v1/chat/completions",
            json={"model": "default", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens},
            timeout=120,
        )
        if r.status_code == 200:
            data = r.json()
            text = ""
            for choice in data.get("choices", []):
                text += choice.get("message", {}).get("content", "")
            from agent.server.security import sanitize_model_response
            state["model_response"] = sanitize_model_response(text)
            return {"response_length": len(text)}
    except Exception:
        pass

    state["model_response"] = "(no model available - prompt prepared, %d chars)" % len(prompt)
    return {"prompt_length": len(prompt), "model": "unavailable"}


@register_step("hallucination_check")
def _step_hallucination(files, state, params):
    """Check model output for hallucinations."""
    from agent.hallucination import detect_hallucinations_exchange

    model_response = state.get("model_response", "")
    chunks = state.get("chunks", [])
    if not model_response or not chunks:
        return {"skipped": "no response or chunks"}

    # Use first chunk group's text as source
    source_parts = []
    for dc in chunks:
        for chunk in dc.get("chunks", []):
            source_parts.append(chunk.text)
    source = "\n".join(source_parts)

    report = detect_hallucinations_exchange(source, model_response)
    state["hallucination_report"] = {
        "score": report.overall_score,
        "kappa_correlation": report.kappa_correlation,
        "n_shared": report.n_shared_entities,
        "n_flags": report.n_flags,
        "summary": report.summary(),
    }
    return state["hallucination_report"]


@register_step("export")
def _step_export(files, state, params):
    """Export results."""
    fmt = params.get("format", "json")
    output = params.get("output", "agent_output")

    if fmt == "safetensors" and state.get("corpus"):
        from agent.training import TrainingExporter
        te = TrainingExporter(state["corpus"])
        path = output + ".safetensors"
        te.export_features(path)
        state["export_path"] = path
        return {"path": path, "n_examples": len(te.examples)}

    if fmt == "json":
        path = output + ".json"
        export = {
            "query": state.get("query", ""),
            "model_response": state.get("model_response", ""),
            "hallucination_report": state.get("hallucination_report", {}),
        }
        Path(path).write_text(json.dumps(export, indent=2, default=str))
        state["export_path"] = path
        return {"path": path}

    if fmt == "rex" and state.get("corpus"):
        from agent.training import TrainingExporter
        te = TrainingExporter(state["corpus"])
        paths = te.export_rex_bundles(output)
        return {"paths": paths}

    return {"skipped": "no exportable data"}


@register_step("training_export")
def _step_training_export(files, state, params):
    """Export training pairs for model fine-tuning."""
    corpus = state.get("corpus")
    if not corpus:
        return {"skipped": "no corpus"}

    from agent.training import TrainingExporter
    te = TrainingExporter(corpus)

    target = params.get("target", "summary")
    output = params.get("output", "training_pairs.safetensors")

    te.export_training_pairs(output, target=target)
    return {"path": output, "n_examples": len(te.examples), "target": target}


@register_step("langgraph_init")
def _step_langgraph_init(files, state, params):
    """Initialize a RexStateGraph for agent execution tracking.

    Creates a state graph that models the agent's execution as a
    relational complex. Subsequent steps are automatically tracked
    as state transitions.

    The structural diagnostics tell you:
    - Is the agent making progress? (gradient-dominated)
    - Is it going in circles? (curl-dominated)
    - Is it stuck in loops it can't break? (harmonic-dominated)
    - Which transitions are structurally supported? (void analysis)
    """
    from agent.integrations.langgraph_rex import RexStateGraph

    rsg = RexStateGraph()

    # Register states from config
    for s in params.get("states", []):
        name = s if isinstance(s, str) else s.get("name", "")
        meta = s.get("metadata", {}) if isinstance(s, dict) else {}
        if name:
            rsg.add_state(name, metadata=meta)

    # Register transitions from config
    for t in params.get("transitions", []):
        rsg.add_transition(
            t.get("from", ""), t.get("to", ""),
            weight=t.get("weight", 1.0),
            sign=t.get("sign", 1),
            transition_type=t.get("type", "default") if isinstance(t.get("type"), str) else "default",
        )

    state["state_graph"] = rsg
    return {"n_states": len(rsg._states), "n_transitions": len(rsg._transitions)}


@register_step("langgraph_analyze")
def _step_langgraph_analyze(files, state, params):
    """Analyze the current agent execution graph.

    Returns structural diagnostics: Hodge decomposition of the
    execution path, cycle detection, transition confidence, and
    should_continue recommendation.
    """
    rsg = state.get("state_graph")
    if rsg is None:
        return {"skipped": "no state graph - add langgraph_init first"}

    result = {}

    # Build the relational complex
    try:
        rsg.build()
    except Exception as e:
        return {"error": f"build failed: {e}"}

    # Full analysis (may fail on some kernels)
    try:
        analysis = rsg.analyze()
        result["analysis"] = analysis
    except Exception:
        # Partial analysis fallback
        try:
            rex = rsg.rex
            import numpy as np
            kappa = rex.coherence
            km = round(float(kappa.mean()), 4) if kappa is not None and len(kappa) > 0 and not np.isnan(kappa.mean()) else 0.0
            result["analysis"] = {
                "nV": rex.nV, "nE": rex.nE,
                "betti": rex.betti,
                "kappa_mean": km,
            }
        except Exception:
            pass

    # These read the complex, not the analysis, so they run whether or not the full
    # analysis succeeded. Indenting them into the handler above made them conditional
    # on rsg.analyze() raising, so a successful run returned none of them.
    try:
        threshold = params.get("harmonic_threshold", 0.4)
        result["should_continue"] = rsg.should_continue(harmonic_threshold=threshold)

        result["cycles"] = rsg.detect_cycles()

        if rsg._execution_log:
            result["path_hodge"] = rsg.decompose_path(rsg._execution_log)
    except Exception as e:
        result["error"] = str(e)

    state["graph_analysis"] = result
    return result


@register_step("langgraph_check")
def _step_langgraph_check(files, state, params):
    """Check a specific transition's structural confidence.

    Uses void analysis and coherence to determine if a transition
    is structurally supported, weak, or nonexistent.
    """
    rsg = state.get("state_graph")
    if rsg is None:
        return {"skipped": "no state graph"}

    src = params.get("from", "")
    tgt = params.get("to", "")
    if not src or not tgt:
        return {"skipped": "specify 'from' and 'to' parameters"}

    return rsg.transition_confidence(src, tgt)


@register_step("langchain_tools")
def _step_langchain_tools(files, state, params):
    """Make LangChain tools available from the current structural analysis.

    Creates RexConfidenceTool, RexAnalyzeTool, RexHodgeTool, and
    RexExplainTool bound to the current document's RexGraph.
    These tools let an LLM agent query structural properties
    during execution.

    The tools give the model:
    - Mathematical confidence (not probability - void counts and kappa)
    - Full structural analysis (betti, character, Hodge)
    - Signal decomposition (gradient/curl/harmonic on any flow)
    - Per-edge/vertex explanation (why this edge matters)
    """
    corpus = state.get("corpus")
    rex = None

    if corpus and corpus.documents:
        for doc in corpus.documents:
            if doc.rex is not None:
                rex = doc.rex
                break

    if rex is None:
        return {"skipped": "no RexGraph available"}

    try:
        from agent.integrations.langchain_tools import (
            RexAnalyzeTool,
            RexConfidenceTool,
            RexExplainTool,
            RexHodgeTool,
        )
        tools = [
            RexConfidenceTool(rex),
            RexAnalyzeTool(rex),
            RexHodgeTool(rex),
            RexExplainTool(rex),
        ]
        state["langchain_tools"] = tools
        return {
            "n_tools": len(tools),
            "tool_names": [t.name for t in tools],
            "note": "Tools available in state['langchain_tools'] for LangChain agent creation",
        }
    except ImportError:
        return {"error": "pip install langchain-core for LangChain tools"}


@register_step("langgraph_record")
def _step_langgraph_record(files, state, params):
    """Record a state visit in the execution log.

    Use this between other steps to build the execution trace.
    The trace is Hodge-decomposed by langgraph_analyze.
    """
    rsg = state.get("state_graph")
    if rsg is None:
        return {"skipped": "no state graph"}

    visit = params.get("state", "")
    if visit:
        # Record state visit in execution log
        if visit not in rsg._states:
            rsg.add_state(visit)
        rsg._execution_log.append(visit)
        rsg._dirty = True
        return {"visited": visit, "log_length": len(rsg._execution_log)}
    return {"skipped": "no state specified"}


class AgentBuilder:
    """Build and run custom agents from configuration."""

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "custom-agent")
        self.description = config.get("description", "")
        self.steps = config.get("steps", [])
        self.defaults = config.get("defaults", {})

    @classmethod
    def from_yaml(cls, path: str) -> AgentBuilder:
        """Load agent config from YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_json(cls, path: str) -> AgentBuilder:
        """Load agent config from JSON."""
        with open(path) as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def load(cls, path: str) -> AgentBuilder:
        """Load from YAML or JSON (auto-detect)."""
        if path.endswith((".yml", ".yaml")):
            return cls.from_yaml(path)
        return cls.from_json(path)

    def save(self, path: str):
        """Save agent config."""
        if path.endswith((".yml", ".yaml")):
            try:
                import yaml
                with open(path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            except ImportError:
                path = path.rsplit(".", 1)[0] + ".json"
                with open(path, "w") as f:
                    json.dump(self.config, f, indent=2)
        else:
            with open(path, "w") as f:
                json.dump(self.config, f, indent=2)
        return path

    def run(self, files: list[str] = None, query: str = None) -> AgentResult:
        """Execute the agent pipeline."""
        t0 = time.time()
        result = AgentResult(name=self.name)
        state = {"query": query or "", "defaults": self.defaults}

        for step_config in self.steps:
            step_type = step_config.get("type", "")
            params = dict(self.defaults)
            params.update(step_config.get("params", {}))

            step_fn = _STEPS.get(step_type)
            if step_fn is None:
                sr = StepResult(step_type=step_type, status="error",
                                error=f"Unknown step type: {step_type}")
                result.steps.append(sr)
                continue

            st = time.time()
            try:
                data = step_fn(files or [], state, params)
                sr = StepResult(step_type=step_type, status="ok",
                                data=data or {}, elapsed=time.time() - st)
            except Exception as e:
                sr = StepResult(step_type=step_type, status="error",
                                error=str(e), elapsed=time.time() - st)
                if step_config.get("required", True):
                    result.steps.append(sr)
                    break

            result.steps.append(sr)

        # Collect accumulated state (audit B1: these were previously
        # never copied back, so the result always looked empty).
        result.chunks = state.get("chunks", []) or []

        qr = state.get("query_result")
        if isinstance(qr, list):
            result.query_results = qr
        elif qr is not None:
            result.query_results = [qr]

        corpus = state.get("corpus")
        if corpus is not None:
            docs = []
            for d in getattr(corpus, "documents", []):
                rex = getattr(d, "rex", None)
                docs.append({
                    "doc_id": getattr(d, "doc_id", ""),
                    "nV": getattr(rex, "nV", 0) if rex is not None else 0,
                    "nE": getattr(rex, "nE", 0) if rex is not None else 0,
                    "nF": getattr(rex, "nF", 0) if rex is not None else 0,
                    "analysis": getattr(d, "analysis", {}) or {},
                })
            result.documents = docs

        result.model_response = state.get("model_response", "")
        result.hallucination_report = state.get("hallucination_report", {})
        result.export_path = state.get("export_path", "")
        result.elapsed = time.time() - t0

        return result

    @staticmethod
    def available_steps() -> list[str]:
        """List all registered step types."""
        return sorted(_STEPS.keys())

    @staticmethod
    def step_help(step_type: str) -> str:
        """Get docstring for a step type."""
        fn = _STEPS.get(step_type)
        return fn.__doc__ if fn else "Unknown step type"

    @staticmethod
    def template(name: str = "default") -> dict:
        """Get a starter template config."""
        templates = {
            "default": {
                "name": "my-agent",
                "steps": [
                    {"type": "corpus", "params": {"depth": "standard"}},
                    {"type": "chunk"},
                    {"type": "query", "params": {"mode": "spectral"}},
                ],
            },
            "rag": {
                "name": "rag-pipeline",
                "description": "RAG with structural verification",
                "steps": [
                    {"type": "ocr", "params": {"strategy": "layout"}},
                    {"type": "corpus", "params": {"depth": "standard"}},
                    {"type": "chunk", "params": {"min_chars": 200}},
                    {"type": "query", "params": {"mode": "spectral", "top_k": 3}},
                    {"type": "model", "params": {"max_tokens": 2048}},
                    {"type": "hallucination_check"},
                ],
            },
            "training": {
                "name": "training-export",
                "description": "Export structural features for model fine-tuning",
                "steps": [
                    {"type": "corpus", "params": {"depth": "standard"}},
                    {"type": "chunk"},
                    {"type": "training_export", "params": {"target": "channel", "output": "training.safetensors"}},
                ],
            },
            "langgraph": {
                "name": "structural-agent",
                "description": "LangGraph agent with structural execution tracking",
                "steps": [
                    {"type": "langgraph_init", "params": {
                        "states": [
                            {"name": "retrieve", "metadata": {"tool": "search"}},
                            {"name": "analyze", "metadata": {"tool": "rexgraph"}},
                            {"name": "reason", "metadata": {"tool": "llm"}},
                            {"name": "verify", "metadata": {"tool": "hallucination_check"}},
                            {"name": "answer", "metadata": {"tool": "output"}},
                        ],
                        "transitions": [
                            {"from": "retrieve", "to": "analyze"},
                            {"from": "analyze", "to": "reason"},
                            {"from": "reason", "to": "verify"},
                            {"from": "verify", "to": "answer"},
                            {"from": "verify", "to": "retrieve", "type": 1},
                            {"from": "reason", "to": "retrieve", "type": 1},
                        ],
                    }},
                    {"type": "corpus", "params": {"depth": "standard"}},
                    {"type": "langgraph_record", "params": {"state": "retrieve"}},
                    {"type": "chunk"},
                    {"type": "query", "params": {"mode": "spectral"}},
                    {"type": "langgraph_record", "params": {"state": "analyze"}},
                    {"type": "model"},
                    {"type": "langgraph_record", "params": {"state": "reason"}},
                    {"type": "hallucination_check"},
                    {"type": "langgraph_record", "params": {"state": "verify"}},
                    {"type": "langgraph_analyze"},
                ],
            },
            "langchain": {
                "name": "langchain-structural",
                "description": "Create LangChain tools from document structure",
                "steps": [
                    {"type": "corpus", "params": {"depth": "standard"}},
                    {"type": "langchain_tools"},
                ],
            },
        }
        return templates.get(name, templates["default"])
