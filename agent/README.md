# RexGraph Agent

The application layer over the compiled `rexgraph` core.

`rexgraph` turns relational data into a relational complex (a graded cell
complex where edges are primary and vertices are derived from edge boundaries)
and decomposes it with Hodge theory and typed Laplacians (see the
[top-level README](../README.md) for the math). `rexgraph-agent` is the platform
built on that core:

- a **FastAPI server and web UI** for upload, analysis, exploration, chat,
  corpus, models, connectors, and the lifecycle;
- a **hive**: a multi-agent swarm orchestrated as a relational complex, where
  agents are cells and their messages are recorded into a live complex, so
  routing is driven by structure and demonstrated history;
- a **setups-driven lifecycle**: named configuration profiles that drive the
  `serve`, `train`, `build`, `deploy`, `test`, `bench`, `finetune`, `ingest`,
  and `pipeline` operations, each run-logged;
- **local LLM inference** through llama.cpp, hardware-aware and tiered;
- a **model builder** for custom ML architectures (CNN, HGNN, language model,
  MLP) trained with the `rexgraph.nn` substrate;
- **connectors** that turn any database into a relational complex;
- **integrations** (TrustGraph knowledge cores and rex-RAG, HuggingFace,
  LangChain, LangGraph) as one part of the ecosystem.

Everything is reachable from the CLI, the HTTP API, and the browser UI.


## Installation

From the repo root (build the core first, then install a profile):

```bash
pip install .                       # rexgraph Cython core
pip install "./agent[standard]"     # full local deployment: UI, connectors, schema, OCR, training
```

Profiles bundle the granular extras. Pick the smallest one that fits:

| Profile | For | Weight |
|---------|-----|--------|
| `server` | web UI and API only | light |
| `standard` | full local deployment: UI, schema tools, connectors, OCR, training export | medium |
| `integrations` | LangChain, LangGraph, TrustGraph | medium |
| `ml` | LoRA fine-tuning and HuggingFace model analysis (torch) | heavy |
| `warehouse` | Snowflake / BigQuery / Redshift / Databricks drivers | heavy |
| `all` | everything that runs on CPU | large |

Granular extras (`schema`, `connectors`, `ocr`, `ocr-paddle`, `training`,
`finetune`, `huggingface`, `langchain`, `langgraph`, `trustgraph`, `mistral`,
`oidc`, `vllm`) combine directly, e.g. `pip install "./agent[server,trustgraph]"`.
Local inference uses the llama.cpp `llama-server` binary; build it with
`NATIVE=1 sh install.sh` from the repo root, or point the runtime at an existing
build.


## Quick Start

Serve the platform and open the UI:

```bash
rcf-server                          # FastAPI server + web UI on http://127.0.0.1:8000
# or: rexgraph-serve  /  python run.py  (run.py opens a browser; --no-browser to skip)
```

Build a hive and route a query through it:

```python
from agent.hive import Hive

hive = Hive()
hive.attach("worker", "http://127.0.0.1:8080", role="worker", model="qwen2.5",
            specialties=["topology", "homology", "hodge"])   # a running llama-server
hive.dispatch("What is the first Betti number of a torus?")  # routes + asks the best bee
```

Turn a knowledge graph into a relational complex and score a retrieval (rex-RAG,
no running service needed):

```python
from agent.integrations.trustgraph_adapter import TrustGraphAdapter, SimpleTriple

adapter = TrustGraphAdapter()
triples = [SimpleTriple("Drug_A", "treats", "Disease_X"),
           SimpleTriple("Drug_A", "targets", "Protein_Y"),
           SimpleTriple("Protein_Y", "involved_in", "Disease_X")]
rex, meta = adapter.from_triples(triples)
conf = adapter.subgraph_confidence(rex, [0, 1, 2])   # structural trust of a subgraph
enrich = adapter.to_enrichment_triples(rex, meta)    # write the analysis back as RDF
```


## The Hive

The hive is a multi-agent swarm orchestrated *as* a relational complex. Agents
("bees") are cells and every message is recorded into a live `agent_complex`, so
routing blends declared specialty with demonstrated interaction history: a cold
hive routes by specialty, a warm one routes by who has been carrying the work.

Bees have three roles (`queen | worker | embedder`) and join three ways:

```python
from agent.hive import Hive
hive = Hive()

# 1. attach: reference an existing OpenAI-compatible endpoint (llama.cpp, vLLM, ...)
hive.attach("queen", "http://127.0.0.1:8080", role="queen", specialties=["planning"])

# 2. spawn: launch and own a managed llama.cpp server for a bee
hive.spawn("coder", "/models/qwen-coder.gguf", role="worker", specialties=["code"])

# 3. add_worker: register ANY callable as a first-class worker (a trained NN, a
#    rexgraph analyzer, an embedder, a statistical model). Not an LLM, not an endpoint.
def analyze(data, **kw):
    from rexgraph.graph import RexGraph
    import numpy as np
    g = RexGraph.from_graph(np.array(data["src"]), np.array(data["tgt"]))
    return {"betti": list(g.betti), "coherence": float(np.mean(g.coherence))}
hive.add_worker("rex-analyzer", analyze, capability="analyze",
                specialties=["structure", "betti"], worker_type="analyzer:rexgraph")

# route, dispatch (route + ask a chat bee), invoke (run a worker capability)
hive.route("explain the homology")          # ranked bees with specialty + history scores
hive.dispatch("write a reversal function")  # -> {routed, bee, reply}
hive.invoke("rex-analyzer", {"src": [0,1,2], "tgt": [1,2,0]})
hive.status(); hive.monitor()               # membership, health, live routing
```

Worker capabilities are `generate` (chat), `predict`, `score`, `embed`,
`analyze`, `transform`. `dispatch` is the chat path and always routes to a
generate-capable bee; other capabilities are reached with `invoke`. This is the
same relational machinery the core uses, pointed outward: agents are cells,
messages are the signal, and the monitor is the structural self-test.

CLI: `rexgraph-hive` starts a hive, `rexgraph-local` manages llama.cpp servers.


## Lifecycle and Setups

A **setup** is a named configuration profile: the shared context (backend,
threads, model choices, paths) that operations run against. A **ComputeSpec**
inside the setup carries the backend and thread width, and resolves through
`rexgraph.compute` and `pick_device` so every operation runs on the same
dynamically selected device (CPU / CUDA / ROCm / MPS) with a live-usable-GPU
probe and a CPU fallback.

Operations are the platform's verbs, each run-logged:

| Operation | Does |
|-----------|------|
| `serve` | run the server + UI |
| `train` | train a model on a relational complex |
| `finetune` | LoRA fine-tune a language model |
| `ingest` | pull a source (DB, knowledge core) into a complex + trainable bundle |
| `pipeline` | run the full analysis pipeline over a corpus |
| `build` | assemble a deployable agent |
| `deploy` | generate a container bundle (Dockerfile, compose, entrypoint, config) |
| `test` | run the platform smoke tests |
| `bench` | benchmark optimizers / architectures |

```bash
rexgraph-ops serve                  # run an operation under the active setup
rexgraph-config                     # manage setups
rexgraph-deploy                     # generate a deployment bundle
```

Training, loading, inference, and saving all ride the resolved backend;
checkpoints are device-agnostic (loaded with `map_location` onto the picked
device), so a model trained on a GPU deploys unchanged to CPU.


## Local LLM inference

The runtime detects the llama.cpp server binary, reads the host hardware, and
serves quantized models locally: no cloud dependency.

```python
from agent import local_runtime as lr

lr.detect_hardware()        # backend, GPU, RAM, model budget (e.g. rocm, 96 GB usable)
lr.discover_local_models()  # GGUF + HF-cache models on disk
lr.recommend()              # tiered model picks that fit the budget (embedder/worker/queen)
lr.start("/models/qwen2.5-0.5b.gguf")  # spawn a server; probe_endpoints() finds it
```

`CATALOG` tiers models by role: an embedder for the swarm's alignment signal,
worker bees for triage and coding, queen models for reasoning. Chat returns
reliability metrics:

```python
from agent import chat_model
chat_model.configure(url="http://127.0.0.1:8080")
out = chat_model.generate_with_metrics("List three properties of a Hodge Laplacian.")
out["metrics"]              # perplexity, mean_surprisal, varentropy: the reliability signal
```

CLI: `rexgraph-local {status, start, stop, pull, recommend}`.


## Models: custom ML architectures

The model builder ships four archetypes, each a starting point that is fully
customizable through its config and trained with the `rexgraph.nn` substrate
(the HodgeAdam optimizer, relational/propagator attention, Green-resolvent
blocks):

| Archetype | Data | Notes |
|-----------|------|-------|
| `mlp` | tabular / vector | classification or regression |
| `cnn` | image | HodgeAdam's conditioning edge shows with `norm=False` |
| `hgnn` | hypergraph / higher-order relational | fiber-bundle advection + diffusion on the complex's signed orientation |
| `lm` | sequence / language | relational (propagator) or standard attention |

```python
from agent.models import list_archetypes, build, run

list_archetypes()                              # cnn / hgnn / lm / mlp with default configs
model = build("hgnn", feat_dim=16, n_classes=4, d_hid=32, n_layers=2)
run("train", archetype="hgnn", steps=200)      # train with HodgeAdam on the resolved backend
```

Because the archetypes are configs over composable `rexgraph.nn` blocks, custom
architectures are the norm, not the exception: vary depth, width, attention
type, orientation, and the optimizer, or assemble blocks directly. Ingesting a
knowledge core yields a labeled node-classification bundle on a relational
complex, which any archetype can train on:

```bash
# POST /api/v1/ml/ingest {triples, labels, train:true, archetype:"hgnn", steps:20}
# -> relational complex -> trainable bundle -> HGNN trained with HodgeAdam
```

CLI: `rexgraph-models {list, build, multistep, fusion}`.


## Connectors

Connectors turn any database into a relational complex, so the whole platform
(analysis, RAG, training) applies to data that already lives somewhere.

```bash
rexgraph-connect list                      # available connectors
rexgraph-connect read  <source>            # pull records
rexgraph-connect validate <source>
rexgraph-connect ingest <source>           # -> relational complex
```

Sources include SQL databases, warehouses (Snowflake / BigQuery / Redshift /
Databricks), documents, graph stores, and streams. Combined with the TrustGraph
adapter, any store becomes a knowledge core and any knowledge core becomes a
relational complex.


## Structural diagnostics

Every edge signal on a relational complex decomposes uniquely and exactly (Hodge
theorem) into three orthogonal components. The analysis pipeline reports these
plus the four-channel structure:

- **Gradient**: hierarchical structure from B1 alone; face-independent; the
  strength of directed chains.
- **Curl**: closed feedback loops from the face structure B2; flow circulating
  within stable triangles.
- **Harmonic**: conserved flow on unfilled cycles; unresolved structural tension.

Derived metrics: **dim(H)** (independent oscillatory modes, `= beta_1`),
**frustration** (harmonic content on the topology channel: tension from the
graph structure, not removable by faces), **coparticipation** (harmonic content
on the geometry channel: how well the face structure supports the dynamics),
**health ratio** (frustration / coparticipation; above 1.0 the structure works
against itself), and **sigma-asymmetry** (per-edge rigidity vs. responsiveness to
face changes).

```python
from agent.auto import auto_rex
from agent.pipeline import AnalysisPipeline

rex = auto_rex("edges.csv")          # or JSON, text, numpy array, DataFrame, triples
result = AnalysisPipeline(rex).run(depth="standard")
result["hodge"]["health_ratio"], result["hodge"]["dim_H"]
```


## Integrations

Integrations are one part of the platform. All soft-gate: they import and their
endpoints stay reachable even when their optional dependency is absent.

### TrustGraph knowledge cores + rex-RAG

The interoperability layer between knowledge graphs and relational complexes,
bidirectional. Standalone mode needs no running service; connected mode uses
`trustgraph-base>=2.4` against a live instance.

**Knowledge graph to relational complex, and back.** `from_triples` turns any
triples into a typed relational complex; `to_enrichment_triples` writes the
structural analysis (coherence, dominant channel, typed-channel weights) back as
RDF triples, so it interoperates with any triple store.

```python
from agent.integrations.trustgraph_adapter import TrustGraphAdapter, SimpleTriple
adapter = TrustGraphAdapter()                       # or url="http://localhost:8088/"
rex, meta = adapter.from_triples([SimpleTriple("Drug_A","treats","Disease_X"), ...])
triples = adapter.to_enrichment_triples(rex, meta)  # (Drug_A, rex:coherence, 0.71), ...
```

**rex-RAG.** Retrieval gets a *structural* trust metric, not just a similarity
score. `subgraph_confidence` / `query_confidence` return coherence (kappa) and
the typed channels (T/G/F/C) of a retrieved subgraph; `decompose_query_signal`
gives the Hodge decomposition of a query over the knowledge complex (gradient /
curl / harmonic, plus a beyond-pairwise fraction); `assess_query` plans a
retrieval: entity coverage, harmonic dimension, and an adjusted token budget.

```python
from agent.integrations.trustgraph_pipeline import TrustGraphPipeline
pipe = TrustGraphPipeline.standalone()
pipe.analyze_triples(triples)                       # -> EngineResult (rex + analysis)
pipe.query_confidence(["Drug_A", "Protein_Y"])      # kappa + typed-channel trust
pipe.decompose_query_signal(signal, rex=rex)        # Hodge decomposition of the query
adapter.assess_query(["Metformin", "mTOR"])         # coverage, dim_H, adjusted_tokens
```

Also: `health_snapshot`, `compare_flows`, `track_evolution` (stabilizing /
fragmenting / stable), and `as_mcp_tool_definitions()` for agent workflows.
Server routes under `/api/v1/trustgraph/*` and `/api/v1/ml/ingest`.

### HuggingFace

Analyze transformer attention as a relational complex: build a complex from
each layer's attention and measure chain-condition violation, Dirac equiweight
deviation, and channel specialization across depth. Requires `[huggingface]`.

```python
from agent.integrations.huggingface_analyzer import analyze_transformer, quick_attention_analysis
report = analyze_transformer(model_name="mistralai/Mistral-7B-v0.1", text="The cat sat.")
result = quick_attention_analysis(attention_matrix)   # pre-extracted weights, no model load
```

### LangChain

Tools that give any LangChain agent structural analysis during reasoning
(`RexAnalyzeTool`, `RexConfidenceTool`, and more), backed by a `RexGraph`.
Requires `[langchain]`.

```python
from agent.integrations.langchain_tools import RexAnalyzeTool
tool = RexAnalyzeTool(rex)          # a LangChain BaseTool over the complex
```

### LangGraph

Model an agent's state machine as a relational complex and read its Hodge
decomposition: gradient means progress, curl means a converging loop, harmonic
means stuck oscillation. Requires `[langgraph]`.

```python
from agent.integrations.langgraph_rex import RexStateGraph
rsg = RexStateGraph()
rsg.add_state("reason"); rsg.add_transition("retrieve", "reason", weight=1.0)
rsg.should_continue(harmonic_threshold=0.4)   # continue / stop / caution
```


## Server and API

The FastAPI app exposes ~151 endpoints. Session/analysis routes are under
`/api`, everything else under `/api/v1`.

| Group | Routes |
|-------|--------|
| sessions / analysis / explore / chat | run analyses, explore a complex, chat with metrics |
| upload / corpus / connectors | ingest data and databases |
| models / ml / pipeline | build, train, and run models and pipelines |
| ops / deploy / schema / rcdb / ontology | lifecycle, deployment, schema, the relational complex DB |
| agents / hive | the swarm |
| trustgraph / ocr / integrations | integrations |

```bash
rcf-server                          # start it; docs at /docs, UI at /
curl http://127.0.0.1:8000/api/health
```


## Deployment

Two stacks: local (one operator, one machine) and network (shared, reachable,
multi-user). Same server binary; what changes is auth, bind, and TLS.

**Local stack.** `rcf-server` (alias `rexgraph-ui`) serves the web UI and REST
API. `rexgraph-serve` is a separate model / OCR inference server, not the app.
Bind stays on loopback (`RCF_HOST=127.0.0.1`, the default); workers and models
run on the local hardware. For open solo dev, start with `RCF_ALLOW_INSECURE=1`
(auth off, a single local-admin identity, loopback only). Without that flag a
fresh server is secure by default (see Authentication).

```bash
RCF_ALLOW_INSECURE=1 rcf-server     # open local dev, loopback only
rexgraph-serve                      # separate model/OCR server (not the web app)
```

**Network stack.** The same server with auth on and per-workspace members,
reachable over the network.

- Put it behind TLS: supply a cert/key (`REXGRAPH_TLS_CERT` / `REXGRAPH_TLS_KEY`)
  or run `RCF_HTTPS=1` to self-sign.
- Binding a public interface (`RCF_HOST=0.0.0.0`) with auth off is refused by the
  bind guard; set `RCF_ALLOW_INSECURE=1` only if you mean it.
- Rate limiting is on by default, tiered per path (`RCF_RATE_LIMIT`,
  `RCF_RATE_LIMIT_AUTH`, `RCF_RATE_LIMIT_HEAVY`). The in-process limiter is
  per-worker; for multiple workers or HA, terminate TLS and rate-limit at the
  proxy.
- `rexgraph-deploy` generates a container bundle (Dockerfile, compose,
  entrypoint, config) for this stack.


## Authentication

Bearer tokens with per-workspace roles. A workspace is shared; each member holds
one token that carries their role in each workspace they belong to.

- **Roles are per workspace.** `admin` manages that workspace's members and runs
  its consequential actions; `user` reads and runs build verbs. The root
  workspace `default` is the instance: its admins enable auth and remove members
  entirely. A member can be admin of one workspace and a user of another.
- **Secure by default.** A fresh server (no `auth.json`) starts with auth ON and
  prints a bootstrap admin token once. `RCF_ALLOW_INSECURE=1` keeps open local
  dev; `REXGRAPH_ADMIN_TOKEN` sets the bootstrap token instead of a random one.
- **Stand up a network** in one step, then add members:

```bash
rexgraph-auth network-init                                   # first admin + recovery key + auth on
rexgraph-auth member add --name alice --role user            # add to 'default'
rexgraph-auth member add --name bob --role admin --workspace proj
rexgraph-auth member list --workspace proj
rexgraph-auth member revoke --name alice                     # from a workspace, or --all
```

- **Enforcement.** A user can PROPOSE a consequential action (a `kill`, say) but
  only an admin of that workspace executes it (with `confirm`). Every command and
  member change is recorded in the activity journal stamped with the acting
  identity, so the journal is the audit trail.
- **Turning auth off** is host-local and passphrase-gated, so a stolen token
  alone cannot open the server. `rexgraph-auth {status, enable, disable,
  passphrase, recover}`.


## Running in parallel (multi-core / multi-GPU)

The compute backend is chosen per run, and work fans out across the hardware you
have. Nothing here is a fixed thread count; it resolves from the machine.

- **Backend selection.** `rexgraph.compute` resolves a `ComputeSpec` (backend +
  thread width) through `pick_device`: CPU / CUDA / ROCm / MPS, with a live-GPU
  probe and a CPU fallback. Pin it in a setup so every operation runs on the same
  device (`rexgraph-config`), or let each run resolve `auto`.
- **Multi-core.** `parallel_map` fans kernel work across CPU cores with
  nested-parallelism-safe budgeting, so the threaded BLAS underneath is not
  oversubscribed. The heavy CPU wins are in sparsification and the shared moment
  engine, not blanket `prange`. Set the width with `compute.set_threads(n)` or
  read the machine with `compute.inventory()`.
- **Multi-GPU.** When more than one GPU is present (`compute.gpu_count()`,
  `compute.gpu_devices()`), the tower dispatches across them for work above
  `multi_gpu_min_work`; the matrix-free Chebyshev / block-CG kernels run on the
  selected device with 1e-15 parity against the CPU path.
- **Hive placement.** Deploy workers across CPU cores while models run on
  GPU / iGPU. Set a model's device in the foundry, or let it resolve; the
  coordinator model can sit on the iGPU while CPU workers handle triage.

```python
from agent.foundry import ModelFoundry
f = ModelFoundry(hive)
f.forge("detector", "mlp", device="cpu")        # a CPU-core worker
f.forge("ranker", "hgnn", device="cuda:0")      # a GPU worker
f.place_llm(hive, "coordinator", model_path, on="igpu")   # LLM on the iGPU
```

Checkpoints are device-agnostic (loaded with `map_location`), so a model trained
on a GPU deploys unchanged to CPU.


## CLI

| Command | Does |
|---------|------|
| `rcf-server` / `rexgraph-ui` | run the web app (UI + REST API) |
| `rexgraph-serve` | run the model / OCR inference server (not the web app) |
| `rexgraph-hive` | start and manage a hive |
| `rexgraph-local` | manage llama.cpp servers and models |
| `rexgraph-models` | list / build / train models |
| `rexgraph-ops` | run lifecycle operations |
| `rexgraph-connect` | database connectors |
| `rexgraph-run` | run the analysis pipeline |
| `rexgraph-deploy` | generate a deployment bundle |
| `rexgraph-setup` / `rexgraph-config` | setups and configuration |
| `rexgraph-auth` | server auth |
| `rexgraph-ocr` | OCR ingestion |
| `rexgraph-test` | platform smoke tests |


## License

Apache License 2.0
