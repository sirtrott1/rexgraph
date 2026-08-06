"""
agent.hive: swarm of local models orchestrated as a relational complex.

A queen (the main driver), worker bees, and an embedder bee, each an OpenAI-compatible endpoint
(a managed llama.cpp subprocess or an attached live server). Every bee-to-bee interaction is
recorded into the live agentic relational complex (`agent_complex`), so the monitor/graph populate
from real traffic.

Composition boundaries:
  - `local_runtime` owns process lifecycle (spawn/health/stop) and hardware detection.
  - `agent_complex` owns the analysis (load-bearing, coherence, Hodge, alignment, routing).
  - `hive` owns the swarm: which bees exist, their role/specialty, dispatching a query to the
    right bee, and relaying every message through the complex.

A bee is managed (this process spawned its llama-server) or attached (a live endpoint, e.g. an
Ollama or vLLM server). Routing is query-reweighting over the interaction history
(`agent_complex.route`) blended with each bee's declared specialties: a fresh hive routes by
specialty, a warm one routes by which bee has been carrying the relevant work.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import re
from dataclasses import dataclass, field

from agent import agent_complex

logger = logging.getLogger(__name__)

VALID_ROLES = ("queen", "worker", "embedder")


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]{3,}", str(text).lower())]


# common function words carry no content; dropping them makes lexical agreement reflect subject
# matter, so "shares nothing" becomes an exact structural signal rather than stopword-inflated. Not
# a tuned threshold, noise removal. (The embedding path does not need this.)
_STOPWORDS = frozenset(
    ["the", "and", "are", "for", "was", "were", "that", "this", "with", "from", "have", "has", "had", "not", "but", "all", "any", "can", "will", "would", "should", "could", "into", "onto", "off", "per", "via", "out", "over", "under", "near", "more", "most", "some", "such", "then", "than", "they", "them", "their", "there", "here", "what", "when", "where", "which", "who", "whom", "how", "why", "our", "your", "its", "his", "her", "about", "also", "been", "being", "does", "did", "done", "each", "other", "same", "only", "very", "just", "like"])


def _is_embed(name: str) -> bool:
    return bool(re.search(r"embed|nomic|bge|gte|e5|minilm", name.lower()))


def _default_rules():
    """Loaded lazily: hive_config imports hive in apply(), so a module-level import would cycle."""
    from agent.hive_config import load_specialty_rules
    return load_specialty_rules()


def _specialty_of(name: str, rules=None):
    """(bee-name base, specialty keywords) for a model name, from the specialty RULES.

    Rules come from config (agent.hive_config.load_specialty_rules), so teaching the hive a new
    model family is a config edit rather than a source edit. Passed explicitly rather than read
    from a global so a caller holding a profile can supply its own, and so this stays pure."""
    for rule in (_default_rules() if rules is None else rules):
        if rule.matches(name):
            return rule.base, list(rule.specialties)
    return None, []


def _worker_name(model_name: str, taken, rules=None) -> str:
    base, _ = _specialty_of(model_name, rules=rules)
    if base is None:
        m = re.match(r"[a-z0-9]+", model_name.lower())
        base = m.group(0) if m else "worker"
    name, i = base, 2
    while name in taken:
        name = f"{base}{i}"; i += 1
    return name


def plan_hive(models, budget_gb: float, *, headroom: float = 0.15,
              kv_factor: float = 1.15, max_workers: int = 4, rules=None) -> dict:
    """Choose a queen, worker bees, and an embedder that fit together within a memory budget,
    given the models detected on disk (local_runtime.discover_local_models). Spawns nothing;
    returns the plan.

    Only GGUF are spawnable via llama.cpp (transformers snapshots are skipped). The queen is the
    largest chat model that fits alone. Workers are the remaining chat models, smallest-first,
    added while the running footprint (file size * kv_factor, for KV-cache overhead) stays under
    the usable budget (budget * (1-headroom)). The cheapest embedder is always included.

    `rules` are the specialty rules used to label each bee; None loads them from config."""
    from agent.hive_config import GENERAL_SPECIALTIES
    rules = _default_rules() if rules is None else rules
    gguf = [m for m in models if m.get("format") == "gguf" and m.get("size_gb", 0) > 0.05]
    embeds = [m for m in gguf if _is_embed(m["name"])]
    chats = [m for m in gguf if not _is_embed(m["name"])]
    usable = max(budget_gb * (1.0 - headroom), 0.0)
    plan, used, taken = [], 0.0, set()

    queen = next((m for m in sorted(chats, key=lambda m: -m["size_gb"])
                  if m["size_gb"] * kv_factor <= usable), None)
    if queen is not None:
        _, spec = _specialty_of(queen["name"], rules=rules)
        plan.append({"name": "queen", "role": "queen", "path": queen["path"],
                     "model": queen["name"], "size_gb": queen["size_gb"],
                     "specialties": spec or ["general", "plan", "summarize"]})
        used += queen["size_gb"] * kv_factor
        taken.add("queen")

    for m in sorted(chats, key=lambda m: m["size_gb"]):
        if queen is not None and m["path"] == queen["path"]:
            continue
        if sum(1 for p in plan if p["role"] == "worker") >= max_workers:
            break
        if used + m["size_gb"] * kv_factor > usable:
            continue
        nm = _worker_name(m["name"], taken, rules=rules); taken.add(nm)
        _, spec = _specialty_of(m["name"], rules=rules)
        # a generalist worker is not a specialist in anything, but an EMPTY list makes it score 0
        # on every cold-hive routing query - unreachable until it somehow accrues history. The
        # queen already had this fallback; the worker branch did not.
        plan.append({"name": nm, "role": "worker", "path": m["path"], "model": m["name"],
                     "size_gb": m["size_gb"], "specialties": spec or list(GENERAL_SPECIALTIES)})
        used += m["size_gb"] * kv_factor

    if embeds:
        em = min(embeds, key=lambda m: m["size_gb"])
        plan.append({"name": "embedder", "role": "embedder", "path": em["path"],
                     "model": em["name"], "size_gb": em["size_gb"], "specialties": []})
        used += em["size_gb"]

    return {"plan": plan, "budget_gb": round(budget_gb, 1), "usable_gb": round(usable, 1),
            "planned_gb": round(used, 1), "n": len(plan),
            "note": ("no chat GGUF fits the budget" if queen is None and not plan else "")}


@dataclass
class Bee:
    """One member of the swarm. Managed bees are subprocesses this hive spawned and owns the
    lifecycle of; attached bees are pre-existing endpoints the hive only references."""
    name: str
    role: str                       # queen | worker | embedder
    url: str
    model: str = ""
    specialties: list[str] = field(default_factory=list)   # concept keywords for routing
    managed: bool = False           # did the hive spawn the process?
    pid: int | None = None
    port: int | None = None
    summary: dict | None = None   # model_io.model_summary (arch/params/dim/quant)
    capability: str = "generate"     # generate (chat) | predict | score | embed | analyze | transform
    worker_type: str = ""            # ':'-scoped kind for the worker-type ontology (e.g. model:mlp)
    _proc: object = None             # Popen for managed worker bees (not serialized)
    _handler: object = None          # local callable for non-chat workers (not serialized)

    def public(self) -> dict:
        return {"name": self.name, "role": self.role, "url": self.url, "model": self.model,
                "specialties": self.specialties, "managed": self.managed, "pid": self.pid,
                "port": self.port, "summary": self.summary, "capability": self.capability,
                "worker_type": self.worker_type, "local": self._handler is not None}


def _chat(url: str, model: str, prompt: str, system: str | None = None,
          max_tokens: int = 512, temperature: float = 0.3, timeout: float = 120.0) -> str | None:
    """Call one bee's OpenAI-compatible /v1/chat/completions. Returns the reply text, or None if
    unreachable or empty. Targets an explicit url so the call goes to a specific bee, not the
    globally-resolved chat backend."""
    try:
        import httpx
    except Exception:
        return None
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    payload: dict = {"messages": msgs, "max_tokens": max_tokens,
                     "temperature": temperature, "stream": False}
    if model:
        payload["model"] = model
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(url.rstrip("/") + "/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        text = "".join(ch.get("message", {}).get("content", "") for ch in data.get("choices", []))
        return text.strip() or None
    except Exception:
        return None


class Hive:
    """Swarm orchestrator. Bees are added (spawned or attached), queries are dispatched to the
    best-matching bee, and every message is recorded into the shared relational complex."""

    def __init__(self, name: str = "default"):
        self._bees: dict[str, Bee] = {}
        self.name = name
        # each hive IS its own coordination complex. The 'default' hive uses the process-wide live
        # complex (so the runtime and existing routes observe it); a named hive is isolated.
        self._own_complex = None
        if name != "default":
            from . import agent_complex
            self._own_complex = agent_complex.AgentComplex()

    @property
    def _complex(self):
        from . import agent_complex
        return agent_complex.get_live() if self._own_complex is None else self._own_complex

    # membership

    def attach(self, name: str, url: str, *, role: str = "worker",
               model: str = "", specialties=None) -> Bee:
        """Register a bee for an already-running endpoint (Ollama/vLLM/llama.cpp/etc). The hive
        references it but does not own its lifecycle. `role` must be queen|worker|embedder."""
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        bee = Bee(name=name, role=role, url=url.rstrip("/"), model=model,
                  specialties=list(specialties or []), managed=False)
        self._bees[name] = bee
        from . import activity as _act
        _act.record("worker:" + name, "attach", detail={"role": role, "model": model})
        return bee

    def add_worker(self, name: str, handler, *, capability: str = "predict",
                   specialties=None, worker_type: str = "", model: str = "") -> Bee:
        """Register any callable as a worker member. `handler(data, **kw)` runs the worker's
        capability on structured input. This is the general primitive: a trained NN, a statistical
        model, a rexgraph analyzer, an embedder, or any inference module becomes a first-class hive
        member, invoked with invoke() and monitored like any bee. `capability` is one of
        predict/score/embed/analyze/transform (generate is the chat path). `worker_type` names its
        kind for the worker-type ontology (':'-scoped, e.g. 'analyzer:stat:logreg'); it defaults to
        'worker:<capability>' so every worker is typed. No HTTP endpoint involved."""
        bee = Bee(name=name, role="worker", url="", model=model,
                  specialties=list(specialties or []), managed=False,
                  capability=capability, worker_type=worker_type or f"worker:{capability}",
                  _handler=handler)
        self._bees[name] = bee
        from . import activity as _act
        _act.record("worker:" + name, "deploy", detail={"type": bee.worker_type, "capability": capability})
        return bee

    def add_model(self, name: str, checkpoint, *, capability: str = "predict",
                  specialties=None, device: str = "cpu", worker_type: str = "") -> Bee:
        """Register a trained model as a worker. A convenience over add_worker whose handler runs
        agent.models.predict on a saved checkpoint (a path, or a (model, config) pair)."""
        from agent import models
        mc = models.load_checkpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        arch = mc[1].get("archetype", "")

        def handler(data, **kw):
            return models.predict(mc, data, device=device, **kw)

        return self.add_worker(name, handler, capability=capability, specialties=specialties,
                               worker_type=worker_type or f"model:{arch}", model=arch)

    def providers(self, capability: str) -> list[str]:
        """Names of worker members that provide a capability (predict/score/embed/analyze/
        transform/generate)."""
        return [b.name for b in self._bees.values()
                if b.capability == capability and b.role != "embedder"]

    def dispatch_capability(self, capability: str, data, *, hint: str = None,
                            sender: str = "user") -> dict:
        """Route a structured request to a worker providing `capability` and invoke it. When more
        than one provides it, `hint` (a query string) breaks ties by specialty overlap. Returns
        {worker, capability, result}. This is the structured-task analog of dispatch()."""
        cands = self.providers(capability)
        if not cands:
            raise ValueError(f"no worker provides capability {capability!r}")
        name = cands[0]
        if hint and len(cands) > 1:
            ht = set(_tokens(hint))
            name = max(cands, key=lambda n: len(
                ht & {t for s in self._bees[n].specialties for t in _tokens(s)}))
        return {"worker": name, "capability": capability,
                "result": self.invoke(name, data, sender=sender)}

    def type_complex(self):
        """Build the worker-type taxonomy as a relational complex via the ontology code
        (agent.ontology_complex). Worker types are ':'-scoped, so 'analyzer:stat:summary' subsumes
        up through 'analyzer:stat' to 'analyzer'; each worker is an instance of its type. Returns
        (rex, meta), or None when no worker declares a type. The result is the same object the
        Hodge/ontology diagnosis reads, so the type hierarchy is consistency-checkable (a harmonic
        subsumption cycle is an inconsistency) and routable like any complex."""
        from agent.ontology_complex import ontology_to_rex, parse_rdf
        triples, seen = [], set()
        for b in self._bees.values():
            wt = b.worker_type
            if not wt:
                continue
            parts = wt.split(":")
            for i in range(1, len(parts)):                 # subsumption chain up the ':' path
                child, parent = ":".join(parts[:i + 1]), ":".join(parts[:i])
                if (child, parent) not in seen:
                    triples.append((child, "rdfs:subClassOf", parent))
                    seen.add((child, parent))
            triples.append((b.name, "rdf:type", wt))       # the worker instance -> its type
        if not triples:
            return None
        return ontology_to_rex(parse_rdf(triples))

    def invoke(self, name: str, data, *, sender: str = "user", record: bool = True, **kw):
        """Call a non-chat worker (a local model/analyzer) on structured input and return its
        result (e.g. predictions). Like ask() for chat bees, the call is recorded into the live
        complex, so a model worker participates in the monitored swarm topology. Use ask() for a
        chat (generate) bee and invoke() for a predict/score/embed/analyze worker."""
        bee = self._bees.get(name)
        if bee is None:
            raise KeyError(f"no bee named {name!r}")
        if bee._handler is None:
            raise ValueError(f"bee {name!r} has no local handler; use ask() for a chat bee")
        if record:
            self.relay(sender, name, f"invoke:{bee.capability}")
        from . import activity as _act
        _use = _act.open_use(bee.model or name, "invoke:" + bee.capability, by="worker:" + name)
        try:
            result = bee._handler(data, **kw)
        finally:
            _act.close_use(_use)
        if record:
            self.relay(name, sender, f"result:{bee.capability}")
        return result

    def spawn(self, name: str, model_path: str, *, role: str = "worker", specialties=None,
              port: int | None = None, ctx_size: int | None = None,
              n_gpu_layers: int | None = None, wait: float = 90.0) -> Bee:
        """Bring a bee up as a managed llama.cpp subprocess. queen registers as the chat backend
        (chat/metrics run on it); embedder registers as the embedding endpoint (so the monitor's
        semantic signal is live); worker is an independent server the hive owns. Requires a built
        llama.cpp binary and the GGUF on disk (see `local_runtime`)."""
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        from agent import local_runtime as L
        if role == "queen":
            st = L.start(model_path, port=port, ctx_size=ctx_size, n_gpu_layers=n_gpu_layers, wait=wait)
            bee = Bee(name=name, role=role, url=st["url"], model=st.get("model", ""),
                      specialties=list(specialties or []), managed=True, pid=st.get("pid"),
                      summary=st.get("model_summary"))
        elif role == "embedder":
            st = L.start_embedder(model_path, port=port, wait=wait)
            bee = Bee(name=name, role=role, url=st["url"], model=st.get("model", ""),
                      specialties=list(specialties or []), managed=True, pid=st.get("pid"))
        else:
            proc, st = L.spawn_server(model_path, port=port, ctx_size=ctx_size,
                                      n_gpu_layers=n_gpu_layers, wait=wait)
            bee = Bee(name=name, role=role, url=st["url"], model=st.get("model", ""),
                      specialties=list(specialties or []), managed=True, pid=st.get("pid"),
                      summary=st.get("model_summary"), _proc=proc)
        self._bees[name] = bee
        return bee

    def remove(self, name: str) -> bool:
        """Stop (if managed) and unregister a bee."""
        bee = self._bees.pop(name, None)
        if bee is None:
            return False
        from . import activity as _act
        _act.record("worker:" + name, "remove", detail={"role": bee.role})
        if bee.managed:
            from agent import local_runtime as L
            if bee.role == "queen":
                L.stop()
            elif bee.role == "embedder":
                L.stop_embedder()
            elif bee._proc is not None:
                try:
                    bee._proc.terminate(); bee._proc.wait(timeout=10)
                except Exception:
                    with contextlib.suppress(Exception):
                        bee._proc.kill()
        return True

    def stop_all(self) -> None:
        for name in list(self._bees):
            self.remove(name)
        coord = getattr(self, "_coord", None)
        if coord is not None and coord.pools is not None:
            coord.pools.shutdown()
            from rexgraph import coordinator as _co
            _co.unregister_hive_share(self.name)
            self._coord = None

    def attach_live(self) -> list[Bee]:
        """Discover running inference servers (local_runtime.probe_endpoints) and attach any not
        already known, so the hive reflects whatever is serving on this host. An endpoint whose
        name/models look like an embedder is tagged as one; the first becomes queen if the hive
        has none yet."""
        from agent import local_runtime as L
        added = []
        known = {b.url for b in self._bees.values()}
        for ep in L.probe_endpoints():
            url = ep["url"].rstrip("/")
            if url in known:
                continue
            models = ep.get("models") or []
            first = (models[0] if models else "").lower()
            is_embed = ep.get("kind") == "ollama" and any("embed" in (m or "").lower() for m in models) \
                or "embed" in first or "nomic" in first or "bge" in first
            role = "embedder" if is_embed else ("queen" if not self.has_queen else "worker")
            name = ep.get("managed") or (models[0] if models else url)
            self.attach(name, url, role=role, model=(models[0] if models else ""))
            added.append(self._bees[name])
            known.add(url)
        return added

    # auto-composition: stand up the best hive that fits, from disk

    def auto_plan(self, budget_gb: float | None = None, **kw) -> dict:
        """Plan a hive (queen, workers, embedder) that fits, from the models on disk and the
        hardware budget. A dry run that spawns nothing. `budget_gb` defaults to the detected
        memory budget."""
        from agent import local_runtime as L
        if budget_gb is None:
            budget_gb = L.detect_hardware().get("model_budget_gb") or 8.0
        return plan_hive(L.discover_local_models(), budget_gb, **kw)

    def compose(self, plan, wait: float = 120.0) -> dict:
        """Spawn every bee in a plan (from `auto_plan`/`plan_hive`). Continues past a bee that
        fails to come up, recording the error, so one bad model does not stop the rest."""
        entries = plan.get("plan", plan) if isinstance(plan, dict) else plan

        def _spawn_one(e):
            try:
                b = self.spawn(e["name"], e["path"], role=e["role"],
                               specialties=e.get("specialties") or [], wait=wait)
                return {"name": b.name, "role": b.role, "url": b.url, "ok": True}
            except Exception as ex:
                return {"name": e.get("name"), "role": e.get("role"), "ok": False, "error": str(ex)}

        # id is index-prefixed so two plan entries with the same name cannot collide into one wave
        # slot (which would silently drop a spawn).
        tasks = [{"id": f"{i}:{e.get('name') or ''}", "kind": "spawn",
                  "fn": functools.partial(_spawn_one, e),
                  "weight": self._task_weight("spawn", str(e.get("name") or ""))}
                 for i, e in enumerate(entries)]
        wave = self._run_wave(tasks)
        # _run_wave omits the id of any task whose fn raises; _spawn_one never raises (fully
        # wrapped in try/except above), but read defensively so a skipped id cannot KeyError.
        results = [wave.get(f"{i}:{e.get('name') or ''}") or
                   {"name": e.get("name"), "role": e.get("role"), "ok": False,
                    "error": "task skipped by coordinator wave"}
                   for i, e in enumerate(entries)]
        return {"spawned": results, "status": self.status()}

    def auto(self, budget_gb: float | None = None, wait: float = 120.0, **kw) -> dict:
        """Plan and stand up the best hive that fits on this machine, from the models on disk."""
        plan = self.auto_plan(budget_gb, **kw)
        if not plan["plan"]:
            return {"plan": plan, "spawned": [], "status": self.status(),
                    "note": plan.get("note") or "no spawnable models found on disk"}
        return {"plan": plan, **self.compose(plan, wait=wait)}

    # accessors

    def bees(self) -> list[Bee]:
        return list(self._bees.values())

    def get(self, name: str) -> Bee | None:
        return self._bees.get(name)

    @property
    def has_queen(self) -> bool:
        return any(b.role == "queen" for b in self._bees.values())

    @property
    def queen(self) -> Bee | None:
        return next((b for b in self._bees.values() if b.role == "queen"), None)

    @property
    def embedder(self) -> Bee | None:
        return next((b for b in self._bees.values() if b.role == "embedder"), None)

    def workers(self) -> list[Bee]:
        return [b for b in self._bees.values() if b.role == "worker"]

    # every interaction is recorded into the relational complex

    def relay(self, sender: str, recipient: str, text: str, **meta):
        """Record one bee-to-bee (or user-to-bee) message into the live agentic complex. This is
        what makes the monitor/graph reflect real swarm traffic. Call it wherever bees message."""
        self._complex.add_message(sender, recipient, text, **meta)

    def ask(self, name: str, prompt: str, *, sender: str = "user", system: str | None = None,
            max_tokens: int = 512, record: bool = True) -> str | None:
        """Send a prompt to one bee, return its reply, and by default record both directions of
        the exchange into the complex so the interaction is part of the monitored structure."""
        bee = self._bees.get(name)
        if bee is None:
            raise KeyError(f"no bee named {name!r}")
        if bee._handler is not None:
            raise ValueError(f"bee {name!r} is a {bee.capability!r} worker; use invoke(), not ask()")
        if record:
            self.relay(sender, name, prompt)
        from . import activity as _act
        _use = _act.open_use(bee.model or name, "ask", by="worker:" + name)   # tracks concurrent use
        try:
            reply = _chat(bee.url, bee.model, prompt, system=system, max_tokens=max_tokens)
        finally:
            _act.close_use(_use)
        if record and reply:
            self.relay(name, sender, reply)
        return reply

    # routing: query-reweighting blended with declared specialty

    def route(self, query: str, top_k: int = 3) -> list[dict]:
        """Rank bees for a query. Blends the interaction-history relevance (agent_complex.route,
        which bee has been carrying this kind of work) with each bee's declared specialty overlap,
        so a cold hive routes by specialty and a warm one routes by demonstrated load. The queen
        is the fallback so a query always has a home."""
        qt = set(_tokens(query))
        # specialty score per bee
        spec = {}
        for b in self._bees.values():
            if b.role == "embedder":
                continue
            st = set(t for s in b.specialties for t in _tokens(s))
            spec[b.name] = (len(qt & st) / (len(st) ** 0.5)) if st else 0.0
        # history score from the live complex (names that match bee names)
        hist = {r["agent"]: r["relevance"] for r in agent_complex.get_live().route(query, top_k=50)}
        ranked = []
        for b in self._bees.values():
            if b.role == "embedder":
                continue
            score = 0.6 * spec.get(b.name, 0.0) + 0.4 * hist.get(b.name, 0.0)
            ranked.append({"bee": b.name, "role": b.role,
                           "score": round(score, 3),
                           "specialty": round(spec.get(b.name, 0.0), 3),
                           "history": round(hist.get(b.name, 0.0), 3)})
        ranked.sort(key=lambda x: -x["score"])
        top = [r for r in ranked if r["score"] > 0][:top_k]
        if not top and self.queen is not None:                # always route to some bee
            top = [{"bee": self.queen.name, "role": "queen", "score": 0.0,
                    "specialty": 0.0, "history": 0.0, "fallback": True}]
        return top

    def dispatch(self, query: str, *, sender: str = "user", system: str | None = None,
                 record: bool = True) -> dict:
        """Route a query to the best bee and ask it, in one call. Returns {routed, bee, reply}.

        dispatch is the CHAT path (it calls ask()), so among the ranked bees it picks the
        highest-scoring GENERATE-capable one - analyze/predict/score/embed/transform workers
        are invoked with invoke(), not ask(), and a query can legitimately rank such a worker
        first (e.g. a topology question matching a rexgraph analyzer's specialties). The queen
        backs a query with no generate-capable specialist match."""
        ranked = self.route(query)
        chosen = None
        for r in ranked:
            b = self._bees.get(r["bee"])
            if b is not None and b.capability == "generate":
                chosen = r["bee"]
                break
        if chosen is None and self.queen is not None and self.queen.capability == "generate":
            chosen = self.queen.name
        if chosen is None:
            return {"routed": ranked, "bee": None, "reply": None,
                    "note": "no generate-capable bee for this query; use invoke() for "
                            "analyze/predict/score/embed/transform workers"}
        reply = self.ask(chosen, query, sender=sender, system=system, record=record)
        return {"routed": ranked, "bee": chosen, "reply": reply}

    # collaboration: dynamic delegation that breaks circular-wait deadlocks structurally

    _HANDOFF_RE = re.compile(r"^\s*HANDOFF\s+([A-Za-z0-9_\-]+)\s*:\s*(.+)", re.I | re.S)

    def _generate_bees(self) -> list[Bee]:
        return [b for b in self._bees.values() if b.capability == "generate"]

    def _coordinator_obj(self):
        """Lazily build this hive's Coordinator with a managed LanePools sized from the active setup
        and this hive's resource share. Cached on the instance."""
        if getattr(self, "_coord", None) is None:
            from rexgraph import coordinator as _co

            from .hive_config import coordinator_settings
            cs = coordinator_settings()
            share_frac = 1.0
            if cs.hive_shares:
                _co.register_hive_share(self.name, cs.hive_shares.get(self.name, 1.0))
                share_frac = _co.share_fraction(self.name)
                # backstop: drop this hive's share from the registry if the hive is garbage-collected
                # without stop_all (the finalizer holds only the name string, not the hive).
                import weakref
                weakref.finalize(self, _co.unregister_hive_share, self.name)
            import os as _os
            cap = _co.capacity(share_frac)
            budget = max(1, int((_os.cpu_count() or 8) * share_frac))   # core share -> inner-thread budget
            pools = _co.LanePools(self.name, idle_ttl_proc=cs.idle_ttl_proc,
                                  idle_ttl_thread=cs.idle_ttl_thread, affinity=cs.affinity,
                                  cap=cap, cores_budget=budget)
            self._coord = _co.Coordinator(pools=pools, cap=cap)
        return self._coord

    def _task_weight(self, kind: str, worker: str = "") -> float:
        from .hive_config import coordinator_settings
        cs = coordinator_settings()
        return float(cs.task_weights.get(kind, 1.0)) * float(cs.worker_weights.get(worker, 1.0))

    def _run_wave(self, tasks: list) -> dict:
        """Dispatch a wave of {id, kind, fn, weight?} tasks through the coordinator, folding timings
        into its cost model. NEVER raises: on any failure (or when the coordinator is disabled) it
        runs the fns serially. Emits a journal event with placement and timings (no content)."""
        from . import activity
        from .coordinator_adapter import work_units
        from .hive_config import coordinator_settings

        def _serial(ts: list) -> dict:
            # per-task try/except so one bad fn cannot sink the rest of the wave, and cannot make
            # _run_wave itself raise: that is the whole point of the fallback.
            out = {}
            for t in ts:
                try:
                    out[t["id"]] = t["fn"]()
                except Exception as ex:
                    logger.warning("hive task '%s' failed in serial run: %s", t["id"], ex)
            return out

        cs = coordinator_settings()
        if not cs.enabled or not tasks:
            return _serial(tasks)
        try:
            units = work_units(tasks)
            co = self._coordinator_obj()
            placement = co.plan(units)
            results = co.pools.run(units, placement, cost=co.cost)
            with contextlib.suppress(Exception):
                activity.record("coordinator", "wave", scope=self.name,
                                detail={"n": len(tasks),
                                        "lanes": {u["id"]: placement[u["id"]] for u in units},
                                        "kinds": {t["id"]: t.get("kind", "") for t in tasks}})
            return results
        except Exception as ex:
            logger.warning("coordinator wave failed, serial fallback: %s", ex)
            return _serial(tasks)

    def _coordination_loops(self) -> int:
        """First Betti number of the live inter-agent complex = number of coordination cycles.
        Cheap (integer rank), so it can be checked after every hand-off."""
        rex = self._complex.interaction_complex()[0]
        return int(rex.betti[1]) if rex is not None else 0

    def _resolve_target(self, name: str, subreq: str) -> str | None:
        """Map a hand-off target to a real generate-capable bee: exact name, else route by the
        request, else the queen."""
        b = self._bees.get(name)
        if b is not None and b.capability == "generate":
            return b.name
        for r in self.route(subreq):
            cand = self._bees.get(r["bee"])
            if cand is not None and cand.capability == "generate":
                return cand.name
        return self.queen.name if self.queen and self.queen.capability == "generate" else None

    def _deadlock_breaker(self, cycle_bees) -> str:
        """Pick a generate-capable bee OUTSIDE the stalled cycle (a fresh perspective), preferring
        the queen; fall back to the queen or any generate-capable bee."""
        cyc = set(cycle_bees)
        q = self.queen
        if q is not None and q.capability == "generate" and q.name not in cyc:
            return q.name
        outside = [b for b in self._generate_bees() if b.name not in cyc]
        if outside:
            return outside[0].name
        return q.name if (q is not None and q.capability == "generate") else self._generate_bees()[0].name

    def collaborate(self, task: str, *, sender: str = "user", max_hops: int = 8,
                    max_tokens: int = 512) -> dict:
        """Solve a task by dynamic delegation, breaking circular-wait deadlocks the instant they form.

        Each bee either answers or hands off with ``HANDOFF <name>: <request>``. Only the bee->bee
        hand-offs are recorded, so the live complex's first Betti number is exactly the count of
        coordination loops. When a hand-off closes a cycle (planner waits on coder waits on reviewer
        waits on planner), b1 rises; rather than loop until a timeout, the hive re-routes to a bee
        outside the cycle and forces a direct resolution. Returns the answer, the hand-off trail, and
        whether a deadlock was broken and where. A pairwise ping-pong (A<->B) is a single undirected
        edge, not a loop; the detector fires on genuine 3+-agent circular waits."""
        HANDOFF_SYS = ("You are one member of a team solving a task. If and only if you truly need a "
                       "teammate's input, reply with exactly `HANDOFF <name>: <what you need>` as the "
                       "entire reply. Otherwise give your best complete answer directly.")
        gen = self._generate_bees()
        if not gen:
            return {"answer": None, "trail": [], "deadlock_broken": False,
                    "note": "no generate-capable bee to collaborate"}

        first = next((r["bee"] for r in self.route(task)
                      if self._bees.get(r["bee"]) in gen), None)
        first = first or (self.queen.name if self.queen else gen[0].name)

        trail, involved = [], []
        current, prompt = first, task
        self.relay(sender, current, task)                     # user -> first bee (a pendant, no cycle)
        prev_b1 = self._coordination_loops()

        for hop in range(max_hops):
            reply = self.ask(current, prompt, system=HANDOFF_SYS, max_tokens=max_tokens, record=False)
            trail.append({"bee": current, "reply": reply})
            if current not in involved:
                involved.append(current)
            m = self._HANDOFF_RE.match(reply or "")
            if not m:                                          # a direct answer -> done
                return {"answer": reply, "bee": current, "trail": trail,
                        "deadlock_broken": False, "hops": hop + 1}
            subreq = m.group(2).strip()
            target = self._resolve_target(m.group(1).strip(), subreq)
            if target is None:
                return {"answer": reply, "bee": current, "trail": trail,
                        "deadlock_broken": False, "hops": hop + 1, "note": "hand-off target unresolved"}
            self.relay(current, target, subreq)                # the bee->bee hand-off edge
            b1 = self._coordination_loops()
            if b1 > prev_b1:                                   # a coordination cycle just closed
                cycle_bees = involved + ([target] if target not in involved else [])
                breaker = self._deadlock_breaker(cycle_bees)
                brief = "\n".join(f"- {t['bee']}: {(t['reply'] or '')[:160]}" for t in trail)
                answer = self.ask(breaker,
                    f"The team hit a circular hand-off and stalled. Transcript:\n{brief}\n\n"
                    f"Original task: {task}\n\nResolve it now with a direct, complete answer.",
                    system="Do not delegate or hand off. Break the deadlock and answer directly.",
                    max_tokens=max_tokens, record=False)
                trail.append({"bee": breaker, "reply": answer, "broke_deadlock": True})
                return {"answer": answer, "bee": breaker, "trail": trail, "deadlock_broken": True,
                        "cycle_at_hop": hop + 1, "cycle_bees": cycle_bees, "hops": hop + 1}
            prev_b1 = b1
            current, prompt = target, subreq

        return {"answer": trail[-1]["reply"] if trail else None, "bee": current, "trail": trail,
                "deadlock_broken": False, "hops": max_hops, "exhausted": True}

    # consensus: aggregate several workers by the STRUCTURE of their agreement

    def _answer_vectors(self, texts, embed_fn):
        """Vectorize answers for the agreement complex: semantic embeddings if an embedder is
        available, else lexical concept-count vectors. Returns an (n, d) array."""
        import numpy as np
        if embed_fn is not None:
            E = np.asarray(embed_fn(texts), dtype=np.float64)
            if E.ndim == 2 and E.shape[0] == len(texts) and E.shape[1] > 0:
                return E
        toks = [[w for w in _tokens(t or "") if w not in _STOPWORDS] for t in texts]
        vocab = sorted({w for ts in toks for w in ts})
        vi = {w: i for i, w in enumerate(vocab)}
        V = np.zeros((len(texts), max(len(vocab), 1)))
        for i, ts in enumerate(toks):
            for w in ts:
                V[i, vi[w]] += 1.0
        return V

    def consensus(self, query: str, *, k: int = 3, workers=None, embed: bool = False,
                  max_tokens: int = 512) -> dict:
        """Answer a query by agreement, not by a single worker's word.

        Fans the query to several workers (an explicit ``workers`` list, else the top-k routed
        generate-capable bees, else one bee sampled k times), builds the agreement complex from
        their answers (embedding cosine when an embedder bee is present, else lexical), and returns
        the coherent consensus answer plus a reliability score = how tightly the consensus cluster
        agreed. The divergent worker, the one with low agreement with the rest, is flagged as the likely
        hallucination and dropped from the answer. With embeddings this separates a genuine
        hallucination from a topically-distinct specialist, which a flat majority vote cannot."""
        import numpy as np
        gen = self._generate_bees()
        if not gen:
            return {"answer": None, "reliability": 0.0, "responders": [], "flagged": [],
                    "note": "no generate-capable bee"}

        if workers:
            names = [w for w in workers if w in self._bees and self._bees[w].capability == "generate"]
        else:
            names = [r["bee"] for r in self.route(query)
                     if self._bees.get(r["bee"]) is not None
                     and self._bees[r["bee"]].capability == "generate"][:k]

        answers = {}
        if len(names) >= 2:
            ask_tasks = [{"id": name, "kind": "ask",
                          "fn": functools.partial(self.ask, name, query, max_tokens=max_tokens),
                          "weight": self._task_weight("ask", name)} for name in names]
            answers = self._run_wave(ask_tasks)
        else:                                               # one specialist: sample it k times
            solo = names[0] if names else (self.queen.name if self.queen else gen[0].name)
            ask_tasks = [{"id": f"{solo}#{i + 1}", "kind": "ask",
                          "fn": functools.partial(self.ask, solo, query, max_tokens=max_tokens),
                          "weight": self._task_weight("ask", solo)} for i in range(max(k, 2))]
            answers = self._run_wave(ask_tasks)

        if not answers:      # every worker errored: the wave dropped them all
            return {"answer": None, "reliability": 0.0, "responders": [], "flagged": [],
                    "n_workers": 0, "note": "no worker responded"}

        labels = list(answers.keys())
        embed_fn = agent_complex.model_embed_fn() if embed else None
        V = self._answer_vectors([answers[l] for l in labels], embed_fn)
        Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        S = Vn @ Vn.T
        n = len(labels)
        avg = np.array([(S[i].sum() - S[i, i]) / (n - 1) if n > 1 else 1.0 for i in range(n)])
        # divergence without a magic cutoff: a data-adaptive Tukey lower fence on the agreement
        # distribution when there are enough workers (the same principled fence the schema linter
        # uses), else the exact structural signal: a worker whose answer shares nothing with the
        # group (orthogonal, agreement ~ 0). _ZERO is a numerical zero, not a policy threshold.
        _ZERO = 1e-9
        if n >= 4:
            q1, q3 = np.percentile(avg, [25, 75])
            fence = q1 - 1.5 * (q3 - q1)
            flagged = [labels[i] for i in range(n) if avg[i] < fence]
        else:
            flagged = [labels[i] for i in range(n) if n > 1 and avg[i] <= _ZERO]
        keep = [i for i in range(n) if labels[i] not in flagged]

        if len(keep) > 1:
            reliability = float(np.mean([S[i, j] for a, i in enumerate(keep) for j in keep[a + 1:]]))
            rep = max(keep, key=lambda i: sum(S[i, j] for j in keep if j != i))
        elif keep:
            reliability, rep = float(avg[keep[0]]), keep[0]
        else:
            reliability, rep = 0.0, int(np.argmax(avg))

        # per-worker STRUCTURAL reliability: build each answer's own complex and read its varentropy
        # gap (the eigen-free H2-H3 collision-entropy reliability flag). Independent of agreement -
        # it flags an internally incoherent answer, not just an odd one out.
        struct = {}
        try:
            from .hive_tasks import structural_of
            metric_tasks = [{"id": l, "kind": "analysis",
                             "fn": functools.partial(structural_of, answers[l] or ""),
                             "weight": self._task_weight("analysis")} for l in labels]
            struct = self._run_wave(metric_tasks)
        except Exception:
            struct = {}

        responders = []
        for i in range(n):
            sm = struct.get(labels[i]) or {}
            responders.append({"worker": labels[i], "agreement": round(float(avg[i]), 3),
                               "reliable": sm.get("reliable"),
                               "varentropy_gap": sm.get("varentropy_gap"),
                               "flag": "divergent" if labels[i] in flagged else "ok"})
        responders.sort(key=lambda d: -d["agreement"])
        return {"answer": answers[labels[rep]], "by": labels[rep],
                "reliability": round(reliability, 3), "responders": responders,
                "flagged": flagged, "n_workers": n}

    def guarded_ask(self, name: str, prompt: str, guard, *, retries: int = 1,
                    autofix: bool = True, max_tokens: int = 512) -> dict:
        """Ask a bee, then run a validity guard over the reply. On a violation, re-ask once with a
        correction note; if it still violates and ``autofix`` is set, apply the guard's fix. Returns
        {reply, violations, corrected, ...}. This is the 'guard bee' pattern: a worker that checks
        another's output against known rules and ensures it gets fixed."""
        reply = self.ask(name, prompt, max_tokens=max_tokens)
        violations = guard.check(reply)
        if not violations:
            return {"reply": reply, "violations": [], "corrected": False}
        if retries > 0:
            note = "; ".join(sorted({v["message"] for v in violations}))
            retry = self.ask(name, f"{prompt}\n\nRevise your answer to fix: {note}", max_tokens=max_tokens)
            if len(guard.check(retry)) < len(violations):
                return {"reply": retry, "violations": guard.check(retry),
                        "corrected": True, "method": "regenerated"}
        if autofix:
            fixed, found = guard.fix(reply)
            return {"reply": fixed, "violations": guard.check(fixed),
                    "corrected": True, "method": "autofixed", "original_violations": found}
        return {"reply": reply, "violations": violations, "corrected": False}

    # status + monitor

    def health(self, bee: Bee, timeout: float = 0.4) -> bool:
        try:
            import httpx
            r = httpx.get(bee.url + "/v1/models", timeout=timeout)
            return r.status_code < 500
        except Exception:
            return False

    def status(self, check_health: bool = False) -> dict:
        """Report every bee with role/url/model, whether there is a queen and embedder, and
        optionally a live health probe per bee."""
        bees = []
        for b in self._bees.values():
            d = b.public()
            if check_health:
                d["alive"] = self.health(b)
            bees.append(d)
        bees.sort(key=lambda d: (d["role"] != "queen", d["role"] != "worker", d["name"]))
        out = {"n_bees": len(bees),
               "queen": self.queen.name if self.queen else None,
               "embedder": self.embedder.name if self.embedder else None,
               "workers": [b.name for b in self.workers()],
               "bees": bees}
        coord = getattr(self, "_coord", None)
        try:
            from .hive_config import coordinator_settings
            cs = coordinator_settings()
            out["coordinator"] = {
                "enabled": cs.enabled,
                "pools": coord.pools.status() if (coord and coord.pools) else
                         {"proc": {"state": "cold"}, "thread": {"state": "cold"}},
                "priorities": {"task_weights": cs.task_weights,
                               "worker_weights": cs.worker_weights,
                               "hive_shares": cs.hive_shares},
            }
        except Exception:
            pass
        return out

    def monitor(self, embed: bool = False, track: bool = False) -> dict:
        """Run the relational-complex monitor over the swarm's traffic (the same live complex the
        hive records into). `embed=True` uses the embedder bee for the semantic alignment signal.
        `track=True` snapshots the drift tracker so repeated calls over time expose which worker is
        starting to detract (a rising-curvature / falling-alignment trend)."""
        fn = agent_complex.model_embed_fn() if embed else None
        out = self._complex.monitor(embed_fn=fn)
        if track:
            d = agent_complex.get_drift().snapshot(out)
            out["drift"] = {"drifting": d.drifting(), "strain_trend": d.strain_trend(),
                            "trends": d.trends()}
        return out

    def snapshot(self) -> dict:
        """The hive as one unified structure: the worker roster (type + capability), the worker-type
        complex, and the live monitor. Model, memory, and topology as a single relational structure."""
        return {
            "workers": [{"name": b.name, "role": b.role, "capability": b.capability,
                         "worker_type": b.worker_type, "model": b.model,
                         "local": b._handler is not None} for b in self._bees.values()],
            "type_complex": self.type_complex(),
            "monitor": self.monitor(),
        }

    def persist(self, store=None, *, name: str = "hive") -> str | None:
        """Catalogue the hive's worker-type structure in the RCDB by structural signature, so the
        hive is a first-class stored complex (model = memory = database, queryable by topology).
        `store` is an open RCStore or an RCDB uri; omit it to use `rcdb.default_store()` (persistent,
        REXGRAPH_RCDB_URI). Pass memory:// only when a throwaway store is what you want. Returns the record
        id, or None when no typed worker structure exists yet."""
        tc = self.type_complex()
        if tc is None:
            return None
        rex, _meta = tc
        from agent.rcdb import default_store, open_store
        if store is None:
            st = default_store()
        else:
            st = open_store(store) if isinstance(store, str) else store
        roster = [{"name": b.name, "capability": b.capability, "worker_type": b.worker_type}
                  for b in self._bees.values()]
        caps = sorted({b.capability for b in self._bees.values()})
        st.put(name, rex, meta={"kind": "hive", "workers": roster, "n_workers": len(self._bees)},
               tags=["hive", *caps])
        return name


# get_hive() returns the 'default' hive of the process-wide hive network (agent.hive_network),
# so single-hive callers are unchanged while named hives become available.

def get_network():
    """The process-wide hive network: the registry of named hives."""
    from .hive_network import get_network as _gn
    return _gn()


def get_hive() -> Hive:
    """The 'default' hive (get-or-create). Same object every call, as before."""
    return get_network().hive("default")


def reset_hive() -> None:
    """Reset the default hive so the next get_hive() is fresh."""
    get_network().reset("default")


def reset_network() -> None:
    """Drop every hive in the network."""
    from .hive_network import reset_network as _rn
    _rn()


def main(argv=None):
    """CLI: `python -m agent.hive <status|up|down|attach|route|ask>`."""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="rexgraph-hive", description=(
        "The agent hive: a swarm of local models as a relational complex."))
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="bring bees up (spawn from paths and/or attach live endpoints)")
    up.add_argument("--queen", help="GGUF path for the queen (main driver)")
    up.add_argument("--worker", action="append", default=[], metavar="NAME=PATH[:specialty,..]",
                    help="a worker bee, repeatable")
    up.add_argument("--embedder", help="GGUF path for the embedder bee")
    up.add_argument("--attach-live", action="store_true", help="attach any running inference servers")
    up.add_argument("--auto", action="store_true",
                    help="auto-compose the best queen+workers+embedder that fit, from models on disk")
    up.add_argument("--recommend", action="store_true",
                    help="print the auto-compose plan and exit (spawn nothing)")
    up.add_argument("--budget", type=float, default=None, help="override the memory budget (GB)")
    st = sub.add_parser("status", help="show the hive"); st.add_argument("--health", action="store_true")
    sub.add_parser("down", help="stop all managed bees")
    at = sub.add_parser("attach", help="attach a live endpoint as a bee")
    at.add_argument("name"); at.add_argument("url")
    at.add_argument("--role", default="worker"); at.add_argument("--spec", default="")
    rt = sub.add_parser("route", help="rank bees for a query"); rt.add_argument("query")
    ak = sub.add_parser("ask", help="ask one bee"); ak.add_argument("name"); ak.add_argument("prompt")
    sub.add_parser("profiles", help="list hive setups (presets + your saved profiles)")
    us = sub.add_parser("use", help="switch to a setup: bring the hive up per a profile (spawns models)")
    us.add_argument("profile")
    ac = sub.add_parser("activate", help="set the active setup only - no spawning, instant")
    ac.add_argument("profile")
    a = ap.parse_args(argv)
    hive = get_hive()

    if a.cmd == "profiles":
        from agent import hive_config
        s = hive_config.get_store(); act = s.active_id()
        for p in s.list():
            mark = "*" if p.id == act else " "
            kind = "preset" if p.builtin else "saved"
            print(f" {mark} {p.id:12s} [{kind}]  {p.name} - compose={p.compose}"
                  + (f" budget={p.budget_gb}GB" if p.budget_gb else ""))
        print("\n* = active. `hive use <id>` to switch."); return
    if a.cmd == "activate":
        from agent import hive_config
        s = hive_config.get_store()
        if s.get(a.profile) is None:
            print(f"no setup '{a.profile}' (see `hive profiles`)"); return
        s.set_active(a.profile)
        p = s.get(a.profile)
        print(f"active setup: {a.profile}  (optimizer={p.optimizer}, attention={p.attention}) - no models spawned")
        return
    if a.cmd == "use":
        from agent import hive_config
        res = hive_config.get_store().apply(a.profile)
        print(f"applied '{res['profile']}' (compose={res['compose']})")
        for s in res.get("spawned", []):
            print(("  up " if s.get("ok") else "  FAILED ") + str(s.get("name"))
                  + ("" if s.get("ok") else f" - {s.get('error')}"))
        for n in res.get("attached", []):
            print(f"  attached {n}")
        print(json.dumps(res["status"], indent=2)); return

    if a.cmd == "up":
        if a.recommend:
            plan = hive.auto_plan(a.budget)
            print(f"budget {plan['budget_gb']}GB (usable {plan['usable_gb']}GB) -> "
                  f"plan {plan['planned_gb']}GB, {plan['n']} bee(s):")
            for e in plan["plan"]:
                sp = (" · " + ",".join(e["specialties"])) if e["specialties"] else ""
                print(f"  {e['role']:8s} {e['name']:12s} ~{e['size_gb']:>5.1f}GB  {e['model']}{sp}")
            if not plan["plan"]:
                print("  (nothing fits - pull a model or lower --budget)")
            return
        if a.auto:
            res = hive.auto(a.budget)
            for s in res.get("spawned", []):
                print(("up " if s["ok"] else "FAILED ")
                      + f"{s['role']}: {s['name']}" + ("" if s["ok"] else f" - {s['error']}"))
        if a.attach_live:
            for b in hive.attach_live():
                print(f"attached {b.role}: {b.name} @ {b.url}")
        if a.queen:
            b = hive.spawn("queen", a.queen, role="queen"); print(f"queen up: {b.name} @ {b.url}")
        for w in a.worker:
            nm, _, rest = w.partition("=")
            path, _, spec = rest.partition(":")
            b = hive.spawn(nm, path, role="worker",
                           specialties=[s for s in spec.split(",") if s])
            print(f"worker up: {b.name} @ {b.url}")
        if a.embedder:
            b = hive.spawn("embedder", a.embedder, role="embedder"); print(f"embedder up: {b.name} @ {b.url}")
        print(json.dumps(hive.status(), indent=2)); return
    if a.cmd == "status":
        print(json.dumps(hive.status(check_health=a.health), indent=2)); return
    if a.cmd == "down":
        hive.stop_all(); print("all managed bees stopped"); return
    if a.cmd == "attach":
        b = hive.attach(a.name, a.url, role=a.role,
                        specialties=[s for s in a.spec.split(",") if s])
        print(f"attached {b.role}: {b.name} @ {b.url}"); return
    if a.cmd == "route":
        print(json.dumps(hive.route(a.query), indent=2)); return
    if a.cmd == "ask":
        reply = hive.ask(a.name, a.prompt)
        print(reply or "(no reply - is the bee reachable?)"); return


if __name__ == "__main__":
    main()
