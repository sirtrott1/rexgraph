"""
agent.agent_complex: the agentic relational complex and the monitor agent.

Agents/models are cells; their messages/interactions are the signals. Build a live rexgraph
complex from a multi-agent message log and run the RCF machinery on it: per-agent
coherence/character, the Hodge decomposition of the interaction flow (coherent vs circulating vs
persistent), effective resistance (which agent is load-bearing), cross-agent output alignment, and
query-reweighting routing (which agent a query surfaces).

The complex is higher-order: the monitor sees the whole topology, where cycles are deadlocks, curl
is circulation/disagreement, and the harmonic component is consensus. The analytic primitives come
from rexgraph; this module is the agent-to-complex adapter plus the monitor/router loop.
"""
from __future__ import annotations

import contextlib
import re
from collections import Counter, defaultdict

import numpy as np

from agent.metrics import coherence_kappa


def _tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]{3,}", str(text).lower())]


def act_complex(events, *, close_trips: bool = True):
    """Oriented acts as a GRADED relational complex: legs at grade 1, trips at grade 2.

    `interaction_complex` below answers a different question and answers it undirected: it
    folds a message and its reply into one weighted edge, because for routing and load what
    matters is the traffic and not who spoke first. Acts are the opposite case, and three
    things about them do not survive being flattened to a graph.

    DIRECTION IS POSITIONAL, not a sign. A column and its negation are the same cell: the
    library canonicalises, which is right, because re-signing is a gauge and leaves the
    spectrum alone. So orientation is WHICH participant carries the single -1. A write
    distinguishes the actor, ``[actor, *objects]``; a read distinguishes the object,
    ``[object, actor]``. Two writers meeting at an object then agree, a writer against a
    reader does not, and that disagreement is the whole content of the frustration channel.
    A read naming several objects is several acts rather than one: a relation has exactly
    one distinguished end, and a gather from many has no single one to nominate.

    ARITY. A leg is k-ary when the act was. One carrier writing to three destinations is
    ONE 4-ary relation, not three edges: expanding it invents C(3,2) pairs that no act
    performed and reports cycles nothing traversed.

    RETURN. A carrier goes somewhere and comes back, and out-and-back is a cycle. Giving
    that cycle a FACE is what says it closes. The face column is SOLVED from ``B_1 c = 0``
    rather than declared, because a declared one fails the chain condition and then bounds
    nothing, which is indistinguishable from having no face at all until the arithmetic is
    checked. So `betti[1]` afterwards counts the circulation that did NOT close: an
    exchange nobody reciprocated, as against a round trip that came home.

    Participants keep their `kind:` prefix as their label, so `subcomplex` and `hyperslice`
    can cut the ambient complex down to one hive, one worker, or one grade without the
    builder needing to know the nesting in advance.

    Returns `(rex, labels)`, or `(None, [])` when no event carried an orientation.
    """
    from rexgraph.graph import RexGraph
    legs, trips = [], defaultdict(list)
    for e in events:
        d = e if isinstance(e, dict) else e.public()
        actor, flow = d.get("entity", ""), d.get("flow", "")
        on = d.get("on")
        if not (actor and on) or flow not in ("read", "write"):
            continue
        objs = list(on) if isinstance(on, (list, tuple)) else [on]
        legs.append((actor, objs, flow))
        tid = (d.get("detail") or {}).get("trip")
        if tid:
            trips[(actor, tid)].append(len(legs) - 1)
    if not legs:
        return None, []

    labels = sorted({actor for actor, _, _ in legs} | {o for _, objs, _ in legs for o in objs})
    idx = {v: i for i, v in enumerate(labels)}
    cells, leg_of = [], []
    for n, (actor, objs, flow) in enumerate(legs):
        if flow == "write":
            cells.append([idx[actor], *(idx[o] for o in objs)])
            leg_of.append(n)
        else:
            for o in objs:                               # a gather is several acts
                cells.append([idx[o], idx[actor]])
                leg_of.append(n)
    rex = RexGraph.from_cells([len(labels), cells])
    legs = [legs[n] for n in leg_of]
    trips = {k: [i for i, n in enumerate(leg_of) if n in set(v)] for k, v in trips.items()}
    if close_trips:
        _close_round_trips(rex, legs, trips)
    return rex, labels


def slice_participants(rex, labels, keep):
    """The subcomplex on a set of participants: one hive, one worker, one kind.

    `subcomplex` takes an edge mask rather than a vertex mask, because which acts survive
    is a decision and not a consequence: an act reaching out of the slice is either cut or
    kept, and the two give different topology. Cut is the answer here. A trip half inside a
    hive is not that hive's trip, and counting it as one would report the hive as closed
    when the closing happened somewhere it cannot see.

    `keep` is a predicate on the label or a set of labels. Returns what `subcomplex`
    returns, `(v_mask, e_mask, f_mask)`: MASKS over the parent complex, so `.sum()` counts
    what survived and the arrays stay aligned with the labels they came from.
    """
    import numpy as _np
    want = keep if callable(keep) else (lambda n, _k=set(keep): n in _k)
    v_mask = _np.array([bool(want(n)) for n in labels], dtype=_np.uint8)
    B1 = rex.B1
    B1 = _np.asarray(B1.todense() if hasattr(B1, "todense") else B1)
    inside = _np.array([bool(v_mask[_np.nonzero(B1[:, j])[0]].all())
                        for j in range(rex.nE)], dtype=_np.uint8)
    return rex.subcomplex(v_mask=v_mask, e_mask=inside)


def _close_round_trips(rex, legs, trips) -> None:
    """Give every out-and-back its face, solved rather than declared.

    A single delivery is a path and does not bound; it is the RECIPROCAL delivery that
    closes it, so trips are paired by carrier and by the two ends taken in either order.
    An unreciprocated trip is left open on purpose, because that is the reading."""
    ends = {}
    for (actor, _tid), leg_ids in trips.items():
        read = [i for i in leg_ids if legs[i][2] == "read"]
        write = [i for i in leg_ids if legs[i][2] == "write"]
        if not (read and write):
            continue                                    # not a there-and-back-again shape
        src = tuple(sorted(legs[read[0]][1]))
        dst = tuple(sorted(legs[write[0]][1]))
        ends.setdefault((actor, frozenset((src, dst))), []).append((src, dst, leg_ids))
    faces = []
    for runs in ends.values():
        outward = {(s, d) for s, d, _ in runs}
        if len(outward) < 2:
            continue                                    # went, never came back
        faces.append(sorted({i for _, _, ids in runs for i in ids}))
    if not faces:
        return
    with contextlib.suppress(Exception):                # a face that will not solve is not one
        rex.add_faces(faces)
        rex._ensure_clean()


class AgentComplex:
    """A live complex over a running swarm. Feed it messages (sender, recipient, text); it builds
    the interaction complex (agents=vertices, messages=weighted edges) and a content view, and the
    monitor runs the RCF analysis on both."""

    def __init__(self):
        self._msgs: list[dict] = []

    def add_message(self, sender, recipient, text, **meta):
        self._msgs.append({"from": str(sender), "to": str(recipient), "text": str(text), "meta": meta})
        return self

    def add_messages(self, rows):
        for r in rows:
            self.add_message(r.get("from", r.get("sender")), r.get("to", r.get("recipient")),
                             r.get("text", ""),
                             **{k: v for k, v in r.items() if k not in ("from", "to", "sender", "recipient", "text")})
        return self

    @classmethod
    def from_conversation(cls, tracker, user: str = "user", agent: str = "assistant"):
        """Adapt a live ConversationTracker into the complex: each exchange is user->agent (the
        prompt) and agent->user (the reply). The chat becomes a monitorable complex; a multi-agent
        swarm's messages carry from/to and flow through the same path."""
        ac = cls()
        exch = tracker.exchanges() if callable(getattr(tracker, "exchanges", None)) else \
            getattr(tracker, "exchanges", []) or getattr(tracker, "_exchanges", [])
        for ex in exch:
            inp = getattr(ex, "input_text", None) or getattr(ex, "input", None) or (ex.get("input") if isinstance(ex, dict) else "")
            out = getattr(ex, "output_text", None) or getattr(ex, "output", None) or (ex.get("output") if isinstance(ex, dict) else "")
            if inp:
                ac.add_message(user, agent, inp)
            if out:
                ac.add_message(agent, user, out)
        return ac

    def agents(self):
        return sorted({m["from"] for m in self._msgs} | {m["to"] for m in self._msgs})

    # the interaction complex: agents = vertices, messages = weighted edges

    def interaction_complex(self):
        from rexgraph.graph import RexGraph
        ags = self.agents(); idx = {a: i for i, a in enumerate(ags)}
        w = defaultdict(float)
        for m in self._msgs:
            a, b = idx[m["from"]], idx[m["to"]]
            if a != b:
                w[(min(a, b), max(a, b))] += 1.0        # undirected interaction strength
        edges = sorted(w)
        if not edges:
            return None, ags, idx, np.zeros(0), []
        src = np.array([e[0] for e in edges], np.int32)
        tgt = np.array([e[1] for e in edges], np.int32)
        we = np.array([w[e] for e in edges], np.float64)
        rex = RexGraph.from_graph(src, tgt, w_E=we)
        return rex, ags, idx, we, edges

    def _agent_concepts(self):
        cc = defaultdict(Counter)
        for m in self._msgs:
            cc[m["from"]].update(_tokens(m["text"]))
        return cc

    def _agent_texts(self):
        by = defaultdict(list)
        for m in self._msgs:
            by[m["from"]].append(m["text"])
        return [(" ".join(by[a]) or a) for a in self.agents()]

    def alignment(self, embed_fn=None):
        """Cross-agent output alignment (cosine similarity of agents' aggregated messages).

        With ``embed_fn`` (semantic embeddings, e.g. ``agent_complex.model_embed_fn()`` backed by
        a running model) it distinguishes a hallucinating or off-topic agent, semantically far from
        the task context, from a topically distinct specialist, semantically related. The lexical
        fallback (concept-cosine) cannot tell them apart. Returns (agents, similarity)."""
        ags = self.agents()
        if embed_fn is not None:
            E = np.asarray(embed_fn(self._agent_texts()), dtype=np.float64)
            if E.ndim == 2 and E.shape[0] == len(ags) and E.shape[1] > 0:
                En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
                return ags, En @ En.T
        cc = self._agent_concepts()                          # lexical fallback
        vocab = sorted({t for c in cc.values() for t in c})
        vi = {t: i for i, t in enumerate(vocab)}
        V = np.zeros((len(ags), max(len(vocab), 1)))
        for i, a in enumerate(ags):
            for t, n in cc[a].items():
                V[i, vi[t]] = n
        Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        return ags, Vn @ Vn.T

    # the monitor agent: run the analysis on the swarm

    def monitor(self, embed_fn=None):
        rex, ags, idx, we, edges = self.interaction_complex()
        out = {"n_agents": len(ags), "n_interactions": len(edges),
               "alignment_mode": "embedding" if embed_fn is not None else "lexical"}
        if rex is None:
            out["note"] = "no inter-agent interactions yet"; out["agents"] = []
            return out
        betti = [int(b) for b in rex.betti]
        kappa = coherence_kappa(rex)
        flow = (we / we.sum()).astype(np.float64)
        try:
            H = rex.hodge_full(flow)
            hodge = {"coherent": round(float(H["pct_grad"]), 3),
                     "circulating": round(float(H["pct_curl"]), 3),   # disagreement/loops
                     "persistent": round(float(H["pct_harm"]), 3)}    # consensus core
        except Exception:
            hodge = None
        try:
            eff = np.asarray(rex._effective_resistance_batch(np.arange(rex.nE)), dtype=np.float64)
        except Exception:
            eff = np.zeros(len(edges))
        load = defaultdict(float)
        for e, (s, t) in enumerate(edges):
            load[s] += float(eff[e]); load[t] += float(eff[e])
        # RCFE field on the coordination (faced) complex. A pairwise hive is a flat 1-complex with
        # no curvature; the field emerges at the higher grade, where a face is a triangle of agents
        # who mutually interact (relevance-gated, not every triple). Curvature is the per-interaction
        # deviation from the coherent ideal and localizes a drifting/hallucinating agent; strain is
        # its total, the network's field energy, trackable over time.
        curv_e = np.zeros(len(edges)); strain = None
        try:
            from rexgraph.graph import RexGraph as _RG
            {(s, t) for (s, t) in edges}
            adj = defaultdict(set)
            for (s, t) in edges:
                adj[s].add(t); adj[t].add(s)
            tris = []
            for (s, t) in edges:                              # a coordination face: s,t and a shared k
                for k in adj[s] & adj[t]:
                    tri = tuple(sorted((s, t, k)))
                    if tri not in tris:
                        tris.append(tri)
            if tris:
                src = np.array([e[0] for e in edges], np.int32)
                tgt = np.array([e[1] for e in edges], np.int32)
                frex = _RG.from_simplicial(src, tgt, np.array(sorted(set(tris)), np.int32))
                curv_e = np.asarray(frex.rcfe_curvature, dtype=np.float64).ravel()
                strain = float(frex.rcfe_strain)
        except Exception:
            pass
        curv = defaultdict(float)
        for e, (s, t) in enumerate(edges):
            if e < len(curv_e):
                curv[s] += float(curv_e[e]); curv[t] += float(curv_e[e])
        ags2, AL = self.alignment(embed_fn=embed_fn)
        avg_align = {a: (float(np.mean([AL[i, j] for j in range(len(ags2)) if j != i])) if len(ags2) > 1 else 1.0)
                     for i, a in enumerate(ags2)}
        # drift is relative to the swarm: an agent whose alignment is far below the median is
        # off-topic or possibly hallucinating, regardless of the absolute scale.
        # "far below the swarm" is an OUTLIER question, and this codebase already has
        # one convention for it: the data-adaptive Tukey lower fence (q1 - 1.5*IQR) used
        # in engine.py and hive.py and described there as "not a fixed magic". The old
        # `avga < 0.5 * med` invented a factor on top of a median; the fence is derived
        # from the alignment distribution itself. With fewer than four agents there are
        # no quartiles, so nothing is flagged rather than a number being invented.
        _al = np.asarray(list(avg_align.values()), dtype=float) if avg_align else np.zeros(0)
        if _al.size >= 4:
            _q1, _q3 = np.percentile(_al, [25.0, 75.0])
            align_fence = float(_q1 - 1.5 * (_q3 - _q1))
        else:
            align_fence = float("-inf")
        report = []
        for i, a in enumerate(ags):
            avga = avg_align.get(a, 0.0)
            report.append({
                "agent": a,
                "coherence": round(float(kappa[i]), 3) if i < len(kappa) else None,
                "load_bearing": round(load.get(i, 0.0), 3),           # effective-resistance centrality
                "curvature": round(curv.get(i, 0.0), 3),              # RCFE deviation localized here
                "alignment": round(avga, 3),                          # agreement with the swarm
                "messages": sum(1 for m in self._msgs if m["from"] == a),
                # low alignment means output diverges from the swarm: off-topic/hallucinating or a
                # topically distinct specialist. The concept-cosine cannot tell them apart;
                # embedding plus task-relevance (model_introspect) is the refinement that does.
                "flag": "divergent" if avga < align_fence else "ok",
            })
        report.sort(key=lambda x: -x["load_bearing"])
        # directed message-flow edges for the graph view (who talks to whom, how much)
        dw = defaultdict(float)
        for m in self._msgs:
            if m["from"] != m["to"]:
                dw[(m["from"], m["to"])] += 1.0
        graph_edges = [{"from": a, "to": b, "weight": int(w)} for (a, b), w in sorted(dw.items())]
        out.update({
            "deadlock_cycles": betti[1] if len(betti) > 1 else 0,     # β₁ interaction loops
            "strain": round(strain, 3) if strain is not None else None,   # total RCFE field energy
            "interaction_hodge": hodge,
            "agents": report,
            "edges": graph_edges,                                    # directed flow for the graph
            "alignment_matrix": {"agents": ags2, "matrix": np.round(AL, 3).tolist()},
        })
        return out

    def route(self, query: str, top_k: int = 3):
        """Query-reweighting: rank agents by relevance to a query (concept overlap, normalized).
        The router's decision signal, which agent(s) a query surfaces."""
        cc = self._agent_concepts(); qt = set(_tokens(query))
        scores = []
        for a in self.agents():
            overlap = sum(cc[a][t] for t in qt)
            scores.append((a, overlap / ((sum(cc[a].values()) ** 0.5) + 1e-9)))
        scores.sort(key=lambda x: -x[1])
        return [{"agent": a, "relevance": round(s, 3)} for a, s in scores[:top_k] if s > 0]


class DriftTracker:
    """Track the RCFE field across successive monitor snapshots to catch an agent that is *starting*
    to detract. The instantaneous monitor is one snapshot; drift is the trend: a worker whose
    curvature (structural deviation) climbs over time, or whose alignment (semantic agreement) falls,
    is beginning to hallucinate or drift from its task before any single snapshot flags it."""

    def __init__(self, window: int = 8):
        self.window = window
        self._hist: list[dict] = []          # per snapshot: {agent: {curvature, alignment}}
        self._strain: list[float] = []

    def snapshot(self, monitor: dict) -> DriftTracker:
        """Record one monitor() result as a time point."""
        self._hist.append({a["agent"]: {"curvature": float(a.get("curvature", 0.0) or 0.0),
                                        "alignment": float(a.get("alignment", 0.0) or 0.0)}
                           for a in monitor.get("agents", [])})
        self._strain.append(float(monitor.get("strain") or 0.0))
        if len(self._hist) > self.window:
            self._hist = self._hist[-self.window:]
            self._strain = self._strain[-self.window:]
        return self

    @staticmethod
    def _slope(series: list[float]) -> float:
        n = len(series)
        if n < 2:
            return 0.0
        mx = (n - 1) / 2.0
        my = sum(series) / n
        num = sum((x - mx) * (y - my) for x, y in enumerate(series))
        den = sum((x - mx) ** 2 for x in range(n)) or 1.0
        return num / den

    def trends(self) -> dict:
        """Per-agent {curvature_slope, alignment_slope, n} over the tracked window."""
        agents = sorted({a for h in self._hist for a in h})
        out = {}
        for a in agents:
            cs = [h[a]["curvature"] for h in self._hist if a in h]
            als = [h[a]["alignment"] for h in self._hist if a in h]
            out[a] = {"curvature_slope": round(self._slope(cs), 4),
                      "alignment_slope": round(self._slope(als), 4), "n": len(cs)}
        return out

    def strain_trend(self) -> float:
        """Slope of the network's total strain over time (rising = the field is heating up)."""
        return round(self._slope(self._strain), 4)

    def drifting(self, *, curv_rise: float = 0.05, align_fall: float = -0.05) -> list[str]:
        """Agents whose curvature is trending up or alignment trending down over the window: the
        ones starting to detract from the swarm."""
        return [a for a, s in self.trends().items()
                if s["n"] >= 2 and (s["curvature_slope"] > curv_rise
                                    or s["alignment_slope"] < align_fall)]


# live registry: the running swarm's shared complex, fed as agents/models interact

_LIVE: AgentComplex | None = None
_DRIFT: DriftTracker | None = None


def get_drift() -> DriftTracker:
    """The process-wide drift tracker: snapshot it with monitor() results over time to detect a
    worker that is beginning to hallucinate or drift from its task (a rising-curvature trend)."""
    global _DRIFT
    if _DRIFT is None:
        _DRIFT = DriftTracker()
    return _DRIFT


def reset_drift():
    global _DRIFT
    _DRIFT = None


def get_live() -> AgentComplex:
    """The process-wide live agentic complex: the shared structure the runtime appends to as
    agents/models exchange messages (model, memory, and database as one complex)."""
    global _LIVE
    if _LIVE is None:
        _LIVE = AgentComplex()
    return _LIVE


def record(sender, recipient, text, **meta):
    """Append one interaction to the live complex. Call this wherever agents/models message each
    other so the monitor runs on the real swarm, not a synthetic log."""
    get_live().add_message(sender, recipient, text, **meta)


def reset_live():
    global _LIVE
    _LIVE = None


def model_embed_fn(url: str | None = None):
    """A semantic embedder for `monitor(embed_fn=...)`, backed by an /v1/embeddings endpoint
    (via model_introspect). Returns None if nothing is serving embeddings, so the monitor falls
    back to the lexical signal. The embedding signal is what turns divergence detection into a
    hallucination-vs-specialist distinction.

    `url` names the endpoint explicitly - an ATTACHED embedder bee, whose process this
    interpreter does not own. Without it only a locally-MANAGED server (local_runtime.start /
    start_embedder) is discoverable, so an attached embedder would be invisible and the monitor
    would silently stay lexical."""
    def _embed(texts):
        from agent import model_introspect
        return model_introspect.embed(texts, url=url)
    if url:
        return _embed
    try:                                                     # offer it if any embedding endpoint is up
        from agent import local_runtime
        if local_runtime.embed_url():                        # dedicated embedder or chat model
            return _embed
    except Exception:
        pass
    return None
