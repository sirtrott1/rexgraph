"""agent.reactive_hive: the monitor -> schema loop that makes the hive self-organizing.

The hive observes the field on its own coordination complex and mutates its own
schema in response, versioning each change with its cause. This closes the loop:

    signal (a field on the complex)  ->  response (a schema mutation)  ->  version

Everything here runs on the same tensor: the signed boundary B, the composite
binary {0,1} (incidence) with {+,-} (orientation), whose entries are {0,+1,-1}.
The detection signals are field quantities read off B with no eigensolve and no
dense solve - the Hodge decomposition (gradient/curl/harmonic of the interaction
flow), the first Betti number (harmonic dimension = coordination deadlocks), and
the effective-resistance / RCFE-curvature fields that localize which worker is
load-bearing or divergent. Because the coordination complex, the database schema,
and the hive's own schema are all B-structured, the same field machinery flows
through all of them.

Default rules:
  * a coordination deadlock (beta_1 > 0, or persistent/harmonic flow)  ->  deploy a
    mediator worker so the hive can route around the stuck cycle.
  * a divergent worker (its curvature/alignment field flags it as off-topic or
    hallucinating)  ->  deploy a guard worker to check its output.
"""
from __future__ import annotations

from typing import Any

from .hive_schema import HiveSchema

# a NEED -> (worker name, worker_type, specialties). Filling a gap deploys the matching worker.
# Extensible: pass `provisioners=` to add or override. These cover the common code-team roles and
# the coordination fixes, so a minimal team grows the exact specialists a task demands.
_PROVISIONERS = {
    "plan":     ("planner", "coordinator:planner", ["plan", "design", "outline"]),
    "review":   ("reviewer", "analyzer:review", ["review", "critique", "bugs"]),
    "test":     ("tester", "analyzer:test", ["test", "validate", "coverage"]),
    "debug":    ("debugger", "analyzer:debug", ["debug", "fix", "trace"]),
    "verify":   ("verifier", "analyzer:verify", ["verify", "check", "consensus"]),
    "mediate":  ("mediator", "coordinator:mediator", ["coordinate", "mediate", "unblock"]),
}


class ReactiveHive:
    """Reads structural signals - the coordination field, a consensus result, a query's schema
    footprint - and mutates the hive's own schema in response (deploying a specialist, attaching a
    database), versioning each change with its cause. Every trigger is exact-structural (a Betti
    number, a set difference, a flag), never a tuned threshold."""

    def __init__(self, hive, schema: HiveSchema | None = None, *, store=None, provisioners=None):
        self.hive = hive
        self.schema = schema or HiveSchema(hive, store=store)
        self._deployed: set = set()          # remember fixes so we do not redeploy each tick
        self.provisioners = dict(_PROVISIONERS)
        if provisioners:
            self.provisioners.update(provisioners)

    def _deploy(self, name, *, worker_type, specialties, cause, key=None) -> dict | None:
        """Deploy a worker, version the schema with the cause, and remember it (idempotent)."""
        key = key or name
        if key in self._deployed:
            return None
        self.hive.add_worker(name, lambda d, **k: d, capability="analyze",
                             worker_type=worker_type, specialties=specialties)
        self._deployed.add(key)
        v = self.schema.snapshot(cause=cause)
        return {"deployed": name, "cause": cause, "version": v.get("version")}

    def _satisfied(self, need: str) -> bool:
        """A need is met when some bee already provides it (its type, specialties, or capability
        mention it). Exact set-membership, no scoring cutoff."""
        n = need.lower()
        for b in self.hive.bees():
            hay = " ".join([b.worker_type or "", b.capability or "", *(b.specialties or [])]).lower()
            if n in hay:
                return True
        return False

    def observe(self) -> dict[str, Any]:
        """Read the field signals off the coordination complex (all eigen-free).

        Beyond the monitor summary, this reads the EXACT harmonic localization and the
        frustration/coparticipation character of the coordination circulation via
        rexgraph.harmonic_health: which agents sit on the stuck mode, and whether the
        deadlock is irreducible topological tension (health_ratio > 1) or fillable overlap.
        """
        m = self.hive.monitor()
        hodge = m.get("interaction_hodge") or {}
        agents = m.get("agents", [])
        # load-bearing = effective-resistance field; a fallback locus if the harmonic read is absent
        load_locus = [a["agent"] for a in sorted(agents, key=lambda x: -(x.get("load_bearing") or 0.0))[:3]]

        health_ratio, harm_locus = None, []
        try:
            from rexgraph import harmonic_health
            rex, ags, idx, we, edges = self.hive._complex.interaction_complex()
            if rex is not None and int(rex.betti[1]) > 0:
                flow = (we / we.sum()) if getattr(we, "sum", None) and we.sum() > 0 else None
                hh = harmonic_health(rex, flow)
                hpe = hh.get("harm_per_edge")
                peak = float(hpe.max()) if hpe is not None and hpe.size else 0.0
                on = set()
                for e, (s, t) in enumerate(edges):                 # exact support of the harmonic mode
                    if peak > 0 and hpe[e] > 1e-9 * peak:
                        on.add(ags[s]); on.add(ags[t])
                harm_locus = sorted(on)
                health_ratio = hh.get("health_ratio")
        except Exception:
            pass

        return {
            "deadlock_cycles": int(m.get("deadlock_cycles", 0) or 0),   # beta_1 of interaction B
            "circulating": float(hodge.get("circulating", 0.0) or 0.0),  # curl fraction
            "persistent": float(hodge.get("persistent", 0.0) or 0.0),    # harmonic fraction
            "health_ratio": health_ratio,                               # frustration/coparticipation
            "strain": m.get("strain"),                                   # RCFE field energy
            "divergent": [a["agent"] for a in agents if a.get("flag") == "divergent"],
            "alignment_mode": m.get("alignment_mode", "lexical"),
            "locus": harm_locus or load_locus,                          # exact harmonic support first
        }

    def react(self) -> list[dict[str, Any]]:
        """Field-driven rules from the coordination complex; each fix versions the schema."""
        obs = self.observe()
        actions: list[dict[str, Any]] = []

        # rule 1: coordination deadlock -> deploy a mediator. EXACT trigger: beta_1 (harmonic
        # dimension) is an integer invariant; > 0 means a cycle exists. No magnitude threshold - the
        # harmonic fraction/character ride in the cause as reported severity, not a gate.
        if obs["deadlock_cycles"] > 0:
            hr = obs.get("health_ratio")
            kind = "" if hr is None else " irreducible" if hr >= 1 else " fillable"
            cause = ("coordination deadlock (beta_1=%d,%s harmonic=%.2f, locus=%s) -> deployed mediator"
                     % (obs["deadlock_cycles"], kind, obs["persistent"], ",".join(obs["locus"]) or "-"))
            name, wt, sp = self.provisioners["mediate"]
            a = self._deploy(name, worker_type=wt, specialties=sp, key="mediate", cause=cause)
            if a:
                a["rule"] = "deadlock"; actions.append(a)

        # rule 2: a divergent (likely hallucinating) worker -> deploy a guard on it. Only on the
        # RELIABLE semantic signal (embedding alignment); lexical alignment can't separate a
        # coordinator from a hallucinator, so it would false-flag a healthy hub.
        divergent = obs["divergent"] if obs["alignment_mode"] == "embedding" else []
        for name in divergent:
            a = self._deploy(f"guard.{name}", worker_type="analyzer:guard",
                             specialties=["review", "validate", name], key=f"guard.{name}",
                             cause=f"divergent worker '{name}' (alignment field) -> deployed guard.{name}")
            if a:
                a["rule"] = "divergence"; actions.append(a)

        return actions

    def require(self, *needs: str, cause: str | None = None) -> list[dict[str, Any]]:
        """Capability gap: for each need NOT already provided by some bee, deploy a specialist for
        it. This is the code-team unlock - a minimal hive declares `require('review','test')` and
        grows the exact roles it lacks. Trigger is exact set-membership (a need is met or it isn't).
        """
        actions = []
        for need in needs:
            if self._satisfied(need) or need in self._deployed:
                continue
            spec = self.provisioners.get(need)
            if spec is None:
                continue
            name, wt, sp = spec
            a = self._deploy(name, worker_type=wt, specialties=sp, key=need,
                             cause=cause or f"capability gap: '{need}' unmet -> deployed {name}")
            if a:
                a["rule"] = "capability"; a["need"] = need
                actions.append(a)
        return actions

    def on_consensus(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Reliability gap: a consensus that flagged a divergent worker, or returned a structurally
        unreliable answer (the library's varentropy `reliable` flag == False), -> deploy a verifier.
        Both triggers are structural facts (a non-empty flag list, a boolean), not a score cutoff."""
        flagged = result.get("flagged") or []
        unreliable = any(r.get("reliable") is False for r in result.get("responders", []))
        if not (flagged or unreliable):
            return []
        why = []
        if flagged:
            why.append(f"flagged {flagged}")
        if unreliable:
            why.append("structurally unreliable answer")
        a = self._deploy("verifier", worker_type="analyzer:verify", specialties=["verify", "check"],
                         key="verify", cause="reliability gap ({}) -> deployed verifier".format("; ".join(why)))
        if a:
            a["rule"] = "reliability"
            return [a]
        return []

    def on_query(self, query_state, *, available=None) -> list[dict[str, Any]]:
        """Data gap: the query references tables that no attached schema can join (unmatched or
        structurally disconnected). If a registered database in `available` (name -> object with
        `.tables()`/`.table_names()`, or an iterable of table names) covers them, attach it and its
        bees. Trigger is an exact set difference between what the query needs and what is bound."""
        sch = getattr(query_state, "schema", query_state)
        if not isinstance(sch, dict) or not sch.get("linked"):
            return []
        need = set(sch.get("unmatched_concepts", [])) | set(sch.get("disconnected_tables", []))
        if not need or not available:
            return []
        actions = []
        for db_name, db in available.items():
            if db_name in self._deployed:
                continue
            if hasattr(db, "table_names"):
                have = set(db.table_names())
            elif hasattr(db, "tables"):
                have = {t["name"] for t in db.tables()}
            else:
                have = set(db)
            covered = need & {t.lower() for t in have} or need & have
            if not covered:
                continue
            bees = db.attach_to_hive(self.hive) if hasattr(db, "attach_to_hive") else []
            self._deployed.add(db_name)
            links = [(b, "reads") for b in bees]
            cause = f"data gap: query needs {sorted(need)} -> attached database '{db_name}'"
            v = self.schema.attach_resource(db_name, "database", links=links, cause=cause)
            actions.append({"rule": "data", "attached": db_name, "bees": bees,
                            "cause": cause, "version": v.get("version")})
        return actions

    def _infer_needs(self, task: str) -> list[str]:
        """The roles a task implies, by which provisioner keys/specialties its words mention. Exact
        word presence (set membership), not a scored match."""
        t = (task or "").lower()
        needs = []
        for need, (_name, _wt, specs) in self.provisioners.items():
            if need in t or any(s in t for s in specs):
                needs.append(need)
        return needs

    def run(self, task: str, *, needs=None, query_state=None, available=None,
            verify: bool = True, consensus_k: int = 3) -> dict[str, Any]:
        """Run a task through the team with the reactive layer live - the team reshapes itself while
        it works. In order: fill the capability gaps the task implies (or `needs`, if given); bind
        any missing data (`on_query`); do the work with `collaborate`; if it deadlocked, `react`
        (deploy a mediator); then cross-check with `consensus` and, on a reliability gap, deploy a
        verifier. Returns the answer, the verification, and every reaction taken (each versioned in
        the self-schema)."""
        reactions: list[dict[str, Any]] = []

        # 1. capability gaps: the team grows the roles the task needs
        reactions += self.require(*(needs if needs is not None else self._infer_needs(task)))

        # 2. data gaps: bind a database the task references but the hive is not attached to
        if query_state is not None:
            reactions += self.on_query(query_state, available=available)

        # 3. the work: dynamic delegation with the deadlock-breaker
        work = self.hive.collaborate(task)

        # 4. if a circular wait formed, react so the structure improves (deploy a mediator)
        if work.get("deadlock_broken"):
            reactions += self.react()

        # 5. verify by consensus, and react to a reliability gap (deploy a verifier)
        verification = None
        if verify and self.hive._generate_bees():
            verification = self.hive.consensus(task, k=consensus_k)
            reactions += self.on_consensus(verification)

        return {"answer": work.get("answer"), "work": work, "verification": verification,
                "reactions": reactions, "team": [b.name for b in self.hive.bees()]}
