"""agent.hive_network: a network of hives as a relational complex one grade up.

A single hive is agents as cells (agent_complex). A network is the same structure lifted a grade:
hives are the cells, inter hive channels are the signals, and the network's health is the same RCFE
field / Hodge / drift read on inter hive traffic: which hive is load bearing, deviating (curvature),
or drifting. Routing and monitoring reuse the hive and agent_complex machinery at the network grade.
"""
from __future__ import annotations

import contextlib
import os
from threading import RLock

from agent import agent_complex
from agent.hive import Hive, _tokens


class HiveNetwork:
    """A set of hives enrolled as cells of one inter hive complex. Route picks a hive then delegates
    to its own routing; monitor runs the relational complex monitor on inter hive traffic."""

    def __init__(self):
        self._registry_lock = RLock()
        self._traffic_lock = RLock()
        self._hives: dict[str, Hive] = {}
        self._specialties: dict[str, list] = {}
        self._net = agent_complex.AgentComplex()          # the inter hive complex (hives = cells)
        self._drift = agent_complex.DriftTracker()        # network grade drift, separate from hives

    def add_hive(self, name: str, hive: Hive, *, specialties=None) -> None:
        """Enroll a hive as a cell in the network, with concept keywords for inter hive routing."""
        with self._registry_lock:
            self._hives[name] = hive
            self._specialties[name] = list(specialties or [])

    def hives(self) -> list[str]:
        with self._registry_lock:
            return sorted(self._hives)

    #### registry: create / address / remove named hives
    def hive(self, name: str = "default"):
        """Get or create a named hive (a cell of the network). This is how the 'default' hive and
        every named hive come into being; creation is logged at network scope."""
        with self._registry_lock:
            h = self._hives.get(name)
            created = h is None
            if created:
                h = Hive(name=name)
                self.add_hive(name, h)
        if created:
            from agent import activity
            activity.record("hive:" + name, "create", scope="hive")
        return h

    def create(self, name: str):
        with self._registry_lock:
            if name in self._hives:
                raise ValueError(f"hive {name!r} already exists")
            return self.hive(name)

    def get(self, name: str):
        with self._registry_lock:
            return self._hives.get(name)

    def names(self) -> list[str]:
        return self.hives()

    def remove(self, name: str) -> bool:
        with self._registry_lock:
            h = self._hives.pop(name, None)
            self._specialties.pop(name, None)
        if h is None:
            return False
        with contextlib.suppress(Exception):
            h.stop_all()
        from agent import activity
        activity.record("hive:" + name, "remove", scope="hive")
        return True

    def reset(self, name: str) -> None:
        self.remove(name)

    def reset_all(self) -> None:
        for n in self.names():
            self.remove(n)

    def status(self) -> dict:
        """Per hive rosters + network totals (the registry view; monitor() is the inter hive field)."""
        hives, total = [], 0
        with self._registry_lock:
            members = list(self._hives.items())
        for n, hive in members:
            st = hive.status()
            total += st["n_bees"]
            hives.append({"name": n, "n_bees": st["n_bees"], "queen": st["queen"],
                          "workers": st["workers"]})
        return {"n_hives": len(hives), "n_bees": total, "hives": hives}

    def relay(self, sender: str, recipient: str, text: str, **meta):
        """Record one inter hive message into the network complex (the grade up analog of Hive.relay)."""
        with self._traffic_lock:
            self._net.add_message(sender, recipient, text, **meta)

    def route(self, query: str, top_k: int = 3) -> list[dict]:
        """Rank hives for a query, blending inter hive interaction history with declared specialty -
        the same query reweighting as Hive.route, one grade up."""
        qt = set(_tokens(query))
        with self._registry_lock:
            specialties = {name: list(values) for name, values in self._specialties.items()}
        with self._traffic_lock:
            hist = {r["agent"]: r["relevance"]
                    for r in self._net.route(query, top_k=max(len(specialties), 1))}
        ranked = []
        for name, values in specialties.items():
            st = {t for s in values for t in _tokens(s)}
            spec = (len(qt & st) / (len(st) ** 0.5 + 1e-9)) if st else 0.0
            ranked.append({"hive": name, "score": round(0.5 * spec + 0.5 * hist.get(name, 0.0), 3),
                           "specialty": round(spec, 3), "history": round(hist.get(name, 0.0), 3)})
        ranked.sort(key=lambda x: -x["score"])
        return ranked[:top_k]

    def dispatch(self, query: str, **kw) -> dict:
        """Route to the best hive, delegate to its dispatch, and record the inter hive hop into the
        network complex. Returns {hive, result}."""
        r = self.route(query, top_k=1)
        if not r:
            raise ValueError("no hives in the network")
        target = r[0]["hive"]
        self.relay("network", target, query)
        hive = self.get(target)
        if hive is None:
            raise ValueError("selected hive was removed before dispatch")
        result = hive.dispatch(query, **kw)
        reply = result.get("reply", "") if isinstance(result, dict) else ""
        self.relay(target, "network", str(reply)[:200])
        return {"hive": target, "result": result}

    def dispatch_capability(self, capability: str, data, *, hint: str = None) -> dict:
        """Route a structured task across the network: pick a hive that has a provider of the
        capability (by inter hive routing on `hint`), then dispatch within it. Returns
        {hive, worker, capability, result}."""
        with self._registry_lock:
            members = list(self._hives.items())
            specialties = {n: list(v) for n, v in self._specialties.items()}
        candidates = {n: h for n, h in members if h.providers(capability)}
        if not candidates:
            raise ValueError(f"no hive provides capability {capability!r}")
        name = next(iter(candidates))
        if hint and len(candidates) > 1:
            ht = set(_tokens(hint))
            name = max(candidates, key=lambda n: len(
                ht & {t for s in specialties[n] for t in _tokens(s)}))
        self.relay("network", name, f"invoke:{capability}")
        out = candidates[name].dispatch_capability(capability, data, hint=hint)
        self.relay(name, "network", f"result:{capability}")
        return {"hive": name, **out}

    def monitor(self, *, track: bool = False) -> dict:
        """The network grade field: the same relational complex monitor on inter hive traffic, so
        hives are the cells: which hive is load bearing, deviating (curvature/strain), or (with
        track=True) drifting over time."""
        with self._traffic_lock:
            out = self._net.monitor()
            if track:
                self._drift.snapshot(out)
                out["drift"] = {"drifting": self._drift.drifting(),
                                "strain_trend": self._drift.strain_trend()}
        out["hives"] = self.hives()
        return out

    def snapshot(self) -> dict:
        """The whole network as one nested structure: each hive's snapshot (workers + type complex),
        the inter hive monitor, and the network field. Network = ambient complex, hive = subcomplex,
        worker = cell: one relational structure across grades."""
        with self._registry_lock:
            members = list(self._hives.items())
        return {"hives": {name: h.snapshot() for name, h in members},
                "network_monitor": self.monitor()}

    def persist(self, store="memory://", *, name: str = "network") -> str | None:
        """Catalogue the inter hive complex in the RCDB by structural signature, and each member
        hive alongside it (network = ambient complex, hives = subcomplexes). `store` is an open
        RCStore or an RCDB uri (pass a shared store or a persistent uri to retrieve later). Returns
        the network record id, or None when there is no inter hive structure yet."""
        from agent.rcdb import open_store
        st = open_store(store) if isinstance(store, str) else store
        with self._traffic_lock:
            rex, ags, idx, we, edges = self._net.interaction_complex()
        with self._registry_lock:
            members = list(self._hives.items())
        for hname, h in members:
            h.persist(st, name=f"{name}:{hname}")
        if rex is None:
            return None
        st.put(name, rex, meta={"kind": "network", "hives": sorted(n for n, _ in members)}, tags=["network"])
        return name


# one network per workspace

_NETWORKS: dict[str, HiveNetwork] = {}
_NETWORKS_LOCK = RLock()


def get_network(workspace: str | None = None) -> HiveNetwork:
    """The hive network for one workspace: its registry of named hives.

    Keyed by workspace rather than process wide. `hive(name)` is get or create, and
    /agents/command reaches it with `scope: "hive:<name>"`, so any tenant could bring a
    named hive into being, and every tenant then shared that one object: the same worker
    bees, and the same coordination complex they each write through `chat` and read
    through `monitor`. Resolved from the bound request when not named, so a caller
    outside a request keeps the single "default" network it always had.
    """
    from agent.server.scope import bound_workspace
    name = workspace or bound_workspace()
    with _NETWORKS_LOCK:
        if name not in _NETWORKS:
            _NETWORKS[name] = HiveNetwork()
        return _NETWORKS[name]


def reset_network(workspace: str | None = None):
    """Drop one workspace's network, or every one of them when none is named."""
    with _NETWORKS_LOCK:
        if workspace is None:
            _NETWORKS.clear()
        else:
            _NETWORKS.pop(workspace, None)


def _after_fork():
    global _NETWORKS, _NETWORKS_LOCK
    _NETWORKS, _NETWORKS_LOCK = {}, RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
