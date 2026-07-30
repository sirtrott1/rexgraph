"""agent.hive_network: a network of hives as a relational complex one grade up.

A single hive is agents-as-cells (agent_complex). A network is the same structure lifted a grade:
hives are the cells, inter-hive channels are the signals, and the network's health is the same RCFE
field / Hodge / drift read on inter-hive traffic - which hive is load-bearing, deviating (curvature),
or drifting. Routing and monitoring reuse the hive and agent_complex machinery at the network grade.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from agent import agent_complex
from agent.hive import Hive, _tokens


class HiveNetwork:
    """A set of hives enrolled as cells of one inter-hive complex. Route picks a hive then delegates
    to its own routing; monitor runs the relational-complex monitor on inter-hive traffic."""

    def __init__(self):
        self._hives: Dict[str, Hive] = {}
        self._specialties: Dict[str, list] = {}
        self._net = agent_complex.AgentComplex()          # the inter-hive complex (hives = cells)
        self._drift = agent_complex.DriftTracker()        # network-grade drift, separate from hives

    def add_hive(self, name: str, hive: Hive, *, specialties=None) -> None:
        """Enroll a hive as a cell in the network, with concept keywords for inter-hive routing."""
        self._hives[name] = hive
        self._specialties[name] = list(specialties or [])

    def hives(self) -> List[str]:
        return sorted(self._hives)

    # -- registry: create / address / remove named hives ----------------------

    def hive(self, name: str = "default"):
        """Get-or-create a named hive (a cell of the network). This is how the 'default' hive and
        every named hive come into being; creation is logged at network scope."""
        h = self._hives.get(name)
        if h is None:
            h = Hive(name=name)
            self.add_hive(name, h)
            from agent import activity
            activity.record("hive:" + name, "create", scope="hive")
        return h

    def create(self, name: str):
        if name in self._hives:
            raise ValueError(f"hive {name!r} already exists")
        return self.hive(name)

    def get(self, name: str):
        return self._hives.get(name)

    def names(self) -> List[str]:
        return self.hives()

    def remove(self, name: str) -> bool:
        h = self._hives.pop(name, None)
        self._specialties.pop(name, None)
        if h is None:
            return False
        try:
            h.stop_all()
        except Exception:
            pass
        from agent import activity
        activity.record("hive:" + name, "remove", scope="hive")
        return True

    def reset(self, name: str) -> None:
        self.remove(name)

    def reset_all(self) -> None:
        for n in list(self._hives):
            self.remove(n)

    def status(self) -> dict:
        """Per-hive rosters + network totals (the registry view; monitor() is the inter-hive field)."""
        hives, total = [], 0
        for n in self.names():
            st = self._hives[n].status()
            total += st["n_bees"]
            hives.append({"name": n, "n_bees": st["n_bees"], "queen": st["queen"],
                          "workers": st["workers"]})
        return {"n_hives": len(hives), "n_bees": total, "hives": hives}

    def relay(self, sender: str, recipient: str, text: str, **meta):
        """Record one inter-hive message into the network complex (the grade-up analog of Hive.relay)."""
        self._net.add_message(sender, recipient, text, **meta)

    def route(self, query: str, top_k: int = 3) -> List[dict]:
        """Rank hives for a query, blending inter-hive interaction history with declared specialty -
        the same query-reweighting as Hive.route, one grade up."""
        qt = set(_tokens(query))
        hist = {r["agent"]: r["relevance"]
                for r in self._net.route(query, top_k=max(len(self._hives), 1))}
        ranked = []
        for name in self._hives:
            st = {t for s in self._specialties[name] for t in _tokens(s)}
            spec = (len(qt & st) / (len(st) ** 0.5 + 1e-9)) if st else 0.0
            ranked.append({"hive": name, "score": round(0.5 * spec + 0.5 * hist.get(name, 0.0), 3),
                           "specialty": round(spec, 3), "history": round(hist.get(name, 0.0), 3)})
        ranked.sort(key=lambda x: -x["score"])
        return ranked[:top_k]

    def dispatch(self, query: str, **kw) -> dict:
        """Route to the best hive, delegate to its dispatch, and record the inter-hive hop into the
        network complex. Returns {hive, result}."""
        r = self.route(query, top_k=1)
        if not r:
            raise ValueError("no hives in the network")
        target = r[0]["hive"]
        self.relay("network", target, query)
        result = self._hives[target].dispatch(query, **kw)
        reply = result.get("reply", "") if isinstance(result, dict) else ""
        self.relay(target, "network", str(reply)[:200])
        return {"hive": target, "result": result}

    def dispatch_capability(self, capability: str, data, *, hint: str = None) -> dict:
        """Route a structured task across the network: pick a hive that has a provider of the
        capability (by inter-hive routing on `hint`), then dispatch within it. Returns
        {hive, worker, capability, result}."""
        candidates = [n for n, h in self._hives.items() if h.providers(capability)]
        if not candidates:
            raise ValueError(f"no hive provides capability {capability!r}")
        name = candidates[0]
        if hint and len(candidates) > 1:
            ht = set(_tokens(hint))
            name = max(candidates, key=lambda n: len(
                ht & {t for s in self._specialties[n] for t in _tokens(s)}))
        self.relay("network", name, f"invoke:{capability}")
        out = self._hives[name].dispatch_capability(capability, data, hint=hint)
        self.relay(name, "network", f"result:{capability}")
        return {"hive": name, **out}

    def monitor(self, *, track: bool = False) -> dict:
        """The network-grade field: the same relational-complex monitor on inter-hive traffic, so
        hives are the cells - which hive is load-bearing, deviating (curvature/strain), or (with
        track=True) drifting over time."""
        out = self._net.monitor()
        out["hives"] = self.hives()
        if track:
            self._drift.snapshot(out)
            out["drift"] = {"drifting": self._drift.drifting(),
                            "strain_trend": self._drift.strain_trend()}
        return out

    def snapshot(self) -> dict:
        """The whole network as one nested structure: each hive's snapshot (workers + type complex),
        the inter-hive monitor, and the network field. Network = ambient complex, hive = subcomplex,
        worker = cell - one relational structure across grades."""
        return {"hives": {name: h.snapshot() for name, h in self._hives.items()},
                "network_monitor": self.monitor()}

    def persist(self, store="memory://", *, name: str = "network") -> Optional[str]:
        """Catalogue the inter-hive complex in the RCDB by structural signature, and each member
        hive alongside it (network = ambient complex, hives = subcomplexes). `store` is an open
        RCStore or an RCDB uri (pass a shared store or a persistent uri to retrieve later). Returns
        the network record id, or None when there is no inter-hive structure yet."""
        from agent.rcdb import open_store
        st = open_store(store) if isinstance(store, str) else store
        rex, ags, idx, we, edges = self._net.interaction_complex()
        for hname, h in self._hives.items():
            h.persist(st, name=f"{name}:{hname}")
        if rex is None:
            return None
        st.put(name, rex, meta={"kind": "network", "hives": self.hives()}, tags=["network"])
        return name


# process-wide singleton

_NETWORK: Optional[HiveNetwork] = None


def get_network() -> HiveNetwork:
    global _NETWORK
    if _NETWORK is None:
        _NETWORK = HiveNetwork()
    return _NETWORK


def reset_network():
    global _NETWORK
    _NETWORK = None
