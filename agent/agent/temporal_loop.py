"""agent.agent.temporal_loop: Slice D online loop closure.

ChangeSource turns the Slice C rcdb change-feed (routed through activity.py) into
a stream of ChangeEvents with the self-write guard; OnlineLoop advances a running
stable-id TemporalRex one step per event, runs FieldNavigator.step + the native
GreensCochainField, and (optionally) writes a guarded derived version back to the
store. The default path is torch-free.
"""
from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

DERIVED_TAG = "__online_derived__"          # reserved tag marking the loop's own write-backs


class ChangeEvent(NamedTuple):
    id: str
    version: int
    action: str                             # "rcdb.put" or "rcdb.delete"
    rex: object                             # RexGraph for put; None for delete


class ChangeSource:
    """Adapt the activity change-feed into ChangeEvents. Skips DERIVED_TAG events
    (the guard), fetches put payloads via store.get_version, and is idempotent on
    (id, version)."""

    def __init__(self, store, *, actions=("rcdb.put",), skip_tag=DERIVED_TAG):
        self.store = store
        self.actions = tuple(actions)
        self.skip_tag = skip_tag
        self._sub = None
        self._seen = set()

    def _event_from_pub(self, pub) -> Optional[ChangeEvent]:
        action = pub.get("action")
        detail = pub.get("detail") or {}
        tags = detail.get("tags") or []
        if self.skip_tag in tags:
            return None                     # the guard: never re-enter on our own writes
        cid = detail.get("id")
        version = detail.get("version")
        if cid is None or version is None:
            return None
        key = (cid, int(version))
        if key in self._seen:
            return None                     # idempotent: already delivered
        if action == "rcdb.delete":
            self._seen.add(key)
            return ChangeEvent(cid, int(version), action, None)
        if action not in self.actions:
            return None
        try:
            rex = self.store.get_version(cid, int(version))
        except Exception:
            rex = None
        if rex is None:
            return None                     # get_version miss: skip without marking seen
        self._seen.add(key)
        return ChangeEvent(cid, int(version), action, rex)

    def start(self, on_event: Callable[[ChangeEvent], None]) -> None:
        from agent import activity

        def _cb(pub):
            try:
                ev = self._event_from_pub(pub)
                if ev is not None:
                    on_event(ev)
            except Exception:
                pass                        # a bad event must never break the bus
        self._sub = _cb
        activity.get_log().subscribe(_cb)

    def stop(self) -> None:
        if self._sub is not None:
            from agent import activity
            activity.get_log().unsubscribe(self._sub)
            self._sub = None

    def poll(self, *, since=None, limit=200) -> List[ChangeEvent]:
        from agent import activity
        pubs = activity.get_log().events(scope="network", since=since, limit=limit)
        out = []
        for pub in reversed(pubs):          # events() is newest-first; replay oldest-first
            ev = self._event_from_pub(pub)
            if ev is not None:
                out.append(ev)
        return out


class StepResult(NamedTuple):
    t: int                                  # snapshot index, or -1 for a delete
    id: str
    version: int
    change: object                          # EdgeChange(added, removed), or None
    nav: dict                               # FieldNavigator.step output
    learn: object                           # predict_then_observe result, or None
    wrote_back: object                      # derived record id (str), or None


class OnlineLoop:
    """Hold the running stable-id TemporalRex, advance one step per change event,
    run the navigator + native field update, optionally write back a guarded
    derived version, and record a per-step StepResult."""

    def __init__(self, store, *, navigator=None, learner=None, observe=None,
                 write_back=False, derived_suffix="::online"):
        from rexgraph.graph import TemporalRex
        from rexgraph.flow.navigator import FieldNavigator
        self.store = store
        self.navigator = navigator if navigator is not None else FieldNavigator()
        if learner is not None:
            self.learner = learner
        else:
            from rexgraph.flow.online import GreensCochainField
            self.learner = GreensCochainField(observe=observe)
        self.write_back = bool(write_back)
        self.derived_suffix = derived_suffix
        self.trex = TemporalRex([])
        self._history: List[StepResult] = []
        self._dropped = set()

    def on_change(self, ev) -> Optional[StepResult]:
        from rexgraph.flow.navigator import changed_edges, removed_region_for
        if ev.action == "rcdb.delete":
            self._dropped.add(ev.id)
            res = StepResult(-1, ev.id, ev.version, None, {"event": False}, None, None)
            self._history.append(res)
            return res
        if ev.rex is None:
            return None
        t = self.trex.append_snapshot(ev.rex)         # key-level stable ids
        curr = self.trex.at(t)                        # materialize the snapshot ONCE
        if t > 0:
            prev = self.trex.at(t - 1)
            change = changed_edges(prev, curr)
            removed_region = removed_region_for(prev, curr, change.removed)
        else:
            change = None
            removed_region = None
        nav = self.navigator.step(curr, change, removed_region)     # reuse curr
        learn = self.learner.predict_then_observe(t, change, curr)  # pass the materialized snapshot
        wrote_back = self._persist_derived(ev, t, learn) if self.write_back else None
        res = StepResult(t, ev.id, ev.version, change, nav, learn, wrote_back)
        self._history.append(res)
        return res

    def _persist_derived(self, ev, t, learn):
        """Persist the current snapshot as a derived version tagged DERIVED_TAG so
        ChangeSource skips the resulting feed event (the loop cannot self-trigger).
        A failed persist is swallowed: the loop keeps consuming."""
        derived_id = ev.id + self.derived_suffix
        try:
            self.store.put(
                derived_id, self.trex.at(t),
                meta={"source": "online-loop", "from": ev.id,
                      "from_version": ev.version,
                      "step_error": (learn or {}).get("error")},
                tags=[DERIVED_TAG],
            )
        except Exception:
            return None
        return derived_id

    def run_stream(self, source) -> None:
        source.start(self.on_change)

    def history(self) -> List[StepResult]:
        return list(self._history)

    def save(self, path_prefix):
        """Persist the running TemporalRex through the Slice B delta serializer and
        the native field (phi) as two named tensors through the safetensors backend
        the rex-state pipeline uses. Returns (trex_path, field_path)."""
        import numpy as np
        from rexgraph.io.safetensors_bridge import temporal_rex_to_safetensors
        from safetensors.numpy import save_file
        trex_path = str(path_prefix) + ".trex.safetensors"
        field_path = str(path_prefix) + ".field.safetensors"
        temporal_rex_to_safetensors(self.trex, trex_path)
        phi = getattr(self.learner, "phi", {})
        keys = np.asarray(sorted(int(k) for k in phi.keys()), dtype=np.int64)
        vals = np.asarray([float(phi[int(k)]) for k in keys.tolist()], dtype=np.float64)
        save_file({"field/keys": keys, "field/values": vals}, field_path)
        return trex_path, field_path

    def load_field(self, field_path):
        """Restore the native field (phi) from a file written by save()."""
        import numpy as np
        from safetensors.numpy import load_file
        d = load_file(field_path)
        keys = np.asarray(d["field/keys"], dtype=np.int64)
        vals = np.asarray(d["field/values"], dtype=np.float64)
        self.learner.phi = {int(k): float(v)
                            for k, v in zip(keys.tolist(), vals.tolist())}
