"""
agent.work_recorder: record the platform's own work as temporal relational state.

A pipeline run and a conversation are both a sequence of states. Recording one
means appending a snapshot to that lineage's TemporalRex and storing it as the next
version of a single RCDB record, so a lineage is one object with two coordinates:

  version   the RCDB chain, bitemporal, readable with get_version / as_of
  step      the position inside the TemporalRex, readable with at / step_at

A chat state is therefore a position in a temporal rex, and the version metadata is
the temporal index rather than a numbering invented here.

Recording is opt-in per workspace and off by default: it writes records the user did
not ask for, so the decision is made once, in settings, not per call. The check lives
here so callers never carry the policy.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: What may be recorded. A caller naming anything else is a bug, not a new kind.
KINDS = ("pipeline-run", "conversation")

#: The workspace setting that turns recording on, and the per-kind switches under it.
SETTING = "record_work"
KIND_SETTING = "record_work_kinds"


def enabled(workspace: str = "default", kind: str | None = None) -> bool:
    """Whether this workspace records its own work, and optionally whether it
    records this kind. Off unless the workspace has switched it on."""
    try:
        from agent.server.persistence import load_settings
        s = load_settings(workspace)
    except Exception:
        return False
    if not s.get(SETTING):
        return False
    if kind is None:
        return True
    kinds = s.get(KIND_SETTING)
    return kind in kinds if isinstance(kinds, list) else True


def _store():
    from agent.server.routes.rcdb import _store as get
    return get()


def _labels_at(rec) -> list[str]:
    return list(((rec.meta or {}).get("vertex_labels") or [])) if rec is not None else []


def record(kind: str, labels: list[str], *, lineage_id: str,
           workspace: str = "default", edges: list | None = None,
           tags: list[str] | None = None, when: float | None = None,
           force: bool = False) -> dict[str, Any] | None:
    """Append one state to a lineage. Returns the version info, or None when
    recording is off or the state is unchanged.

    `force` records regardless of the workspace setting, for an explicit request
    rather than an automatic one.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    if not force and not enabled(workspace, kind):
        return None
    labels = [str(x) for x in (labels or [])]
    if len(labels) < 2:
        return None                      # nothing to relate yet

    from agent.lineage_adapters import conversation_to_rex, run_to_rex
    build = conversation_to_rex if kind == "conversation" else run_to_rex
    rex, meta = build(labels, edges=edges)
    if rex is None:
        return None

    store = _store()
    at = float(when if when is not None else time.time())

    # The record IS the temporal rex: load what is there, append this state, store
    # the whole thing as the next version. A repeated identical state is not a step.
    from rexgraph.graph import TemporalRex
    prev = None
    try:
        prev = store.get(lineage_id)
    except Exception:
        prev = None
    rec = None
    try:
        rec = store.get_record(lineage_id)
    except Exception:
        rec = None
    if _labels_at(rec) == labels:
        return {"lineage_id": lineage_id, "version": getattr(rec, "version", None),
                "unchanged": True}

    if isinstance(prev, TemporalRex):
        temporal = prev
    else:
        temporal = TemporalRex([])
        if prev is not None:             # a lineage recorded before this ran
            try:
                temporal.append_snapshot(prev, at=at)
            except Exception:
                logger.debug("could not carry the previous snapshot into %s", lineage_id)
    try:
        step = temporal.append_snapshot(rex, at=at)
    except Exception as e:
        logger.warning("could not append to %s: %s", lineage_id, e)
        return None

    meta = dict(meta)
    meta.update({"kind": kind, "workspace": workspace,
                 "temporal": {"step": int(step), "at": at, "T": int(temporal.T)}})
    all_tags = sorted(set((tags or []) + [kind, "recorded-work"]))

    from agent.rcdb import put_version
    info = put_version(store, lineage_id, temporal, meta=meta, tags=all_tags,
                       valid_from=at)
    info["unchanged"] = False
    info["step"] = int(step)
    info["T"] = int(temporal.T)
    return info


def state_at(lineage_id: str, when: float):
    """The recorded state current at a moment: the position in the temporal rex,
    and the complex reconstructed there. Returns (step, rex) or (None, None)."""
    temporal = _store().get(lineage_id)
    from rexgraph.graph import TemporalRex
    if not isinstance(temporal, TemporalRex):
        return None, None
    step = temporal.step_at(float(when))
    if step is None:
        return None, None
    return int(step), temporal.reconstruct_at(int(step))


def recorded(workspace: str | None = None, kind: str | None = None) -> list[dict]:
    """The lineages this platform has recorded, newest first."""
    out = []
    for rec in _store().query(tags_any=["recorded-work"]):
        m = rec.meta or {}
        if workspace and m.get("workspace") != workspace:
            continue
        if kind and m.get("kind") != kind:
            continue
        out.append({"id": rec.id, "kind": m.get("kind"), "version": rec.version,
                    "T": (m.get("temporal") or {}).get("T"),
                    "at": (m.get("temporal") or {}).get("at"),
                    "labels": (m.get("vertex_labels") or [])[-4:]})
    out.sort(key=lambda r: r.get("at") or 0, reverse=True)
    return out
