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

import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

#: Kinds `record` builds a complex FOR, from labels, through a lineage adapter.
KINDS = ("pipeline-run", "conversation")

#: Kinds `record_complex` accepts a complex for, because the caller already has one.
#:
#: An agent that has analysed something, a hive that has placed its tasks, an edit to a
#: stored graph: each of those IS a complex before it is anything else, and flattening it
#: to labels so an adapter can rebuild a different one loses what it knew.
COMPLEX_KINDS = ("edit", "agent-state", "hive-state", "pipeline-run", "conversation")

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


def record_complex(kind: str, rex, *, lineage_id: str, workspace: str = "default",
                   meta: dict | None = None, tags: list[str] | None = None,
                   when: float | None = None, force: bool = False) -> dict[str, Any] | None:
    """Append a complex the caller already has as the next state of a lineage.

    `record` builds a complex from labels through an adapter, which is right when the
    caller has a list of things and no complex. This is for the other case, which is most
    of them: an edit to a stored graph, an agent's state, a hive's placement. Each is a
    complex before it is anything else, and there is nothing to reconstruct it from.

    The lineage is one object with two coordinates, exactly as `record` leaves it:

        version   the RCDB chain, bitemporal, read with get_version or as_of
        step      the position inside the TemporalRex, read with at or step_at

    so an edit history is a temporal complex rather than a log beside one, and any state
    in it reconstructs as a RexGraph that can be analysed, queried or drawn like any
    other. `TemporalRex.append_snapshot` keeps the checkpoint/delta index incrementally,
    O(delta) per edit, so a long history costs a diff per step and not a rebuild.

    Returns the version info, or None when recording is off. Unlike `record` this does
    not de-duplicate: two edits that happen to produce the same complex are still two
    edits, and a history that silently drops one is not a history.
    """
    if kind not in COMPLEX_KINDS:
        raise ValueError(
            f"unknown kind {kind!r}; expected one of {', '.join(COMPLEX_KINDS)}")
    if not force and not enabled(workspace, kind):
        return None
    if rex is None:
        return None

    from rexgraph.graph import TemporalRex

    store = _store()
    at = float(when if when is not None else time.time())
    try:
        prev = store.get(lineage_id)
    except Exception:
        prev = None

    if isinstance(prev, TemporalRex):
        temporal = prev
    else:
        temporal = TemporalRex([])
        if prev is not None:
            try:
                temporal.append_snapshot(prev, at=at)
            except Exception:
                logger.debug("could not carry the previous snapshot into %s", lineage_id)
    step = temporal.append_snapshot(rex, at=at)

    record_meta = {"kind": kind, "workspace": workspace, "step": int(step),
                   "at": at, "nV": int(rex.nV), "nE": int(rex.nE),
                   "nF": int(rex.nF_hodge)}
    record_meta.update(meta or {})
    from agent.rcdb import put_version
    info = put_version(store, lineage_id, temporal,
                       meta=record_meta, tags=list(tags or []) + [kind],
                       valid_from=at)
    return {**info, "step": int(step), "steps": int(temporal.T)}


def history(lineage_id: str) -> list[dict]:
    """Every recorded state of a lineage, as steps with their times.

    The edit log, read off the temporal store rather than kept beside it.
    """
    from rexgraph.graph import TemporalRex

    try:
        temporal = _store().get(lineage_id)
    except Exception:
        return []
    if not isinstance(temporal, TemporalRex):
        return []
    # `times` is an ndarray, so `or []` asks for its truth value and raises
    times = np.asarray(getattr(temporal, "times", [])).ravel().tolist()
    out = []
    for step in range(int(temporal.T)):
        # reconstruct_at, not at: `at` reads the raw snapshot list, which a store
        # round-trip does not materialise, and the checkpoint/delta index is what
        # survives. Same complex either way, one rebuild from the nearest checkpoint.
        snapshot = temporal.reconstruct_at(step)
        out.append({"step": step,
                    "at": float(times[step]) if step < len(times) else None,
                    "nV": int(snapshot.nV), "nE": int(snapshot.nE)})
    return out


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
