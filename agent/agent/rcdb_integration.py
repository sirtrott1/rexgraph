"""
agent.rcdb_integration: give the sibling rcdb package this application's policy.

rcdb does not import the agent, so everything the agent adds to a store arrives through
these four callables: what a change records into the activity feed, how a request narrows
the default store, how metadata is projected before it is stored, and how a candidate is
scored. Without them the store still works, just without any of it.

configure() is called once when the agent package is imported, so a caller that uses
agent.rcdb gets the agent's behaviour without having to know this wiring exists.
"""
from __future__ import annotations


def _activity(entity, action, detail):
    from agent import activity
    activity.record(entity, action, scope="network", detail=detail)


def _scope(store):
    from agent.server.scope import scoped
    return scoped(store)


def _privacy(meta):
    from agent.interfaces import apply_label_privacy
    return apply_label_privacy(meta)


def _similarity(rex, doc_labels, query_labels, *, reading=True):
    from agent.scoring import interfacing_score
    return interfacing_score(rex, doc_labels, query_labels, reading=reading)


def configure() -> None:
    """Install this application's policy into the store package."""
    from rcdb import configure_hooks
    configure_hooks(activity=_activity, scope=_scope, privacy=_privacy,
                    similarity=_similarity)
