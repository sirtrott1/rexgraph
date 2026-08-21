"""Structure-specific answerers.

Each answerer holds one structure, answers only the questions that structure supports,
and declines the rest with a reason. Results compose as a union: every answerer is asked
and each keeps its own provenance.

    lexical     WordNet definitions and typed relations
    linkage     Roget categories and Wiktionary link kinds
    ontology    asserted subsumption and relations from a ParsedOntology
    passage     spans of a document corpus that hold the query terms

`register_with_hive` adds them as typed hive workers. The union is separate from
`Hive.dispatch_capability`, which routes to one provider.
"""
from __future__ import annotations

from agent.answerers.lexical import LexicalAnswerer
from agent.answerers.linkage import LinkageAnswerer
from agent.answerers.ontology import OntologyAnswerer
from agent.answerers.passage import PassageAnswerer

__all__ = ["LexicalAnswerer", "LinkageAnswerer", "OntologyAnswerer",
           "PassageAnswerer", "register", "exact_answers", "registered",
           "register_with_hive"]

#: name -> (answerer, render). Instantiated once per process: a lexicon is 161,705
#: entries and 2.6 s to read, and an ontology is a parse plus a complex build, so a
#: per-query construction would make an exact answer look like an expensive one.
_REGISTRY: dict[str, tuple] = {}


def register(name: str, answerer, render=None) -> None:
    """Add an answerer to the exact-answer union. `render(result) -> str` defaults to
    the one its own module defines."""
    if render is None:
        import importlib
        render = getattr(importlib.import_module(type(answerer).__module__),
                         "render", None)
    _REGISTRY[str(name)] = (answerer, render)


def registered() -> dict:
    """name -> capability, for the hive and for reporting what can be asked."""
    return {n: getattr(a, "capability", None)
            for n, (a, _r) in _default_registry().items()}


def _default_registry() -> dict:
    if not _REGISTRY:
        register("lexical", LexicalAnswerer())
        # an ontology is registered only when one is configured: there is no default
        # ontology, and an answerer with nothing loaded declines every query, which is
        # correct but is not worth a construction per process.
        import os
        # the linkage sources register when their files are on disk. Each is cheap to
        # read (Roget 1.5 MB, the Wiktionary index 63 MB of binary) and each answers
        # kinds the others do not, so the union covers what no single lexicon does.
        from agent.answerers.linkage import DEFAULT_ROGET, DEFAULT_WIKTIONARY
        if os.path.exists(DEFAULT_ROGET):
            register("roget", LinkageAnswerer.roget())
        if os.path.exists(DEFAULT_WIKTIONARY):
            register("wiktionary", LinkageAnswerer.wiktionary())
        path = os.environ.get("REXGRAPH_ONTOLOGY")
        if path and os.path.exists(path):
            register("ontology", OntologyAnswerer.from_file(path))
    return _REGISTRY


def exact_answers(query: str) -> list[dict]:
    """Every exact answer the registered structures support for this query.

    A union, in registration order. An answerer that declines contributes nothing and
    costs nothing, because each one checks its own interface before touching its structure,
    so this is cheap for the common case where a query is not answerable exactly at all.
    """
    out = []
    for name, (answerer, render) in _default_registry().items():
        try:
            got = answerer.answer(query)
        except Exception:
            # The result is a UNION, so one structure being unreachable (its source
            # not on disk, its index stale) removes that answerer's contribution and
            # not the others'. An answerer that CAN answer and declines returns
            # answered=False instead, which is a different statement.
            continue
        if not got.get("answered"):
            continue
        out.append({
            "answerer": name,
            "kind": got.get("relation") or got.get("asked"),
            "subject": got.get("subject"),
            "source": got.get("source", name),
            "text": (render(got) if render else ""),
            "result": got,
        })
    return out


def register_with_hive(hive=None) -> list[str]:
    """Register every answerer as a hive worker, and return the names.

    WHY THIS IS SEPARATE FROM `exact_answers`, since both look like registries. The hive
    routes a task to ONE provider of a capability, which is what `dispatch_capability`
    does, and it is the right shape for work that any one worker can do. Answering is the
    other shape: every structure is asked, each declines what it cannot support, and the
    survivors are ALL reported, because a union of exact answers is the composition and
    picking one would throw away an exact answer that another structure had.

    So the union stays here and the hive gets the members, which is what makes them
    visible to `providers`, `status`, the member roster and the activity journal like
    every other bee. `as_worker` was written for this and had no caller but its test.
    """
    from agent import hive as hivemod
    h = hive if hive is not None else hivemod.get_hive()
    names = []
    for name, (answerer, _render) in _default_registry().items():
        handler, capability, worker_type = answerer.as_worker()
        h.add_worker(name, handler, capability=capability, worker_type=worker_type)
        names.append(name)
    return names
