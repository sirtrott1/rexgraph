"""
agent.ontology_releases: an ontology across its releases, as one changing complex.

An ontology is not a file, it is a series of files. GO ships monthly, terms are
introduced, obsoleted, merged and split, and the question a curator or a downstream
pipeline actually has is what changed and whether it matters. Diffing two OBO files
as text answers the first badly and the second not at all.

Here the series is a `TemporalRex`: one complex with continuous identity, and the
releases are its snapshots over a shared term vocabulary. That is what makes the
comparison structural rather than textual. Three readings come out of it:

**Lifecycle.** When each term first and last appears. A term that stops appearing is
obsoleted; one that starts is introduced.

**Merges, stated rather than inferred.** OBO records a merge by giving the surviving
term an `alt_id` naming the one it absorbed. The parser already reads `alt_id` as an
alias, so a merge is read off what the file says instead of guessed from a
disappearance. That distinction matters: a term vanishing because it was merged and a
term vanishing because it was deleted are different events with the same textual
signature.

**Surprise.** `FieldNavigator` walks the series and stays idle unless the change at a
step is a surprise against the trend, so a run of ordinary monthly growth costs
nothing and the release that reorganised a branch is the one that reports. What it
returns is localized to the region that changed, not the whole ontology.

The vocabulary is shared across snapshots on purpose. Each release parsed on its own
would index its terms differently and the "same" term would be a different vertex at
each step, which is exactly the identity a temporal complex exists to keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Release:
    """One release of an ontology."""

    label: str
    parsed: object                       # ParsedOntology
    terms: set = field(default_factory=set)
    relations: set = field(default_factory=set)   # (subject, predicate, object)

    @property
    def n_terms(self) -> int:
        return len(self.terms)


def load_releases(sources, labels=None) -> list[Release]:
    """Read each source as one release, in the order given.

    Order is the caller's: a filename is not a date, and guessing one from it would
    silently reorder a series.
    """
    from agent.adapters import ontology_formats as OF

    names = list(labels or [])
    out = []
    for i, src in enumerate(sources):
        label = names[i] if i < len(names) else f"release_{i}"
        parsed = (src if hasattr(src, "triples")
                  else OF.read_ontology(str(src)))
        terms = {s for s, _p, _o in parsed.triples} | {
            o for _s, _p, o in parsed.triples} | set(parsed.labels)
        out.append(Release(label=label, parsed=parsed, terms=terms,
                           relations={tuple(t) for t in parsed.triples}))
    return out


def shared_vocabulary(releases) -> list[str]:
    """Every term any release mentions, in one stable order.

    The order is what gives a term the same vertex index in every snapshot, which is
    what `TemporalRex` needs to treat them as one thing over time.
    """
    seen = set()
    for r in releases:
        seen |= r.terms
    return sorted(seen)


def temporal_complex(releases, *, vocabulary=None):
    """The series as one `TemporalRex` over a shared vocabulary.

    Returns `(temporal, vocabulary)`. A release with no relations is still appended,
    because an empty snapshot is a real state of the series and skipping it would
    shift every later index.
    """
    from rexgraph.graph import RexGraph, TemporalRex

    vocab = list(vocabulary) if vocabulary is not None else shared_vocabulary(releases)
    index = {t: i for i, t in enumerate(vocab)}
    temporal = TemporalRex([])
    for t, rel in enumerate(releases):
        src, tgt = [], []
        for s, _p, o in sorted(rel.relations):
            if s in index and o in index and s != o:
                src.append(index[s])
                tgt.append(index[o])
        rex = RexGraph(sources=np.asarray(src, np.int32),
                       targets=np.asarray(tgt, np.int32))
        rex._agent_meta = {"vertex_labels": vocab, "release": rel.label,
                           "input_type": "ontology"}
        temporal.append_snapshot(rex, at=float(t))
    return temporal, vocab


def term_lifecycle(releases) -> dict:
    """When each term appears and disappears across the series.

    `obsoleted_at` is the index of the first release the term is missing from after
    having been present. A term absent from the start is not obsoleted, it was never
    there.
    """
    n = len(releases)
    present: dict[str, list[int]] = {}
    for t, rel in enumerate(releases):
        for term in rel.terms:
            present.setdefault(term, []).append(t)

    out = {}
    for term, steps in present.items():
        first, last = steps[0], steps[-1]
        out[term] = {
            "first_seen": first,
            "last_seen": last,
            "present_in": len(steps),
            "obsoleted_at": (last + 1) if last < n - 1 else None,
            "introduced": first > 0,
            "gaps": [t for t in range(first, last) if t not in set(steps)],
        }
    return out


def merges(releases) -> list[dict]:
    """Terms absorbed into another, as the files themselves record it.

    OBO gives the surviving term an `alt_id` naming what it absorbed. A term that
    stops appearing AND turns up as another term's `alt_id` was merged; one that stops
    appearing with no such record was deleted, and the two are reported apart because
    they mean different things to anything downstream holding the old id.
    """
    lifecycle = term_lifecycle(releases)
    found = []
    for t in range(1, len(releases)):
        aliases = releases[t].parsed.aliases or {}
        absorbed = {a: survivor for survivor, alist in aliases.items() for a in alist}
        for term, info in lifecycle.items():
            if info["obsoleted_at"] != t:
                continue
            survivor = absorbed.get(term)
            found.append({
                "term": term,
                "at": t,
                "release": releases[t].label,
                "merged_into": survivor,
                "kind": "merged" if survivor else "removed",
            })
    return found


def release_diff(a: Release, b: Release) -> dict:
    """What changed between two releases."""
    added_terms = sorted(b.terms - a.terms)
    removed_terms = sorted(a.terms - b.terms)
    added_rel = sorted(b.relations - a.relations)
    removed_rel = sorted(a.relations - b.relations)
    return {
        "from": a.label, "to": b.label,
        "n_terms": [a.n_terms, b.n_terms],
        "added_terms": added_terms[:200], "n_added_terms": len(added_terms),
        "removed_terms": removed_terms[:200], "n_removed_terms": len(removed_terms),
        "added_relations": [list(r) for r in added_rel[:200]],
        "n_added_relations": len(added_rel),
        "removed_relations": [list(r) for r in removed_rel[:200]],
        "n_removed_relations": len(removed_rel),
        "unchanged": not (added_terms or removed_terms or added_rel or removed_rel),
    }


def navigate(releases, *, gate=None) -> dict:
    """Walk the series and report only the releases that were a surprise.

    The navigator is idle by default: a run of ordinary growth costs one gate
    observation per step and no flow work. A release that reorganises a branch fires
    the gate, and what comes back is localized to the region that changed.

    A single navigator carries its baseline across the run, so it is built here and
    used once. Reusing one across two series would blend the first series' baseline
    into the second.
    """
    from rexgraph.flow.navigator import FieldNavigator

    temporal, vocab = temporal_complex(releases)
    nav = FieldNavigator(gate=gate)
    steps = []
    for t in range(int(temporal.T)):
        rex = temporal.reconstruct_at(t)
        out = nav.step(rex)
        steps.append({
            "t": t,
            "release": releases[t].label if t < len(releases) else str(t),
            "surprise": bool(out.get("event")),
            "n_region": int(len(out["region"])) if out.get("region") is not None else 0,
        })
    return {
        "n_releases": len(releases),
        "n_terms_total": len(vocab),
        "flow_calls": int(nav.flow_calls),
        "surprising": [s for s in steps if s["surprise"]],
        "steps": steps,
        "reading": ("a step is reported only when the change at it is a surprise "
                    "against the trend, so steady growth is silent by design"),
    }


def summary(releases) -> dict:
    """Everything the series says about itself, in one call."""
    life = term_lifecycle(releases)
    merged = merges(releases)
    return {
        "n_releases": len(releases),
        "releases": [{"label": r.label, "n_terms": r.n_terms,
                      "n_relations": len(r.relations)} for r in releases],
        "n_terms_total": len(life),
        "n_introduced": sum(1 for v in life.values() if v["introduced"]),
        "n_obsoleted": sum(1 for v in life.values()
                           if v["obsoleted_at"] is not None),
        "merges": [m for m in merged if m["kind"] == "merged"],
        "removals": [m for m in merged if m["kind"] == "removed"],
        "diffs": [release_diff(a, b)
                  for a, b in zip(releases, releases[1:], strict=False)],
    }
