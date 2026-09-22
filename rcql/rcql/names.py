"""Canonical operator names and collision checks shared by both registries."""
from __future__ import annotations

import re


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z", re.ASCII)
_SYNTAX_NAMES = frozenset({"NOT", "TRUE", "FALSE", "NONE"})

# Names retired in v1.2.1 for names that state the implementation or the relational method.
RETIRED = {
    "MARKOV_VIEW": "PARTICIPATION_WALK",
    "PAGERANK": "RESOLVENT_RANK for the relational ranking, or PAGERANK_ITERATION on a PARTICIPATION_WALK",
    "PAGERANK_EXACT": "RESOLVENT_RANK for the relational ranking, or PAGERANK_SOLVE on a PARTICIPATION_WALK",
}


def canonical_name(name: str) -> str:
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("operator names must be ASCII identifiers starting with a letter")
    key = name.upper()
    if key == "NOT":
        raise ValueError(f"operator name {key!r} conflicts with expression syntax")
    return key


def insert_unique(registry, name, value):
    """Reject replacement, including an alternate spelling of an existing name."""
    key = canonical_name(name)
    if key in _SYNTAX_NAMES:
        raise ValueError(f"operator name {key!r} conflicts with expression syntax")
    if key in registry:
        raise ValueError(f"RCQL operator {key!r} is already registered")
    registry[key] = value
    return value

