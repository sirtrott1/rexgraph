"""Description of Rex values exposed through RCQL."""
from __future__ import annotations

import contextlib
from typing import Any

from rexgraph.graded_boundary import graded_boundaries_from_rex


def describe_rex(rex) -> dict[str, Any]:
    """Return the structural shape of a Rex without computing dense operators."""
    rex._ensure_clean()
    B = graded_boundaries_from_rex(rex)
    sizes = [int(B[0].shape[0])] + [int(M.shape[1]) for M in B] if B else [0]
    out = {
        "kind": "Rex",
        "dimension": max(0, len(sizes) - 1),
        "grades": tuple(range(len(sizes))),
        "cells": tuple(sizes),
        "boundaries": tuple(tuple(int(x) for x in M.shape) for M in B),
    }
    with contextlib.suppress(Exception):        # betti is optional in a description
        out["betti"] = tuple(int(x) for x in rex.betti)
    for name in ("nV", "nE", "nF"):
        if hasattr(rex, name):
            out[name] = int(getattr(rex, name))
    return out
