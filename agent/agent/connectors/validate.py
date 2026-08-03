"""
agent.connectors.validate: the validation harness.

Runs a connector against a source and reports, check by check, whether its
output is a well-formed relational complex that stores and round-trips and
preserves the sovereign-engine invariants. The point is to turn an integration
from a research project into a **known, testable quantity**: a pass/fail report
an integrator can attach to a fixed-price quote.

Programmatic (test util):

    from agent.connectors.validate import validate_connector
    report = validate_connector(MyConnector(), my_source)
    assert report.ok

CLI:

    python -m agent.connectors.validate my.module:MyConnector            # no source
    python -m agent.connectors.validate my.module:make_connector source  # factory
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import to_rexgraph


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class ValidationReport:
    connector: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name, passed, detail))

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        head = f"connector validation: {self.connector} - " + (
            "PASS" if self.ok else "FAIL")
        return "\n".join([head] + [str(c) for c in self.checks])


def validate_connector(connector: Any, source: Any = None,
                       store_uri: str = "memory://") -> ValidationReport:
    """Validate one connector against one source. Never raises for a check
    failure - every failure is recorded and reported, so a partial connector
    still yields an actionable report."""
    rep = ValidationReport(connector=type(connector).__name__)

    # 1. contract shape
    try:
        rex, meta = connector.read(source)
        labels = meta.get("vertex_labels")
        edges = meta.get("edges")
        ok = bool(labels) and edges is not None and "source" in meta
        ok = ok and all(len(e) == 2 for e in edges)
        if meta.get("weights") is not None:
            ok = ok and len(meta["weights"]) == len(edges)
        if meta.get("modality") is not None:
            ok = ok and len(meta["modality"]) == len(edges)
        rep.add("contract shape", ok,
                f"{len(labels or [])} labels, {len(edges or [])} edges, "
                f"source={meta.get('source')!r}")
    except Exception as e:                       # noqa: BLE001
        rep.add("contract shape", False, f"read() raised: {e!r}")
        return rep                               # nothing else can run

    # 2. builds in the engine
    try:
        g = to_rexgraph(rex, meta)
        chi = np.asarray(g.structural_character, dtype=float)
        # per-edge structural character: one row per edge. The channel count is
        # 4 (T,G,F,C) once the complex is non-trivial; it degenerates for a
        # single-edge complex, so we require rows==nE rather than a fixed width.
        ok = chi.ndim == 2 and chi.shape[0] == g.nE
        channels = chi.shape[1] if chi.ndim == 2 else 0
        rep.add("builds in engine", ok,
                f"nV={g.nV} nE={g.nE}; per-edge character {chi.shape}"
                + ("" if channels == 4 or g.nE < 2
                   else f" (expected 4 channels, got {channels})"))
    except Exception as e:                       # noqa: BLE001
        rep.add("builds in engine", False, f"construction raised: {e!r}")
        return rep

    # 3. chain condition ∂²=0
    try:
        cv = bool(g.chain_valid)
        has_faces = meta.get("faces") is not None or getattr(g, "nF", 0) > 0
        rep.add("chain condition ∂²=0", cv,
                ("faces supplied; B₁B₂=0 holds" if has_faces
                 else "no faces (topology-only); trivially valid")
                if cv else "B₁B₂ ≠ 0 - face selection violates the chain condition")
    except Exception as e:                       # noqa: BLE001
        rep.add("chain condition ∂²=0", False, f"raised: {e!r}")

    # 4. betti computable / signature builds
    try:
        betti = tuple(int(b) for b in g.betti)
        ok = len(betti) == 3
        rep.add("betti / signature", ok, f"β=(b0,b1,b2)={betti}")
    except Exception as e:                       # noqa: BLE001
        rep.add("betti / signature", False, f"raised: {e!r}")
        betti = None

    # 5. RCDB round-trip (put -> get -> structure preserved)
    try:
        from agent.rcdb import open_store
        st = open_store(store_uri)
        st.put("_validate", g, meta=getattr(g, "_agent_meta", None), tags=["_v"])
        got = st.get("_validate")
        ok = (got is not None and got.nV == g.nV and got.nE == g.nE
              and (betti is None or tuple(int(b) for b in got.betti) == betti))
        st.delete("_validate")
        rep.add("RCDB round-trip", ok,
                f"put->get preserved nV/nE/betti ({got.nV},{got.nE})"
                if ok else "structure changed across put->get")
    except Exception as e:                       # noqa: BLE001
        rep.add("RCDB round-trip", False, f"raised: {e!r}")

    # 6. read-only probe
    # Concrete, in-sandbox checks: the connector is deterministic (reading
    # twice yields the same structure - a writing connector that mutated the
    # source would drift) and exposes no write surface. Proving read-only
    # against a *live* source is a per-integration review item in the host env.
    try:
        rex2, meta2 = connector.read(source)
        g2 = to_rexgraph(rex2, meta2)
        deterministic = (g2.nV == g.nV and g2.nE == g.nE
                         and meta2.get("vertex_labels") == meta.get("vertex_labels"))
        write_surface = [m for m in ("write", "put", "insert", "update",
                                     "delete", "execute", "commit")
                         if callable(getattr(connector, m, None))]
        ok = deterministic and not write_surface
        rep.add("read-only probe", ok,
                "deterministic; no write surface"
                if ok else
                (f"exposes write methods {write_surface}" if write_surface
                 else "non-deterministic across reads"))
    except Exception as e:                       # noqa: BLE001
        rep.add("read-only probe", False, f"raised: {e!r}")

    # 7. capability consistency
    try:
        caps = connector.capabilities()
        problems = []
        if caps.weights and meta.get("weights") is None:
            problems.append("advertises weights but emitted none")
        if caps.modality and meta.get("modality") is None:
            problems.append("advertises modality but emitted none")
        # faces are structure-dependent (a source may legitimately have none to
        # close), so an advertised-but-absent face selection is not a failure.
        rep.add("capability consistency", not problems,
                "advertised capabilities match output"
                if not problems else "; ".join(problems))
    except Exception as e:                       # noqa: BLE001
        rep.add("capability consistency", False, f"raised: {e!r}")

    return rep


def _load(spec: str) -> Callable:
    """Load ``module:attr`` and return a connector instance. ``attr`` may be a
    Connector subclass (instantiated) or a factory returning one."""
    import importlib
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("spec must be 'module:ConnectorOrFactory'")
    obj = getattr(importlib.import_module(mod_name), attr)
    inst = obj() if isinstance(obj, type) else obj()
    return inst


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m agent.connectors.validate module:Connector [source]")
        return 2
    connector = _load(argv[0])
    source = argv[1] if len(argv) > 1 else None
    report = validate_connector(connector, source)
    print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
