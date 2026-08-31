# Repository Instructions

## Working together

- Two agents share this repository: Codex owns `rexgraph/`, and Claude owns `agent/`. Root and build files are shared: they require Art's authorization, and both agents must agree on the diff before it is committed.
- Coordinate through `~/co-writers`: `say <who> "text" --to <who>` writes a message, `read N` shows history, and `wait <who>` blocks until the other writes. The log uses the activity-journal format, so the conversation is itself a relational complex readable by `agent_complex.act_complex`.
- Ownership limits edits, not inspection. Report relevant discoveries across lane boundaries, especially when they affect code owned by the other agent.
- Before implementing behavior, announce what you intend to build, ask whether an implementation or equivalent already exists, and search the current tree. Current source and tests outrank archived or reference code. Port behavior and invariants in small slices; never bulk-copy an older tree over current work.

## Core ownership and invariants

- Codex owns `rexgraph/`. Do not edit `agent/`; coordinate any root, build, or shared file before touching it.
- Preserve edge/relation primacy, `B_k B_(k+1) = 0`, `Geometry_k = Topology_(k+1)`, exact integer/rational paths, grade shapes, and exact-zero behavior for a missing upper grade. Hodge down/up operators must remain symmetric positive semidefinite and mutually annihilating where the chain law applies.
- A new abstraction must use current compiled operators and preserve newer channel-tower, ternary, coboundary-volume, and exact-rank behavior. Do not regress a live fast or exact path to an archived fallback.
- Core tests that exercise optional higher packages must import-skip at the narrow test site. Never add Agent as a core dependency.

## Agent ownership and invariants

- Claude owns `agent/`. Do not edit `rexgraph/`; coordinate any root, build, or shared file before touching it.
- A worker is any capability, not necessarily a language model. `add_worker` accepts a callable with capability `predict`, `score`, `embed`, `analyze`, or `transform`; a chat model is only the `generate` case. Do not add an LLM-shaped assumption to the member contract.
- A bee is reachable in exactly two ways: an in-process callable or an OpenAI-compatible endpoint. `external_bee` is not a third transport; it serves that same endpoint and lets the responder poll rather than listen, so a harness that can call out but cannot be called still attaches unchanged. Extend the contract if a genuinely new transport is needed rather than special-casing a caller.
- Acts carry an orientation. `activity.record` takes `on` and `flow`. Direction is positional (which participant holds the single `-1`), because the library canonicalizes a column against its negation. Encoding a read as a negated write silently produces a write.
- `rcdb.copy_record` is the one place a record crosses between stores. Both migration and courier paths use it; do not add a second copy path.
- `agent.client.RexClient` is the only client for `/rex/v1`. Do not write another.
- The security boundary engages at the socket. `mcp_tools.Context`, `server/scope.py`, and `interfaces.LocalIdentity` define in-process execution as one trusted operator. If a boundary is required between in-process callers, state that requirement explicitly. Do not add a second security model beside the existing bearer, workspace, and audit-chain model without a stated trust-boundary reason.
- `get_hive()` returns a process-wide singleton. It refuses a second caller that registers the same tool name rather than silently rebinding it; separate hives are how two callers coexist.

## Isolation and build truth

- Never install with `/home/art/micromamba/envs/rexgraph/bin/python` or its `pip`. Read and test that environment, but never mutate it. Every install or build experiment gets a dedicated virtual environment under `/tmp`.
- A repository-root test can mix source Python with extensions from an editable install. Before claiming a cold build, report the Git HEAD, `rexgraph.__file__`, at least one compiled module's `__file__`, the interpreter path, and the installed metadata version. Verify imports from a neutral directory when testing a non-editable wheel.
- `run_core_tests.sh` does not by itself prove a cold build. Distinguish build, import, collection, and execution claims.

## Traps that cost real time

- Tests can pass here because this machine has both packages installed. A defect that appears only on a core-only or non-editable install is invisible to suites run in place. Validate packaging claims from a cold clone.
- Identity comparisons on a stored signature must name the fields they compare. A denylist over the whole signature can silently disagree across backends because optional analytics differ between a store that keeps the object and one that serializes it.
- A silent watcher looks exactly like a quiet channel. If a monitor or filter emits nothing, prove that it can emit before concluding that nothing happened.

## Verification and commits

- Core suite: `sh run_core_tests.sh -q`. Agent suite: `~/micromamba/envs/rexgraph/bin/python -m pytest agent/tests -q`. Record measured counts and environment-specific skips or failures; never substitute a promised baseline for an observed number.
- Never run `git add -A`. Commit with an explicit pathspec. Use a `scope: lowercase subject`; the prose body explains why and ends with test counts. Do not add `Co-authored-by`, `Signed-off-by`, or an AI trailer. The author remains `sirtrott1`. Do not push or delete Art's files without asking.
