"""Static checks on the single-file frontend.

app.jsx is hyperscript with no build step, so nothing type-checks it and nothing
resolves its identifiers before a browser does. A render body that reads another
component's state is valid JavaScript: it parses, it ships, and it throws
ReferenceError the first time that screen mounts. These checks are the substitute
for the compiler the file does not have.
"""
from __future__ import annotations

import collections
import pathlib
import re

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "frontend" / "app.jsx"

# Names the file may use without declaring: language keywords, browser globals, and
# the React bindings destructured at the top of the file.
GLOBAL = set("""
function null true false this new typeof instanceof return var let const if else for
while do switch case break continue throw try catch finally delete void in of class
extends super yield await async
window document Math JSON Object Array String Number Boolean Date Promise RegExp Error
Set Map Symbol console alert confirm prompt setTimeout setInterval clearTimeout
clearInterval requestAnimationFrame fetch Blob URL FormData File FileReader EventSource
Headers AbortController encodeURIComponent decodeURIComponent parseInt parseFloat isNaN
isFinite localStorage sessionStorage navigator location history undefined NaN Infinity
TextDecoder TextEncoder
React ReactDOM h useState useEffect useRef useMemo useCallback
""".split())


def _mask(s: str, fill: str) -> str:
    """Replace a span with filler, keeping newlines so line numbers stay true."""
    return "".join(ch if ch == "\n" else fill for ch in s)


def _blank(s: str) -> str:
    """Mask strings, comments and regex literals so brace matching and identifier
    scanning see code only. A `/` opens a regex when the last significant character
    cannot end an expression.

    String and regex bodies become `0`, not spaces: a blanked-to-space `?"a":b` reads
    as an empty ternary branch, which is one of the things these checks look for.
    Comments become spaces because nothing inside them is an operand.
    """
    out: list[str] = []
    i, n, last = 0, len(s), ""
    while i < n:
        c = s[i]
        if c in "\"'`":
            q, j = c, i + 1
            while j < n and s[j] != q:
                j += 2 if s[j] == "\\" else 1
            out.append(_mask(s[i:j + 1], "0")); i = j + 1; last = "0"
        elif c == "/" and i + 1 < n and s[i + 1] == "/":
            j = s.find("\n", i)
            j = n if j < 0 else j
            out.append(_mask(s[i:j], " ")); i = j
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i) + 2
            out.append(_mask(s[i:j], " ")); i = j
        elif c == "/" and last not in ")]}" and not (last or "(").isalnum():
            j = i + 1
            while j < n and s[j] != "/":
                j += 2 if s[j] == "\\" else 1
            out.append(_mask(s[i:j + 1], "0")); i = j + 1; last = "0"
        else:
            out.append(c)
            if not c.isspace():
                last = c
            i += 1
    return "".join(out)


def _declared(body: str) -> set[str]:
    """Every name a scope binds: var/let/const, function declarations, parameters,
    catch bindings."""
    names: set[str] = set()
    for mm in re.finditer(r"\b(?:var|let|const)\s+([^;\n]+)", body):
        for part in mm.group(1).split(","):
            nm = re.match(r"\s*([A-Za-z_$][\w$]*)", part)
            if nm:
                names.add(nm.group(1))
    names |= set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", body))
    for mm in re.finditer(r"function\s*[A-Za-z_$\w]*\s*\(([^)]*)\)", body):
        for p in mm.group(1).split(","):
            nm = re.match(r"\s*([A-Za-z_$][\w$]*)", p)
            if nm:
                names.add(nm.group(1))
    names |= set(re.findall(r"catch\s*\(\s*([A-Za-z_$][\w$]*)", body))
    return names


def _functions(clean: str) -> dict[str, tuple[int, int, str]]:
    """Top-level `function NAME(...)` spans, located by brace matching."""
    out = {}
    for m in re.finditer(r"^function ([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{", clean, re.M):
        depth, i = 1, m.end()
        while i < len(clean) and depth:
            if clean[i] == "{":
                depth += 1
            elif clean[i] == "}":
                depth -= 1
            i += 1
        out[m.group(1)] = (m.start(), i, m.group(2))
    return out


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


def test_every_identifier_resolves(src):
    """No component reads a name it does not own.

    Catches the failure where a screen's markup and the state it reads drift into
    different function bodies during an edit: the file still parses, and the screen
    throws on mount.
    """
    clean = _blank(src)
    funcs = _functions(clean)
    assert len(funcs) > 40, "function scan found too little; the matcher is broken"

    scopes = {n: _declared(clean[a:b]) for n, (a, b, _) in funcs.items()}
    for n, (_, _, params) in funcs.items():
        scopes[n] |= {p.strip() for p in params.split(",") if p.strip()}

    spans = sorted((a, b) for a, b, _ in funcs.values())
    outside = "".join(clean[e:s] for (s, _), (_, e) in
                      zip(spans + [(len(clean), 0)], [(0, 0)] + spans, strict=False))
    module = _declared(outside) | set(funcs)

    owner = collections.defaultdict(set)
    for n, names in scopes.items():
        for x in names:
            owner[x].add(n)

    bad = []
    for n, (a, b, _) in funcs.items():
        used = set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*(?![\w$]*\s*:)", clean[a:b]))
        line = src[:a].count("\n") + 1
        for u in sorted(used):
            if u in scopes[n] or u in module or u in GLOBAL:
                continue
            other = sorted(owner.get(u, set()) - {n})
            where = f"declared only in {', '.join(other)}" if other else "declared nowhere"
            bad.append(f"{n} (line {line}) uses '{u}': {where}")
    assert not bad, "unresolved identifiers:\n  " + "\n  ".join(bad)


def test_no_react_node_in_string_concatenation(src):
    """`"x " + h(Icon, ...)` renders the literal text [object Object]. It is valid
    JavaScript, so only a reader or this check catches it."""
    hits = [f"line {src[:m.start()].count(chr(10)) + 1}"
            for m in re.finditer(r"\+\s*h\(Icon|h\(Icon,\{[^}]*\}\)\s*\+", src)]
    assert not hits, "React node concatenated into a string at: " + ", ".join(hits)


def test_every_rendered_component_exists(src):
    """A capitalised name passed to h() has to be a component defined in this file.
    `Comp` is the screen chosen from TAB_MAP at runtime."""
    defined = set(re.findall(r"function ([A-Z]\w*)\(", src)) | {"Comp"}
    used = set(re.findall(r"h\(([A-Z]\w*)\s*,", src))
    assert not (used - defined), f"rendered but never defined: {sorted(used - defined)}"


def test_no_dangling_ternary(src):
    """Lifting a node out of a `cond ? a : b` leaves `? :` or `: )`, which parses as
    a syntax error only sometimes and as a silently empty branch otherwise."""
    clean = _blank(src)
    hits = [f"line {clean[:m.start()].count(chr(10)) + 1}"
            for m in re.finditer(r"\?\s*[:\)]|:\s*\)", clean)]
    assert not hits, "empty ternary branch at: " + ", ".join(hits)


def test_every_screen_and_sub_tab_renders():
    """Shallow-render each screen, in each of its sub-tabs, with stub hooks.

    Catches what the static checks cannot see: a prop shape a primitive rejects, a
    component that throws while building its tree. It is the complement of
    `test_every_identifier_resolves`, which reaches branches this cannot because
    they need data to render.
    """
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = pathlib.Path(__file__).parent / "render_screens.js"
    r = subprocess.run([node, str(harness)], capture_output=True, text=True, timeout=120)
    out = r.stdout.strip()
    assert "LOAD_FAIL" not in out, out
    fails = [ln for ln in out.splitlines() if ln.startswith("FAIL ")]
    assert not fails, "screens that do not render:\n  " + "\n  ".join(fails)
    assert "OK" in out, out
    views = int(next(ln for ln in out.splitlines() if ln.startswith("VIEWS ")).split()[1])
    assert views >= 40, f"only {views} views rendered; the sub-tab walk is not running"
