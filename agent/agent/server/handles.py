"""
agent.server.handles: workspace-scoped file handles.

The tools take files. If a file is named by its path, then whoever can call a tool can
read any file the server process can read, and `files: ["/etc/passwd"]` is a valid
request. A path is a name in the server's namespace, not in the caller's, so it cannot
be checked without a list of what the caller is allowed to reach.

A handle is a name in the caller's namespace. Content is stored under the workspace
that put it there and addressed by the digest of its bytes, so resolution is a lookup
inside one directory rather than a decision about a path:

    workspaces/<workspace>/files/<sha256>

A handle from another workspace does not resolve, because it is not there. That is the
whole isolation argument: no comparison to get wrong, no traversal to normalise, and
knowing another tenant's digest buys nothing. Identical bytes uploaded twice cost one
copy, and a handle is stable across re-upload.

Local use is unchanged. With auth off the server is a single operator on their own
machine, where naming a file by path is the point, so paths pass through. With auth on
they do not, and `REXGRAPH_REQUIRE_HANDLES=1` forces the strict rule regardless.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

#: a handle is the hex sha256 of the content it names
_HANDLE_RE = re.compile(r"\A[0-9a-f]{64}\Z")

#: workspace names become directory names, so they are held to the same rule.
#: A dot is NOT permitted, which looks over-strict for a name and is not: the
#: previous class allowed one, so "." and ".." were valid workspaces and resolved
#: to the parent of the workspace root. That is a namespace two tenants share and
#: neither asked for. There is exactly one rule and it lives here, because the
#: reason it exists is that these names become paths.
WORKSPACE_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
_WORKSPACE_RE = WORKSPACE_RE          # the old private name, kept for callers


def config_dir() -> Path:
    """Where this deployment keeps its credentials and trail."""
    return Path(os.environ.get("REXGRAPH_CONFIG_DIR",
                               os.path.expanduser("~/.config/rexgraph"))).resolve()


def allowed_roots() -> list[str]:
    """The directories a request may name a path inside.

    Built here because three routes had assembled the same list from the same two
    sources, and a fourth needed it: a list that is rebuilt at each seam is a list that
    can be tightened at one and left loose at another.
    """
    roots = [os.path.realpath(os.getcwd()), "/tmp"]
    extra = os.environ.get("REXGRAPH_ALLOWED_DIRS", "")
    if extra:
        roots.extend(d for d in extra.split(":") if d)
    return roots


def path_allowed(path: str) -> bool:
    """Whether a caller-supplied path may be read or written."""
    return path_within(path, allowed_roots())


def path_within(path: str, roots) -> bool:
    """Whether a resolved path lies inside one of `roots`, by PATH COMPONENT.

    `resolved.startswith(root)` is a string test wearing a path test's clothes: it
    admits /tmpfoo for /tmp and /home/artifacts for /home/art, because a prefix of the
    text is not containment in the tree. `Path.is_relative_to` compares components,
    which is the question actually being asked.

    The deployment's own config directory is refused even when it falls inside an
    allowed root, because the common allow-list includes a home directory and that is
    where auth.json, connections.json and the audit journal live. An allow-list whose
    widest entry contains the credential store is not an allow-list.
    """
    try:
        # expanduser BEFORE resolving, because every sink downstream expands it:
        # models/store.py mkdirs os.path.expanduser(path) and models/data.py reads it.
        # Path.resolve() leaves "~" alone, so "~/x" resolved to "<cwd>/~/x", passed as
        # inside the allow-list, and then wrote to the real home directory. Fired: a
        # save_to of "~/x" put 3.6 MB of weights in $HOME.
        #
        # Environment variables are deliberately NOT expanded, because the sinks do not
        # expand them either. A guard that normalises differently from the sink is
        # checking a path nobody writes to.
        r = Path(os.path.expanduser(path)).resolve()
    except (OSError, ValueError):
        return False
    cfg = config_dir()
    if r == cfg or cfg in r.parents:
        return False
    for root in roots:
        try:
            if r == Path(root).resolve() or Path(root).resolve() in r.parents:
                return True
        except (OSError, ValueError):
            continue
    return False


def valid_workspace(name: str) -> bool:
    """Whether a name may be used as a workspace.

    One predicate rather than a regex repeated at each seam. Two copies had already
    drifted apart, and the looser of the two was the one reached from a request
    header, so the strict rule guarded the door nobody was coming through.
    """
    return bool(WORKSPACE_RE.match(name or ""))

#: refuse to store a single file larger than this (bytes)
DEFAULT_MAX_FILE = 512 * 1024 * 1024


class HandleError(ValueError):
    """A handle that does not name content this caller may read."""


def _root() -> Path:
    base = Path(os.environ.get("REXGRAPH_CONFIG_DIR",
                               Path.home() / ".config" / "rexgraph"))
    return base / "workspaces"


#: a stored suffix, kept so the format layer can still tell what a file IS. Content
#: addressing names bytes; every reader in the stack dispatches on extension, so a
#: handle that resolved to a bare digest was content nothing could interpret.
_SUFFIX_RE = re.compile(r"\A\.[A-Za-z0-9][A-Za-z0-9.]{0,15}\Z")


def _files_dir(workspace: str) -> Path:
    if not _WORKSPACE_RE.match(workspace or ""):
        raise HandleError(f"{workspace!r} is not a usable workspace name")
    return _root() / workspace / "files"


def _labels_dir(workspace: str) -> Path:
    """Original filenames, kept beside the content rather than in its name.

    A separate directory because a sidecar living next to the content as
    `<digest>.name` is ambiguous the moment `.name` is itself a real extension.
    """
    if not _WORKSPACE_RE.match(workspace or ""):
        raise HandleError(f"{workspace!r} is not a usable workspace name")
    return _root() / workspace / "labels"


def _suffix_of(name: str) -> str:
    """The extension to store content under, or "" when there is nothing usable.

    Taken from the caller's filename and held to a strict shape, because it becomes
    part of a path. Compound suffixes survive (`.obo.gz` stays `.obo.gz`) since the
    readers dispatch on them.
    """
    base = str(name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in base:
        return ""
    suffix = base[base.index("."):] if base[0] != "." else ""
    return suffix if _SUFFIX_RE.match(suffix) else ""


def mint(workspace: str, data: bytes, *, name: str = "",
         max_bytes: int = DEFAULT_MAX_FILE) -> dict:
    """Store bytes for a workspace and return the handle that names them.

    Content-addressed, so storing the same bytes twice returns the same handle and
    keeps one copy. The original filename is kept beside the content as a label: it is
    reported back to the caller and never used to build a path.
    """
    if len(data) > max_bytes:
        raise HandleError(f"file is {len(data)} bytes, over the {max_bytes} limit")

    digest = hashlib.sha256(data).hexdigest()
    d = _files_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"{digest}{_suffix_of(name)}"
    if not target.exists():
        # write beside and rename, so a reader never sees a partial file under a
        # digest that promises complete content
        tmp = d / f".{digest}.partial"
        tmp.write_bytes(data)
        tmp.replace(target)
    if name:
        labels = _labels_dir(workspace)
        labels.mkdir(parents=True, exist_ok=True)
        (labels / digest).write_text(str(name)[:255], encoding="utf-8")

    return {"handle": digest, "bytes": len(data), "name": name,
            "workspace": workspace}


def mint_path(workspace: str, path: str, **kw) -> dict:
    """Store an existing file's content under a workspace.

    For the local operator and the CLI, which have a real path in hand and want a
    handle they can pass to a tool.
    """
    p = Path(path)
    if not p.is_file():
        raise HandleError(f"{path!r} is not a file")
    return mint(workspace, p.read_bytes(), name=kw.pop("name", p.name), **kw)


def resolve(workspace: str, handle: str) -> Path:
    """The path a handle names, or raise.

    The only way content leaves this module. A handle is a digest and nothing else, so
    it cannot carry a separator, a parent reference or an absolute root, and it is
    joined onto exactly one directory.
    """
    h = str(handle or "").strip().lower()
    if not _HANDLE_RE.match(h):
        raise HandleError("not a handle")
    d = _files_dir(workspace)
    # the digest is the name; whatever suffix it was stored under travels with it, so
    # the format layer can still tell what the content is
    for p in (d / h, *sorted(d.glob(f"{h}.*"))):
        if p.is_file():
            return p
    raise HandleError("no such handle in this workspace")


def label(workspace: str, handle: str) -> str:
    """The filename a handle was stored under, for display."""
    h = str(handle or "").strip().lower()
    if not _HANDLE_RE.match(h):
        return ""
    try:
        side = _labels_dir(workspace) / h
        return side.read_text(encoding="utf-8").strip() if side.is_file() else ""
    except (HandleError, OSError):
        return ""


def listing(workspace: str) -> list[dict]:
    """Every handle a workspace holds."""
    d = _files_dir(workspace)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        stem = p.name.split(".", 1)[0]
        if p.is_file() and _HANDLE_RE.match(stem):
            out.append({"handle": stem, "bytes": p.stat().st_size,
                        "name": label(workspace, stem), "stored_as": p.name})
    return out


def forget(workspace: str, handle: str) -> bool:
    """Drop one handle's content from a workspace."""
    try:
        p = resolve(workspace, handle)
    except HandleError:
        return False
    p.unlink()
    side = _labels_dir(workspace) / str(handle).strip().lower()
    if side.is_file():
        side.unlink()
    return True


def drop_workspace(workspace: str) -> int:
    """Remove every file a workspace holds. Returns how many were removed."""
    d = _files_dir(workspace)
    if not d.is_dir():
        return 0
    n = len([p for p in d.iterdir() if _HANDLE_RE.match(p.name)])
    shutil.rmtree(d)
    return n


def paths_allowed(auth_enabled: bool) -> bool:
    """Whether a raw filesystem path may stand in for a handle.

    Off whenever auth is on, because a token holder is not the operator of the box.
    `REXGRAPH_REQUIRE_HANDLES=1` turns it off unconditionally, for a local server that
    is nonetheless reachable by something the operator does not control.
    """
    if os.environ.get("REXGRAPH_REQUIRE_HANDLES") == "1":
        return False
    return not auth_enabled


def resolve_inputs(workspace: str, files, *, auth_enabled: bool) -> list[str]:
    """Turn a tool's `files` argument into paths, under the policy above.

    The single gate every tool that reads files goes through. A caller that sends a
    path where paths are not allowed gets told that it is a handle that is wanted,
    rather than a file-not-found that would confirm whether the path exists.
    """
    if isinstance(files, (str, bytes)):
        files = [files]
    allow = paths_allowed(auth_enabled)
    out = []
    for item in list(files or []):
        s = str(item)
        if _HANDLE_RE.match(s.strip().lower()):
            out.append(str(resolve(workspace, s)))
        elif allow:
            out.append(s)
        else:
            raise HandleError(
                "inputs must be handles from this workspace; upload the file first "
                "and pass the handle it returns")
    return out
