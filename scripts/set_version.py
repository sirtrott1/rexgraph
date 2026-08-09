#!/usr/bin/env python3
"""Set the version in every file that declares one, in one go.

Five files state the version, because each serves something that cannot read the others:
meson needs its own at configure time, pip reads pyproject before any code runs, and
`__version__` has to answer without the package being installed. Editing five files by
hand is why meson.build sat at 1.0.1 through two 1.0.6 releases.

    python scripts/set_version.py 1.0.7        write it everywhere
    python scripts/set_version.py --show       print what each file says now
    python scripts/set_version.py 1.0.7 -n     show the edits without making them

Each file is matched by an anchored pattern rather than a bare string replace, so a
version number appearing in prose, a dependency pin, or a changelog entry is left alone.
A file whose pattern does not match is an error and stops the run: a silent miss here is
exactly the drift this exists to prevent.

`rexgraph/tests/test_version_consistency.py` checks the result. This writes it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: file -> (pattern with ONE capturing group around the version, replacement template)
TARGETS = {
    "pyproject.toml": (re.compile(r'^(version\s*=\s*")[^"]+(")', re.M), r"\g<1>{v}\g<2>"),
    "agent/pyproject.toml": (re.compile(r'^(version\s*=\s*")[^"]+(")', re.M),
                             r"\g<1>{v}\g<2>"),
    "meson.build": (re.compile(r"^(\s*version:\s*')[^']+(',)", re.M), r"\g<1>{v}\g<2>"),
    "rexgraph/__init__.py": (re.compile(r'^(__version__\s*=\s*")[^"]+(")', re.M),
                             r"\g<1>{v}\g<2>"),
    "agent/agent/__init__.py": (re.compile(r'^(__version__\s*=\s*")[^"]+(")', re.M),
                                r"\g<1>{v}\g<2>"),
}

VERSION_RE = re.compile(r"\d+\.\d+\.\d+([abrc]\d+|\.dev\d+|\.post\d+)?")


def current() -> dict[str, str]:
    """What each file says right now."""
    out = {}
    for name, (pattern, _) in TARGETS.items():
        text = (ROOT / name).read_text()
        match = pattern.search(text)
        if match is None:
            out[name] = "<pattern did not match>"
            continue
        # the version is whatever sits between the two captured anchors
        out[name] = text[match.end(1):match.start(2)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", nargs="?", help="the new version, e.g. 1.0.7")
    ap.add_argument("--show", action="store_true", help="print current values and exit")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="report the edits without writing them")
    args = ap.parse_args(argv)

    if args.show or not args.version:
        for name, value in current().items():
            print(f"  {value:<12} {name}")
        distinct = set(current().values())
        print(f"\n{'consistent' if len(distinct) == 1 else 'INCONSISTENT'}: "
              f"{len(distinct)} distinct value(s)")
        return 0 if len(distinct) == 1 else 1

    if not VERSION_RE.fullmatch(args.version):
        print(f"error: {args.version!r} is not a release number "
              f"(N.N.N with an optional a/b/rc/dev/post suffix)", file=sys.stderr)
        return 2

    changed = []
    for name, (pattern, template) in TARGETS.items():
        path = ROOT / name
        text = path.read_text()
        new, count = pattern.subn(template.format(v=args.version), text)
        if count == 0:
            print(f"error: no version line matched in {name}; refusing to continue "
                  f"with a partial update", file=sys.stderr)
            return 3
        if new != text:
            changed.append(name)
            if not args.dry_run:
                path.write_text(new)

    verb = "would set" if args.dry_run else "set"
    print(f"{verb} {args.version} in {len(changed)} file(s):")
    for name in changed:
        print(f"  {name}")
    if not changed:
        print("  (everything already at that version)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
