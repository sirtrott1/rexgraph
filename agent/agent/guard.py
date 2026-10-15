"""Check generated text against configured regular expression rules.

OutputGuard reports matching spans in a complete reply or growing token stream.
Rules may exempt allowed text and define replacements for automatic correction.
The supplied relational complex rules normalize selected project terminology.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["GuardRule", "OutputGuard", "RELATIONAL_COMPLEX_RULES", "relational_complex_guard"]


@dataclass
class GuardRule:
    name: str
    pattern: str                              # regex marking a violation
    message: str
    fix: str | None = None                 # replacement (regex sub) for auto correct
    allow: str | None = None               # regex whose matches inside a hit are exempt
    flags: int = re.IGNORECASE

    def __post_init__(self):
        self._re = re.compile(self.pattern, self.flags)
        self._allow = re.compile(self.allow, self.flags) if self.allow else None

    def _exempt(self, hit: str) -> bool:
        return bool(self._allow and self._allow.search(hit))

    def violations(self, text: str) -> list[dict]:
        out = []
        for m in self._re.finditer(text or ""):
            if self._exempt(m.group(0)):
                continue
            out.append({"rule": self.name, "span": [m.start(), m.end()],
                        "matched": m.group(0), "message": self.message})
        return out

    def apply_fix(self, text: str) -> str:
        """Substitute `fix` for every non exempt hit.

        `fix` is expanded as a regex template, so it may carry group references and a
        rule can preserve what it matched. That is how the plural is kept: a fix of
        r"relational complex\\1" against a pattern ending "(es|)" yields "relational
        complexes" for a plural hit and "relational complex" for a singular one. A fix
        with no references expands to itself.
        """
        if self.fix is None or not text:
            return text or ""

        def _sub(m):
            if self._exempt(m.group(0)):
                return m.group(0)
            try:
                return m.expand(self.fix)
            except re.error:
                return self.fix          # a fix that is not a valid template is literal
        return self._re.sub(_sub, text)


class OutputGuard:
    """A set of GuardRules applied to generated text or a token stream."""

    def __init__(self, rules: Iterable[GuardRule]):
        self.rules = list(rules)

    def check(self, text: str) -> list[dict]:
        v: list[dict] = []
        for r in self.rules:
            v.extend(r.violations(text or ""))
        return sorted(v, key=lambda d: (d["span"][0], d["rule"]))

    def fix(self, text: str) -> tuple[str, list[dict]]:
        """Return (corrected_text, violations_found_in_the_original)."""
        found = self.check(text)
        out = text or ""
        for r in self.rules:
            out = r.apply_fix(out)
        return out, found

    def scan_stream(self, chunks: Iterable[str]):
        """Feed generated chunks as they arrive. Yields (accumulated, new_violations) after each
        chunk, so a violation is caught the instant the offending text completes: not after the
        whole generation finishes. A phrase split across chunks ('chain ' + 'complex') is caught
        when the second chunk lands."""
        acc = ""
        seen = 0
        for ch in chunks:
            acc += ch
            v = self.check(acc)
            new, seen = v[seen:], len(v)
            yield acc, new


#### presets
RELATIONAL_COMPLEX_RULES = [
    # "(es|)" rather than "(es)?" so the group always participates and the fix template
    # can echo it back; an optional group that does not match makes m.expand raise.
    GuardRule("relational-complex-term", r"chain[ -]complex(es|)",
              "use 'relational complex', not 'chain complex'",
              fix=r"relational complex\1"),
    GuardRule("relational-complex-hyphen", r"relational-complex",
              "write 'relational complex' unhyphenated, even as an adjective",
              fix="relational complex"),
]


def relational_complex_guard() -> OutputGuard:
    """Return the supplied terminology guard for the phrase relational complex.

    The rules normalize selected variants of that phrase and preserve chain condition.
    """
    return OutputGuard(RELATIONAL_COMPLEX_RULES)
