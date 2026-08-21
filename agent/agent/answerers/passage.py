"""Passage answerer over a document corpus.

Reports which retrieved spans hold which query terms, with document, section and byte
offsets. A document complex records co-occurrence, so it locates terms and asserts no
predication.
"""
from __future__ import annotations

from agent.answerers import _question as Q

__all__ = ["PassageAnswerer"]

class PassageAnswerer:
    """Retrieved spans, with what each one exactly contains and where it is."""

    capability = "passage"

    def __init__(self, *, max_passages: int = 3, snippet: int = 320):
        self._max = int(max_passages)
        self._snippet = int(snippet)

    @staticmethod
    def _content_terms(query: str) -> list[str]:
        """The query's terms that can localise a passage.

        The gate is the LEXICON'S OWN record of which words are function words: every
        recorded reading an article, conjunction, determiner, particle, postposition,
        preposition or pronoun, plus the words that name a relation to some structure,
        which are question vocabulary. Both are facts about a structure rather than a
        curated list of English, which is what this used to be.

        With no lexicon on disk both sets are empty and nothing is excluded. That is the
        canonical stance and not a silent fallback: `record_response` weights seeds by
        1/deg and excludes nothing for the same reason, since a term in most records
        already says so by its degree.
        """
        gate = Q.function_words() | Q.interface_words()
        seen, out = set(), []
        for w in Q.tokens(query):
            if w not in gate and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def answer(self, query: str, sections, *, document: str = "") -> dict:
        terms = self._content_terms(query)
        if not terms:
            return {"answered": False,
                    "reason": "the question names no term to locate",
                    "capability": self.capability}
        if not sections:
            return {"answered": False, "reason": "retrieval returned no passage",
                    "capability": self.capability}

        from rexgraph.corpus_profile import TEXT, tokenize
        want = set(terms)
        hits = []
        for s in sections:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            # WHICH terms this span holds is a set intersection over the span's own
            # tokens: exact, and the only claim the co-occurrence structure licenses.
            present = {w for w, _a, _b in tokenize(text, TEXT)} & want
            if not present:
                continue
            hits.append({
                "document": document or s.get("document", ""),
                "section": s.get("section_id"),
                "layer": s.get("layer"),
                "span": s.get("span"),
                "contains": sorted(present),
                "missing": sorted(want - present),
                "reading": s.get("reading"),
                "agree": s.get("agree"),
                "channels": s.get("channels"),
                "channel_names": s.get("channel_names"),
                "text": text[:self._snippet] + ("…" if len(text) > self._snippet else ""),
            })
            if len(hits) >= self._max:
                break

        if not hits:
            return {"answered": False,
                    "reason": f"no retrieved passage contains any of {terms[:6]}",
                    "terms": terms, "capability": self.capability}
        covered = sorted({t for h in hits for t in h["contains"]})
        return {"answered": True, "terms": terms, "covered": covered,
                "uncovered": sorted(want - set(covered)),
                "passages": hits, "capability": self.capability}

    def as_worker(self):
        def handler(data):
            d = data if isinstance(data, dict) else {}
            return self.answer(str(d.get("query", data)), d.get("sections") or [],
                               document=str(d.get("document", "")))
        return handler, self.capability, "answerer:passage"


def render(result: dict) -> str:
    """The answer as text. Extractive: every line is a span the corpus can reproduce."""
    if not result.get("answered"):
        return ""
    lines = []
    for i, h in enumerate(result["passages"], 1):
        where = h.get("section") or "?"
        span = h.get("span")
        # a span is (byte offset, byte length), the address `section_text` seeks to,
        # not a start/end pair.
        at = f" bytes {span[0]}+{span[1]}" if span and len(span) >= 2 else ""
        doc = f"{h['document']}, " if h.get("document") else ""
        lines.append(f"[{i}] {doc}{where}{at}: contains "
                     f"{', '.join(h['contains'])}")
        lines.append(f"    {h['text']}")
        prof, names = h.get("channels"), h.get("channel_names")
        if prof and names:
            lines.append("    " + "  ".join(f"{n}={v:.3f}"
                                            for n, v in zip(names, prof, strict=True)))
        if h.get("reading"):
            agreed = " (both readings agree)" if h.get("agree") else ""
            lines.append(f"    found by: {h['reading']}{agreed}")
    if result.get("uncovered"):
        lines.append(f"\nNot located in any retrieved passage: "
                     f"{', '.join(result['uncovered'])}.")
    lines.append("\nThese are spans that contain your terms. The corpus records "
                 "co-occurrence, so it can show you where something is discussed, "
                 "not assert what it means.")
    return "\n".join(lines)
