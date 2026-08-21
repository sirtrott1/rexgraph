"""The grammar as a lookup a document can actually use.

WordNet records the frames (39 of them) and which each SENSE admits (25,100 links). A
document has neither senses nor sense ids (it has words) so the usable form is
`word -> the frames any of its senses admits`, which is 11,585 words.

A frame IS an oriented relation and that is what makes it the orientation source: the
verb heads it and the argument slots share. "Somebody ----s somebody something" is arity
4, so the verb carries the `-1` and agent, recipient and theme each carry `1/3` because
the boundary column says so. Part-of-speech tagging and dependency parsing approximate
exactly this, statistically, from labelled corpora; here it is read.

`head_of` is deliberately narrow. A relation of arity k is headed by the token admitting
a frame of arity k, and it makes a claim ONLY when exactly one token does. Zero is an
honest absence (nothing in the span subcategorises for that shape) and several is real
ambiguity, and neither is something to break by picking. A caller that gets no claim
keeps positional orientation and records that no frame governed, which is a different
statement from "the first token heads it" and has to stay distinguishable.

Any grammar with a `head_of` implements this. For a programming language it is the
parser, which answers exactly and never abstains (an assignment heads on its target, a
call on its callee) so code is the easy case rather than a second implementation.
"""
from __future__ import annotations

__all__ = ["FrameGrammar", "wordnet_grammar"]


class FrameGrammar:
    """Which participant heads a relation, from recorded subcategorisation frames.

    `frames` is `{frame_id: (subcategorization_string, arity)}` and `word_frames` is
    `{word: frozenset(frame_id)}`. Both are facts read out of the lexicon; nothing here
    is trained or tuned.
    """

    __slots__ = ("frames", "word_frames", "_by_arity")

    def __init__(self, frames, word_frames):
        self.frames = {str(k): (str(v[0]), int(v[1])) for k, v in frames.items()}
        self.word_frames = {str(w).lower(): frozenset(map(str, f))
                            for w, f in word_frames.items()}
        #: word -> {arity: frame_id}. A word admitting two frames of the SAME arity is
        #: not ambiguous for this purpose: both say the head is this word.
        self._by_arity = {}
        for w, fids in self.word_frames.items():
            m = {}
            for f in fids:
                a = self.frames.get(f, ("", 0))[1]
                if a >= 2:
                    m.setdefault(a, f)
            if m:
                self._by_arity[w] = m

    def arity_of(self, frame_id):
        """The relation arity a frame states: the verb plus its slots."""
        return self.frames.get(str(frame_id), ("", 0))[1]

    def head_of(self, tokens):
        """`(index, frame_id)` for the token that heads this relation, or None.

        The relation's arity is its number of DISTINCT participants, matching the
        boundary column. A claim is made only when exactly one token subcategorises for
        that arity; otherwise there is no claim, which the caller must record rather than
        replace with a guess.
        """
        toks = [str(t).lower() for t in tokens]
        k = len({t for t in toks})
        if k < 2:
            return None                      # a witness has one participant and no frame
        hits = [(i, self._by_arity[t][k])
                for i, t in enumerate(toks)
                if t in self._by_arity and k in self._by_arity[t]]
        if len(hits) != 1:
            return None                      # nothing subcategorises, or several do
        return hits[0]

    def __len__(self):
        return len(self.word_frames)

    def __repr__(self):
        return (f"FrameGrammar({len(self.frames)} frames, "
                f"{len(self.word_frames)} words)")


def wordnet_grammar(wn):
    """Build a `FrameGrammar` from what `load_wordnet` returned.

    The sense layer is collapsed to words here because that is the join a document can
    make. It loses which SENSE admitted the frame, which is real information, but sense
    disambiguation is a separate reading, and carrying an unresolved sense id into an
    orientation decision would be carrying a question, not an answer.
    """
    from agent.adapters.lexical import frame_slots

    frames = {}
    for fid, text in (wn.get("frames") or {}).items():
        slots, _unknown = frame_slots(text)
        frames[fid] = (text, 1 + len(slots))

    lemma_of = wn.get("lemma_of") or {}
    sense_of = wn.get("sense_of") or {}
    forms_of = wn.get("forms_of") or {}
    word_frames = {}
    for sid, fids in (wn.get("frames_of") or {}).items():
        eid, _ss = sense_of.get(sid, (None, None))
        if eid is None:
            continue
        # the lemma AND its recorded inflections, because a document is inflected text
        # and a lexicon keyed only on lemmas misses "chased" and "gave" entirely
        for written in [lemma_of.get(eid), *(forms_of.get(eid) or ())]:
            if written:
                word_frames.setdefault(str(written).lower(), set()).update(fids)
    return FrameGrammar(frames, word_frames)
