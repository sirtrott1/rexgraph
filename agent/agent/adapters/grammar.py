"""Orient document relations using frames supplied by the caller.

FrameGrammar maps tokens to declared frame identifiers and relation arities.
A relation has a head only when exactly one token supports its arity.
"""
from __future__ import annotations

__all__ = ["FrameGrammar"]


class FrameGrammar:
    """Which participant heads a relation, from recorded subcategorisation frames.

    `frames` is `{frame_id: (subcategorization_string, arity)}` and `word_frames` is
    `{word: frozenset(frame_id)}`. The caller supplies both mappings.
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
