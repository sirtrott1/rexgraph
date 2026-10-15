"""Document Merkle trees with binary sibling verification paths at each layer.

The document hierarchy supplies sentence, paragraph, chapter and root levels.
Binary trees within sibling sets retain those named levels while bounding each
level's path length by its fanout. Native state retains the data needed to derive leaves.
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["LayerMerkle", "build_merkle", "verify_proof", "layer_chain",
           "pack_merkle", "unpack_merkle"]

DIGEST_SIZE = 32
EMPTY = b"\x00" * DIGEST_SIZE


def _h(*parts):
    """One node hash. Parts are length prefixed so concatenation is unambiguous."""
    d = hashlib.blake2b(digest_size=DIGEST_SIZE)
    for p in parts:
        b = p if isinstance(p, bytes) else str(p).encode("utf-8")
        d.update(len(b).to_bytes(8, "little"))
        d.update(b)
    return d.digest()


def _binary_levels(leaves):
    """Every level of the binary tree over ONE sibling set, leaves first.

    An odd node is paired with itself. That is the standard choice and it matters that it
    is stated: duplicating rather than promoting keeps the level widths a function of the
    count alone, so a verifier reconstructs the shape without being told it.
    """
    lvl = list(leaves) or [EMPTY]
    levels = [lvl]
    while len(lvl) > 1:
        lvl = [_h(lvl[i], lvl[i + 1] if i + 1 < len(lvl) else lvl[i])
               for i in range(0, len(lvl), 2)]
        levels.append(lvl)
    return levels


def _path(levels, idx):
    """`[(sibling, sibling_is_right)]` from a leaf up to the root of one sibling set."""
    out = []
    for lv in levels[:-1]:
        if len(lv) <= 1:
            break
        sib = idx ^ 1
        if sib >= len(lv):
            sib = idx                       # the self paired odd node
        out.append((lv[sib], bool(sib > idx)))
        idx //= 2
    return out


def layer_chain(store, base=None):
    """The layers ordered finest to coarsest, following `refines`.

    A chain is discovered rather than declared: the base is the layer that owns cells
    directly, and each later one is whoever refines the one before it. Several chains can
    exist over one complex (a sentence/paragraph/chapter chain and, say, a speaker
    chain), so the base is nameable.
    """
    owns = [s.name for s in store.values() if not s.is_derived]
    if base is None:
        if len(owns) != 1:
            raise ValueError(
                f"{len(owns)} layers own cells ({sorted(owns)}); name one as base=")
        base = owns[0]
    if base not in store:
        raise ValueError(f"{base!r} is not a sectioning of this complex")
    chain, seen = [base], {base}
    while True:
        nxt = [s.name for s in store.values() if s.refines == chain[-1]]
        if not nxt:
            break
        if len(nxt) > 1:
            raise ValueError(f"{chain[-1]!r} is refined by {sorted(nxt)}; a Merkle chain "
                             f"needs one parent per level")
        if nxt[0] in seen:
            raise ValueError(f"the refines graph cycles at {nxt[0]!r}")
        chain.append(nxt[0]); seen.add(nxt[0])
    return chain


def _leaf_digests(rex, sect):
    """One digest per section of the base layer, from the STATE and nothing else.

    A leaf commits to the section's identity (its label), where it lives in the source
    (its span, when it has one) and the boundary columns of the cells it owns. The last
    is the structural content: the columns carry the signs, so a re orientation changes
    the leaf even when the support does not, which is the thing a digest over the support
    alone would miss.
    """
    rex._ensure_clean()
    B = rex._B1_dual
    out = []
    for i in range(sect.n_sections):
        cells = np.sort(np.asarray(sect.cells(i), dtype=np.int64))
        parts = [sect.labels[i].encode("utf-8")]
        if sect.spans is not None and i < len(sect.spans):
            parts.append(np.asarray(sect.spans[i], dtype=np.int64).tobytes())
        for c in cells.tolist():
            lo, hi = B.col_ptr[c], B.col_ptr[c + 1]
            parts.append(np.asarray(B.row_idx[lo:hi], dtype=np.int64).tobytes())
            parts.append(np.ascontiguousarray(B.vals_csc[lo:hi], dtype=np.float64).tobytes())
        out.append(_h(*parts))
    return out


class LayerMerkle:
    """The built tree: leaves, the interior node of every layer, and the root."""

    __slots__ = ("chain", "leaves", "levels", "groups", "roots", "parents", "root")

    def __init__(self, chain, leaves, levels, groups, roots, parents, root):
        self.chain = list(chain)          #: finest -> coarsest, then the implicit root
        self.leaves = list(leaves)
        self.levels = levels              #: {layer: [levels per sibling set]}
        self.groups = groups              #: {layer: [[child indices]]}
        self.roots = roots                #: {layer: [digest per section]}, free
        self.parents = parents            #: {layer: parent array over the finer layer}
        self.root = root

    def layer_digest(self, layer, i):
        """The digest of section `i` of `layer`, which is an interior node of this tree."""
        if layer == self.chain[0]:
            return self.leaves[i]
        return self.roots[layer][i]

    def proof(self, leaf):
        """Return (sibling, is_right, layer) entries from a leaf to the root.

        Each layer tag identifies the named sectioning level traversed by
        that part of the path. The sibling digests and orientation flags
        supply the inputs required to recompute the document root.
        """
        out, idx = [], int(leaf)
        for k, layer in enumerate(self.chain):
            coarser = self.chain[k + 1] if k + 1 < len(self.chain) else None
            par = self.parents.get(coarser) if coarser else None
            g = self.groups[layer]
            j = int(par[idx]) if par is not None else 0
            pos = g[j].index(idx)
            out += [(s, r, layer) for s, r in _path(self.levels[layer][j], pos)]
            idx = j
        return out


def verify_proof(leaf_digest, proof, root):
    """Recompute the root from a leaf and its path. No document, no complex."""
    h = leaf_digest
    for sib, is_right, _layer in proof:
        h = _h(h, sib) if is_right else _h(sib, h)
    return h == root


def build_merkle(rex, *, base=None, leaves=None):
    """Build the hybrid tree over `rex`'s sectioning hierarchy.

    `leaves` skips `_leaf_digests`, which is the only part that touches B1. Passing the
    STORED leaves rebuilds the interior alone, which is what verifying a loaded bundle
    needs: it checks that those leaves hash to that root without re deriving them from
    the complex, so leaf tampering and root tampering are both caught.
    """
    from rexgraph.sectioning import sectionings_of

    store = sectionings_of(rex)
    if not store:
        raise ValueError("no sectionings: there is no hierarchy to build a tree over")
    chain = layer_chain(store, base)
    sect = store[chain[0]]
    if not sect.is_partition():
        raise ValueError(
            f"{chain[0]!r} is a cover, not a partition: a leaf would have several "
            f"parents and the tree is not well defined. Store the partition.")

    leaves = list(leaves) if leaves is not None else _leaf_digests(rex, sect)
    if len(leaves) != sect.n_sections:
        raise ValueError(
            f"{len(leaves)} leaves for {sect.n_sections} sections of {chain[0]!r}")
    levels, groups, roots, parents = {}, {}, {}, {}
    below = leaves
    for k, layer in enumerate(chain):
        coarser = chain[k + 1] if k + 1 < len(chain) else None
        if coarser:
            par = np.asarray(store[coarser].parent, dtype=np.int64)
            n_up = store[coarser].n_sections
            parents[coarser] = par
        else:
            par, n_up = np.zeros(len(below), dtype=np.int64), 1
        g = [[] for _ in range(n_up)]
        for i, p in enumerate(par.tolist()):
            g[p].append(i)
        lv = [_binary_levels([below[i] for i in members]) for members in g]
        levels[layer], groups[layer] = lv, g
        below = [x[-1][0] for x in lv]
        if coarser:
            roots[coarser] = below
    return LayerMerkle(chain, leaves, levels, groups, roots, parents,
                       below[0] if below else EMPTY)


# serialisation

def pack_merkle(rex, t, h, *, base=None):
    """Return Merkle metadata derived from the complex's sectioning state.

    The metadata contains the layer chain, root digest and leaf count.
    Leaf and interior digests are recomputed from boundary columns, labels,
    spans and parent maps already carried by the complex. No digest arrays
    are added to t or h, and source text is not read.

    Returns None if sectionings are absent or a valid hierarchy cannot
    be built. unpack_merkle reconstructs the tree and compares its root
    with this metadata when verification is requested.
    """
    from rexgraph.sectioning import sectionings_of

    if not sectionings_of(rex):
        return None
    try:
        m = build_merkle(rex, base=base)
    except ValueError:
        return None                        # a cover or a broken chain is not an error here
    return {"chain": list(m.chain), "root": m.root.hex(), "n_leaves": len(m.leaves)}


def unpack_merkle(rex, t, h, *, verify=True):
    """Rebuild a Merkle tree from state and optionally verify its stored root.

    Derive leaves from the loaded boundary columns, labels and spans. Verification
    compares the resulting root with the stored root.
    """
    meta = h.get("merkle")
    if not meta:
        return None
    if not verify or not meta.get("root"):
        return None, meta
    try:
        rebuilt = build_merkle(rex, base=(meta.get("chain") or [None])[0])
    except ValueError as exc:
        raise ValueError(f"the stored layer hierarchy does not rebuild: {exc}") from exc
    if rebuilt.root.hex() != meta["root"]:
        raise ValueError(
            "this complex does not hash to the stored Merkle root: its layer tree is "
            "not what was written")
    rex._merkle_leaves = rebuilt.leaves
    return rebuilt.leaves, meta
