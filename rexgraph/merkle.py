"""A Merkle tree whose shape is the document's own layer hierarchy.

Two trees were available and each gave up something the other kept. A balanced binary
tree over every sentence proves inclusion in ceil(log2 n) hashes but its interior nodes
mean nothing: the path from a sentence to the root passes through nodes that are not
paragraphs, not chapters, not anything a reader could name. A tree shaped like the
document proves through nodes that ARE the paragraph and the chapter, but combining a
chapter's 31 paragraphs flat costs all 31 sibling hashes.

Neither trade is necessary. The layer hierarchy supplies the LEVELS and a binary tree
runs INSIDE each sibling set, so a proof carries log2 of each fanout instead of all of
it while the path still reads sentence -> paragraph -> chapter -> document. Measured on
a 314 KB book (2,802 sentences, 1,122 paragraphs, 36 chapters): 14 hashes at the median
against 70 for the flat-sibling form and 12 for the binary tree that names nothing. 448
bytes to keep the semantics, against 384 to throw them away.

Two consequences fall out rather than being built:

    layer digests are free    the paragraph and chapter digests ARE this tree's interior
                              nodes. Nothing hashes them separately, and nothing stores
                              a 2 kB homomorphic digest per section to make a coarsening
                              composable, because composition is what a tree already is.
    the partition is required a leaf must have exactly ONE parent, so this works because
                              the stored sectioning is a partition. Under a cover a
                              relation belongs to several sentences at once and the tree
                              is not well defined. The choice of the partition as the
                              canonical form and the availability of this tree are the
                              same decision, not two.

What it buys over the flat digest, which stays as the container seal: an inclusion proof
that travels without the document, and a single-sentence update that rehashes a path
instead of the book (10.5 us against 3.0 ms, 281x).
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["LayerMerkle", "build_merkle", "verify_proof", "layer_chain",
           "pack_merkle", "unpack_merkle"]

DIGEST_SIZE = 32
EMPTY = b"\x00" * DIGEST_SIZE


def _h(*parts):
    """One node hash. Parts are length-prefixed so concatenation is unambiguous."""
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
            sib = idx                       # the self-paired odd node
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
    is the structural content: the columns carry the signs, so a re-orientation changes
    the leaf even when the support does not, which is the thing a digest over the support
    alone would miss.
    """
    from rexgraph.core._sparse import to_scipy_csr
    B = to_scipy_csr(rex._B1_dual).tocsc()
    out = []
    for i in range(sect.n_sections):
        cells = np.sort(np.asarray(sect.cells(i), dtype=np.int64))
        parts = [sect.labels[i].encode("utf-8")]
        if sect.spans is not None and i < len(sect.spans):
            parts.append(np.asarray(sect.spans[i], dtype=np.int64).tobytes())
        for c in cells.tolist():
            lo, hi = B.indptr[c], B.indptr[c + 1]
            parts.append(B.indices[lo:hi].astype(np.int64).tobytes())
            parts.append(np.ascontiguousarray(B.data[lo:hi], dtype=np.float64).tobytes())
        out.append(_h(*parts))
    return out


class LayerMerkle:
    """The built tree: leaves, the interior node of every layer, and the root."""

    __slots__ = ("chain", "leaves", "levels", "groups", "roots", "parents", "root")

    def __init__(self, chain, leaves, levels, groups, roots, parents, root):
        self.chain = list(chain)          #: finest -> coarsest, then the implicit root
        self.leaves = list(leaves)
        self.levels = levels              #: {layer: [levels-per-sibling-set]}
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
        """`[(sibling, is_right, layer)]` from base section `leaf` to the document root.

        The `layer` tag is the point of the whole construction: a verifier can say which
        paragraph and which chapter it climbed through, not merely that some path exists.
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
    needs: it checks that those leaves hash to that root without re-deriving them from
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


#### serialisation #############################################################

def pack_merkle(rex, t, h, *, base=None):
    """Store the LEAVES and the root; every interior node is recomputable from them.

    Nothing is stored: the tree is a pure function of state the file already carries.
    The interior is the leaves and the parent maps; the leaves are what `_leaf_digests`
    builds "from the STATE and nothing else": the section labels, the spans, and the
    boundary columns of the cells it owns, every one of them already a tensor here. A
    stored digest is a cached hash of data sitting beside it.

    It used to store both. The leaves were kept on the grounds that a proof must be
    checkable "without the source text the leaves were built from", which is not what
    they are built from: `_leaf_digests` never touches the text. They were also 35% of
    a document blob and, being digests, at full entropy: the one part of the file no
    compressor can touch, so they dominated the COMPRESSED size far more than the raw.

    Deriving them makes verification stronger rather than weaker. Checking stored leaves
    against a stored root asks "do these leaves make this root". Recomputing from state
    asks "does this complex make this root", which is the question that catches a
    tampered boundary column, and the old form explicitly could not ask it.

    That is what this says and it used to also write `layer_roots`, every interior node
    of every coarser layer, as hex strings in the JSON header: the exact thing the
    paragraph above says it does not do. Nothing read it: `unpack_merkle` takes `chain`
    and `root`, and `layer_digest` reads the `roots` that `build_merkle` recomputes. On
    Webster's Unabridged (nV 223,609, nE 1,991,070) it was 72.8 MiB of header, and the
    header is written twice, so 145.6 MiB against safetensors' 100 MB limit: the
    document could not be stored at all. Digests are bytes and belong in a tensor, which
    is where the leaves already are.
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
    """Rebuild the tree from state and, when asked, CHECK it makes the stored root.

    This asks "does THIS COMPLEX make this root", which is the whole tree derived from
    the boundary columns, the labels and the spans as loaded. A rewritten column, a moved
    span or a rewritten root all fail it. The previous form compared stored leaves to a
    stored root and so could only catch the last two.

    Cost is one `_leaf_digests` pass, measured at 2.3 us a section.
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
