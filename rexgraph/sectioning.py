"""Sectionings: a partition of a complex's cells, carried as a cochain.

The document is the canonical field. Its chapters, paragraphs and sentences are not
smaller complexes hanging off it, they are PARTITIONS of the one field, and a traversal
from chapter to paragraph to sentence is a refinement of that partition rather than a
walk into a nested object. That is what this module stores.

A partition of the cells at grade `k` is an integer cochain at grade `k`: one value per
cell naming its owner. So a sectioning needs no new storage type, it is a first-class
object of the model already, and SEVERAL sectionings coexist over ONE field the same way
several cochains do. Encoding it as CSR (section -> its cells) covers both readings a
document actually needs:

    partition   every cell in exactly one section. Masses total rank(B_k) exactly, which
                is what makes the accounting close and why this is the canonical form.
    cover       a cell in several sections, which is what sentence spans really are: a
                pair of words recurring in two sentences belongs to both. Masses then
                exceed the rank, and the excess is not an error, it IS the sharing.

`spans` carries each section's byte range in the raw source. That is the pointer layer:
the text stays an addressable heap and a section names where it lives, so recovering a
section's prose is one seek and one read rather than a re-parse.

Each sectioning carries its OWN digest over its own tensors, so a layer can be checked
without loading the complex it sections, matching what nested states already do.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Sectioning", "add_sectioning", "add_coarsening", "sectionings_of",
           "drop_sectioning",
           "sectioning_summary", "pack_sectionings", "unpack_sectionings"]


def _fit(values, n):
    """The narrowest unsigned dtype that holds `n`, which is how the index layer packs."""
    a = np.asarray(values, dtype=np.int64)
    if n <= np.iinfo(np.uint16).max:
        return a.astype(np.uint16)
    if n <= np.iinfo(np.uint32).max:
        return a.astype(np.uint32)
    return a.astype(np.int64)


class Sectioning:
    """One sectioning of one grade: CSR from section to the cells it owns.

    `indptr`/`indices` are the CSR. `labels[i]` names section `i`. `spans[i]` is its
    `(offset, length)` in the raw source when there is one, and is absent otherwise
    because most sectionings are not derived from bytes.
    """

    __slots__ = ("name", "grade", "indptr", "indices", "labels", "spans", "n_cells",
                 "method", "refines", "parent")

    def __init__(self, name, grade, indptr, indices, labels, *, spans=None,
                 n_cells=0, method="", refines="", parent=None):
        self.name = str(name)
        self.grade = int(grade)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = [str(x) for x in labels]
        self.spans = None if spans is None else np.asarray(spans, dtype=np.int64)
        self.n_cells = int(n_cells)
        #: When this layer is a COARSENING of a finer one, `refines` names that layer and
        #: `parent[i]` is the section of THIS layer that finer section `i` belongs to.
        #: The CSR is then derived rather than stored, which is the whole saving: a
        #: paragraph layer over 2,662 sentences is 2,662 numbers, not the 121,877 cell
        #: memberships it would re-list. That is also the honest model, because a
        #: paragraph does not own cells directly, it owns sentences that own cells.
        self.refines = str(refines or "")
        self.parent = None if parent is None else np.asarray(parent, dtype=np.int64)
        #: how the sectioning was ARRIVED AT, not what it says. A reader has to be able
        #: to tell a chapter split that matched real headings from one that fell back to
        #: blank lines, because the two support different claims.
        self.method = str(method)

    def __len__(self):
        return len(self.labels) if self.refines else max(len(self.indptr) - 1, 0)

    @property
    def n_sections(self):
        return len(self)

    @property
    def is_derived(self):
        """True when the CSR is derived from a finer layer rather than stored."""
        return bool(self.refines) and self.indptr.size <= 1

    def resolved(self, store=None):
        """This layer with its own CSR, deriving it from the finer layer if need be.

        `store` is `{name: Sectioning}`, normally `sectionings_of(rex)`. A derived layer
        cannot answer `cells` on its own by design: it does not know the finer layer's
        memberships, and copying them in would undo the saving.
        """
        if not self.is_derived:
            return self
        if not store or self.refines not in store:
            raise ValueError(
                f"{self.name!r} refines {self.refines!r} and needs it to resolve. Pass "
                f"store=sectionings_of(rex).")
        finer = store[self.refines].resolved(store)
        order = np.argsort(self.parent, kind="stable")
        counts = np.bincount(self.parent, minlength=len(self))
        sizes = np.diff(finer.indptr)
        indptr = np.zeros(len(self) + 1, dtype=np.int64)
        chunks = []
        at = 0
        for j in range(len(self)):
            fin = order[at:at + counts[j]]
            at += counts[j]
            chunks.extend(finer.indices[finer.indptr[f]:finer.indptr[f] + sizes[f]]
                          for f in fin)
            indptr[j + 1] = indptr[j] + int(sizes[fin].sum()) if fin.size else indptr[j]
        idx = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        return Sectioning(self.name, self.grade, indptr, idx, self.labels,
                          spans=self.spans, n_cells=self.n_cells, method=self.method)

    def cells(self, i):
        """The cell ids owned by section `i`."""
        if self.is_derived:
            raise ValueError(f"{self.name!r} is derived; call resolved(store) first")
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def as_sections(self, store=None):
        """`{label: [cell ids]}`, the mapping `partition.section_readings` takes."""
        s = self.resolved(store) if self.is_derived else self
        return {s.labels[i]: s.cells(i).tolist() for i in range(len(s))}

    def owner_cochain(self, n_cells=None):
        """The partition as a grade-`k` cochain: `owner[c]` is c's section, or -1.

        This is the form the claim rests on. It only exists when the sectioning IS a
        partition; a cover has no single owner per cell and gets -1 collisions resolved
        by first writer, so `is_partition` decides whether it means anything.
        """
        n = int(n_cells if n_cells is not None else self.n_cells)
        owner = np.full(n, -1, dtype=np.int64)
        for i in range(len(self)):
            c = self.cells(i)
            fresh = c[owner[c] < 0]
            owner[fresh] = i
        return owner

    def is_partition(self, n_cells=None):
        """True when every cell is owned exactly once. Cheap: counts, not sorting."""
        n = int(n_cells if n_cells is not None else self.n_cells)
        if self.indices.size != n:
            return False
        seen = np.zeros(n, dtype=bool)
        seen[self.indices] = True
        return bool(seen.all())

    def __repr__(self):
        return (f"Sectioning({self.name!r}, grade={self.grade}, "
                f"{self.n_sections} sections over {self.indices.size} cells)")


def from_mapping(name, sections, *, grade=1, n_cells=0, spans=None, method=""):
    """Build a `Sectioning` from `{label: [cell ids]}`, preserving the given order."""
    labels, indptr, indices = [], [0], []
    for lab, cells in sections.items():
        labels.append(lab)
        c = np.asarray(list(cells), dtype=np.int64).ravel()
        indices.append(c)
        indptr.append(indptr[-1] + c.size)
    idx = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
    sp = None
    if spans is not None:
        sp = np.asarray([spans[lab] for lab in labels], dtype=np.int64).reshape(-1, 2)
    return Sectioning(name, grade, indptr, idx, labels, spans=sp,
                      n_cells=int(n_cells), method=method)


def add_coarsening(rex, name, refines, parent, labels, *, spans=None, method="",
                   grade=1):
    """Attach a layer that COARSENS an existing one, storing only the parent map.

    `parent[i]` is the index of the section of THIS layer that section `i` of `refines`
    belongs to. Nothing about the cells is stored, because a paragraph owns sentences and
    the sentences already say which cells they own; re-listing them cost 45x what the map
    costs and could disagree with the finer layer, which a derived one cannot.
    """
    store = getattr(rex, "_sectionings", None) or {}
    if refines not in store:
        raise ValueError(f"{name!r} refines {refines!r}, which is not attached")
    finer = store[refines]
    par = np.asarray(parent, dtype=np.int64).ravel()
    if par.size != finer.n_sections:
        raise ValueError(
            f"{name!r} gives {par.size} parents for {finer.n_sections} sections of "
            f"{refines!r}: a coarsening assigns every finer section exactly once")
    labs = [str(x) for x in labels]
    if par.size and (par.min() < 0 or par.max() >= len(labs)):
        raise ValueError(f"{name!r} names parent {int(par.max())} of {len(labs)} labels")
    sp = None if spans is None else np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    s = Sectioning(name, grade, np.zeros(1, dtype=np.int64),
                   np.zeros(0, dtype=np.int64), labs, spans=sp,
                   n_cells=finer.n_cells, method=method, refines=refines, parent=par)
    if getattr(rex, "_sectionings", None) is None:
        rex._sectionings = {}
    rex._sectionings[str(name)] = s
    return s


def add_sectioning(rex, name, sections, *, grade=1, spans=None, method="",
                   replace=True):
    """Attach a sectioning to `rex` so it serialises with the complex.

    `sections` is `{label: [cell ids]}` or a `Sectioning`. Cell ids are validated against
    the grade's own count, because a sectioning that names a cell the complex does not
    have is not a partition of anything.
    """
    n_cells = int(rex.nV) if int(grade) == 0 else int(rex.nE) if int(grade) == 1 else 0
    if int(grade) >= 2:
        n_cells = int(getattr(rex, "nF", 0) or 0)
    s = (sections if isinstance(sections, Sectioning)
         else from_mapping(name, sections, grade=grade, n_cells=n_cells, spans=spans,
                           method=method))
    s.name, s.n_cells = str(name), n_cells
    if s.indices.size and (s.indices.min() < 0 or s.indices.max() >= max(n_cells, 1)):
        raise ValueError(
            f"sectioning {name!r} names cell {int(s.indices.max())} at grade {grade}, "
            f"but the complex has {n_cells}. A sectioning partitions the cells that "
            f"exist; it does not introduce any.")
    store = getattr(rex, "_sectionings", None)
    if store is None:
        store = {}
        rex._sectionings = store
    if name in store and not replace:
        raise ValueError(f"{name!r} is already a sectioning of this complex")
    store[str(name)] = s
    return s


def sectionings_of(rex):
    """`{name: Sectioning}` attached to `rex`, empty when there are none."""
    return dict(getattr(rex, "_sectionings", None) or {})


def drop_sectioning(rex, name):
    store = getattr(rex, "_sectionings", None) or {}
    return store.pop(str(name), None)


def sectioning_summary(rex):
    """Per-sectioning summary small enough for a store index to carry and query.

    This is what makes the layers queryable without opening the complex: a caller asks
    which documents have a paragraph layer, or how many sections one has, and reads it
    off the index instead of deserialising 61,354 blobs.
    """
    out = []
    store = sectionings_of(rex)
    for s in store.values():
        r = s.resolved(store) if s.is_derived else s
        sizes = np.diff(r.indptr) if len(r) else np.zeros(0, dtype=np.int64)
        out.append({
            "name": s.name, "grade": int(s.grade),
            "n_sections": int(s.n_sections), "n_cells_covered": int(r.indices.size),
            "is_partition": bool(r.is_partition()), "refines": s.refines,
            "max_section": int(sizes.max()) if sizes.size else 0,
            "min_section": int(sizes.min()) if sizes.size else 0,
            "has_spans": s.spans is not None,
            "method": s.method,
        })
    return out


#### serialisation #############################################################

def pack_sectionings(rex, t, h):
    """Write every sectioning into the tensor dict under `sections/<name>/*`.

    Mirrors how nested states are packed, including the per-layer digest: each sectioning
    is digested over its OWN tensors so a caller can check one layer without the rest.
    """
    from rexgraph.io.rex_state import DIGEST_ALGO, _pack_strings, state_digest

    entries = []
    for s in sectionings_of(rex).values():
        pref = f"sections/{s.name}/"
        names = []
        # every index array is FITTED. These are all bounded counts (cells, sections,
        # byte offsets within one document), and int64 doubled or quadrupled each of
        # them for range nothing here uses.
        if s.is_derived:
            t[pref + "parent"] = np.ascontiguousarray(
                _fit(s.parent, max(len(s.labels), 1)))
            names.append(pref + "parent")
        else:
            t[pref + "indptr"] = np.ascontiguousarray(
                _fit(s.indptr, max(int(s.indices.size), 1)))
            t[pref + "indices"] = np.ascontiguousarray(
                _fit(s.indices, max(s.n_cells, 1)))
            names += [pref + "indptr", pref + "indices"]
        buf, offs = _pack_strings(s.labels)
        t[pref + "labels"] = np.frombuffer(buf, dtype=np.uint8).copy()
        t[pref + "label_offsets"] = np.ascontiguousarray(_fit(offs, max(len(buf), 1)))
        names += [pref + "labels", pref + "label_offsets"]
        if s.spans is not None:
            t[pref + "spans"] = np.ascontiguousarray(
                _fit(s.spans.ravel(), max(int(s.spans.max()) + 1, 1))
                .reshape(s.spans.shape))
            names.append(pref + "spans")
        entries.append({"name": s.name, "grade": int(s.grade),
                        "n_cells": int(s.n_cells), "method": s.method,
                        "refines": s.refines,
                        # the framing travels with the layer, for the same reason it
                        # travels with the container: a digest written under one rule and
                        # checked under another reports every stored object as corrupt.
                        # Two other writers had to learn this the hard way.
                        "digest_algo": int(DIGEST_ALGO),
                        "digest": state_digest(t, names)})
    return entries


def unpack_sectionings(rex, t, h):
    """Restore sectionings from the tensor dict, verifying each layer's own digest."""
    from rexgraph.io.rex_state import _unpack_strings, state_digest

    out = {}
    for e in h.get("sectionings", []) or []:
        pref = f"sections/{e['name']}/"
        derived = pref + "parent" in t
        if not derived and pref + "indptr" not in t:
            continue
        names = ([pref + "parent"] if derived else [pref + "indptr", pref + "indices"])
        names += [pref + "labels", pref + "label_offsets"]
        spans = None
        if pref + "spans" in t:
            spans = np.asarray(t[pref + "spans"])
            names.append(pref + "spans")
        algo = int(e.get("digest_algo", 1))
        if e.get("digest") and state_digest(t, names, algo=algo) != e["digest"]:
            raise ValueError(
                f"sectioning {e['name']!r} does not match its digest: the layer is not "
                f"what was written")
        labels = _unpack_strings(t[pref + "labels"], t[pref + "label_offsets"])
        out[e["name"]] = Sectioning(
            e["name"], int(e.get("grade", 1)),
            np.zeros(1, dtype=np.int64) if derived else np.asarray(t[pref + "indptr"]),
            np.zeros(0, dtype=np.int64) if derived else np.asarray(t[pref + "indices"]),
            labels, spans=spans, n_cells=int(e.get("n_cells", 0)),
            method=str(e.get("method", "")), refines=str(e.get("refines", "")),
            parent=np.asarray(t[pref + "parent"]) if derived else None)
    if out:
        rex._sectionings = out
    return out
