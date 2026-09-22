"""Explicit finite Q-complexes and exact sparse chain map certificates.

No target boundary is inferred, and no map is assumed to be a partition,
projection, injection or metric isometry. A certificate checks both complexes
and all commuting squares. Composition requires the identical middle complex,
not just matching dimensions. All supplied coefficients must be exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256

from rexgraph.graded_boundary import (
    _exact_compose_columns,
    _exact_composition_residual,
    _integer_columns,
)
from rexgraph.native_sparse import raw_boundary_carriers, sparse_columns, sparse_arrays
from rexgraph.type_accession import CoordinateSpace, TypeAccession, _digest, _entries

__all__ = ["CoordinateComplex", "GradedMap", "ChainMap", "ChainHomotopy",
           "SymmetryGroup", "symmetry_generators"]


def _columns(entries, n):
    result = [{} for _ in range(n)]
    for i, j, value in entries:
        result[j][i] = value
    return result


def _triples(columns):
    return tuple((i, j, value) for j, column in enumerate(columns)
                 for i, value in sorted(column.items()))


def _exact_entries(entries, shape):
    entries, exact = _entries(entries, shape)
    if not exact:
        raise TypeError("chain-map declarations require integers or Fractions, not floats")
    return entries


def _chain_residual(value):
    maps = [_columns(entries, value.sizes[k])
            for k, entries in enumerate(value.boundaries, 1)]
    return max((_exact_composition_residual(a, b) for a, b in zip(maps[:-1], maps[1:], strict=True)),
               default=Fraction(0))


@dataclass(frozen=True, eq=False)
class CoordinateComplex:
    """Declared ordered spaces C0..Cg and boundaries B1..Bg, over Q.

    Construction validates coefficients and axes, not the chain law. Explicit
    zero maps retain empty grades. ``from_rex`` captures the *full* present tower
    in canonical cell order, retaining primary C1 shares and integral higher
    boundaries. A hand built target is a coordinate complex, not a new RexGraph.
    """

    spaces: tuple[CoordinateSpace, ...]
    boundaries: tuple
    source: object = field(init=False, default=None, repr=False)
    _primary: tuple = field(init=False, default=(), repr=False)
    _relation_ids: tuple | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        spaces, boundaries = tuple(self.spaces), tuple(self.boundaries)
        if not spaces or any(not isinstance(s, CoordinateSpace) for s in spaces):
            raise TypeError("complex requires ordered CoordinateSpaces starting at grade zero")
        if len(boundaries) != len(spaces) - 1:
            raise ValueError("complex requires exactly one boundary per adjacent pair of grades")
        boundaries = tuple(_exact_entries(b, (len(spaces[k-1].keys), len(spaces[k].keys)))
                           for k, b in enumerate(boundaries, 1))
        object.__setattr__(self, "spaces", spaces)
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def sizes(self):
        return tuple(len(s.keys) for s in self.spaces)

    @property
    def coefficient_digest(self):
        # Names/order and primary supports are part of the boundary state contract.
        header = repr((tuple((s.name, s.keys) for s in self.spaces), self._primary, self._relation_ids))
        coefficients = "|".join(_digest(b, f"B{k}|") for k, b in enumerate(self.boundaries, 1))
        return sha256(("coordinate-complex-v1|" + header + coefficients).encode()).hexdigest()

    @classmethod
    def from_rex(cls, source):
        source._ensure_clean()
        maps = raw_boundary_carriers(source)
        shapes = tuple(sparse_arrays(b)[3] for b in maps)
        sizes = (shapes[0][0], *(shape[1] for shape in shapes))
        if sizes[:2] != (int(source.nV), int(source.nE)):
            raise ValueError("source boundary axes do not match the canonical cell populations")
        primary = tuple(tuple(int(v) for v in support) for support in source.relation_supports())
        from rexgraph.native_rank import primary_columns as _primary_columns
        columns = _primary_columns(source)
        if len(columns) != sizes[1]:
            raise ValueError("primary relation count does not match the boundary")
        # Primary incidence is authoritative. Check its sparse display agrees;
        # never infer exact branching shares by rationalizing floating entries. The
        # columns come from the stored CSR, with any declared head or share, so the
        # support reading is checked against that same CSR rather than assumed equal
        # to it: two readings of the primary incidence that disagree certify nothing.
        ptr, idx = source._boundary_ptr, source._boundary_idx
        for j, support in enumerate(primary):
            if support != tuple(int(v) for v in idx[int(ptr[j]):int(ptr[j + 1])]):
                raise ValueError("primary incidence and stored B1 disagree")
        for column, stored in zip(columns, sparse_columns(maps[0]), strict=True):
            if stored != {i: float(v) for i, v in column.items()}:
                raise ValueError("primary incidence and stored B1 disagree")
        boundaries = [_triples(columns)]
        for k, b in enumerate(maps[1:], 2):
            if shapes[k-1] != (sizes[k-1], sizes[k]):
                raise ValueError("source boundary tower has incompatible adjacent shapes")
            columns = _integer_columns(b)
            if columns is None:
                raise ValueError("exact chain maps require integral stored higher boundaries")
            boundaries.append(_triples(columns))
        result = cls(tuple(CoordinateSpace(f"C{k}", tuple(str(i) for i in range(n)))
                           for k, n in enumerate(sizes)), tuple(boundaries))
        object.__setattr__(result, "source", source)
        object.__setattr__(result, "_primary", primary)
        ids = source.relation_ids
        if ids is not None:
            object.__setattr__(result, "_relation_ids", tuple(int(i) for i in ids))
        return result

    def check_state(self):
        if self.source is not None:
            current = type(self).from_rex(self.source)
            if current.coefficient_digest != self.coefficient_digest:
                raise ValueError("chain-map boundary state changed; bind a fresh complex and map")


@dataclass(frozen=True, eq=False)
class GradedMap:
    """An explicit map P_k at every grade; chain preservation is not assumed."""

    domain: CoordinateComplex
    codomain: CoordinateComplex
    components: tuple

    def __post_init__(self):
        if not isinstance(self.domain, CoordinateComplex) or not isinstance(self.codomain, CoordinateComplex):
            raise TypeError("graded map requires explicit domain and codomain complexes")
        if len(self.domain.spaces) != len(self.codomain.spaces):
            raise ValueError("graded map requires equal grade ranges; declare absent grades as empty spaces")
        components = tuple(self.components)
        if len(components) != len(self.domain.spaces):
            raise ValueError("graded map requires one component at every grade, including zero")
        object.__setattr__(self, "components", tuple(_exact_entries(p, shape)
                           for p, shape in zip(components, self.shapes, strict=True)))

    @property
    def shapes(self):
        return tuple(zip(self.codomain.sizes, self.domain.sizes, strict=True))

    @property
    def coefficient_digest(self):
        return sha256(("graded-map-v1|" + self.domain.coefficient_digest + self.codomain.coefficient_digest
                       + "|".join(_digest(p, f"P{k}|") for k, p in enumerate(self.components))).encode()).hexdigest()

    @classmethod
    def from_accessions(cls, accessions, codomain):
        """Assemble one named type across *all* source grades in canonical order.

        The target boundaries and coordinates must be supplied, not induced by a
        pseudoinverse. Coordinate free ambient accessions use canonical Ck spaces.
        This does not change the independent accessions' unproved status or
        assert covariant cochain transport: the checked square is for chains.
        """
        accessions = tuple(accessions)
        if not accessions or any(not isinstance(a, TypeAccession) for a in accessions):
            raise TypeError("graded accession map requires TypeAccessions")
        first = accessions[0]
        domain = CoordinateComplex.from_rex(first.source)
        if not isinstance(codomain, CoordinateComplex) or len(accessions) != len(domain.spaces):
            raise ValueError("supply the full ordered accession tower and an explicit target complex")
        if len(codomain.spaces) != len(domain.spaces):
            raise ValueError("target and accession towers must have the same grade range")
        for k, a in enumerate(accessions):
            if a.source is not first.source or a.grade != k or a.name != first.name or a.cell_keys is not None:
                raise ValueError("accessions require one source, type name and canonical basis, ordered by grade")
            if not a.exact:
                raise TypeError("chain-map accessions must be exact, including zero declarations")
            coordinates = domain.spaces[k] if a.coordinates is None else a.coordinates
            if a.shape != (codomain.sizes[k], domain.sizes[k]) or coordinates != codomain.spaces[k]:
                raise ValueError("accession axes and ordered coordinates must match the declared complexes")
        return cls(domain, codomain, tuple(a.entries for a in accessions))

    def check_state(self):
        self.domain.check_state()
        if self.codomain is not self.domain:
            self.codomain.check_state()

    def then(self, following):
        """Return Q o P, still unverified, using exact sparse column products."""
        if not isinstance(following, GradedMap) or self.codomain is not following.domain:
            raise ValueError("composition requires the identical declared middle complex")
        self.check_state()
        following.check_state()
        components = tuple(_triples(_exact_compose_columns(_columns(q, nq), _columns(p, np)))
                           for p, q, np, nq in zip(self.components, following.components,
                                                  self.domain.sizes, following.domain.sizes, strict=True))
        return GradedMap(self.domain, following.codomain, components)

    def verify(self):
        return ChainMap(self)


@dataclass(frozen=True, eq=False)
class ChainMap:
    """Verified finite chain map. Construction always checks, never trusts a flag.

    Residuals are exact max entry norms, not tolerances. The proof is about the
    captured boundary state; call ``check_state`` before reuse. RCQL and ``then``
    do so automatically. No induced homology matrix or homotopy is constructed.
    """

    declaration: GradedMap
    source_residual: Fraction = field(init=False)
    target_residual: Fraction = field(init=False)
    commutation_residuals: tuple = field(init=False)

    def __post_init__(self):
        p = self.declaration
        if not isinstance(p, GradedMap):
            raise TypeError("ChainMap requires an explicit GradedMap")
        p.check_state()
        source, target = _chain_residual(p.domain), _chain_residual(p.codomain)
        if source or target:
            raise ValueError(f"chain law failed: source residual {source}, target residual {target}")
        columns = [_columns(e, n) for e, n in zip(p.components, p.domain.sizes, strict=True)]
        residuals = []
        for k, (b, d) in enumerate(zip(p.domain.boundaries, p.codomain.boundaries, strict=True), 1):
            left = _exact_compose_columns(columns[k-1], _columns(b, p.domain.sizes[k]))
            right = _exact_compose_columns(_columns(d, p.codomain.sizes[k]), columns[k])
            residual = max((abs(a.get(i, 0) - b.get(i, 0)) for a, b in zip(left, right, strict=True)
                            for i in a.keys() | b.keys()), default=Fraction(0))
            if residual:
                raise ValueError(f"chain-map square failed at grade {k}: exact residual {residual}")
            residuals.append(residual)
        object.__setattr__(self, "source_residual", source)
        object.__setattr__(self, "target_residual", target)
        object.__setattr__(self, "commutation_residuals", tuple(residuals))

    def check_state(self):
        self.declaration.check_state()

    def then(self, following):
        if not isinstance(following, ChainMap):
            raise TypeError("verified composition requires another ChainMap")
        return self.declaration.then(following.declaration).verify()


def symmetry_generators(generators):
    """Certify exact Euclidean chain automorphisms on one declared complex.

    A transpose is an inverse only after every square component satisfies
    U transpose U = I over Q. This does not assert weighted metric preservation,
    preservation of primary support slots or preservation of the G channel.
    """
    if not isinstance(generators, (list, tuple)) or not generators:
        raise ValueError("symmetry requires a nonempty sequence of graded maps")
    certificates = []
    domain = None
    for value in generators:
        p = value.declaration if isinstance(value, ChainMap) else value
        if not isinstance(p, GradedMap):
            raise TypeError("symmetry generators must be GradedMap or ChainMap declarations")
        if p.domain is not p.codomain or domain is not None and p.domain is not domain:
            raise ValueError("symmetry requires one identical declared domain and codomain complex")
        domain = p.domain
        certificate = p.verify()
        for k, (entries, n) in enumerate(zip(p.components, domain.sizes, strict=True)):
            transpose = tuple((j, i, v) for i, j, v in entries)
            gram = _exact_compose_columns(_columns(transpose, n), _columns(entries, n))
            if any(column != {j: Fraction(1)} for j, column in enumerate(gram)):
                raise ValueError(f"symmetry component at grade {k} is not exactly Euclidean orthogonal")
        certificates.append(certificate)
    return tuple(certificates)


@dataclass(frozen=True, eq=False)
class SymmetryGroup:
    """A generated subgroup of exact Euclidean chain automorphisms.

    Signed letters name generators or their transposes. Products act from right
    to left; the empty word is identity. The selected map is materialized lazily
    by existing sparse composition. Products can gain entries. No enumeration,
    finiteness, minimal generator set or completeness claim is made.
    """

    generators: tuple
    word: tuple = ()

    def __post_init__(self):
        from rexgraph.rational_operator import generator_word
        certificates = symmetry_generators(self.generators)
        object.__setattr__(self, "generators", certificates)
        object.__setattr__(self, "word", generator_word(len(certificates), self.word))

    @property
    def domain(self):
        return self.generators[0].declaration.domain

    @property
    def sizes(self):
        self.check_state()
        return self.domain.sizes

    @property
    def generator_count(self):
        return len(self.generators)

    def check_state(self):
        self.domain.check_state()

    def element(self, word):
        from rexgraph.rational_operator import generator_word
        self.check_state()
        result = object.__new__(type(self))
        object.__setattr__(result, "generators", self.generators)
        object.__setattr__(result, "word", generator_word(len(self.generators), word))
        return result

    @property
    def inverse(self):
        return self.element(tuple(-i for i in reversed(self.word)))

    @property
    def identity(self):
        return self.element(())

    @property
    def map(self):
        self.check_state()
        domain = self.domain
        result = GradedMap(domain, domain,
            tuple(tuple((i, i, 1) for i in range(n)) for n in domain.sizes))
        for letter in reversed(self.word):
            p = self.generators[abs(letter)-1].declaration
            if letter < 0:
                p = GradedMap(domain, domain,
                    tuple(tuple((j, i, v) for i, j, v in entries) for entries in p.components))
            result = result.then(p)
        return result.verify()


@dataclass(frozen=True, eq=False)
class ChainHomotopy:
    """Verified G-F = B_D H + H B_C at every grade, with an explicit Q witness.

    H_k maps C_k to D_(k+1). Its top component has zero rows and must be
    supplied as an empty sparse declaration. Endpoints share the identical
    declared complexes, as for composition. No witness is solved or inferred.
    """

    left: ChainMap | GradedMap
    right: ChainMap | GradedMap
    witness: tuple
    residuals: tuple = field(init=False)

    def __post_init__(self):
        def checked(value):
            value = value.declaration if isinstance(value, ChainMap) else value
            if not isinstance(value, GradedMap):
                raise TypeError("homotopy endpoints must be explicit graded maps")
            return value.verify()

        left, right = checked(self.left), checked(self.right)
        f, g = left.declaration, right.declaration
        if f.domain is not g.domain or f.codomain is not g.codomain:
            raise ValueError("homotopy endpoints require the identical declared domain and codomain complexes")
        sizes, target = f.domain.sizes, f.codomain.sizes
        if len(self.witness) != len(sizes):
            raise ValueError("homotopy requires one witness component at every grade, including the empty top")
        shapes = tuple((target[k+1] if k+1 < len(target) else 0, n) for k, n in enumerate(sizes))
        witness = tuple(_exact_entries(h, shape) for h, shape in zip(self.witness, shapes, strict=True))
        hcols = [_columns(h, n) for h, n in zip(witness, sizes, strict=True)]
        residuals = []
        for k, n in enumerate(sizes):
            below = (_exact_compose_columns(_columns(f.codomain.boundaries[k], target[k+1]), hcols[k])
                     if k+1 < len(target) else [{} for _ in range(n)])
            above = (_exact_compose_columns(hcols[k-1], _columns(f.domain.boundaries[k-1], n))
                     if k else [{} for _ in range(n)])
            a, b = _columns(f.components[k], n), _columns(g.components[k], n)
            residual = max((abs(v.get(i, 0) - u.get(i, 0) - d.get(i, 0) - e.get(i, 0))
                            for u, v, d, e in zip(a, b, below, above, strict=True)
                            for i in u.keys() | v.keys() | d.keys() | e.keys()), default=Fraction(0))
            if residual:
                raise ValueError(f"homotopy equation failed at grade {k}: exact residual {residual}")
            residuals.append(residual)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "witness", witness)
        object.__setattr__(self, "residuals", tuple(residuals))

    @property
    def shapes(self):
        f = self.left.declaration
        target = f.codomain.sizes
        return tuple((target[k+1] if k+1 < len(target) else 0, n) for k, n in enumerate(f.domain.sizes))

    @property
    def coefficient_digest(self):
        return sha256(("chain-homotopy-v1|" + self.left.declaration.coefficient_digest
                       + self.right.declaration.coefficient_digest
                       + "|".join(_digest(h, f"H{k}|") for k, h in enumerate(self.witness))).encode()).hexdigest()

    def check_state(self):
        self.left.check_state()
        self.right.check_state()
