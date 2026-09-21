"""Native molecular observations and exact geometry on supplied coordinates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
import json


from rexgraph.chain_map import CoordinateComplex
from rexgraph.coordinate_map import CoordinateMap, _identity
from rexgraph.tensor_field import FieldSource, TensorField, TensorChannels
from rexgraph.type_accession import CoordinateSpace

__all__ = ["MolecularView", "molecular_changes", "conformation_field", "conformation_direction"]


def _key(values):
    return json.dumps(tuple(values), ensure_ascii=False, separators=(",", ":"))


def _refs(*fields):
    refs = {s.coefficient_digest: s for f in fields for s in
            (*f.dependencies, *((f.source,) if f.source else ()))}
    return tuple(refs.values())


@dataclass(frozen=True, eq=False)
class MolecularView:
    """A selected molecular observation of one complete native source."""
    source: FieldSource
    name: str
    _manifest: dict = field(repr=False)
    _record: dict = field(repr=False)

    @classmethod
    def from_source(cls, source, name, *, record_id=None, version=None):
        reference = source if isinstance(source, FieldSource) else FieldSource(source, record_id, version)
        reference.check()
        manifest = deepcopy(getattr(reference.source, "_agent_meta", {}).get("source_manifest", {}))
        if manifest.get("schema") != "molecular_bundle/1":
            raise ValueError("selected source is not a molecular bundle")
        from hashlib import sha256
        payload = {k: v for k, v in manifest.items() if k != "bundle_digest"}
        digest = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if digest != manifest.get("bundle_digest"):
            raise ValueError("molecular declaration digest differs")
        selected = [r for r in manifest["molecules"] if r["name"] == name]
        if len(selected) != 1:
            raise ValueError("select one named molecular record")
        return cls(reference, name, manifest, selected[0])

    def _check(self):
        self.source.check()
        current = getattr(self.source.source, "_agent_meta", {}).get("source_manifest", {})
        if current != self._manifest:
            raise ValueError("molecular view declaration changed")
        if self._record != next((r for r in current.get("molecules", ()) if r["name"] == self.name), None):
            raise ValueError("molecular view selection changed")

    @property
    def manifest(self):
        self._check()
        return deepcopy(self._manifest)

    @property
    def record(self):
        self._check()
        return deepcopy(self._record)

    @property
    def coefficient_digest(self):
        return _identity(("molecular_view_v1", self.source.coefficient_digest, self.name, self._manifest["bundle_digest"]))

    @property
    def atom_space(self):
        self._check()
        return CoordinateSpace("molecular_atoms/"+self._record["coordinate_scope"], tuple(a["key"] for a in self._record["atoms"]))

    @property
    def bond_space(self):
        keys = self.atom_space.keys
        return CoordinateSpace("molecular_bonds/"+self._record["coordinate_scope"],
                               tuple(_key(sorted((keys[a], keys[b]))) for a, b in
                                     (bond["atoms"] for bond in self._record["bonds"])))

    def bond_complex(self):
        """Observe the bond boundary without adding aromatic groups or atom witnesses."""
        self._check()
        keys = self.atom_space.keys
        entries = []
        for j, bond in enumerate(self._record["bonds"]):
            a, b = bond["atoms"]
            if keys[a] > keys[b]:
                a, b = b, a
            entries.extend(((a, j, -1), (b, j, 1)))
        return CoordinateComplex((self.atom_space, self.bond_space), (tuple(entries),))

    def atom_lift(self):
        """Retain atom occurrences in the canonical boundary object coordinates."""
        native = CoordinateComplex.from_rex(self.source.source)
        return CoordinateMap(self.atom_space, native.spaces[0],
                             tuple((a["cell"], i, 1) for i, a in enumerate(self._record["atoms"])))

    def bond_lift(self):
        """Lift the selected oriented bond coordinates into their primary relations."""
        native = CoordinateComplex.from_rex(self.source.source)
        keys = self.atom_space.keys
        entries = []
        for j, bond in enumerate(self._record["bonds"]):
            a, b = bond["atoms"]
            entries.append((bond["cell"], j, 1 if keys[a] < keys[b] else -1))
        return CoordinateMap(self.bond_space, native.spaces[1], tuple(entries))

    def field(self, reading="bond_order", *, native=False):
        self._check()
        if not isinstance(native, bool):
            raise TypeError("native field flag must be boolean")
        atoms, bonds = self._record["atoms"], self._record["bonds"]
        if reading == "bond_order":
            values, space, grade = [Fraction(b["order"]) for b in bonds], self.bond_space, 1
        elif reading == "bond_presence":
            values, space, grade = [1]*len(bonds), self.bond_space, 1
        elif reading == "aromatic_bond":
            values, space, grade = [int(b["aromatic"]) for b in bonds], self.bond_space, 1
        elif reading in {"charge", "hydrogens", "atomic_number", "isotope", "radicals"}:
            values, space, grade = [a[reading] for a in atoms], self.atom_space, 0
        elif reading == "atom_presence":
            values, space, grade = [1]*len(atoms), self.atom_space, 0
        else:
            raise ValueError("unknown molecular reading")
        result = TensorField(space, values, source=self.source, grade=grade, variance="chain",
                             provenance=(self.coefficient_digest, _identity(("molecular_reading", reading))),)
        if native:
            from rexgraph.tensor_field import apply_tensor
            if grade != 1:
                raise ValueError("native injection currently requires a selected bond field")
            result = replace(apply_tensor(self.bond_lift(), result), grade=1)
        return result

    def conformation(self, name):
        self._check()
        profile = self._record["conformers"].get(name)
        if profile is None:
            raise ValueError("selected molecule has no supplied conformer with this name")
        tensor = self.source.source._cell_metadata[0][profile["cell"]][profile["attribute"]]
        if (not isinstance(tensor, TensorField) or tensor.coefficient_digest != profile["field_digest"]
                or tensor.space != self.atom_space):
            raise ValueError("stored conformation differs from its declaration")
        return replace(tensor, source=self.source, provenance=(*tensor.provenance, self.coefficient_digest))

    def balance(self):
        counts = {}
        for atom in self._record["atoms"]:
            key = (atom["atomic_number"], atom["isotope"])
            counts[key] = counts.get(key, 0) + 1
            hydrogens = atom["hydrogens"]
            if hydrogens:
                counts[(1, 0)] = counts.get((1, 0), 0) + hydrogens
        return tuple((number, isotope, count) for (number, isotope), count in sorted(counts.items()))

    def info(self):
        record = self.record
        return {**record, "element_counts": self.balance(), "formal_charge": sum(a["charge"] for a in record["atoms"]),
                "source": self.source.as_record(), "view_digest": self.coefficient_digest,
                "parser_profile": self._manifest["profile"]}


def molecular_changes(old, new, alignment=None):
    """Compare declared atom identities without inferring an atom mapping."""
    if not isinstance(old, MolecularView) or not isinstance(new, MolecularView):
        raise TypeError("molecular comparison requires two native views")
    a, b = old.record, new.record
    old_keys, new_keys = old.atom_space.keys, new.atom_space.keys
    if alignment is None:
        if not a["complete_mapping"] or not b["complete_mapping"] or old.atom_space.name != new.atom_space.name:
            raise ValueError("comparison needs a complete shared atom map or explicit alignment")
        union = sorted(set(old_keys) | set(new_keys))
        alignment = tuple((k if k in old_keys else None, k if k in new_keys else None, k) for k in union)
    else:
        alignment = tuple(tuple(v) for v in alignment)
    if (any(len(v) != 3 or not isinstance(v[2], str) or not v[2] or (v[0] is None and v[1] is None) for v in alignment)
            or len({v[2] for v in alignment}) != len(alignment)):
        raise ValueError("atom alignment requires unique named comparison coordinates")
    for side, keys in ((0, old_keys), (1, new_keys)):
        declared = [v[side] for v in alignment if v[side] is not None]
        if len(declared) != len(set(declared)) or set(declared) != set(keys):
            raise ValueError("alignment must account for every source atom exactly once")
    maps = [{x[side]: x[2] for x in alignment if x[side] is not None} for side in (0, 1)]
    scope = _identity(("atom_alignment_v1", old.atom_space.name, new.atom_space.name, alignment))
    atom_space = CoordinateSpace("atom_comparison/"+scope, tuple(v[2] for v in alignment))
    atom_tables = [{maps[side][atom["key"]]: atom for atom in record["atoms"]} for side, record in enumerate((a, b))]
    bond_tables = []
    for side, record in enumerate((a, b)):
        keys = [maps[side][v["key"]] for v in record["atoms"]]
        bond_tables.append({_key(sorted((keys[x], keys[y]))): bond for bond in record["bonds"] for x, y in [bond["atoms"]]})
    bond_keys = tuple(sorted(set(bond_tables[0]) | set(bond_tables[1])))
    bond_space = CoordinateSpace("bond_comparison/"+scope, bond_keys)
    dependencies = (old.source, new.source)
    names, fields = [], []
    def emit(label, space, left, right):
        for suffix, values in (("old", left), ("new", right), ("delta", [r-l for l, r in zip(left, right, strict=True)])):
            names.append(label+"/"+suffix)
            fields.append(TensorField(space, values, dependencies=dependencies,
                                     provenance=(old.coefficient_digest, new.coefficient_digest, scope, label)))
    emit("bond_presence", bond_space, *[[int(k in table) for k in bond_keys] for table in bond_tables])
    emit("bond_order", bond_space, *[[Fraction(table[k]["order"]) if k in table else Fraction(0) for k in bond_keys] for table in bond_tables])
    emit("aromatic_bond", bond_space, *[[int(table[k]["aromatic"]) if k in table else 0 for k in bond_keys] for table in bond_tables])
    stereo_pairs = tuple(sorted({(key, bond["stereo"]) for table in bond_tables for key, bond in table.items()}))
    stereo_space = CoordinateSpace("bond_stereo_comparison/"+scope, tuple(_key(pair) for pair in stereo_pairs))
    emit("bond_stereo", stereo_space, *[[int(key in table and table[key]["stereo"] == label) for key, label in stereo_pairs] for table in bond_tables])
    emit("atom_presence", atom_space, *[[int(k in table) for k in atom_space.keys] for table in atom_tables])
    for attribute in ("charge", "hydrogens", "isotope", "radicals"):
        emit(attribute, atom_space, *[[table[k][attribute] if k in table else 0 for k in atom_space.keys] for table in atom_tables])
    element_pairs = tuple(sorted({(key, atom["atomic_number"]) for table in atom_tables for key, atom in table.items()}))
    element_space = CoordinateSpace("atom_element_comparison/"+scope, tuple(_key(pair) for pair in element_pairs))
    emit("element", element_space, *[[int(key in table and table[key]["atomic_number"] == number) for key, number in element_pairs] for table in atom_tables])
    stereo_keys = tuple(sorted({(key, atom["cip"] or "unassigned") for table in atom_tables for key, atom in table.items()}))
    stereo_space = CoordinateSpace("atom_cip_comparison/"+scope, tuple(_key(pair) for pair in stereo_keys))
    emit("cip", stereo_space, *[[int(key in table and (table[key]["cip"] or "unassigned") == label) for key, label in stereo_keys] for table in atom_tables])
    balance_keys = tuple(sorted({(z, iso) for view in (old, new) for z, iso, _ in view.balance()}))
    balance_space = CoordinateSpace("element_inventory", tuple(_key(pair) for pair in balance_keys))
    balances = [{(z, iso): n for z, iso, n in view.balance()} for view in (old, new)]
    emit("inventory", balance_space, *[[table.get(k, 0) for k in balance_keys] for table in balances])
    return TensorChannels(tuple(names), tuple(fields), _identity(("molecular_delta_v1", scope, old.coefficient_digest, new.coefficient_digest)), dependencies)


@dataclass(frozen=True)
class _Dual:
    value: Fraction
    direction: Fraction = Fraction(0)

    @staticmethod
    def of(value):
        return value if isinstance(value, _Dual) else _Dual(Fraction(value))

    def __add__(self, other):
        other = self.of(other)
        return _Dual(self.value+other.value, self.direction+other.direction)

    __radd__ = __add__

    def __neg__(self):
        return _Dual(-self.value, -self.direction)

    def __sub__(self, other):
        return self + (-self.of(other))

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        other = self.of(other)
        return _Dual(self.value*other.value, self.direction*other.value+self.value*other.direction)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        if not other.value:
            raise ValueError("geometric spread is undefined on a zero direction")
        return _Dual(self.value/other.value, (self.direction*other.value-self.value*other.direction)/(other.value*other.value))

    def __rtruediv__(self, other):
        return self.of(other)/self


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b, strict=True)), Fraction(0))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _spread(a, b):
    denominator = _dot(a, a)*_dot(b, b)
    if (denominator.value if isinstance(denominator, _Dual) else denominator) == 0:
        raise ValueError("geometric spread is undefined on a zero direction")
    dot = _dot(a, b)
    return 1-dot*dot/denominator


_READINGS = {"pair_quadrance": (2, 2), "angle_dot": (3, 2), "angle_spread": (3, 0),
             "torsion_dot": (4, 4), "torsion_spread": (4, 0), "oriented_volume": (4, 3)}


def _geometry(values, indices, reading):
    points = [tuple(values[i]) for i in indices]
    if reading == "pair_quadrance":
        vector = _sub(points[1], points[0])
        return _dot(vector, vector)
    if reading.startswith("angle"):
        a, b = _sub(points[0], points[1]), _sub(points[2], points[1])
        return _dot(a, b) if reading == "angle_dot" else _spread(a, b)
    a, b, c = (_sub(points[i+1], points[i]) for i in range(3))
    n1, n2 = _cross(a, b), _cross(b, c)
    if reading == "oriented_volume":
        return _dot(n1, c)
    return _dot(n1, n2) if reading == "torsion_dot" else _spread(n1, n2)


def _geometry_contract(field, reading, selections):
    if not isinstance(field, TensorField) or len(field.axes) != 1 or field.axes[0].keys != ("x", "y", "z"):
        raise ValueError("geometry requires a field with one explicit Cartesian axis")
    field.check_state()
    if not field.axes[0].name.startswith("cartesian/"):
        raise ValueError("Cartesian coordinates must declare unit and reference frame")
    if reading not in _READINGS:
        raise ValueError("unknown conformation reading")
    selections = tuple(tuple(s) for s in selections)
    arity, power = _READINGS[reading]
    if any(len(s) != arity or len(set(s)) != arity for s in selections):
        raise ValueError("geometric selections require distinct atoms of the declared arity")
    if len(set(selections)) != len(selections):
        raise ValueError("geometric selections must be distinct")
    index = {k: i for i, k in enumerate(field.space.keys)}
    if any(k not in index for s in selections for k in s):
        raise ValueError("geometric selection names an absent atom")
    indices = tuple(tuple(index[k] for k in s) for s in selections)
    declaration = field.axes[0].name.split("/", 2)
    if len(declaration) != 3 or not declaration[1] or not declaration[2]:
        raise ValueError("Cartesian coordinates require a nonempty unit and frame")
    unit = declaration[1]
    space = CoordinateSpace("geometry/"+field.space.name+"/"+reading+"/"+unit+"^"+str(power), tuple(_key(s) for s in selections))
    return space, indices


def conformation_field(field, reading, selections):
    """Evaluate supplied geometry without fitting an alignment.

    The oriented volume reading is the determinant, or six signed tetrahedron volumes.
    """
    space, indices = _geometry_contract(field, reading, selections)
    values = tuple(_geometry(field.values, i, reading) for i in indices)
    return TensorField(space, values, source=field.source, dependencies=field.dependencies,
                       provenance=(*field.provenance, field.coefficient_digest, "supplied_coordinate_geometry"))


def conformation_direction(field, direction, reading, selections, *, parameter, parameter_unit):
    """Differentiate the declared geometry in a supplied coordinate direction."""
    space, indices = _geometry_contract(field, reading, selections)
    if not isinstance(direction, TensorField) or (direction.space, direction.axes) != (field.space, field.axes):
        raise ValueError("coordinate derivative must match the conformation axes")
    direction.check_state()
    if not all(isinstance(s, str) and s for s in (parameter, parameter_unit)):
        raise ValueError("sensitivity requires a parameter name and unit")
    dual = [[_Dual(x, dx) for x, dx in zip(row, drow, strict=True)] for row, drow in zip(field.values, direction.values, strict=True)]
    values = tuple(_geometry(dual, i, reading).direction for i in indices)
    target = CoordinateSpace(space.name+"/per/"+parameter+"/"+parameter_unit, space.keys)
    return TensorField(target, values, dependencies=_refs(field, direction),
                       provenance=(field.coefficient_digest, direction.coefficient_digest, "exact_directional_derivative"))
