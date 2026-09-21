"""Registered molecular notation readers with exact source attachments."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import gzip
import json
from numbers import Integral
from pathlib import Path
import re

import numpy as np

from . import EdgeConstruction

__all__ = ["SmilesBundle", "read_smiles", "load_smiles", "load_reaction_smiles", "match_smarts"]


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value):
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")
    return value


def _rational(value):
    if isinstance(value, bool):
        raise TypeError("exact molecular values reject booleans")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("exact molecular values require integers, Fractions or rational text")


def _qrecord(value):
    value = _rational(value)
    return [str(value.numerator), str(value.denominator)]


def _backend():
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise ImportError("the smiles reader requires the optional rdkit parser") from exc
    return Chem, rdBase.rdkitVersion


def _tokens(text, offset):
    """Retain lexical occurrences without deciding molecular chemistry."""
    if not text or not text.isascii():
        raise ValueError("SMILES requires a nonempty ASCII notation field")
    output = []
    i = 0
    while i < len(text):
        start = i
        ch = text[i]
        if ch == "[":
            stop = text.find("]", i + 1)
            if stop < 0:
                raise ValueError("unclosed bracket atom")
            i = stop + 1
            kind = "atom"
        elif text[i:i+2] in {"Cl", "Br"}:
            i += 2
            kind = "atom"
        elif ch in "BCNOPSFIbcnops*":
            i += 1
            kind = "atom"
        elif ch.isdigit():
            i += 1
            kind = "ring"
        elif ch == "%" and re.fullmatch(r"%[0-9]{2}", text[i:i+3]):
            i += 3
            kind = "ring"
        elif ch in "-=#$:/\\":
            i += 1
            kind = "bond"
        elif ch in "().":
            i += 1
            kind = {"(": "open", ")": "close", ".": "disconnect"}[ch]
        else:
            raise ValueError("unsupported SMILES token at position " + str(i))
        output.append({"kind": kind, "text": text[start:i],
                       "start": offset + start, "stop": offset + i})
    return output


def _connections(tokens):
    atoms, bonds, stack, rings = [], [], [], {}
    current, pending = None, []
    for ti, token in enumerate(tokens):
        kind = token["kind"]
        if kind == "atom":
            new = len(atoms)
            atoms.append(ti)
            if current is not None:
                bonds.append((current, new, tuple(pending)))
            elif pending:
                raise ValueError("bond token has no left atom")
            current, pending = new, []
        elif kind == "bond":
            if pending:
                raise ValueError("consecutive bond tokens are not supported")
            pending = [ti]
        elif kind == "open":
            if current is None or pending:
                raise ValueError("invalid branch opening")
            stack.append(current)
        elif kind == "close":
            if not stack or pending:
                raise ValueError("invalid branch closing")
            current = stack.pop()
        elif kind == "disconnect":
            if current is None or pending:
                raise ValueError("invalid disconnected component")
            current = None
        else:
            if current is None:
                raise ValueError("ring reference has no atom")
            key = int(token["text"].lstrip("%"))
            if key in rings:
                other, first_tokens = rings.pop(key)
                bonds.append((other, current, (*first_tokens, *pending, ti)))
            else:
                rings[key] = (current, (*pending, ti))
            pending = []
    if stack or rings or pending or current is None:
        raise ValueError("incomplete SMILES syntax")
    return atoms, bonds


def _parse_molecule(text, offset, name, role, scope, Chem):
    tokens = _tokens(text, offset)
    atom_tokens, connections = _connections(tokens)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.parseName = False
    params.allowCXSMILES = False
    mol = Chem.MolFromSmiles(text, params)
    if mol is None:
        raise ValueError("invalid molecular structure for " + name)
    if mol.GetNumAtoms() != len(atom_tokens) or mol.GetNumBonds() != len(connections):
        raise ValueError("parser and source incidence counts disagree")
    parsed = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b for b in mol.GetBonds()}
    if len(parsed) != len(connections) or set(parsed) != {tuple(sorted((a, b))) for a, b, _ in connections}:
        raise ValueError("parser and lexical bond identities disagree")
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    complete = bool(maps) and all(maps) and len(set(maps)) == len(maps)
    coordinate_scope = scope if complete else scope + "/occurrence/" + _digest((name, text))
    atoms = []
    for atom, token_index in zip(mol.GetAtoms(), atom_tokens, strict=True):
        atoms.append({"key": "map/"+str(atom.GetAtomMapNum()) if complete else "occurrence/"+str(atom.GetIdx()),
                      "map": str(atom.GetAtomMapNum()), "element": atom.GetSymbol(),
                      "atomic_number": atom.GetAtomicNum(), "isotope": atom.GetIsotope(),
                      "charge": atom.GetFormalCharge(), "hydrogens": atom.GetTotalNumHs(),
                      "radicals": atom.GetNumRadicalElectrons(), "aromatic": atom.GetIsAromatic(),
                      "chiral_tag": str(atom.GetChiralTag()),
                      "cip": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
                      "token": token_index})
    orders = {"SINGLE": "1", "DOUBLE": "2", "TRIPLE": "3", "QUADRUPLE": "4", "AROMATIC": "3/2"}
    bonds = []
    for a, b, refs in connections:
        bond = parsed[tuple(sorted((a, b)))]
        kind = str(bond.GetBondType())
        if kind not in orders:
            raise ValueError("unsupported bond interpretation " + kind)
        bonds.append({"atoms": [a, b], "kind": kind, "order": orders[kind],
                      "aromatic": bond.GetIsAromatic(), "direction": str(bond.GetBondDir()),
                      "stereo": str(bond.GetStereo()), "stereo_atoms": list(bond.GetStereoAtoms()),
                      "parser_begin": bond.GetBeginAtomIdx(), "parser_end": bond.GetEndAtomIdx(),
                      "tokens": list(dict.fromkeys((atom_tokens[a], *refs, atom_tokens[b])))})
    return {"name": name, "role": role, "smiles": text, "coordinate_scope": coordinate_scope,
            "complete_mapping": complete, "atoms": atoms, "bonds": bonds, "tokens": tokens,
            "conformers": {}, "conditions": {}, "time": None}


def _conditions(values):
    result = {}
    for name, value in (values or {}).items():
        _name(name, "condition name")
        if not isinstance(value, dict) or set(value) - {"value", "unit", "label", "origin", "source"}:
            raise ValueError("condition requires a declared quantity or category")
        if ("value" in value) == ("label" in value):
            raise ValueError("condition requires either value or label")
        entry = {"origin": _name(value.get("origin"), "condition origin"),
                 "source": _name(value.get("source"), "condition source")}
        if "value" in value:
            entry.update(value=_qrecord(value["value"]), unit=_name(value.get("unit"), "condition unit"))
        else:
            entry["label"] = _name(value["label"], "condition label")
        result[name] = entry
    return result


@dataclass(frozen=True)
class SmilesBundle:
    """One immutable source bundle with declared molecular realizations."""
    manifest_json: str
    fields: tuple

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    @property
    def digest(self):
        return self.construction().source_manifest["bundle_digest"]

    def construction(self):
        from rexgraph.span import SpanAttachment, SpanBlock
        manifest = self.manifest
        labels, sources, targets, branching, kinds, attrs, identities = [], [], [], [], [], {0: {}, 1: {}}, []
        pairs, groups = [], []
        for record in manifest["molecules"]:
            atoms = record["atoms"]
            base = len(labels)
            for i, atom in enumerate(atoms):
                index = base + i
                atom["cell"] = index
                label = _json((manifest["document_id"], record["name"], "atom", atom["key"]))
                labels.append(label)
                attrs[0][index] = {"molecular_record": record["name"], "atom_key": atom["key"],
                                   "element": atom["element"], "formal_charge": atom["charge"],
                                   "isotope": atom["isotope"], "atom_map": atom["map"]}
                token = record["tokens"][atom["token"]]
                block = SpanBlock(label+"/source", manifest["byte_axis"], "byte",
                                  (("atom", token["start"], token["stop"]),))
                attrs[0][index]["molecular_source"] = SpanAttachment(label+"/source", label, "atom", manifest["source_digest"], text=block)
            used = set()
            for bi, bond in enumerate(record["bonds"]):
                a, b = bond["atoms"]
                used.update((a, b))
                bond_id = sorted((atoms[a]["key"], atoms[b]["key"])) if record["complete_mapping"] else bi
                ident = _json((manifest["document_id"], record["name"], "bond", bond_id))
                components = tuple(("token/"+str(t), record["tokens"][t]["start"], record["tokens"][t]["stop"])
                                   for t in bond["tokens"])
                block = SpanBlock(ident+"/source", manifest["byte_axis"], "byte", components)
                values = {"molecular_record": record["name"], "bond_kind": bond["kind"],
                          "bond_order_exact": bond["order"], "bond_stereo": bond["stereo"],
                          "molecular_source": SpanAttachment(ident+"/source", ident, "bond", manifest["source_digest"], text=block)}
                pairs.append((base+a, base+b, bond["kind"], ident, values, bond))
            for i in range(len(atoms)):
                if i not in used:
                    ident = _json((manifest["document_id"], record["name"], "isolated_atom", atoms[i]["key"]))
                    groups.append(([base+i], "atomic_witness", ident, {"molecular_record": record["name"], "construction": "isolated atom assertion"}))
            if manifest["profile"]["aromatic_systems"]:
                adj = {}
                for bond in record["bonds"]:
                    if bond["aromatic"]:
                        a, b = bond["atoms"]
                        adj.setdefault(a, set()).add(b)
                        adj.setdefault(b, set()).add(a)
                seen = set()
                for root in sorted(adj):
                    if root in seen:
                        continue
                    todo, members = [root], []
                    while todo:
                        a = todo.pop()
                        if a in seen:
                            continue
                        seen.add(a)
                        members.append(a)
                        todo.extend(sorted(adj[a] - seen))
                    members.sort()
                    if len(members) > 2:
                        ident = _json((manifest["document_id"], record["name"], "aromatic_system", sorted(atoms[i]["key"] for i in members)))
                        groups.append(([base+i for i in members], "aromatic_system", ident,
                                       {"molecular_record": record["name"], "construction": "parser aromatic component"}))
        for cell, (a, b, kind, ident, values, bond) in enumerate(pairs):
            sources.append(a)
            targets.append(b)
            kinds.append(kind)
            attrs[1][cell] = values
            identities.append(ident)
            bond["cell"] = cell
        for support, kind, ident, values in groups:
            cell = len(pairs) + len(branching)
            branching.append(support)
            kinds.append(kind)
            attrs[1][cell] = values
            identities.append(ident)
        for record_name, conformer_name, tensor in self.fields:
            record = next(r for r in manifest["molecules"] if r["name"] == record_name)
            key = "conformer/" + conformer_name
            cell = record["atoms"][0]["cell"]
            attrs[0][cell][key] = tensor
            record["conformers"][conformer_name]["cell"] = cell
            record["conformers"][conformer_name]["attribute"] = key
            record["conformers"][conformer_name]["field_digest"] = tensor.coefficient_digest
        manifest["bundle_digest"] = _digest({k: v for k, v in manifest.items() if k != "bundle_digest"})
        names = sorted(set(kinds))
        type_indices = np.asarray([names.index(k) for k in kinds], dtype=np.int32)
        numeric = [int(sha256(i.encode()).hexdigest()[:16], 16) & ((1 << 63)-1) for i in identities]
        if len(numeric) != len(set(numeric)):
            raise ValueError("primary molecular identity hash collision")
        count = len(kinds)
        return EdgeConstruction(np.asarray(sources, dtype=np.int32), np.asarray(targets, dtype=np.int32),
            np.asarray([Fraction(1)]*count, dtype=object), np.ones(count, dtype=np.float64),
            type_indices, labels, len(names), names, branching=branching, attributes=attrs,
            origin="smiles", relation_ids=np.asarray(numeric, dtype=np.int64), source_manifest=manifest)


def read_smiles(path, *, document_id, reaction=False, map_namespace=None, aromatic_systems=True,
                conformers=None, conditions=None, times=None):
    """Read a SMILES table with a declared parser and exact optional observations."""
    from rexgraph.tensor_field import TensorField
    from rexgraph.type_accession import CoordinateSpace
    _name(document_id, "document_id")
    if not isinstance(reaction, bool) or not isinstance(aromatic_systems, bool):
        raise TypeError("reader policies must be boolean")
    if map_namespace is not None:
        _name(map_namespace, "map_namespace")
    path = Path(path)
    raw = path.read_bytes()
    decoded = gzip.decompress(raw) if path.suffix.lower() == ".gz" else raw
    decoded.decode("utf-8", errors="strict")
    source_digest = sha256(decoded).hexdigest()
    Chem, version = _backend()
    records, offset, row_names = [], 0, set()
    for row, line in enumerate(decoded.splitlines(keepends=True), 1):
        content = line.rstrip(b"\r\n")
        match = re.fullmatch(rb"[ \t]*(\S+)(?:[ \t]+(.*?))?[ \t]*", content)
        if not content.strip():
            offset += len(line)
            continue
        if match is None:
            raise ValueError("invalid SMILES table row " + str(row))
        notation = match.group(1).decode("ascii")
        name = (match.group(2) or ("row/"+str(row)).encode()).decode("utf-8").strip()
        _name(name, "molecule name")
        if name in row_names:
            raise ValueError("molecule row names must be unique")
        row_names.add(name)
        start = offset + match.start(1)
        if reaction:
            parts = notation.split(">")
            if len(parts) != 3 or not parts[0] or not parts[2]:
                raise ValueError("reaction reader requires nonempty reactants and products")
            local = 0
            for role, part in zip(("reactants", "agents", "products"), parts, strict=True):
                if part:
                    records.append(_parse_molecule(part, start+local, name+"/"+role, role,
                                                   map_namespace or document_id+"/"+name, Chem))
                local += len(part) + 1
        else:
            if ">" in notation:
                raise ValueError("use the reaction reader for reaction notation")
            records.append(_parse_molecule(notation, start, name, "molecule",
                                           map_namespace or document_id+"/"+name, Chem))
        offset += len(line)
    if not records:
        raise ValueError("SMILES table has no molecules")
    names = {r["name"] for r in records}
    for supplied in (conformers, conditions, times):
        if supplied is not None and (not isinstance(supplied, dict) or set(supplied)-names):
            raise ValueError("observation keys must name supplied molecular records")
    fields = []
    for record in records:
        record["conditions"] = _conditions((conditions or {}).get(record["name"]))
        time = (times or {}).get(record["name"])
        if time is not None:
            if set(time) != {"value", "axis", "unit"}:
                raise ValueError("time requires value, axis and unit")
            record["time"] = {"value": _qrecord(time["value"]), "axis": _name(time["axis"], "time axis"),
                              "unit": _name(time["unit"], "time unit")}
        for name, declaration in (conformers or {}).get(record["name"], {}).items():
            _name(name, "conformer name")
            if set(declaration) != {"coordinates", "unit", "frame", "origin", "source"}:
                raise ValueError("conformer requires coordinates, unit, frame, origin and source")
            keys = tuple(a["key"] for a in record["atoms"])
            coords = declaration["coordinates"]
            if not isinstance(coords, dict) or set(coords) != set(keys):
                raise ValueError("conformer coordinates must name every atom exactly once")
            values = [[_rational(x) for x in coords[key]] for key in keys]
            if any(len(x) != 3 for x in values):
                raise ValueError("conformer positions need three Cartesian coordinates")
            profile = {k: _name(declaration[k], "conformer "+k) for k in ("unit", "frame", "origin", "source")}
            if "/" in profile["unit"]:
                raise ValueError("position unit must have one named length unit")
            field = TensorField(CoordinateSpace("molecular_atoms/"+record["coordinate_scope"], keys), values,
                                (CoordinateSpace("cartesian/"+profile["unit"]+"/"+profile["frame"], ("x", "y", "z")),),
                                provenance=(_digest(("supplied_conformer", profile)),))
            fields.append((record["name"], name, field))
            record["conformers"][name] = profile
    manifest = {"schema": "molecular_bundle/1", "document_id": document_id,
                "source_digest": source_digest, "byte_axis": "utf8/"+source_digest,
                "files": [{"name": path.name, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}],
                "profile": {"parser": "rdkit", "parser_version": version, "remove_hydrogens": False,
                            "reaction": reaction, "aromatic_systems": aromatic_systems,
                            "source_notation": "restricted_standard_smiles", "atom_maps": "explicit_unique_for_alignment"},
                "molecules": records}
    manifest["bundle_digest"] = _digest(manifest)
    return SmilesBundle(_json(manifest), tuple(fields))


def load_smiles(path, **options):
    return read_smiles(path, **options).construction()


def load_reaction_smiles(path, **options):
    if "reaction" in options:
        raise ValueError("reaction policy is fixed by the registered reader")
    return read_smiles(path, reaction=True, **options).construction()


def match_smarts(source, selection, pattern, *, max_matches=1000, use_chirality=True):
    """Return explicit substructure occurrences with their original source spans."""
    from rexgraph.molecular_field import MolecularView
    view = MolecularView.from_source(source, selection)
    if isinstance(max_matches, bool) or not isinstance(max_matches, int) or max_matches <= 0:
        raise ValueError("max_matches must be a positive integer")
    if not isinstance(use_chirality, bool):
        raise TypeError("use_chirality must be boolean")
    Chem, version = _backend()
    if view.manifest["profile"]["parser_version"] != version:
        raise ValueError("substructure evaluation requires the recorded parser version")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(view.record["smiles"], params)
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError("invalid SMARTS pattern")
    matches = mol.GetSubstructMatches(query, uniquify=True, useChirality=use_chirality, maxMatches=max_matches+1)
    if len(matches) > max_matches:
        raise ValueError("substructure result exceeds max_matches")
    atoms, bonds = view.record["atoms"], view.record["bonds"]
    result = []
    for match in matches:
        mapped_bonds = {tuple(sorted((match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]))) for b in query.GetBonds()}
        selected_bonds = [b for b in bonds if tuple(sorted(b["atoms"])) in mapped_bonds]
        token_ids = sorted({atoms[i]["token"] for i in match} | {t for b in selected_bonds for t in b["tokens"]})
        result.append({"atom_keys": tuple(atoms[i]["key"] for i in match),
                       "bond_cells": tuple(b["cell"] for b in selected_bonds),
                       "byte_spans": tuple((view.record["tokens"][i]["start"], view.record["tokens"][i]["stop"]) for i in token_ids)})
    return tuple(result)


def register(register_reader):
    register_reader("smiles", load_smiles, extensions=(".smi", ".smiles"))
    register_reader("reaction_smiles", load_reaction_smiles, extensions=(".rsmi", ".rsmiles"))
