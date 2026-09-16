"""Exact sparse differences of accession measurements across an explicit map."""
from fractions import Fraction

from rexgraph.chain_map import _columns, _triples
from rexgraph.field_delta import validate_correspondence
from rexgraph.graded_boundary import _exact_compose_columns
from rexgraph.cells import cell_count
from rexgraph.type_accession import TypeAccession


def validate_accession_delta(old, new, correspondence):
    """Check axes and source states without computing either matrix product."""
    mapping = validate_correspondence(correspondence)
    if not isinstance(old, TypeAccession) or not isinstance(new, TypeAccession):
        raise TypeError("accession delta requires two TypeAccessions")
    if old.grade != new.grade or not 0 <= old.grade < len(mapping.components):
        raise ValueError("accession delta requires one carried grade at both endpoints")
    if old.source is not mapping.domain.source or new.source is not mapping.codomain.source:
        raise ValueError("accession endpoints must be the correspondence's actual source and target Rex")
    for value in (old, new):
        if not value.exact:
            raise TypeError("accession delta requires integer or rational measurements")
        if value.cell_keys is not None or value.n_cells != cell_count(value.source, value.grade):
            raise ValueError("accession delta requires current canonical ambient axes")
    if old.coordinates != new.coordinates:
        raise ValueError("rectangular accession delta requires the same declared output coordinates")
    return mapping


def accession_delta(old, new, correspondence):
    """Return A_new J minus J A_old, or A_new J minus A_old on a shared output.

    Ambient endomorphisms use the supplied J on both axes. Rectangular
    measurements instead require identical named output coordinates, declaring
    identity correspondence on that output. No output map, time interval,
    projection property or chain preservation of an accession is inferred.
    Products reuse the Core exact sparse column composition. Storage follows
    actual product support; fill is possible and no dense matrix is allocated.
    """
    mapping = validate_accession_delta(old, new, correspondence)
    grade = old.grade
    j = _columns(mapping.components[grade], old.n_cells)
    left = _exact_compose_columns(_columns(new.entries, new.n_cells), j)
    if old.coordinates is None:
        right = _exact_compose_columns(j, _columns(old.entries, old.n_cells))
        target = mapping.codomain.spaces[grade]
        formula = "A_new J - J A_old"
    else:
        right = _columns(old.entries, old.n_cells)
        target = old.coordinates
        formula = "A_new J - A_old"
    columns = []
    for a, b in zip(left, right, strict=True):
        difference = dict(a)
        for row, value in b.items():
            difference[row] = difference.get(row, Fraction(0)) - value
            if not difference[row]:
                del difference[row]
        columns.append(difference)
    domain = mapping.domain.spaces[grade]
    return {"grade": grade, "shape": (len(target.keys), old.n_cells),
            "entries": _triples(columns), "implicit_zero": Fraction(0),
            "domain_name": domain.name, "domain_keys": domain.keys,
            "codomain_name": target.name, "codomain_keys": target.keys,
            "formula": formula, "coefficient_domain": "Q",
            "old_accession_digest": old.coefficient_digest, "new_accession_digest": new.coefficient_digest,
            "correspondence_digest": mapping.coefficient_digest}
