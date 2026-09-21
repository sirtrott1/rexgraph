"""Checked record projection; never Python attribute traversal or method calls."""
from __future__ import annotations

from collections.abc import Mapping

from .types import Domain, Exactness, RCType, ShapeRef, ValueKind


def project(value, name):
    from rexgraph.chain_map import SymmetryGroup
    if isinstance(value, SymmetryGroup):
        if name not in {"word", "map", "inverse", "identity", "sizes", "generator_count"}:
            raise TypeError(f"SymmetryGroup has no declared member {name!r}")
        value.check_state()
        return getattr(value, name)
    from rexgraph.rational_operator import ResolventGroup
    if isinstance(value, ResolventGroup):
        if name not in {"word", "scales", "inverse", "identity"}:
            raise TypeError(f"ResolventGroup has no declared member {name!r}")
        value.check_state()
        return getattr(value, name)
    from rexgraph.void_state import VoidState
    if isinstance(value, VoidState):
        if name not in {"source_state", "region", "potential", "void_indices", "columns", "shape", "n_voids", "n_potential", "strain", "homology"}:
            raise TypeError(f"VoidState has no declared member {name!r}")
        value.check_state()
        return getattr(value, name)
    from rexgraph.relative_quotient import RelativeQuotient
    if isinstance(value, RelativeQuotient):
        if name not in {"projection", "sizes", "boundaries", "cell_maps", "removed_cells", "readings",
                        "source_boundary_digest", "coefficient_digest", "residuals"}:
            raise TypeError(f"RelativeQuotient has no declared member {name!r}")
        value.check_state()
        return getattr(value, name)
    from rexgraph.boundary_difference import BoundaryDifference
    if isinstance(value, BoundaryDifference):
        if name not in {"shape", "entries", "readings", "vertex_keys", "relation_pairs", "matching", "source_digests"}:
            raise TypeError(f"BoundaryDifference has no declared member {name!r}")
        value.check_state()
        return getattr(value, name)
    # Explicit read only result fields, never arbitrary Python traversal. The
    # lazy import keeps parse-only/type-only callers independent of the core.
    if not isinstance(value, Mapping):
        from rexgraph.io.partition_state import RexPartition
        if isinstance(value, RexPartition) and name in {"rex", "manifest", "digest", "cell_maps"}:
            value.check_state()
            return getattr(value, name)
        from rexgraph.cell_neighborhood import Hyperslice
        from rexgraph.column_expansion import ColumnExpansion, ColumnLegs, PrimaryColumnLift
        if isinstance(value, Hyperslice) and name in {"cell", "below", "above", "lateral"}:
            return getattr(value, name)
        if isinstance(value, ColumnExpansion) and name in {"legs", "lift", "references", "groups", "coefficient_digest"}:
            value.check_state()
            return getattr(value, name)
        if isinstance(value, (ColumnLegs, PrimaryColumnLift)) and name in {"shape", "entries"}:
            value.expansion.check_state()
            return getattr(value, name)
        from rexgraph.chain_map import ChainHomotopy
        if isinstance(value, ChainHomotopy) and name in {"residuals", "shapes", "coefficient_digest"}:
            value.check_state()
            return getattr(value, name)
        from rexgraph.cells import Cell
        if isinstance(value, Cell) and name in {"grade", "index"}:
            return getattr(value, name)
        # CatalogEntry is an explicit metadata value, not a source handle.
        from rexgraph.io.catalog import CatalogEntry
        if isinstance(value, CatalogEntry) and name in value.__dataclass_fields__:
            return getattr(value, name)
        from rexgraph.section_calculus import SectionFamily, SectionImage
        if isinstance(value, SectionFamily) and name == "dimension":
            value.check_state()
            return value.dimension
        if isinstance(value, SectionImage) and name == "determined":
            value.family.check_state()
            return value.determined
        from rexgraph.sheaf import ExactGlueResult, ExactSectionCheck
        allowed = ({"compatible", "incidence_count", "comparison_count", "obstructions"}
                   if isinstance(value, ExactSectionCheck) else
                   {"ratio", "gluable", "glued", "h0", "obstruction_count", "components", "obstructions", "failed_pairs"}
                   if isinstance(value, ExactGlueResult) else set())
        if name in allowed:
            return getattr(value, name)
    if name.startswith("_") or not isinstance(value, Mapping):
        raise TypeError("member access requires a public record key, not an object attribute")
    if name not in value:
        raise KeyError(f"record has no member {name!r}")
    return value[name]


def member_type(parent, name, context):
    if isinstance(parent, Mapping):
        return project(parent, name)
    if not isinstance(parent, RCType):
        raise TypeError("member access requires a declared record")
    if parent.name == "ReadoutEquivalence":
        kinds = {"equivalent": ValueKind.BOOLEAN, "family_dimension": ValueKind.EXACT_INTEGER,
                 "family_digest": ValueKind.TEXT, "left_digest": ValueKind.TEXT,
                 "right_digest": ValueKind.TEXT, "certificate_digest": ValueKind.TEXT,
                 "scope": ValueKind.TEXT, "offset": ValueKind.TENSOR_FIELD,
                 "variation": ValueKind.COORDINATE_MAP}
        if name not in kinds:
            raise TypeError("unknown readout certificate member")
        return RCType(kinds[name].value, kind=kinds[name], source=parent.source,
                      exactness=Exactness.RATIONAL if name == "offset" else Exactness.STRUCTURAL)
    if parent.name == "PathChange" and parent.kind is ValueKind.RECORD:
        kinds = {"terms": ValueKind.SEQUENCE, "seeds": ValueKind.SEQUENCE,
                 "weights": ValueKind.SEQUENCE, "event": ValueKind.BOOLEAN,
                 "carried": ValueKind.EXACT_INTEGER, "n_turns": ValueKind.EXACT_INTEGER,
                 "baseline_turns": ValueKind.EXACT_INTEGER, "H_T": ValueKind.UNKNOWN,
                 "status": ValueKind.TEXT}
        if name not in kinds:
            raise TypeError(f"PathChange has no declared member {name!r}")
        return RCType(kinds[name].value, kind=kinds[name],
            domain=Domain.REAL if name in {"weights", "H_T"} else Domain.METADATA,
            exactness=Exactness.APPROXIMATE if name in {"weights", "H_T"} else Exactness.STRUCTURAL,
            source=parent.source, temporal=parent.temporal)
    if parent.name == "CorpusField" and parent.kind is ValueKind.RECORD:
        kinds = {"scores": ValueKind.SEQUENCE, "ids": ValueKind.SEQUENCE,
                 "versions": ValueKind.SEQUENCE, "fields": ValueKind.SEQUENCE,
                 "snapshot_digest": ValueKind.TEXT, "reading": ValueKind.TEXT,
                 "coefficient_domain": ValueKind.TEXT, "exact": ValueKind.BOOLEAN,
                 "as_of": ValueKind.UNKNOWN, "valid_at": ValueKind.UNKNOWN}
        if name not in kinds:
            raise TypeError(f"CorpusField has no declared member {name!r}")
        return RCType(kinds[name].value, kind=kinds[name],
            source=parent.source, temporal=parent.temporal,
            domain=parent.domain if name == "scores" else Domain.METADATA,
            exactness=(Exactness.RATIONAL if parent.domain is Domain.RATIONAL else Exactness.APPROXIMATE)
                      if name == "scores" else Exactness.STRUCTURAL)
    if parent.name == "SymmetryGroup" and parent.kind is ValueKind.RECORD:
        if name in {"inverse", "identity"}:
            return parent
        kinds = {"word": ValueKind.SEQUENCE, "sizes": ValueKind.SEQUENCE,
                 "generator_count": ValueKind.EXACT_INTEGER, "map": ValueKind.CHAIN_MAP}
        if name not in kinds:
            raise TypeError(f"SymmetryGroup has no declared member {name!r}")
        return RCType(kinds[name].value, kind=kinds[name],
                      domain=Domain.RATIONAL if name == "map" else Domain.INTEGER,
                      exactness=Exactness.STRUCTURAL if name == "map" else Exactness.INTEGER,
                      source=parent.source, temporal=parent.temporal)
    if parent.operator is not None and dict(parent.operator.parameters).get("kind") == "RESOLVENT_GROUP":
        if name in {"inverse", "identity"}:
            from dataclasses import replace
            parameters = dict(parent.operator.parameters)
            parameters["word"] = () if name == "identity" else tuple(-i for i in reversed(parameters["word"]))
            return parent.with_(operator=replace(parent.operator, parameters=tuple(parameters.items())))
        if name in {"word", "scales"}:
            return RCType("GroupCoefficients", kind=ValueKind.SEQUENCE, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source, temporal=parent.temporal)
        raise TypeError(f"ResolventGroup has no declared member {name!r}")
    if parent.name in {"VoidState", "VoidHomology"} and parent.kind is ValueKind.RECORD:
        names = ({"source_state", "region", "potential", "void_indices", "columns", "shape", "n_voids", "n_potential", "strain", "homology"}
                 if parent.name == "VoidState" else {"rank_before", "rank_after", "independent_fillings", "methods"})
        if name not in names:
            raise TypeError(f"{parent.name} has no declared member {name!r}")
        integer = name in {"n_voids", "n_potential", "strain", "rank_before", "rank_after", "independent_fillings"}
        return RCType("VoidHomology" if name == "homology" else "VoidMember",
                      kind=ValueKind.RECORD if name == "homology" else ValueKind.EXACT_INTEGER if integer else
                      ValueKind.TEXT if name == "source_state" else ValueKind.SEQUENCE,
                      domain=Domain.INTEGER if integer else Domain.METADATA,
                      exactness=Exactness.INTEGER if integer else Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    if parent.name == "LineageAlignment" and parent.kind is ValueKind.RECORD:
        names = {"keys", "entries", "presence", "cell_maps", "shape", "steps", "times",
                 "identity", "coefficient_domain", "implicit_zero", "missing"}
        if name not in names:
            raise TypeError(f"LineageAlignment has no declared member {name!r}")
        text = name in {"identity", "coefficient_domain", "missing"}
        return RCType("AlignmentMember", kind=ValueKind.TEXT if text else ValueKind.EXACT_INTEGER
                      if name == "implicit_zero" else ValueKind.SEQUENCE,
                      domain=Domain.INTEGER if name == "implicit_zero" else Domain.METADATA,
                      exactness=Exactness.INTEGER if name == "implicit_zero" else Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    if parent.name == "RelationValidation" and parent.kind is ValueKind.RECORD:
        if name not in {"grade", "valid", "closed", "nonzero", "integral", "storable", "residuals", "accepted"}:
            raise TypeError(f"RelationValidation has no declared member {name!r}")
        integer, rational = name in {"grade", "accepted"}, name == "residuals"
        return RCType("Integer" if name == "grade" else "Sequence",
                      kind=ValueKind.EXACT_INTEGER if name == "grade" else ValueKind.SEQUENCE,
                      domain=Domain.INTEGER if integer else Domain.RATIONAL if rational else Domain.METADATA,
                      exactness=Exactness.INTEGER if integer else Exactness.RATIONAL if rational else Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    if parent.name in {"RelativeQuotient", "RelativeHomology"} and parent.kind is ValueKind.RECORD:
        kinds = ({"projection": ValueKind.CHAIN_MAP, "sizes": ValueKind.SEQUENCE,
                  "boundaries": ValueKind.SEQUENCE, "cell_maps": ValueKind.SEQUENCE,
                  "removed_cells": ValueKind.SEQUENCE, "residuals": ValueKind.SEQUENCE,
                  "readings": ValueKind.RECORD, "source_boundary_digest": ValueKind.TEXT,
                  "coefficient_digest": ValueKind.TEXT} if parent.name == "RelativeQuotient" else
                 {"betti": ValueKind.SEQUENCE, "ranks": ValueKind.SEQUENCE, "methods": ValueKind.SEQUENCE})
        if name not in kinds:
            raise TypeError(f"{parent.name} has no declared member {name!r}")
        integer = name in {"sizes", "betti", "ranks"}
        rational = name == "residuals"
        return RCType("RelativeHomology" if name == "readings" else kinds[name].value,
                      kind=kinds[name], domain=Domain.INTEGER if integer else
                      Domain.RATIONAL if rational or name == "projection" else Domain.METADATA,
                      exactness=Exactness.INTEGER if integer else Exactness.RATIONAL if rational else
                      Exactness.STRUCTURAL, source=parent.source, temporal=parent.temporal)
    if parent.kind is ValueKind.RECORD and parent.name in {"SectionResponse", "SemanticClosure"}:
        kinds = ({"scores": ValueKind.SEQUENCE, "labels": ValueKind.SEQUENCE,
                  "layer": ValueKind.TEXT, "reading": ValueKind.TEXT, "seed_weight": ValueKind.TEXT,
                  "coefficient_domain": ValueKind.TEXT, "exact": ValueKind.BOOLEAN}
                 if parent.name == "SectionResponse" else
                 {"seed": ValueKind.EXACT_INTEGER, "grade": ValueKind.EXACT_INTEGER,
                  "depth": ValueKind.UNKNOWN, "converged": ValueKind.BOOLEAN,
                  "steps": ValueKind.SEQUENCE, "relations": ValueKind.SEQUENCE,
                  "vertices": ValueKind.SEQUENCE, "reading": ValueKind.TEXT})
        if name not in kinds:
            raise TypeError(f"{parent.name} has no declared member {name!r}")
        numeric = name == "scores"
        integer = kinds[name] is ValueKind.EXACT_INTEGER
        return RCType(kinds[name].value, kind=kinds[name], source=parent.source, temporal=parent.temporal,
            domain=parent.domain if numeric else Domain.INTEGER if integer else Domain.METADATA,
            exactness=(Exactness.RATIONAL if parent.domain is Domain.RATIONAL else Exactness.APPROXIMATE)
                      if numeric else Exactness.INTEGER if integer else Exactness.STRUCTURAL)
    if parent.kind is ValueKind.BOUNDARY_DIFFERENCE:
        if name not in {"shape", "entries", "readings", "vertex_keys", "relation_pairs", "matching", "source_digests"}:
            raise TypeError(f"BoundaryDifference has no declared member {name!r}")
        return RCType("DifferenceMember", kind=ValueKind.RECORD if name == "readings" else
                      ValueKind.TEXT if name == "matching" else ValueKind.SEQUENCE,
                      domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    if parent.name in {"ChannelMoments", "CoordinateDelta"} and parent.kind is ValueKind.RECORD:
        allowed = ({"names", "values", "shape", "total", "kernel_digest", "coefficient_domain"}
                   if parent.name == "ChannelMoments" else
                   {"names", "space_name", "space_keys", "fields", "values", "shape", "moments",
                    "operation_digest", "coefficient_domain"})
        if name not in allowed:
            raise TypeError(f"{parent.name} has no declared member {name!r}")
        if name == "moments":
            return RCType("ChannelMoments", kind=ValueKind.RECORD, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source, temporal=parent.temporal)
        if name == "total":
            return RCType("Rational", kind=ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                          exactness=Exactness.RATIONAL, source=parent.source, temporal=parent.temporal)
        kind = ValueKind.SEQUENCE if name in {"names", "values", "shape", "fields", "space_keys"} else ValueKind.TEXT
        return RCType(kind.value, kind=kind, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    if parent.name == "AccessionDelta" and parent.kind is ValueKind.RECORD:
        if name in {"grade", "implicit_zero"}:
            rational = name == "implicit_zero"
            return RCType("Rational" if rational else "Integer",
                          kind=ValueKind.EXACT_RATIONAL if rational else ValueKind.EXACT_INTEGER,
                          domain=Domain.RATIONAL if rational else Domain.INTEGER,
                          exactness=Exactness.RATIONAL if rational else Exactness.INTEGER,
                          source=parent.source, temporal=parent.temporal)
        if name in {"shape", "entries", "domain_keys", "codomain_keys"}:
            return RCType("Coordinates", kind=ValueKind.SEQUENCE, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source, temporal=parent.temporal)
        if name in {"domain_name", "codomain_name", "formula", "coefficient_domain",
                    "old_accession_digest", "new_accession_digest", "correspondence_digest", "output_correspondence_digest"}:
            return RCType("Text", kind=ValueKind.TEXT, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                          source=parent.source, temporal=parent.temporal)
        raise TypeError(f"AccessionDelta has no declared member {name!r}")
    if parent.name == "FieldDelta" and parent.kind is ValueKind.RECORD:
        if name in {"moment", "oriented_moment", "down_quadrance", "up_quadrance"}:
            return RCType("Rational", kind=ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL, exactness=Exactness.RATIONAL,
                          source=parent.source, temporal=parent.temporal)
        if name in {"down", "up"}:
            return RCType("DefectCoordinates", kind=ValueKind.RECORD, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                          source=parent.source, temporal=parent.temporal)
        if name not in {"grade", "metrics", "coefficient_domain", "correspondence_digest", "source_boundary_digest", "target_boundary_digest", "metric_digest"}:
            raise TypeError(f"FieldDelta has no declared member {name!r}")
        if name == "grade":
            return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER, exactness=Exactness.INTEGER,
                          source=parent.source, temporal=parent.temporal)
        return RCType("Text", kind=ValueKind.TEXT, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    if parent.name == "DefectCoordinates" and parent.kind is ValueKind.RECORD:
        if name in {"grade", "implicit_zero"}:
            return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER, exactness=Exactness.INTEGER,
                          source=parent.source, temporal=parent.temporal)
        if name in {"keys", "shape", "values"}:
            return RCType("Coordinates", kind=ValueKind.SEQUENCE, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                          source=parent.source, temporal=parent.temporal)
        if name == "name":
            return RCType("OptionalCoordinateName", domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                          source=parent.source, temporal=parent.temporal)
        raise TypeError(f"DefectCoordinates has no declared member {name!r}")
    if parent.name == "ParquetExport" and parent.kind is ValueKind.RECORD:
        kinds = {"payload": ValueKind.ARTIFACT_BYTES, "manifest": ValueKind.RECORD, "digest": ValueKind.TEXT}
        if name not in kinds:
            raise TypeError(f"ParquetExport has no declared member {name!r}")
        return RCType(kinds[name].value, kind=kinds[name], domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL)
    if parent.kind is ValueKind.REX_PARTITION:
        kinds = {"rex": ValueKind.REX, "manifest": ValueKind.RECORD,
                 "digest": ValueKind.TEXT, "cell_maps": ValueKind.SEQUENCE}
        if name not in kinds:
            raise TypeError(f"RexPartition has no declared member {name!r}")
        return RCType(kinds[name].value, kind=kinds[name], domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL, source=parent.source)
    if parent.kind is ValueKind.HYPERSLICE:
        from dataclasses import replace
        from .types import Variance
        if name not in {"cell", "below", "above", "lateral"}:
            raise TypeError(f"Hyperslice has no declared member {name!r}")
        if name == "below" and parent.grade == 0:
            return None
        grade = parent.grade + {"cell": 0, "lateral": 0, "below": -1, "above": 1}[name]
        kind = ValueKind.CELL if name == "cell" else ValueKind.CELL_SET
        return parent.with_(name=kind.value, kind=kind, grade=grade, variance=Variance.CELL,
                            basis=replace(parent.basis, grade=grade))
    if parent.kind is ValueKind.COLUMN_EXPANSION:
        kinds = {"legs": ValueKind.COLUMN_LEGS, "lift": ValueKind.PRIMARY_COLUMN_LIFT}
        if name in kinds:
            return parent.with_(name=kinds[name].value, kind=kinds[name])
        if name in {"references", "groups", "coefficient_digest"}:
            return RCType("ExpansionDiagnostic", kind=ValueKind.TEXT if name == "coefficient_digest" else ValueKind.SEQUENCE,
                          domain=Domain.METADATA, exactness=Exactness.STRUCTURAL, source=parent.source)
        raise TypeError(f"ColumnExpansion has no declared member {name!r}")
    if parent.kind in {ValueKind.COLUMN_LEGS, ValueKind.PRIMARY_COLUMN_LIFT}:
        if name not in {"shape", "entries"}:
            raise TypeError(f"column factor has no declared member {name!r}")
        return RCType("FactorDiagnostic", kind=ValueKind.SEQUENCE, domain=Domain.METADATA,
                      exactness=Exactness.STRUCTURAL, source=parent.source)
    if parent.kind is ValueKind.CHAIN_HOMOTOPY:
        if name not in {"residuals", "shapes", "coefficient_digest"}:
            raise TypeError(f"ChainHomotopy has no declared member {name!r}")
        return RCType("HomotopyDiagnostic", kind=ValueKind.TEXT if name == "coefficient_digest" else ValueKind.SEQUENCE,
                      domain=Domain.METADATA, exactness=Exactness.STRUCTURAL, source=parent.source)
    if parent.kind is ValueKind.CELL and name in {"grade", "index"}:
        return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
                      exactness=Exactness.INTEGER, source=parent.source, temporal=parent.temporal)
    if parent.kind is ValueKind.HODGE_SPLIT:
        if name not in {"gradient", "curl", "harmonic"}:
            raise TypeError(f"HodgeSplit has no member {name!r}")
        shape = parent.shape
        if shape is None and context.native:
            shape = ShapeRef((context.grade(parent.grade),))
        return parent.with_(name="Cochain", kind=ValueKind.COCHAIN, shape=shape)
    if parent.kind in {ValueKind.SECTION_FAMILY, ValueKind.SECTION_IMAGE}:
        if name == "dimension" and parent.kind is ValueKind.SECTION_FAMILY:
            return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
                          exactness=Exactness.INTEGER, source=parent.source)
        if name == "determined" and parent.kind is ValueKind.SECTION_IMAGE:
            return RCType("Boolean", kind=ValueKind.BOOLEAN, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source)
        raise TypeError(f"{parent.kind.value} has no declared member {name!r}")
    if parent.kind in {ValueKind.EXACT_GLUE, ValueKind.EXACT_SECTION_CHECK}:
        integer = ({"gluable", "glued", "h0", "obstruction_count"}
                   if parent.kind is ValueKind.EXACT_GLUE else {"incidence_count", "comparison_count"})
        sequences = ({"components", "obstructions", "failed_pairs"}
                     if parent.kind is ValueKind.EXACT_GLUE else {"obstructions"})
        if name in integer:
            return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
                          exactness=Exactness.INTEGER, source=parent.source)
        if name == "compatible" and parent.kind is ValueKind.EXACT_SECTION_CHECK:
            return RCType("Boolean", kind=ValueKind.BOOLEAN, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source)
        if name == "ratio" and parent.kind is ValueKind.EXACT_GLUE:
            return RCType("Rational", kind=ValueKind.EXACT_RATIONAL, domain=Domain.RATIONAL,
                          exactness=Exactness.RATIONAL, source=parent.source)
        if name in sequences:
            return RCType("SectionDiagnosticSequence", kind=ValueKind.SEQUENCE, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source)
        raise TypeError(f"{parent.kind.value} has no declared member {name!r}")
    if parent.kind is ValueKind.STRUCTURAL_DESCRIPTION:
        if parent.name == "Dependence":
            if name in {"rank", "nullity"}:
                return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
                              exactness=Exactness.INTEGER, source=parent.source)
            if name == "dependent":
                return RCType("Boolean", kind=ValueKind.BOOLEAN, domain=Domain.METADATA,
                              exactness=Exactness.STRUCTURAL, source=parent.source)
            if name in {"kernel", "coordinates"}:
                return RCType("FamilyKernel" if name == "kernel" else "Text",
                              kind=ValueKind.SEQUENCE if name == "kernel" else ValueKind.TEXT,
                              domain=Domain.METADATA, exactness=Exactness.STRUCTURAL, source=parent.source)
            raise TypeError(f"Dependence has no declared member {name!r}")
        if name in {"nV", "nE", "nF", "dimension"}:
            return RCType("Integer", kind=ValueKind.EXACT_INTEGER, domain=Domain.INTEGER,
                          exactness=Exactness.INTEGER, source=parent.source, temporal=parent.temporal)
        if name == "kind":
            return RCType("Text", kind=ValueKind.TEXT, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL, source=parent.source)
        if name in {"grades", "cells", "boundaries", "betti"}:
            return RCType("IntegerSequenceView", kind=ValueKind.SEQUENCE,
                          domain=Domain.INTEGER, exactness=Exactness.INTEGER, source=parent.source)
        raise TypeError(f"StructuralDescription has no member {name!r}")
    # These signatures promise metadata records, not arbitrary live objects.
    # Key existence is data dependent; the plan must not execute the reader to
    # discover it. The runtime checks the Mapping protocol and actual key.
    if parent.kind in {ValueKind.STORE_STATS, ValueKind.CATALOG_ENTRY, ValueKind.SECURITY_STATUS, ValueKind.RECORD}:
        return RCType("MetadataMember", domain=Domain.METADATA, exactness=Exactness.STRUCTURAL,
                      source=parent.source, temporal=parent.temporal)
    raise TypeError(f"{parent.kind.value} has no declared record-member contract")
