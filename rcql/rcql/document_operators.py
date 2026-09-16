"""Document query adapters; Core owns the response arithmetic and closure."""
from .execution_trace import record_method


def _seeds(source, value):
    from rexgraph.cells import Cell, CellSet
    from .operators import _typed_cells
    if isinstance(value, (Cell, CellSet)):
        _typed_cells(source, value, operator="document response")
        if value.grade != 0:
            raise ValueError("document response requires C0 seeds")
        return [value.index] if isinstance(value, Cell) else value.indices
    return value


def document_field(source, seeds, reading="mass", seed_weight="invdeg", exact=True):
    from rexgraph.partition import document_field as core_field
    result = core_field(source, _seeds(source, seeds), reading=reading, seed_weight=seed_weight, exact=exact)
    record_method("core-sparse-rational-response", reading=reading, exact=exact, seed_weight=seed_weight)
    return result


def section_response(source, layer, seeds, reading="mass", seed_weight="invdeg", exact=True):
    from rexgraph.partition import section_response as core_response
    from .document_contracts import section_layer
    scores, labels = core_response(source, section_layer(source, layer), _seeds(source, seeds),
                                  propagator=reading, seed_weight=seed_weight, exact=exact)
    record_method("core-sparse-rational-response", reading=reading, exact=exact, seed_weight=seed_weight, layer=layer)
    return {"scores": tuple(scores.tolist()), "labels": tuple(labels), "layer": layer,
            "reading": reading, "seed_weight": seed_weight,
            "coefficient_domain": "Q" if exact else "real", "exact": exact}


def text_overlap_view(source):
    from rexgraph.text_overlap import TextOverlapView
    result = TextOverlapView(source)
    record_method("core-primary-text-overlap", diagonal="removed")
    return result


def install(register):
    from .operators import closure
    register("DOCUMENT_FIELD")(document_field)
    register("SECTION_RESPONSE")(section_response)
    register("SEMANTIC_CLOSURE")(closure)
    register("TEXT_OVERLAP_VIEW")(text_overlap_view)
