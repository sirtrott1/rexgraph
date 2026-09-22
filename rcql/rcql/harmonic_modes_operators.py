"""Read the core exact sector traces; RCQL forms no trace of its own."""
from .execution_trace import record_method


def _traces(source, grade, sector, weight, calculus):
    from rexgraph.harmonic_modes import grade_traces
    if calculus is not None and calculus.source is not None and calculus.source is not source:
        raise ValueError("field calculus belongs to another selected source")
    traces = grade_traces(source if calculus is None else calculus, grade)
    record_method("native-exact-sector-traces", grade=traces.grade, sector=sector, weight=weight,
                  size=traces.size, betti=traces.betti,
                  metric=("field-calculus" if calculus is not None
                          else "identity" if source.edge_metric_exact is None else "declared"))
    return traces


def effective_modes(source, grade, sector="completed", weight="unit", calculus=None):
    return _traces(source, grade, sector, weight, calculus).effective_modes(sector, weight)


def harmonic_log(source, grade, sector="completed", weight="unit", calculus=None):
    return _traces(source, grade, sector, weight, calculus).harmonic_log(sector, weight)


def install(register):
    register("EFFECTIVE_MODES")(effective_modes)
    register("HARMONIC_LOG")(harmonic_log)
