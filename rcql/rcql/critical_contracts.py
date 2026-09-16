"""Explicit sigma family declarations and critical operator contracts."""
from __future__ import annotations

from fractions import Fraction

from .types import Domain, Exactness, OperatorDescriptor, RCType, ShapeRef, ValueKind


ARGUMENTS = {
    "SIGMA_OPERATOR": (("sigma", "amplitudes", "rates", "channels", "c_channel", "grade"), (1,)),
    "CRITICAL_COMMUTATOR": (("sigma", "amplitudes", "rates", "channels", "c_channel", "grade"), (1,)),
    "CRITICAL_RATE": (("amplitudes", "rates", "channels", "c_channel", "reading", "grade"), ("generator", 1)),
}


def refine(typed, context):
    from rexgraph.sigma_operator import _source_state, family_parameters
    if not context.native:
        raise TypeError("sigma actions require a native Rex source")
    args, name = typed.args, typed.operator
    reading = "operator"
    if name == "CRITICAL_RATE":
        reading = args[4] if len(args) > 4 else "generator"
        if reading not in ("generator", "slope"):
            raise ValueError("critical rate reading must be generator or slope")
        parameters = family_parameters(0.5, *args[:4], args[5] if len(args) > 5 else 1)
    else:
        parameters = family_parameters(*args[:5], args[5] if len(args) > 5 else 1)
    sigma, amplitudes, rates, channels, c_channel = parameters
    if len(amplitudes) != context.grade(0):
        raise ValueError("sigma family must name every vertex in the canonical C0 order")
    _source_state(context.binding.value)
    n = context.grade(1)
    basis = typed.result.basis
    declarations = (("family", "explicit-exponential-vertex"), ("sigma", sigma),
                    ("amplitudes", amplitudes), ("rates", rates), ("channels", channels),
                    ("g_channel", "raw"), ("c_channel", c_channel),
                    ("column_normalization", "unit-before-relation-metric"),
                    ("channel_normalization", "trace"), ("reading", reading),
                    ("relation_metric_rescaling", "common-absolute-maximum-cancels-in-hats"),
                    ("derivative", "analytic"))
    ordinary = name == "SIGMA_OPERATOR"
    desc = OperatorDescriptor("sigma-family" if ordinary else "sigma-critical", basis, basis, (n, n),
                              Domain.REAL, Exactness.APPROXIMATE, symmetric=ordinary, psd=ordinary,
                              transpose_available=True, exact_action=False, exact_transpose=False,
                              action_variance="cochain", parameters=declarations,
                              euclidean_skew_adjoint=not ordinary)
    return typed.result.with_(operator=desc, shape=ShapeRef((n, n)))


def install(register):
    from .signatures import OperatorSignature, TypePattern
    number = TypePattern("sigma", literal=(int, float, Fraction))
    sequence = TypePattern("explicit sequence", literal=(list, tuple))
    choice = TypePattern("C channel", literal=str)
    grade = TypePattern("grade", literal=int, optional=True)
    family = (sequence, sequence, sequence, choice)
    for name in ARGUMENTS:
        inputs = (number, *family, grade) if name != "CRITICAL_RATE" else (
            *family, TypePattern("reading", literal=str, optional=True), grade)
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=inputs,
            result=RCType("SigmaAction", grade=1, kind=ValueKind.OPERATOR, domain=Domain.METADATA,
                          exactness=Exactness.STRUCTURAL),
            implementation_key="rex." + name.lower(), memoizable=True,
            preconditions=("explicit exponential vertex amplitudes and rates, channel list and C convention",
                           "C1 only; raw G channel, fixed relation metric, unit boundary columns, trace normalized channels",
                           "numerical action and analytic derivative; finite runtime range is checked on evaluation")))
