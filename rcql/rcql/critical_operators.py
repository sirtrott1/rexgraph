"""Adapters to the core sigma family; RCQL does not reconstruct its channels."""
from .execution_trace import record_method


def sigma_operator(source, sigma, amplitudes, rates, channels, c_channel, grade=1):
    from rexgraph.sigma_operator import sigma_operator as build
    result = build(source, sigma, amplitudes, rates, channels, c_channel, grade)
    record_method("native-incidence-sigma", arithmetic="numerical", family="explicit-exponential-vertex",
                  derivative="analytic", channels=tuple(channels), c_channel=c_channel)
    return result


def critical_commutator(source, sigma, amplitudes, rates, channels, c_channel, grade=1):
    from rexgraph.sigma_operator import critical_commutator as build
    result = build(source, sigma, amplitudes, rates, channels, c_channel, grade)
    record_method("native-factored-critical-commutator", arithmetic="numerical", sigma=float(sigma))
    return result


def critical_rate(source, amplitudes, rates, channels, c_channel, reading="generator", grade=1):
    from rexgraph.sigma_operator import critical_rate as build
    result = build(source, amplitudes, rates, channels, c_channel, reading, grade)
    record_method("native-analytic-critical-rate", arithmetic="numerical", reading=reading, sigma=0.5)
    return result


ADAPTERS = {"SIGMA_OPERATOR": sigma_operator, "CRITICAL_COMMUTATOR": critical_commutator,
            "CRITICAL_RATE": critical_rate}


def install(register):
    for name, fn in ADAPTERS.items():
        register(name)(fn)
