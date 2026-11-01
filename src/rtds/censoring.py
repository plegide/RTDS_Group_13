"""Solve the exponential censoring rate that gives the target censored fraction."""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

# Gauss-Hermite quadrature: deterministic, so the rate is the same on every machine.
_NODES, _WEIGHTS = hermegauss(100)
_WEIGHTS = _WEIGHTS / np.sqrt(2 * np.pi)


def expected_over_driver(fn):
    """E[fn(x)] for x ~ N(0, 1), by quadrature."""
    return np.sum(_WEIGHTS * fn(_NODES))


def censored_fraction(rate, h0, b):
    # P(censored | x) = c / (c + lam(x)), with lam(x) = h0*exp(b*x), averaged over x ~ N(0, 1).
    return expected_over_driver(lambda x: rate / (rate + h0 * np.exp(b * x)))


def solve_censoring_rate(h0, b, target):
    return brentq(lambda c: censored_fraction(c, h0, b) - target, 1e-8, 1e3, xtol=1e-14)


def marginal_survival(t, h0, b):
    """Population S(t), for comparison with Kaplan-Meier."""
    t = np.atleast_1d(t)
    return np.array([expected_over_driver(lambda x: np.exp(-h0 * np.exp(b * x) * ti)) for ti in t])
