"""Score one explained patient: top visit, hit (with ties) and credit share of the driver."""

import math

import numpy as np


def score_patient(credit, driver_col, tie_rel_tol):
    """credit: {visit column: credit}."""
    names = list(credit)
    values = np.array([credit[n] for n in names])
    best = values.max()
    tied = [n for n, v in zip(names, values) if v >= best * (1 - tie_rel_tol)]
    total = values.sum()
    return {
        # Ties are joined, e.g. "1,3".
        "top_visit": ",".join(n.removeprefix("visit_") for n in tied),
        "hit": 1 / len(tied) if driver_col in tied else 0.0,
        "n_tied": len(tied),
        # NaN if all credits are 0; the caller logs it.
        "credit_share": credit[driver_col] / total if total > 0 else math.nan,
    }
