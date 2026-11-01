"""Compare a scenario's Table 1 with scenario A's after relabelling the visits.

For the order-blind forest the credits should match, visit for visit.
"""

import numpy as np
import pandas as pd


def visit_map(cfg, scenario):
    """{visit number in `scenario`: the same base column's visit number in scenario A}."""
    n = cfg["simulation"]["n_visits"]

    def order(s):
        d = cfg["scenarios"][s]
        return [d] + [k for k in range(1, n + 1) if k != d]

    a, s = order("A"), order(scenario)
    return {s[j]: a[j] for j in range(n)}


def compare_with_a(cfg, scenario, table_s, table_a):
    """Match rows on (rho, dataset_seed, patient_id) and return the largest differences."""
    m = table_s.merge(table_a, on=["rho", "dataset_seed", "patient_id"], suffixes=("", "_a"))
    rel = []
    for k, ka in visit_map(cfg, scenario).items():
        diff = (m[f"credit_visit_{k}"] - m[f"credit_visit_{ka}_a"]).abs()
        rel.append((diff / m[[f"credit_visit_{j}_a" for j in (1, 2, 3)]].sum(axis=1)).max())
    return {
        "rows_matched": len(m),
        "max_relative_credit_diff": float(np.max(rel)),
        "hits_identical": bool((m["hit"] == m["hit_a"]).all()),
        "max_share_diff": float((m["credit_share"] - m["credit_share_a"]).abs().max()),
        "c_index_identical": bool(np.array_equal(m["heldout_c_index"].to_numpy(), m["heldout_c_index_a"].to_numpy())),
    }
