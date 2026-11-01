"""Run 2: how much the model relies on each visit, by shuffling it in the held-out set.

Two measures from the same shuffles: the C-index drop (ranking only) and the integrated
Brier score rise (the whole curve, so the closer comparison to SurvSHAP(t) credit).
"""

import numpy as np

from rtds import seeds
from rtds.metrics import c_index, ibs


def run2(cfg, model, X_test, y_train, y_test, grid, dataset_index):
    """{visit column: (C-index drop, Brier increase)}; positive = shuffling made the model worse."""
    gen = seeds.rng(cfg["master_seed"], "shuffle", dataset_index)
    base_c = c_index(y_test, model.predict_risk(X_test))
    base_ibs = ibs(model, X_test, y_train, y_test, grid)
    out = {}
    for col in X_test.columns:
        c_vals, ibs_vals = [], []
        for _ in range(cfg["runs"]["run2_n_shuffles"]):
            X = X_test.copy()
            X[col] = gen.permutation(X[col].to_numpy())
            c_vals.append(c_index(y_test, model.predict_risk(X)))
            ibs_vals.append(ibs(model, X, y_train, y_test, grid))
        out[col] = (base_c - np.mean(c_vals), np.mean(ibs_vals) - base_ibs)
    return out


def table2_rows(version, scenario, rho, dataset_index, drops):
    """Run 2 results as Table 2 rows, one per visit."""
    return [[version, scenario, rho, dataset_index, int(col.removeprefix("visit_")), c, b]
            for col, (c, b) in drops.items()]
