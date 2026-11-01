"""Explain a dataset's chosen patients into Table 1 rows (Run 1 and Run 3)."""

import math

from rtds.explain_survshap import SurvShapExplainer
from rtds.scores import score_patient
from rtds.selection import background_rows, explained_patients


def explain_dataset(cfg, model, dataset, dataset_index, builder, grid, driver_col, key,
                    heldout_c_index=None, background_idx=None):
    """key: (version, scenario, rho, run). background_idx: only for the background-size check.
    Returns (Table 1 rows, log rows)."""
    tool = cfg["tool"]
    if background_idx is None:
        background_idx = background_rows(cfg, dataset_index)
    background = builder.build(dataset.train.visits[background_idx])
    patient_ids = explained_patients(cfg, dataset_index)
    patients = builder.build(dataset.test.visits[patient_ids])
    explainer = SurvShapExplainer(background, dataset.train.y[background_idx], grid,
                                  tool["calculation_method"], tool["aggregation_method"])
    version, scenario, rho, run = key
    driver = int(driver_col.removeprefix("visit_"))

    rows, log = [], []
    for row_pos, pid in enumerate(patient_ids):
        credit = explainer.explain(model, patients.iloc[[row_pos]].reset_index(drop=True))
        s = score_patient(credit, driver_col, cfg["scores"]["tie_rel_tol"])
        prefix = [version, scenario, rho, dataset_index, int(pid), run]
        rows.append(prefix + [driver, credit["visit_1"], credit["visit_2"], credit["visit_3"],
                              s["top_visit"], s["hit"], s["credit_share"], heldout_c_index])
        if s["n_tied"] > 1 or math.isnan(s["credit_share"]):
            log.append(prefix + [s["n_tied"], math.isnan(s["credit_share"])])
    return rows, log
