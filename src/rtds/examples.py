"""Worked examples: SurvSHAP(t) over time for patients chosen by fixed rules.

Table 1 keeps only total credit, so each model is rebuilt from its seed and the curves recomputed.

A. typical:  forest, rho = typical_rho; the first patient in the seeded list.
B. failure:  forest, rho = failure_rho; the miss with the lowest driver share.
C. settings: LSTM, rho of the seed check; the patient of A, with fixed and with tuned settings.
"""

import numpy as np
import pandas as pd

from rtds.explain_survshap import SurvShapExplainer
from rtds.inputs import FlatVisitColumns
from rtds.models import make_model
from rtds.results import scenario_dir
from rtds.selection import background_rows, explained_patients
from rtds.simulate import simulate_dataset

SCENARIO = "A"
SAME = 1e-6  # largest relative gap allowed to the saved credit


def choose(cfg):
    """Apply the three rules to the saved tables. Returns one dict per explanation to draw."""
    ex = cfg["examples"]
    i = ex["dataset_index"]
    driver = cfg["scenarios"][SCENARIO]
    first = int(explained_patients(cfg, i)[0])
    forest = pd.read_csv(scenario_dir(SCENARIO) / "table1_main.csv")
    forest = forest[forest["dataset_seed"] == i]
    misses = forest[(forest["rho"] == ex["failure_rho"]) & (forest["hit"] < 1)].sort_values("credit_share")
    failing = int(misses["patient_id"].iloc[0])
    seed_rho = cfg["seed_stability"]["rho"]
    return [
        {"case": "A", "label": "Typical: forest, first seeded patient", "version": "forest", "rho": ex["typical_rho"],
         "patient": first, "settings": "fixed"},
        {"case": "B", "label": f"Failure: forest, worst of the {len(misses)} misses", "version": "forest",
         "rho": ex["failure_rho"], "patient": failing, "settings": "fixed"},
        {"case": "C", "label": "LSTM, fixed settings (main run)", "version": "lstm", "rho": seed_rho,
         "patient": first, "settings": "fixed"},
        {"case": "C", "label": "LSTM, tuned settings (main run)", "version": "lstm_tuned", "rho": seed_rho,
         "patient": first, "settings": "tuned"},
    ], driver


def saved_credits(cfg, item):
    i = cfg["examples"]["dataset_index"]
    t = pd.read_csv(scenario_dir(SCENARIO, item["version"]) / "table1_main.csv")
    row = t[(t["dataset_seed"] == i) & (t["rho"] == item["rho"]) & (t["patient_id"] == item["patient"])].iloc[0]
    return np.array([row[f"credit_visit_{k}"] for k in (1, 2, 3)]), row["heldout_c_index"]


def curves(cfg, item, grid):
    """Returns SurvSHAP(t) per visit and the total credits."""
    i = cfg["examples"]["dataset_index"]
    builder = FlatVisitColumns()
    ds = simulate_dataset(cfg, i, item["rho"], SCENARIO)
    X_train = builder.build(ds.train.visits)
    model = make_model(cfg, item["version"], i, grid, SCENARIO, item["rho"])
    model.fit(X_train, ds.train.y)
    bg = background_rows(cfg, i)
    explainer = SurvShapExplainer(builder.build(ds.train.visits[bg]), ds.train.y[bg], grid,
                                  cfg["tool"]["calculation_method"], cfg["tool"]["aggregation_method"])
    result = explainer.explain_full(model, builder.build(ds.test.visits[[item["patient"]]])).set_index("variable_name")
    tcols = [c for c in result.columns if c.startswith("t = ")]
    phi = np.array([result.loc[f"visit_{k}", tcols].to_numpy(dtype=float) for k in (1, 2, 3)])
    credit = np.array([result.loc[f"visit_{k}", "aggregated_change"] for k in (1, 2, 3)])
    return phi, credit


def run_examples(cfg, grid):
    """Returns the items with their curves, the driver and a summary table. Fails if a credit does not reproduce."""
    items, driver = choose(cfg)
    rows = []
    for item in items:
        item["phi"], credit = curves(cfg, item, grid)
        saved, c_index = saved_credits(cfg, item)
        gap = float(np.max(np.abs(credit - saved)) / saved.sum())
        assert gap <= SAME, f"case {item['case']}: recomputed credit differs from Table 1 ({gap:.1e})"
        item["credit"], item["c_index"] = credit, c_index
        rows.append({"case": item["case"], "label": item["label"], "model": item["version"], "scenario": SCENARIO,
                     "rho": item["rho"], "dataset_seed": cfg["examples"]["dataset_index"], "patient_id": item["patient"],
                     "settings": item["settings"], "model_c_index": c_index,
                     "credit_visit_1": credit[0], "credit_visit_2": credit[1], "credit_visit_3": credit[2],
                     "driver_share": credit[driver - 1] / credit.sum(), "top_visit": int(np.argmax(credit)) + 1,
                     "max_relative_gap_to_table1": gap})
    return items, driver, pd.DataFrame(rows)
