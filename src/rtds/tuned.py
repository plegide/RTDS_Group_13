"""Read back the settings chosen by scripts/tune.py."""

import json

import pandas as pd

from rtds.results import RESULTS_DIR

TUNING_DIR = RESULTS_DIR / "tuning"


def best_path(model):
    return TUNING_DIR / f"{model}_best.csv"


def best_settings(model, scenario, rho):
    """The chosen settings as a dict."""
    best = pd.read_csv(best_path(model))
    if model == "forest":
        scenario = "A"  # tuned on A only: B is A with the visits relabelled
    row = best[(best["model"] == model) & (best["scenario"] == scenario) & (best["rho"] == rho)]
    assert len(row) == 1, f"no tuned {model} settings for scenario {scenario}, rho {rho}"
    return json.loads(row["settings"].iloc[0])
