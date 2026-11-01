"""The results tables: their columns and how they are written."""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def scenario_dir(scenario, version="forest"):
    """The fixed forest sits at the top of the scenario's folder, other versions in a subfolder."""
    base = RESULTS_DIR / f"scenario_{scenario}"
    return base if version == "forest" else base / version

TABLE1_COLUMNS = [
    "version", "scenario", "rho", "dataset_seed", "patient_id", "run", "true_driver_visit",
    "credit_visit_1", "credit_visit_2", "credit_visit_3", "top_visit", "hit",
    "credit_share", "heldout_c_index",
]
# Run 2; positive = the model got worse.
TABLE2_COLUMNS = ["version", "scenario", "rho", "dataset_seed", "visit", "c_index_drop", "ibs_increase"]

# Per dataset: the model's C-index, and the true risk's as a reference.
DATASET_COLUMNS = ["version", "scenario", "rho", "dataset_seed", "heldout_c_index", "true_risk_c_index"]

# Ties and NaN shares.
LOG_COLUMNS = ["version", "scenario", "rho", "dataset_seed", "patient_id", "run", "n_tied", "share_is_nan"]


def start_table(path, columns):
    """Create (or overwrite) a table with just its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_rows(path, rows, columns):
    """Append rows to a table."""
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=False, index=False)
