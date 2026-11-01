"""Run 1 (sanity check): SurvSHAP(t) on the true survival function. The driver should get all the credit.

Writes results/scenario_<X>/table1_run1.csv and log_run1.csv.
Run from the repository root:  python scripts/run1_true_curve.py [--scenario B]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd

from rtds.cli import parse_scenario
from rtds.config import load_config
from rtds.explain_dataset import explain_dataset
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.results import LOG_COLUMNS, TABLE1_COLUMNS, append_rows, scenario_dir, start_table
from rtds.simulate import simulate_dataset
from rtds.true_model import TrueSurvivalModel

VERSION = "forest"  # table label only; Run 1 explains the truth, not the forest
RUN = 1


def main():
    scenario = parse_scenario(__doc__)
    cfg = load_config()
    sim = cfg["simulation"]
    driver_col = f"visit_{cfg['scenarios'][scenario]}"
    grid = time_grid(cfg)
    builder = FlatVisitColumns()
    model = TrueSurvivalModel(sim["h0"], sim["b"], driver_col, grid)

    out = scenario_dir(scenario)
    table_path, log_path = out / "table1_run1.csv", out / "log_run1.csv"
    start_table(table_path, TABLE1_COLUMNS)
    start_table(log_path, LOG_COLUMNS)

    start = time.perf_counter()
    for rho in sim["rho_levels"]:
        for i in range(cfg["runs"]["run1_n_datasets"]):
            dataset = simulate_dataset(cfg, i, rho, scenario)
            rows, log = explain_dataset(cfg, model, dataset, i, builder, grid, driver_col,
                                        key=(VERSION, scenario, rho, RUN))
            append_rows(table_path, rows, TABLE1_COLUMNS)
            append_rows(log_path, log, LOG_COLUMNS)
            print(f"rho = {rho}, dataset {i}: done ({time.perf_counter() - start:.0f} s elapsed)", flush=True)

    summarise(pd.read_csv(table_path), pd.read_csv(log_path), driver_col)


def summarise(table, log, driver_col):
    others = [c for c in ("credit_visit_1", "credit_visit_2", "credit_visit_3") if c != f"credit_{driver_col}"]
    print(f"\nRun 1 summary ({len(table)} explained patients; average over patients per dataset, then over datasets)")
    for rho, g in table.groupby("rho"):
        per_ds = g.groupby("dataset_seed")[["hit", "credit_share"]].mean()
        print(f"  rho = {rho}: hit = {per_ds['hit'].mean():.4f}, credit share = {per_ds['credit_share'].mean():.9f}, "
              f"lowest single-patient share = {g['credit_share'].min():.9f}, "
              f"largest non-driver credit = {g[others].to_numpy().max():.2e}, "
              f"driver credit range = {g[f'credit_{driver_col}'].min():.3f} to {g[f'credit_{driver_col}'].max():.3f}")
    print(f"  ties logged: {int((log['n_tied'] > 1).sum())}, NaN shares logged: {int(log['share_is_nan'].sum())}")


if __name__ == "__main__":
    main()
