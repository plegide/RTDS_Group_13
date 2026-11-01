"""The forest in scenario B vs scenario A with the visits relabelled: an order-blind model should match.

Writes results/scenario_B/table1_mirror.csv, table2_mirror.csv and mirror_check.csv.
Run from the repository root:  python scripts/check_forest_mirror.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd
from joblib import Parallel, delayed

from rtds.config import load_config
from rtds.grid import time_grid
from rtds.mirror import compare_with_a
from rtds.results import scenario_dir
from rtds.run_dataset import run_dataset, task_paths

SCENARIO = "B"


def main():
    cfg = load_config()
    grid = time_grid(cfg)
    base = scenario_dir(SCENARIO)
    out_dir = base / "main"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, rho) for i in range(cfg["mirror"]["n_datasets"]) for rho in cfg["simulation"]["rho_levels"]]
    Parallel(n_jobs=cfg["runs"]["n_jobs"])(delayed(run_dataset)(cfg, i, rho, SCENARIO, grid, out_dir) for i, rho in tasks)

    merged = {name: pd.concat([pd.read_csv(task_paths(out_dir, SCENARIO, rho, i)[name]) for i, rho in tasks],
                              ignore_index=True) for name in ("table1", "table2")}
    merged["table1"].to_csv(base / "table1_mirror.csv", index=False)
    merged["table2"].to_csv(base / "table2_mirror.csv", index=False)

    a = pd.read_csv(scenario_dir("A") / "table1_main.csv")
    rows = []
    for rho in cfg["simulation"]["rho_levels"]:
        b = merged["table1"][merged["table1"]["rho"] == rho]
        rows.append({"rho": rho, **compare_with_a(cfg, SCENARIO, b, a)})
    check = pd.DataFrame(rows)
    check.to_csv(base / "mirror_check.csv", index=False)

    t1 = merged["table1"]
    print(f"Forest, scenario B, datasets 0-{cfg['mirror']['n_datasets'] - 1}: "
          f"{len(t1)} explained patients compared with scenario A (visits relabelled)")
    print(check.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    summary = t1.groupby("rho")[["hit", "credit_share"]].mean()
    a_same = a[a["dataset_seed"] < cfg["mirror"]["n_datasets"]].groupby("rho")[["hit", "credit_share"]].mean()
    print("\nhit rate and credit share on these datasets, B vs A:")
    print(summary.join(a_same, rsuffix="_A").to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
