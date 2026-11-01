"""Is 100 background rows enough? Run 3 on one dataset, each size compared with the largest. Not a result.

Writes results/scenario_A/background_check.csv.
Run from the repository root:  python scripts/check_background_size.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rtds import seeds
from rtds.config import load_config
from rtds.explain_dataset import explain_dataset
from rtds.forest import ForestModel
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.metrics import c_index
from rtds.results import TABLE1_COLUMNS, scenario_dir
from rtds.selection import background_rows
from rtds.simulate import simulate_dataset

SCENARIO = "A"
VERSION = "forest"
RUN = 3


def nested_order(cfg, dataset_index):
    """Training row indices: the real background first, then the other rows in a seeded order."""
    base = background_rows(cfg, dataset_index)
    rest = np.setdiff1d(np.arange(cfg["simulation"]["n_train"]), base)
    extra = seeds.rng(cfg["master_seed"], "background_check", dataset_index).permutation(rest)
    return np.concatenate([base, extra])


def main():
    cfg = load_config()
    check = cfg["background_check"]
    i, rho = check["dataset_index"], check["rho"]
    grid = time_grid(cfg)
    builder = FlatVisitColumns()
    driver_col = f"visit_{cfg['scenarios'][SCENARIO]}"

    ds = simulate_dataset(cfg, i, rho, SCENARIO)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
    model = ForestModel(cfg, i).fit(X_train, ds.train.y)
    model.assert_grid_inside(X_test, grid)
    c = c_index(ds.test.y, model.predict_risk(X_test))
    order = nested_order(cfg, i)
    assert np.array_equal(np.sort(order[:cfg["tool"]["n_background"]]), np.sort(background_rows(cfg, i)))

    def run_size(size):
        start = time.perf_counter()
        rows, _ = explain_dataset(cfg, model, ds, i, builder, grid, driver_col,
                                  key=(VERSION, SCENARIO, rho, RUN), heldout_c_index=c,
                                  background_idx=order[:size])
        return size, pd.DataFrame(rows, columns=TABLE1_COLUMNS).assign(background_size=size), time.perf_counter() - start

    done = Parallel(n_jobs=len(check["sizes"]))(delayed(run_size)(size) for size in check["sizes"])
    tables = {size: t for size, t, _ in done}
    seconds = {size: sec for size, _, sec in done}

    out = pd.concat(tables.values(), ignore_index=True)
    out.to_csv(scenario_dir(SCENARIO) / "background_check.csv", index=False)

    ref_size = max(check["sizes"])
    ref = tables[ref_size].set_index("patient_id")
    print(f"\ndataset {i}, rho = {rho}, {len(ref)} patients; compared with the {ref_size}-row background")
    for size, t in tables.items():
        t = t.set_index("patient_id").loc[ref.index]
        print(f"  {size:4d} rows: hit rate = {t['hit'].mean():.4f}, mean share = {t['credit_share'].mean():.4f}, "
              f"same top visit as {ref_size} = {(t['top_visit'] == ref['top_visit']).mean():.2f}, "
              f"mean |share - share_{ref_size}| = {(t['credit_share'] - ref['credit_share']).abs().mean():.4f}, "
              f"{seconds[size] / len(t):.2f} s per patient")


if __name__ == "__main__":
    main()
