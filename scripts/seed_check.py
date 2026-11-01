"""Seed check: is the LSTM's choice between look-alike visits decided by the data or by training randomness?

Each dataset gets 5 extra LSTMs that differ only in the training seed, each through Run 2 and
Run 3; the main-run model is read back as a sixth. Does the favoured visit change between seeds?

Writes results/scenario_<X>/<model>/seed_check/.
Run from the repository root:  python scripts/seed_check.py [--scenario B] [--model lstm_tuned]
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rtds.cli import parse_scenario_and_model
from rtds.config import load_config
from rtds.explain_dataset import explain_dataset
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.metrics import c_index
from rtds.models import make_model
from rtds.reliance import run2, table2_rows
from rtds.results import TABLE1_COLUMNS, TABLE2_COLUMNS, scenario_dir
from rtds.simulate import simulate_dataset

MAIN_SEED = {"lstm": "6d", "lstm_tuned": "6f"}  # label of the main-run model


def one_model(cfg, version, scenario, rho, i, seed, grid):
    builder = FlatVisitColumns()
    driver_col = f"visit_{cfg['scenarios'][scenario]}"
    ds = simulate_dataset(cfg, i, rho, scenario)
    X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
    model = make_model(cfg, version, i, grid, scenario, rho, replicate=seed).fit(X_train, ds.train.y)
    model.assert_grid_inside(X_test, grid)
    c = c_index(ds.test.y, model.predict_risk(X_test))
    t2 = table2_rows(version, scenario, rho, i, run2(cfg, model, X_test, ds.train.y, ds.test.y, grid, i))
    t1, _ = explain_dataset(cfg, model, ds, i, builder, grid, driver_col,
                            key=(version, scenario, rho, 3), heldout_c_index=c)
    return [r + [str(seed)] for r in t1], [r + [str(seed)] for r in t2]


def favoured(values):
    """The visit (1, 2 or 3) with the largest value."""
    return int(np.argmax(values)) + 1


def main():
    scenario, version = parse_scenario_and_model(__doc__)
    assert version in MAIN_SEED, "the seed check is for the LSTM only (--model lstm or lstm_tuned)"
    cfg = load_config()
    p = cfg["seed_stability"]
    rho = p["rho"]
    grid = time_grid(cfg)
    out = scenario_dir(scenario, version) / "seed_check"
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(i, s) for i in range(p["n_datasets"]) for s in range(p["n_seeds"])]
    done = Parallel(n_jobs=cfg["runs"]["n_jobs"])(delayed(one_model)(cfg, version, scenario, rho, i, s, grid) for i, s in jobs)
    t1 = pd.DataFrame([r for rows, _ in done for r in rows], columns=TABLE1_COLUMNS + ["seed"])
    t2 = pd.DataFrame([r for _, rows in done for r in rows], columns=TABLE2_COLUMNS + ["seed"])

    # The main-run model, from its saved tables
    base = scenario_dir(scenario, version)
    keep = lambda t: t[(t["rho"] == rho) & (t["dataset_seed"] < p["n_datasets"])].assign(seed=MAIN_SEED[version])
    t1 = pd.concat([keep(pd.read_csv(base / "table1_main.csv")), t1], ignore_index=True)
    t2 = pd.concat([keep(pd.read_csv(base / "table2_main.csv")), t2], ignore_index=True)
    t1.to_csv(out / "table1_seeds.csv", index=False)
    t2.to_csv(out / "table2_seeds.csv", index=False)

    # One row per model
    total = t1[["credit_visit_1", "credit_visit_2", "credit_visit_3"]].sum(axis=1)
    for k in (1, 2, 3):
        t1[f"share_visit_{k}"] = t1[f"credit_visit_{k}"] / total
    driver = cfg["scenarios"][scenario]
    models = []
    for (i, seed), g in t1.groupby(["dataset_seed", "seed"]):
        r2 = t2[(t2["dataset_seed"] == i) & (t2["seed"] == seed)].sort_values("visit")
        shares = [g[f"share_visit_{k}"].mean() for k in (1, 2, 3)]
        models.append({"dataset_seed": i, "seed": seed, "c_index": g["heldout_c_index"].iloc[0],
                       "hit_rate": g["hit"].mean(), "driver_share": g["credit_share"].mean(),
                       "share_visit_1": shares[0], "share_visit_2": shares[1], "share_visit_3": shares[2],
                       "run3_favoured": favoured(shares),
                       "run2_c_favoured": favoured(r2["c_index_drop"].to_numpy()),
                       "run2_ibs_favoured": favoured(r2["ibs_increase"].to_numpy())})
    models = pd.DataFrame(models)
    models.to_csv(out / "models.csv", index=False)

    # One row per dataset
    datasets = []
    for i, g in models.groupby("dataset_seed"):
        pt = t1[t1["dataset_seed"] == i].pivot(index="patient_id", columns="seed", values="top_visit")
        pairs = list(itertools.combinations(pt.columns, 2))
        agree = np.mean([(pt[a] == pt[b]).mean() for a, b in pairs])
        datasets.append({"dataset_seed": i, "n_models": len(g),
                         "run3_favoured_visits": ",".join(str(v) for v in sorted(g["run3_favoured"].unique())),
                         "run2_c_favoured_visits": ",".join(str(v) for v in sorted(g["run2_c_favoured"].unique())),
                         "run2_ibs_favoured_visits": ",".join(str(v) for v in sorted(g["run2_ibs_favoured"].unique())),
                         "run3_flips": g["run3_favoured"].nunique() > 1,
                         "c_index_min": g["c_index"].min(), "c_index_max": g["c_index"].max(),
                         "hit_rate_min": g["hit_rate"].min(), "hit_rate_max": g["hit_rate"].max(),
                         "patient_top_visit_agreement": agree})
    datasets = pd.DataFrame(datasets)
    datasets.to_csv(out / "datasets.csv", index=False)

    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"Stage 6e: {version}, scenario {scenario} (driver visit_{driver}), rho = {rho}; "
        f"{p['n_datasets']} datasets x ({p['n_seeds']} extra seeds + the 6d model)")
    say(datasets.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    say(f"\nDatasets where the visit Run 3 favours changes between seeds: {int(datasets['run3_flips'].sum())} of {len(datasets)}")
    say(f"Mean share of patients for whom two seeds name the same top visit: "
        f"{datasets['patient_top_visit_agreement'].mean():.3f}")
    say(f"Spread of C-index between seeds of the same dataset: up to "
        f"{(datasets['c_index_max'] - datasets['c_index_min']).max():.4f}")
    say(f"Models where Run 3 favours the driver: {int((models['run3_favoured'] == driver).sum())} of {len(models)}; "
        f"where Run 2 (C-index) favours it: {int((models['run2_c_favoured'] == driver).sum())}; "
        f"where Run 2 (Brier) favours it: {int((models['run2_ibs_favoured'] == driver).sum())}")
    say(f"Models where Run 3 and Run 2 (Brier) favour the same visit: "
        f"{int((models['run3_favoured'] == models['run2_ibs_favoured']).sum())} of {len(models)}")
    (out / "seed_check_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
