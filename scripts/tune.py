"""Hyperparameter tuning. Forest: full grid per rho. LSTM: one Optuna study per scenario x rho.

Writes results/tuning/<model>_trials.csv, <model>_best.csv (read by the tuned main runs) and
<model>_tuning_output.txt.
Run from the repository root:  python scripts/tune.py --model forest   (or --model lstm)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd
from joblib import Parallel, delayed

from rtds.config import load_config
from rtds.grid import time_grid
from rtds.tuned import TUNING_DIR, best_path
from rtds.tuning import forest_candidate, lstm_study

SCENARIOS = ("A", "B")


def fixed_settings(cfg, model):
    """The fixed setting in the same form as the tuned ones."""
    if model == "forest":
        f = cfg["forest"]
        return {"max_features": round(f["max_features"] * cfg["simulation"]["n_visits"]),
                "min_samples_leaf": f["min_samples_leaf"]}
    return {k: cfg["lstm"][k] for k in ("hidden_size", "learning_rate", "batch_size", "patience", "weight_decay")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["forest", "lstm"])
    model = parser.parse_args().model
    cfg = load_config()
    grid = time_grid(cfg)
    rhos = cfg["simulation"]["rho_levels"]
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if model == "forest":
        space = cfg["tuning"]["forest"]
        cells = list(itertools.product(rhos, space["max_features"], space["min_samples_leaf"]))
        trials = Parallel(n_jobs=cfg["runs"]["n_jobs"], verbose=5)(
            delayed(forest_candidate)(cfg, rho, mf, leaf, grid) for rho, mf, leaf in cells)
    else:
        settings = [(s, rho) for s in SCENARIOS for rho in rhos]
        studies = Parallel(n_jobs=cfg["runs"]["n_jobs"], verbose=5)(
            delayed(lstm_study)(cfg, s, rho, k, grid) for k, (s, rho) in enumerate(settings))
        trials = [row for study in studies for row in study]
    minutes = (time.perf_counter() - start) / 60
    trials = pd.DataFrame(trials)
    fixed = json.dumps(fixed_settings(cfg, model))
    trials["is_fixed_setting"] = trials["settings"].apply(lambda s: json.loads(s) == json.loads(fixed))
    trials.to_csv(TUNING_DIR / f"{model}_trials.csv", index=False)

    # The chosen setting: lowest mean Brier score.
    best = []
    for (scen, rho), g in trials.groupby(["scenario", "rho"], sort=False):
        top = g.loc[g["ibs"].idxmin()]
        fx = g[g["is_fixed_setting"]].iloc[0]
        best.append({"model": model, "scenario": scen, "rho": rho, "settings": top["settings"],
                     "ibs": top["ibs"], "c_index": top["c_index"], "best_epoch": top["best_epoch"],
                     "fixed_ibs": fx["ibs"], "fixed_c_index": fx["c_index"], "fixed_best_epoch": fx["best_epoch"],
                     "same_as_fixed": bool(top["is_fixed_setting"])})
    best = pd.DataFrame(best)
    best.to_csv(best_path(model), index=False)

    lines = [f"Stage 6f, {model}: {len(trials)} candidates scored in {minutes:.1f} min "
             f"(tuning datasets {cfg['tuning']['first_dataset']}-"
             f"{cfg['tuning']['first_dataset'] + cfg['tuning']['n_datasets'] - 1}; score = held-out integrated Brier score)",
             f"Fixed setting: {fixed}", ""]
    for _, r in best.iterrows():
        lines.append(f"scenario {r['scenario']}, rho = {r['rho']:g}: chosen {r['settings']}")
        lines.append(f"  Brier {r['ibs']:.5f} (fixed {r['fixed_ibs']:.5f}), C-index {r['c_index']:.4f} "
                     f"(fixed {r['fixed_c_index']:.4f}), same as fixed: {r['same_as_fixed']}"
                     + (f", best epoch {r['best_epoch']:.0f} (fixed {r['fixed_best_epoch']:.0f})" if model == "lstm" else ""))
    print("\n".join(lines))
    (TUNING_DIR / f"{model}_tuning_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
