"""Score candidate hyperparameters on the tuning datasets, and the two searches.

Score: held-out integrated Brier score. No explanation is computed here, so the choice
depends on prediction quality only.
"""

import json

import numpy as np
import optuna

from rtds import seeds
from rtds.forest import ForestModel
from rtds.inputs import FlatVisitColumns
from rtds.lstm_model import LSTMModel
from rtds.metrics import c_index, ibs
from rtds.simulate import simulate_dataset


def tuning_datasets(cfg):
    t = cfg["tuning"]
    return range(t["first_dataset"], t["first_dataset"] + t["n_datasets"])


def score_settings(cfg, model_name, settings, scenario, rho, grid):
    """Brier score (lower = better), C-index and, for the LSTM, the kept epoch, averaged over the tuning datasets."""
    builder = FlatVisitColumns()
    ibs_vals, c_vals, epochs = [], [], []
    for i in tuning_datasets(cfg):
        ds = simulate_dataset(cfg, i, rho, scenario)
        X_train, X_test = builder.build(ds.train.visits), builder.build(ds.test.visits)
        if model_name == "forest":
            model = ForestModel(cfg, i, settings)
        else:
            model = LSTMModel(cfg, i, grid, settings=settings)
        model.fit(X_train, ds.train.y)
        model.assert_grid_inside(X_test, grid)
        ibs_vals.append(ibs(model, X_test, ds.train.y, ds.test.y, grid))
        c_vals.append(c_index(ds.test.y, model.predict_risk(X_test)))
        epochs.append(getattr(model, "best_epoch", np.nan))
    return {"ibs": float(np.mean(ibs_vals)), "c_index": float(np.mean(c_vals)), "best_epoch": float(np.mean(epochs))}


def forest_candidate(cfg, rho, max_features, min_samples_leaf, grid):
    """One cell of the forest grid (scenario A; B is A with the visits relabelled)."""
    settings = {"max_features": max_features, "min_samples_leaf": min_samples_leaf}
    return {"model": "forest", "scenario": "A", "rho": rho, "settings": json.dumps(settings),
            **settings, **score_settings(cfg, "forest", settings, "A", rho, grid)}


def lstm_study(cfg, scenario, rho, study_index, grid):
    """One Optuna study for one scenario and rho. Returns one row per trial."""
    space = cfg["tuning"]["lstm"]

    def objective(trial):
        settings = {
            "hidden_size": trial.suggest_categorical("hidden_size", list(space["hidden_size"])),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "batch_size": trial.suggest_categorical("batch_size", list(space["batch_size"])),
            "patience": trial.suggest_categorical("patience", list(space["patience"])),
        }
        # 0 cannot sit on a log scale, hence the switch.
        on = trial.suggest_categorical("use_weight_decay", [False, True])
        settings["weight_decay"] = trial.suggest_float("weight_decay", *space["weight_decay"], log=True) if on else 0.0
        s = score_settings(cfg, "lstm", settings, scenario, rho, grid)
        trial.set_user_attr("settings", json.dumps(settings))
        trial.set_user_attr("c_index", s["c_index"])
        trial.set_user_attr("best_epoch", s["best_epoch"])
        return s["ibs"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Seeded, with trials in sequence, so the study is reproducible.
    sampler = optuna.samplers.TPESampler(seed=seeds.int_seed(cfg["master_seed"], "tuning", study_index))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # Trial 0 is the fixed setting, so the search can only match or beat it.
    fixed = cfg["lstm"]
    study.enqueue_trial({k: fixed[k] for k in ("hidden_size", "learning_rate", "batch_size", "patience")}
                        | {"use_weight_decay": False})
    study.optimize(objective, n_trials=space["n_trials"])
    return [{"model": "lstm", "scenario": scenario, "rho": rho, "trial": t.number,
             "settings": t.user_attrs["settings"], **json.loads(t.user_attrs["settings"]),
             "ibs": t.value, "c_index": t.user_attrs["c_index"], "best_epoch": t.user_attrs["best_epoch"]}
            for t in study.trials]
