"""Pick the model by name. The pipeline only sees the SurvivalModel interface."""

from rtds.forest import ForestModel
from rtds.lstm_model import LSTMModel
from rtds.tuned import best_settings

# *_tuned: the settings chosen by scripts/tune.py; otherwise the fixed settings in config.toml.
VERSIONS = ("forest", "lstm", "forest_tuned", "lstm_tuned")


def make_model(cfg, version, dataset_index, grid, scenario=None, rho=None, replicate=None):
    if version not in VERSIONS:
        raise ValueError(f"unknown model version {version!r}; expected one of {VERSIONS}")
    base = version.removesuffix("_tuned")
    # Same seeds as the fixed model, so only the settings differ.
    settings = best_settings(base, scenario, rho) if version.endswith("_tuned") else None
    if base == "forest":
        return ForestModel(cfg, dataset_index, settings)
    return LSTMModel(cfg, dataset_index, grid, replicate, settings)
