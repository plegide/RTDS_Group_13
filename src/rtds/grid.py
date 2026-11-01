"""The fixed time grid the tool is evaluated on, identical for every dataset and run."""

import numpy as np

from rtds.simulate import reference_sample


def time_grid(cfg):
    tool = cfg["tool"]
    times = reference_sample(cfg).time  # observed times, censored or not
    lo, hi = np.quantile(times, [tool["grid_lower_quantile"], tool["grid_upper_quantile"]])
    return np.linspace(lo, hi, tool["n_grid"])
