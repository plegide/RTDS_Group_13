"""Do the training and held-out patients follow the same distribution? The split is not stratified.

Times and the driver are identical at every rho, so rho = 0 covers them all.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from rtds.simulate import simulate_dataset


def train_test_check(cfg, scenario="A"):
    """One row per dataset: censored fraction, median time and KS p-values, training vs held-out."""
    driver = cfg["scenarios"][scenario] - 1
    rows = []
    for i in range(cfg["runs"]["n_datasets"]):
        ds = simulate_dataset(cfg, i, 0.0, scenario)
        tr, te = ds.train, ds.test
        rows.append({"dataset_seed": i,
                     "censored_train": 1 - tr.event.mean(), "censored_test": 1 - te.event.mean(),
                     "median_time_train": np.median(tr.time), "median_time_test": np.median(te.time),
                     "ks_p_time": ks_2samp(tr.time, te.time).pvalue,
                     "ks_p_driver": ks_2samp(tr.visits[:, driver], te.visits[:, driver]).pvalue})
    return pd.DataFrame(rows)
