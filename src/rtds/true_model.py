"""The simulator's true survival function, shaped like a fitted sksurv model (Run 1)."""

import numpy as np
from sksurv.functions import StepFunction


class TrueSurvivalModel:
    def __init__(self, h0, b, driver_column, grid):
        self.h0 = h0
        self.b = b
        self.driver_column = driver_column
        # A StepFunction is exact only at its own x-points, so the explainer must always get this grid.
        self.grid = np.asarray(grid, dtype=float)

    def fit(self, X, y):
        return self  # nothing to learn

    def predict_survival_function(self, X):
        x = X[self.driver_column].to_numpy(dtype=float)
        rate = self.h0 * np.exp(self.b * x)
        surv = np.exp(-np.outer(rate, self.grid))
        return np.array([StepFunction(self.grid, s) for s in surv])
