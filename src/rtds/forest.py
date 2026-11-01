"""The random survival forest behind the SurvivalModel interface."""

from sksurv.ensemble import RandomSurvivalForest

from rtds import seeds


class ForestModel:
    def __init__(self, cfg, dataset_index, settings=None):
        # settings: tuned values; None = the fixed settings.
        f = {**cfg["forest"], **(settings or {})}
        self.rsf = RandomSurvivalForest(
            n_estimators=f["n_estimators"],
            max_features=f["max_features"],
            min_samples_leaf=f["min_samples_leaf"],
            # Same seed at every rho, so the comparison between rho levels is paired.
            random_state=seeds.int_seed(cfg["master_seed"], "model", dataset_index),
            n_jobs=1,  # parallel across datasets, not inside a forest
        )

    def fit(self, X, y):
        self.rsf.fit(X, y)
        return self

    def predict_survival_function(self, X):
        return self.rsf.predict_survival_function(X)

    def predict_risk(self, X):
        """Risk score for Harrell's C-index (higher = earlier event)."""
        return self.rsf.predict(X)

    def assert_grid_inside(self, X, grid):
        # A forest step function raises ValueError outside its range.
        lo, hi = self.predict_survival_function(X.iloc[[0]])[0].domain
        assert lo <= grid[0] and grid[-1] <= hi, f"grid [{grid[0]}, {grid[-1]}] outside forest range [{lo}, {hi}]"
        return lo, hi
