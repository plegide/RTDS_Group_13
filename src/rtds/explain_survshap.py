"""SurvSHAP(t) behind the Explainer interface: one patient in, one credit per visit out."""

import numpy as np
from survshap import PredictSurvSHAP, SurvivalModelExplainer


class SurvShapExplainer:
    def __init__(self, background, background_y, grid, calculation_method, aggregation_method):
        # background fills masked visits. background_y is required by survshap but does not affect credit.
        self.background = background
        self.background_y = background_y
        self.grid = np.asarray(grid, dtype=float)
        self.calculation_method = calculation_method
        self.aggregation_method = aggregation_method

    def explain_full(self, model, patient):
        """The raw survshap result table (one row per visit, one column per grid time)."""
        if len(patient) != 1:
            raise ValueError("PredictSurvSHAP.fit takes exactly one patient")
        explainer = SurvivalModelExplainer(
            model,
            self.background,
            self.background_y,
            predict_survival_function=lambda m, d: m.predict_survival_function(d),
        )
        shap = PredictSurvSHAP(
            calculation_method=self.calculation_method,
            aggregation_method=self.aggregation_method,
        )
        # Always pass the fixed grid; otherwise survshap uses the model's own times.
        shap.fit(explainer, patient, timestamps=self.grid)
        return shap.result

    def explain(self, model, patient):
        result = self.explain_full(model, patient)
        # Match by variable_name, never by row position.
        credit = result.set_index("variable_name")["aggregated_change"]
        return {col: float(credit.loc[col]) for col in self.background.columns}
