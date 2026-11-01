"""Check the simulator against the design.

1. Censored fraction matches the config (large sample and every dataset).
2. Correlation between visits matches each rho (large sample).
3. Event times are identical across correlation levels for the same dataset index.
4. Kaplan-Meier of a large sample matches the population survival curve.
5. A Cox model on a large sample puts the effect on the driver and about zero elsewhere.
6. The same seed gives identical data.
7. (Scenario B only) Same patients as scenario A: identical times, events and driver values.

Run from the repository root:  python scripts/check_simulator.py [--scenario B]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

from rtds.cli import parse_scenario
from rtds.censoring import censored_fraction, marginal_survival
from rtds.config import load_config
from rtds.grid import time_grid
from rtds.inputs import FlatVisitColumns
from rtds.simulate import censor_rate_from_config, reference_sample, simulate_dataset



def main():
    scenario = parse_scenario(__doc__)
    cfg = load_config()
    sim = cfg["simulation"]
    rhos = sim["rho_levels"]
    n_datasets = cfg["runs"]["n_datasets"]
    driver = cfg["scenarios"][scenario]
    builder = FlatVisitColumns()
    large = {rho: reference_sample(cfg, rho, scenario) for rho in rhos}
    print(f"large sample: {sim['n_reference']} patients; scenario {scenario}, driver = visit_{driver}")

    # 1. Censoring
    rate = censor_rate_from_config(cfg)
    target = sim["target_censored_fraction"]
    print(f"\n[1] solved censoring rate = {rate:.6f}; censored fraction by quadrature = "
          f"{censored_fraction(rate, sim['h0'], sim['b']):.6f} (target {target})")
    large_cens = 1 - large[rhos[0]].event.mean()
    se = np.sqrt(target * (1 - target) / sim["n_reference"])
    print(f"    large sample: censored = {large_cens:.4f}, events = {1 - large_cens:.4f} "
          f"(binomial SE {se:.4f})")
    assert abs(large_cens - target) < 4 * se
    per_dataset = [1 - simulate_dataset(cfg, i, rhos[0], scenario).train.event.mean() for i in range(n_datasets)]
    print(f"    {n_datasets} training sets: censored mean = {np.mean(per_dataset):.4f}, "
          f"min = {np.min(per_dataset):.4f}, max = {np.max(per_dataset):.4f}")

    # 2. Correlation
    print("\n[2] off-diagonal correlations between visits (large sample)")
    for rho in rhos:
        c = np.corrcoef(large[rho].visits, rowvar=False)
        off = c[np.triu_indices_from(c, k=1)]
        print(f"    rho = {rho}: corr(1,2) = {off[0]:.4f}, corr(1,3) = {off[1]:.4f}, corr(2,3) = {off[2]:.4f}; "
              f"marginal sd = {np.round(large[rho].visits.std(axis=0), 4)}")
        assert np.all(np.abs(off - rho) < 0.01)

    # 3. Event times identical across rho
    same = True
    for i in range(n_datasets):
        ds = [simulate_dataset(cfg, i, rho, scenario) for rho in rhos]
        for d in ds[1:]:
            for part in ("train", "test"):
                a, b = getattr(ds[0], part), getattr(d, part)
                same &= np.array_equal(a.time, b.time) and np.array_equal(a.event, b.event)
                same &= np.array_equal(a.visits[:, driver - 1], b.visits[:, driver - 1])
    print(f"\n[3] {n_datasets} datasets: times, events and driver visit bit-identical across rho {list(rhos)}: {same}")
    assert same

    # 4. Kaplan-Meier vs the population survival curve, on the time grid
    grid = time_grid(cfg)
    ref = large[rhos[0]]
    km_t, km_s = kaplan_meier_estimator(ref.event, ref.time)
    km_on_grid = km_s[np.searchsorted(km_t, grid, side="right") - 1]
    truth = marginal_survival(grid, sim["h0"], sim["b"])
    diff = np.abs(km_on_grid - truth)
    print(f"\n[4] time grid: {len(grid)} points from {grid[0]:.4f} to {grid[-1]:.4f}")
    print(f"    max |KM - true S(t)| on grid = {diff.max():.4f} at t = {grid[diff.argmax()]:.3f} "
          f"(true S there = {truth[diff.argmax()]:.4f})")
    for t_idx in (0, len(grid) // 2, len(grid) - 1):
        print(f"    t = {grid[t_idx]:7.3f}: KM = {km_on_grid[t_idx]:.4f}, true = {truth[t_idx]:.4f}")
    assert diff.max() < 0.01

    # 5. Cox model on the large sample, at every rho
    print(f"\n[5] Cox coefficients on the large sample (true: visit_{driver} = ln 2 = {np.log(2):.4f}, others 0)")
    for rho in rhos:
        X = builder.build(large[rho].visits)
        cox = CoxPHSurvivalAnalysis(ties="breslow").fit(X, large[rho].y)
        coef = dict(zip(X.columns, cox.coef_))
        print(f"    rho = {rho}: " + ", ".join(f"{k} = {v:+.4f}" for k, v in coef.items()))
        for k, v in coef.items():
            expected = np.log(2) if k == f"visit_{driver}" else 0.0
            assert abs(v - expected) < 0.05

    # 6. Reproducibility
    a, b = simulate_dataset(cfg, 3, rhos[1], scenario), simulate_dataset(cfg, 3, rhos[1], scenario)
    c = simulate_dataset(cfg, 4, rhos[1], scenario)
    identical = all(np.array_equal(getattr(a.train, f), getattr(b.train, f)) for f in ("visits", "time", "event"))
    differs = not np.array_equal(a.train.time, c.train.time)
    print(f"\n[6] dataset 3 simulated twice identical: {identical}; dataset 4 differs from dataset 3: {differs}")
    assert identical and differs

    # 7. Same patients as scenario A
    if scenario != "A":
        driver_a = cfg["scenarios"]["A"]
        same = True
        for i in range(n_datasets):
            for rho in rhos:
                a, b = simulate_dataset(cfg, i, rho, "A"), simulate_dataset(cfg, i, rho, scenario)
                for part in ("train", "test"):
                    pa, pb = getattr(a, part), getattr(b, part)
                    same &= np.array_equal(pa.time, pb.time) and np.array_equal(pa.event, pb.event)
                    same &= np.array_equal(pa.visits[:, driver_a - 1], pb.visits[:, driver - 1])
        print(f"\n[7] {n_datasets} datasets x {len(rhos)} rho: times, events and driver values bit-identical "
              f"to scenario A (visit_{driver} here = visit_{driver_a} there): {same}")
        assert same

    print("\nAll simulator checks passed.")


if __name__ == "__main__":
    main()
