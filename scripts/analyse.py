"""Final analysis: every summary number, test and result figure of the report, from the saved tables.

The forest's scenario B numbers are scenario A's with the visits relabelled (evidence:
results/scenario_B/mirror_check.csv).

Writes results/final/.
Run from the repository root:  python scripts/analyse.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use src/ without installing it

import pandas as pd

from rtds import seeds
from rtds.config import load_config
from rtds.examples import run_examples
from rtds.figures import example_curves, hit_rate_chart, per_visit_panels
from rtds.grid import time_grid
from rtds.mirror import visit_map
from rtds.results import RESULTS_DIR, scenario_dir
from rtds.split_check import train_test_check
from rtds.stats import bootstrap_ci, paired_wilcoxon, per_dataset

SCORES = ["hit", "credit_share"]
METRICS = SCORES + ["share_visit_1", "share_visit_2", "share_visit_3", "drop_visit_1", "drop_visit_2",
                    "drop_visit_3", "ibs_visit_1", "ibs_visit_2", "ibs_visit_3",
                    "heldout_c_index", "true_risk_c_index"]


def load_per_dataset(scenario, model):
    """One row per (rho, dataset): scores averaged over patients, Run 2 drops, C-indexes."""
    base = scenario_dir(scenario, model)
    t1 = pd.read_csv(base / "table1_main.csv")
    t2 = pd.read_csv(base / "table2_main.csv")
    ds = pd.read_csv(base / "datasets_main.csv")
    log = pd.read_csv(base / "log_main.csv")
    drops = (t2.pivot_table(index=["rho", "dataset_seed"], columns="visit", values="c_index_drop")
             .rename(columns=lambda k: f"drop_visit_{k}").reset_index())
    brier = (t2.pivot_table(index=["rho", "dataset_seed"], columns="visit", values="ibs_increase")
             .rename(columns=lambda k: f"ibs_visit_{k}").reset_index())
    drops = drops.merge(brier, on=["rho", "dataset_seed"])
    out = (per_dataset(t1).merge(drops, on=["rho", "dataset_seed"])
           .merge(ds[["rho", "dataset_seed", "heldout_c_index", "true_risk_c_index"]], on=["rho", "dataset_seed"]))
    counts = {rho: (int((t1["rho"] == rho).sum()), int(((log["rho"] == rho) & (log["n_tied"] > 1)).sum()),
                    int(((log["rho"] == rho) & log["share_is_nan"]).sum())) for rho in out["rho"].unique()}
    return out, counts


def relabel_from_a(cfg, per_ds_a, scenario):
    """Scenario A's per-dataset values with the visits renamed to `scenario`'s labels."""
    out = per_ds_a.copy()
    for k, ka in visit_map(cfg, scenario).items():
        for kind in ("share", "drop", "ibs"):
            out[f"{kind}_visit_{k}"] = per_ds_a[f"{kind}_visit_{ka}"]
    return out


def summarise(cfg, per, counts, settings, gen):
    """One row per (model, scenario, rho) with bootstrap intervals over datasets."""
    a = cfg["analysis"]
    rows = []
    for (model, scen), p in per.items():
        for rho in cfg["simulation"]["rho_levels"]:
            g = p[p["rho"] == rho]
            n_pat, n_ties, n_nan = counts[(model, scen)][rho]
            base = model.removesuffix("_tuned")
            row = {"model": base, "settings": settings, "scenario": scen, "rho": rho, "driver": cfg["scenarios"][scen],
                   "n_datasets": len(g), "n_patients": n_pat, "n_ties": n_ties, "n_nan_shares": n_nan,
                   "source": f"relabelled from {model} A" if base == "forest" and scen == "B" else "main runs"}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(g[m], gen, a["n_bootstrap"], a["ci_level"])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_tests_for(per, forest, lstm, rhos):
    """The planned paired Wilcoxon tests for one pair of models."""
    tests = []

    def test(family, label, metric, rho, x, y):
        n, ma, mb, med, stat, pv = paired_wilcoxon(x.set_index("dataset_seed")[metric], y.set_index("dataset_seed")[metric])
        tests.append({"family": family, "comparison": label, "metric": metric, "rho": rho, "n_datasets": n,
                      "mean_first": ma, "mean_second": mb, "median_diff_second_minus_first": med,
                      "wilcoxon_stat": stat, "p_value": pv})

    for model, scen in ((forest, "A"), (lstm, "A"), (lstm, "B")):  # forest B = forest A, not retested
        p = per[(model, scen)]
        for r1, r2 in itertools.combinations(rhos, 2):
            for m in SCORES:
                test("between rho levels", f"{model} {scen}: rho {r1:g} vs {r2:g}", m, f"{r1:g} vs {r2:g}",
                     p[p["rho"] == r1], p[p["rho"] == r2])
    for scen in ("A", "B"):
        for rho in rhos:
            for m in SCORES:
                f, l = per[(forest, scen)], per[(lstm, scen)]
                test("forest vs LSTM", f"scenario {scen}: {forest} vs {lstm}", m, rho, f[f["rho"] == rho], l[l["rho"] == rho])
    for rho in rhos:
        for m in SCORES:
            la, lb = per[(lstm, "A")], per[(lstm, "B")]
            test("scenario A vs B", f"{lstm}: scenario A vs B", m, rho, la[la["rho"] == rho], lb[lb["rho"] == rho])
    return pd.DataFrame(tests)


def seed_check_rows(cfg):
    """One row per seed check (fixed and tuned LSTM, scenarios A and B)."""
    rows = []
    for version in ("lstm", "lstm_tuned"):
        for scen in cfg["seed_stability"]["scenarios"]:
            d = scenario_dir(scen, version) / "seed_check"
            if not (d / "models.csv").exists():
                continue
            models, ds = pd.read_csv(d / "models.csv"), pd.read_csv(d / "datasets.csv")
            driver = cfg["scenarios"][scen]
            fav = models["run3_favoured"].value_counts()
            rows.append({"model": "lstm", "settings": "tuned" if version.endswith("_tuned") else "fixed",
                         "scenario": scen, "driver": driver, "rho": cfg["seed_stability"]["rho"],
                         "n_datasets": len(ds), "n_models": len(models),
                         "datasets_where_run3_favourite_flips": int(ds["run3_flips"].sum()),
                         "patient_top_visit_agreement": ds["patient_top_visit_agreement"].mean(),
                         "max_c_index_spread": (ds["c_index_max"] - ds["c_index_min"]).max(),
                         **{f"models_run3_favours_visit_{k}": int(fav.get(k, 0)) for k in (1, 2, 3)},
                         "models_run2_brier_favours_driver": int((models["run2_ibs_favoured"] == driver).sum()),
                         "models_run3_matches_run2_brier": int((models["run3_favoured"] == models["run2_ibs_favoured"]).sum()),
                         "mean_hit_rate": models["hit_rate"].mean(), "mean_driver_share": models["driver_share"].mean()})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    a = cfg["analysis"]
    rhos = list(cfg["simulation"]["rho_levels"])
    final = RESULTS_DIR / "final"
    figs = final / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    # One bootstrap stream per setting, so the tuned results cannot change the fixed numbers.
    per, counts, summaries, test_sets = {}, {}, [], []
    for settings, suffix, stream in (("fixed", "", 1), ("tuned", "_tuned", 2)):
        forest, lstm = "forest" + suffix, "lstm" + suffix
        per[(forest, "A")], counts[(forest, "A")] = load_per_dataset("A", forest)
        per[(forest, "B")], counts[(forest, "B")] = relabel_from_a(cfg, per[(forest, "A")], "B"), counts[(forest, "A")]
        for scen in ("A", "B"):
            per[(lstm, scen)], counts[(lstm, scen)] = load_per_dataset(scen, lstm)
        keys = [(forest, "A"), (forest, "B"), (lstm, "A"), (lstm, "B")]
        summaries.append(summarise(cfg, {k: per[k] for k in keys}, counts, settings,
                                   seeds.rng(cfg["master_seed"], "bootstrap", stream)))
        test_sets.append(paired_tests_for(per, forest, lstm, rhos))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(final / "summary.csv", index=False)
    tests, tests_tuned = test_sets
    tests.to_csv(final / "tests.csv", index=False)
    tests_tuned.to_csv(final / "tests_tuned.csv", index=False)

    # Run 1
    run1 = []
    for scen in ("A", "B"):
        t = pd.read_csv(scenario_dir(scen) / "table1_run1.csv")
        for rho, g in t.groupby("rho"):
            run1.append({"scenario": scen, "rho": rho, "n_patients": len(g), "hit": g["hit"].mean(),
                         "credit_share": g["credit_share"].mean(), "lowest_share": g["credit_share"].min()})
    run1 = pd.DataFrame(run1)
    run1.to_csv(final / "run1_summary.csv", index=False)

    # Figures
    hit_rate_chart(summary, figs / "fig_hit_rate.pdf")
    fixed = summary[summary["settings"] == "fixed"]
    s = fixed[(fixed["model"] == "lstm") & (fixed["scenario"] == "A")].reset_index(drop=True)
    per_visit_panels(s, figs / "fig_per_visit_lstm_A.pdf", cfg["scenarios"]["A"])

    # Report text
    pct = int(a["ci_level"] * 100)
    say(f"Final analysis. Means over datasets with {pct}% bootstrap intervals ({a['n_bootstrap']} resamples of datasets).")
    say("Forest scenario B = forest scenario A relabelled (mirror shown in Stages B4 and B5).")
    for _, r in summary.iterrows():
        f = lambda m, d=3: f"{r[m + '_mean']:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]"
        say(f"\n{r['model']} ({r['settings']} settings), scenario {r['scenario']} (driver visit_{r['driver']}), rho = {r['rho']:g}: "
            f"{r['n_datasets']} datasets, {r['n_patients']} patients, ties = {r['n_ties']}, NaN shares = {r['n_nan_shares']}")
        say(f"  hit rate {f('hit')}   credit share {f('credit_share')}")
        say(f"  share per visit: 1 = {f('share_visit_1')}, 2 = {f('share_visit_2')}, 3 = {f('share_visit_3')}")
        say(f"  Run 2 drop: 1 = {f('drop_visit_1', 4)}, 2 = {f('drop_visit_2', 4)}, 3 = {f('drop_visit_3', 4)}")
        say(f"  Run 2 Brier increase: 1 = {f('ibs_visit_1', 4)}, 2 = {f('ibs_visit_2', 4)}, 3 = {f('ibs_visit_3', 4)}")
        say(f"  C-index model {f('heldout_c_index', 4)}, true risk {f('true_risk_c_index', 4)}")
    for name, t in (("fixed settings, the planned analysis", tests), ("tuned settings, robustness check", tests_tuned)):
        say(f"\nPaired Wilcoxon signed-rank tests, {name}: {len(t)} comparisons "
            f"({', '.join(f'{k}: {v}' for k, v in t['family'].value_counts(sort=False).items())}); "
            f"p < 0.05 in {int((t['p_value'] < 0.05).sum())}.")
        say(t.drop(columns=["family"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    say("\nRun 1 (true survival curve):")
    say(run1.to_string(index=False, formatters={"rho": "{:g}".format, "hit": "{:.4f}".format,
                                               "credit_share": "{:.9f}".format, "lowest_share": "{:.9f}".format}))
    # Seed checks
    checks = seed_check_rows(cfg)
    checks.to_csv(final / "seed_checks.csv", index=False)
    for _, r in checks.iterrows():
        say(f"\nSeed check, LSTM ({r['settings']} settings), scenario {r['scenario']} (driver visit_{r['driver']}), "
            f"rho = {r['rho']:g}, {r['n_datasets']} datasets x {r['n_models'] // r['n_datasets']} models: the visit Run 3 "
            f"favours changes between seeds in {r['datasets_where_run3_favourite_flips']} of {r['n_datasets']} datasets; "
            f"two seeds name the same top visit for {r['patient_top_visit_agreement']:.3f} of patients on average; "
            f"C-index spread within a dataset at most {r['max_c_index_spread']:.4f}; Run 3 favours visit 1 / 2 / 3 in "
            f"{r['models_run3_favours_visit_1']} / {r['models_run3_favours_visit_2']} / {r['models_run3_favours_visit_3']} "
            f"of {r['n_models']} models; Run 2 (Brier) favours the driver in {r['models_run2_brier_favours_driver']}; "
            f"Run 3 and Run 2 (Brier) favour the same visit in {r['models_run3_matches_run2_brier']} of {r['n_models']}.")
    # Training vs held-out patients
    split = train_test_check(cfg)
    split.to_csv(final / "train_test_check.csv", index=False)
    say(f"\nTraining vs held-out patients ({len(split)} datasets): censored {split['censored_train'].mean():.3f} vs "
        f"{split['censored_test'].mean():.3f} on average (largest gap within a dataset "
        f"{(split['censored_train'] - split['censored_test']).abs().max():.3f}); median observed time "
        f"{split['median_time_train'].mean():.2f} vs {split['median_time_test'].mean():.2f}; two-sample KS test "
        f"p < 0.05 in {int((split['ks_p_time'] < 0.05).sum())} datasets for the time and "
        f"{int((split['ks_p_driver'] < 0.05).sum())} for the driver visit (about {0.05 * len(split):.1f} expected by chance).")
    # Worked examples
    grid = time_grid(cfg)
    items, ex_driver, examples = run_examples(cfg, grid)
    examples.to_csv(final / "examples.csv", index=False)
    example_curves(items, grid, ex_driver, figs / "fig_examples.png")
    say(f"\nWorked examples (credits reproduce Table 1; largest relative gap "
        f"{examples['max_relative_gap_to_table1'].max():.1e}):")
    say(examples.drop(columns=["max_relative_gap_to_table1"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (final / "analysis_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
