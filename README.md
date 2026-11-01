# Do explanation tools credit the right visit?

Code and results for the RTDS 2026 group project of Group 13 (Stockholm University).

We simulate patients with three correlated visits, of which one known visit drives the risk. Then we check how often SurvSHAP(t) gives that visit the most credit, for a random survival forest (RSF) and an LSTM, at correlation 0, 0.5 and 0.9, with the driver read last (scenario A) or first (scenario B).

Every number and result figure in the report can be regenerated from this repository. Every random draw comes from one master seed in `config.toml`, which also holds every setting.

## Install

Python 3.12 (scikit-survival needs a package with no prebuilt wheel for 3.13 on Windows).

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

| Path | Contents |
|---|---|
| `config.toml` | every setting: simulation, models, tool, tuning, analysis, master seed |
| `src/rtds/` | the library: simulator, models, explainer, scores, reliance, statistics, figures |
| `scripts/` | the eight scripts below, run from the repository root |
| `results/` | the saved tables of every run, and `results/final/` with the report's numbers and figures |
| `report_figures/` | LaTeX sources of the report's Figs. 1 and 2 (pipeline and data point) |

## Reproduce

The scripts in order. Times are from our laptop CPU, with several datasets in parallel (`n_jobs` in `config.toml`). Each main run saves after every dataset and skips finished datasets if restarted.

| Step | Command | What it gives in the report | Time |
|---|---|---|---|
| 1 | `python scripts/check_simulator.py` (and `--scenario B`) | simulator checks: correlation, same event times at every ρ, Kaplan-Meier vs true curve, Cox model | 1 min |
| 2 | `python scripts/run1_true_curve.py` (and `--scenario B`) | the tool on the true survival curve (driver share > 0.9999999) | 4 min each |
| 3 | `python scripts/tune.py --model forest`, then `--model lstm` | the tuned hyperparameters (Table II) | 7 and 11 min |
| 4 | `python scripts/run_main.py --scenario A` | RSF, fixed hyperparameters | 160 min |
|   | `python scripts/run_main.py --scenario A --model lstm` (and `--scenario B`) | LSTM, fixed | 5 min each |
|   | `python scripts/run_main.py --scenario A --model forest_tuned` | RSF, tuned | 185 min |
|   | `python scripts/run_main.py --scenario A --model lstm_tuned` (and `--scenario B`) | LSTM, tuned | 7 min each |
| 5 | `python scripts/check_forest_mirror.py` | the RSF gives the same results in scenarios A and B | 17 min |
| 6 | `python scripts/seed_check.py --scenario A --model lstm` (and `--scenario B`, and both with `--model lstm_tuned`) | the seed check: 5 extra training seeds on 10 datasets at ρ = 0.9 | 2 min (fixed, scenario A) |
| 7 | `python scripts/check_background_size.py` | background size on one dataset (hit rate and time per patient) | |
| 8 | `python scripts/analyse.py` | Table III, the Wilcoxon tests, the bootstrap intervals, the train/test check, the seed-check counts and the three result figures | 1 min |

Step 8 needs only the saved tables in `results/` (it refits the four example models from their seeds), so it can be run on its own right after cloning.

## Where the report's results are

- `results/final/analysis_output.txt`: every number in the Results section, in plain text.
- `results/final/summary.csv`: hit rate, credit share, per-visit credit, both reliance measures and C-indexes, with 95% bootstrap intervals over datasets.
- `results/final/tests.csv` and `tests_tuned.csv`: the 36 paired Wilcoxon tests for fixed and for tuned hyperparameters.
- `results/final/seed_checks.csv`, `train_test_check.csv`, `run1_summary.csv`, `examples.csv`.
- `results/final/figures/`: `fig_hit_rate.pdf`, `fig_per_visit_lstm_A.pdf` and `fig_examples.png`.

Results are per explained patient (Table 1 files, `table1_*.csv`) and per visit and dataset (Table 2 files, `table2_*.csv`), in `results/scenario_<A|B>/` for the fixed RSF and in a subfolder per model version otherwise (`lstm/`, `forest_tuned/`, `lstm_tuned/`).
