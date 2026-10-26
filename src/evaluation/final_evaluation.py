"""
Final evaluation (2026-10-24 freeze): every Python feature set refit with a
properly chosen L2 penalty, then scored once on the held-out season.

Every recorded Python experiment (backtest.py, derby_analysis.py) used one
penalty, alpha=1.0, on UNSTANDARDIZED features. That penalizes smooth
long-horizon features harder than noisy 5-match form, and it changes which
features look useful (docs/PROGRESS.md item 15). This script settles it
with the procedure the docs stated before any of these numbers existed:

  1. Standardize every feature (mean 0 / sd 1 from the training rows only).
  2. For each feature set, choose alpha from config.yaml ->
     final_evaluation.alpha_grid by walk-forward validation INSIDE the
     training seasons: each validation season is predicted by a fit on the
     seasons before it, predictions are pooled, and the lowest Brier score
     wins (ties go to the larger alpha). The holdout season is dropped
     before this step, so it can't influence the choice.
  3. Refit each set on all training seasons at its own alpha and score the
     holdout season ONCE, on the same matches as backtest.py, with paired
     bootstraps against its baseline, the no-information base rate, and R.

Decision rule, fixed beforehand: a feature "improves out-of-sample
predictions" only if its Brier 95% CI against its own baseline excludes
zero in the improving direction on the holdout (step 3). The nested
walk-forward below checks whether the direction holds on a second season;
it is not a second chance to pass.

Secondary: nested walk-forward over 2024-25 + 2025-26. Each test season's
alpha is re-chosen from validation seasons strictly before it, so no season
ever tunes its own predictions. It pools ~2x the holdout's matches and puts
R's walk-forward predictions (R has no hyperparameter) on the same footing.
Squad quality isn't in it: its first usable training season is 2023-24,
which leaves no validation season before 2025-26.

What no procedure here can fix: the holdout season was reported on by every
earlier experiment, so it isn't a pristine test of the choice of features
to try. See docs/METHODOLOGY.md, "Final evaluation".

Usage:
    python -m src.evaluation.backtest            # the recorded alpha=1.0 CSVs, compared against below
    Rscript src/models/dixon_coles_model.R       # R's holdout + walk-forward CSVs
    python -m src.evaluation.final_evaluation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.backtest import (base_rate_forecast, common_match_ids, compare_feature_sets,
                                     experiment_1_feature_sets, experiment_2_feature_sets, paired_bootstrap,
                                     per_match_brier, run_poisson_backtest)
from src.evaluation.derby_analysis import walk_forward_poisson
from src.evaluation.evaluate_predictions import compute_metrics, per_match_log_loss
from src.models.poisson_model import PoissonMatchModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

KEYS = ["match_date", "home_team", "away_team"]
METRIC_KEYS = ["n_evaluated", "winner_accuracy", "exact_score_accuracy",
               "mae_home_goals", "mae_away_goals", "brier_score", "log_loss"]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def select_alpha(df: pd.DataFrame, home_features: list, away_features: list, test_season: int,
                  validation_seasons: list, alpha_grid: list, max_goals: int,
                  standardize: bool = True) -> tuple:
    """
    Chooses the L2 penalty for one feature set without looking at
    `test_season` or anything after it: every season >= test_season is
    dropped first. Each usable validation season (one with an earlier
    season left to train on) is predicted by a fit on the seasons before
    it, and the alpha with the lowest pooled Brier wins; ties go to the
    larger alpha. Returns (alpha, table of every alpha's validation scores).
    """
    df = df[df["season"] < test_season].dropna(subset=home_features + away_features)
    seasons = set(df["season"])
    folds = [s for s in validation_seasons if s in seasons and any(t < s for t in seasons)]
    if not folds:
        raise ValueError(f"No validation season before {test_season} has earlier seasons to train on "
                         f"(seasons with data: {sorted(seasons)}).")

    rows = []
    for alpha in alpha_grid:
        preds = walk_forward_poisson(df, home_features, away_features, folds, max_goals, alpha=alpha,
                                     standardize=standardize, verbose=False)
        rows.append({
            "alpha": alpha,
            "validation_seasons": "+".join(str(s) for s in folds),
            "n_validation": len(preds),
            "brier_score": float(per_match_brier(preds).mean()),
            "log_loss": float(per_match_log_loss(preds).mean()),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(["brier_score", "alpha"], ascending=[True, False]).iloc[0]["alpha"]
    table["chosen"] = table["alpha"] == best
    return float(best), table


def choose_alphas(df: pd.DataFrame, feature_sets: dict, test_season: int, final_cfg: dict,
                   max_goals: int, analysis: str) -> tuple:
    """select_alpha for every feature set; returns ({name: alpha}, one combined table)."""
    alphas, tables = {}, []
    for name, (home_features, away_features) in feature_sets.items():
        alphas[name], table = select_alpha(df, home_features, away_features, test_season,
                                           final_cfg["validation_seasons"], final_cfg["alpha_grid"],
                                           max_goals, standardize=final_cfg["standardize"])
        tables.append(table.assign(analysis=analysis, test_season=test_season, feature_set=name))
    return alphas, pd.concat(tables, ignore_index=True)


def nested_walk_forward(df: pd.DataFrame, feature_sets: dict, test_seasons: list, final_cfg: dict,
                         max_goals: int) -> tuple:
    """
    For each test season: choose each set's alpha on validation seasons
    before it (select_alpha), fit on every season before it, and predict
    the test season on the matches every set can score. Returns pooled
    predictions per set (with a `season` column) and the alpha table.
    """
    preds = {name: [] for name in feature_sets}
    tables = []
    for season in test_seasons:
        alphas, table = choose_alphas(df, feature_sets, season, final_cfg, max_goals, "nested_walk_forward")
        tables.append(table)
        upto = df[df["season"] <= season]
        eval_ids = common_match_ids(upto, feature_sets, season)
        for name, (home_features, away_features) in feature_sets.items():
            fold = run_poisson_backtest(upto, home_features, away_features, season, max_goals,
                                        eval_match_ids=eval_ids, alpha=alphas[name],
                                        standardize=final_cfg["standardize"], verbose=False)
            preds[name].append(fold.assign(season=season))
        print(f"  test season {season}: {len(eval_ids)} matches; alphas "
              + ", ".join(f"{n}={a:g}" for n, a in alphas.items()))
    return ({name: pd.concat(frames, ignore_index=True) for name, frames in preds.items()},
            pd.concat(tables, ignore_index=True))


def align_on_matches(a: pd.DataFrame, b: pd.DataFrame) -> tuple:
    """Restricts two prediction frames to the matches they share, in the same
    order and with the same date format, so they can go into paired_bootstrap
    (e.g. R, which skips promoted teams, against a Python model)."""
    def keyed(df):
        out = df.copy()
        out["match_date"] = pd.to_datetime(out["match_date"]).dt.strftime("%Y-%m-%d")
        return out.drop_duplicates(KEYS)
    a, b = keyed(a), keyed(b)
    shared = a[KEYS].merge(b[KEYS], on=KEYS).sort_values(KEYS)
    return (shared.merge(a, on=KEYS, how="left").reset_index(drop=True),
            shared.merge(b, on=KEYS, how="left").reset_index(drop=True))


def base_rate_by_season(template: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """base_rate_forecast for pooled walk-forward predictions: each season's
    rows get the frequencies of the seasons before it only."""
    return pd.concat([base_rate_forecast(template[template["season"] == s], matches[matches["season"] < s])
                      for s in sorted(template["season"].unique())], ignore_index=True)


def record(rows: list, analysis: str, baseline: str, challenger: str, base_df: pd.DataFrame,
           chal_df: pd.DataFrame, show: bool = True, level: float = 0.95) -> None:
    """Paired bootstrap of challenger - baseline, appended to `rows` (one row
    per metric) and optionally printed."""
    boot = paired_bootstrap(base_df, chal_df, level=level)
    for metric in ("brier_diff", "log_loss_diff", "accuracy_diff"):
        b = boot[metric]
        rows.append({"analysis": analysis, "challenger": challenger, "baseline": baseline, "n": boot["n"],
                     "metric": metric, "level": level, "mean": b["mean"], "ci_low": b["ci95"][0],
                     "ci_high": b["ci95"][1], "excludes_zero": b["excludes_zero"]})
    if show:
        b, l, acc = boot["brier_diff"], boot["log_loss_diff"], boot["accuracy_diff"]
        flag = lambda x: "*" if x["excludes_zero"] else " "
        print(f"    {challenger + ' vs. ' + baseline:<66} n={boot['n']:<4}"
              f" Brier {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]{flag(b)}"
              f"  log loss {l['mean']:+.4f} [{l['ci95'][0]:+.4f}, {l['ci95'][1]:+.4f}]{flag(l)}"
              f"  acc {100 * acc['mean']:+.1f} pts [{100 * acc['ci95'][0]:+.1f}, {100 * acc['ci95'][1]:+.1f}]{flag(acc)}")


def metrics_row(analysis: str, model: str, preds: pd.DataFrame, alpha=None) -> dict:
    return {"analysis": analysis, "model": model, "alpha": alpha, **compute_metrics(preds)}


def print_metrics(rows: list) -> None:
    print(f"    {'model':<40}{'alpha':>7}" + "".join(f"{k:>15}" for k in METRIC_KEYS))
    for r in rows:
        alpha = "" if r["alpha"] is None or pd.isna(r["alpha"]) else f"{r['alpha']:g}"
        print(f"    {r['model']:<40}{alpha:>7}" + "".join(f"{r[k]:>15}" for k in METRIC_KEYS))


def print_coefficients(name: str, df: pd.DataFrame, home_features: list, away_features: list,
                        holdout_season: int, alpha: float, max_goals: int) -> None:
    """Explainability: each standardized coefficient as a rate ratio -- the
    multiplier on expected goals for a 1-sd higher feature value."""
    train = df[df["season"] != holdout_season].dropna(subset=home_features + away_features)
    model = PoissonMatchModel(home_features, away_features, max_goals=max_goals, alpha=alpha,
                              standardize=True).fit(train)
    for side, features, pipe in (("home goals", home_features, model.home_model),
                                 ("away goals", away_features, model.away_model)):
        reg = pipe[-1]
        terms = ", ".join(f"{f} x{np.exp(c):.3f}" for f, c in zip(features, reg.coef_))
        print(f"    {name:<34} {side}: baseline rate {np.exp(reg.intercept_):.2f}; per +1 sd: {terms}")


def main():
    cfg = load_config()
    final_cfg = cfg["final_evaluation"]
    holdout_season = cfg["evaluation"]["holdout_season"]
    windows = cfg["features"]["form_windows"]
    max_goals = cfg["model"]["max_goals_simulated"]
    match_feat_path = PROJECT_ROOT / cfg["paths"]["gold_match_features"] / "match_features.csv"
    pred_dir = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"]
    out_dir = PROJECT_ROOT / cfg["paths"]["gold_final_evaluation"]
    r_holdout_path = pred_dir / "dixon_coles_holdout_backtest.csv"
    r_wf_path = pred_dir / "dixon_coles_walkforward.csv"

    for path, fix in [(match_feat_path, "python -m src.features.build_match_features"),
                      (r_holdout_path, "Rscript src/models/dixon_coles_model.R"),
                      (r_wf_path, "Rscript src/models/dixon_coles_model.R")]:
        if not path.exists():
            print(f"Missing {path.relative_to(PROJECT_ROOT)}. Run: {fix}")
            return

    df = pd.read_csv(match_feat_path, parse_dates=["date"])
    exp1, exp2 = experiment_1_feature_sets(windows), experiment_2_feature_sets(windows)
    squad_home, squad_away = exp2["squad_quality"]
    squad_df = df.dropna(subset=squad_home + squad_away)
    train_matches = df[df["season"] != holdout_season]
    r_holdout, r_wf = pd.read_csv(r_holdout_path), pd.read_csv(r_wf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons, metrics = [], []

    # ---- 1. Alpha selection inside the training seasons ---------------------
    print(f"=== 1. Choosing each feature set's alpha on training seasons only "
          f"(standardize={final_cfg['standardize']}, grid {final_cfg['alpha_grid']}) ===")
    alphas1, table1 = choose_alphas(df, exp1, holdout_season, final_cfg, max_goals, "holdout")
    alphas2, table2 = choose_alphas(squad_df, exp2, holdout_season, final_cfg, max_goals, "holdout")
    alphas = {**alphas1, **alphas2}
    selection = pd.concat([table1, table2], ignore_index=True)
    for name, alpha in alphas.items():
        t = selection[(selection["feature_set"] == name) & (selection["analysis"] == "holdout")]
        chosen = t[t["chosen"]].iloc[0]
        at_one = t[t["alpha"] == 1.0].iloc[0]
        print(f"    {name:<34} alpha={alpha:<6g} validation {chosen['validation_seasons']} "
              f"(n={chosen['n_validation']}): Brier {chosen['brier_score']:.4f} "
              f"(alpha=1: {at_one['brier_score']:.4f})")

    # ---- 2. The holdout, scored once ------------------------------------------
    print(f"\n=== 2. PRIMARY: season {holdout_season}, scored once with the chosen alphas ===")
    print("--- Experiment 1 (form / team-level) ---")
    final1 = compare_feature_sets(df, exp1, holdout_season, max_goals, out_dir, alpha=alphas1,
                                  standardize=final_cfg["standardize"], file_prefix="holdout_")
    print("\n--- Experiment 2 (squad quality; every arm fit on the rows where it exists) ---")
    final2 = compare_feature_sets(squad_df, exp2, holdout_season, max_goals, out_dir, alpha=alphas2,
                                  standardize=final_cfg["standardize"], file_prefix="holdout_")
    final = {**final1, **final2}

    base1 = base_rate_forecast(final1["rolling_form_baseline"], train_matches)
    base2 = base_rate_forecast(final2["rolling_form_baseline_squad_rows"], train_matches)
    r_base = base_rate_forecast(r_holdout, train_matches)
    base1.to_csv(out_dir / "holdout_base_rate.csv", index=False)

    print("\n  Summary (each Python row: standardized features, its own validation-chosen alpha):")
    metric_rows = [metrics_row("holdout", "base_rate (experiment 1 rows)", base1)]
    metric_rows += [metrics_row("holdout", name, final1[name], alphas[name]) for name in exp1]
    metric_rows += [metrics_row("holdout", "base_rate (experiment 2 rows)", base2)]
    metric_rows += [metrics_row("holdout", name, final2[name], alphas[name]) for name in exp2]
    metric_rows += [metrics_row("holdout", "base_rate (R rows)", r_base),
                    metrics_row("holdout", "r_dixon_coles", r_holdout)]
    print_metrics(metric_rows)
    metrics += metric_rows

    print("\n  Paired bootstraps (challenger - baseline; * = 95% CI excludes 0). "
          "Brier/log loss: negative = better.")
    print("  Each factor vs. its own baseline (the research question):")
    for name in list(exp1)[1:]:
        record(comparisons, "holdout_tuned", "rolling_form_baseline", name,
               final1["rolling_form_baseline"], final1[name])
    for name in list(exp2)[1:]:
        record(comparisons, "holdout_tuned", "rolling_form_baseline_squad_rows", name,
               final2["rolling_form_baseline_squad_rows"], final2[name])
    record(comparisons, "holdout_tuned", "prev_season_team_goals_control", "squad_quality",
           final2["prev_season_team_goals_control"], final2["squad_quality"])
    # Seven challengers were tested against the same holdout, so a single 95%
    # interval excluding zero is weaker evidence than it looks alone.
    family = [(r["baseline"], r["challenger"]) for r in comparisons
              if r["analysis"] == "holdout_tuned" and r["metric"] == "brier_diff"]
    level = 1 - 0.05 / len(family)
    print(f"  Bonferroni-corrected ({100 * level:.2f}%) intervals for the {len(family)} comparisons above:")
    for baseline, challenger in family:
        record(comparisons, "holdout_tuned_bonferroni", baseline, challenger, final[baseline], final[challenger],
               level=level)

    print("  Every model vs. the no-information base rate:")
    for name in exp1:
        record(comparisons, "holdout_vs_base_rate", "base_rate", name, base1, final1[name])
    for name in exp2:
        record(comparisons, "holdout_vs_base_rate", "base_rate", name, base2, final2[name])
    record(comparisons, "holdout_vs_base_rate", "base_rate", "r_dixon_coles", r_base, r_holdout)

    print("  R Dixon-Coles vs. each Python model, on the matches both scored:")
    for name, preds in final.items():
        py, r = align_on_matches(preds, r_holdout)
        record(comparisons, "holdout_r_vs_python", name, "r_dixon_coles", py, r)

    # The same comparisons with the recorded alpha=1.0 CSVs, to show what
    # choosing the penalty properly changed. Same matches, same order.
    recorded = {name: pred_dir / f"poisson_holdout_backtest_{name}.csv" for name in final}
    if all(p.exists() for p in recorded.values()):
        rec = {name: pd.read_csv(p, parse_dates=["match_date"]) for name, p in recorded.items()}
        print("  Recorded experiments (unstandardized, alpha=1.0 for every set; from backtest.py):")
        for name in list(exp1)[1:]:
            record(comparisons, "holdout_recorded_alpha1", "rolling_form_baseline", name,
                   rec["rolling_form_baseline"], rec[name])
        for name in list(exp2)[1:]:
            record(comparisons, "holdout_recorded_alpha1", "rolling_form_baseline_squad_rows", name,
                   rec["rolling_form_baseline_squad_rows"], rec[name])
        record(comparisons, "holdout_recorded_alpha1", "prev_season_team_goals_control", "squad_quality",
               rec["prev_season_team_goals_control"], rec["squad_quality"])
        record(comparisons, "holdout_vs_base_rate", "base_rate", "rolling_form_baseline (alpha=1.0)",
               base1, rec["rolling_form_baseline"])
        print("  What tuning did to each set (tuned - recorded alpha=1.0):")
        for name in final:
            record(comparisons, "holdout_tuned_vs_recorded", f"{name} (alpha=1.0)", f"{name} (tuned)",
                   rec[name], final[name])
    else:
        print("  [skip] recorded alpha=1.0 CSVs not found -- run `python -m src.evaluation.backtest` first.")

    print("\n  Fitted effects (standardized; multiplier on expected goals per +1 sd of each feature):")
    for name in ("rolling_form_baseline", "team_level_venue_split"):
        print_coefficients(name, df, *exp1[name], holdout_season, alphas[name], max_goals)
    print_coefficients("squad_quality", squad_df, *exp2["squad_quality"], holdout_season,
                       alphas["squad_quality"], max_goals)

    # ---- 3. Secondary: nested walk-forward ------------------------------------
    test_seasons = final_cfg["nested_walk_forward_test_seasons"]
    print(f"\n=== 3. SECONDARY: nested walk-forward, test seasons {test_seasons} "
          f"(experiment 1 sets; alpha re-chosen before each season) ===")
    wf, wf_table = nested_walk_forward(df, exp1, test_seasons, final_cfg, max_goals)
    selection = pd.concat([selection, wf_table], ignore_index=True)
    same = all(wf[name][wf[name]["season"] == holdout_season].drop(columns="season").reset_index(drop=True)
               .equals(final1[name].reset_index(drop=True)) for name in exp1)
    print(f"  {holdout_season} fold reproduces section 2's holdout predictions exactly: {same}")
    wf_base = base_rate_by_season(wf["rolling_form_baseline"], df)
    r_wf = r_wf[r_wf["season"].isin(test_seasons)]
    r_wf_base = base_rate_by_season(r_wf, df)
    pd.concat([p.assign(model=n) for n, p in {**wf, "base_rate": wf_base}.items()], ignore_index=True) \
        .to_csv(out_dir / "nested_walkforward_predictions.csv", index=False)

    wf_rows = [metrics_row("nested_walk_forward", "base_rate (python rows)", wf_base)]
    wf_rows += [metrics_row("nested_walk_forward", name, preds) for name, preds in wf.items()]
    wf_rows += [metrics_row("nested_walk_forward", "base_rate (R rows)", r_wf_base),
                metrics_row("nested_walk_forward", "r_dixon_coles", r_wf)]
    print_metrics(wf_rows)
    metrics += wf_rows

    print("  Each factor vs. the baseline, pooled:")
    for name in list(exp1)[1:]:
        record(comparisons, "nested_wf", "rolling_form_baseline", name, wf["rolling_form_baseline"], wf[name])
    # The season before the holdout is an independent replication: its alpha
    # came from 2023-24 alone, and none of its predictions are in section 2.
    for season in test_seasons:
        print(f"  Each factor vs. the baseline, season {season} only:")
        in_season = {name: preds[preds["season"] == season] for name, preds in wf.items()}
        for name in list(exp1)[1:]:
            record(comparisons, f"nested_wf_season_{season}", "rolling_form_baseline", name,
                   in_season["rolling_form_baseline"], in_season[name])
    print("  vs. the base rate, pooled:")
    for name in exp1:
        record(comparisons, "nested_wf_vs_base_rate", "base_rate", name, wf_base, wf[name])
    record(comparisons, "nested_wf_vs_base_rate", "base_rate", "r_dixon_coles", r_wf_base, r_wf)
    print("  R Dixon-Coles vs. each Python model, on the matches both scored:")
    for name, preds in wf.items():
        py, r = align_on_matches(preds, r_wf)
        record(comparisons, "nested_wf_r_vs_python", name, "r_dixon_coles", py, r)

    selection.round(6).to_csv(out_dir / "alpha_selection.csv", index=False)
    pd.DataFrame(metrics).to_csv(out_dir / "final_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "final_comparisons.csv", index=False)
    print(f"\nSaved alpha selection, metrics, comparisons and predictions -> {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
