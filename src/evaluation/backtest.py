"""
Real chronological train/test backtest for the forecasting models.

Every metric produced elsewhere in this project up to now (the Poisson
model's own "quick sanity check", the R model's historical predictions) is
in-sample: the model saw the match it's being scored on during training.
This script fits the Python Poisson model on everything BEFORE the holdout
season only, then scores it on the holdout season it never trained on --
using the same metrics as src/evaluation/evaluate_predictions.py so results
are directly comparable to live prediction-log evaluation.

The R Dixon-Coles model's held-out backtest lives in
src/models/dixon_coles_model.R (same holdout season, same output schema) --
run that separately (`Rscript src/models/dixon_coles_model.R`) since this
project keeps the R model's own tooling in R.

Usage:
    python -m src.evaluation.backtest
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.evaluate_predictions import compute_metrics, per_match_log_loss
from src.models.poisson_model import (PoissonMatchModel, build_feature_lists, build_prev_season_control_feature_lists,
                                      build_sos_feature_lists, build_squad_feature_lists,
                                      build_team_level_feature_lists)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_poisson_backtest(df: pd.DataFrame, home_features: list, away_features: list,
                          holdout_season: int, max_goals: int, eval_match_ids: set = None,
                          alpha: float = 1.0) -> pd.DataFrame:
    """
    eval_match_ids, if given, restricts the SCORED test rows to this exact
    set of match_ids -- used to put multiple feature sets on an identical
    held-out sample even though each drops a slightly different set of rows
    to NaN (a fair like-for-like comparison needs the same test matches,
    not just the same holdout season). Training still uses each feature
    set's own full available training rows.
    """
    feature_cols = home_features + away_features
    df = df.dropna(subset=feature_cols)

    train_df = df[df["season"] != holdout_season]
    test_df = df[df["season"] == holdout_season]
    if eval_match_ids is not None:
        test_df = test_df[test_df["match_id"].isin(eval_match_ids)]
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Empty train ({len(train_df)}) or test ({len(test_df)}) split for "
            f"holdout_season={holdout_season}. Check config.yaml -> evaluation.holdout_season."
        )

    print(f"Training on {len(train_df)} matches (seasons != {holdout_season}), "
          f"holding out {len(test_df)} matches (season {holdout_season})...")

    model = PoissonMatchModel(home_features, away_features, max_goals=max_goals, alpha=alpha)
    model.fit(train_df)

    rows = []
    for _, row in test_df.iterrows():
        pred = model.predict_match(row)
        rows.append({
            "match_date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            **pred,
            "actual_home_goals": row["home_goals"],
            "actual_away_goals": row["away_goals"],
            "actual_result": row["result"],
        })
    return pd.DataFrame(rows)


def per_match_brier(df: pd.DataFrame) -> np.ndarray:
    onehot = pd.get_dummies(df["actual_result"]).reindex(columns=["H", "D", "A"], fill_value=0).values.astype(float)
    probs = df[["home_win_prob", "draw_prob", "away_win_prob"]].values
    return ((probs - onehot) ** 2).sum(axis=1)


def per_match_correct(df: pd.DataFrame) -> np.ndarray:
    label = {"home_win_prob": "H", "draw_prob": "D", "away_win_prob": "A"}
    predicted = df[["home_win_prob", "draw_prob", "away_win_prob"]].idxmax(axis=1).map(label)
    return (predicted == df["actual_result"]).astype(float).values


def paired_bootstrap(baseline_df: pd.DataFrame, challenger_df: pd.DataFrame,
                      n_boot: int = 10000, seed: int = 0) -> dict:
    """
    Paired bootstrap over matches for (challenger - baseline) differences in
    Brier score, log loss, and winner accuracy. Both frames must cover the identical
    matches in the same order. With one holdout season a gap of a percentage
    point or two is often noise; the 95% interval says whether it is.
    """
    keys = ["match_date", "home_team", "away_team"]
    if not baseline_df[keys].reset_index(drop=True).equals(challenger_df[keys].reset_index(drop=True)):
        raise ValueError("Paired bootstrap needs both frames to hold the identical matches in the same order.")

    diffs = {
        "brier_diff": per_match_brier(challenger_df) - per_match_brier(baseline_df),
        "log_loss_diff": per_match_log_loss(challenger_df) - per_match_log_loss(baseline_df),
        "accuracy_diff": per_match_correct(challenger_df) - per_match_correct(baseline_df),
    }
    rng = np.random.default_rng(seed)
    n = len(baseline_df)
    idx = rng.integers(0, n, size=(n_boot, n))

    out = {"n": n}
    for name, d in diffs.items():
        boot_means = d[idx].mean(axis=1)
        low, high = np.percentile(boot_means, [2.5, 97.5])
        out[name] = {
            "mean": round(float(d.mean()), 4),
            "ci95": (round(float(low), 4), round(float(high), 4)),
            "excludes_zero": bool(low > 0 or high < 0),
        }
    return out


def print_bootstrap(baseline_name: str, baseline_df: pd.DataFrame, name: str, challenger_df: pd.DataFrame):
    boot = paired_bootstrap(baseline_df, challenger_df)
    print(f"  {name} vs. {baseline_name} (n={boot['n']})")
    for key in ("brier_diff", "log_loss_diff", "accuracy_diff"):
        b = boot[key]
        verdict = "interval excludes 0" if b["excludes_zero"] else "interval includes 0 -> not distinguishable from noise"
        print(f"    {key:<14} mean {b['mean']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  ({verdict})")


def compare_feature_sets(df: pd.DataFrame, feature_sets: dict, holdout_season: int, max_goals: int,
                          out_dir: Path, alpha: float = 1.0) -> dict:
    """
    Fits every named feature set on the non-holdout seasons, scores each on
    the identical set of holdout matches, saves each one's predictions,
    prints a metrics table, and prints a paired bootstrap of every
    challenger against the FIRST entry (the baseline). Returns each feature
    set's held-out predictions, for any further paired comparisons.
    """
    # A fair comparison needs every feature set scored on the SAME matches --
    # take the intersection of each set's valid (non-NaN) holdout-season rows.
    common_eval_ids = None
    for home_features, away_features in feature_sets.values():
        valid = df.dropna(subset=home_features + away_features)
        ids = set(valid.loc[valid["season"] == holdout_season, "match_id"])
        common_eval_ids = ids if common_eval_ids is None else common_eval_ids & ids
    print(f"Common evaluation set across all feature sets: {len(common_eval_ids)} matches "
          f"(season {holdout_season}).\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    backtest_frames = {}
    for name, (home_features, away_features) in feature_sets.items():
        backtest_df = run_poisson_backtest(df, home_features, away_features, holdout_season, max_goals,
                                            eval_match_ids=common_eval_ids, alpha=alpha)
        out_path = out_dir / f"poisson_holdout_backtest_{name}.csv"
        backtest_df.to_csv(out_path, index=False)
        print(f"Saved {len(backtest_df)} held-out prediction(s) -> {out_path.relative_to(PROJECT_ROOT)}")
        results[name] = compute_metrics(backtest_df)
        backtest_frames[name] = backtest_df

    print(f"\nPython Poisson -- held-out backtest comparison (season {holdout_season} never seen in training):")
    metric_keys = ["n_evaluated", "winner_accuracy", "exact_score_accuracy",
                    "mae_home_goals", "mae_away_goals", "brier_score", "log_loss"]
    header = f"  {'metric':<22}" + "".join(f"{name:>34}" for name in results)
    print(header)
    for key in metric_keys:
        row = f"  {key:<22}" + "".join(f"{results[name].get(key, 'n/a'):>34}" for name in results)
        print(row)

    baseline_name = next(iter(feature_sets))
    print(f"\nPaired bootstrap vs. '{baseline_name}' (challenger - baseline; 95% CI over matches, "
          f"10,000 resamples). Brier/log loss: negative = better. Accuracy: positive = better.")
    for name in feature_sets:
        if name != baseline_name:
            print_bootstrap(baseline_name, backtest_frames[baseline_name], name, backtest_frames[name])
    return backtest_frames


def main():
    cfg = load_config()
    holdout_season = cfg["evaluation"]["holdout_season"]
    windows = cfg["features"]["form_windows"]
    max_goals = cfg["model"]["max_goals_simulated"]
    alpha = cfg["model"]["poisson_alpha"]
    match_feat_path = PROJECT_ROOT / cfg["paths"]["gold_match_features"] / "match_features.csv"
    out_dir = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"]

    if not match_feat_path.exists():
        print(f"Missing {match_feat_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.features.build_match_features")
        return

    df = pd.read_csv(match_feat_path, parse_dates=["date"])

    # Named candidate feature sets, all scored on the exact same held-out
    # season so results are directly comparable -- see docs/PROGRESS.md for
    # the research question behind each one. windows[0] is the baseline
    # window; the other configured windows are challengers (2026-10-03).
    feature_sets = {
        "rolling_form_baseline": build_feature_lists(windows),
        "team_level_venue_split": build_team_level_feature_lists(),
        "sos_adjusted_form": build_sos_feature_lists(windows),
    }
    for w in windows[1:]:
        feature_sets[f"rolling_form_last{w}"] = build_feature_lists([w])
    print(f"Poisson L2 penalty: alpha={alpha} (config.yaml -> model.poisson_alpha)\n")
    print("=== Experiment 1: form/team-level feature sets ===")
    compare_feature_sets(df, feature_sets, holdout_season, max_goals, out_dir, alpha=alpha)

    # Squad quality (2026-10-10) is undefined for the first season with
    # player data (no previous season to look back at), so its model can
    # only train on later seasons. Every arm here is refit on exactly those
    # rows, so differences are the features', not the training sets'. The
    # control arm (last season's TEAM goals, no player data) separates
    # "player information helps" from "any longer-horizon strength helps".
    squad_home, squad_away = build_squad_feature_lists(windows)
    squad_df = df.dropna(subset=squad_home + squad_away)
    print("\n=== Experiment 2: squad quality (all arms trained on the rows where it exists) ===")
    frames = compare_feature_sets(squad_df, {
        "rolling_form_baseline_squad_rows": build_feature_lists(windows),
        "prev_season_team_goals_control": build_prev_season_control_feature_lists(windows),
        "squad_quality": (squad_home, squad_away),
    }, holdout_season, max_goals, out_dir, alpha=alpha)
    print("\nWhat the roster adds beyond last season's team goals:")
    print_bootstrap("prev_season_team_goals_control", frames["prev_season_team_goals_control"],
                    "squad_quality", frames["squad_quality"])

    print("\nCompare against src/models/dixon_coles_model.R's holdout output "
          "(same season, same schema) for a like-for-like model comparison.")


if __name__ == "__main__":
    main()
