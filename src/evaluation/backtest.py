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

import pandas as pd
import yaml

from src.evaluation.evaluate_predictions import compute_metrics
from src.models.poisson_model import PoissonMatchModel, build_feature_lists, build_team_level_feature_lists

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_poisson_backtest(df: pd.DataFrame, home_features: list, away_features: list,
                          holdout_season: int, max_goals: int, eval_match_ids: set = None) -> pd.DataFrame:
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

    model = PoissonMatchModel(home_features, away_features, max_goals=max_goals)
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


def main():
    cfg = load_config()
    holdout_season = cfg["evaluation"]["holdout_season"]
    windows = cfg["features"]["form_windows"]
    max_goals = cfg["model"]["max_goals_simulated"]
    match_feat_path = PROJECT_ROOT / cfg["paths"]["gold_match_features"] / "match_features.csv"
    out_dir = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"]

    if not match_feat_path.exists():
        print(f"Missing {match_feat_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.features.build_match_features")
        return

    df = pd.read_csv(match_feat_path, parse_dates=["date"])

    # Named candidate feature sets, all scored on the exact same held-out
    # season so results are directly comparable -- see docs/PROGRESS.md's
    # 2026-09-26 entry for the research question this answers.
    feature_sets = {
        "rolling_form_baseline": build_feature_lists(windows),
        "team_level_venue_split": build_team_level_feature_lists(),
    }

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
    for name, (home_features, away_features) in feature_sets.items():
        backtest_df = run_poisson_backtest(df, home_features, away_features, holdout_season, max_goals,
                                            eval_match_ids=common_eval_ids)
        out_path = out_dir / f"poisson_holdout_backtest_{name}.csv"
        backtest_df.to_csv(out_path, index=False)
        print(f"Saved {len(backtest_df)} held-out prediction(s) -> {out_path.relative_to(PROJECT_ROOT)}")
        results[name] = compute_metrics(backtest_df)

    print(f"\nPython Poisson -- held-out backtest comparison (season {holdout_season} never seen in training):")
    metric_keys = ["n_evaluated", "winner_accuracy", "exact_score_accuracy",
                    "mae_home_goals", "mae_away_goals", "brier_score"]
    header = f"  {'metric':<22}" + "".join(f"{name:>26}" for name in results)
    print(header)
    for key in metric_keys:
        row = f"  {key:<22}" + "".join(f"{results[name].get(key, 'n/a'):>26}" for name in results)
        print(row)

    print("\nCompare against src/models/dixon_coles_model.R's holdout output "
          "(same season, same schema) for a like-for-like model comparison.")


if __name__ == "__main__":
    main()
