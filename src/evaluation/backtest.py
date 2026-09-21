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
from src.models.poisson_model import PoissonMatchModel, build_feature_lists

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_poisson_backtest(df: pd.DataFrame, home_features: list, away_features: list,
                          holdout_season: int, max_goals: int) -> pd.DataFrame:
    feature_cols = home_features + away_features
    df = df.dropna(subset=feature_cols)

    train_df = df[df["season"] != holdout_season]
    test_df = df[df["season"] == holdout_season]
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
    out_path = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"] / "poisson_holdout_backtest.csv"

    if not match_feat_path.exists():
        print(f"Missing {match_feat_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.features.build_match_features")
        return

    df = pd.read_csv(match_feat_path, parse_dates=["date"])
    home_features, away_features = build_feature_lists(windows)

    backtest_df = run_poisson_backtest(df, home_features, away_features, holdout_season, max_goals)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(out_path, index=False)
    print(f"Saved {len(backtest_df)} held-out prediction(s) -> {out_path.relative_to(PROJECT_ROOT)}")

    metrics = compute_metrics(backtest_df)
    print(f"\nPython Poisson -- held-out backtest (season {holdout_season} never seen in training):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nCompare against src/models/dixon_coles_model.R's holdout output "
          "(same season, same schema) for a like-for-like model comparison.")


if __name__ == "__main__":
    main()
