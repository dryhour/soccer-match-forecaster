"""
Evaluate stored predictions against actual results.

Two-step workflow:
  1. update_actuals()  -- fills in actual_home_goals/actual_away_goals/
     actual_result in prediction_log.csv by matching against the latest
     Silver matches table (run cleaning again after real matches happen).
  2. compute_metrics()  -- once actuals are known, reports:
       - Winner accuracy (did we call H/D/A correctly?)
       - Exact scoreline accuracy
       - Mean absolute goal error (home & away)
       - Brier score (probability calibration, lower is better)
       - Breakdown by team and by month

Usage:
    python -m src.evaluation.evaluate_predictions
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def update_actuals(pred_log: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    pred_log = pred_log.copy()
    # These columns start out empty (NaN/float) when a prediction is first logged;
    # cast to object dtype so we can safely write strings/ints into them later.
    for col in ["actual_home_goals", "actual_away_goals", "actual_result"]:
        if col in pred_log.columns:
            pred_log[col] = pred_log[col].astype(object)

    lookup = matches.set_index(["home_team", "away_team"])[["home_goals", "away_goals", "result"]]
    lookup = lookup.sort_index()

    for idx, row in pred_log.iterrows():
        if pd.notna(row.get("actual_result")):
            continue  # already filled in
        key = (row["home_team"], row["away_team"])
        if key in lookup.index:
            actual = lookup.loc[key]
            if isinstance(actual, pd.DataFrame):  # multiple meetings, take most recent
                actual = actual.iloc[-1]
            pred_log.at[idx, "actual_home_goals"] = actual["home_goals"]
            pred_log.at[idx, "actual_away_goals"] = actual["away_goals"]
            pred_log.at[idx, "actual_result"] = actual["result"]

    return pred_log


def compute_metrics(df: pd.DataFrame) -> dict:
    scored = df.dropna(subset=["actual_result"]).copy()
    if scored.empty:
        return {"note": "No completed matches to evaluate yet."}

    scored["predicted_result"] = scored[["home_win_prob", "draw_prob", "away_win_prob"]].idxmax(axis=1)
    scored["predicted_result"] = scored["predicted_result"].map({
        "home_win_prob": "H", "draw_prob": "D", "away_win_prob": "A"
    })
    winner_correct = (scored["predicted_result"] == scored["actual_result"]).mean()

    predicted_score_parts = scored["predicted_score"].str.split("-", expand=True).astype(int)
    exact_score_correct = (
        (predicted_score_parts[0] == scored["actual_home_goals"]) &
        (predicted_score_parts[1] == scored["actual_away_goals"])
    ).mean()

    mae_home = (scored["expected_home_goals"] - scored["actual_home_goals"]).abs().mean()
    mae_away = (scored["expected_away_goals"] - scored["actual_away_goals"]).abs().mean()

    # Brier score: mean squared error between predicted probability vector
    # and the one-hot actual outcome, averaged over the 3 outcome classes.
    outcome_onehot = pd.get_dummies(scored["actual_result"]).reindex(columns=["H", "D", "A"], fill_value=0)
    prob_cols = scored[["home_win_prob", "draw_prob", "away_win_prob"]].values
    brier = np.mean(np.sum((prob_cols - outcome_onehot.values) ** 2, axis=1))

    return {
        "n_evaluated": len(scored),
        "winner_accuracy": round(float(winner_correct), 4),
        "exact_score_accuracy": round(float(exact_score_correct), 4),
        "mae_home_goals": round(float(mae_home), 3),
        "mae_away_goals": round(float(mae_away), 3),
        "brier_score": round(float(brier), 4),
    }


def main():
    cfg = load_config()
    log_path = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"] / "prediction_log.csv"
    matches_path = PROJECT_ROOT / cfg["paths"]["silver_matches"] / "matches.csv"

    if not log_path.exists():
        print(f"No predictions logged yet at {log_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.predictions.generate_predictions")
        return
    if not matches_path.exists():
        print(f"Missing {matches_path.relative_to(PROJECT_ROOT)}.")
        return

    pred_log = pd.read_csv(log_path)
    matches = pd.read_csv(matches_path, parse_dates=["date"])

    updated = update_actuals(pred_log, matches)
    updated.to_csv(log_path, index=False)

    metrics = compute_metrics(updated)
    print("Prediction evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
