"""
Generate and store predictions for one or more upcoming matches.

Given a fixture list (home team, away team, date), this pulls each team's
most recent rolling-form row from the Gold team_features table, runs the
trained model, and appends the result to a prediction log
(data/gold/prediction_features/prediction_log.csv) so it can later be
compared against actual results (see src/evaluation/evaluate_predictions.py).

Usage:
    python -m src.predictions.generate_predictions --fixtures fixtures.csv

    fixtures.csv format:
        date,home_team,away_team
        2026-08-30,Arsenal,Chelsea
"""

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.models.poisson_model import PoissonMatchModel, predict_matchup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_prediction_row(model: PoissonMatchModel, team_history: pd.DataFrame,
                          home_team: str, away_team: str, match_date: str) -> dict:
    pred = predict_matchup(model, team_history, home_team, away_team)
    pred.update({
        "prediction_id": str(uuid.uuid4())[:8],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "actual_home_goals": None,
        "actual_away_goals": None,
        "actual_result": None,
    })
    return pred


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for upcoming fixtures.")
    parser.add_argument("--fixtures", required=True, help="CSV with columns: date,home_team,away_team")
    args = parser.parse_args()

    cfg = load_config()
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    model_path = models_dir / "poisson_baseline.pkl"
    team_hist_path = PROJECT_ROOT / cfg["paths"]["gold_team_features"] / "team_match_history.csv"
    log_dir = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "prediction_log.csv"

    if not model_path.exists():
        print(f"No trained model at {model_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.models.poisson_model")
        return
    if not team_hist_path.exists():
        print(f"Missing {team_hist_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.features.build_match_features")
        return

    model = PoissonMatchModel.load(model_path)
    team_history = pd.read_csv(team_hist_path, parse_dates=["date"])
    fixtures = pd.read_csv(Path(args.fixtures))

    predictions = []
    for _, fx in fixtures.iterrows():
        try:
            pred = build_prediction_row(model, team_history, fx["home_team"], fx["away_team"], fx["date"])
            predictions.append(pred)
            print(f"{fx['home_team']} {pred['predicted_score'].replace('-', ' - ')} {fx['away_team']}  "
                  f"(H {pred['home_win_prob']:.0%} / D {pred['draw_prob']:.0%} / A {pred['away_win_prob']:.0%})")
        except ValueError as e:
            print(f"  [skip] {fx['home_team']} vs {fx['away_team']}: {e}")

    if not predictions:
        print("No predictions generated.")
        return

    new_df = pd.DataFrame(predictions)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(log_path, index=False)
    print(f"\nSaved {len(new_df)} prediction(s) -> {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
