"""
Baseline forecasting model: Poisson regression.

Why start here: goals in a soccer match are reasonably well approximated by
independent Poisson processes. Fitting two Poisson regressions (expected home
goals, expected away goals) from attack/defense strength features gives a
simple, fast, EXPLAINABLE baseline that every fancier model should be
compared against -- per the project's "don't assume complex = better" goal.

Model:
    home_goals ~ Poisson(lambda_home)
    away_goals ~ Poisson(lambda_away)

    log(lambda_home) = f(home_attack_form, away_defense_form, home_advantage)
    log(lambda_away) = f(away_attack_form, home_defense_form)

From (lambda_home, lambda_away) we derive the full scoreline probability
grid, and from that: win/draw/loss probabilities and the most likely score.

Usage:
    python -m src.models.poisson_model            # train + save
    python -m src.models.poisson_model --predict "Arsenal" "Chelsea"
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class PoissonMatchModel:
    """Two independent Poisson regressors: one for home goals, one for away."""

    def __init__(self, feature_cols_home: list, feature_cols_away: list, max_goals: int = 10):
        self.feature_cols_home = feature_cols_home
        self.feature_cols_away = feature_cols_away
        self.max_goals = max_goals
        self.home_model = PoissonRegressor(alpha=1.0, max_iter=500)
        self.away_model = PoissonRegressor(alpha=1.0, max_iter=500)

    def fit(self, df: pd.DataFrame):
        X_home = df[self.feature_cols_home].values
        X_away = df[self.feature_cols_away].values
        self.home_model.fit(X_home, df["home_goals"].values)
        self.away_model.fit(X_away, df["away_goals"].values)
        return self

    def predict_lambdas(self, row: pd.Series):
        x_home = row[self.feature_cols_home].values.reshape(1, -1).astype(float)
        x_away = row[self.feature_cols_away].values.reshape(1, -1).astype(float)
        lam_home = float(self.home_model.predict(x_home)[0])
        lam_away = float(self.away_model.predict(x_away)[0])
        return lam_home, lam_away

    def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        """Probability of every scoreline from 0-0 up to max_goals-max_goals."""
        gh = np.arange(0, self.max_goals + 1)
        ga = np.arange(0, self.max_goals + 1)
        p_home = poisson.pmf(gh, lam_home)
        p_away = poisson.pmf(ga, lam_away)
        return np.outer(p_home, p_away)  # [home_goals, away_goals]

    def outcome_probabilities(self, matrix: np.ndarray):
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        return home_win, draw, away_win

    def predict_match(self, row: pd.Series) -> dict:
        lam_home, lam_away = self.predict_lambdas(row)
        matrix = self.score_matrix(lam_home, lam_away)
        home_win, draw, away_win = self.outcome_probabilities(matrix)

        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        confidence = float(matrix.max())  # probability mass on the single most likely score

        return {
            "expected_home_goals": round(lam_home, 2),
            "expected_away_goals": round(lam_away, 2),
            "predicted_score": f"{best_idx[0]}-{best_idx[1]}",
            "home_win_prob": round(float(home_win), 4),
            "draw_prob": round(float(draw), 4),
            "away_win_prob": round(float(away_win), 4),
            "prediction_confidence": round(confidence, 4),
        }

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "PoissonMatchModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_feature_lists(windows: list):
    w = windows[0]
    home_features = [f"home_avg_goals_for_last{w}", f"away_avg_goals_against_last{w}"]
    away_features = [f"away_avg_goals_for_last{w}", f"home_avg_goals_against_last{w}"]
    return home_features, away_features


def main():
    parser = argparse.ArgumentParser(description="Train (or query) the baseline Poisson model.")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"),
                         help="Predict a hypothetical match using each team's most recent form.")
    args = parser.parse_args()

    cfg = load_config()
    windows = cfg["features"]["form_windows"]
    match_feat_path = PROJECT_ROOT / cfg["paths"]["gold_match_features"] / "match_features.csv"
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "poisson_baseline.pkl"

    if not match_feat_path.exists():
        print(f"Missing {match_feat_path.relative_to(PROJECT_ROOT)}. "
              f"Run: python -m src.features.build_match_features")
        return

    df = pd.read_csv(match_feat_path, parse_dates=["date"])
    home_features, away_features = build_feature_lists(windows)
    train_df = df.dropna(subset=home_features + away_features)

    if args.predict:
        if not model_path.exists():
            print("No trained model found -- training first...")
        else:
            model = PoissonMatchModel.load(model_path)
            home_team, away_team = args.predict
            print(f"Model loaded. To predict {home_team} vs {away_team}, "
                  f"pull each team's latest feature row from "
                  f"data/gold/team_features/team_match_history.csv and call "
                  f"model.predict_match(row). See README for a full example.")
            return

    print(f"Training on {len(train_df)} matches with complete rolling-form features "
          f"(window={windows[0]})...")
    model = PoissonMatchModel(home_features, away_features, max_goals=cfg["model"]["max_goals_simulated"])
    model.fit(train_df)
    model.save(model_path)
    print(f"Saved model -> {model_path.relative_to(PROJECT_ROOT)}")

    # Quick sanity check on a held-out slice (last 10% of matches chronologically)
    n_test = max(1, int(len(train_df) * 0.1))
    test_df = train_df.sort_values("date").tail(n_test)
    correct = 0
    for _, row in test_df.iterrows():
        pred = model.predict_match(row)
        probs = {"H": pred["home_win_prob"], "D": pred["draw_prob"], "A": pred["away_win_prob"]}
        predicted_result = max(probs, key=probs.get)
        if predicted_result == row["result"]:
            correct += 1
    print(f"Quick sanity check: {correct}/{len(test_df)} correct winner predictions "
          f"on the most recent {n_test} matches ({100 * correct / len(test_df):.1f}%).")
    print("(This is a rough in-sample-ish check, not real backtesting -- "
          "see src/evaluation/evaluate_predictions.py for proper evaluation.)")


if __name__ == "__main__":
    main()
