"""
Unit tests for prediction-row generation.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.poisson_model import PoissonMatchModel, build_feature_lists
from src.predictions.generate_predictions import build_prediction_row


def make_fitted_model():
    df = pd.DataFrame({
        "home_avg_goals_for_last5": [1.0, 2.0, 1.5, 0.5, 2.5, 1.0],
        "away_avg_goals_against_last5": [1.0, 1.5, 1.0, 2.0, 0.5, 1.2],
        "away_avg_goals_for_last5": [1.2, 0.8, 1.5, 2.0, 0.5, 1.0],
        "home_avg_goals_against_last5": [1.0, 1.5, 0.8, 1.0, 2.0, 1.5],
        "home_goals": [1, 3, 2, 0, 3, 1],
        "away_goals": [1, 0, 2, 3, 0, 1],
    })
    home_features, away_features = build_feature_lists([5])
    model = PoissonMatchModel(home_features, away_features, max_goals=6)
    return model.fit(df)


def make_history():
    return pd.DataFrame({
        "team": ["Arsenal", "Chelsea"],
        "date": pd.to_datetime(["2026-02-01", "2026-02-01"]),
        "avg_goals_for_last5": [2.0, 1.2],
        "avg_goals_against_last5": [1.0, 1.3],
    })


def test_build_prediction_row_has_expected_fields():
    model = make_fitted_model()
    row = build_prediction_row(model, make_history(), "Arsenal", "Chelsea", "2026-08-30")
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["match_date"] == "2026-08-30"
    assert row["actual_result"] is None
    assert "predicted_score" in row
    assert "prediction_id" in row
    assert "generated_at" in row


def test_build_prediction_row_unknown_team_raises():
    model = make_fitted_model()
    history = pd.DataFrame({
        "team": ["Arsenal"],
        "date": pd.to_datetime(["2026-02-01"]),
        "avg_goals_for_last5": [2.0],
        "avg_goals_against_last5": [1.0],
    })
    try:
        build_prediction_row(model, history, "Arsenal", "Nowhere FC", "2026-08-30")
        assert False, "expected ValueError"
    except ValueError:
        pass
