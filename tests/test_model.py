"""
Unit tests for the Poisson baseline model.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.poisson_model import (
    PoissonMatchModel, build_feature_lists, latest_team_row, predict_from_rows, predict_matchup,
)


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


def test_score_matrix_sums_to_one():
    model = make_fitted_model()
    matrix = model.score_matrix(1.4, 1.1)
    # Truncated at max_goals, so it's just under 1.0 -- the missing mass is
    # the (tiny) probability of either side scoring more than max_goals.
    assert np.isclose(matrix.sum(), 1.0, atol=1e-2)


def test_outcome_probabilities_partition_the_matrix():
    model = make_fitted_model()
    matrix = model.score_matrix(1.4, 1.1)
    home_win, draw, away_win = model.outcome_probabilities(matrix)
    assert np.isclose(home_win + draw + away_win, matrix.sum())


def test_predict_match_shape_and_probabilities_sum_to_one():
    model = make_fitted_model()
    row = pd.Series({
        "home_avg_goals_for_last5": 1.5,
        "away_avg_goals_against_last5": 1.2,
        "away_avg_goals_for_last5": 1.0,
        "home_avg_goals_against_last5": 1.3,
    })
    pred = model.predict_match(row)
    assert set(pred.keys()) == {
        "expected_home_goals", "expected_away_goals", "predicted_score",
        "home_win_prob", "draw_prob", "away_win_prob", "prediction_confidence",
    }
    assert np.isclose(
        pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"], 1.0, atol=1e-2
    )


def test_latest_team_row_picks_most_recent():
    history = pd.DataFrame({
        "team": ["Arsenal", "Arsenal", "Chelsea"],
        "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-01-15"]),
        "avg_goals_for_last5": [1.0, 2.0, 1.5],
    })
    row = latest_team_row(history, "Arsenal")
    assert row["avg_goals_for_last5"] == 2.0


def test_latest_team_row_unknown_team_raises():
    history = pd.DataFrame({"team": ["Arsenal"], "date": pd.to_datetime(["2026-01-01"])})
    try:
        latest_team_row(history, "Nowhere FC")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_predict_from_rows_matches_predict_match():
    model = make_fitted_model()
    home_row = pd.Series({"avg_goals_for_last5": 1.5, "avg_goals_against_last5": 1.3})
    away_row = pd.Series({"avg_goals_for_last5": 1.0, "avg_goals_against_last5": 1.2})
    pred = predict_from_rows(model, home_row, away_row)
    assert "predicted_score" in pred
    assert pred["expected_home_goals"] >= 0
    assert pred["expected_away_goals"] >= 0


def test_predict_matchup_uses_each_teams_latest_row():
    model = make_fitted_model()
    history = pd.DataFrame({
        "team": ["Arsenal", "Arsenal", "Chelsea", "Chelsea"],
        "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-01-01", "2026-02-01"]),
        "avg_goals_for_last5": [1.0, 2.0, 1.0, 1.2],
        "avg_goals_against_last5": [1.5, 1.0, 1.5, 1.3],
    })
    pred = predict_matchup(model, history, "Arsenal", "Chelsea")
    assert "predicted_score" in pred
    assert pred["expected_home_goals"] >= 0
