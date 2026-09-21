"""
Unit tests for the chronological train/test backtest.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.backtest import run_poisson_backtest
from src.models.poisson_model import build_feature_lists


def make_match_features(n_train=20, n_test=6):
    """Synthetic match_features-shaped rows across two seasons."""
    home_features, away_features = build_feature_lists([5])
    rows = []
    for i in range(n_train):
        rows.append({
            "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
            "season": 2223, "home_team": "A", "away_team": "B",
            home_features[0]: 1.5, home_features[1]: 1.0,
            away_features[0]: 1.2, away_features[1]: 1.3,
            "home_goals": 2, "away_goals": 1, "result": "H",
        })
    for i in range(n_test):
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "season": 2526, "home_team": "A", "away_team": "B",
            home_features[0]: 1.8, home_features[1]: 0.9,
            away_features[0]: 1.1, away_features[1]: 1.4,
            "home_goals": 1, "away_goals": 1, "result": "D",
        })
    return pd.DataFrame(rows), home_features, away_features


def test_backtest_only_scores_holdout_season():
    df, home_features, away_features = make_match_features()
    result = run_poisson_backtest(df, home_features, away_features, holdout_season=2526, max_goals=10)
    assert len(result) == 6
    assert set(result.columns) >= {
        "expected_home_goals", "expected_away_goals", "predicted_score",
        "actual_home_goals", "actual_away_goals", "actual_result",
    }


def test_backtest_raises_on_empty_split():
    df, home_features, away_features = make_match_features()
    with pytest.raises(ValueError):
        run_poisson_backtest(df, home_features, away_features, holdout_season=9999, max_goals=10)
