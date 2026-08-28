"""
Unit tests for prediction evaluation.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_predictions import update_actuals, compute_metrics


def make_pred_log():
    return pd.DataFrame({
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Chelsea", "Liverpool"],
        "home_win_prob": [0.5, 0.3],
        "draw_prob": [0.25, 0.3],
        "away_win_prob": [0.25, 0.4],
        "predicted_score": ["2-1", "1-1"],
        "expected_home_goals": [2.0, 1.0],
        "expected_away_goals": [1.0, 1.0],
        "actual_home_goals": [None, None],
        "actual_away_goals": [None, None],
        "actual_result": [None, None],
    })


def make_matches():
    return pd.DataFrame({
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
        "home_goals": [2],
        "away_goals": [1],
        "result": ["H"],
    })


def test_update_actuals_fills_known_match():
    updated = update_actuals(make_pred_log(), make_matches())
    row = updated.iloc[0]
    assert row["actual_home_goals"] == 2
    assert row["actual_away_goals"] == 1
    assert row["actual_result"] == "H"


def test_update_actuals_leaves_unmatched_row_empty():
    updated = update_actuals(make_pred_log(), make_matches())
    assert pd.isna(updated.iloc[1]["actual_result"])


def test_compute_metrics_on_resolved_predictions():
    updated = update_actuals(make_pred_log(), make_matches())
    metrics = compute_metrics(updated)
    # Only the Arsenal-Chelsea row has a known actual result.
    assert metrics["n_evaluated"] == 1
    assert metrics["winner_accuracy"] == 1.0
    assert metrics["exact_score_accuracy"] == 1.0
    assert metrics["mae_home_goals"] == 0.0
    assert metrics["mae_away_goals"] == 0.0


def test_compute_metrics_with_no_resolved_predictions():
    pred_log = make_pred_log()
    metrics = compute_metrics(pred_log)
    assert "note" in metrics
