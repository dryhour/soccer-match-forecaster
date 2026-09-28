"""
Unit tests for prediction evaluation.
Run with: pytest tests/
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_predictions import update_actuals, compute_metrics, per_match_log_loss


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


def test_compute_metrics_log_loss_matches_hand_calculation():
    # Arsenal-Chelsea resolves to "H", predicted with home_win_prob 0.5.
    updated = update_actuals(make_pred_log(), make_matches())
    metrics = compute_metrics(updated)
    assert metrics["log_loss"] == round(-math.log(0.5), 4)


def test_per_match_log_loss_picks_actual_outcome_and_clips_zero():
    df = pd.DataFrame({
        "home_win_prob": [0.2, 0.0],
        "draw_prob": [0.3, 0.5],
        "away_win_prob": [0.5, 0.5],
        "actual_result": ["D", "H"],
    })
    ll = per_match_log_loss(df)
    assert ll[0] == pytest.approx(-math.log(0.3))
    assert math.isfinite(ll[1]) and ll[1] > 30  # clipped at 1e-15, not inf
