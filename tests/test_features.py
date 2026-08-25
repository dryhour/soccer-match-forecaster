"""
Unit tests for Gold-layer feature engineering — specifically that rolling
form features don't leak the current match's own result into itself.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_match_features import to_team_long, add_rolling_form


def make_sample_matches():
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15"]),
        "season": ["2526"] * 3,
        "home_team": ["Arsenal", "Chelsea", "Arsenal"],
        "away_team": ["Chelsea", "Arsenal", "Chelsea"],
        "home_goals": [3, 0, 1],
        "away_goals": [1, 2, 1],
        "result": ["H", "A", "D"],
    })


def test_to_team_long_row_count():
    matches = make_sample_matches()
    long_df = to_team_long(matches)
    assert len(long_df) == len(matches) * 2


def test_no_leakage_in_rolling_form():
    matches = make_sample_matches()
    long_df = to_team_long(matches)
    form = add_rolling_form(long_df, windows=[2], min_matches=1)

    # Arsenal's form entering match m3 should reflect ONLY m1 (3 goals for)
    # and m2 (0 goals for as away team doesn't apply -- m2 Arsenal was away,
    # scored 2), not m3 itself (1 goal).
    arsenal_m3 = form[(form["team"] == "Arsenal") & (form["match_id"] == "m3")]
    assert not arsenal_m3.empty
    avg_goals_for = arsenal_m3.iloc[0]["avg_goals_for_last2"]
    # Should be mean of [3 (m1), 2 (m2)] = 2.5, NOT including m3's own value of 1
    assert avg_goals_for == 2.5
