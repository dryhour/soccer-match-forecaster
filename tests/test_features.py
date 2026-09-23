"""
Unit tests for Gold-layer feature engineering — specifically that rolling
form features don't leak the current match's own result into itself.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_match_features import to_team_long, add_rolling_form, add_venue_split_averages


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


def make_arsenal_venue_sequence():
    """Arsenal alternating home/away, opponent irrelevant to this test."""
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3", "m4", "m5"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22", "2026-01-29"]),
        "season": ["2526"] * 5,
        "home_team": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
        "away_team": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
        "home_goals": [2, 3, 4, 6, 3],   # m2/m4 home_goals is Chelsea's, irrelevant here
        "away_goals": [1, 5, 0, 1, 2],   # m2/m4 away_goals is Arsenal's (5, then 1)
        "result": ["H", "H", "H", "H", "H"],
    })


def test_venue_split_ignores_other_venue_matches():
    matches = make_arsenal_venue_sequence()
    long_df = to_team_long(matches)
    form = add_venue_split_averages(long_df, min_matches=1)

    # Arsenal's 2nd HOME match (m3) should look back only at its 1st home
    # match (m1: 2 goals), ignoring the away match (m2: 5 goals) played in between.
    arsenal_m3 = form[(form["team"] == "Arsenal") & (form["match_id"] == "m3") & (form["is_home"] == 1)]
    assert arsenal_m3.iloc[0]["avg_goals_for_by_venue"] == 2.0

    # Arsenal's 2nd AWAY match (m4) should look back only at its 1st away
    # match (m2: 5 goals), ignoring the two home matches played around it.
    arsenal_m4 = form[(form["team"] == "Arsenal") & (form["match_id"] == "m4") & (form["is_home"] == 0)]
    assert arsenal_m4.iloc[0]["avg_goals_for_by_venue"] == 5.0


def test_venue_split_no_leakage_of_current_match():
    matches = make_arsenal_venue_sequence()
    long_df = to_team_long(matches)
    form = add_venue_split_averages(long_df, min_matches=1)

    # Arsenal's 3rd home match (m5, scored 3) should average its first two
    # home matches (m1: 2, m3: 4) = 3.0, never including m5's own 3 goals.
    arsenal_m5 = form[(form["team"] == "Arsenal") & (form["match_id"] == "m5") & (form["is_home"] == 1)]
    assert arsenal_m5.iloc[0]["avg_goals_for_by_venue"] == 3.0

    # Arsenal's very first home match has no prior home history -> NaN.
    arsenal_m1 = form[(form["team"] == "Arsenal") & (form["match_id"] == "m1") & (form["is_home"] == 1)]
    assert pd.isna(arsenal_m1.iloc[0]["avg_goals_for_by_venue"])
