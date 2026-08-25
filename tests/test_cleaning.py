"""
Unit tests for the Silver-layer cleaning logic.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning.clean_matches import standardize_team, add_derived_stats


def test_standardize_team_known_alias():
    assert standardize_team("Man United") == "Manchester United"
    assert standardize_team("Spurs") == "Tottenham Hotspur"


def test_standardize_team_unknown_passthrough():
    assert standardize_team("Arsenal") == "Arsenal"


def test_standardize_team_handles_non_string():
    assert standardize_team(None) is None


def test_add_derived_stats():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-30"]),
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
        "home_goals": [2],
        "away_goals": [1],
        "result": ["H"],
    })
    out = add_derived_stats(df)
    assert out.loc[0, "total_goals"] == 3
    assert out.loc[0, "goal_diff"] == 1
    assert out.loc[0, "result_label"] == "Home Win"
    assert out.loc[0, "match_id"] == "20260830_Arsenal_Chelsea"
