"""
Unit tests for Gold-layer feature engineering — specifically that rolling
form features don't leak the current match's own result into itself.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_match_features import (to_team_long, add_rolling_form, add_sos_adjusted_form,
                                               add_venue_split_averages, build_prev_season_team_goals,
                                               build_squad_quality, previous_season)


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


def make_sos_sequence(a_goals_m4=0):
    """Three teams, one match per date, so league averages are easy to hand-compute."""
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3", "m4"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22"]),
        "season": ["2526"] * 4,
        "home_team": ["A", "C", "A", "B"],
        "away_team": ["B", "B", "C", "A"],
        "home_goals": [2, 3, 1, 0],
        "away_goals": [0, 1, 1, a_goals_m4],
        "result": ["H", "H", "D", "D" if a_goals_m4 == 0 else "A"],
    })


def sos_row(form, team, match_id):
    return form[(form["team"] == team) & (form["match_id"] == match_id)].iloc[0]


def test_sos_adjustment_matches_hand_calculation():
    form = add_sos_adjusted_form(to_team_long(make_sos_sequence()), windows=[2], min_matches=1)

    # A vs C at m3: league avg before m3 = (2+0+3+1)/4 = 1.5 goals per team per match.
    # C entering m3 had conceded 1 and scored 3 per match (m2 only).
    # adj_for = 1 - (1 - 1.5) = 1.5 ; adj_against = 1 - (3 - 1.5) = -0.5
    a_m3 = sos_row(form, "A", "m3")
    assert a_m3["adj_goals_for"] == 1.5
    assert a_m3["adj_goals_against"] == -0.5

    # A entering m4: m1 has no prior league history (NaN, skipped), so the
    # 2-match window holds only m3's adjusted values.
    a_m4 = sos_row(form, "A", "m4")
    assert a_m4["avg_sos_goals_for_last2"] == 1.5
    assert a_m4["avg_sos_goals_against_last2"] == -0.5


def test_sos_opponent_without_history_means_no_adjustment():
    form = add_sos_adjusted_form(to_team_long(make_sos_sequence()), windows=[2], min_matches=1)
    # C's first match is m2, so B's m2 opponent is treated as league-average:
    # adjusted goals equal raw goals.
    b_m2 = sos_row(form, "B", "m2")
    assert b_m2["adj_goals_for"] == 1.0
    assert b_m2["adj_goals_against"] == 3.0


def test_sos_no_leakage_of_current_match():
    base = add_sos_adjusted_form(to_team_long(make_sos_sequence(a_goals_m4=0)), windows=[2], min_matches=1)
    changed = add_sos_adjusted_form(to_team_long(make_sos_sequence(a_goals_m4=5)), windows=[2], min_matches=1)
    # Changing m4's own score must not move either side's features for m4.
    for team in ("A", "B"):
        for col in ("avg_sos_goals_for_last2", "avg_sos_goals_against_last2"):
            assert sos_row(base, team, "m4")[col] == sos_row(changed, team, "m4")[col]


def make_squad_inputs():
    """Two seasons. Striker scores 19 for Villa in 2324, moves to Spurs for
    2425 and scores 30 there. Wolves are in the PL in 2324 but their article
    didn't parse; Luton are promoted for 2425 (not in the PL in 2324)."""
    matches = pd.DataFrame({
        "home_team": ["Villa", "Wolves", "Villa", "Spurs", "Luton", "Wolves"],
        "away_team": ["Spurs", "Villa", "Spurs", "Luton", "Wolves", "Villa"],
        "season": [2324, 2324, 2425, 2425, 2425, 2425],
    })
    player_stats = pd.DataFrame({
        "team":   ["Villa", "Villa",    "Spurs", "Villa", "Spurs",   "Spurs", "Luton",    "Wolves"],
        "season": [2324,    2324,       2324,    2425,    2425,      2425,    2425,       2425],
        "player": ["Striker", "Winger", "Mid",   "Winger", "Striker", "Mid",  "Newcomer", "Keeper"],
        "league_goals": [19, 6,         4,       8,       30,        5,       12,         0],
    })
    return build_squad_quality(player_stats, matches).set_index(["team", "season"])["squad_prior_goals_per_match"]


def test_previous_season():
    assert previous_season(2526) == 2425
    assert previous_season(2223) == 2122


def test_squad_quality_uses_previous_season_goals_only():
    squad = make_squad_inputs()
    # Spurs 2425 roster = Striker (19 last season, at Villa) + Mid (4 last
    # season). Their 2425 goals (30, 5) are this season's and must not count.
    assert squad[("Spurs", 2425)] == (19 + 4) / 38
    # Villa keeps only Winger's 6; Striker's 19 left with him.
    assert squad[("Villa", 2425)] == 6 / 38


def test_squad_quality_missing_data_is_nan_not_zero():
    squad = make_squad_inputs()
    assert pd.isna(squad[("Villa", 2324)])   # no player data for 2223 at all
    assert pd.isna(squad[("Wolves", 2425)])  # in the PL in 2324, but that article is missing


def test_squad_quality_promoted_team_is_computed():
    squad = make_squad_inputs()
    # Luton weren't in the PL in 2324: no PL goals last season is a real 0,
    # not missing data. Newcomer's 12 goals are 2425's own and don't count.
    assert squad[("Luton", 2425)] == 0.0


def test_prev_season_team_goals_control():
    matches = pd.DataFrame({
        "match_id": ["a", "b", "c"],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2025-01-01"]),
        "season": [2324, 2324, 2425],
        "home_team": ["Villa", "Spurs", "Villa"],
        "away_team": ["Spurs", "Villa", "Luton"],
        "home_goals": [3, 2, 1],
        "away_goals": [1, 0, 1],
        "result": ["H", "H", "D"],
    })
    ctrl = build_prev_season_team_goals(matches).set_index(["team", "season"])["prev_season_goals_per_match"]
    assert ctrl[("Villa", 2425)] == (3 + 0) / 2   # Villa's 2324 goals: 3 at home, 0 away
    assert ctrl[("Luton", 2425)] == 0.0           # promoted: not in the PL in 2324
    assert pd.isna(ctrl[("Villa", 2324)])         # 2223 isn't in the data at all
