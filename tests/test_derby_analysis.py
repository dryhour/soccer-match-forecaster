"""
Unit tests for the derby / rivalry analysis.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.derby_analysis import (MDE_Z, check_team_names, compare_groups, derby_pairs, detection_power,
                                           label_derbies, per_match_scores, walk_forward_poisson)
from src.models.poisson_model import build_feature_lists

DERBY_CFG = {
    "metro_areas": {
        "London": ["Arsenal", "Chelsea", "Tottenham Hotspur"],
        "Merseyside": ["Liverpool", "Everton"],
    },
    "classic": {"North London derby": ["Arsenal", "Tottenham Hotspur"]},
    "rivalry": {
        "North London derby": ["Arsenal", "Tottenham Hotspur"],
        "M23 derby": ["Brighton", "Crystal Palace"],
    },
}


def test_local_pairs_are_every_within_area_combination():
    pairs = derby_pairs(DERBY_CFG, "local")
    assert len(pairs) == 3 + 1  # C(3,2) London pairs + 1 Merseyside pair
    assert pairs[frozenset(("Chelsea", "Arsenal"))] == "London derby"
    assert frozenset(("Arsenal", "Liverpool")) not in pairs


def test_label_derbies_is_order_independent_and_per_definition():
    df = pd.DataFrame({
        "home_team": ["Arsenal", "Tottenham Hotspur", "Chelsea", "Brighton", "Arsenal"],
        "away_team": ["Tottenham Hotspur", "Arsenal", "Arsenal", "Crystal Palace", "Liverpool"],
    })
    out = label_derbies(df, DERBY_CFG)
    assert out["derby_local"].tolist() == [True, True, True, False, False]
    assert out["derby_classic"].tolist() == [True, True, False, False, False]
    # A between-cities rivalry is a rivalry but not a local derby.
    assert out["derby_rivalry"].tolist() == [True, True, False, True, False]
    assert out.loc[0, "pairing"] == out.loc[1, "pairing"]


def test_check_team_names_rejects_misspelled_team():
    known = {"Arsenal", "Chelsea", "Tottenham Hotspur", "Liverpool", "Everton", "Brighton", "Crystal Palace"}
    check_team_names(DERBY_CFG, known)
    with pytest.raises(ValueError, match="Tottenham Hotspur"):
        check_team_names(DERBY_CFG, known - {"Tottenham Hotspur"})


def _pred(p_home, p_draw, p_away, result, home_goals=1, away_goals=1, exp_home=1.5, exp_away=1.0):
    return {"match_date": "2025-01-01", "home_team": "A", "away_team": "B",
            "home_win_prob": p_home, "draw_prob": p_draw, "away_win_prob": p_away,
            "expected_home_goals": exp_home, "expected_away_goals": exp_away,
            "actual_home_goals": home_goals, "actual_away_goals": away_goals, "actual_result": result}


def test_expected_loss_is_probability_weighted_loss_over_outcomes():
    p = (0.5, 0.3, 0.2)
    scores = per_match_scores(pd.DataFrame([_pred(*p, r) for r in "HDA"]))
    # Expected loss under the model = sum_k p_k * (loss if outcome k happens).
    assert scores["expected_brier"].iloc[0] == pytest.approx(np.dot(p, scores["brier"]))
    assert scores["expected_log_loss"].iloc[0] == pytest.approx(np.dot(p, scores["log_loss"]))
    assert scores["expected_brier"].iloc[0] == pytest.approx(1 - (0.25 + 0.09 + 0.04))


def test_excess_loss_averages_zero_when_outcomes_follow_the_model():
    rng = np.random.default_rng(0)
    p = (0.6, 0.25, 0.15)
    results = rng.choice(list("HDA"), size=20000, p=p)
    scores = per_match_scores(pd.DataFrame([_pred(*p, r) for r in results]))
    assert scores["excess_brier"].mean() == pytest.approx(0, abs=0.01)
    assert scores["excess_log_loss"].mean() == pytest.approx(0, abs=0.02)


def test_calibration_residuals_by_hand():
    scores = per_match_scores(pd.DataFrame([
        _pred(0.6, 0.25, 0.15, "H", home_goals=2, away_goals=0, exp_home=1.5, exp_away=1.0),
        _pred(0.2, 0.3, 0.5, "H", home_goals=1, away_goals=0, exp_home=1.0, exp_away=1.4),
    ]))
    assert scores["resid_draw"].tolist() == pytest.approx([-0.25, -0.3])
    # Home favorite won (1 - 0.6); away favorite lost (0 - 0.5).
    assert scores["resid_favorite_win"].tolist() == pytest.approx([0.4, -0.5])
    # Actual goal diff minus expected: (2 - 0.5), (1 - (-0.4)).
    assert scores["resid_goal_diff"].tolist() == pytest.approx([1.5, 1.4])


def test_compare_groups_difference_and_interval():
    values = np.array([1.0, 1.2, 0.8, 1.0, 0.0, 0.1, -0.1, 0.0, 0.05, -0.05])
    is_derby = np.array([True] * 4 + [False] * 6)
    res = compare_groups(values, is_derby)
    assert res["n_derby"] == 4 and res["n_other"] == 6
    assert res["diff"] == pytest.approx(1.0)
    assert res["ci_low"] < 1.0 < res["ci_high"]
    assert res["excludes_zero"]


def test_compare_groups_with_singleton_clusters_matches_match_level():
    rng = np.random.default_rng(1)
    values = rng.normal(size=200)
    is_derby = np.arange(200) < 40
    match = compare_groups(values, is_derby, seed=3)
    cluster = compare_groups(values, is_derby, clusters=np.arange(200), seed=3)
    assert cluster["ci_low"] == pytest.approx(match["ci_low"], abs=0.05)
    assert cluster["ci_high"] == pytest.approx(match["ci_high"], abs=0.05)


def test_compare_groups_needs_two_matches_per_group():
    with pytest.raises(ValueError):
        compare_groups([1.0, 2.0, 3.0], [True, False, False])


def test_detection_power_at_minimum_detectable_difference_is_80_percent():
    assert detection_power(MDE_Z * 0.01, 0.01) == pytest.approx(0.80, abs=0.005)
    assert detection_power(0.0, 0.01) == pytest.approx(0.05, abs=0.001)


def test_walk_forward_never_trains_on_later_seasons():
    home_features, away_features = build_feature_lists([5])
    rng = np.random.default_rng(0)
    rows = []
    for season, goal_level in [(2223, 1), (2324, 1), (2425, 6)]:
        for i in range(30):
            hf, af = rng.uniform(0.5, 2.5, size=2)
            rows.append({
                "date": pd.Timestamp(f"20{str(season)[:2]}-09-01") + pd.Timedelta(days=i),
                "season": season, "home_team": "A", "away_team": "B",
                home_features[0]: hf, home_features[1]: af, away_features[0]: af, away_features[1]: hf,
                "home_goals": goal_level + i % 3, "away_goals": i % 2, "result": "H",
            })
    df = pd.DataFrame(rows)
    with_future = walk_forward_poisson(df, home_features, away_features, [2324], max_goals=10)
    without_future = walk_forward_poisson(df[df["season"] < 2425], home_features, away_features,
                                          [2324], max_goals=10)
    # The 2425 season (very different goal levels) must not change 2324's predictions.
    pd.testing.assert_frame_equal(with_future, without_future)
    assert set(with_future["season"]) == {2324}
