"""
Unit tests for the final evaluation (penalty chosen inside the training
seasons, holdout scored once, nested walk-forward).
Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.backtest import paired_bootstrap
from src.evaluation.final_evaluation import align_on_matches, base_rate_by_season, nested_walk_forward, select_alpha
from src.models.poisson_model import build_feature_lists

HOME, AWAY = build_feature_lists([5])
GRID = [0.0, 0.1, 10.0]
FINAL_CFG = {"standardize": True, "alpha_grid": GRID, "validation_seasons": [2324, 2425]}


def make_seasons(seasons=(2223, 2324, 2425, 2526), n=40, seed=0):
    """Synthetic match_features rows whose goals really depend on the features."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(n):
            hf, ha, af, aa = rng.uniform(0.5, 2.5, size=4)
            hg, ag = rng.poisson(0.6 * hf * aa / 1.5), rng.poisson(0.5 * af * ha / 1.5)
            rows.append({
                "match_id": f"{season}-{i}", "season": season,
                "date": pd.Timestamp(f"20{str(season)[:2]}-08-01") + pd.Timedelta(days=i),
                "home_team": f"T{i % 6}", "away_team": f"T{(i + 1) % 6}",
                HOME[0]: hf, HOME[1]: aa, AWAY[0]: af, AWAY[1]: ha,
                "home_goals": hg, "away_goals": ag, "result": "H" if hg > ag else "D" if hg == ag else "A",
            })
    return pd.DataFrame(rows)


def test_select_alpha_never_sees_the_test_season_or_later():
    df = make_seasons()
    alpha, table = select_alpha(df, HOME, AWAY, 2526, [2324, 2425], GRID, max_goals=8)
    # Rewrite the test season with nonsense goals: nothing about the choice may move.
    scrambled = df.copy()
    test = scrambled["season"] == 2526
    scrambled.loc[test, "home_goals"] = 9
    alpha2, table2 = select_alpha(scrambled, HOME, AWAY, 2526, [2324, 2425], GRID, max_goals=8)
    assert alpha == alpha2
    pd.testing.assert_frame_equal(table, table2)
    assert set(table["validation_seasons"]) == {"2324+2425"}
    assert table.loc[table["chosen"], "brier_score"].iloc[0] == table["brier_score"].min()


def test_select_alpha_skips_validation_seasons_without_earlier_training_data():
    df = make_seasons(seasons=(2324, 2425, 2526))
    _, table = select_alpha(df, HOME, AWAY, 2526, [2324, 2425], GRID, max_goals=8)
    assert set(table["validation_seasons"]) == {"2425"}  # 2324 has nothing before it
    with pytest.raises(ValueError):
        select_alpha(df, HOME, AWAY, 2425, [2324, 2425], GRID, max_goals=8)


def test_nested_walk_forward_never_uses_later_seasons():
    df = make_seasons()
    sets = {"base": (HOME, AWAY)}
    preds, _ = nested_walk_forward(df, sets, [2425], FINAL_CFG, max_goals=8)
    preds_without_future, _ = nested_walk_forward(df[df["season"] <= 2425], sets, [2425], FINAL_CFG, max_goals=8)
    pd.testing.assert_frame_equal(preds["base"], preds_without_future["base"])
    assert set(preds["base"]["season"]) == {2425}


def test_align_on_matches_pairs_shared_matches_across_date_formats():
    py = pd.DataFrame({"match_date": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-03"]),
                       "home_team": ["C", "A", "E"], "away_team": ["D", "B", "F"],
                       "home_win_prob": [0.5, 0.4, 0.3], "draw_prob": [0.25, 0.3, 0.3],
                       "away_win_prob": [0.25, 0.3, 0.4], "actual_result": ["H", "D", "A"]})
    r = pd.DataFrame({"match_date": ["2026-01-01", "2026-01-02"],  # R skipped E-F
                      "home_team": ["A", "C"], "away_team": ["B", "D"],
                      "home_win_prob": [0.6, 0.2], "draw_prob": [0.2, 0.3],
                      "away_win_prob": [0.2, 0.5], "actual_result": ["D", "H"]})
    a, b = align_on_matches(py, r)
    assert a[["home_team", "away_team"]].values.tolist() == [["A", "B"], ["C", "D"]]
    assert a["home_win_prob"].tolist() == [0.4, 0.5] and b["home_win_prob"].tolist() == [0.6, 0.2]
    paired_bootstrap(a, b, n_boot=20)  # accepted as the same matches in the same order


def test_base_rate_by_season_uses_only_earlier_seasons():
    matches = pd.DataFrame({"season": [2223, 2223, 2324, 2324],
                            "result": ["H", "H", "A", "A"],
                            "home_goals": [1, 1, 0, 0], "away_goals": [0, 0, 1, 1]})
    template = pd.DataFrame({"season": [2324, 2425], "match_date": ["2024-01-01", "2025-01-01"],
                             "home_team": ["A", "A"], "away_team": ["B", "B"],
                             "actual_home_goals": [0, 0], "actual_away_goals": [1, 1],
                             "actual_result": ["A", "A"]})
    base = base_rate_by_season(template, matches)
    # 2324 sees only 2223 (all home wins); 2425 sees both seasons (half and half).
    assert base["home_win_prob"].tolist() == [1.0, 0.5]
    assert base["away_win_prob"].tolist() == [0.0, 0.5]


def test_paired_bootstrap_wider_level_gives_wider_interval():
    rng = np.random.default_rng(1)
    n = 80
    base = pd.DataFrame({"match_date": pd.date_range("2026-01-01", periods=n), "home_team": "A", "away_team": "B",
                         "home_win_prob": 0.45, "draw_prob": 0.25, "away_win_prob": 0.30,
                         "actual_result": rng.choice(list("HDA"), size=n)})
    chal = base.copy()
    chal["home_win_prob"] = rng.uniform(0.3, 0.6, size=n)
    chal["away_win_prob"] = 0.75 - chal["home_win_prob"]
    ci95 = paired_bootstrap(base, chal, n_boot=500)["brier_diff"]["ci95"]
    ci99 = paired_bootstrap(base, chal, n_boot=500, level=0.99)["brier_diff"]["ci95"]
    assert ci99[0] < ci95[0] and ci99[1] > ci95[1]
