"""
Gold-layer feature engineering: team & match features.

Reads data/silver/matches_clean/matches.csv and produces two Gold tables:

1. team_features/team_match_history.csv
   One row per (team, match) with that team's rolling form BEFORE the match
   (goals scored/conceded, points, win rate, home/away splits) over the
   windows configured in config.yaml -- e.g. last 5 and last 10 matches.
   Using "before this match" values (not including the match itself) avoids
   leaking the outcome into its own features.

2. match_features/match_features.csv
   One row per match, joining home-team and away-team rolling form together
   with the actual result -- this is the model-ready table.

Usage:
    python -m src.features.build_match_features
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def to_team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape one row-per-match into two rows-per-match (one per team),
    which makes rolling "form" stats much simpler to compute per team.
    """
    home = matches.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "goals_for", "away_goals": "goals_against",
    }).copy()
    home["is_home"] = 1
    home["points"] = home["result"].map({"H": 3, "D": 1, "A": 0})

    away = matches.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "goals_for", "home_goals": "goals_against",
    }).copy()
    away["is_home"] = 0
    away["points"] = away["result"].map({"A": 3, "D": 1, "H": 0})

    keep = ["match_id", "date", "season", "team", "opponent",
            "goals_for", "goals_against", "is_home", "points"]
    long_df = pd.concat([home[keep], away[keep]], ignore_index=True)
    return long_df.sort_values(["team", "date"])


def add_rolling_form(long_df: pd.DataFrame, windows: list, min_matches: int) -> pd.DataFrame:
    """
    For each team, compute rolling averages over the last N matches,
    SHIFTED by 1 so the current match's own result never leaks into its
    own feature row (features describe form entering the match).
    """
    long_df = long_df.copy()
    grouped = long_df.groupby("team", group_keys=False)

    for w in windows:
        long_df[f"avg_goals_for_last{w}"] = grouped["goals_for"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=min_matches).mean()
        )
        long_df[f"avg_goals_against_last{w}"] = grouped["goals_against"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=min_matches).mean()
        )
        long_df[f"avg_points_last{w}"] = grouped["points"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=min_matches).mean()
        )

    # Season-to-date average, blended across home and away matches
    # (expanding mean, shifted).
    long_df["season_avg_goals_for"] = grouped["goals_for"].apply(
        lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
    )
    long_df["season_avg_goals_against"] = grouped["goals_against"].apply(
        lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
    )

    return long_df


def add_venue_split_averages(long_df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """
    Season-to-date average goals scored/conceded computed ONLY from a
    team's past matches at the SAME venue (home or away) as this row's own
    match -- e.g. a team's value here on a home-match row is its average
    scoring/conceding rate across its own past home matches only, never
    blended with its away record. This is a different signal from
    season_avg_goals_for/against above (which blends both venues), since
    a team's home and away performance often differ a lot.

    Shifted by 1 WITHIN each venue-specific sequence (not within the
    team's full match sequence) so a match's own result never leaks into
    its own feature, and so a team's 3rd home game of the season looks back
    at its first 2 home games only -- not at away games played in between.
    """
    long_df = long_df.copy()
    long_df["avg_goals_for_by_venue"] = float("nan")
    long_df["avg_goals_against_by_venue"] = float("nan")

    for is_home_flag in (1, 0):
        mask = long_df["is_home"] == is_home_flag
        subset = long_df.loc[mask].sort_values(["team", "date"])
        grouped = subset.groupby("team", group_keys=False)
        long_df.loc[mask, "avg_goals_for_by_venue"] = grouped["goals_for"].apply(
            lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
        )
        long_df.loc[mask, "avg_goals_against_by_venue"] = grouped["goals_against"].apply(
            lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
        )

    return long_df


def build_match_features(matches: pd.DataFrame, team_form: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Join home-team and away-team rolling form back onto each match."""
    form_cols = [c for c in team_form.columns if c.startswith("avg_") or c.startswith("season_avg_")]

    home_form = team_form[team_form["is_home"] == 1][["match_id", "team"] + form_cols]
    home_form = home_form.rename(columns={c: f"home_{c}" for c in form_cols}).drop(columns="team")

    away_form = team_form[team_form["is_home"] == 0][["match_id", "team"] + form_cols]
    away_form = away_form.rename(columns={c: f"away_{c}" for c in form_cols}).drop(columns="team")

    feat = matches.merge(home_form, on="match_id", how="left")
    feat = feat.merge(away_form, on="match_id", how="left")

    # Simple derived matchup features
    w = windows[0]
    feat["form_diff_points"] = feat[f"home_avg_points_last{w}"] - feat[f"away_avg_points_last{w}"]
    feat["form_diff_goals_for"] = feat[f"home_avg_goals_for_last{w}"] - feat[f"away_avg_goals_for_last{w}"]

    return feat


def main():
    cfg = load_config()
    silver_matches_path = PROJECT_ROOT / cfg["paths"]["silver_matches"] / "matches.csv"
    team_out_dir = PROJECT_ROOT / cfg["paths"]["gold_team_features"]
    match_out_dir = PROJECT_ROOT / cfg["paths"]["gold_match_features"]
    team_out_dir.mkdir(parents=True, exist_ok=True)
    match_out_dir.mkdir(parents=True, exist_ok=True)

    if not silver_matches_path.exists():
        print(f"Missing {silver_matches_path.relative_to(PROJECT_ROOT)}. "
              f"Run cleaning first: python -m src.cleaning.clean_matches")
        return

    matches = pd.read_csv(silver_matches_path, parse_dates=["date"])
    windows = cfg["features"]["form_windows"]
    min_matches = cfg["features"]["min_matches_for_form"]

    long_df = to_team_long(matches)
    team_form = add_rolling_form(long_df, windows, min_matches)
    team_form = add_venue_split_averages(team_form, min_matches)

    team_out_path = team_out_dir / "team_match_history.csv"
    team_form.to_csv(team_out_path, index=False)
    print(f"Saved team-level rolling form -> {team_out_path.relative_to(PROJECT_ROOT)}")

    match_feat = build_match_features(matches, team_form, windows)
    match_out_path = match_out_dir / "match_features.csv"
    match_feat.to_csv(match_out_path, index=False)
    print(f"Saved match-level features -> {match_out_path.relative_to(PROJECT_ROOT)}")

    usable = match_feat.dropna(subset=[f"home_avg_points_last{windows[0]}", f"away_avg_points_last{windows[0]}"])
    print(f"{len(usable)}/{len(match_feat)} matches have enough history to be model-ready.")


if __name__ == "__main__":
    main()
