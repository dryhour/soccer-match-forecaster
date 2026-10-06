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

    # Expanding mean over the team's WHOLE history in the data (shifted),
    # blended across home and away. Despite the "season_" prefix it is NOT
    # reset at season boundaries -- grouped by team only.
    long_df["season_avg_goals_for"] = grouped["goals_for"].apply(
        lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
    )
    long_df["season_avg_goals_against"] = grouped["goals_against"].apply(
        lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
    )

    return long_df


def add_venue_split_averages(long_df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """
    Average goals scored/conceded computed ONLY from a team's past matches
    at the SAME venue (home or away) as this row's own match, over its whole
    history in the data (expanding mean, not reset each season) -- e.g. a team's value here on a home-match row is its average
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


def add_sos_adjusted_form(long_df: pd.DataFrame, windows: list, min_matches: int) -> pd.DataFrame:
    """
    Strength-of-schedule-adjusted rolling form (2026-10-03 experiment).

    Each past match's goals are re-scored relative to how strong that
    opponent was GOING INTO that match:

        adj_goals_for     = goals_for     - (opp_avg_goals_against - league_avg)
        adj_goals_against = goals_against - (opp_avg_goals_for     - league_avg)

    so 3 goals against a leaky defence counts for less than 3 against a
    tight one. opp_avg_* is the opponent's expanding mean over all its
    matches strictly before this one; league_avg is the mean goals per team
    per match over all matches on strictly earlier dates. When the opponent
    has fewer than min_matches of history (e.g. newly promoted), it is
    treated as league-average, i.e. no adjustment.

    The rolling mean of the adjusted values is then shifted by 1 like
    add_rolling_form, so a match's own result never enters its own feature.
    """
    long_df = long_df.copy()
    grouped = long_df.groupby("team", group_keys=False)
    long_df["_pre_gf"] = grouped["goals_for"].apply(
        lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
    )
    long_df["_pre_ga"] = grouped["goals_against"].apply(
        lambda s: s.shift(1).expanding(min_periods=min_matches).mean()
    )

    # League average from strictly earlier dates only (no same-day matches).
    daily = long_df.groupby("date")["goals_for"].agg(["sum", "count"]).sort_index()
    prior = daily.cumsum().shift(1)
    league_avg = (prior["sum"] / prior["count"]).rename("_league_avg")
    long_df = long_df.merge(league_avg, left_on="date", right_index=True, how="left")

    opp = long_df[["match_id", "team", "_pre_gf", "_pre_ga"]].rename(columns={
        "team": "opponent", "_pre_gf": "_opp_pre_gf", "_pre_ga": "_opp_pre_ga",
    })
    long_df = long_df.merge(opp, on=["match_id", "opponent"], how="left")
    opp_ga = long_df["_opp_pre_ga"].fillna(long_df["_league_avg"])
    opp_gf = long_df["_opp_pre_gf"].fillna(long_df["_league_avg"])
    long_df["adj_goals_for"] = long_df["goals_for"] - (opp_ga - long_df["_league_avg"])
    long_df["adj_goals_against"] = long_df["goals_against"] - (opp_gf - long_df["_league_avg"])

    long_df = long_df.sort_values(["team", "date"])
    grouped = long_df.groupby("team", group_keys=False)
    for w in windows:
        long_df[f"avg_sos_goals_for_last{w}"] = grouped["adj_goals_for"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=min_matches).mean()
        )
        long_df[f"avg_sos_goals_against_last{w}"] = grouped["adj_goals_against"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=min_matches).mean()
        )

    return long_df.drop(columns=["_pre_gf", "_pre_ga", "_league_avg", "_opp_pre_gf", "_opp_pre_ga"])


MATCHES_PER_SEASON = 38


def previous_season(season: int) -> int:
    """football-data.co.uk season code of the season before: 2526 -> 2425."""
    return (season // 100 - 1) * 100 + (season % 100 - 1)


def build_squad_quality(player_stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Squad-quality feature (2026-10-10 experiment), one row per (team, season):

        squad_prior_goals_per_match = (sum over this season's roster of each
                                       player's PREVIOUS-season Premier League
                                       goals, at any club) / 38

    Previous season, because the Wikipedia data are end-of-season totals:
    any same-season player stat would include goals from the match being
    predicted and every match after it. Last season's totals are all known
    before this season starts, and summing over the CURRENT roster picks up
    transfers in both directions.

    A sum rather than a per-appearance rate, because a rate lets a player
    with 1 goal in 1 appearance dominate the team metric; in a sum his
    contribution is bounded by his (small) goal total, so no
    minimum-appearances floor is needed.

    Players with no PL goals last season -- including new arrivals from
    other leagues and most of a newly promoted squad -- contribute 0, so
    promoted teams score low by construction.

    Known look-ahead (roster composition only, not results): rosters come
    from end-of-season articles, so a January signing also counts toward
    the squad for that season's August-December matches.

    NaN -- not 0 -- when the value can't be computed: no parsed roster for
    the team-season, no player data at all for the previous season (e.g.
    2223, whose previous season was never ingested), or the team was in the
    PL last season but its own article for that season didn't parse.
    """
    pl_team_seasons = pd.concat([
        matches[["home_team", "season"]].rename(columns={"home_team": "team"}),
        matches[["away_team", "season"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates()
    # PL team-seasons only -- e.g. a parsed Championship article must not
    # count as "previous-season PL goals".
    stats = player_stats.merge(pl_team_seasons, on=["team", "season"], how="inner")

    prior_goals = (stats.groupby(["player", "season"], as_index=False)["league_goals"].sum()
                   .rename(columns={"season": "prev_season", "league_goals": "prior_goals"}))
    roster = stats[["team", "season", "player"]].drop_duplicates()
    roster["prev_season"] = roster["season"].map(previous_season)
    roster = roster.merge(prior_goals, on=["player", "prev_season"], how="left")
    roster["prior_goals"] = roster["prior_goals"].fillna(0)

    squad = roster.groupby(["team", "season", "prev_season"], as_index=False)["prior_goals"].sum()
    squad["squad_prior_goals_per_match"] = squad["prior_goals"] / MATCHES_PER_SEASON

    parsed = set(zip(stats["team"], stats["season"]))
    in_pl = set(zip(pl_team_seasons["team"], pl_team_seasons["season"]))
    no_prev_data = ~squad["prev_season"].isin(set(stats["season"]))
    own_prev_missing = pd.Series([(t, p) in in_pl and (t, p) not in parsed
                                  for t, p in zip(squad["team"], squad["prev_season"])], index=squad.index)
    squad.loc[no_prev_data | own_prev_missing, "squad_prior_goals_per_match"] = float("nan")
    return squad[["team", "season", "squad_prior_goals_per_match"]]


def build_prev_season_team_goals(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Control arm for the squad-quality experiment, one row per (team,
    season): the TEAM's own goals per match last season, from match results
    alone -- no player data, no roster, so no look-ahead of any kind.

    Teams not in the PL last season (promoted) get 0, mirroring the squad
    feature's ~0 for promoted squads, so the comparison isolates what the
    roster adds rather than just a "promoted" flag. NaN when the previous
    season isn't in the data at all (the earliest season).

    If squad_prior_goals_per_match beats the baseline but not this, the
    gain is a longer-horizon team-strength signal, not player information.
    """
    long_df = to_team_long(matches)
    season_gpm = long_df.groupby(["team", "season"])["goals_for"].mean()
    team_seasons = long_df[["team", "season"]].drop_duplicates()
    prev = team_seasons["season"].map(previous_season)
    team_seasons["prev_season_goals_per_match"] = [
        season_gpm.get((t, p), 0.0) for t, p in zip(team_seasons["team"], prev)
    ]
    team_seasons.loc[~prev.isin(set(long_df["season"])), "prev_season_goals_per_match"] = float("nan")
    return team_seasons


def add_squad_quality(long_df: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    """Attach a season-level (team, season) table -- squad quality or its
    control -- to every one of the team's match rows."""
    return long_df.merge(squad, on=["team", "season"], how="left")


def build_match_features(matches: pd.DataFrame, team_form: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Join home-team and away-team rolling form back onto each match."""
    form_cols = [c for c in team_form.columns if c.startswith(("avg_", "season_avg_", "squad_", "prev_season_"))]

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
    team_form = add_sos_adjusted_form(team_form, windows, min_matches)

    player_stats_path = PROJECT_ROOT / cfg["paths"]["silver_players"] / "player_stats.csv"
    if player_stats_path.exists():
        squad = build_squad_quality(pd.read_csv(player_stats_path), matches)
    else:
        print(f"Missing {player_stats_path.relative_to(PROJECT_ROOT)} -- squad-quality feature left blank. "
              f"Run: python -m src.cleaning.clean_player_stats")
        squad = pd.DataFrame({"team": pd.Series(dtype=str), "season": pd.Series(dtype=matches["season"].dtype),
                              "squad_prior_goals_per_match": pd.Series(dtype=float)})
    team_form = add_squad_quality(team_form, squad)
    team_form = add_squad_quality(team_form, build_prev_season_team_goals(matches))

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
