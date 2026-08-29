"""
Silver-layer cleaning: upcoming fixtures.

Reads data/bronze/fixtures/fixtures.csv (a combined, multi-league file --
see src/ingestion/football_data_co_uk.py), filters to the active league(s),
standardizes team names with the same alias map used for historical
results, and writes data/silver/fixtures_clean/fixtures.csv.

The source file's "Div" column is occasionally unreliable (a handful of
rows have been observed with the wrong league code), so rows are also
cross-checked against the set of teams that actually appear in our cleaned
historical match data -- a row naming a team we've never seen in the
league is dropped rather than trusted.

Usage:
    python -m src.cleaning.clean_fixtures
"""

from pathlib import Path

import pandas as pd
import yaml

from src.cleaning.clean_matches import standardize_team

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

COLUMN_MAP = {
    "Div": "league",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Referee": "referee",
}
REQUIRED_COLS = ["date", "home_team", "away_team"]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    bronze_path = PROJECT_ROOT / cfg["paths"]["bronze_fixtures"] / "fixtures.csv"
    silver_dir = PROJECT_ROOT / cfg["paths"]["silver_fixtures"]
    matches_path = PROJECT_ROOT / cfg["paths"]["silver_matches"] / "matches.csv"
    silver_dir.mkdir(parents=True, exist_ok=True)

    if not bronze_path.exists():
        print(f"Missing {bronze_path.relative_to(PROJECT_ROOT)}. "
              f"Run ingestion first: python -m src.ingestion.football_data_co_uk")
        return
    if not matches_path.exists():
        print(f"Missing {matches_path.relative_to(PROJECT_ROOT)}. "
              f"Run cleaning first: python -m src.cleaning.clean_matches")
        return

    df = pd.read_csv(bronze_path, encoding="utf-8-sig")
    present_cols = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(present_cols.keys())].rename(columns=present_cols)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"Missing required columns {missing} in fixtures.csv.")
        return

    active_leagues = cfg["active"]["leagues"]
    df = df[df["league"].isin(active_leagues)].copy()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["home_team"] = df["home_team"].apply(standardize_team)
    df["away_team"] = df["away_team"].apply(standardize_team)
    df = df.dropna(subset=REQUIRED_COLS)

    known_teams = set(pd.read_csv(matches_path, usecols=["home_team", "away_team"]).values.ravel())
    before = len(df)
    df = df[df["home_team"].isin(known_teams) & df["away_team"].isin(known_teams)]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) naming a team not seen in the historical league data "
              f"(likely a mislabeled 'Div' in the source).")

    df = df.sort_values("date")
    out_path = silver_dir / "fixtures.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} upcoming fixture(s) -> {out_path.relative_to(PROJECT_ROOT)}")
    if not df.empty:
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    main()
