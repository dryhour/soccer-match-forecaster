"""
Silver-layer cleaning: matches.

Reads every raw CSV in data/bronze/matches/, standardizes it, and writes a
single combined, de-duplicated table to data/silver/matches_clean/matches.csv

Standardization performed:
  - Keep only the columns we actually need (raw files also carry betting odds
    from many bookmakers -- dropped here, they're still in Bronze if needed).
  - Rename to consistent snake_case column names.
  - Parse dates to real datetime (source format varies: dd/mm/yy vs dd/mm/yyyy).
  - Standardize team names via a manual alias map (source names sometimes
    drift year to year, e.g. "Man United" vs "Manchester United").
  - Add a stable match_id.
  - Drop duplicate/incomplete rows.
  - Compute basic derived stats: total goals, goal difference, result label.

Usage:
    python -m src.cleaning.clean_matches
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Raw column -> clean column. Extend as you ingest more stat columns.
COLUMN_MAP = {
    "Div": "league",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",       # H / D / A
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "Referee": "referee",
}

REQUIRED_COLS = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]

# Team name standardization. Add entries as you discover inconsistencies
# across seasons/sources. Keys are lowercased for lookup.
TEAM_ALIASES = {
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers",
    "newcastle": "Newcastle United",
    "west brom": "West Bromwich Albion",
    "west ham": "West Ham United",
    "leicester": "Leicester City",
    "nott'm forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "sheffield utd": "Sheffield United",
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def standardize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    return TEAM_ALIASES.get(key, name.strip())


def clean_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    present_cols = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(present_cols.keys())].rename(columns=present_cols)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  [skip] {path.name}: missing required columns {missing}")
        return pd.DataFrame()

    # Dates in these files can be dd/mm/yy or dd/mm/yyyy depending on season.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    df["home_team"] = df["home_team"].apply(standardize_team)
    df["away_team"] = df["away_team"].apply(standardize_team)

    df = df.dropna(subset=REQUIRED_COLS)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # source filename encodes league + season, e.g. E0_2425.csv
    stem_parts = path.stem.split("_")
    df["season"] = stem_parts[1] if len(stem_parts) > 1 else "unknown"
    df["source_file"] = path.name

    return df


def add_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["result_label"] = df["result"].map({"H": "Home Win", "D": "Draw", "A": "Away Win"})
    df["match_id"] = (
        df["date"].dt.strftime("%Y%m%d") + "_" +
        df["home_team"].str.replace(" ", "") + "_" +
        df["away_team"].str.replace(" ", "")
    )
    return df


def main():
    cfg = load_config()
    bronze_dir = PROJECT_ROOT / cfg["paths"]["bronze_matches"]
    silver_dir = PROJECT_ROOT / cfg["paths"]["silver_matches"]
    silver_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(bronze_dir.glob("*.csv"))
    if not raw_files:
        print(f"No raw files found in {bronze_dir.relative_to(PROJECT_ROOT)}/. "
              f"Run ingestion first: python -m src.ingestion.football_data_co_uk")
        return

    print(f"Cleaning {len(raw_files)} raw file(s)...")
    frames = [clean_one_file(f) for f in raw_files]
    frames = [f for f in frames if not f.empty]

    if not frames:
        print("No valid rows after cleaning.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = add_derived_stats(combined)
    combined = combined.drop_duplicates(subset=["match_id"]).sort_values("date")

    out_path = silver_dir / "matches.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} clean matches -> {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"Teams: {combined['home_team'].nunique()} unique home-team names")


if __name__ == "__main__":
    main()
