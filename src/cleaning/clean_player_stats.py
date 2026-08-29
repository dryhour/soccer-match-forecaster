"""
Silver-layer cleaning: player appearance/goal stats.

Reads every raw Wikipedia article HTML in data/bronze/players/, parses each
club-season article's player statistics, and writes a single combined table
to data/silver/players_clean/player_stats.csv -- one row per (team, season,
player).

Wikipedia season articles present this in two different layouts depending
on the club/editor:
  1. Separate tables under "Appearances" and "Goals" headings, each with a
     "Premier League" column (e.g. "35+2" meaning 35 starts + 2 sub apps).
  2. ONE combined table under an "Appearances and goals" heading, with a
     two-level header: competition (e.g. "Premier League") x stat
     ("Apps"/"Goals").
Both are handled; a file matching neither is skipped rather than crashing
the whole run, since Wikipedia's exact formatting isn't guaranteed to stay
consistent across ~100 club-season articles written by different editors.

Usage:
    python -m src.cleaning.clean_player_stats
"""

import io
import re
from pathlib import Path

import pandas as pd
import yaml
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

FILENAME_RE = re.compile(r"^(.*)_(\d{4})$")
OUTPUT_COLUMNS = ["team", "season", "player", "position", "league_starts", "league_subs", "league_apps", "league_goals"]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_table_after_heading(soup: BeautifulSoup, heading_id: str):
    anchor = soup.find(id=heading_id)
    if anchor is None:
        return None
    heading = anchor.find_parent(["h2", "h3", "h4"]) or anchor
    return heading.find_next("table")


def read_table(table) -> pd.DataFrame:
    return pd.read_html(io.StringIO(str(table)))[0]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def drop_divider_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Some tables include a section-divider row (e.g. "Players who departed
    the club on loan..." or "...permanently..."), rendered by pandas as one
    long string repeated across every column since the source <td> spans the
    whole row. A real player row never has every cell identical, so that's a
    more robust signal than any specific column name (which varies -- "No."
    vs "Squad number" -- across articles)."""
    return df[df.apply(lambda row: row.astype(str).nunique() > 1, axis=1)]


def parse_apps_value(value) -> tuple:
    """'35+2' -> (35, 2) starts/subs. '6(20)' -> (6, 20). Plain '38' ->
    (38, 0). Blank/dash/anything unrecognized -> (0, 0)."""
    value = str(value).strip()
    if not value or value in ("-", "–", "—", "nan"):
        return 0, 0
    m = re.match(r"^(\d+)\s*[+(]\s*(\d+)\)?$", value)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)$", value)
    if m:
        return int(m.group(1)), 0
    return 0, 0


def clean_player_name(name) -> str:
    """Strips trailing footnote markers Wikipedia uses on player names, e.g.
    'Bukayo Saka*' or 'Ethan Nwaneri†' -> the plain name."""
    return re.sub(r"[*†‡]+$", "", str(name)).strip()


def clean_separate_tables(soup: BeautifulSoup) -> pd.DataFrame:
    """Layout 1: separate 'Appearances' and 'Goals' tables."""
    apps_table = find_table_after_heading(soup, "Appearances")
    if apps_table is None:
        return pd.DataFrame()

    apps_df = flatten_columns(read_table(apps_table))
    if "Player" not in apps_df.columns or "Premier League" not in apps_df.columns:
        return pd.DataFrame()

    apps_df = drop_divider_rows(apps_df)
    apps_df["player"] = apps_df["Player"].apply(clean_player_name)
    starts_subs = apps_df["Premier League"].apply(parse_apps_value)
    apps_df["league_starts"] = starts_subs.apply(lambda t: t[0])
    apps_df["league_subs"] = starts_subs.apply(lambda t: t[1])
    apps_df["league_apps"] = apps_df["league_starts"] + apps_df["league_subs"]
    apps_df["position"] = apps_df["Pos."] if "Pos." in apps_df.columns else None

    result = apps_df[["player", "position", "league_starts", "league_subs", "league_apps"]].copy()
    result = result.drop_duplicates(subset="player")

    goals_table = find_table_after_heading(soup, "Goals")
    if goals_table is not None:
        goals_df = flatten_columns(read_table(goals_table))
        if "Player" in goals_df.columns and "Premier League" in goals_df.columns:
            goals_df = drop_divider_rows(goals_df)
            goals_df["player"] = goals_df["Player"].apply(clean_player_name)
            goals_df["league_goals"] = pd.to_numeric(goals_df["Premier League"], errors="coerce").fillna(0).astype(int)
            goals_df = goals_df.drop_duplicates(subset="player")
            result = result.merge(goals_df[["player", "league_goals"]], on="player", how="left")

    if "league_goals" not in result.columns:
        result["league_goals"] = 0
    result["league_goals"] = result["league_goals"].fillna(0).astype(int)
    return result


def find_stat_column(df: pd.DataFrame, top: str, bottom: str):
    for col in df.columns:
        if isinstance(col, tuple) and col[0] == top and col[-1] == bottom:
            return col
    return None


COMBINED_TABLE_HEADINGS = ["Appearances_and_goals", "Appearances", "Squad_statistics"]
PLAYER_COLUMN_NAMES = ["Player", "Name"]
LEAGUE_COLUMN_NAMES = ["Premier League", "League"]


def clean_combined_table(soup: BeautifulSoup) -> pd.DataFrame:
    """Layout 2: one table with a two-level header (competition x
    Apps/Goals). The table shape is what determines this parser, not the
    heading name or exact column labels -- both vary across articles (e.g.
    heading 'Squad_statistics' with columns 'Name'/'League' instead of
    'Player'/'Premier League'), so a few aliases of each are tried."""
    table = None
    for heading_id in COMBINED_TABLE_HEADINGS:
        table = find_table_after_heading(soup, heading_id)
        if table is not None:
            break
    if table is None:
        return pd.DataFrame()

    df = read_table(table)
    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()

    player_col = next((c for name in PLAYER_COLUMN_NAMES if (c := find_stat_column(df, name, name)) is not None), None)
    apps_col = next(
        (c for name in LEAGUE_COLUMN_NAMES if (c := find_stat_column(df, name, "Apps")) is not None), None
    )
    goals_col = next(
        (c for name in LEAGUE_COLUMN_NAMES if (c := find_stat_column(df, name, "Goals")) is not None), None
    )
    if player_col is None or apps_col is None:
        return pd.DataFrame()

    pos_col = find_stat_column(df, "Pos", "Pos") or find_stat_column(df, "Pos.", "Pos.")

    df = drop_divider_rows(df)

    result = pd.DataFrame()
    result["player"] = df[player_col].apply(clean_player_name)
    result["position"] = df[pos_col] if pos_col is not None else None
    starts_subs = df[apps_col].apply(parse_apps_value)
    result["league_starts"] = starts_subs.apply(lambda t: t[0])
    result["league_subs"] = starts_subs.apply(lambda t: t[1])
    result["league_apps"] = result["league_starts"] + result["league_subs"]
    result["league_goals"] = (
        pd.to_numeric(df[goals_col], errors="coerce").fillna(0).astype(int) if goals_col is not None else 0
    )
    return result.drop_duplicates(subset="player")


def clean_icon_header_table(soup: BeautifulSoup) -> pd.DataFrame:
    """Layout 3: a single table straight under the 'Statistics' heading (no
    'Appearances'/'Goals' subheading at all), where each competition spans 4
    icon-only sub-columns in a fixed order: shirt=apps, ball=goals, yellow
    card, red card. No separate starts/subs breakdown is available here."""
    table = find_table_after_heading(soup, "Statistics")
    if table is None:
        return pd.DataFrame()

    df = read_table(table)
    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()

    pl_cols = [c for c in df.columns if c[0] == "Premier League"]
    player_col = find_stat_column(df, "Player", "Player")
    if len(pl_cols) < 2 or player_col is None:
        return pd.DataFrame()
    apps_col, goals_col = pl_cols[0], pl_cols[1]

    pos_col = find_stat_column(df, "Pos.", "Pos.") or find_stat_column(df, "Pos", "Pos")

    df = drop_divider_rows(df)

    result = pd.DataFrame()
    result["player"] = df[player_col].apply(clean_player_name)
    result["position"] = df[pos_col] if pos_col is not None else None
    result["league_apps"] = pd.to_numeric(df[apps_col], errors="coerce").fillna(0).astype(int)
    result["league_starts"] = result["league_apps"]  # no starts/subs split in this layout
    result["league_subs"] = 0
    result["league_goals"] = pd.to_numeric(df[goals_col], errors="coerce").fillna(0).astype(int)
    return result.drop_duplicates(subset="player")


def clean_one_file(path: Path, team: str, season: str) -> pd.DataFrame:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    result = clean_separate_tables(soup)
    if result.empty:
        result = clean_combined_table(soup)
    if result.empty:
        result = clean_icon_header_table(soup)
    if result.empty:
        print(f"  [skip] {path.name}: no recognizable appearances table found")
        return pd.DataFrame()

    result["team"] = team
    result["season"] = season
    return result[OUTPUT_COLUMNS]


def main():
    cfg = load_config()
    bronze_dir = PROJECT_ROOT / cfg["paths"]["bronze_players"]
    silver_dir = PROJECT_ROOT / cfg["paths"]["silver_players"]
    silver_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(bronze_dir.glob("*.html"))
    if not raw_files:
        print(f"No raw files found in {bronze_dir.relative_to(PROJECT_ROOT)}/. "
              f"Run ingestion first: python -m src.ingestion.wikipedia_player_stats")
        return

    print(f"Cleaning {len(raw_files)} raw file(s)...")
    frames = []
    for path in raw_files:
        m = FILENAME_RE.match(path.stem)
        if not m:
            print(f"  [skip] {path.name}: unrecognized filename pattern")
            continue
        team, season = m.group(1).replace("_", " "), m.group(2)
        try:
            frame = clean_one_file(path, team, season)
        except Exception as e:
            print(f"  [skip] {path.name}: unexpected parse error ({e})")
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        print("No valid rows after cleaning.")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_path = silver_dir / "player_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} player-season rows -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
