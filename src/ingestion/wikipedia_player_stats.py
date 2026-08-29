"""
Bronze-layer ingestion: player appearance/goal data from Wikipedia club-season
articles (e.g. "2023-24 Arsenal F.C. season").

Uses Wikipedia's MediaWiki API (action=parse) rather than scraping the raw
page -- a documented, stable contract that doesn't require rendering JS.
Saves the parsed article HTML UNCHANGED into data/bronze/players/, one file
per (team, season). No cleaning happens here -- that's the Silver layer's
job (see src/cleaning/clean_player_stats.py).

Rate limits: Wikipedia's anonymous API allows a modest request rate. This
script sleeps between requests and backs off on HTTP 429, and skips files
that already exist so re-runs only fetch what's missing. See
https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits

Usage:
    python -m src.ingestion.wikipedia_player_stats
    python -m src.ingestion.wikipedia_player_stats --team Arsenal --season 2526
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "soccer-match-forecaster/1.0 "
    "(personal research project; https://github.com/dryhour/soccer-match-forecaster)"
)
REQUEST_DELAY_SECONDS = 1.5

# Team name (as used in match data) -> full Wikipedia club name, used to
# build article titles like "2023-24 Arsenal F.C. season". Extend as new
# teams appear in the match data. Verified against Wikipedia for teams in
# the 2025-26 season; older/relegated teams are best-effort standard names
# and may need a fix here if a title has moved.
TEAM_WIKI_NAME = {
    "Arsenal": "Arsenal F.C.",
    "Aston Villa": "Aston Villa F.C.",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford F.C.",
    "Brighton": "Brighton & Hove Albion F.C.",
    "Burnley": "Burnley F.C.",
    "Chelsea": "Chelsea F.C.",
    "Crystal Palace": "Crystal Palace F.C.",
    "Everton": "Everton F.C.",
    "Fulham": "Fulham F.C.",
    "Ipswich": "Ipswich Town F.C.",
    "Leeds": "Leeds United F.C.",
    "Leicester City": "Leicester City F.C.",
    "Liverpool": "Liverpool F.C.",
    "Luton": "Luton Town F.C.",
    "Manchester City": "Manchester City F.C.",
    "Manchester United": "Manchester United F.C.",
    "Newcastle United": "Newcastle United F.C.",
    "Nottingham Forest": "Nottingham Forest F.C.",
    "Sheffield United": "Sheffield United F.C.",
    "Southampton": "Southampton F.C.",
    "Sunderland": "Sunderland A.F.C.",
    "Tottenham Hotspur": "Tottenham Hotspur F.C.",
    "West Ham United": "West Ham United F.C.",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers F.C.",
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def season_wiki_range(season_code: str) -> str:
    """'2526' -> '2025–26' (en dash), Wikipedia's article-title convention."""
    return f"20{season_code[:2]}–{season_code[2:]}"


def article_title(team: str, season_code: str) -> str:
    wiki_name = TEAM_WIKI_NAME.get(team)
    if wiki_name is None:
        raise ValueError(f"No Wikipedia club name mapped for team '{team}'. Add it to TEAM_WIKI_NAME.")
    return f"{season_wiki_range(season_code)} {wiki_name} season"


def fetch_article_html(title: str):
    """Returns the parsed article HTML, or None if the page doesn't exist."""
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2}
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(5):
        resp = requests.get(API_URL, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    [rate limited] waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None  # e.g. missingtitle
        return data["parse"]["text"]

    print(f"    [error] gave up on '{title}' after repeated rate limiting")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download player stats from Wikipedia club-season articles into the Bronze layer."
    )
    parser.add_argument("--team", help="Single team name, e.g. 'Arsenal' (default: every team in the season)")
    parser.add_argument("--season", help="Single season code, e.g. 2526 (default: every season in the match data)")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["paths"]["bronze_players"]
    out_dir.mkdir(parents=True, exist_ok=True)

    matches_path = PROJECT_ROOT / cfg["paths"]["silver_matches"] / "matches.csv"
    if not matches_path.exists():
        print(f"Missing {matches_path.relative_to(PROJECT_ROOT)}. "
              f"Run cleaning first: python -m src.cleaning.clean_matches")
        return
    matches = pd.read_csv(matches_path, dtype={"season": str})

    seasons = [args.season] if args.season else sorted(matches["season"].unique())

    saved, skipped = [], []
    for season in seasons:
        season_matches = matches[matches["season"] == season]
        teams = (
            [args.team] if args.team
            else sorted(set(season_matches["home_team"]) | set(season_matches["away_team"]))
        )
        for team in teams:
            out_path = out_dir / f"{team.replace(' ', '_')}_{season}.html"
            if out_path.exists():
                continue  # idempotent -- only fetch what's missing

            try:
                title = article_title(team, season)
            except ValueError as e:
                print(f"  [skip] {e}")
                skipped.append(team)
                continue

            print(f"  fetching {title} ...")
            html = fetch_article_html(title)
            time.sleep(REQUEST_DELAY_SECONDS)
            if html is None:
                print(f"    [skip] article not found: '{title}'")
                skipped.append(f"{team} {season}")
                continue

            out_path.write_text(html, encoding="utf-8")
            print(f"    [ok] -> {out_path.relative_to(PROJECT_ROOT)}")
            saved.append(out_path)

    print(f"\nDone. {len(saved)} article(s) saved, {len(skipped)} skipped.")
    if skipped:
        print("Skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
