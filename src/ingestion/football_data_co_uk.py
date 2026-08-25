"""
Bronze-layer ingestion: football-data.co.uk match results.

football-data.co.uk publishes free CSVs of match results (+ basic stats and
betting odds) for most major European leagues, no signup or API key needed.

This script downloads raw CSVs for the leagues/seasons listed in
config/config.yaml and stores them UNCHANGED in data/bronze/matches/.
No cleaning, renaming, or transformation happens here -- that's the Silver
layer's job. Bronze = "as close to the source as possible".

Usage:
    python -m src.ingestion.football_data_co_uk
    python -m src.ingestion.football_data_co_uk --league E0 --season 2425
"""

import argparse
import sys
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def download_one(base_url: str, league: str, season: str, out_dir: Path) -> Path:
    """Download a single league/season CSV. Returns the saved file path."""
    url = f"{base_url}/{season}/{league}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{league}_{season}.csv"

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or len(resp.content) < 100:
        print(f"  [skip] {url} -> HTTP {resp.status_code} (not available yet, e.g. future season)")
        return None

    out_path.write_bytes(resp.content)
    print(f"  [ok]   {url} -> {out_path.relative_to(PROJECT_ROOT)} ({len(resp.content)} bytes)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download raw match CSVs into the Bronze layer.")
    parser.add_argument("--league", help="Single league code, e.g. E0 (overrides config)")
    parser.add_argument("--season", help="Single season code, e.g. 2425 (overrides config)")
    args = parser.parse_args()

    cfg = load_config()
    base_url = cfg["data_sources"]["football_data_co_uk"]["base_url"]
    out_dir = PROJECT_ROOT / cfg["paths"]["bronze_matches"]

    leagues = [args.league] if args.league else cfg["active"]["leagues"]
    seasons = [args.season] if args.season else cfg["active"]["seasons"]

    print(f"Downloading {len(leagues)} league(s) x {len(seasons)} season(s) "
          f"into {out_dir.relative_to(PROJECT_ROOT)}/")

    saved = []
    for league in leagues:
        for season in seasons:
            path = download_one(base_url, league, season, out_dir)
            if path:
                saved.append(path)

    print(f"\nDone. {len(saved)} file(s) saved.")
    if not saved:
        print("Nothing downloaded -- check your network/config, or that the season has started.")
        sys.exit(1)


if __name__ == "__main__":
    main()
