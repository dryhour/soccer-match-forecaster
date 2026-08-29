"""
Unit tests for upcoming-fixtures cleaning.
Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.cleaning.clean_fixtures as clean_fixtures


def setup_project(tmp_path, monkeypatch, fixtures_csv_text, matches_rows):
    (tmp_path / "data" / "bronze" / "fixtures").mkdir(parents=True)
    (tmp_path / "data" / "silver" / "matches_clean").mkdir(parents=True)
    (tmp_path / "data" / "silver" / "fixtures_clean").mkdir(parents=True)

    (tmp_path / "data" / "bronze" / "fixtures" / "fixtures.csv").write_text(
        fixtures_csv_text, encoding="utf-8-sig"
    )
    pd.DataFrame(matches_rows, columns=["home_team", "away_team"]).to_csv(
        tmp_path / "data" / "silver" / "matches_clean" / "matches.csv", index=False
    )

    cfg = {
        "active": {"leagues": ["E0"]},
        "paths": {
            "bronze_fixtures": "data/bronze/fixtures",
            "silver_fixtures": "data/silver/fixtures_clean",
            "silver_matches": "data/silver/matches_clean",
        },
    }
    monkeypatch.setattr(clean_fixtures, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(clean_fixtures, "load_config", lambda: cfg)


def test_filters_to_active_league_and_standardizes_names(tmp_path, monkeypatch):
    fixtures_csv = (
        "Div,Date,Time,HomeTeam,AwayTeam,Referee\n"
        "E0,28/08/2026,20:00,Crystal Palace,Man City,A Madley\n"
        "B1,29/08/2026,15:00,Genk,Beveren,\n"
    )
    matches = [("Crystal Palace", "Manchester City"), ("Manchester City", "Crystal Palace")]
    setup_project(tmp_path, monkeypatch, fixtures_csv, matches)

    clean_fixtures.main()

    out = pd.read_csv(tmp_path / "data" / "silver" / "fixtures_clean" / "fixtures.csv")
    assert len(out) == 1
    assert out.iloc[0]["home_team"] == "Crystal Palace"
    assert out.iloc[0]["away_team"] == "Manchester City"  # "Man City" standardized


def test_drops_rows_with_unrecognized_teams(tmp_path, monkeypatch):
    fixtures_csv = (
        "Div,Date,Time,HomeTeam,AwayTeam,Referee\n"
        "E0,29/08/2026,15:00,Coventry,Hull,J Smith\n"
        "E0,28/08/2026,20:00,Crystal Palace,Man City,A Madley\n"
    )
    matches = [("Crystal Palace", "Manchester City")]
    setup_project(tmp_path, monkeypatch, fixtures_csv, matches)

    clean_fixtures.main()

    out = pd.read_csv(tmp_path / "data" / "silver" / "fixtures_clean" / "fixtures.csv")
    assert len(out) == 1
    assert out.iloc[0]["home_team"] == "Crystal Palace"
