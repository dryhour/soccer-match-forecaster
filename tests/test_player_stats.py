"""
Unit tests for player-stats cleaning (Wikipedia club-season articles).
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning.clean_player_stats import clean_one_file, clean_player_name, parse_apps_value

SAMPLE_HTML = """
<html><body>
<h3><span class="mw-headline" id="Appearances">Appearances</span></h3>
<table class="wikitable">
<tr><th>No.</th><th>Pos.</th><th>Player</th><th>Premier League</th><th>Season total</th></tr>
<tr><td>1</td><td>GK</td><td>Test Keeper</td><td>38</td><td>40</td></tr>
<tr><td>9</td><td>FW</td><td>Test Striker*</td><td>20+5</td><td>28+6</td></tr>
<tr>
<td>Players who departed the club on loan</td>
<td>Players who departed the club on loan</td>
<td>Players who departed the club on loan</td>
<td>Players who departed the club on loan</td>
<td>Players who departed the club on loan</td>
</tr>
</table>
<h3><span class="mw-headline" id="Goals">Goals</span></h3>
<table class="wikitable">
<tr><th>No.</th><th>Pos.</th><th>Player</th><th>Premier League</th><th>Season total</th></tr>
<tr><td>9</td><td>FW</td><td>Test Striker*</td><td>15</td><td>18</td></tr>
</table>
</body></html>
"""


def test_parse_apps_value_plain_number():
    assert parse_apps_value("38") == (38, 0)


def test_parse_apps_value_starts_plus_subs():
    assert parse_apps_value("20+5") == (20, 5)


def test_parse_apps_value_blank_or_dash():
    assert parse_apps_value("") == (0, 0)
    assert parse_apps_value("-") == (0, 0)


def test_clean_player_name_strips_footnote_marker():
    assert clean_player_name("Test Striker*") == "Test Striker"
    assert clean_player_name("Test Keeper") == "Test Keeper"


def test_clean_one_file_end_to_end(tmp_path):
    path = tmp_path / "Test_Team_2324.html"
    path.write_text(SAMPLE_HTML, encoding="utf-8")

    df = clean_one_file(path, "Test Team", "2324")

    assert len(df) == 2  # divider row filtered out
    assert set(df["player"]) == {"Test Keeper", "Test Striker"}

    keeper = df[df["player"] == "Test Keeper"].iloc[0]
    assert keeper["league_starts"] == 38
    assert keeper["league_subs"] == 0
    assert keeper["league_apps"] == 38
    assert keeper["league_goals"] == 0  # not in the goals table

    striker = df[df["player"] == "Test Striker"].iloc[0]
    assert striker["league_starts"] == 20
    assert striker["league_subs"] == 5
    assert striker["league_apps"] == 25
    assert striker["league_goals"] == 15


COMBINED_TABLE_HTML = """
<html><body>
<h3><span class="mw-headline" id="Appearances_and_goals">Appearances and goals</span></h3>
<table class="wikitable">
<tr>
<th rowspan="2">No.</th><th rowspan="2">Pos</th><th rowspan="2">Player</th>
<th colspan="2">Total</th><th colspan="2">Premier League</th>
</tr>
<tr><th>Apps</th><th>Goals</th><th>Apps</th><th>Goals</th></tr>
<tr><td>1</td><td>GK</td><td>Combined Keeper</td><td>40</td><td>0</td><td>38</td><td>0</td></tr>
<tr><td>9</td><td>FW</td><td>Combined Striker†</td><td>28</td><td>18</td><td>20+5</td><td>15</td></tr>
</table>
</body></html>
"""


def test_clean_one_file_combined_table_format(tmp_path):
    path = tmp_path / "Combined_Team_2526.html"
    path.write_text(COMBINED_TABLE_HTML, encoding="utf-8")

    df = clean_one_file(path, "Combined Team", "2526")

    assert len(df) == 2
    striker = df[df["player"] == "Combined Striker"].iloc[0]
    assert striker["league_starts"] == 20
    assert striker["league_subs"] == 5
    assert striker["league_apps"] == 25
    assert striker["league_goals"] == 15

    keeper = df[df["player"] == "Combined Keeper"].iloc[0]
    assert keeper["league_apps"] == 38
    assert keeper["league_goals"] == 0


def test_clean_one_file_missing_appearances_table_returns_empty(tmp_path):
    path = tmp_path / "No_Stats_Team_2324.html"
    path.write_text("<html><body><p>No stats here.</p></body></html>", encoding="utf-8")

    df = clean_one_file(path, "No Stats Team", "2324")
    assert df.empty


# Header row 3 is a positional-group label ("Goalkeepers") that pandas reads
# as a third header level -- the column lookup must still find League/Apps.
THREE_LEVEL_HEADER_HTML = """
<html><body>
<h3><span class="mw-headline" id="Appearances_and_goals">Appearances and goals</span></h3>
<table class="wikitable">
<tr>
<th rowspan="2">No.</th><th rowspan="2">Pos</th><th rowspan="2">Player</th>
<th colspan="2">Premier League</th>
</tr>
<tr><th>Apps</th><th>Goals</th></tr>
<tr><th colspan="5">Goalkeepers</th></tr>
<tr><td>1</td><td>GK</td><td>Group Keeper</td><td>38</td><td>0</td></tr>
<tr><th colspan="5">Forwards</th></tr>
<tr><td>9</td><td>FW</td><td>Group Striker</td><td>30+4</td><td>12</td></tr>
</table>
</body></html>
"""


def test_clean_one_file_three_level_header(tmp_path):
    path = tmp_path / "Group_Team_2324.html"
    path.write_text(THREE_LEVEL_HEADER_HTML, encoding="utf-8")

    df = clean_one_file(path, "Group Team", "2324")

    assert set(df["player"]) == {"Group Keeper", "Group Striker"}  # "Forwards" divider dropped
    striker = df[df["player"] == "Group Striker"].iloc[0]
    assert striker["league_apps"] == 34
    assert striker["league_goals"] == 12


# Apps/Starts table with NO goals column, plus a separate "Goalscorers"
# table carrying "Own goals"/"Totals" footer rows (Aston Villa/Liverpool style).
APPS_STARTS_WITH_SCORERS_HTML = """
<html><body>
<h3><span class="mw-headline" id="Appearances">Appearances</span></h3>
<table class="wikitable">
<tr>
<th rowspan="2">No.</th><th rowspan="2">Pos.</th><th rowspan="2">Player</th>
<th colspan="2">Premier League</th>
</tr>
<tr><th>Apps</th><th>Starts</th></tr>
<tr><td>2</td><td>DF</td><td>Starts Defender</td><td>29</td><td>23</td></tr>
<tr><td>11</td><td>FW</td><td>Starts Striker</td><td>37</td><td>37</td></tr>
<tr><td>Total</td><td>Total</td><td>Total</td><td>38</td><td>38</td></tr>
</table>
<h3><span class="mw-headline" id="Goalscorers">Goalscorers</span></h3>
<table class="wikitable">
<tr><th>Rank</th><th>Player</th><th>Premier League</th><th>Total</th></tr>
<tr><td>1</td><td>Starts Striker</td><td>19</td><td>27</td></tr>
<tr><td>Own goals</td><td>Own goals</td><td>4</td><td>4</td></tr>
<tr><td>Totals</td><td>Totals</td><td>23</td><td>31</td></tr>
</table>
</body></html>
"""


def test_clean_one_file_apps_starts_table_takes_goals_from_scorers_table(tmp_path):
    path = tmp_path / "Starts_Team_2324.html"
    path.write_text(APPS_STARTS_WITH_SCORERS_HTML, encoding="utf-8")

    df = clean_one_file(path, "Starts Team", "2324")

    defender = df[df["player"] == "Starts Defender"].iloc[0]
    assert (defender["league_starts"], defender["league_subs"], defender["league_apps"]) == (23, 6, 29)
    assert defender["league_goals"] == 0
    assert df[df["player"] == "Starts Striker"].iloc[0]["league_goals"] == 19


def test_clean_one_file_drops_summary_rows(tmp_path):
    path = tmp_path / "Starts_Team_2324.html"
    path.write_text(APPS_STARTS_WITH_SCORERS_HTML, encoding="utf-8")

    df = clean_one_file(path, "Starts Team", "2324")

    assert set(df["player"]) == {"Starts Defender", "Starts Striker"}
    assert df["league_goals"].sum() == 19  # not doubled by a "Totals" row
