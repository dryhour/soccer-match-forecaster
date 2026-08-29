"""
Unit tests for Wikipedia player-stats ingestion. No real network calls --
requests.get is monkeypatched.
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ingestion.wikipedia_player_stats as ingestion
from src.ingestion.wikipedia_player_stats import article_title, fetch_article_html, season_wiki_range


class FakeResponse:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def test_season_wiki_range():
    assert season_wiki_range("2324") == "2023–24"
    assert season_wiki_range("2526") == "2025–26"


def test_article_title_known_team():
    assert article_title("Arsenal", "2324") == "2023–24 Arsenal F.C. season"


def test_article_title_unknown_team_raises():
    try:
        article_title("Nowhere FC", "2324")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fetch_article_html_success(monkeypatch):
    monkeypatch.setattr(
        ingestion.requests, "get",
        lambda *a, **k: FakeResponse(200, {"parse": {"text": "<html>ok</html>"}}),
    )
    assert fetch_article_html("2023–24 Arsenal F.C. season") == "<html>ok</html>"


def test_fetch_article_html_missing_page(monkeypatch):
    monkeypatch.setattr(
        ingestion.requests, "get",
        lambda *a, **k: FakeResponse(200, {"error": {"code": "missingtitle"}}),
    )
    assert fetch_article_html("Not A Real Page") is None


def test_fetch_article_html_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429)
        return FakeResponse(200, {"parse": {"text": "<html>ok</html>"}})

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    monkeypatch.setattr(ingestion.time, "sleep", lambda _: None)
    assert fetch_article_html("2023–24 Arsenal F.C. season") == "<html>ok</html>"
    assert calls["n"] == 2
