"""
Unit tests for Bronze-layer ingestion. No real network calls -- requests.get
is monkeypatched.
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ingestion.football_data_co_uk as ingestion
from src.ingestion.football_data_co_uk import download_fixtures, download_one


class FakeResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content


def test_download_one_skips_non_200(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.requests, "get", lambda url, timeout: FakeResponse(404, b""))
    result = download_one("http://example.com/base", "E0", "9999", tmp_path)
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_download_one_skips_short_content(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.requests, "get", lambda url, timeout: FakeResponse(200, b"tiny"))
    result = download_one("http://example.com/base", "E0", "9999", tmp_path)
    assert result is None


def test_download_one_saves_valid_response(monkeypatch, tmp_path):
    content = b"Div,Date,HomeTeam,AwayTeam\n" + b"x" * 100
    monkeypatch.setattr(ingestion.requests, "get", lambda url, timeout: FakeResponse(200, content))
    # download_one prints out_path.relative_to(PROJECT_ROOT) on success; point
    # PROJECT_ROOT at tmp_path itself so that succeeds for this out-of-tree dir.
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", tmp_path)
    result = download_one("http://example.com/base", "E0", "9999", tmp_path)
    assert result == tmp_path / "E0_9999.csv"
    assert result.read_bytes() == content


def test_download_fixtures_saves_valid_response(monkeypatch, tmp_path):
    content = b"Div,Date,Time,HomeTeam,AwayTeam,Referee\n" + b"x" * 100
    monkeypatch.setattr(ingestion.requests, "get", lambda url, timeout: FakeResponse(200, content))
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", tmp_path)
    result = download_fixtures("http://example.com/fixtures.csv", tmp_path)
    assert result == tmp_path / "fixtures.csv"
    assert result.read_bytes() == content


def test_download_fixtures_skips_non_200(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion.requests, "get", lambda url, timeout: FakeResponse(500, b""))
    result = download_fixtures("http://example.com/fixtures.csv", tmp_path)
    assert result is None
