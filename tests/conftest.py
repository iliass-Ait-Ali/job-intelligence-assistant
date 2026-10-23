"""Shared test helpers. Everything here is offline; fixtures are SYNTHETIC, not live data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from job_intel.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClient:
    """Stands in for HttpClient: returns scripted responses in order; Exceptions are raised."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict | None]] = []

    @property
    def requests_made(self) -> int:
        return len(self.calls)

    def get_json(self, url: str, params: dict | None = None) -> Any:
        self.calls.append((url, params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(user_agent="job-intel-tests", data_dir=tmp_path / "data")


@pytest.fixture
def arbeitnow_pages() -> list[dict]:
    return [load_fixture("arbeitnow_page1.json"), load_fixture("arbeitnow_page2.json")]


@pytest.fixture
def remotive_payload() -> dict:
    return load_fixture("remotive.json")
