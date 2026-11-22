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


# ---- Phase 3 fixtures (need the optional search/app dependencies) --------------------------------


@pytest.fixture(scope="session")
def match_searcher(tmp_path_factory):
    pytest.importorskip("chromadb")
    from match_data import match_jobs

    from job_intel.search.document import ChunkingConfig
    from job_intel.search.embedder import HashingEmbedder
    from job_intel.search.index import IndexConfig, JobIndex
    from job_intel.search.retrieve import Searcher

    embedder = HashingEmbedder(max_seq_length=128)
    records = match_jobs()
    index = JobIndex.open_for_build(tmp_path_factory.mktemp("match_idx"), IndexConfig.build(embedder, ChunkingConfig(100, 20)))
    index.sync(records, embedder)
    return Searcher(index, embedder, {j["id"]: j for j in records})


def _profile(name: str):
    from datetime import date

    from job_intel.match.cv_parser import parse_cv
    from job_intel.match.cv_text import clean_cv_text

    path = Path(__file__).parent.parent / "eval" / "synthetic_cvs" / name
    return parse_cv(clean_cv_text(path.read_text(encoding="utf-8")), today=date(2026, 10, 23))


@pytest.fixture(scope="session")
def student_profile():
    return _profile("cv_en_student_data_science.txt")


@pytest.fixture(scope="session")
def ml_profile():
    return _profile("cv_en_ml_engineer.txt")


# ---- platform fixtures (services, API, UI) ---------------------------------------------------------


class FakeReranker:
    """Deterministic stand-in for the cross-encoder: scores a passage by query-word overlap."""

    name = "fake-reranker"

    def score(self, query, passages):
        import numpy as np

        words = set(query.casefold().split())
        return np.asarray([len(words & set(p.casefold().split())) for p in passages], dtype=np.float32)


@pytest.fixture(scope="session")
def platform_service(match_searcher):
    """A JobIntelService over the synthetic match corpus, with FAISS and a fake re-ranker (no downloads)."""
    from match_data import match_jobs

    from job_intel.enrich.postings import enrich_all
    from job_intel.search.faiss_index import FaissChunkIndex
    from job_intel.search.retrieve import Searcher
    from job_intel.services import JobIntelService

    postings, rejected = enrich_all(match_jobs())
    assert not rejected
    records = {j["id"]: j for j in match_jobs()}
    searcher = Searcher(FaissChunkIndex(match_searcher.index), match_searcher.embedder, records, reranker=FakeReranker())
    return JobIntelService(Settings(user_agent="tests"), searcher_factory=lambda: searcher,
                           postings_override={p.id: p for p in postings})
