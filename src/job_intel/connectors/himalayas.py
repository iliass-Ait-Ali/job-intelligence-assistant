"""Himalayas remote-jobs API (https://himalayas.app/docs/remote-jobs-api).

Terms (API documentation, checked 2026-11-22): free public API, no key required; every "Apply"
action must use the supplied ``applicationLink``; Himalayas must be credited with a visible source
link; jobs may be shown in our own app but must not be passed on to other job boards, aggregators,
Google Jobs or LinkedIn Jobs. Every Job keeps ``applicationLink`` as its URL and source="himalayas".

The full feed holds >100k jobs of every kind, so we use the *search* endpoint with a short list of
target-role queries (data / ML / AI), 20 results per page, a few pages per query, paced by the
shared HTTP client (default 2 s between requests).
"""

from __future__ import annotations

import logging
from typing import Any

from job_intel.connectors.base import Connector, FetchResult, stable_id
from job_intel.http import FetchError, HttpClient
from job_intel.models import InvalidRecordError, Job
from job_intel.textutils import canonical_job_type, clean_str, html_to_text, parse_datetime

log = logging.getLogger(__name__)

SEARCH_URL = "https://himalayas.app/jobs/api/search"
DEFAULT_QUERIES = (
    "data scientist",
    "machine learning engineer",
    "data engineer",
    "data analyst",
    "AI engineer",
    "MLOps",
    "NLP",
    "analytics engineer",
    "LLM",
    "data science intern",
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


class HimalayasConnector(Connector):
    name = "himalayas"

    def __init__(
        self,
        client: HttpClient,
        queries: tuple[str, ...] = DEFAULT_QUERIES,
        pages_per_query: int = 3,
        url: str = SEARCH_URL,
    ) -> None:
        self.client = client
        self.queries = queries
        self.pages_per_query = pages_per_query
        self.url = url

    def fetch(self) -> FetchResult:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        result = FetchResult(records=records, meta={"queries": list(self.queries), "pages_per_query": self.pages_per_query})
        failures = 0
        for query in self.queries:
            for page in range(1, self.pages_per_query + 1):
                try:
                    payload = self.client.get_json(self.url, params={"q": query, "page": page})
                except FetchError as exc:
                    failures += 1
                    result.warnings.append(f"query {query!r} page {page} failed: {exc}")
                    break
                jobs = payload.get("jobs") if isinstance(payload, dict) else None
                if not isinstance(jobs, list):
                    result.warnings.append(f"query {query!r} page {page}: response has no 'jobs' list")
                    break
                for job in jobs:
                    key = str(job.get("guid") or job.get("applicationLink") or "")
                    if key and key not in seen:
                        seen.add(key)
                        job = dict(job)
                        job["_query"] = query
                        records.append(job)
                if len(jobs) < 20:  # last page of this query
                    break
        result.requests_made = self.client.requests_made
        if not records:
            raise FetchError(f"Himalayas returned no jobs ({failures} failed request(s))")
        log.info("himalayas: %d unique records from %d queries", len(records), len(self.queries))
        return result

    def normalize(self, raw: dict[str, Any], collected_at: str) -> Job:
        title = clean_str(raw.get("title"))
        url = clean_str(raw.get("applicationLink")) or clean_str(raw.get("guid"))
        if not title or not url:
            raise InvalidRecordError("Himalayas record needs a title and an applicationLink")
        guid = clean_str(raw.get("guid")) or url
        native = guid.rstrip("/").rsplit("/jobs/", 1)[-1] if "/jobs/" in guid else None
        company_slug = clean_str(raw.get("companySlug"))
        locations = _as_list(raw.get("locationRestrictions"))
        tags = _as_list(raw.get("categories")) + _as_list(raw.get("parentCategories")) + _as_list(raw.get("seniority"))
        employment = clean_str(raw.get("employmentType"))
        return Job(
            id=stable_id(self.name, f"{company_slug}/{native}" if native and company_slug else native, url),
            title=title,
            company=clean_str(raw.get("companyName")),
            # Himalayas lists countries where the remote job may be done; none means worldwide.
            location=", ".join(locations) if locations else "Worldwide",
            job_type=canonical_job_type(employment, title),
            remote=True,  # every Himalayas listing is a remote job
            description=html_to_text(raw.get("description")),
            source=self.name,
            source_url=url,
            published_at=parse_datetime(raw.get("pubDate")),
            collected_at=collected_at,
            tags=tuple(tags),
        )
