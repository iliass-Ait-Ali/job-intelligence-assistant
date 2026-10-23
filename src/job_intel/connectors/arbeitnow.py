"""Arbeitnow job-board API (https://www.arbeitnow.com/api/job-board-api).

Terms (from the API's own response): free public API, "please do not abuse", link back to
the site. Every Job keeps the original arbeitnow.com listing URL and source="arbeitnow".
"""

from __future__ import annotations

import logging
from typing import Any

from job_intel.connectors.base import Connector, FetchResult, stable_id
from job_intel.http import FetchError, HttpClient
from job_intel.models import InvalidRecordError, Job
from job_intel.textutils import canonical_job_type, clean_str, html_to_text, parse_datetime

log = logging.getLogger(__name__)

API_URL = "https://www.arbeitnow.com/api/job-board-api"


class ArbeitnowConnector(Connector):
    name = "arbeitnow"

    def __init__(self, client: HttpClient, max_pages: int = 10, url: str = API_URL) -> None:
        self.client = client
        self.max_pages = max_pages
        self.url = url

    def fetch(self) -> FetchResult:
        records: list[dict[str, Any]] = []
        result = FetchResult(records=records)
        for page in range(1, self.max_pages + 1):
            try:
                payload = self.client.get_json(self.url, params={"page": page})
                data = self._extract_data(payload)
            except FetchError as exc:
                if page == 1:
                    raise
                result.warnings.append(f"page {page} failed, keeping {page - 1} earlier page(s): {exc}")
                break
            if page == 1:
                result.meta["terms"] = (payload.get("meta") or {}).get("terms")
            records.extend(data)
            result.meta["pages_fetched"] = page
            log.info("arbeitnow page %d: %d records", page, len(data))
            if not data or not (payload.get("links") or {}).get("next"):
                break
        result.requests_made = self.client.requests_made
        return result

    @staticmethod
    def _extract_data(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise FetchError("Arbeitnow response has no 'data' list (API format changed?)")
        return payload["data"]

    def normalize(self, raw: dict[str, Any], collected_at: str) -> Job:
        title = clean_str(raw.get("title"))
        url = clean_str(raw.get("url"))
        if not title or not url:
            raise InvalidRecordError("Arbeitnow record needs a title and a url")
        tags = [t for t in (raw.get("tags") or []) if isinstance(t, str)]
        job_types = [t for t in (raw.get("job_types") or []) if isinstance(t, str)]
        remote = raw.get("remote")
        return Job(
            id=stable_id(self.name, raw.get("slug"), url),
            title=title,
            company=clean_str(raw.get("company_name")),
            location=clean_str(raw.get("location")),
            job_type=canonical_job_type(job_types, title),
            remote=remote if isinstance(remote, bool) else None,
            description=html_to_text(raw.get("description")),
            source=self.name,
            source_url=url,
            published_at=parse_datetime(raw.get("created_at")),
            collected_at=collected_at,
            tags=tuple(tags + job_types),
        )
