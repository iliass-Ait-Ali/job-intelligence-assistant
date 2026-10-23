"""Remotive remote-jobs API (https://remotive.com/api/remote-jobs).

Terms (from the API's legal notice): jobs may be shared further only with a link back to the
Remotive URL and Remotive named as the source; do not submit them to third-party job
aggregators (Jooble, Google Jobs, LinkedIn Jobs, ...); the data is delayed 24h; fetch at most
about 4 times a day. Every Job keeps the original remotive.com URL and source="remotive";
the pipeline's cache TTL (default 6h) enforces the request budget.
"""

from __future__ import annotations

import logging
from typing import Any

from job_intel.connectors.base import Connector, FetchResult, stable_id
from job_intel.http import FetchError, HttpClient
from job_intel.models import InvalidRecordError, Job
from job_intel.textutils import canonical_job_type, clean_str, html_to_text, parse_datetime

log = logging.getLogger(__name__)

API_URL = "https://remotive.com/api/remote-jobs"


class RemotiveConnector(Connector):
    name = "remotive"

    def __init__(self, client: HttpClient, url: str = API_URL) -> None:
        self.client = client
        self.url = url

    def fetch(self) -> FetchResult:
        payload = self.client.get_json(self.url)  # a single request per run
        if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
            raise FetchError("Remotive response has no 'jobs' list (API format changed?)")
        jobs = payload["jobs"]
        log.info("remotive: %d records (API reports job-count=%s)", len(jobs), payload.get("job-count"))
        return FetchResult(
            records=jobs,
            requests_made=self.client.requests_made,
            meta={
                "legal_notice": payload.get("0-legal-notice"),
                "job_count": payload.get("job-count"),
                "total_job_count": payload.get("total-job-count"),
            },
        )

    def normalize(self, raw: dict[str, Any], collected_at: str) -> Job:
        title = clean_str(raw.get("title"))
        url = clean_str(raw.get("url"))
        if not title or not url:
            raise InvalidRecordError("Remotive record needs a title and a url")
        tags = [t.strip() for t in (raw.get("tags") or []) if isinstance(t, str) and t.strip()]
        if category := clean_str(raw.get("category")):
            tags.append(category)
        return Job(
            id=stable_id(self.name, raw.get("id"), url),
            title=title,
            company=clean_str(raw.get("company_name")),
            location=clean_str(raw.get("candidate_required_location")),
            job_type=canonical_job_type(clean_str(raw.get("job_type")), title),
            remote=True,  # every Remotive listing is a remote job
            description=html_to_text(raw.get("description")),
            source=self.name,
            source_url=url,
            published_at=parse_datetime(raw.get("publication_date")),
            collected_at=collected_at,
            tags=tuple(tags),
        )
