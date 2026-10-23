"""Duplicate removal by stable identifier, canonical source URL, and reposted content."""

from __future__ import annotations

import re
from collections import Counter

from job_intel.models import Job
from job_intel.textutils import canonical_url


def _content_key(job: Job) -> tuple[str, str, str] | None:
    """(title, company, location), case/space-insensitive; None when there is no company.

    Catches reposts that get a fresh id and URL (seen live: the same Perplexity internship
    twice, and a Mozilla contract on arbeitnow.com and arbeitnow.co.uk). A different location
    keeps two otherwise identical titles apart (multi-site postings).
    """
    if not job.company:
        return None
    norm = lambda text: re.sub(r"\s+", " ", (text or "").casefold()).strip()  # noqa: E731
    return norm(job.title), norm(job.company), norm(job.location)


def deduplicate(jobs: list[Job]) -> tuple[list[Job], Counter[str]]:
    """Keep the first occurrence of each job; order the input by preference first.

    Two jobs are duplicates if they share a stable `id`, the same canonical `source_url`
    (scheme/host case, trailing slash, fragment and tracking parameters are ignored), or
    the same title + company + location. Returns the kept jobs and a Counter of removals by
    the first rule that matched ("id", "source_url", "title_company_location").
    """
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    seen_content: set[tuple[str, str, str]] = set()
    kept: list[Job] = []
    removed: Counter[str] = Counter()
    for job in jobs:
        url_key, content_key = canonical_url(job.source_url), _content_key(job)
        if job.id in seen_ids:
            removed["id"] += 1
        elif url_key in seen_urls:
            removed["source_url"] += 1
        elif content_key is not None and content_key in seen_content:
            removed["title_company_location"] += 1
        else:
            kept.append(job)
        seen_ids.add(job.id)
        seen_urls.add(url_key)
        if content_key is not None:
            seen_content.add(content_key)
    return kept, removed
