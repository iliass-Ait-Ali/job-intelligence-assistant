"""Eligibility filters, shared by every search engine so comparisons use the same candidate set.

Semantics of missing or unknown values (also documented in the README):

* ``job_types``: match the canonical job_type. Jobs with no job_type (about a quarter of the
  data) match only if "unknown" is listed or ``include_unknown`` is set.
* ``remote``: only ``True`` can be filtered. Phase 1 stores ``remote=false`` for listings that
  do not say they are remote, so "false" means "not stated", not "on-site". With ``remote=True``
  those listings are excluded and ``include_unknown`` does NOT bring them back.
* ``since``: jobs published on or after the date. Jobs without a publication date are excluded
  unless ``include_unknown`` is set.
* ``locations``: accent/spelling-insensitive (see ``locations.py``), city-to-country aware, OR
  across the listed values. Jobs without a location are excluded unless ``include_unknown``.
* Different filters are combined with AND; values inside one filter with OR.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from job_intel.search.errors import SearchError
from job_intel.search.locations import location_matches

UNKNOWN = "unknown"
_KEYS = {"sources", "job_types", "remote", "locations", "since", "include_unknown"}


@dataclass(frozen=True)
class IndexedJob:
    """The job-level facts stored in the index, as seen by filters and result display."""

    job_id: str
    title: str
    company: str | None
    location: str | None
    job_type: str | None
    remote: bool | None
    source: str
    url: str
    published_ts: int | None
    n_chunks: int = 1
    content_hash: str = ""

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any]) -> IndexedJob:
        return cls(
            job_id=meta["job_id"], title=meta["title"], company=meta.get("company"),
            location=meta.get("location"), job_type=meta.get("job_type"), remote=meta.get("remote"),
            source=meta["source"], url=meta["url"], published_ts=meta.get("published_ts"),
            n_chunks=int(meta.get("n_chunks", 1)), content_hash=meta.get("content_hash", ""),
        )  # fmt: skip

    @property
    def published_date(self) -> date | None:
        return datetime.fromtimestamp(self.published_ts, tz=UTC).date() if self.published_ts is not None else None


@dataclass(frozen=True)
class JobFilter:
    sources: tuple[str, ...] = ()
    job_types: tuple[str, ...] = ()
    remote: bool | None = None  # only True is meaningful
    locations: tuple[str, ...] = ()
    since: date | None = None
    include_unknown: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.sources or self.job_types or self.remote or self.locations or self.since)

    def matches(self, job: IndexedJob) -> bool:
        if self.sources and job.source not in self.sources:
            return False
        if self.job_types:
            value = job.job_type or UNKNOWN
            if value not in self.job_types and not (job.job_type is None and self.include_unknown):
                return False
        if self.remote and job.remote is not True:
            return False
        if self.locations:
            if not job.location:
                if not self.include_unknown:
                    return False
            elif not any(location_matches(job.location, wanted) for wanted in self.locations):
                return False
        if self.since:
            published = job.published_date
            if published is None:
                if not self.include_unknown:
                    return False
            elif published < self.since:
                return False
        return True

    def eligible(self, jobs: Iterable[IndexedJob]) -> list[IndexedJob]:
        return [job for job in jobs if self.matches(job)]

    def describe(self) -> str:
        parts = []
        if self.sources:
            parts.append(f"source in {list(self.sources)}")
        if self.job_types:
            parts.append(f"job_type in {list(self.job_types)}")
        if self.remote:
            parts.append("remote (explicitly stated)")
        if self.locations:
            parts.append(f"location in {list(self.locations)}")
        if self.since:
            parts.append(f"published since {self.since.isoformat()}")
        if parts and self.include_unknown:
            parts.append("unknown values allowed")
        return "; ".join(parts) or "no filters"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> JobFilter:
        """Build a filter from a JSON object (eval/queries.json); rejects unknown keys."""
        data = dict(data or {})
        if unknown := set(data) - _KEYS:
            raise SearchError(f"unknown filter key(s): {sorted(unknown)}; allowed: {sorted(_KEYS)}")

        def as_tuple(value: Any) -> tuple[str, ...]:
            return () if value is None else ((value,) if isinstance(value, str) else tuple(value))

        since = data.get("since")
        try:
            since_date = date.fromisoformat(since) if isinstance(since, str) else since
        except ValueError as exc:
            raise SearchError(f"invalid 'since' date {since!r} (use YYYY-MM-DD)") from exc
        return cls(
            sources=as_tuple(data.get("sources")),
            job_types=as_tuple(data.get("job_types")),
            remote=True if data.get("remote") else None,
            locations=as_tuple(data.get("locations")),
            since=since_date,
            include_unknown=bool(data.get("include_unknown", False)),
        )
