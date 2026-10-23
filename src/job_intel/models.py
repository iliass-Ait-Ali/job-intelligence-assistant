"""The normalized record shared by every connector and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Exported columns, in order. `Job.tags` is deliberately not part of the export.
FIELDS = (
    "id",
    "title",
    "company",
    "location",
    "job_type",
    "remote",
    "description",
    "source",
    "source_url",
    "published_at",
    "collected_at",
)


class InvalidRecordError(ValueError):
    """A raw API record is unusable (e.g. no title or URL) and must be skipped."""


@dataclass(slots=True)
class Job:
    id: str
    title: str
    company: str | None
    location: str | None
    job_type: str | None
    remote: bool | None
    description: str | None
    source: str
    source_url: str
    published_at: str | None
    collected_at: str
    # Source-provided labels, used for relevance scoring only.
    tags: tuple[str, ...] = field(default=(), repr=False)

    def to_record(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FIELDS}
