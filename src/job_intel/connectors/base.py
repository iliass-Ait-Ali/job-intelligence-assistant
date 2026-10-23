"""Connector contract: fetch raw records from one provider, normalize one record to a Job."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from job_intel.models import Job


@dataclass
class FetchResult:
    records: list[dict[str, Any]]
    requests_made: int = 0
    # Provider notices, page counts, ... stored with the raw snapshot.
    meta: dict[str, Any] = field(default_factory=dict)
    # Non-fatal problems, e.g. a later page failing after earlier pages succeeded.
    warnings: list[str] = field(default_factory=list)


class Connector(ABC):
    name: str

    @abstractmethod
    def fetch(self) -> FetchResult:
        """Return raw provider records. Raise FetchError if nothing usable came back."""

    @abstractmethod
    def normalize(self, raw: dict[str, Any], collected_at: str) -> Job:
        """Convert one raw record. Raise InvalidRecordError if it is unusable."""


def stable_id(source: str, native_id: Any, url: str) -> str:
    """`<source>:<provider id>`, falling back to a hash of the URL if the provider gave none."""
    native = str(native_id).strip() if native_id not in (None, "") else ""
    return f"{source}:{native or hashlib.sha1(url.encode()).hexdigest()[:16]}"
