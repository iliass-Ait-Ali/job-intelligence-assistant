"""The ETL pipeline: fetch (or reuse a fresh snapshot) -> normalize -> filter -> dedupe -> export."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from job_intel.config import Settings
from job_intel.connectors import ArbeitnowConnector, Connector, RemotiveConnector
from job_intel.dedup import deduplicate
from job_intel.http import FetchError, HttpClient
from job_intel.models import InvalidRecordError, Job
from job_intel.relevance import Relevance, filter_relevant
from job_intel.storage import RawStore, utcnow, write_csv, write_json, write_jsonl

log = logging.getLogger(__name__)

SOURCE_NAMES = ("arbeitnow", "remotive")
ATTRIBUTION = {
    "arbeitnow": "Jobs courtesy of Arbeitnow (https://www.arbeitnow.com). Link back to the listing URL.",
    "remotive": "Jobs courtesy of Remotive (https://remotive.com). Link back to the listing URL; "
    "do not resubmit to third-party job aggregators.",
}


@dataclass
class SourceStats:
    source: str
    fetched: int = 0
    invalid: int = 0
    relevant: int = 0
    final: int = 0
    requests_made: int = 0
    from_cache: bool = False
    fetched_at: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class CollectionReport:
    started_at: str
    sources: dict[str, SourceStats] = field(default_factory=dict)
    total_fetched: int = 0
    invalid_skipped: int = 0
    normalized: int = 0
    filtered_out: int = 0  # normalized but not relevant to the target roles
    relevant: int = 0
    duplicates_removed: int = 0
    duplicates_by_reason: dict[str, int] = field(default_factory=dict)
    final: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_job_type: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    attribution: dict[str, str] = field(default_factory=dict)

    @property
    def all_sources_failed(self) -> bool:
        return bool(self.sources) and all(s.error for s in self.sources.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_connectors(settings: Settings, names: tuple[str, ...] | list[str]) -> dict[str, Connector]:
    def client() -> HttpClient:  # one client per provider: each has its own request pacing/count
        return HttpClient(
            settings.user_agent,
            timeout=(settings.connect_timeout, settings.read_timeout),
            min_interval=settings.request_delay,
            retries=settings.max_retries,
        )

    factories = {
        "arbeitnow": lambda: ArbeitnowConnector(client(), max_pages=settings.arbeitnow_max_pages),
        "remotive": lambda: RemotiveConnector(client()),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"unknown source(s): {', '.join(sorted(unknown))}")
    return {name: factories[name]() for name in names}


def _collect_source(
    connector: Connector,
    store: RawStore,
    ttl_hours: float,
    force_refresh: bool,
    now: datetime,
    stats: SourceStats,
) -> list[Job]:
    """Fetch (or reuse a fresh snapshot of) one source and normalize its records."""
    snapshot = None if force_refresh else store.load_fresh(connector.name, ttl_hours, now)
    if snapshot:
        log.info("%s: reusing snapshot %s (fresh within %.1fh)", connector.name, snapshot.path.name, ttl_hours)
        records, fetched_at = snapshot.records, snapshot.fetched_at
        stats.from_cache = True
    else:
        result = connector.fetch()
        records, fetched_at = result.records, now
        stats.requests_made = result.requests_made
        stats.warnings.extend(result.warnings)
        store.save(connector.name, records, result.meta, fetched_at)

    stats.fetched = len(records)
    stats.fetched_at = fetched_at.isoformat()
    collected_at = fetched_at.isoformat()
    jobs: list[Job] = []
    for raw in records:
        try:
            jobs.append(connector.normalize(raw, collected_at))
        except InvalidRecordError as exc:
            stats.invalid += 1
            log.debug("%s: skipped record: %s", connector.name, exc)
        except Exception as exc:  # a malformed record must never abort the whole run
            stats.invalid += 1
            log.warning("%s: unexpected error normalizing a record (%r): %s", connector.name, type(exc).__name__, exc)
    return jobs


def run_collection(
    settings: Settings,
    connectors: dict[str, Connector],
    *,
    force_refresh: bool = False,
    now: datetime | None = None,
    export: bool = True,
    output_dir: Path | None = None,
) -> tuple[CollectionReport, list[Job]]:
    now = now or utcnow()
    store = RawStore(settings.raw_dir)
    report = CollectionReport(started_at=now.isoformat())
    report.attribution = {name: ATTRIBUTION.get(name, "") for name in connectors}
    normalized: list[Job] = []

    for name, connector in connectors.items():
        stats = report.sources[name] = SourceStats(source=name)
        try:
            jobs = _collect_source(connector, store, settings.ttl_hours.get(name, 1.0), force_refresh, now, stats)
        except FetchError as exc:
            stats.error = str(exc)
            log.error("%s: fetch failed: %s", name, exc)
            continue
        normalized.extend(jobs)

    report.total_fetched = sum(s.fetched for s in report.sources.values())
    report.invalid_skipped = sum(s.invalid for s in report.sources.values())
    report.normalized = len(normalized)

    scored: list[tuple[Job, Relevance]] = filter_relevant(normalized)
    report.relevant = len(scored)
    report.filtered_out = report.normalized - report.relevant
    for job, _ in scored:
        report.sources[job.source].relevant += 1

    relevance_by_id: dict[str, Relevance] = {}
    for job, rel in scored:
        relevance_by_id.setdefault(job.id, rel)
    final, removed = deduplicate([job for job, _ in scored])  # scored is best-first, so the best copy wins
    report.duplicates_by_reason = dict(removed)
    report.duplicates_removed = sum(removed.values())
    report.final = len(final)

    categories: Counter[str] = Counter()
    for job in final:
        report.sources[job.source].final += 1
        rel = relevance_by_id[job.id]
        categories.update(rel.categories or ("early_career_data_signal",))
    report.by_category = dict(categories.most_common())
    report.by_job_type = dict(Counter(job.job_type or "unknown" for job in final).most_common())

    if export and not report.all_sources_failed:
        out = output_dir or settings.processed_dir
        report.outputs = {
            "csv": str(out / "jobs.csv"),
            "jsonl": str(out / "jobs.jsonl"),
            "summary": str(out / "collection_summary.json"),
        }
        write_csv(out / "jobs.csv", final)
        write_jsonl(out / "jobs.jsonl", final)
        write_json(out / "collection_summary.json", report.to_dict())
    return report, final
