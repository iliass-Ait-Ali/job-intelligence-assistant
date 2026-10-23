"""Raw snapshot cache (data/raw) and atomic exports (data/processed)."""

from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from job_intel.models import FIELDS, Job

KEEP_SNAPSHOTS = 3
_STAMP = "%Y%m%dT%H%M%SZ"


@dataclass
class Snapshot:
    source: str
    fetched_at: datetime
    records: list[dict[str, Any]]
    meta: dict[str, Any]
    path: Path


class RawStore:
    """Gzipped JSON snapshots of raw API responses: `<source>_<UTC timestamp>.json.gz`.

    They make reruns cheap and polite (a fresh snapshot is reused instead of hitting the
    API again) and keep the unmodified provider data for debugging and reprocessing.
    """

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def _paths(self, source: str) -> list[Path]:
        return sorted(self.raw_dir.glob(f"{source}_*.json.gz"))

    def save(self, source: str, records: list[dict[str, Any]], meta: dict[str, Any], fetched_at: datetime) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{source}_{fetched_at.strftime(_STAMP)}.json.gz"
        payload = {"source": source, "fetched_at": fetched_at.isoformat(), "meta": meta, "records": records}
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
        for old in self._paths(source)[:-KEEP_SNAPSHOTS]:
            old.unlink(missing_ok=True)
        return path

    def load_latest(self, source: str) -> Snapshot | None:
        for path in reversed(self._paths(source)):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    data = json.load(fh)
                return Snapshot(
                    source=source,
                    fetched_at=datetime.fromisoformat(data["fetched_at"]),
                    records=data["records"],
                    meta=data.get("meta", {}),
                    path=path,
                )
            except (OSError, ValueError, KeyError):
                continue  # corrupt snapshot: fall back to an older one or a refetch
        return None

    def load_fresh(self, source: str, ttl_hours: float, now: datetime) -> Snapshot | None:
        snap = self.load_latest(source)
        if snap and now - snap.fetched_at < timedelta(hours=ttl_hours):
            return snap
        return None


def utcnow() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def write_csv(path: Path, jobs: list[Job]) -> None:
    def _write(tmp: Path) -> None:
        # utf-8-sig so Excel on Windows detects UTF-8; pandas/csv read it transparently.
        with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            for job in jobs:
                row = job.to_record()
                if row["remote"] is not None:
                    row["remote"] = str(row["remote"]).lower()
                writer.writerow(row)  # None is written as an empty cell

    _atomic_write(path, _write)


def write_jsonl(path: Path, jobs: list[Job]) -> None:
    def _write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for job in jobs:
                fh.write(json.dumps(job.to_record(), ensure_ascii=False) + "\n")

    _atomic_write(path, _write)


def write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write(
        path,
        lambda tmp: tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"),
    )
