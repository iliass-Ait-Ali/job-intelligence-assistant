"""Glue between settings, files, index, embedder and searcher (used by the CLI)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from job_intel.config import Settings
from job_intel.search.document import ChunkingConfig
from job_intel.search.embedder import Embedder, load_embedder
from job_intel.search.errors import IndexConfigMismatch, JobsFileError
from job_intel.search.index import IndexConfig, JobIndex, SyncReport, content_hash
from job_intel.search.retrieve import Searcher

log = logging.getLogger(__name__)


def load_jobs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise JobsFileError(f"{path} not found. Run `jobintel collect` first (Phase 1).")
    jobs = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                jobs.append(json.loads(line))
            except ValueError as exc:
                raise JobsFileError(f"{path} line {number}: invalid JSON ({exc})") from exc
    if not jobs:
        raise JobsFileError(f"{path} contains no jobs")
    return jobs


def build_index(
    settings: Settings, *, jobs_path: Path | None = None, model: str | None = None, rebuild: bool = False,
    exclude_sources: tuple[str, ...] = (), progress: Callable[[int, int], None] | None = None,
) -> tuple[SyncReport, Embedder, IndexConfig]:  # fmt: skip
    jobs = load_jobs(jobs_path or settings.jobs_path)
    if exclude_sources:
        jobs = [j for j in jobs if j.get("source") not in exclude_sources]
    embedder = load_embedder(model or settings.embedding_model, device=settings.device)
    chunking = ChunkingConfig(settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    config = IndexConfig.build(embedder, chunking)
    index = JobIndex.open_for_build(settings.index_dir, config, rebuild=rebuild)
    report = index.sync(jobs, embedder, progress=progress)
    return report, embedder, config


def open_searcher(
    settings: Settings, *, jobs_path: Path | None = None, model: str | None = None
) -> tuple[Searcher, list[str]]:  # fmt: skip
    """Open the index with the model it was built with. Returns the searcher and warnings."""
    index = JobIndex.open_existing(settings.index_dir)
    config = index.config
    if model and model != config.model_name:
        raise IndexConfigMismatch(
            f"the index was built with {config.model_name!r}, not {model!r}. Queries must use the same model; "
            f"run `jobintel index --model {model} --rebuild` to switch."
        )
    embedder = load_embedder(config.model_name, device=settings.device)
    if (embedder.dim, embedder.query_prefix, embedder.passage_prefix) != (
        config.embedding_dim, config.query_prefix, config.passage_prefix
    ):
        raise IndexConfigMismatch(
            "the loaded model does not match the index (dimension or prefixes differ); "
            "run `jobintel index --rebuild`."
        )
    jobs = {j["id"]: j for j in load_jobs(jobs_path or settings.jobs_path)}
    searcher = Searcher(index, embedder, jobs)

    notes = []
    changed = sum(1 for jid, ij in searcher.table.items() if jid in jobs and content_hash(jobs[jid]) != ij.content_hash)
    new = sum(1 for jid in jobs if jid not in searcher.table)
    gone = sum(1 for jid in searcher.table if jid not in jobs)
    if changed or new or gone:
        notes.append(f"index is out of date vs the jobs file ({changed} changed, {new} new, {gone} removed): run `jobintel index`")
    return searcher, notes


def index_status(settings: Settings) -> dict[str, Any]:
    index = JobIndex.open_existing(settings.index_dir)
    table = index.job_table()
    by_source: dict[str, int] = {}
    for job in table.values():
        by_source[job.source] = by_source.get(job.source, 0) + 1
    return {
        "path": str(settings.index_dir),
        "jobs": len(table),
        "chunks": index.count_chunks(),
        "jobs_by_source": by_source,
        "config": index.config.__dict__,
    }
