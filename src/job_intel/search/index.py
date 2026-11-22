"""The persistent ChromaDB index: one record per chunk, grouped by job.

Chunk id      ``<job id>#<chunk number>`` (stable across runs)
Embedding     computed here and passed to Chroma (Chroma never loads its own model); normalized
Distance      cosine. Chroma returns ``distance = 1 - cosine_similarity``; ``ChunkHit`` carries both
Config        model, dimension, prefixes, chunking and distance are stored with the collection and
              checked on every open, so vectors from different settings are never mixed
Idempotency   each job stores a content hash; ``sync`` adds new jobs, replaces changed or
              incomplete ones, deletes jobs that left the dataset and leaves the rest untouched
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from job_intel.search.document import CHUNKER_VERSION, ChunkingConfig, build_chunks
from job_intel.search.errors import DependencyError, IndexConfigMismatch, IndexNotFoundError, SearchError
from job_intel.search.filters import IndexedJob

if TYPE_CHECKING:
    from job_intel.search.embedder import Embedder

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
COLLECTION_NAME = "jobs"
CONFIG_KEY = "jobintel_config"
HNSW = {"space": "cosine", "ef_construction": 200, "ef_search": 400}
_PAGE = 5000


@dataclass(frozen=True)
class IndexConfig:
    schema_version: int
    model_name: str
    embedding_dim: int
    max_seq_length: int
    passage_prefix: str
    query_prefix: str
    normalized: bool
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    chunker_version: int
    distance: str = "cosine"

    @classmethod
    def build(cls, embedder: Embedder, chunking: ChunkingConfig) -> IndexConfig:
        return cls(
            schema_version=SCHEMA_VERSION, model_name=embedder.name, embedding_dim=embedder.dim,
            max_seq_length=embedder.max_seq_length, passage_prefix=embedder.passage_prefix,
            query_prefix=embedder.query_prefix, normalized=True, chunk_max_tokens=chunking.max_tokens,
            chunk_overlap_tokens=chunking.overlap_tokens, chunker_version=CHUNKER_VERSION,
        )  # fmt: skip

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> IndexConfig:
        return cls(**json.loads(text))

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(self.chunk_max_tokens, self.chunk_overlap_tokens)

    def differences(self, other: IndexConfig) -> list[str]:
        mine, theirs = asdict(self), asdict(other)
        return [f"{k}: index has {mine[k]!r}, requested {theirs[k]!r}" for k in mine if mine[k] != theirs[k]]


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    job_id: str
    chunk_index: int
    text: str
    distance: float  # Chroma cosine distance, 0 = identical direction
    similarity: float  # 1 - distance = cosine similarity


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    duplicate_inputs: int = 0
    total_jobs: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    seconds: float = 0.0
    chunk_token_stats: dict[str, float] = field(default_factory=dict)


def _chromadb():
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DependencyError(
            f'chromadb could not be imported ({type(exc).__name__}: {exc}). '
            'If it is not installed, run: python -m pip install -e ".[search]"'
        ) from exc
    return chromadb, Settings


def content_hash(job: Mapping[str, Any]) -> str:
    """Hash of everything that influences a job's chunks or stored metadata."""
    keys = ("title", "company", "location", "job_type", "remote", "description", "source", "source_url", "published_at")
    payload = json.dumps({k: job.get(k) for k in keys}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{CHUNKER_VERSION}|{payload}".encode()).hexdigest()[:20]


def _timestamp(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp())


def _chunk_metadata(job: Mapping[str, Any], digest: str, index: int, total: int) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "job_id": job["id"], "chunk_index": index, "n_chunks": total, "content_hash": digest,
        "source": job.get("source") or "", "title": job["title"], "url": job.get("source_url") or "",
    }  # fmt: skip
    # Chroma metadata cannot hold null: unknown values are simply absent.
    optional = {
        "company": job.get("company"), "location": job.get("location"), "job_type": job.get("job_type"),
        "remote": job.get("remote"), "published_ts": _timestamp(job.get("published_at")),
    }  # fmt: skip
    meta.update({k: v for k, v in optional.items() if v is not None})
    return meta


class JobIndex:
    def __init__(self, collection: Any, config: IndexConfig) -> None:
        self._col = collection
        self.config = config

    # ---- opening -------------------------------------------------------------------------

    @staticmethod
    def _client(path: Path):
        chromadb, Settings = _chromadb()
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))

    @classmethod
    def open_existing(cls, path: Path) -> JobIndex:
        """Open for querying. The stored config tells the caller which model to load."""
        if not path.exists():
            raise IndexNotFoundError(f"no index at {path}. Build one with: jobintel index")
        client = cls._client(path)
        try:
            col = client.get_collection(COLLECTION_NAME, embedding_function=None)
        except Exception as exc:
            raise IndexNotFoundError(f"no index at {path}. Build one with: jobintel index") from exc
        raw = (col.metadata or {}).get(CONFIG_KEY)
        if not raw:
            raise IndexConfigMismatch("the index has no stored configuration; rebuild it with: jobintel index --rebuild")
        config = IndexConfig.from_json(raw)
        if config.schema_version != SCHEMA_VERSION:
            raise IndexConfigMismatch(
                f"index schema v{config.schema_version} != code schema v{SCHEMA_VERSION}; run: jobintel index --rebuild"
            )
        return cls(col, config)

    @classmethod
    def open_for_build(cls, path: Path, config: IndexConfig, *, rebuild: bool = False) -> JobIndex:
        client = cls._client(path)
        existing = None
        try:
            existing = client.get_collection(COLLECTION_NAME, embedding_function=None)
        except Exception:
            pass
        if existing is not None:
            raw = (existing.metadata or {}).get(CONFIG_KEY)
            diffs = IndexConfig.from_json(raw).differences(config) if raw else ["no stored configuration"]
            if diffs and not rebuild:
                raise IndexConfigMismatch(
                    "the existing index was built with different settings:\n  - "
                    + "\n  - ".join(diffs)
                    + "\nRe-run with --rebuild to discard it and re-embed everything."
                )
            if rebuild:
                log.info("rebuilding: dropping the existing index")
                client.delete_collection(COLLECTION_NAME)
                existing = None
        if existing is None:
            existing = client.create_collection(
                COLLECTION_NAME,
                configuration={"hnsw": dict(HNSW)},
                metadata={CONFIG_KEY: config.to_json()},
                embedding_function=None,
            )
        return cls(existing, config)

    # ---- reading -------------------------------------------------------------------------

    def _all_metadatas(self) -> list[Mapping[str, Any]]:
        out: list[Mapping[str, Any]] = []
        offset = 0
        while True:
            page = self._col.get(include=["metadatas"], limit=_PAGE, offset=offset)
            out.extend(page["metadatas"])
            if len(page["ids"]) < _PAGE:
                return out
            offset += _PAGE

    def job_table(self) -> dict[str, IndexedJob]:
        jobs: dict[str, IndexedJob] = {}
        for meta in self._all_metadatas():
            jobs.setdefault(meta["job_id"], IndexedJob.from_metadata(meta))
        return jobs

    def count_chunks(self) -> int:
        return int(self._col.count())

    def chunks_of(self, job_id: str) -> list[str]:
        """The job's chunk texts in document order."""
        page = self._col.get(where={"job_id": job_id}, include=["documents", "metadatas"])
        pairs = sorted(zip(page["metadatas"], page["documents"]), key=lambda md: int(md[0]["chunk_index"]))
        return [doc or "" for _, doc in pairs]

    def query(self, vector: Sequence[float], n_results: int, job_ids: Iterable[str] | None = None) -> list[ChunkHit]:
        """Nearest chunks, optionally restricted (before the vector search) to the given jobs."""
        where = None
        if job_ids is not None:
            ids = sorted(set(job_ids))
            if not ids:
                return []
            where = {"job_id": {"$in": ids}}
        res = self._col.query(
            query_embeddings=[list(map(float, vector))], n_results=max(1, n_results), where=where,
            include=["documents", "metadatas", "distances"],
        )  # fmt: skip
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                ChunkHit(cid, meta["job_id"], int(meta["chunk_index"]), doc or "", float(dist), max(-1.0, min(1.0, 1.0 - float(dist))))
            )
        expected = min(max(1, n_results), self._eligible_count(where))
        if len(hits) < expected:
            # The approximate (HNSW) search can return fewer neighbours than exist, e.g. when many chunks are
            # near-identical. Callers rely on "fewer than asked" meaning "no more candidates", so fall back to an
            # exact scan, which is cheap at this scale.
            log.warning("HNSW returned %d of %d requested chunks; using an exact scan", len(hits), expected)
            hits = self._exact_query(vector, max(1, n_results), where)
        return hits

    def _eligible_count(self, where: dict | None) -> int:
        if where is None:
            return self.count_chunks()
        return len(self._col.get(where=where, include=[])["ids"])

    def _exact_query(self, vector: Sequence[float], n_results: int, where: dict | None) -> list[ChunkHit]:
        import numpy as np

        ids: list[str] = []
        docs: list[str] = []
        metas: list[Mapping[str, Any]] = []
        vectors = []
        offset = 0
        while True:
            page = self._col.get(where=where, include=["embeddings", "documents", "metadatas"], limit=_PAGE, offset=offset)
            ids += page["ids"]
            docs += page["documents"]
            metas += page["metadatas"]
            vectors += list(page["embeddings"])
            if len(page["ids"]) < _PAGE:
                break
            offset += _PAGE
        if not ids:
            return []
        matrix = np.asarray(vectors, dtype=np.float32)
        query = np.asarray(vector, dtype=np.float32)
        sims = (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)
        order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))[:n_results]
        return [
            ChunkHit(ids[i], metas[i]["job_id"], int(metas[i]["chunk_index"]), docs[i] or "", 1.0 - float(sims[i]),
                     max(-1.0, min(1.0, float(sims[i]))))
            for i in order
        ]

    # ---- writing -------------------------------------------------------------------------

    def sync(
        self,
        jobs: Iterable[Mapping[str, Any]],
        embedder: Embedder,
        *,
        batch_size: int = 32,
        progress: Callable[[int, int], None] | None = None,
    ) -> SyncReport:
        """Make the index mirror `jobs`. Safe to run repeatedly."""
        started = time.perf_counter()
        report = SyncReport()
        wanted: dict[str, Mapping[str, Any]] = {}
        for job in jobs:
            if not job.get("id") or not job.get("title"):
                raise SearchError(f"job record without id/title: {dict(job)!r:.200}")
            if job["id"] in wanted:
                report.duplicate_inputs += 1  # first copy wins
                continue
            wanted[job["id"]] = job

        stored: dict[str, dict[str, Any]] = defaultdict(lambda: {"hash": None, "ids": [], "n": 0})
        offset = 0
        while True:  # chunk ids + metadata of everything currently stored
            page = self._col.get(include=["metadatas"], limit=_PAGE, offset=offset)
            for cid, meta in zip(page["ids"], page["metadatas"]):
                entry = stored[meta["job_id"]]
                entry["hash"], entry["n"] = meta["content_hash"], int(meta["n_chunks"])
                entry["ids"].append(cid)
            if len(page["ids"]) < _PAGE:
                break
            offset += _PAGE

        to_delete: list[str] = []
        to_process: list[Mapping[str, Any]] = []
        for job_id, entry in stored.items():
            if job_id not in wanted:
                to_delete += entry["ids"]
                report.removed += 1
        for job_id, job in wanted.items():
            entry, digest = stored.get(job_id), content_hash(job)
            if entry is None:
                report.added += 1
            elif entry["hash"] == digest and len(entry["ids"]) == entry["n"]:
                report.unchanged += 1
                continue
            else:  # changed, or a previous run stopped half-way through this job
                report.updated += 1
                to_delete += entry["ids"]
            to_process.append(job)

        for start in range(0, len(to_delete), 1000):
            self._col.delete(ids=to_delete[start : start + 1000])

        records: list[tuple[str, str, str, dict[str, Any]]] = []  # id, embed_text, body, metadata
        chunking = self.config.chunking
        token_counts: list[int] = []
        for job in to_process:
            digest = content_hash(job)
            chunks = build_chunks(job, embedder.count_passage_tokens, chunking, embedder.max_seq_length)
            for chunk in chunks:
                token_counts.append(chunk.tokens)
                records.append(
                    (f"{job['id']}#{chunk.index}", chunk.embed_text, chunk.text,
                     _chunk_metadata(job, digest, chunk.index, len(chunks)))
                )  # fmt: skip
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            vectors = embedder.embed_passages([r[1] for r in batch])
            self._col.upsert(
                ids=[r[0] for r in batch], embeddings=vectors, documents=[r[2] for r in batch],
                metadatas=[r[3] for r in batch],
            )  # fmt: skip
            report.embedded_chunks += len(batch)
            if progress:
                progress(report.embedded_chunks, len(records))

        report.total_jobs = len(wanted)
        report.total_chunks = self.count_chunks()
        if token_counts:
            ordered = sorted(token_counts)
            report.chunk_token_stats = {
                "min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1],
                "limit": min(chunking.max_tokens, embedder.max_seq_length),
            }  # fmt: skip
        report.seconds = time.perf_counter() - started
        return report
