"""FAISS vector search over the vectors persisted in ChromaDB.

Division of labour
* **ChromaDB** is the *store*: it persists chunk texts, metadata and embeddings on disk, and its
  ``sync`` makes indexing incremental and idempotent (content hashes).
* **FAISS** is the *search engine*: at start-up the stored vectors are loaded once into an
  ``IndexFlatIP``, i.e. an **exact** inner-product scan. Vectors are L2-normalised, so the inner
  product IS the cosine similarity: ``cos(q, d) = q·d / (‖q‖‖d‖) = q·d``.

Why exact search: with a few thousand chunks a brute-force scan costs ~1 ms, so an approximate
index (HNSW, IVF) would only add recall loss. Chroma's own HNSW index is approximate; the
benchmark (``evaluation/vector_backends.py``) measures how often HNSW and FAISS-flat agree.
Filtering happens *inside* the search with an ``IDSelectorBatch`` (pre-filtering), so a filter can
never leave fewer than ``k`` results while eligible chunks exist.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from job_intel.search.errors import DependencyError
from job_intel.search.filters import IndexedJob
from job_intel.search.index import _PAGE, ChunkHit, JobIndex

log = logging.getLogger(__name__)


def _faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DependencyError('faiss is not installed: python -m pip install faiss-cpu') from exc
    return faiss


class FaissChunkIndex:
    """Drop-in replacement for ``JobIndex`` at query time (same ``query`` / ``job_table`` contract)."""

    backend = "faiss-flat-ip"

    def __init__(self, base: JobIndex) -> None:
        faiss = _faiss()
        self.base = base
        self.config = base.config
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        vectors: list[Any] = []
        offset = 0
        while True:
            page = base._col.get(include=["embeddings", "documents", "metadatas"], limit=_PAGE, offset=offset)
            ids += page["ids"]
            docs += page["documents"]
            metas += page["metadatas"]
            vectors += list(page["embeddings"])
            if len(page["ids"]) < _PAGE:
                break
            offset += _PAGE
        self._ids, self._docs, self._metas = ids, docs, metas
        dim = base.config.embedding_dim
        matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, dim) if vectors else np.zeros((0, dim), np.float32)
        faiss.normalize_L2(matrix)  # already normalised at indexing time; cheap insurance
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(matrix)
        self._rows_by_job: dict[str, list[int]] = defaultdict(list)
        for row, meta in enumerate(metas):
            self._rows_by_job[meta["job_id"]].append(row)
        self._table: dict[str, IndexedJob] = {}
        for meta in metas:
            self._table.setdefault(meta["job_id"], IndexedJob.from_metadata(meta))
        log.info("FAISS flat index: %d chunks, dim %d", self._index.ntotal, dim)

    def job_table(self) -> dict[str, IndexedJob]:
        return dict(self._table)

    def count_chunks(self) -> int:
        return int(self._index.ntotal)

    def chunks_of(self, job_id: str) -> list[str]:
        rows = sorted(self._rows_by_job.get(job_id, []), key=lambda r: int(self._metas[r]["chunk_index"]))
        return [self._docs[r] for r in rows]

    def query(self, vector: Sequence[float], n_results: int, job_ids: Iterable[str] | None = None) -> list[ChunkHit]:
        faiss = _faiss()
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        params = None
        if job_ids is not None:
            rows = [r for jid in set(job_ids) for r in self._rows_by_job.get(jid, [])]
            if not rows:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.asarray(rows, dtype=np.int64)))
            limit = len(rows)
        else:
            limit = self.count_chunks()
        k = max(1, min(n_results, limit))
        if limit == 0:
            return []
        sims, idx = self._index.search(query, k, params=params) if params else self._index.search(query, k)
        hits = []
        for sim, row in zip(sims[0], idx[0]):
            if row < 0:
                continue
            meta = self._metas[row]
            s = max(-1.0, min(1.0, float(sim)))
            hits.append(ChunkHit(self._ids[row], meta["job_id"], int(meta["chunk_index"]), self._docs[row] or "", 1.0 - s, s))
        hits.sort(key=lambda h: (-h.similarity, h.chunk_id))
        return hits
