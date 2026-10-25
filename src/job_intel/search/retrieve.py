"""Query-time retrieval: semantic (ChromaDB) and lexical (BM25) behind one interface.

Pipeline for the semantic engine
1. Resolve the eligible jobs with the metadata filter (``JobFilter``, shared with BM25).
2. Search only those jobs: the eligibility list is a Chroma ``where`` clause, so filtering happens
   *before* the vector search, never as a post-filter of a fixed-size candidate list.
3. Fetch chunks, group them by job (job score = its best chunk's cosine similarity) and, if the
   chunks seen do not yet cover ``k`` distinct jobs, double the number of chunks and repeat until
   they do or every eligible chunk has been seen. Once k jobs are covered the top-k is exact:
   any job not seen has all its chunks below the lowest score fetched.
4. Optionally (presentation only) cap the number of jobs per company.

Scores are never mixed between engines: ``score_kind`` says what ``score`` means.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from job_intel.search.filters import IndexedJob, JobFilter
from job_intel.search.index import JobIndex
from job_intel.search.lexical import Bm25Index, best_snippet, job_document

if TYPE_CHECKING:
    from job_intel.search.embedder import Embedder

SEMANTIC = "semantic"
BM25 = "bm25"
HYBRID = "hybrid"  # Reciprocal Rank Fusion of the semantic and BM25 rankings
ENGINES = (SEMANTIC, BM25, HYBRID)
INITIAL_CHUNKS = 50
RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009); not tuned on this data


@dataclass(frozen=True)
class JobHit:
    rank: int
    job: IndexedJob
    score: float
    score_kind: str  # "cosine_similarity" | "bm25" | "rrf"
    snippet: str
    distance: float | None = None  # semantic only: 1 - score


@dataclass
class SearchResponse:
    engine: str
    query: str
    hits: list[JobHit]
    eligible_jobs: int
    total_jobs: int
    chunks_fetched: int = 0
    exhausted: bool = False  # every eligible chunk was examined
    hidden_by_cap: int = 0
    notes: list[str] = field(default_factory=list)


def apply_company_cap(
    ranked: list[tuple[IndexedJob, Any]], max_per_company: int | None
) -> tuple[list[tuple[IndexedJob, Any]], int]:
    """Presentation-layer diversity: keep at most N jobs per company, preserving rank order."""
    if not max_per_company:
        return ranked, 0
    seen: dict[str, int] = {}
    kept = []
    for item in ranked:
        key = (item[0].company or item[0].job_id).casefold()
        if seen.get(key, 0) < max_per_company:
            seen[key] = seen.get(key, 0) + 1
            kept.append(item)
    return kept, len(ranked) - len(kept)


class Searcher:
    def __init__(self, index: JobIndex, embedder: Embedder, jobs: Mapping[str, Mapping[str, Any]]) -> None:
        """`jobs` maps job id -> processed record; it feeds BM25 and lexical snippets."""
        self.index, self.embedder = index, embedder
        self.table = index.job_table()
        self._records = {job_id: jobs[job_id] for job_id in self.table if job_id in jobs}
        self.missing_records = len(self.table) - len(self._records)
        self._bm25: Bm25Index | None = None

    @property
    def bm25(self) -> Bm25Index:
        if self._bm25 is None:
            self._bm25 = Bm25Index({jid: job_document(rec) for jid, rec in self._records.items()})
        return self._bm25

    def record(self, job_id: str) -> Mapping[str, Any] | None:
        """The processed job record (description etc.) behind an indexed job, if the jobs file has it."""
        return self._records.get(job_id)

    def eligible(self, flt: JobFilter) -> list[IndexedJob]:
        return flt.eligible(self.table.values())

    def search(
        self, query: str, *, engine: str = SEMANTIC, k: int = 10, flt: JobFilter | None = None,
        max_per_company: int | None = None, allowed_ids: Collection[str] | None = None,
    ) -> SearchResponse:  # fmt: skip
        """`allowed_ids` further restricts the candidates BEFORE ranking (used for hard constraints)."""
        flt = flt or JobFilter()
        eligible = self.eligible(flt)
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            eligible = [j for j in eligible if j.job_id in allowed]
        response = SearchResponse(engine, query, [], len(eligible), len(self.table))
        if self.missing_records:
            response.notes.append(
                f"{self.missing_records} indexed job(s) are missing from the jobs file (index out of date?)"
            )
        if not eligible:
            response.notes.append(f"no job satisfies the filters ({flt.describe()})")
            return response
        if engine == SEMANTIC:
            self._semantic(response, query, k, eligible, max_per_company)
        elif engine == BM25:
            self._lexical(response, query, k, eligible, max_per_company)
        elif engine == HYBRID:
            self._hybrid(response, query, k, eligible, max_per_company)
        else:
            raise ValueError(f"unknown engine {engine!r}; choose from {ENGINES}")
        return response

    def _semantic(self, response, query, k, eligible, max_per_company) -> None:
        eligible_ids = {j.job_id for j in eligible}
        eligible_chunks = sum(j.n_chunks for j in eligible)
        vector = self.embedder.embed_queries([query])[0]
        where_ids = None if len(eligible) == len(self.table) else eligible_ids
        n = min(max(INITIAL_CHUNKS, 5 * k), eligible_chunks)
        while True:
            chunk_hits = self.index.query(vector, n, where_ids)
            best: dict[str, Any] = {}
            for hit in chunk_hits:  # query() returns chunks nearest first
                best.setdefault(hit.job_id, hit)
            ranked = [(self.table[jid], hit) for jid, hit in best.items()]
            shown, hidden = apply_company_cap(ranked, max_per_company)
            exhausted = n >= eligible_chunks or len(chunk_hits) < n
            if len(shown) >= k or exhausted:
                break
            n = min(n * 2, eligible_chunks)
        response.chunks_fetched, response.exhausted, response.hidden_by_cap = len(chunk_hits), exhausted, hidden
        for rank, (job, hit) in enumerate(shown[:k], start=1):
            # Show the line of the matching chunk that shares most words with the query (else its start).
            response.hits.append(
                JobHit(rank, job, hit.similarity, "cosine_similarity", best_snippet(hit.text, query), distance=hit.distance)
            )
        if len(shown) < k:
            response.notes.append(f"only {len(shown)} job(s) available for this query and filters")

    def _hybrid(self, response, query, k, eligible, max_per_company) -> None:
        """RRF over the FULL semantic and BM25 rankings of the eligible jobs (each engine uncapped)."""
        n = len(eligible)
        parts = []
        for engine_fn in (self._semantic, self._lexical):
            part = SearchResponse("", query, [], len(eligible), len(self.table))
            engine_fn(part, query, n, eligible, None)
            parts.append(part)
        fused: dict[str, float] = {}
        snippets: dict[str, str] = {}
        for part in parts:
            for hit in part.hits:  # BM25 lists only jobs sharing a term; the others get no lexical vote
                fused[hit.job.job_id] = fused.get(hit.job.job_id, 0.0) + 1.0 / (RRF_K + hit.rank)
                snippets.setdefault(hit.job.job_id, hit.snippet)
        ranked = [(self.table[jid], score) for jid, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
        shown, hidden = apply_company_cap(ranked, max_per_company)
        response.chunks_fetched, response.exhausted, response.hidden_by_cap = parts[0].chunks_fetched, True, hidden
        for rank, (job, score) in enumerate(shown[:k], start=1):
            response.hits.append(JobHit(rank, job, score, "rrf", snippets[job.job_id]))
        if len(shown) < k:
            response.notes.append(f"only {len(shown)} job(s) available for this query and filters")

    def _lexical(self, response, query, k, eligible, max_per_company) -> None:
        ranked_ids = self.bm25.search(query, k=len(self.table), eligible=[j.job_id for j in eligible])
        ranked = [(self.table[jid], score) for jid, score in ranked_ids]
        shown, hidden = apply_company_cap(ranked, max_per_company)
        response.hidden_by_cap, response.exhausted = hidden, True
        for rank, (job, score) in enumerate(shown[:k], start=1):
            snippet = best_snippet(self._records[job.job_id].get("description"), query)
            response.hits.append(JobHit(rank, job, score, "bm25", snippet.replace("\n", " ")))
        if len(shown) < k:
            response.notes.append(f"only {len(shown)} job(s) share a term with the query")
