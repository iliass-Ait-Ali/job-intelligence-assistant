"""Application service layer: ONE object that the API and the UI both call.

It owns the expensive, shared resources (embedding model, FAISS index, cross-encoder, postings,
market frame) and loads each lazily, once, thread-safely. Business logic stays in the domain
modules (search/, match/, market/, assistant/); this layer only wires them together, so the UI and
the API cannot drift apart and every feature is testable without a UI.

Privacy: CV text and profiles are passed in and returned, never stored here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from job_intel.config import Settings
from job_intel.enrich.postings import JobPosting, load_postings
from job_intel.match.cv_parser import parse_cv
from job_intel.match.cv_text import ExtractedCV, from_file, from_pasted_text
from job_intel.match.gap import GapReport, PlanStep, analyse_gap, learning_plan, market_demand
from job_intel.match.matcher import JobMatch, MatchResult, match_candidate
from job_intel.match.models import CandidateProfile, Preferences
from job_intel.match.scoring import CompatibilityScore, load_weights, rank_by_score
from job_intel.search.filters import JobFilter

log = logging.getLogger(__name__)
SCORING_CONFIG = Path(__file__).resolve().parents[2] / "config" / "scoring.yaml"  # repo checkout; else defaults


@dataclass
class Recommendations:
    result: MatchResult
    ranked: list[tuple[JobMatch, CompatibilityScore]]
    seconds: float


@dataclass
class JobIntelService:
    settings: Settings
    use_faiss: bool = True
    reranker_model: str | None = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    searcher_factory: Callable[[], Any] | None = None  # tests inject a small searcher
    postings_override: dict[str, JobPosting] | None = None
    _searcher: Any = None
    _postings: dict[str, JobPosting] | None = None
    _market_df: Any = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    timings: dict[str, float] = field(default_factory=dict)

    # ---- resources -----------------------------------------------------------------------------

    @property
    def postings(self) -> dict[str, JobPosting]:
        with self._lock:
            if self._postings is None:
                if self.postings_override is not None:
                    self._postings = dict(self.postings_override)
                else:
                    started = time.perf_counter()
                    self._postings = {p.id: p for p in load_postings(self.settings.enriched_path)}
                    self.timings["load_postings_s"] = time.perf_counter() - started
            return self._postings

    def searcher(self):
        with self._lock:
            if self._searcher is None:
                started = time.perf_counter()
                if self.searcher_factory is not None:
                    self._searcher = self.searcher_factory()
                else:
                    self._searcher = self._open_searcher()
                self.timings["open_searcher_s"] = time.perf_counter() - started
            return self._searcher

    def _open_searcher(self):
        from job_intel.search import service
        from job_intel.search.embedder import resolve_device
        from job_intel.search.retrieve import Searcher

        base, notes = service.open_searcher(self.settings)
        for note in notes:
            log.warning(note)
        index = base.index
        if self.use_faiss:
            from job_intel.search.faiss_index import FaissChunkIndex

            index = FaissChunkIndex(base.index)
        jobs = {jid: base.record(jid) for jid in base.table if base.record(jid) is not None}
        reranker = None
        if self.reranker_model:
            model, device = self.reranker_model, resolve_device(self.settings.device)

            def reranker():  # loaded on first hybrid_rerank query only
                from job_intel.search.rerank import CrossEncoderReranker

                return CrossEncoderReranker(model, device=device)

        return Searcher(index, base.embedder, jobs, reranker=reranker)

    def warm_up(self) -> dict[str, float]:
        """Load everything a first request would otherwise pay for (model, FAISS, BM25, requirement cache)."""
        from job_intel.match.matcher import _requirements

        started = time.perf_counter()
        searcher = self.searcher()
        _ = searcher.bm25, searcher.keyword
        for job_id in searcher.table:
            record = searcher.record(job_id) or {}
            _requirements(searcher.table[job_id].title, record.get("description"))
        _ = self.market_df
        self.timings["warm_up_s"] = time.perf_counter() - started
        return dict(self.timings)

    @property
    def market_df(self):
        with self._lock:
            if self._market_df is None:
                from job_intel.market.analytics import postings_frame

                self._market_df = postings_frame(self.postings.values())
            return self._market_df

    def eval_snapshot_df(self):
        """The frozen 2026-10-23 corpus, enriched on the fly (for snapshot comparisons), or None."""
        path = self.settings.data_dir / "eval_corpus" / "2026-10-23" / "jobs.jsonl"
        if not path.exists():
            return None
        from job_intel.enrich.postings import enrich_all
        from job_intel.market.analytics import postings_frame
        from job_intel.search.service import load_jobs

        postings, _ = enrich_all(load_jobs(path))
        return postings_frame(postings)

    def llm_client(self):
        from job_intel.rag.ollama import OllamaClient

        return OllamaClient(model=self.settings.llm_model, base_url=self.settings.ollama_url, timeout=self.settings.llm_timeout)

    def llm_status(self) -> tuple[bool, str]:
        from job_intel.rag.ollama import LLMError

        try:
            return True, self.llm_client().check()
        except LLMError as exc:
            return False, str(exc)

    def status(self) -> dict[str, Any]:
        postings = self.postings
        by_source: dict[str, int] = {}
        for p in postings.values():
            by_source[p.source] = by_source.get(p.source, 0) + 1
        dates = sorted(p.published_at for p in postings.values() if p.published_at)
        return {"postings": len(postings), "by_source": by_source,
                "published_from": dates[0][:10] if dates else None, "published_to": dates[-1][:10] if dates else None,
                "companies": len({p.company for p in postings.values() if p.company}),
                "countries": len({c for p in postings.values() for c in p.countries}), "timings": dict(self.timings)}  # fmt: skip

    # ---- jobs & search -------------------------------------------------------------------------

    def get_job(self, job_id: str) -> JobPosting | None:
        return self.postings.get(job_id)

    def search(self, query: str, *, engine: str = "hybrid", k: int = 10, flt: JobFilter | None = None,
               max_per_company: int | None = None):  # fmt: skip
        searcher = self.searcher()
        started = time.perf_counter()
        response = searcher.search(query, engine=engine, k=k, flt=flt, max_per_company=max_per_company)
        response.notes.append(f"{engine}: {1000 * (time.perf_counter() - started):.0f} ms")
        return response

    # ---- CV & recommendations ------------------------------------------------------------------

    @staticmethod
    def read_cv(*, path: str | Path | None = None, text: str | None = None) -> ExtractedCV:
        return from_file(path) if path else from_pasted_text(text or "")

    @staticmethod
    def parse_profile(cv: ExtractedCV) -> CandidateProfile:
        profile = parse_cv(cv.text)
        profile.warnings[:0] = cv.warnings
        return profile

    def weights(self) -> dict[str, float]:
        return load_weights(SCORING_CONFIG if SCORING_CONFIG.exists() else None)

    def recommend(self, profile: CandidateProfile, prefs: Preferences, *, weights: dict[str, float] | None = None,
                  sort: str = "score") -> Recommendations:  # fmt: skip
        from job_intel.match.scoring import score_match

        started = time.perf_counter()
        # Assess the whole candidate pool (max_results=100), then let the chosen order pick the top.
        wide = prefs.model_copy(update={"max_results": 100})
        result = match_candidate(profile, wide, self.searcher(), pool_size=max(80, prefs.max_results * 3))
        w = weights or self.weights()
        if sort == "score":
            ranked = rank_by_score(result.matches, profile, w)
        else:  # the matcher's rule-based evidence order
            ranked = [(m, score_match(m, profile, w)) for m in result.matches]
        result.warnings = [x for x in result.warnings if "max_results" not in x]
        return Recommendations(result, ranked[: prefs.max_results], time.perf_counter() - started)

    # ---- gap analysis & plan -------------------------------------------------------------------

    @staticmethod
    def gap(profile: CandidateProfile, match: JobMatch, declared: dict[str, str] | None = None) -> GapReport:
        return analyse_gap(profile, match, declared)

    def plan(self, profile: CandidateProfile, ranked: list[tuple[JobMatch, CompatibilityScore]],
             declared: dict[str, str] | None = None, top: int = 10, max_steps: int = 8) -> list[PlanStep]:  # fmt: skip
        reports = [(analyse_gap(profile, m, declared), sc.score) for m, sc in ranked[:top]]
        return learning_plan(reports, market_demand(self.postings.values()), max_steps=max_steps)

    # ---- assistant -----------------------------------------------------------------------------

    def assistant_session(self, profile: CandidateProfile | None = None,
                          ranked: list[tuple[JobMatch, CompatibilityScore]] | None = None,
                          selected_job_id: str | None = None):  # fmt: skip
        from job_intel.assistant.chat import AssistantSession

        return AssistantSession(self.postings, self.searcher(), profile, list(ranked or []), selected_job_id, self.market_df)

    def ask(self, session, question: str, *, use_llm: bool = False):
        from job_intel.assistant.chat import ask

        return ask(session, question, llm_client=self.llm_client() if use_llm else None)
