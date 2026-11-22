"""REST API (FastAPI). Run: ``jobintel api`` then open http://127.0.0.1:8000/docs.

Design
* Stateless: every request carries what it needs (a profile is sent back by the client); the server
  stores no CV and no profile. Uploaded CV files are read in memory and deleted immediately.
* One shared ``JobIntelService`` (dependency ``get_service``) owns the models and indexes; tests
  override it with a small in-memory corpus.
* Errors are mapped to clear HTTP codes: 400 unreadable CV, 404 unknown job, 422 invalid input
  (Pydantic), 503 index not built / LLM unavailable.
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from job_intel import __version__
from job_intel.api.schemas import (
    AskRequest,
    ComponentOut,
    ConstraintOut,
    GapRequest,
    ParseResponse,
    ProfileRequest,
    RecommendationOut,
    RecommendRequest,
    RecommendResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponseOut,
    SkillOut,
)  # fmt: skip
from job_intel.config import Settings
from job_intel.match.cv_text import CV_SUFFIXES, CVExtractionError
from job_intel.search.errors import SearchError
from job_intel.search.filters import JobFilter
from job_intel.services import JobIntelService

log = logging.getLogger(__name__)

app = FastAPI(
    title="Job & Internship Intelligence API",
    version=__version__,
    description="Hybrid job search, CV parsing, explainable CV-job matching, skill-gap analysis, market "
    "intelligence and a grounded career assistant. Compatibility scores are indices, not hiring probabilities.",
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_methods=["*"], allow_headers=["*"])  # fmt: skip


@lru_cache(maxsize=1)
def get_service() -> JobIntelService:
    return JobIntelService(Settings.from_env())


Service = Annotated[JobIntelService, Depends(get_service)]


def _job_or_404(service: JobIntelService, job_id: str):
    posting = service.get_job(job_id)
    if posting is None:
        raise HTTPException(404, f"unknown job id {job_id!r}")
    return posting


def recommendation_out(rank: int, match, score, posting) -> RecommendationOut:
    return RecommendationOut(
        rank=rank, job_id=match.job.job_id, title=match.job.title, company=match.job.company, location=match.job.location,
        contract_type=posting.contract_type if posting else match.job.job_type,
        apply_url=str(posting.apply_url) if posting else match.job.url, source=match.job.source,
        compatibility=score.score, ranking_score=score.ranking_score, coverage=score.coverage, band=score.band,
        evidence_level=match.evidence_level,
        components=[ComponentOut(name=c.name, value=c.value, weight=c.weight, points=round(score.contribution(c.name), 1),
                                 detail=c.detail) for c in score.components],
        skills=[SkillOut(skill=s.skill, status=s.status, job_level=s.job_level, job_evidence=s.job_evidence,
                         cv_evidence=[e.text for e in s.cv_evidence], via=s.via) for s in match.skills],
        constraints=[ConstraintOut(name=c.name, status=c.status, hard=c.hard, detail=c.detail) for c in match.constraints],
        notes=match.notes,
    )  # fmt: skip


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/status", tags=["meta"])
def status(service: Service) -> dict[str, Any]:
    ok, message = service.llm_status()
    return {**service.status(), "llm": {"available": ok, "message": message}}


@app.get("/jobs", tags=["jobs"])
def list_jobs(service: Service, country: str | None = None, role: str | None = None, contract: str | None = None,
              level: str | None = None, skill: str | None = None, limit: int = Query(20, ge=1, le=200),
              offset: int = Query(0, ge=0)) -> dict[str, Any]:  # fmt: skip
    items = list(service.postings.values())
    if country:
        items = [p for p in items if country.casefold() in p.countries or (p.region or "") == country.casefold()]
    if role:
        items = [p for p in items if role in p.role_family]
    if contract:
        items = [p for p in items if p.contract_type == contract]
    if level:
        items = [p for p in items if p.experience_level == level]
    if skill:
        items = [p for p in items if skill.casefold() in {s.name.casefold() for s in p.skills}]
    items.sort(key=lambda p: (p.published_at or ""), reverse=True)
    page = items[offset : offset + limit]
    return {"total": len(items), "offset": offset,
            "items": [p.model_dump(mode="json", exclude={"description"}) for p in page]}  # fmt: skip


@app.get("/jobs/{job_id:path}", tags=["jobs"])
def get_job(job_id: str, service: Service) -> dict[str, Any]:
    return _job_or_404(service, job_id).model_dump(mode="json")


@app.post("/search", response_model=SearchResponseOut, tags=["search"])
def search(req: SearchRequest, service: Service) -> SearchResponseOut:
    f = req.filters
    flt = JobFilter(sources=tuple(f.sources), job_types=tuple(f.job_types), remote=f.remote,
                    locations=tuple(f.locations), include_unknown=f.include_unknown)  # fmt: skip
    try:
        response = service.search(req.query, engine=req.engine, k=req.k, flt=flt, max_per_company=req.max_per_company)
    except SearchError as exc:
        raise HTTPException(503, str(exc)) from exc
    hits = []
    for h in response.hits:
        p = service.get_job(h.job.job_id)
        hits.append(SearchHitOut(rank=h.rank, job_id=h.job.job_id, title=h.job.title, company=h.job.company,
                                 location=h.job.location, contract_type=p.contract_type if p else h.job.job_type,
                                 score=float(h.score), score_kind=h.score_kind, snippet=h.snippet,
                                 apply_url=str(p.apply_url) if p else h.job.url))  # fmt: skip
    return SearchResponseOut(engine=response.engine, query=response.query, eligible_jobs=response.eligible_jobs,
                             total_jobs=response.total_jobs, hits=hits, notes=response.notes)  # fmt: skip


@app.post("/cv/parse", response_model=ParseResponse, tags=["cv"])
async def parse_cv_endpoint(service: Service, file: UploadFile | None = File(None),
                            text: str | None = Form(None)) -> ParseResponse:  # fmt: skip
    """Upload a PDF / DOCX / TXT CV **or** send pasted text. Nothing is stored."""
    try:
        if file is not None:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in CV_SUFFIXES:
                raise HTTPException(400, f"unsupported file type {suffix!r}; use PDF, DOCX or TXT")
            data = await file.read()
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                cv = service.read_cv(path=tmp)
            finally:
                os.unlink(tmp)
        elif text:
            cv = service.read_cv(text=text)
        else:
            raise HTTPException(400, "send a file or a text field")
    except CVExtractionError as exc:
        raise HTTPException(400, str(exc)) from exc
    profile = service.parse_profile(cv)
    return ParseResponse(profile=profile, source=cv.source, characters=len(cv.text), warnings=profile.warnings)


@app.post("/recommendations", response_model=RecommendResponse, tags=["matching"])
def recommendations(req: RecommendRequest, service: Service) -> RecommendResponse:
    try:
        recs = service.recommend(req.profile, req.preferences, sort=req.sort)
    except SearchError as exc:
        raise HTTPException(503, str(exc)) from exc
    out = [recommendation_out(i, m, sc, service.get_job(m.job.job_id)) for i, (m, sc) in enumerate(recs.ranked, start=1)]
    excluded = [{"job_id": m.job.job_id, "title": m.job.title, "reasons": m.excluded_because} for m in recs.result.excluded[:50]]
    return RecommendResponse(recommendations=out, excluded=excluded, eligible_total=recs.result.eligible_total,
                             corpus_size=recs.result.corpus_size, seconds=round(recs.seconds, 2),
                             warnings=recs.result.warnings)  # fmt: skip


def _find_match(service: JobIntelService, req: ProfileRequest, job_id: str):
    recs = service.recommend(req.profile, req.preferences.model_copy(update={"max_results": 100}))
    for m, sc in recs.ranked:
        if m.job.job_id == job_id:
            return m, sc
    for m in recs.result.excluded:
        if m.job.job_id == job_id:
            return m, None
    raise HTTPException(404, f"job {job_id!r} is not among the candidate's retrieved jobs")


@app.post("/gap", tags=["skills"])
def gap(req: GapRequest, service: Service) -> dict[str, Any]:
    _job_or_404(service, req.job_id)
    match, score = _find_match(service, req, req.job_id)
    report = service.gap(req.profile, match, req.declared)
    return {"job_id": report.job_id, "title": report.title, "company": report.company,
            "compatibility": score.score if score else None,
            "items": [vars(i) for i in report.items],
            "summary": {"mastered": len(report.mastered), "partial": len(report.partial), "missing": len(report.missing),
                        "missing_required": len(report.missing_required)}}  # fmt: skip


@app.post("/plan", tags=["skills"])
def plan(req: ProfileRequest, service: Service) -> dict[str, Any]:
    recs = service.recommend(req.profile, req.preferences)
    steps = service.plan(req.profile, recs.ranked)
    return {"based_on_jobs": len(recs.ranked), "steps": [vars(s) for s in steps]}


@app.get("/market/skills", tags=["market"])
def market_skills(service: Service, role: str | None = None, country: str | None = None, top: int = Query(20, ge=1, le=100),
                  required_only: bool = False) -> dict[str, Any]:  # fmt: skip
    from job_intel.market import analytics as ma

    df = ma.filter_frame(service.market_df, role_family=[role] if role else None, country=[country] if country else None)
    demand = ma.skill_demand(df, column="required_skills" if required_only else "skills", top=top)
    return {"postings": len(df), "skills": demand.to_dict(orient="records")}


@app.get("/market/cooccurrence", tags=["market"])
def market_cooccurrence(service: Service, top_skills: int = Query(20, ge=5, le=60), min_count: int = Query(5, ge=1)) -> dict[str, Any]:
    from job_intel.market import analytics as ma

    return {"pairs": ma.cooccurrence(service.market_df, top_skills, min_count).head(50).to_dict(orient="records")}


@app.get("/market/companies", tags=["market"])
def market_companies(service: Service, skill: str | None = None, top: int = Query(15, ge=1, le=100)) -> dict[str, Any]:
    from job_intel.market import analytics as ma

    df = service.market_df
    table = ma.top_companies_for_skill(df, skill, top) if skill else ma.counts_by(df, "company", top)
    return {"skill": skill, "companies": table.to_dict(orient="records")}


@app.post("/assistant/ask", tags=["assistant"])
def assistant_ask(req: AskRequest, service: Service) -> dict[str, Any]:
    ranked = service.recommend(req.profile, req.preferences).ranked if req.profile else []
    session = service.assistant_session(req.profile, ranked, req.selected_job_id)
    answer = service.ask(session, req.question, use_llm=req.use_llm)
    return {"question": answer.question, "intent": answer.intent, "mode": answer.mode, "language": answer.language,
            "sentences": [vars(s) for s in answer.sentences], "documents": [vars(d) for d in answer.docs],
            "rejected": [{"text": t, "reason": r} for t, r in answer.rejected], "notes": answer.notes,
            "latency_s": answer.latency_s}  # fmt: skip
