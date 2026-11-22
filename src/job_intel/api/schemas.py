"""Request / response models of the REST API (they generate the OpenAPI documentation at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from job_intel.match.models import CandidateProfile, Preferences


class SearchFilters(BaseModel):
    job_types: list[str] = Field(default_factory=list, examples=[["internship"]])
    locations: list[str] = Field(default_factory=list, examples=[["France"]])
    remote: bool | None = None
    sources: list[str] = Field(default_factory=list)
    include_unknown: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500, examples=["stage data analyst à Paris"])
    engine: Literal["keyword", "bm25", "semantic", "hybrid", "hybrid_rerank"] = "hybrid"
    k: int = Field(10, ge=1, le=50)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    max_per_company: int | None = Field(None, ge=1, le=10)


class SearchHitOut(BaseModel):
    rank: int
    job_id: str
    title: str
    company: str | None
    location: str | None
    contract_type: str | None
    score: float
    score_kind: str
    snippet: str
    apply_url: str


class SearchResponseOut(BaseModel):
    engine: str
    query: str
    eligible_jobs: int
    total_jobs: int
    hits: list[SearchHitOut]
    notes: list[str]


class ParseResponse(BaseModel):
    profile: CandidateProfile
    source: str
    characters: int
    warnings: list[str]


class ProfileRequest(BaseModel):
    profile: CandidateProfile
    preferences: Preferences = Field(default_factory=Preferences)


class RecommendRequest(ProfileRequest):
    sort: Literal["score", "evidence"] = "score"


class ComponentOut(BaseModel):
    name: str
    value: float | None
    weight: float
    points: float
    detail: str


class SkillOut(BaseModel):
    skill: str
    status: str
    job_level: str
    job_evidence: str
    cv_evidence: list[str]
    via: list[str] = Field(default_factory=list)


class ConstraintOut(BaseModel):
    name: str
    status: str
    hard: bool
    detail: str


class RecommendationOut(BaseModel):
    rank: int
    job_id: str
    title: str
    company: str | None
    location: str | None
    contract_type: str | None
    apply_url: str
    source: str
    compatibility: float = Field(description="0-100 compatibility index over observable criteria; NOT a hiring probability")
    ranking_score: float
    coverage: float = Field(description="share of the criteria weight that could be observed (0-1)")
    band: str
    evidence_level: str
    components: list[ComponentOut]
    skills: list[SkillOut]
    constraints: list[ConstraintOut]
    notes: list[str]


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationOut]
    excluded: list[dict[str, Any]]
    eligible_total: int
    corpus_size: int
    seconds: float
    warnings: list[str]


class GapRequest(ProfileRequest):
    job_id: str
    declared: dict[str, Literal["have", "lack"]] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    profile: CandidateProfile | None = None
    preferences: Preferences = Field(default_factory=Preferences)
    selected_job_id: str | None = None
    use_llm: bool = Field(False, description="use the LOCAL Ollama model (CV data never leaves the machine)")
