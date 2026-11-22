"""Module 1 - structured job postings.

Turns a cleaned listing (``data/processed/jobs.jsonl``, Phase 1) into a validated ``JobPosting``
with every field the specification asks for. Each extracted field keeps its evidence or its rule:

| field              | how it is obtained                                                     |
|--------------------|------------------------------------------------------------------------|
| countries / region | location gazetteer (``search/locations.py``), city -> country          |
| contract_type      | provider label, overridden by an explicit title ("Stage", "Werkstudent") |
| role_family        | strict title patterns (``relevance.py``)                                |
| sector             | keyword heuristic with a minimum of 2 hits (``enrich/sector.py``)        |
| required/preferred | requirement extractor: section headings + cue words (``match/requirements.py``) |
| soft_skills        | multilingual vocabulary (``match/softskills.py``)                         |
| experience_level   | title seniority words, else the stated minimum years                    |
| degree / languages | requirement extractor, only when the listing states them               |

Unknown is always ``None`` / empty, never a default value that looks like information.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from job_intel.enrich.sector import classify_sector
from job_intel.match.cv_parser import detect_language
from job_intel.match.requirements import extract_requirements
from job_intel.match.softskills import find_soft_skills
from job_intel.match.taxonomy import category_of
from job_intel.relevance import ROLE_PATTERNS
from job_intel.search.locations import countries_of, region_of

log = logging.getLogger(__name__)

ExperienceLevel = Literal["internship", "junior", "mid", "senior", "lead", "unspecified"]


class SkillRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    category: str
    level: Literal["required", "preferred", "mentioned"]
    evidence: str


class JobPosting(BaseModel):
    """One validated, structured listing. ``apply_url`` is always the provider's original link."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3)
    title: str = Field(min_length=2)
    company: str | None = None
    location: str | None = None
    countries: list[str] = Field(default_factory=list)
    region: str | None = None
    remote: bool | None = None
    contract_type: str | None = None
    role_family: list[str] = Field(default_factory=list)
    sector: str | None = None
    sector_evidence: list[str] = Field(default_factory=list)
    description: str = ""
    description_language: str = "unknown"
    skills: list[SkillRequirement] = Field(default_factory=list)
    soft_skills: dict[str, str] = Field(default_factory=dict)
    experience_level: ExperienceLevel = "unspecified"
    min_years: int | None = None
    degree_level: str | None = None
    degree_accepts_students: bool | None = None
    languages: list[str] = Field(default_factory=list)
    published_at: str | None = None
    collected_at: str | None = None
    source: str
    apply_url: HttpUrl

    @property
    def required_skills(self) -> list[str]:
        return [s.name for s in self.skills if s.level == "required"]

    @property
    def preferred_skills(self) -> list[str]:
        return [s.name for s in self.skills if s.level == "preferred"]

    @property
    def technical_skills(self) -> list[str]:
        return [s.name for s in self.skills]


def _level(seniority: str | None, years: int | None) -> ExperienceLevel:
    mapping: dict[str, ExperienceLevel] = {"intern": "internship", "junior": "junior", "senior": "senior",
                                           "lead_or_above": "lead"}  # fmt: skip
    if seniority in mapping:
        return mapping[seniority]
    if years is None:
        return "unspecified"
    if years <= 2:
        return "junior"
    return "mid" if years <= 5 else "senior"


def enrich_job(job: Mapping[str, Any]) -> JobPosting:
    title = job["title"]
    description = job.get("description") or ""
    reqs = extract_requirements(title, description)
    years = reqs.required_years
    sector, sector_hits = classify_sector(job.get("company"), description)
    contract = job.get("job_type")
    if contract is None and reqs.seniority == "intern":
        contract = "internship"
    return JobPosting(
        id=job["id"],
        title=title,
        company=job.get("company"),
        location=job.get("location"),
        countries=countries_of(job.get("location")),
        region=region_of(job.get("location")),
        remote=job.get("remote"),
        contract_type=contract,
        role_family=[cat for cat, pats in ROLE_PATTERNS.items() if any(p.search(title) for p in pats)],
        sector=sector,
        sector_evidence=sector_hits,
        description=description,
        description_language=detect_language(description),
        skills=[SkillRequirement(name=s.name, category=category_of(s.name), level=s.level, evidence=s.evidence)
                for s in reqs.skills],  # fmt: skip
        soft_skills=find_soft_skills(description),
        experience_level=_level(reqs.seniority, years),
        min_years=years,
        degree_level=reqs.degree[0] if reqs.degree else None,
        degree_accepts_students=reqs.degree[3] if reqs.degree else None,
        languages=[lang.language for lang in reqs.languages if lang.level != "preferred"],
        published_at=job.get("published_at"),
        collected_at=job.get("collected_at"),
        source=job["source"],
        apply_url=job["source_url"],
    )


def enrich_all(jobs: Iterable[Mapping[str, Any]]) -> tuple[list[JobPosting], list[dict[str, str]]]:
    """Validated postings plus the records that failed validation (id + reason), never silently dropped."""
    ok: list[JobPosting] = []
    rejected: list[dict[str, str]] = []
    for job in jobs:
        try:
            ok.append(enrich_job(job))
        except (ValidationError, KeyError, ValueError) as exc:
            rejected.append({"id": str(job.get("id")), "reason": str(exc).splitlines()[0][:300]})
    return ok, rejected


def quality_report(postings: list[JobPosting], rejected: list[dict[str, str]]) -> dict[str, Any]:
    """Field completeness: the share of postings where each field is known."""
    n = len(postings) or 1
    def share(pred) -> float:
        return round(sum(1 for p in postings if pred(p)) / n, 3)

    return {
        "postings": len(postings),
        "rejected": rejected,
        "completeness": {
            "company": share(lambda p: p.company),
            "location": share(lambda p: p.location),
            "country": share(lambda p: p.countries or p.region),
            "contract_type": share(lambda p: p.contract_type),
            "sector": share(lambda p: p.sector),
            "technical_skills": share(lambda p: p.skills),
            "required_skills_marked": share(lambda p: p.required_skills),
            "soft_skills": share(lambda p: p.soft_skills),
            "experience_level": share(lambda p: p.experience_level != "unspecified"),
            "min_years": share(lambda p: p.min_years is not None),
            "degree_level": share(lambda p: p.degree_level),
            "languages": share(lambda p: p.languages),
            "published_at": share(lambda p: p.published_at),
        },
    }


def write_postings(path: Path, postings: list[JobPosting]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for p in postings:
            fh.write(p.model_dump_json() + "\n")
    tmp.replace(path)


def load_postings(path: Path) -> list[JobPosting]:
    with path.open(encoding="utf-8") as fh:
        return [JobPosting.model_validate(json.loads(line)) for line in fh if line.strip()]
