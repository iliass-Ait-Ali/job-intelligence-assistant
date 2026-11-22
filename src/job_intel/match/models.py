"""Pydantic models for the candidate profile, preferences and match results.

Design rules (privacy and honesty):
* Contact details (name, e-mail, phone, address) are never extracted: they are not needed for matching.
* Every extracted item keeps the CV text it came from (`Evidence`), so any claim can be traced back.
* Nothing is inferred: no proficiency levels, no years of experience beyond what dates in the CV state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DegreeLevel = Literal["secondary", "associate", "bachelor", "master", "phd"]
DEGREE_ORDER: dict[str, int] = {"secondary": 0, "associate": 1, "bachelor": 2, "master": 3, "phd": 4}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(Strict):
    """A verbatim line from the CV."""

    text: str = Field(min_length=1, max_length=600)
    section: str | None = None


class SkillItem(Strict):
    name: str = Field(min_length=1, max_length=80)  # canonical taxonomy name, or a user-added skill
    category: str = "tool"
    # "demonstrated": appears inside an experience / project entry; "listed": only in a skills list.
    # This is where the CV mentions it, not a proficiency level.
    context: Literal["demonstrated", "listed"] = "listed"
    evidence: list[Evidence] = Field(default_factory=list, max_length=4)


class Education(Strict):
    degree: str | None = None
    level: DegreeLevel | None = None
    field: str | None = None
    institution: str | None = None
    start: str | None = None
    end: str | None = None
    in_progress: bool | None = None  # only set when the CV says so ("present", "en cours", a future year...)
    evidence: list[Evidence] = Field(default_factory=list, max_length=6)


class Experience(Strict):
    title: str | None = None
    organization: str | None = None
    kind: Literal["internship", "job", "unspecified"] = "unspecified"
    start: str | None = None  # "YYYY" or "YYYY-MM", exactly as far as the CV states it
    end: str | None = None
    current: bool = False
    # Bounds computed ONLY from the stated dates (None when dates are missing or incomplete).
    duration_months_min: int | None = None
    duration_months_max: int | None = None
    bullets: list[str] = Field(default_factory=list, max_length=20)
    skills: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list, max_length=25)


class Project(Strict):
    name: str | None = None
    description: list[str] = Field(default_factory=list, max_length=12)
    skills: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list, max_length=15)


class SpokenLanguage(Strict):
    language: str
    level_text: str | None = None  # exactly as written in the CV ("C1", "native", "courant"); never interpreted


class CandidateProfile(Strict):
    detected_language: Literal["en", "fr", "de", "unknown"] = "unknown"
    headline: str | None = None  # first line(s) of the CV when they look like a title, e.g. "Data Science student"
    summary: str | None = None
    target_roles: list[str] = Field(default_factory=list)  # only roles the CV says it is looking for
    education: list[Education] = Field(default_factory=list)
    skills: list[SkillItem] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    spoken_languages: list[SpokenLanguage] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)  # lines of a Certifications section, verbatim
    # Sector of each dated experience, estimated from its own lines (enrich/sector.py); empty when unclear.
    experience_sectors: list[str] = Field(default_factory=list)
    soft_skills: dict[str, str] = Field(default_factory=dict)  # claimed soft skill -> CV line (never scored)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("target_roles")
    @classmethod
    def _unique_roles(cls, roles: list[str]) -> list[str]:
        seen, out = set(), []
        for role in (r.strip() for r in roles):
            if role and role.casefold() not in seen:
                seen.add(role.casefold())
                out.append(role)
        return out

    # --- views requested in the specification -------------------------------------------------

    @property
    def programming_languages(self) -> list[str]:
        return [s.name for s in self.skills if s.category == "programming_language"]

    @property
    def frameworks_and_technologies(self) -> list[str]:
        return [s.name for s in self.skills if s.category not in {"programming_language", "concept"}]

    @property
    def technical_skills(self) -> list[str]:
        return [s.name for s in self.skills]

    @property
    def skill_names(self) -> set[str]:
        return {s.name for s in self.skills}

    @property
    def highest_degree_level(self) -> DegreeLevel | None:
        """Highest COMPLETED degree level (in-progress studies are not counted)."""
        done = [e.level for e in self.education if e.level and e.in_progress is not True]
        return max(done, key=lambda lv: DEGREE_ORDER[lv], default=None)

    @property
    def in_progress_degree_level(self) -> DegreeLevel | None:
        ongoing = [e.level for e in self.education if e.level and e.in_progress is True]
        return max(ongoing, key=lambda lv: DEGREE_ORDER[lv], default=None)

    def experience_months_bounds(self) -> tuple[int, int] | None:
        """(lower, upper) bound of stated professional experience in months, or None if nothing is dated.

        Lower: dated *jobs* only (internships may not count as experience for an employer).
        Upper: jobs + internships. Overlaps are not merged, which can only raise the upper bound.
        """
        jobs = [e for e in self.experiences if e.kind != "internship"]
        every = self.experiences
        if not any(e.duration_months_max is not None for e in every):
            return None
        lower = sum(e.duration_months_min or 0 for e in jobs)
        upper = sum(e.duration_months_max if e.duration_months_max is not None else 10**4 for e in every)
        return lower, upper


class Preferences(Strict):
    target_roles: list[str] = Field(default_factory=list)
    job_types: list[str] = Field(default_factory=list)  # canonical: internship, full_time, working_student, ...
    locations: list[str] = Field(default_factory=list)
    remote: Literal["any", "prefer", "only"] = "any"
    # Constraints that EXCLUDE a job when clearly violated. Unknown never excludes.
    # Others are reported as warnings only. Allowed: job_type, location, remote, experience, education, language.
    hard_constraints: list[str] = Field(default_factory=lambda: ["job_type", "location"])
    max_results: int = Field(default=15, ge=1, le=100)

    @field_validator("hard_constraints")
    @classmethod
    def _known(cls, names: list[str]) -> list[str]:
        allowed = {"job_type", "location", "remote", "experience", "education", "language"}
        bad = set(names) - allowed
        if bad:
            raise ValueError(f"unknown constraint(s) {sorted(bad)}; allowed: {sorted(allowed)}")
        return names
