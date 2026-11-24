"""Skill-gap analysis and personalised learning plan (Module 5).

Status of each skill a listing asks for, from strongest to weakest evidence:

| status            | meaning                                                                      |
|-------------------|------------------------------------------------------------------------------|
| ``demonstrated``  | the CV names the skill inside an experience or project (used, not just listed) |
| ``listed``        | the CV names the skill only in a skills list                                  |
| ``implied``       | the CV names a skill that *implies* it (PostgreSQL => SQL, PyTorch => deep learning) |
| ``transferable``  | the CV names a *related* skill of the same family (Tableau for Power BI)      |
| ``self_declared`` | not in the CV, but the user said "I have it" (the CV under-reports it)        |
| ``not_mentioned`` | no evidence in the CV. NOT the same as absent: the CV is the only source       |
| ``absent``        | the user confirmed they do not have it                                         |

The distinction "absent vs not mentioned" cannot be made from the CV alone - so the system does not
pretend to: it asks the user (``declared`` argument, a checkbox in the UI) and otherwise reports
``not_mentioned`` with a hint when related skills suggest the CV may simply be incomplete.

The learning plan ranks the gaps by

    priority(s) = Σ_jobs  w_level(s, job) · fit(job)   ×   (1 + demand(s))

* ``w_level`` = 1.0 required, 0.5 preferred, 0.3 mentioned - what the target jobs ask for;
* ``fit(job)`` = compatibility score / 100 - gaps of jobs you almost fit matter more;
* ``demand(s)`` = share of the whole corpus asking for s - market-wide usefulness;

then orders them so that prerequisites come first (topological sort on the knowledge-base graph,
priority as tie-break).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from job_intel.match.models import CandidateProfile
from job_intel.match.taxonomy import category_of, related_skills

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "skills_kb.yaml"
LEVEL_WEIGHT = {"required": 1.0, "preferred": 0.5, "mentioned": 0.3}
GapStatus = Literal["demonstrated", "listed", "implied", "transferable", "self_declared", "not_mentioned", "absent"]
COVERED = {"demonstrated", "listed", "self_declared"}
PARTIAL = {"implied", "transferable"}
MISSING = {"not_mentioned", "absent"}


@lru_cache(maxsize=1)
def knowledge_base(path: Path = KB_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _entry(skill: str) -> dict:
    kb = knowledge_base()
    entry = dict(kb["categories"].get(category_of(skill), {}))
    entry.update({k: v for k, v in (kb["skills"].get(skill) or {}).items() if v})
    entry.setdefault("implies", [])
    entry.setdefault("prerequisites", [])
    entry.setdefault("resources", [])
    entry.setdefault("interview", [])
    return entry


def implied_by(cv_skills: Iterable[str]) -> dict[str, str]:
    """Skill -> the CV skill that implies it (one level of implication, deterministic)."""
    out: dict[str, str] = {}
    for name in sorted(cv_skills):
        for target in _entry(name)["implies"]:
            out.setdefault(target, name)
    return out


@dataclass
class SkillGap:
    skill: str
    status: GapStatus
    job_level: str
    job_evidence: str
    cv_evidence: list[str] = field(default_factory=list)
    via: str | None = None  # implying / related CV skill
    hint: str | None = None


@dataclass
class GapReport:
    job_id: str
    title: str
    company: str | None
    items: list[SkillGap]

    def by_status(self, *statuses: str) -> list[SkillGap]:
        return [i for i in self.items if i.status in statuses]

    @property
    def mastered(self) -> list[SkillGap]:
        return self.by_status(*COVERED)

    @property
    def partial(self) -> list[SkillGap]:
        return self.by_status(*PARTIAL)

    @property
    def missing(self) -> list[SkillGap]:
        return self.by_status(*MISSING)

    @property
    def missing_required(self) -> list[SkillGap]:
        return [i for i in self.missing if i.job_level == "required"]

    @property
    def optional(self) -> list[SkillGap]:
        return [i for i in self.items if i.job_level != "required"]


def analyse_gap(profile: CandidateProfile, match, declared: Mapping[str, str] | None = None) -> GapReport:
    """`match` is a ``matcher.JobMatch``; `declared` maps skill -> "have" | "lack" (user answers)."""
    declared = dict(declared or {})
    cv = {s.name: s for s in profile.skills}
    implied = implied_by(cv)
    items: list[SkillGap] = []
    for sm in match.skills:
        name = sm.skill
        cv_lines = [e.text for e in cv[name].evidence[:2]] if name in cv else []
        if name in cv:
            status: GapStatus = "demonstrated" if cv[name].context == "demonstrated" else "listed"
            items.append(SkillGap(name, status, sm.job_level, sm.job_evidence, cv_lines))
            continue
        if declared.get(name) == "have":
            items.append(SkillGap(name, "self_declared", sm.job_level, sm.job_evidence,
                                  hint="Not in your CV: add it with a concrete example."))  # fmt: skip
            continue
        if name in implied:
            src = implied[name]
            items.append(SkillGap(name, "implied", sm.job_level, sm.job_evidence,
                                  [e.text for e in cv[src].evidence[:1]], via=src,
                                  hint=f"{src} implies working knowledge of {name}; name it explicitly in your CV."))  # fmt: skip
            continue
        if sm.status == "transferable" and sm.via:
            items.append(SkillGap(name, "transferable", sm.job_level, sm.job_evidence,
                                  [e.text for e in sm.cv_evidence[:3]], via=sm.via[0],
                                  hint=f"You know {', '.join(sm.via)}; {name} is related but not equivalent."))  # fmt: skip
            continue
        if declared.get(name) == "lack":
            items.append(SkillGap(name, "absent", sm.job_level, sm.job_evidence))
            continue
        neighbours = sorted(related_skills(name) & set(cv))
        hint = (f"Your CV mentions {', '.join(neighbours[:3])}: if you have used {name}, the CV may just not say it."
                if neighbours else "No evidence in the CV. If you have it, add it; otherwise see the learning plan.")  # fmt: skip
        items.append(SkillGap(name, "not_mentioned", sm.job_level, sm.job_evidence, hint=hint))
    order = {"required": 0, "preferred": 1, "mentioned": 2}
    items.sort(key=lambda i: (order.get(i.job_level, 3), i.skill))
    return GapReport(match.job.job_id, match.job.title, match.job.company, items)


# ---------------------------------------------------------------------------------------------
# learning plan


@dataclass
class PlanStep:
    order: int
    skill: str
    priority: float
    reasons: list[str]
    prerequisites: list[str]
    resources: list[dict]
    project: str | None
    interview: list[str]
    jobs: list[str]


def market_demand(postings: Iterable) -> dict[str, float]:
    """Share of postings that ask for each skill (any level)."""
    postings = list(postings)
    counts: dict[str, int] = {}
    for p in postings:
        for name in {s.name for s in p.skills}:
            counts[name] = counts.get(name, 0) + 1
    n = len(postings) or 1
    return {k: v / n for k, v in counts.items()}


def learning_plan(reports: list[tuple[GapReport, float]], demand: Mapping[str, float] | None = None,
                  max_steps: int = 8) -> list[PlanStep]:  # fmt: skip
    """`reports` = (gap report, compatibility score 0-100) for the jobs the candidate targets."""
    demand = demand or {}
    priority: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    jobs: dict[str, list[str]] = {}
    for report, score in reports:
        fit = max(score, 1.0) / 100.0
        for item in report.items:
            if item.status in COVERED:
                continue
            weight = LEVEL_WEIGHT.get(item.job_level, 0.3) * (0.5 if item.status in PARTIAL else 1.0)
            priority[item.skill] = priority.get(item.skill, 0.0) + weight * fit
            jobs.setdefault(item.skill, []).append(f"{report.title} ({report.company or '?'})")
            if item.job_level == "required":
                reasons.setdefault(item.skill, []).append(f"required by {report.title}")
    for skill in priority:
        priority[skill] *= 1.0 + demand.get(skill, 0.0)
        if demand.get(skill):
            reasons.setdefault(skill, []).append(f"asked for in {demand[skill]:.0%} of all listings")
    chosen = sorted(priority, key=lambda s: (-priority[s], s))[:max_steps]
    chosen_set = set(chosen)

    # topological order: a skill comes after its (chosen) prerequisites; priority breaks ties (Kahn's algorithm)
    deps = {s: [p for p in _entry(s)["prerequisites"] if p in chosen_set] for s in chosen}
    done: list[str] = []
    remaining = set(chosen)
    while remaining:
        ready = sorted((s for s in remaining if all(p in done for p in deps[s])), key=lambda s: (-priority[s], s))
        if not ready:  # a cycle in the KB: fall back to priority order
            ready = sorted(remaining, key=lambda s: (-priority[s], s))
        done.append(ready[0])
        remaining.discard(ready[0])

    steps = []
    for i, skill in enumerate(done, start=1):
        entry = _entry(skill)
        steps.append(PlanStep(i, skill, round(priority[skill], 3), reasons.get(skill, [])[:3], entry["prerequisites"],
                              entry["resources"][:3], entry.get("project"), entry["interview"][:3],
                              sorted(set(jobs.get(skill, [])))[:5]))  # fmt: skip
    return steps


def interview_prep(report: GapReport, max_questions: int = 10) -> list[tuple[str, str]]:
    """(skill, question) pairs: covered skills first (you will be tested on what you claim), then gaps."""
    out: list[tuple[str, str]] = []
    for group in (report.mastered, report.partial, report.missing_required):
        for item in group:
            for q in _entry(item.skill)["interview"][:2]:
                out.append((item.skill, q))
    return out[:max_questions]
