"""Interpretable compatibility score (Module 4).

    score = 100 * Σ_i w_i c_i / Σ_{i known} w_i          coverage = Σ_{i known} w_i / Σ_i w_i

* ``c_i ∈ [0, 1]`` is one *component* (required skills, semantic similarity, role, ...). A component
  whose inputs are missing is **unknown** and is left out of BOTH sums: missing information neither
  helps nor hurts, and ``coverage`` says how much of the weight was actually observable.
* **Eliminatory vs ranking criteria.** Hard constraints the user chose (contract type, location, ...)
  are applied *before* scoring by the matcher: a clearly violated one excludes the job, with its
  reason. Everything else only moves the score.
* The score is a **compatibility index**, not a probability of being hired: no hiring outcome was
  ever observed, so nothing here is calibrated to such an event. It ranks and explains.

Weights live in ``config/scoring.yaml`` (defaults below); ``evaluation/calibrate_scoring.py`` fits
them against relevance judgments with leave-one-candidate-out cross-validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from job_intel.match.models import DEGREE_ORDER, CandidateProfile
from job_intel.match.requirements import JobRequirements

DEFAULT_WEIGHTS: dict[str, float] = {
    "required_skills": 0.30,   # what the listing explicitly asks for: the core of fit
    "semantic": 0.20,          # whole-text similarity: catches vocabulary the taxonomy misses
    "role": 0.15,              # is it the kind of job the candidate targets?
    "experience": 0.10,        # years asked vs dated experience in the CV
    "preferred_skills": 0.08,  # nice-to-haves: bonus, smaller weight
    "education": 0.05,
    "languages": 0.05,
    "location": 0.04,          # soft preference (hard ones already excluded the job)
    "contract": 0.03,
}  # fmt: skip
# Cosine similarities of multilingual-e5-base between a CV query and a listing chunk live in a narrow
# band; these bounds map it to [0, 1]. They are the 5th / 95th percentiles measured on this corpus
# (see docs/learn/04_matching_scoring.md), not magic numbers.
SEMANTIC_BOUNDS = (0.78, 0.89)
TRANSFERABLE_CREDIT = 0.5  # a related skill counts half: evidence of a neighbour, not of the skill


@dataclass
class Component:
    name: str
    value: float | None  # None = unknown
    weight: float
    detail: str

    @property
    def known(self) -> bool:
        return self.value is not None


@dataclass
class CompatibilityScore:
    score: float  # 0-100
    coverage: float  # 0-1, share of the weight that was observable
    components: list[Component] = field(default_factory=list)

    @property
    def band(self) -> str:
        if self.score >= 70:
            return "high"
        return "medium" if self.score >= 45 else "low"

    def contribution(self, name: str) -> float:
        """Points (out of 100) this component adds to the score."""
        known = sum(c.weight for c in self.components if c.known) or 1.0
        for c in self.components:
            if c.name == name and c.known:
                return 100.0 * c.weight * float(c.value) / known
        return 0.0


def load_weights(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return dict(DEFAULT_WEIGHTS)
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in (data.get("weights") or {}).items():
        if key not in DEFAULT_WEIGHTS:
            raise ValueError(f"unknown scoring component {key!r}; allowed: {sorted(DEFAULT_WEIGHTS)}")
        if float(value) < 0:
            raise ValueError(f"weight of {key} must be >= 0")
        weights[key] = float(value)
    return weights


def _skill_component(skills, weight: float, name: str) -> Component:
    if not skills:
        return Component(name, None, weight, "the listing marks no such skills")
    exact = sum(s.status == "exact" for s in skills)
    transferable = sum(s.status == "transferable" for s in skills)
    value = (exact + TRANSFERABLE_CREDIT * transferable) / len(skills)
    return Component(name, value, weight, f"{exact} exact + {transferable} transferable of {len(skills)}")


def _semantic_component(similarity: float | None, weight: float) -> Component:
    if similarity is None:
        return Component("semantic", None, weight, "no semantic retrieval score")
    lo, hi = SEMANTIC_BOUNDS
    value = min(1.0, max(0.0, (similarity - lo) / (hi - lo)))
    return Component("semantic", value, weight, f"best chunk cosine {similarity:.3f} (mapped from [{lo}, {hi}])")


def _status_component(name: str, constraints, weight: float, partial: float | None = None) -> Component:
    rel = [c for c in constraints if c.name == name]
    if not rel:
        return Component(name, None, weight, "no requirement or preference")
    if any(c.status == "violated" for c in rel):
        value = partial if partial is not None else 0.0
        return Component(name, value, weight, next(c.detail for c in rel if c.status == "violated"))
    if all(c.status == "satisfied" for c in rel):
        return Component(name, 1.0, weight, rel[0].detail)
    known = [c for c in rel if c.status != "unknown"]
    if known:  # e.g. two languages asked, one satisfied, one unknown
        return Component(name, 1.0, weight, "; ".join(c.detail for c in known))
    return Component(name, None, weight, rel[0].detail)


def _experience_partial(profile: CandidateProfile, reqs: JobRequirements) -> float | None:
    years, bounds = reqs.required_years, profile.experience_months_bounds()
    if not years or bounds is None:
        return None
    return min(1.0, bounds[1] / (12.0 * years))


def _education_partial(profile: CandidateProfile, reqs: JobRequirements) -> float | None:
    if reqs.degree is None:
        return None
    have = profile.highest_degree_level or profile.in_progress_degree_level
    if have is None:
        return None
    return (DEGREE_ORDER[have] + 1) / (DEGREE_ORDER[reqs.degree[0]] + 1)


def _language_component(constraints, weight: float) -> Component:
    rel = [c for c in constraints if c.name == "language"]
    if not rel or all(c.status == "unknown" for c in rel):
        return Component("languages", None, weight, rel[0].detail if rel else "no language requirement")
    known = [c for c in rel if c.status != "unknown"]
    value = sum(c.status == "satisfied" for c in known) / len(known)
    return Component("languages", value, weight, "; ".join(c.detail for c in known))


def score_match(match, profile: CandidateProfile, weights: Mapping[str, float] | None = None) -> CompatibilityScore:
    """`match` is a ``matcher.JobMatch``."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    required = [s for s in match.skills if s.job_level == "required"]
    base = required or [s for s in match.skills if s.job_level == "mentioned"]
    preferred = [s for s in match.skills if s.job_level == "preferred"]
    role_value = {"match": 1.0, "no_match": 0.0}.get(match.role_status)
    comps = [
        _skill_component(base, w["required_skills"], "required_skills"),
        _semantic_component(match.retrieval.best_similarity, w["semantic"]),
        Component("role", role_value, w["role"],
                  f"title matches {', '.join(match.matched_roles)}" if role_value == 1.0
                  else ("title is not one of your target roles" if role_value == 0.0 else "no target role given")),
        _status_component("experience", match.constraints, w["experience"], _experience_partial(profile, match.requirements)),
        _skill_component(preferred, w["preferred_skills"], "preferred_skills"),
        _status_component("education", match.constraints, w["education"], _education_partial(profile, match.requirements)),
        _language_component(match.constraints, w["languages"]),
        _status_component("location", match.constraints, w["location"]),
        _status_component("job_type", match.constraints, w["contract"]),
    ]  # fmt: skip
    comps[-1].name = "contract"
    total = sum(c.weight for c in comps) or 1.0
    known_w = sum(c.weight for c in comps if c.known)
    score = 100.0 * sum(c.weight * float(c.value) for c in comps if c.known) / known_w if known_w else 0.0
    return CompatibilityScore(round(score, 1), round(known_w / total, 2), comps)
