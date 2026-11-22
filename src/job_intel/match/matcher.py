"""Transparent CV -> job matching.

Pipeline (each step is a plain function that can be tested and inspected):

A. ``build_queries``      profile + preferences -> labelled search queries (roles, skills, each experience,
                          each project, summary), so nothing in the CV is reduced to one keyword bag.
B. ``retrieve_candidates`` every query runs on BOTH Phase 2 engines (semantic + BM25); the ranked lists are
                          fused with Reciprocal Rank Fusion (RRF) into a broad candidate pool.
C. ``evaluate_constraints`` job type / location / remote / experience / education / language, each
                          ``satisfied | violated | unknown``. Missing information is ``unknown``, never a
                          mismatch. Only constraints the user marked hard, and only when ``violated``, exclude
                          a job; excluded jobs are listed with the reason, not silently dropped.
D. ``rank``               an ordinal evidence level (strong / moderate / weak) from explicit rules, then
                          counts of exact / transferable / not-evidenced skills, then retrieval rank.
                          There is NO percentage score: the numbers would not be calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from job_intel.match.models import DEGREE_ORDER, CandidateProfile, Evidence, Preferences
from job_intel.match.requirements import JobRequirements, extract_requirements
from job_intel.match.taxonomy import closest_related
from job_intel.relevance import ROLE_PATTERNS
from job_intel.search.filters import IndexedJob
from job_intel.search.locations import location_matches, remote_allows
from job_intel.search.retrieve import BM25, SEMANTIC, Searcher

Status = Literal["satisfied", "violated", "unknown"]
RRF_K = 60

# Heuristic thresholds for the evidence level. They are documented constants, not calibrated values.
STRONG_MIN_EXACT = 3
STRONG_MIN_COVERAGE = 0.5
MODERATE_MIN_EXACT = 2
LEVEL_RANK = {"strong": 0, "moderate": 1, "weak": 2, "insufficient_information": 3}
ROLE_RANK = {"match": 0, "unknown": 1, "no_match": 2}


# ---------------------------------------------------------------------------------------------
# A. candidate representation


@dataclass(frozen=True)
class CandidateQuery:
    text: str
    kind: Literal["role", "skills", "experience", "project", "summary"]
    label: str  # which CV item produced it, for traceability


def build_queries(profile: CandidateProfile, prefs: Preferences, max_queries: int = 12) -> list[CandidateQuery]:
    roles = prefs.target_roles or profile.target_roles
    demonstrated = [s.name for s in profile.skills if s.context == "demonstrated"]
    top = (demonstrated or profile.technical_skills)[:8]
    hint = "internship stage" if "internship" in prefs.job_types else ""
    queries: list[CandidateQuery] = []
    for role in roles[:3]:
        queries.append(CandidateQuery(" ".join([role, *top[:4], hint]).strip(), "role", f"role: {role}"))
    if top:
        queries.append(CandidateQuery(" ".join([*top, hint]).strip(), "skills", "skills"))
    for exp in profile.experiences[:4]:
        first = exp.bullets[0] if exp.bullets else ""
        text = " ".join(filter(None, [exp.title, " ".join(exp.skills[:6]), first[:200]]))
        if text.strip():
            queries.append(CandidateQuery(text, "experience", f"experience: {exp.title or exp.organization or '?'}"))
    for proj in profile.projects[:3]:
        text = " ".join(filter(None, [proj.name, " ".join(proj.skills[:6]), (proj.description[0] if proj.description else "")[:200]]))
        if text.strip():
            queries.append(CandidateQuery(text, "project", f"project: {proj.name or '?'}"))
    if profile.summary:
        queries.append(CandidateQuery(profile.summary[:300], "summary", "summary"))
    seen, unique = set(), []
    for q in queries:
        key = q.text.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[:max_queries]


# ---------------------------------------------------------------------------------------------
# B. candidate retrieval


@dataclass
class RetrievalTrace:
    rrf: float = 0.0
    best_similarity: float | None = None  # best cosine similarity of any chunk for any query
    best_semantic_rank: int | None = None
    best_bm25_rank: int | None = None
    found_by: list[str] = field(default_factory=list)  # query labels that retrieved the job


def retrieve_candidates(searcher: Searcher, queries: list[CandidateQuery], *, k_per_query: int = 30,
                        pool_size: int = 80, allowed_ids: Collection[str] | None = None) -> dict[str, RetrievalTrace]:  # fmt: skip
    traces: dict[str, RetrievalTrace] = {}
    for query in queries:
        for engine in (SEMANTIC, BM25):
            response = searcher.search(query.text, engine=engine, k=k_per_query, allowed_ids=allowed_ids)
            for hit in response.hits:
                t = traces.setdefault(hit.job.job_id, RetrievalTrace())
                t.rrf += 1.0 / (RRF_K + hit.rank)
                if query.label not in t.found_by:
                    t.found_by.append(query.label)
                if engine == SEMANTIC:
                    t.best_similarity = max(t.best_similarity or -1.0, hit.score)
                    t.best_semantic_rank = min(t.best_semantic_rank or hit.rank, hit.rank)
                else:
                    t.best_bm25_rank = min(t.best_bm25_rank or hit.rank, hit.rank)
    ordered = sorted(traces.items(), key=lambda kv: (-kv[1].rrf, kv[0]))[:pool_size]
    return dict(ordered)


# ---------------------------------------------------------------------------------------------
# C. constraints


@dataclass
class ConstraintResult:
    name: str
    status: Status
    hard: bool
    detail: str


EARLY_CAREER = {"internship", "working_student"}
REMOTE_BOARDS = {"himalayas", "remotive"}  # sources whose location means "where candidates may work from"


def _job_type_constraint(job: IndexedJob, reqs: JobRequirements, prefs: Preferences) -> ConstraintResult | None:
    if not prefs.job_types:
        return None
    hard = "job_type" in prefs.hard_constraints
    if job.job_type is None:
        # No structured type: fall back to what the TITLE says explicitly (never to the description's tone).
        wants_only_early = set(prefs.job_types) <= EARLY_CAREER
        if reqs.seniority in {"senior", "lead_or_above"} and wants_only_early:
            return ConstraintResult("job_type", "violated", hard,
                                    f"The title signals a {reqs.seniority.replace('_', ' ')} role; you are looking for {', '.join(prefs.job_types)}.")  # fmt: skip
        if reqs.seniority == "intern":
            if set(prefs.job_types) & EARLY_CAREER:
                return ConstraintResult("job_type", "satisfied", hard,
                                        "The listing states no job type, but its title signals an internship/student role.")  # fmt: skip
            return ConstraintResult("job_type", "violated", hard, "The title signals an internship/student role.")
        return ConstraintResult("job_type", "unknown", hard, "The listing does not state a job type.")
    if job.job_type in prefs.job_types:
        return ConstraintResult("job_type", "satisfied", hard, f"Job type '{job.job_type}' is among your preferences.")
    if job.job_type == "permanent" and "full_time" in prefs.job_types:
        return ConstraintResult("job_type", "unknown", hard, "Permanent contract; working hours are not stated.")
    return ConstraintResult("job_type", "violated", hard,
                            f"Job type '{job.job_type}' is not among your preferences ({', '.join(prefs.job_types)}).")  # fmt: skip


def _location_constraint(job: IndexedJob, prefs: Preferences) -> ConstraintResult | None:
    if not prefs.locations:
        return None
    hard = "location" in prefs.hard_constraints
    if job.remote is True and job.source in REMOTE_BOARDS:
        # On remote-only boards the location field lists where candidates may live, not an office.
        verdicts = [remote_allows(job.location, wanted) for wanted in prefs.locations]
        if any(v is True for v in verdicts):
            return ConstraintResult("location", "satisfied", hard, f"Remote job open to candidates in: {job.location}.")
        if verdicts and all(v is False for v in verdicts):
            return ConstraintResult("location", "violated", hard,
                                    f"Remote job restricted to {job.location}; you want {', '.join(prefs.locations)}.")  # fmt: skip
        return ConstraintResult("location", "unknown", hard, f"Remote job; allowed locations: {job.location or 'not stated'}.")
    if job.remote is True:
        return ConstraintResult("location", "satisfied", hard, "The listing states remote work.")
    if not job.location:
        return ConstraintResult("location", "unknown", hard, "The listing does not state a location.")
    if any(location_matches(job.location, wanted) for wanted in prefs.locations):
        return ConstraintResult("location", "satisfied", hard, f"Location '{job.location}' matches your preferences.")
    return ConstraintResult("location", "violated", hard,
                            f"Location '{job.location}' does not match {', '.join(prefs.locations)}.")  # fmt: skip


def _remote_constraint(job: IndexedJob, prefs: Preferences) -> ConstraintResult | None:
    if prefs.remote == "any":
        return None
    hard = prefs.remote == "only" or "remote" in prefs.hard_constraints
    if job.remote is True:
        return ConstraintResult("remote", "satisfied", hard, "The listing states remote work.")
    # Phase 1 stores remote=false for "not stated", so absence is never proof of on-site work.
    return ConstraintResult("remote", "unknown", hard, "The listing does not state that remote work is possible.")


def _experience_constraint(profile: CandidateProfile, reqs: JobRequirements, prefs: Preferences) -> ConstraintResult | None:
    years = reqs.required_years
    if years is None:
        return None
    hard = "experience" in prefs.hard_constraints
    need = years * 12
    bounds = profile.experience_months_bounds()
    evidence = next((e for y, e, lvl in reqs.years if y == years and lvl != "preferred"), "")
    if bounds is None:
        return ConstraintResult("experience", "unknown", hard, f"The listing asks for {years}+ years; the CV has no dated experience. “{evidence[:120]}”")
    lo, hi = bounds
    if lo >= need:
        return ConstraintResult("experience", "satisfied", hard, f"Dated jobs in the CV total at least {lo} months; the listing asks for {years} years.")
    if hi < need:
        return ConstraintResult("experience", "violated", hard, f"The listing asks for {years}+ years; the CV's dates add up to at most {hi} months.")
    return ConstraintResult("experience", "unknown", hard, f"The listing asks for {years}+ years; the CV's dates give {lo}-{min(hi, 9999)} months (depends on whether internships count).")


def _education_constraint(profile: CandidateProfile, reqs: JobRequirements, prefs: Preferences) -> ConstraintResult | None:
    if reqs.degree is None:
        return None
    required, _evidence, _lvl, enrolled_ok = reqs.degree
    hard = "education" in prefs.hard_constraints
    done, ongoing = profile.highest_degree_level, profile.in_progress_degree_level
    need = DEGREE_ORDER[required]
    if done and DEGREE_ORDER[done] >= need:
        return ConstraintResult("education", "satisfied", hard, f"The CV states a completed {done} degree; the listing asks for {required}.")
    if ongoing and DEGREE_ORDER[ongoing] >= need:
        if enrolled_ok:
            return ConstraintResult("education", "satisfied", hard, f"The CV states an in-progress {ongoing}; the listing accepts enrolled students.")
        return ConstraintResult("education", "unknown", hard, f"The CV states an in-progress {ongoing}; the listing asks for {required} without mentioning students.")
    if done or ongoing:
        return ConstraintResult("education", "violated", hard, f"Highest stated level is {done or ongoing}; the listing asks for {required}.")
    return ConstraintResult("education", "unknown", hard, f"The listing asks for a {required} degree; no degree level could be read from the CV.")


def _language_constraints(profile: CandidateProfile, reqs: JobRequirements, prefs: Preferences) -> list[ConstraintResult]:
    hard = "language" in prefs.hard_constraints
    out = []
    known = {s.language: s for s in profile.spoken_languages}
    for need in reqs.languages:
        if need.level == "preferred":
            continue
        if not known:
            out.append(ConstraintResult("language", "unknown", hard, f"The listing mentions {need.language}; the CV lists no spoken languages."))
        elif need.language in known:
            lvl = known[need.language].level_text
            out.append(ConstraintResult("language", "satisfied", hard,
                                        f"The CV lists {need.language}{f' ({lvl})' if lvl else ''}; the level is not compared automatically."))  # fmt: skip
        else:
            out.append(ConstraintResult("language", "violated", hard,
                                        f"The listing mentions {need.language}; the CV lists {', '.join(known)} but not {need.language}."))  # fmt: skip
    return out


def evaluate_constraints(job: IndexedJob, reqs: JobRequirements, profile: CandidateProfile, prefs: Preferences) -> list[ConstraintResult]:
    results = [
        _job_type_constraint(job, reqs, prefs), _location_constraint(job, prefs), _remote_constraint(job, prefs),
        _experience_constraint(profile, reqs, prefs), _education_constraint(profile, reqs, prefs),
    ]  # fmt: skip
    return [r for r in results if r] + _language_constraints(profile, reqs, prefs)


# ---------------------------------------------------------------------------------------------
# D. skills, role relevance, ranking


@dataclass
class SkillMatch:
    skill: str
    status: Literal["exact", "transferable", "not_evidenced"]
    job_level: Literal["required", "preferred", "mentioned"]
    job_evidence: str
    cv_evidence: list[Evidence] = field(default_factory=list)
    via: list[str] = field(default_factory=list)  # CV skills offered as transferable evidence


def match_skills(profile: CandidateProfile, reqs: JobRequirements) -> list[SkillMatch]:
    cv = {s.name: s for s in profile.skills}
    out = []
    for js in reqs.skills:
        if js.name in cv:
            out.append(SkillMatch(js.name, "exact", js.level, js.evidence, cv[js.name].evidence))
            continue
        via = closest_related(js.name, set(cv))
        if via:
            evidence = [e for name in via for e in cv[name].evidence[:1]]
            out.append(SkillMatch(js.name, "transferable", js.level, js.evidence, evidence, via[:3]))
        else:
            out.append(SkillMatch(js.name, "not_evidenced", js.level, js.evidence))
    return out


def role_categories(text: str) -> set[str]:
    return {cat for cat, patterns in ROLE_PATTERNS.items() if any(p.search(text) for p in patterns)}


def role_relevance(title: str, roles: list[str]) -> tuple[Literal["match", "no_match", "unknown"], list[str]]:
    if not roles:
        return "unknown", []
    job_cats = role_categories(title)
    matched = []
    for role in roles:
        cats = role_categories(role)
        if (cats and cats & job_cats) or (not cats and role.casefold() in title.casefold()):
            matched.append(role)
    return ("match", matched) if matched else ("no_match", [])


@dataclass
class JobMatch:
    job: IndexedJob
    record: Mapping[str, Any]
    retrieval: RetrievalTrace
    requirements: JobRequirements
    skills: list[SkillMatch]
    constraints: list[ConstraintResult]
    role_status: str
    matched_roles: list[str]
    evidence_level: str
    points: float
    excluded_because: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def skills_with(self, status: str, *levels: str) -> list[SkillMatch]:
        return [s for s in self.skills if s.status == status and (not levels or s.job_level in levels)]

    @property
    def counts(self) -> dict[str, int]:
        base = self.base_skills
        return {st: sum(1 for s in base if s.status == st) for st in ("exact", "transferable", "not_evidenced")}

    @property
    def soft_violations(self) -> int:
        """Stated requirements the CV clearly does not meet (hard ones were already excluded)."""
        return sum(c.status == "violated" for c in self.constraints)

    @property
    def coverage_bucket(self) -> float:
        """Share of the base skills with exact CV evidence, rounded to 0.1 (a tie-break, not a score)."""
        base = self.base_skills
        return round(sum(s.status == "exact" for s in base) / len(base), 1) if base else 0.0

    @property
    def base_skills(self) -> list[SkillMatch]:
        """The skills the level is judged on: those marked required, or all listed ones when nothing is marked."""
        required = [s for s in self.skills if s.job_level == "required"]
        return required or self.skills


def _evidence_level(skills: list[SkillMatch], role_status: str) -> tuple[str, float]:
    required = [s for s in skills if s.job_level == "required"]
    base = required or skills
    exact = sum(s.status == "exact" for s in base)
    transferable = sum(s.status == "transferable" for s in base)
    points = exact + 0.5 * transferable + 0.25 * sum(s.status == "exact" for s in skills if s not in base)
    if not skills and role_status != "match":
        return "insufficient_information", 0.0
    coverage = exact / len(base) if base else 0.0
    if role_status != "no_match" and exact >= STRONG_MIN_EXACT and coverage >= STRONG_MIN_COVERAGE:
        return "strong", points
    if (
        (role_status == "match" and exact >= 1)
        or (role_status == "unknown" and exact >= MODERATE_MIN_EXACT)
        or (role_status == "no_match" and exact >= STRONG_MIN_EXACT)
        or (role_status != "no_match" and exact + transferable >= 3)
    ):
        return "moderate", points
    return "weak", points


@dataclass
class MatchResult:
    matches: list[JobMatch]
    excluded: list[JobMatch]
    queries: list[CandidateQuery]
    pool_size: int
    warnings: list[str] = field(default_factory=list)
    corpus_size: int = 0  # jobs in the index
    eligible_total: int = 0  # jobs in the whole index that no hard constraint clearly excludes


@lru_cache(maxsize=4096)
def _requirements(title: str, description: str | None) -> JobRequirements:
    return extract_requirements(title, description)


def _assess(job_id: str, trace: RetrievalTrace, profile: CandidateProfile, prefs: Preferences, searcher: Searcher,
            roles: list[str]) -> JobMatch:  # fmt: skip
    """Everything we can say about one job for one candidate."""
    job, record = searcher.table[job_id], searcher.record(job_id) or {}
    reqs = _requirements(job.title, record.get("description"))
    skills = match_skills(profile, reqs)
    role_status, matched_roles = role_relevance(job.title, roles)
    level, points = _evidence_level(skills, role_status)
    constraints = evaluate_constraints(job, reqs, profile, prefs)
    reasons = [f"{c.name}: {c.detail}" for c in constraints if c.hard and c.status == "violated"]
    if prefs.remote == "only" and job.remote is not True:
        reasons.append("remote: you asked for remote-only and the listing does not state remote work.")
    notes = []
    if not reqs.has_requirements_section:
        notes.append("The listing has no recognisable requirements section; all skills it mentions are treated as the base set.")
    if reqs.ignored_instruction_lines:
        notes.append(f"{reqs.ignored_instruction_lines} line(s) in the listing look like instructions to an AI system and were ignored.")
    if reqs.seniority in {"senior", "lead_or_above"}:
        notes.append("The job title signals a senior-level role.")
    return JobMatch(job, record, trace, reqs, skills, constraints, role_status, matched_roles, level, points, reasons, notes)


def _sort_key(m: JobMatch):
    return (LEVEL_RANK[m.evidence_level], ROLE_RANK[m.role_status], m.soft_violations, -m.coverage_bucket, -m.points,
            -m.retrieval.rrf, m.job.job_id)  # fmt: skip


def match_candidate(profile: CandidateProfile, prefs: Preferences, searcher: Searcher, *,
                    k_per_query: int = 30, pool_size: int = 80, prefilter: bool = True) -> MatchResult:  # fmt: skip
    """Rank jobs for a candidate.

    ``prefilter=True`` (default) applies the HARD constraints to the whole index BEFORE retrieval, so the
    candidate pool is filled with jobs that are not clearly excluded. ``prefilter=False`` is the Phase 3
    behaviour (retrieve first, exclude afterwards), kept for comparison: with a small pool it wasted most
    slots on jobs that were then excluded, so eligible jobs outside the top-N never surfaced.
    """
    queries = build_queries(profile, prefs)
    result = MatchResult([], [], queries, 0, corpus_size=len(searcher.table))
    if not queries:
        result.warnings.append("The profile has no roles, skills, experiences or projects to search with.")
        return result
    roles = prefs.target_roles or profile.target_roles
    allowed: set[str] | None = None
    excluded_all: list[JobMatch] = []
    if prefilter:
        allowed = set()
        for job_id in searcher.table:
            assessed = _assess(job_id, RetrievalTrace(), profile, prefs, searcher, roles)
            (excluded_all if assessed.excluded_because else []).append(assessed)
            if not assessed.excluded_because:
                allowed.add(job_id)
        result.eligible_total = len(allowed)
    traces = retrieve_candidates(searcher, queries, k_per_query=k_per_query, pool_size=pool_size, allowed_ids=allowed)
    result.pool_size = len(traces)
    pooled = [_assess(job_id, trace, profile, prefs, searcher, roles) for job_id, trace in traces.items()]
    if prefilter:
        result.excluded = excluded_all
        ranked = pooled
    else:
        result.excluded = [m for m in pooled if m.excluded_because]
        ranked = [m for m in pooled if not m.excluded_because]
    ranked.sort(key=_sort_key)
    result.matches = ranked[: prefs.max_results]
    if len(ranked) > prefs.max_results:
        result.warnings.append(f"{len(ranked) - prefs.max_results} more eligible candidates were not shown (max_results={prefs.max_results}).")
    return result
