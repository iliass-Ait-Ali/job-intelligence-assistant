"""Rule-based relevance: which listings are Data/ML/AI roles or data-flavoured internships.

Title patterns decide *what* a job is (deliberately strict: a bare "AI" in a title such as
"Legal AI Scaleup" or "AI Marketing" does not qualify). Descriptions only add to the score
for regular roles, so a keyword-stuffed body cannot make an unrelated job relevant.
Early-career roles (internship / working student) are kept only with a data/AI signal:
a data term in the title or tags, an AI term in a non-marketing/design/sales title, or
at least two strong data-science skills in the description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from job_intel.models import Job

_AI_PREFIX = r"(?:ai|a\.i\.|ki|genai|gen[\s-]?ai|llm|generative ai|artificial intelligence|agentic ai)"
_AI_ROLE = r"(?:engineer(?:ing)?|developer|scientist|researcher|architect|ingenieur\w*|entwickler\w*)"
_AI_SUFFIX = r"(?:applied |physical |generative )?ai(?:\s+(?:team|foundations|platform))?(?!\s+[a-z])"

ROLE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    category: tuple(re.compile(p, re.I) for p in patterns)
    for category, patterns in {
        "data_scientist": (r"data[\s-]*scien(?:ce|tist)", r"datenwissenschaft\w*", r"applied scientist"),
        "data_analyst": (
            r"data[\s-]*analy(?:st|sts|ist|tics|sis)",
            r"\bdatenanaly\w*",
            r"business intelligence",
            r"\bbi[\s-]*(?:analyst|developer|engineer)\b",
        ),
        "ml_engineer": (
            r"machine[\s-]*learning",
            r"maschinelles lernen",
            r"\bml\b",
            r"\bmlops\b",
            r"deep[\s-]*learning",
            r"computer[\s-]*vision",
        ),
        "ai_engineer": (
            rf"\b{_AI_PREFIX}\b[^,;|&]{{0,30}}?\b{_AI_ROLE}",
            # "Software Engineer - AI", "Staff Software Engineer: Applied AI"
            rf"\bengineer\s*[-:,–]\s*{_AI_SUFFIX}",
        ),
        "data_engineer": (
            r"data[\s-]*engineer\w*",
            r"dateningenieur\w*",
            r"analytics engineer",
            r"\betl\b",
            r"data platform",
            r"big data",
            r"data[\s-]*architect",
        ),
        "nlp": (
            r"\bnlp\b",
            r"natural[\s-]*language",
            r"computational linguist\w*",
            r"sprachverarbeitung",
            r"conversational ai",
            r"\bllms?\b",
            r"text mining",
        ),
    }.items()
}

EARLY_CAREER_TYPES = {"internship", "working_student"}

# Early-career signals, checked on title + tags.
_DATA_SIGNAL = re.compile(
    r"\b(?:data|daten|analytics|analytik|nlp|statistic\w*|statistik|python|sql)\b", re.I
)
_AI_SIGNAL = re.compile(r"\b(?:ai|ki|ia|artificial intelligence)\b", re.I)
# An AI mention in these titles is about the business function, not an AI role.
_NON_TECH_TITLE = re.compile(
    r"marketing|design|grafi[ck]|ontwerp|content|sales|vertrieb|recruit|\bhr\b|communication"
    r"|fundraising|founder|business development|growth|\bgtm\b|operations",
    re.I,
)

_STRONG_SKILLS = (
    "machine learning", "deep learning", "nlp", "natural language", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "data science", "data analysis", "llm", "tableau",
    "power bi", "spark", "airflow", "dbt", "statistics",
)  # fmt: skip
_COMMON_SKILLS = ("python", "sql", "etl")  # only add to the score


def _compile(terms: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(rf"(?<![\w-]){re.escape(t)}(?![\w-])", re.I) for t in terms)


_STRONG_RES = _compile(_STRONG_SKILLS)
_COMMON_RES = _compile(_COMMON_SKILLS)


@dataclass(frozen=True)
class Relevance:
    score: int
    categories: tuple[str, ...]  # matched role families, empty for signal-only internships
    early_career: bool


def _hits(patterns: tuple[re.Pattern[str], ...], text: str | None) -> int:
    return sum(1 for pattern in patterns if pattern.search(text or ""))


def _has_early_career_signal(job: Job, strong_skill_hits: int) -> bool:
    labels = " ".join((job.title, *job.tags))
    if _DATA_SIGNAL.search(labels):
        return True
    if _AI_SIGNAL.search(labels) and not _NON_TECH_TITLE.search(job.title):
        return True
    return strong_skill_hits >= 2


def assess(job: Job) -> Relevance | None:
    """Return a Relevance if the job qualifies, else None."""
    categories = tuple(
        category
        for category, patterns in ROLE_PATTERNS.items()
        if any(p.search(job.title) for p in patterns)
    )
    early_career = job.job_type in EARLY_CAREER_TYPES
    strong_hits = _hits(_STRONG_RES, job.description)

    if not categories and not (early_career and _has_early_career_signal(job, strong_hits)):
        return None

    score = 10 + 2 * (len(categories) - 1) if categories else 4
    score += 5 if early_career else 0
    score += min(strong_hits + _hits(_COMMON_RES, job.description), 5)
    return Relevance(score=score, categories=categories, early_career=early_career)


def filter_relevant(jobs: list[Job]) -> list[tuple[Job, Relevance]]:
    """Keep qualifying jobs, best first (score, then newest)."""
    scored = [(job, rel) for job in jobs if (rel := assess(job)) is not None]
    scored.sort(key=lambda pair: (pair[1].score, pair[0].published_at or ""), reverse=True)
    return scored
