"""Intent routing for the career assistant: rule-based, bilingual (EN / FR), inspectable.

Why rules and not an LLM router: the intents are few and lexically distinct; a rule router is
deterministic, testable, instant, and never sends the question anywhere. Its accuracy is measured
on the assistant evaluation set (``evaluation/rag_bench.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from job_intel.match.cv_parser import fold
from job_intel.match.taxonomy import find_skills

INTENTS = ("top_jobs", "why_match", "missing_skills", "companies_with_skill", "compare", "interview",
           "projects", "market", "search")  # fmt: skip

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx))
    for name, rx in (
        ("compare", r"\bcompar\w*|\bversus\b|\bvs\.?\b|difference entre|differences? between"),
        ("interview", r"interview|entretien|prepare\w* (?:for|pour|mon|my)|questions? techniques?|technical questions?"),
        ("projects", r"\bprojects?\b|\bprojets?\b|portfolio|ameliorer mon profil|improve my profile|what should i build|que dois-je (?:realiser|construire)"),
        ("missing_skills", r"\bmissing\b|\bmanqu\w*|\bgap\b|lacun\w*|\black\b|what skills do i need|quelles competences (?:me )?(?:faut|manque)|learn next|apprendre"),
        ("why_match", r"\bwhy\b|pourquoi|correspond\w*|\bfit\b|\bmatch(?:es)?\b(?! engine)|explain|explique"),
        ("companies_with_skill", r"(?:which|what|quelles?) (?:companies|entreprises|societes|employers)|who (?:is )?hiring|qui recrute|entreprises? (?:qui )?(?:recherch|recrut)"),
        ("market", r"most (?:in[- ])?demand\w*|plus demande\w*|plus recherche\w*|trend\w*|tendance\w*|market|marche de l'emploi|statisti\w*|how many|combien"),
        ("top_jobs", r"\btop\b|\bbest\b|meilleur\w*|plus pertinent\w*|most relevant|recommend\w*|recommand\w*|which jobs|quelles offres|suited|adaptee?s?"),
    )
)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "premiere": 1, "premier": 1, "deuxieme": 2,
             "second_fr": 2, "troisieme": 3, "quatrieme": 4, "cinquieme": 5}  # fmt: skip
_JOB_REF = re.compile(r"(?:#|\b(?:job|offre|offer|poste|annonce|number|numero|n°|no\.?)\s*#?\s*)(\d{1,2})\b")
_NUMBER = re.compile(r"\b(\d{1,2})\b")
_THIS = re.compile(r"\b(?:this|that|cette|ce|cet)\s+(?:job|offer|offre|poste|role|annonce|opportunit\w*)\b")


@dataclass
class Route:
    intent: str
    language: str  # "fr" | "en"
    job_refs: list[int] = field(default_factory=list)  # 1-based positions in the recommendation list
    this_job: bool = False
    skills: list[str] = field(default_factory=list)
    count: int | None = None


def question_language(question: str) -> str:
    folded = fold(question)
    fr = len(re.findall(r"\b(?:quel\w*|pourquoi|comment|est-ce|mon|ma|mes|les|des|offres?|entreprises?|competences?|"
                        r"je|dois|cette|avec|pour)\b", folded))  # fmt: skip
    en = len(re.findall(r"\b(?:what|which|why|how|my|the|jobs?|companies|skills?|should|this|with|for|do|i)\b", folded))
    return "fr" if fr > en else "en"


def route(question: str) -> Route:
    folded = fold(question)
    intent = next((name for name, rx in _RULES if rx.search(folded)), "search")
    refs = [int(m.group(1)) for m in _JOB_REF.finditer(folded)]
    refs += [n for word, n in _ORDINALS.items() if re.search(rf"\b{word}\b", folded)]
    count = None
    if intent in {"top_jobs", "market", "companies_with_skill"}:
        nums = [int(n) for n in _NUMBER.findall(folded)]
        words = {"three": 3, "trois": 3, "five": 5, "cinq": 5, "ten": 10, "dix": 10}
        nums += [v for w, v in words.items() if re.search(rf"\b{w}\b", folded)]
        count = next((n for n in nums if 1 <= n <= 25), None)
    skills = find_skills(question)
    # "missing skills for this job" is a gap question even if it says "skills"; "companies looking for ML" needs a skill
    if intent == "companies_with_skill" and not skills:
        intent = "market"
    return Route(intent, question_language(question), sorted(dict.fromkeys(refs)), bool(_THIS.search(folded)), skills, count)
