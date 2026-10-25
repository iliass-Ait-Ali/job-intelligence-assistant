"""Extract explicit requirements from a job listing (skills, years, degree, languages, seniority).

Rule-based and quote-preserving: each requirement keeps the listing line it came from, so a
recommendation can always point at the sentence that justifies it. Nothing is guessed from
company names or industry: if the listing does not say it, it is not a requirement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from job_intel.match.cv_parser import LANGUAGE_NAMES, fold
from job_intel.match.models import DEGREE_ORDER
from job_intel.match.taxonomy import BY_NAME, find_skills
from job_intel.rag.sanitize import looks_like_injection

Level = Literal["required", "preferred", "mentioned"]
_RANK = {"required": 0, "preferred": 1, "mentioned": 2}

_REQ_HEAD = re.compile(
    r"requirement|qualification|\bprofil\b|what you(?:'ll)? (?:bring|need)|your profile|you have|you bring|must[- ]have|required|"
    r"who you are|about you|looking for|what we expect|profil recherche|votre profil|vous avez|vous etes|"
    r"competences requises|competences attendues|prerequis|ce que nous recherchons|ihr profil|anforderung|das bringst du mit|"
    r"dein profil|qualifikation|wir erwarten|was du mitbringst|ihre qualifikation|your background|experience required|"
    r"skills? (?:&|and) experience|minimum", re.I)  # fmt: skip
_PREF_HEAD = re.compile(
    r"nice[- ]to[- ]have|bonus|preferred|good to have|would be a plus|a plus|ideally|souhaite|idealement|apprecie|"
    r"von vorteil|wunschenswert|pluspunkte|pluspunkt", re.I)  # fmt: skip
_SKIP_HEAD = re.compile(
    r"benefit|what we offer|we offer|perks|why join|about us|about the company|who we are|our (?:story|mission|values)|"
    r"ce que nous offrons|nous offrons|avantages|a propos de nous|wir bieten|unser angebot|equal opportunit|"
    r"how to apply|application process|diversity|salary|remuneration|compensation|pourquoi nous rejoindre", re.I)  # fmt: skip
_RESP_HEAD = re.compile(
    r"responsibilit|what you(?:'ll)? do|your role|the role|your mission|missions?\b|tasks|aufgaben|your tasks|"
    r"vos missions|le poste|job description|description du poste|stellenbeschreibung|the opportunity|day to day", re.I)  # fmt: skip
_PREF_CUE = re.compile(r"nice to have|is a plus|a plus\b|bonus|preferred|ideally|appreci|souhait|idealement|von vorteil|wunschenswert|desirable|beneficial", re.I)
_REQ_CUE = re.compile(r"\b(?:must|required|requirement|essential|mandatory|obligatoire|indispensable|zwingend|unbedingt)\b", re.I)

_EXP_CUE = re.compile(r"experience|experiences|erfahrung|exp\b|background|expertise|practice", re.I)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "ten": 10,
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six_": 6, "sept": 7, "huit": 8, "dix": 10,
    "ein": 1, "eine": 1, "zwei": 2, "drei": 3, "vier": 4, "funf": 5, "sechs": 6, "acht": 8, "zehn": 10,
}  # fmt: skip
_YEARS = re.compile(
    r"(?<![\w.])(\d{1,2}|" + "|".join(k for k in _NUMBER_WORDS if not k.endswith("_")) + r")\s*(?:\+|plus)?\s*"
    r"(?:(?:-|–|to|a|à|bis)\s*(\d{1,2})\s*)?(?:years?|yrs?|ans|jahre?n?)\b", re.I)  # fmt: skip

_DEGREE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("phd", re.compile(r"\bph\.?\s?d\b|doctorate|doctorat\b|promotion\b", re.I)),
    ("master", re.compile(r"\bmaster['’]?s?\b|\bmsc\b|\bm\.sc\b|\bmba\b|bac ?\+ ?5|diplome d'ingenieur|ecole d'ingenieur|"
                          r"masterabschluss|diplom-ingenieur|ingenieurdiplom|diplome d ingenieur", re.I)),
    ("bachelor", re.compile(r"\bbachelor['’]?s?\b|\bbsc\b|\bb\.sc\b|\blicence\b|bac ?\+ ?3|bachelorabschluss", re.I)),
)
_DEGREE_CUE = re.compile(r"degree|diplom|bac ?\+|bachelor|master['’]?s|msc|bsc|phd|ph\.d|doctor|licence|studium|studies|"
                         r"studied|etudes|formation|abschluss|university|universit|ecole|school|graduat|enrolled|student", re.I)

_LANGUAGE_CUE = re.compile(
    r"fluent|fluency|native|proficien|business[- ]level|\bc1\b|\bc2\b|\bb2\b|courant|bilingue|maternelle|verhandlungssicher|"
    r"fliessend|muttersprache|command|written|spoken|speaking|parle|sprechen|sehr gute|\bgute\b|kenntnisse|excellent|"
    r"strong|\bgood\b|knowledge|skills|maitrise|niveau|vloeiend", re.I)  # fmt: skip
_ENROLLED = re.compile(r"enrolled|currently|actuellement|en cours|etudiant|student|derniere annee|final year|preparez|"
                       r"last year|studierende|immatricul", re.I)
_BOILERPLATE = re.compile(r"find more .{0,40}jobs|on arbeitnow|arbeitnow\.com|remotive\.com", re.I)
_LANG_WORD = {fold(k): v for k, v in LANGUAGE_NAMES.items()}

_SENIORITY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("intern", re.compile(r"\bintern(?:ship)?\b|\bstage\b|stagiaire|praktik\w*|werkstudent\w*|working student|alternan\w*|apprenti\w*|apprentice", re.I)),
    ("junior", re.compile(r"\bjunior\b|\bjr\.?\b|entry[- ]level|graduate|debutant|berufseinsteiger|nachwuchs|trainee", re.I)),
    ("lead_or_above", re.compile(r"\blead\b|principal|\bstaff\b|head of|director|\bmanager\b|\bvp\b|architect", re.I)),
    ("senior", re.compile(r"\bsenior\b|\bsr\.?\b|confirme|experienced|experimente|erfahren|\bmid[- ]senior\b", re.I)),
)


@dataclass
class JobSkill:
    name: str
    level: Level
    evidence: str


@dataclass
class LanguageReq:
    language: str
    level: Level
    evidence: str


@dataclass
class JobRequirements:
    skills: list[JobSkill] = field(default_factory=list)
    years: list[tuple[int, str, Level]] = field(default_factory=list)  # (years, listing line, level)
    # (minimum degree level, listing line, requirement level, listing accepts people still enrolled)
    degree: tuple[str, str, Level, bool] | None = None
    languages: list[LanguageReq] = field(default_factory=list)
    seniority: str | None = None
    has_requirements_section: bool = False
    ignored_instruction_lines: int = 0  # lines that look like instructions to an AI; never used as requirements

    @property
    def required_years(self) -> int | None:
        values = [y for y, _, lvl in self.years if lvl != "preferred"]
        return max(values) if values else None

    def by_level(self, *levels: Level) -> list[JobSkill]:
        return [s for s in self.skills if s.level in levels]


def _evidence_for(text: str, skill: str, limit: int = 300) -> str:
    """The listing line, shortened around the mention of `skill` so the quoted evidence still contains it."""
    if len(text) <= limit:
        return text
    hit = BY_NAME[skill].pattern.search(text) if skill in BY_NAME else None
    start = 0 if hit is None else max(0, min(hit.start() - limit // 3, len(text) - limit))
    return ("…" if start else "") + text[start : start + limit].strip() + ("…" if start + limit < len(text) else "")


def _heading_kind(line: str) -> str | None:
    text = line.strip()
    if not text or len(text) > 80 or text.startswith("- "):
        return None
    words = text.split()
    if not (text.endswith(":") or (len(words) <= 6 and not re.search(r"[.!?]$", text))):
        return None
    folded = fold(text)
    for kind, rx in (("skip", _SKIP_HEAD), ("pref", _PREF_HEAD), ("req", _REQ_HEAD), ("resp", _RESP_HEAD)):
        if rx.search(folded):
            return kind
    return None


def seniority_of(title: str) -> str | None:
    folded = fold(title)
    for name, rx in _SENIORITY:
        if rx.search(folded):
            return name
    return None


def _level_for(section: str | None, line: str) -> Level:
    if _PREF_CUE.search(fold(line)):
        return "preferred"
    if _REQ_CUE.search(line):
        return "required"
    return {"req": "required", "pref": "preferred"}.get(section or "", "mentioned")


def extract_requirements(title: str, description: str | None) -> JobRequirements:
    req = JobRequirements(seniority=seniority_of(title))
    best: dict[str, JobSkill] = {}
    section: str | None = None
    lines = [ln.strip() for ln in (description or "").split("\n")]
    # the title itself often names the core skill ("Machine Learning Engineer")
    for name in find_skills(title):
        best[name] = JobSkill(name, "mentioned", title)
    for line in lines:
        if not line:
            continue
        if (kind := _heading_kind(line)) is not None:
            section = kind
            req.has_requirements_section |= kind == "req"
            continue
        if section == "skip" or _BOILERPLATE.search(line):
            continue
        if looks_like_injection(line):
            req.ignored_instruction_lines += 1
            continue
        text = line[2:].strip() if line.startswith("- ") else line
        level = _level_for(section, text)

        for name in find_skills(text):
            current = best.get(name)
            if current is None or _RANK[level] < _RANK[current.level]:
                best[name] = JobSkill(name, level, _evidence_for(text, name))

        if _EXP_CUE.search(fold(text)):
            for m in _YEARS.finditer(text):
                raw = m.group(1).lower()
                value = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(fold(raw))
                if value and 0 < value <= 20:
                    req.years.append((value, text[:300], level))

        folded = fold(text)
        if _DEGREE_CUE.search(folded):
            levels = [lv for lv, rx in _DEGREE_PATTERNS if rx.search(folded)]
            if levels:
                ranks = [DEGREE_ORDER[lv] for lv in levels]
                alternatives = bool(re.search(r"\b(?:or|ou|oder|/)\b", folded))
                chosen = min(ranks) if alternatives else max(ranks)
                lvl_name = next(k for k, v in DEGREE_ORDER.items() if v == chosen)
                if level != "preferred" and (req.degree is None or DEGREE_ORDER[req.degree[0]] < chosen):
                    req.degree = (lvl_name, text[:300], level, bool(_ENROLLED.search(folded)))

        if _LANGUAGE_CUE.search(folded):
            words = re.findall(r"[a-z0-9]+", folded)
            for idx, word in enumerate(words):  # a proficiency cue must sit within 5 words of the language name
                language = _LANG_WORD.get(word)
                if language and _LANGUAGE_CUE.search(" ".join(words[max(0, idx - 5) : idx + 6])):
                    if not any(l.language == language for l in req.languages):
                        req.languages.append(LanguageReq(language, level, text[:300]))

    req.skills = sorted(best.values(), key=lambda s: (_RANK[s.level], s.name))
    return req
