"""Rule-based CV parser: text -> CandidateProfile.

Deliberately deterministic and conservative:
* sections are found from English / French / German headings;
* skills come from the shared taxonomy and keep the CV lines that mention them;
* dates are parsed only as far as the CV states them, and durations are *bounds*, never guesses;
* no proficiency, seniority or "years of experience" is inferred;
* contact details are never extracted.
Anything the rules cannot place is simply left out (the user can edit the profile in the interface).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from job_intel.match.models import (
    CandidateProfile,
    Education,
    Evidence,
    Experience,
    Project,
    SkillItem,
    SpokenLanguage,
)
from job_intel.match.taxonomy import category_of, find_skills

# ---------------------------------------------------------------------------------------------
# helpers


def fold(text: str) -> str:
    """Casefold + strip accents, for comparisons only."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


SECTION_HEADINGS: dict[str, set[str]] = {
    "summary": {"profile", "professional profile", "summary", "professional summary", "about me", "about", "objective",
                "career objective", "profil", "profil professionnel", "resume", "objectif", "objectif professionnel",
                "a propos", "zusammenfassung", "uber mich", "kurzprofil", "personal statement"},
    "education": {"education", "academic background", "studies", "academic education", "formation", "formations",
                  "etudes", "parcours academique", "diplomes", "scolarite", "ausbildung", "bildung", "studium",
                  "akademischer werdegang", "education and training"},
    "experience": {"experience", "work experience", "professional experience", "experiences", "employment",
                   "employment history", "career", "internships", "internship experience", "experience professionnelle",
                   "experiences professionnelles", "stages", "parcours professionnel", "berufserfahrung", "praktika",
                   "werdegang", "work history", "professional background"},
    "projects": {"projects", "personal projects", "academic projects", "selected projects", "projets",
                 "projets personnels", "projets academiques", "realisations", "projekte", "side projects"},
    "skills": {"skills", "technical skills", "key skills", "core competencies", "technologies", "tech stack", "tools",
               "competences", "competences techniques", "competences cles", "outils", "kenntnisse", "fahigkeiten",
               "technische fahigkeiten", "it kenntnisse", "technical competencies", "skills and tools"},
    "languages": {"languages", "spoken languages", "language skills", "langues", "sprachen", "sprachkenntnisse"},
    "certifications": {"certifications", "certificates", "licenses", "certificats", "zertifikate", "certification",
                       "formations complementaires", "courses"},
    "ignored": {"interests", "hobbies", "references", "volunteering", "centres d interet", "loisirs", "referencesque",
                "engagement", "interessen", "hobbys", "vie associative", "associations", "extracurricular", "awards",
                "publications", "distinctions", "activites extra professionnelles", "centres d'interet"},
}
_HEADING_SPLIT = re.compile(r"\s*(?:&|/|\band\b|\bet\b|\bund\b|\+)\s*")


def heading_of(line: str) -> str | None:
    stripped = re.sub(r"^[\W\d_]+|[\W\d_]+$", "", line.strip())
    if not stripped or len(stripped.split()) > 6 or len(stripped) > 50:
        return None
    norm = fold(stripped).replace("'", " ").replace("’", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    candidates = [norm, *_HEADING_SPLIT.split(norm)]
    for cand in candidates:
        for name, headings in SECTION_HEADINGS.items():
            if cand in headings:
                return name
    return None


MONTHS = {
    **dict.fromkeys(["jan", "janv", "january", "janvier", "januar"], 1),
    **dict.fromkeys(["feb", "fev", "fevr", "february", "fevrier", "februar"], 2),
    **dict.fromkeys(["mar", "mars", "march", "marz", "maerz"], 3),
    **dict.fromkeys(["apr", "avr", "april", "avril"], 4),
    **dict.fromkeys(["may", "mai"], 5),
    **dict.fromkeys(["jun", "juin", "june", "juni"], 6),
    **dict.fromkeys(["jul", "juil", "july", "juillet", "juli"], 7),
    **dict.fromkeys(["aug", "aout", "august"], 8),
    **dict.fromkeys(["sep", "sept", "september", "septembre"], 9),
    **dict.fromkeys(["oct", "october", "octobre", "oktober"], 10),
    **dict.fromkeys(["nov", "november", "novembre"], 11),
    **dict.fromkeys(["dec", "december", "decembre", "dezember"], 12),
}
_MONTH_RE = "|".join(sorted([*MONTHS, "février", "fév", "août", "décembre", "déc", "märz"], key=len, reverse=True))
_DATE_TOKEN = rf"(?:(?:{_MONTH_RE})\.?\s+)?(?:19|20)\d{{2}}|\d{{1,2}}/(?:19|20)\d{{2}}"
_PRESENT = r"present|présent|présente|current|currently|now|today|aujourd'hui|aujourd’hui|ce jour|en cours|ongoing|heute|aktuell|jusqu'a ce jour"
DATE_RANGE = re.compile(
    rf"(?P<s>{_DATE_TOKEN})\s*(?:-|–|—|to|a|au|bis|until|jusqu'a|jusqu’à|à)\s*(?P<e>{_PRESENT}|{_DATE_TOKEN})",
    re.IGNORECASE,
)
SINGLE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def parse_date_token(token: str) -> tuple[int, int | None] | None:
    token = fold(token).strip(". ")
    if m := re.fullmatch(r"(\d{1,2})/((?:19|20)\d{2})", token):
        month = int(m.group(1))
        return int(m.group(2)), month if 1 <= month <= 12 else None
    if m := re.fullmatch(r"(?:([a-z]+)\.?\s+)?((?:19|20)\d{2})", token):
        month = MONTHS.get(m.group(1)) if m.group(1) else None
        return int(m.group(2)), month
    return None


def _fmt(ym: tuple[int, int | None]) -> str:
    return f"{ym[0]}-{ym[1]:02d}" if ym[1] else str(ym[0])


def date_range(text: str, today: date) -> dict | None:
    """First date range in `text`: {start, end, current, min, max (months)}; None if there is none."""
    m = DATE_RANGE.search(text)
    if not m:
        return None
    start = parse_date_token(m.group("s"))
    end_raw = fold(m.group("e"))
    current = bool(re.fullmatch(fold(_PRESENT), end_raw, re.IGNORECASE))
    end = (today.year, today.month) if current else parse_date_token(m.group("e"))
    if start is None or end is None:
        return None
    if start[1] and end[1]:  # month resolution on both sides
        elapsed = (end[0] - start[0]) * 12 + (end[1] - start[1])
        lo, hi = max(0, elapsed - 1), max(0, elapsed + 1)
    else:  # year resolution: only widely spaced bounds are honest
        lo, hi = max(0, (end[0] - start[0] - 1) * 12), max(0, (end[0] - start[0] + 1) * 12 - 1)
    if end < start:
        return None
    return {"start": _fmt(start), "end": "present" if current else _fmt(end), "current": current, "min": lo, "max": hi,
            "span": m.span(), "end_year": end[0]}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# education

_LEVELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("phd", re.compile(r"\b(?:ph\.?\s?d|doctorate|doctorat|doctoral|dr\.?-ing)\b", re.I)),
    ("master", re.compile(r"\b(?:master|msc|m\.sc|m1|m2|mba|mastere|ingenieur|engineering degree|engineer'?s degree|"
                          r"dipl\.?-?ing|diplom-ingenieur|bac ?\+ ?5)\b", re.I)),
    ("bachelor", re.compile(r"\b(?:bachelor|bsc|b\.sc|licence|bac ?\+ ?3|undergraduate|bachelorstudium)\b", re.I)),
    ("associate", re.compile(r"\b(?:bts|dut|bac ?\+ ?2|associate degree)\b", re.I)),
    ("secondary", re.compile(r"\b(?:baccalaureat|baccalaureate|high school|abitur|lycee)\b", re.I)),
)
_INSTITUTION = re.compile(r"\b(?:university|universite|universitat|ecole|school|institut|institute|college|polytechni\w*|"
                          r"hochschule|academy|lycee|faculty|faculte|ensae|ensimag|centrale|emines|insa|mines)\b", re.I)
_IN_PROGRESS = re.compile(r"\b(?:expected|prevu|en cours|currently|current|candidate|in progress|etudiant en|student)\b", re.I)


def _level_of(text: str) -> str | None:
    folded = fold(text)
    for level, pattern in _LEVELS:
        if pattern.search(folded):
            return level
    return None


def _blocks(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Split (index, line) pairs into blank-line separated blocks."""
    blocks, current = [], []
    for item in lines:
        if item[1].strip():
            current.append(item)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_education(lines: list[tuple[int, str]], today: date) -> list[Education]:
    entries: list[list[str]] = []
    for block in _blocks(lines):
        current: list[str] = []
        for _, line in block:
            starts_new = bool(_level_of(line)) and any(_level_of(prev) for prev in current)
            if starts_new:
                entries.append(current)
                current = []
            current.append(line)
        if current:
            entries.append(current)
    out = []
    for entry in entries:
        joined = " | ".join(entry)
        level = max((lv for lv in map(_level_of, entry) if lv), key=lambda lv: ["secondary", "associate", "bachelor", "master", "phd"].index(lv), default=None)
        degree_line = next((ln for ln in entry if _level_of(ln) == level and level), entry[0])
        dr = date_range(joined, today)
        years = [int(y) for y in SINGLE_YEAR.findall(joined)]
        end_year = dr["end_year"] if dr else (max(years) if years else None)
        in_progress: bool | None = None  # None = the CV does not say, and the dates do not settle it
        if (dr and dr["current"]) or _IN_PROGRESS.search(fold(joined)):
            in_progress = True
        elif end_year is not None and end_year != today.year:
            in_progress = end_year > today.year
        degree_clean = re.sub(DATE_RANGE, "", degree_line).strip(" ,-|")
        degree_part, *rest = re.split(r"\s[-–—|]\s", degree_clean, maxsplit=1)
        field_match = re.search(r"(?:\bin\b|\ben\b|\bof\b|\bde\b|:|\bfor\b)\s+([A-Z][\w &/'’-]{3,60})", degree_part)
        institution = next(
            (re.sub(DATE_RANGE, "", ln).strip(" ,-|") for ln in entry if ln != degree_line and _INSTITUTION.search(fold(ln))),
            rest[0].strip() if rest else None,
        )
        out.append(Education(
            degree=degree_part.strip() or None, level=level,  # type: ignore[arg-type]
            field=field_match.group(1).strip() if field_match and level else None,
            institution=institution or None,
            start=dr["start"] if dr else None, end=dr["end"] if dr else (str(end_year) if end_year else None),
            in_progress=in_progress, evidence=[Evidence(text=ln[:600], section="education") for ln in entry[:6]],
        ))  # fmt: skip
    return [e for e in out if e.degree or e.institution]


# ---------------------------------------------------------------------------------------------
# experience and projects

_ROLE_WORDS = re.compile(
    r"\b(?:engineer|ingenieur|intern|internship|stagiaire|stage|analyst|analyste|developer|developpeur|scientist|"
    r"assistant|assistante|consultant|manager|researcher|chercheur|apprenti|alternant|alternance|werkstudent|praktikant|"
    r"technician|technicien|lead|director|designer|architect|data|ml|ai|ia|freelance|teaching|tutor|moniteur|"
    r"research|developer|programmer|statistician|statisticien)\b", re.I)  # fmt: skip
_INTERNSHIP = re.compile(r"\b(?:intern|internship|stagiaire|stage|praktik\w*|werkstudent|alternan\w*|apprenti\w*|apprentice|co-?op)\b", re.I)
_BULLET = re.compile(r"^\s*[-•*·]\s+")
_HEADER_SPLIT = re.compile(r"\s+[|•·@–—]\s+|\s+-\s+|\s+(?:at|chez|bei)\s+|\s*\|\s*")


def _split_entry(block: list[tuple[int, str]]) -> tuple[list[str], list[str]]:
    """(header lines, bullet texts) of one entry; wrapped lines are glued to the previous bullet."""
    header: list[str] = []
    bullets: list[str] = []
    for _, line in block:
        if _BULLET.match(line):
            bullets.append(_BULLET.sub("", line).strip())
        elif bullets and not DATE_RANGE.search(line) and (line[:1].islower() or not re.search(r"[.!?:;]$", bullets[-1])):
            bullets[-1] = f"{bullets[-1]} {line.strip()}"
        elif not bullets:
            header.append(line.strip())
        else:
            bullets.append(line.strip())
    return header, bullets


def _entry_blocks(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Blank-line blocks, further split where a new dated header starts after bullets."""
    out: list[list[tuple[int, str]]] = []
    for block in _blocks(lines):
        current: list[tuple[int, str]] = []
        seen_bullet = False
        for item in block:
            is_bullet = bool(_BULLET.match(item[1]))
            if current and seen_bullet and not is_bullet and DATE_RANGE.search(item[1]):
                out.append(current)
                current, seen_bullet = [], False
            current.append(item)
            seen_bullet = seen_bullet or is_bullet
        out.append(current)
    return out


def parse_experience(lines: list[tuple[int, str]], today: date) -> list[Experience]:
    out = []
    for block in _entry_blocks(lines):
        header, bullets = _split_entry(block)
        if not header and not bullets:
            continue
        header_text = " | ".join(header)
        dr = date_range(header_text, today)
        cleaned = DATE_RANGE.sub("", header_text) if dr else header_text
        pieces = [p.strip(" ,-|•·") for p in _HEADER_SPLIT.split(cleaned) if p and p.strip(" ,-|•·")]
        title = next((p for p in pieces if _ROLE_WORDS.search(fold(p))), None)
        organization = next((p for p in pieces if p != title and not SINGLE_YEAR.fullmatch(p)), None)
        all_text = "\n".join([header_text, *bullets])
        kind = "internship" if _INTERNSHIP.search(fold(header_text)) else ("job" if (title or organization) else "unspecified")
        out.append(Experience(
            title=title, organization=organization, kind=kind,  # type: ignore[arg-type]
            start=dr["start"] if dr else None, end=dr["end"] if dr else None, current=bool(dr and dr["current"]),
            duration_months_min=dr["min"] if dr else None, duration_months_max=dr["max"] if dr else None,
            bullets=[b[:400] for b in bullets[:20]], skills=find_skills(all_text),
            evidence=[Evidence(text=t[:600], section="experience") for t in [*header, *bullets][:25] if t],
        ))  # fmt: skip
    return out


def parse_projects(lines: list[tuple[int, str]]) -> list[Project]:
    out = []
    for block in _blocks(lines):
        header, bullets = _split_entry(block)
        name = re.sub(r"https?://\S+", "", header[0]).strip(" -|:•") if header else None
        description = [*(header[1:]), *bullets]
        text = "\n".join([*header, *bullets])
        if not name and not description:
            continue
        out.append(Project(
            name=(name or None) and name[:120], description=[d[:400] for d in description[:12]], skills=find_skills(text),
            evidence=[Evidence(text=t[:600], section="projects") for t in [*header, *bullets][:15] if t],
        ))  # fmt: skip
    return out


def parse_unstructured(lines: list[tuple[int, str]], today: date) -> tuple[list[Experience], list[Education]]:
    """Fallback for CVs without recognisable Experience/Education headings (prose, unusual headings).

    A line with a date range and no degree keyword is treated as an experience whose title/organisation come from
    the text BEFORE the dates; a line naming a degree level becomes an education entry. Less reliable than the
    sectioned path, so the caller adds a warning.
    """
    experiences, edu_lines = [], []
    previous = ""  # last non-empty, non-bullet, undated line: the header of an entry whose dates sit on the next line
    for index, line in lines:
        if not line.strip():
            continue
        if _level_of(line):
            edu_lines.append((index, line))
            continue
        dr = date_range(line, today)
        if not dr:
            if not _BULLET.match(line):
                previous = line.strip()
            continue
        head = line[: dr["span"][0]].strip(" ,;:-|(")
        tail = line[dr["span"][1] :].strip(" .,;:-|)")
        if not head and previous:
            head = previous
        pieces = [p.strip(" ,-|") for p in re.split(r"\s+[-\u2013\u2014|]\s+|,\s+", head) if p.strip(" ,-|")]
        title = next((p for p in pieces if _ROLE_WORDS.search(fold(p))), None)
        if not (title or pieces):
            continue
        organization = next((p for p in pieces if p != title), None)
        text = f"{head} {tail}"
        experiences.append(Experience(
            title=title, organization=organization,
            kind="internship" if _INTERNSHIP.search(fold(head)) else "job",
            start=dr["start"], end=dr["end"], current=dr["current"], duration_months_min=dr["min"], duration_months_max=dr["max"],
            bullets=[t for t in re.split(r"(?<=[.!?])\s+", tail) if t][:6], skills=find_skills(text),
            evidence=[Evidence(text=line[:600], section="experience")],
        ))  # fmt: skip
    return experiences, parse_education(edu_lines, today)


# ---------------------------------------------------------------------------------------------
# languages, roles, language detection

LANGUAGE_NAMES = {
    "english": "English", "anglais": "English", "englisch": "English", "french": "French", "francais": "French",
    "français": "French", "französisch": "French", "franzosisch": "French", "german": "German", "allemand": "German",
    "deutsch": "German", "dutch": "Dutch", "neerlandais": "Dutch", "nederlands": "Dutch", "niederlandisch": "Dutch",
    "spanish": "Spanish", "espagnol": "Spanish", "espanol": "Spanish", "italian": "Italian", "italien": "Italian",
    "portuguese": "Portuguese", "portugais": "Portuguese", "arabic": "Arabic", "arabe": "Arabic",
    "chinese": "Chinese", "chinois": "Chinese", "mandarin": "Chinese",
}  # fmt: skip


_INLINE_LANGUAGES = re.compile(r"^\s*(?:spoken\s+)?(?:languages?|langues|sprachen|sprachkenntnisse)\s*[:\-\u2013]\s*(.+)$", re.IGNORECASE)


def inline_language_lines(lines: list[str]) -> list[tuple[int, str]]:
    found = [(i, m.group(1)) for i, ln in enumerate(lines) if (m := _INLINE_LANGUAGES.match(ln))]
    return found or loose_language_lines(lines)


def loose_language_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Lines made only of 'Language (level)' items, e.g. 'English (fluent), French (native)', under any heading."""
    out = []
    for i, ln in enumerate(lines):
        pieces = [p.strip() for p in re.split(r"[,;]", ln) if p.strip()]
        if 1 <= len(pieces) <= 6 and all(re.match(r"([A-Za-zÀ-ÿ]+)", p) and LANGUAGE_NAMES.get(fold(re.match(r"([A-Za-zÀ-ÿ]+)", p).group(1))) for p in pieces) \
                and any(re.search(r"[(:\-]|\b(?:native|fluent|basic|natif|courant|notions|[abc][12])\b", p, re.I) for p in pieces):
            out.append((i, ln))
    return out


def parse_spoken_languages(lines: list[tuple[int, str]]) -> list[SpokenLanguage]:
    out: dict[str, SpokenLanguage] = {}
    for _, line in lines:
        for piece in re.split(r"[,;|•·\n]|\s{3,}", line):
            piece = piece.strip(" -–—")
            if not piece:
                continue
            first = re.match(r"([A-Za-zÀ-ÿ]+)", piece)
            language = LANGUAGE_NAMES.get(fold(first.group(1))) if first else None
            if language and language not in out:
                rest = piece[first.end():].strip(" :-–—()[]")  # type: ignore[union-attr]
                out[language] = SpokenLanguage(language=language, level_text=rest[:60] or None)
    return list(out.values())


ROLE_LEXICON: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.I))
    for name, pattern in (
        ("Data Scientist", r"data[\s-]*scientist|scientifique des donn[ée]es"),
        ("Data Analyst", r"data[\s-]*analyst|analyste (?:de )?donn[ée]es|analyste data|datenanalyst"),
        ("Machine Learning Engineer", r"(?:machine learning|ml) engineer|ing[ée]nieur (?:en )?(?:machine learning|ml|apprentissage)"),
        ("AI Engineer", r"\bai engineer|\bia engineer|ing[ée]nieur (?:en )?(?:ia|intelligence artificielle)|\bki-ingenieur"),
        ("Data Engineer", r"data[\s-]*engineer|ing[ée]nieur (?:data|donn[ée]es)|dateningenieur"),
        ("NLP Engineer", r"nlp engineer|ing[ée]nieur nlp"),
        ("Analytics Engineer", r"analytics engineer"),
        ("Business Intelligence Analyst", r"business intelligence|\bbi analyst|analyste bi"),
        ("Research Scientist", r"research scientist|chercheur en (?:ia|machine learning)"),
        ("MLOps Engineer", r"mlops engineer"),
        ("Software Engineer", r"software engineer|ing[ée]nieur logiciel|d[ée]veloppeur"),
    )
)
_OBJECTIVE = re.compile(r"\b(?:seeking|looking for|searching for|aiming|aspiring|objective|goal|recherche|a la recherche|"
                        r"à la recherche|souhaite|objectif|en quete|en quête|suche|interested in|cherche)\b", re.I)


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ']+", text.lower())[:2500]
    stop = {
        "en": {"the", "and", "of", "to", "in", "for", "with", "my", "on", "a", "as", "using"},
        "fr": {"le", "la", "les", "des", "et", "de", "du", "un", "une", "pour", "dans", "avec", "sur", "au", "par"},
        "de": {"der", "die", "das", "und", "für", "mit", "von", "zu", "im", "ein", "eine", "auf"},
    }
    scores = {lang: sum(w in vocab for w in words) for lang, vocab in stop.items()}
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] >= 5 else "unknown"


# ---------------------------------------------------------------------------------------------
# main entry point


def parse_cv(text: str, *, today: date | None = None, fallbacks: bool = True) -> CandidateProfile:
    """`fallbacks=False` reproduces the Phase 3 parser (no guessing without headings, no inline language lines)."""
    today = today or date.today()
    lines = text.split("\n")
    sections: dict[str, list[tuple[int, str]]] = {k: [] for k in (*SECTION_HEADINGS, "header")}
    current = "header"
    section_of: list[str] = []
    for i, line in enumerate(lines):
        if (h := heading_of(line)) and not _BULLET.match(line):
            current = h
            section_of.append("heading")
            continue
        section_of.append(current)
        sections[current].append((i, line))

    # --- skills: every line outside languages / ignored sections, remembering where it was found
    mentions: dict[str, list[tuple[str, str]]] = {}
    for i, line in enumerate(lines):
        sec = section_of[i]
        if sec in {"heading", "languages", "ignored"} or not line.strip() or "@" in line:
            continue
        if _OBJECTIVE.search(line):  # "looking for ... LLM roles", "interested in NLP": a wish, not a skill claim
            continue
        for skill in find_skills(line):
            mentions.setdefault(skill, []).append((sec, line.strip()))
    skills = []
    for name, found in mentions.items():
        demonstrated = [m for m in found if m[0] in {"experience", "projects"}]
        ordered = demonstrated + [m for m in found if m not in demonstrated]
        seen, evidence = set(), []
        for sec, line in ordered:
            if line not in seen and len(evidence) < 4:
                seen.add(line)
                evidence.append(Evidence(text=line[:600], section=sec))
        skills.append(SkillItem(name=name, category=category_of(name), context="demonstrated" if demonstrated else "listed", evidence=evidence))
    skills.sort(key=lambda s: (s.context != "demonstrated", -len(mentions[s.name]), s.name))

    experiences = parse_experience(sections["experience"], today)
    education = parse_education(sections["education"], today)
    if fallbacks and not sections["experience"] and not sections["education"]:
        experiences, education = parse_unstructured(
            [(i, ln) for sec in ("header", "summary", "skills", "projects", "certifications") for i, ln in sections[sec]], today)
    projects = parse_projects(sections["projects"])
    spoken = parse_spoken_languages(sections["languages"] or (inline_language_lines(lines) if fallbacks else []))

    summary_lines = [ln.strip() for _, ln in sections["summary"] if ln.strip()]
    summary = " ".join(summary_lines)[:700] or None

    head = [ln.strip() for _, ln in sections["header"] if ln.strip()][:6]
    def _looks_like_headline(ln: str) -> bool:
        if len(ln) > 90 or "@" in ln or "http" in ln.lower() or re.search(r"\d{6,}", ln):
            return False
        return bool(_ROLE_WORDS.search(fold(ln)) or re.search(r"\b(?:student|etudiant)\b", fold(ln)))

    headline = next((ln for ln in head if _looks_like_headline(ln)), None)
    objective_zone = [*summary_lines, *(head or [])]
    roles: list[str] = []
    for ln in objective_zone:
        if _OBJECTIVE.search(ln):
            roles += [name for name, rx in ROLE_LEXICON if rx.search(ln)]

    certifications = [ln.strip().removeprefix("- ").strip()[:200] for _, ln in sections["certifications"]
                      if len(ln.strip()) >= 4][:15]  # fmt: skip
    from job_intel.enrich.sector import classify_sector
    from job_intel.match.softskills import find_soft_skills

    sectors: list[str] = []
    for exp in experiences:  # the sector of each experience, from its own lines only
        sector, _hits = classify_sector(exp.organization, exp.title, " ".join(exp.bullets))
        if sector and sector not in sectors:
            sectors.append(sector)
    soft = find_soft_skills("\n".join(ln for i, ln in enumerate(lines) if section_of[i] not in {"heading", "ignored"}))

    profile = CandidateProfile(
        detected_language=detect_language(text),  # type: ignore[arg-type]
        headline=headline, summary=summary, target_roles=roles, education=education, skills=skills,
        experiences=experiences, projects=projects, spoken_languages=spoken, certifications=certifications,
        experience_sectors=sectors, soft_skills=soft,
    )  # fmt: skip
    warn = profile.warnings
    if not any(sections[k] for k in ("experience", "education", "skills", "projects")):
        warn.append("No standard section headings (Education, Experience, Skills, Projects...) were recognised: "
                    "only skills mentioned anywhere in the text could be extracted.")
    if fallbacks and not sections["experience"] and not sections["education"] and (experiences or education):
        warn.append("No Experience/Education headings were recognised: entries were guessed from dates and degree names. Please review them.")
    if not skills:
        warn.append("No technical skills from the supported vocabulary were found.")
    if not experiences and not projects:
        warn.append("No experience or project entries were recognised.")
    if experiences and not any(e.start for e in experiences):
        warn.append("No dates were recognised in the experience entries, so experience duration is unknown.")
    return profile
