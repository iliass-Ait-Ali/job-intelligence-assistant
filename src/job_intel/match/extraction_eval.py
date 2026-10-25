"""Measure the CV parser against SYNTHETIC CVs whose ground truth is known by construction.

Each document is generated from a structured specification (skills, degree, dated experiences, spoken languages,
target role) in one of several layouts and languages. The gold annotation is the specification itself, so no human
judgment is involved and nothing is "invented": we know exactly which skills, dates and degree were written.

What this measures: how well the parser recovers information from different layouts (headings, inline skills, no
headings, two-column text order, numeric dates, French/German). What it does NOT measure: behaviour on real CVs,
whose layouts and vocabulary are more varied. Use `jobintel profile --cv your.pdf` to inspect a real one.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from job_intel.match.cv_parser import parse_cv
from job_intel.match.taxonomy import BY_NAME

TODAY = date(2026, 10, 23)
MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]

PROFILES = [
    {"role": "Data Scientist", "title": "Data Scientist", "skills": ["Python", "SQL", "pandas", "scikit-learn", "PyTorch", "Statistics", "Docker"],
     "extra": ["XGBoost", "Git", "Tableau", "Time series"], "degree": ("master", "Master in Data Science"), "spoken": [("English", "fluent"), ("French", "native")]},
    {"role": "Data Engineer", "title": "Data Engineer", "skills": ["Python", "SQL", "Spark", "Airflow", "dbt", "Snowflake", "Docker"],
     "extra": ["Kafka", "Terraform", "AWS", "PostgreSQL"], "degree": ("master", "Diplôme d'ingénieur"), "spoken": [("French", "native"), ("English", "B2")]},
    {"role": "Machine Learning Engineer", "title": "Machine Learning Engineer", "skills": ["Python", "PyTorch", "TensorFlow", "MLflow", "Kubernetes", "AWS", "FastAPI"],
     "extra": ["LangChain", "Git", "Hugging Face", "Deep learning"], "degree": ("phd", "PhD in Computer Science"), "spoken": [("English", "native"), ("German", "B1")]},
    {"role": "Data Analyst", "title": "Data Analyst", "skills": ["SQL", "Excel", "Power BI", "Python", "Tableau", "Data visualization", "Statistics"],
     "extra": ["R", "Looker", "A/B testing"], "degree": ("bachelor", "Bachelor in Applied Mathematics"), "spoken": [("English", "C1"), ("Spanish", "basic")]},
]
ORGS = ["Acme Analytics", "Northwind Data", "Globex Labs", "Initech Systems", "Contoso Retail"]
STYLES = ["en_classic", "fr_classic", "de_classic", "inline_skills", "no_headings", "two_column", "numeric_dates_caps", "pipe_table",
          "unusual_headings", "compact_dates"]


@dataclass
class GoldCV:
    id: str
    style: str
    text: str
    skills: set[str]
    degree_level: str
    experiences: list[tuple[str, str, str]]  # (title, start, end) as the parser should normalise them
    languages: set[str]
    target_roles: set[str]
    notes: list[str] = field(default_factory=list)


def _alias(rng: random.Random, name: str) -> str:
    """Sometimes write a skill through one of its taxonomy aliases (the gold stays the canonical name)."""
    skill = BY_NAME[name]
    safe = [a for a in skill.aliases if a.lower() not in {"js", "ml", "eda", "tal", "ab testing", "k8s"} and len(a) > 3]
    return rng.choice(safe) if safe and rng.random() < 0.25 and not skill.ambiguous else name


def _fmt_date(style: str, y: int, m: int) -> str:
    if style == "fr_classic":
        return f"{MONTHS_FR[m - 1]} {y}"
    if style == "de_classic":
        return f"{MONTHS_DE[m - 1]} {y}"
    if style in {"numeric_dates_caps", "compact_dates"}:
        return f"{m:02d}/{y}"
    return f"{MONTHS_EN[m - 1]} {y}"


def _render(style: str, spec: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    lang = {"fr_classic": "fr", "de_classic": "de"}.get(style, "en")
    h = {
        "en": dict(sum="PROFILE", edu="EDUCATION", exp="EXPERIENCE", skl="SKILLS", lng="LANGUAGES", present="present", seek="Looking for a {role} position."),
        "fr": dict(sum="PROFIL", edu="FORMATION", exp="EXPÉRIENCES PROFESSIONNELLES", skl="COMPÉTENCES", lng="LANGUES", present="Présent", seek="À la recherche d'un poste de {role}."),
        "de": dict(sum="PROFIL", edu="AUSBILDUNG", exp="BERUFSERFAHRUNG", skl="KENNTNISSE", lng="SPRACHEN", present="heute", seek="Looking for a {role} position."),
    }[lang]
    if style == "numeric_dates_caps":
        h = {**h, "exp": "EXPERIENCE:", "edu": "EDUCATION:", "skl": "SKILLS:", "lng": "LANGUAGES:", "sum": "PROFILE:"}
    if style == "unusual_headings":  # headings the parser's vocabulary does not know
        h = {**h, "exp": "WHERE I WORKED", "edu": "Where I studied", "skl": "My toolbox", "lng": "Tongues", "sum": "Who I am"}
    roles = [e["title"] for e in spec["experiences"]]
    skills_text = {s: _alias(rng, s) for s in spec["skills"]}
    exps = []
    for e in spec["experiences"]:
        end = h["present"] if e["current"] else _fmt_date(style, *e["end"])
        sep = {"fr_classic": " – ", "compact_dates": "–"}.get(style, " - ")
        dates = f"{_fmt_date(style, *e['start'])}{sep}{end}"
        mine = [skills_text[s] for s in e["skills"]]
        if style == "inline_skills":
            bullets = [f"- Delivered projects using {', '.join(mine[:3])} and {mine[3]}." if len(mine) > 3 else f"- Delivered projects using {', '.join(mine)}."]
        else:
            glyph = "•" if style == "compact_dates" else "-"
            bullets = [f"{glyph} Built pipelines and models with {', '.join(mine[:2])}.", f"{glyph} Worked on dashboards and reporting with {mine[-1]}."]
        exps.append((f"{e['title']} - {e['org']}", dates, bullets))
    skills_line = ", ".join(skills_text[s] for s in spec["skills"])
    deg_level, degree = spec["degree"]
    edu = f"{degree} - Example University\n{spec['edu_dates']}"
    spoken = ", ".join(f"{lang_} ({lvl})" for lang_, lvl in spec["spoken"])
    seek = h["seek"].format(role=spec["role"]) if spec["state_role"] else None

    if style == "no_headings":
        body = [f"{seek or ''} I hold a {degree} ({spec['edu_dates']})."]
        for title, dates, bullets in exps:
            body.append(f"{title}, {dates}. " + " ".join(b.lstrip('- ') for b in bullets))
        body.append(f"Technical skills: {skills_line}.")
        body.append(f"Languages: {spoken}")
        return "\n\n".join(body), {"headings": False}
    parts = []
    if seek:
        parts.append(f"{h['sum']}\n{seek}")
    parts.append(f"{h['edu']}\n{edu}")
    if style == "two_column":
        left = [skills_text[x] for x in spec["skills"]]  # first column: skills; second column: unrelated text, as in a 2-column PDF
        right = [t for t, _, _ in exps] + [d for _, d, _ in exps]
        pad = max(len(a) for a in left) + 3  # real extractors keep a gap between columns; gluing words is not what we test here
        lines = [f"{a:<{pad}}{right[i % len(right)]}" for i, a in enumerate(left)]
        parts.append(f"{h['exp']}\n" + "\n".join(f"{t}\n{d}\n" + "\n".join(b) for t, d, b in exps))
        parts.append(f"{h['skl']}\n" + "\n".join(lines))
    elif style == "pipe_table":
        parts.append(f"{h['exp']}\n" + "\n\n".join(f"{t}\n{d}\n" + "\n".join(b) for t, d, b in exps))
        half = len(spec["skills"]) // 2
        s = [skills_text[x] for x in spec["skills"]]
        parts.append(f"{h['skl']}\nCore | " + " | ".join(s[:half]) + "\nTools | " + " | ".join(s[half:]))
    else:
        parts.append(f"{h['exp']}\n" + "\n\n".join(f"{t}\n{d}\n" + "\n".join(b) for t, d, b in exps))
        if style != "inline_skills":
            parts.append(f"{h['skl']}\n{skills_line}")
    parts.append(f"{h['lng']}\n{spoken}")
    return "\n\n".join(parts), {"headings": True, "roles": roles}


def generate_corpus(seed: int = 0, per_style: int = 4) -> list[GoldCV]:
    rng = random.Random(seed)
    corpus: list[GoldCV] = []
    for style in STYLES:
        for k in range(per_style):
            prof = PROFILES[(k + STYLES.index(style)) % len(PROFILES)]
            skills = prof["skills"] + rng.sample(prof["extra"], 2)
            exps = []
            year = 2018 + rng.randint(0, 2)
            for j in range(2):
                start = (year + j * 2, rng.randint(1, 12))
                current = j == 1
                end = (start[0] + 1, rng.randint(1, 12)) if not current else (TODAY.year, TODAY.month)
                chosen = skills[j * 3 : j * 3 + 4] if len(skills[j * 3 : j * 3 + 4]) == 4 else skills[:4]
                exps.append({"title": prof["title"] if j else f"Junior {prof['title']}", "org": rng.choice(ORGS), "start": start, "end": end, "current": current, "skills": chosen})
            used_skills = {s for e in exps for s in e["skills"]}
            title_skills = {"Machine learning"} if "Machine Learning" in prof["title"] else set()  # a mention in a job title counts
            if style == "inline_skills":
                spec_skills = [s for s in skills if s in used_skills]
            else:
                spec_skills = skills
            spec = {"role": prof["role"], "skills": spec_skills, "experiences": exps, "degree": prof["degree"], "spoken": prof["spoken"],
                    "edu_dates": f"{2012 + k} - {2015 + k}", "state_role": style not in {"two_column", "pipe_table"}}  # fmt: skip
            text, info = _render(style, spec, rng)
            if style == "inline_skills":
                spec_skills = [s for s in spec_skills if s in used_skills]
            gold_exps = []
            for e in exps:
                s_ = f"{e['start'][0]}-{e['start'][1]:02d}"
                e_ = "present" if e["current"] else f"{e['end'][0]}-{e['end'][1]:02d}"
                gold_exps.append((e["title"], s_, e_))
            corpus.append(GoldCV(
                id=f"{style}_{k}", style=style, text=text, skills=set(spec_skills) | title_skills, degree_level=prof["degree"][0],
                experiences=gold_exps, languages={lang for lang, _ in prof["spoken"]},
                target_roles={prof["role"]} if spec["state_role"] else set(),
            ))  # fmt: skip
    return corpus


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gold: set, got: set) -> None:
        self.tp += len(gold & got)
        self.fp += len(got - gold)
        self.fn += len(gold - got)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class ExtractionReport:
    per_style: dict[str, dict[str, float]]
    overall: dict[str, float]
    failures: list[str]


def evaluate_extraction(corpus: list[GoldCV] | None = None, *, fallbacks: bool = True) -> ExtractionReport:
    corpus = corpus or generate_corpus()
    acc: dict[str, dict[str, Any]] = defaultdict(lambda: {"skills": Counts(), "langs": Counts(), "roles": Counts(), "deg": [0, 0], "dates": [0, 0], "exp": Counts()})
    failures: list[str] = []
    for doc in corpus:
        profile = parse_cv(doc.text, today=TODAY, fallbacks=fallbacks)
        a = acc[doc.style]
        got_skills = profile.skill_names
        a["skills"].add(doc.skills, got_skills)
        a["langs"].add(doc.languages, {s.language for s in profile.spoken_languages})
        a["roles"].add(doc.target_roles, set(profile.target_roles))
        got_levels = {e.level for e in profile.education}
        a["deg"][0] += doc.degree_level in got_levels
        a["deg"][1] += 1
        got_exps = {(e.start, e.end) for e in profile.experiences}
        gold_dates = {(s, e) for _, s, e in doc.experiences}
        a["dates"][0] += len(gold_dates & got_exps)
        a["dates"][1] += len(gold_dates)
        a["exp"].add({(t, s) for t, s, _ in doc.experiences}, {(e.title, e.start) for e in profile.experiences})
        if doc.skills != got_skills:
            failures.append(f"{doc.id}: skills missed {sorted(doc.skills - got_skills)} extra {sorted(got_skills - doc.skills)}")
        if doc.degree_level not in got_levels:
            failures.append(f"{doc.id}: degree level {doc.degree_level} not found (got {sorted(x for x in got_levels if x)})")
        if gold_dates - got_exps:
            failures.append(f"{doc.id}: experience dates missed {sorted(gold_dates - got_exps)} (got {sorted(x for x in got_exps if x[0])})")
    per_style, tot = {}, {"skills": Counts(), "langs": Counts(), "roles": Counts(), "exp": Counts(), "deg": [0, 0], "dates": [0, 0]}
    for style, a in acc.items():
        per_style[style] = {
            "skill_precision": a["skills"].precision, "skill_recall": a["skills"].recall, "skill_f1": a["skills"].f1,
            "degree_level_accuracy": a["deg"][0] / a["deg"][1], "experience_date_recall": a["dates"][0] / a["dates"][1],
            "language_f1": a["langs"].f1, "target_role_f1": a["roles"].f1,
        }  # fmt: skip
        for key in ("skills", "langs", "roles", "exp"):
            tot[key].tp += a[key].tp
            tot[key].fp += a[key].fp
            tot[key].fn += a[key].fn
        for key in ("deg", "dates"):
            tot[key][0] += a[key][0]
            tot[key][1] += a[key][1]
    overall = {
        "documents": float(len(corpus)), "skill_precision": tot["skills"].precision, "skill_recall": tot["skills"].recall, "skill_f1": tot["skills"].f1,
        "degree_level_accuracy": tot["deg"][0] / tot["deg"][1], "experience_date_recall": tot["dates"][0] / tot["dates"][1],
        "language_f1": tot["langs"].f1, "target_role_f1": tot["roles"].f1, "experience_title_start_f1": tot["exp"].f1,
    }  # fmt: skip
    return ExtractionReport(per_style, overall, failures)


def format_extraction(report: ExtractionReport, show_failures: int = 12) -> str:
    cols = ["skill_precision", "skill_recall", "skill_f1", "degree_level_accuracy", "experience_date_recall", "language_f1", "target_role_f1"]
    short = ["skillP", "skillR", "skillF1", "degree", "dates", "langF1", "roleF1"]
    lines = [f"Synthetic CVs with known ground truth: {int(report.overall['documents'])} documents, {len(report.per_style)} layouts",
             f"{'layout':<20}" + "".join(f"{s:>9}" for s in short)]  # fmt: skip
    for style, m in report.per_style.items():
        lines.append(f"{style:<20}" + "".join(f"{m[c]:>9.2f}" for c in cols))
    lines.append(f"{'ALL':<20}" + "".join(f"{report.overall[c]:>9.2f}" for c in cols))
    if report.failures:
        lines += ["", f"{len(report.failures)} imperfect document(s); first {min(show_failures, len(report.failures))}:"] + [f"  {f}" for f in report.failures[:show_failures]]
    lines.append("Ground truth is known by construction (synthetic CVs); this says nothing about real CVs' layouts.")
    return "\n".join(lines)
