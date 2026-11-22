"""Soft-skill vocabulary (EN / FR / DE), used on both sides: job listings and CVs.

Soft skills are extracted with the same discipline as technical skills: a canonical name, explicit
aliases, and the line that mentions it as evidence. Nothing is inferred from tone ("dynamic team"
is not evidence of teamwork in the *candidate*). On the job side they are reported as asked-for
qualities; on the CV side as claims the CV makes. They never contribute to the technical-skill
coverage score (see ``match/scoring.py``), because a CV saying "team player" proves little.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from job_intel.match.cv_parser import fold


@dataclass(frozen=True)
class SoftSkill:
    name: str
    aliases: tuple[str, ...]
    pattern: re.Pattern[str] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        alts = "|".join(re.escape(fold(a)) for a in sorted(self.aliases, key=len, reverse=True))
        object.__setattr__(self, "pattern", re.compile(rf"(?<![a-z])(?:{alts})(?![a-z])"))


SOFT_SKILLS: tuple[SoftSkill, ...] = (
    SoftSkill("Communication", ("communication skills", "communicate", "communication", "communicator",
                                "communication ecrite", "aisance relationnelle", "kommunikationsfahigkeit",
                                "kommunikationsstark", "kommunikativ", "presentation skills", "storytelling")),
    SoftSkill("Teamwork", ("team player", "teamwork", "work in a team", "collaborative", "collaboration",
                           "esprit d'equipe", "travail en equipe", "teamfahigkeit", "teamplayer", "cross-functional")),
    SoftSkill("Problem solving", ("problem solving", "problem-solving", "solve complex problems", "resolution de problemes",
                                  "problemlosung", "analytical thinking", "analytical mindset", "esprit d'analyse",
                                  "analytisches denken", "analytical skills")),
    SoftSkill("Autonomy", ("autonomous", "autonomy", "autonome", "autonomie", "self-starter", "independently",
                           "selbststandig", "eigenverantwortlich", "self-motivated", "proactive", "proactif")),
    SoftSkill("Leadership", ("leadership", "lead a team", "people management", "leading teams", "fuhrungskompetenz",
                             "management d'equipe", "mentoring", "mentor")),
    SoftSkill("Curiosity", ("curious", "curiosity", "curiosite", "curieux", "eager to learn", "willingness to learn",
                            "lernbereitschaft", "neugier", "growth mindset", "passion for learning")),
    SoftSkill("Adaptability", ("adaptability", "adaptable", "flexible", "flexibility", "adaptabilite",
                               "fast-paced environment", "ambiguity", "anpassungsfahigkeit")),
    SoftSkill("Attention to detail", ("attention to detail", "detail-oriented", "detail oriented", "rigueur", "rigoureux",
                                      "rigorous", "sorgfalt", "sorgfaltig", "meticulous")),
    SoftSkill("Stakeholder management", ("stakeholder management", "stakeholders", "business partners",
                                         "parties prenantes", "client-facing", "customer-facing")),
    SoftSkill("Critical thinking", ("critical thinking", "esprit critique", "kritisches denken", "sound judgment")),
    SoftSkill("Ownership", ("ownership", "take ownership", "accountability", "sense of responsibility",
                            "sens des responsabilites", "verantwortungsbewusstsein")),
    SoftSkill("Organisation", ("organised", "organized", "organisation skills", "organizational skills",
                               "time management", "prioritize", "prioritise", "gestion du temps", "organisationstalent")),
    SoftSkill("Creativity", ("creative", "creativity", "creativite", "creatif", "kreativ", "innovative thinking")),
)
SOFT_NAMES = tuple(s.name for s in SOFT_SKILLS)


def find_soft_skills(text: str | None) -> dict[str, str]:
    """Soft skill -> first line mentioning it (verbatim, up to 240 chars)."""
    found: dict[str, str] = {}
    for line in (text or "").split("\n"):
        folded = fold(line)
        if not folded.strip():
            continue
        for skill in SOFT_SKILLS:
            if skill.name not in found and skill.pattern.search(folded):
                found[skill.name] = line.strip()[:240]
    return found
