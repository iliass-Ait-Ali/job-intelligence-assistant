"""Label-free checks on explanations, plus the plumbing for a human audit.

Two measurements, both about the VERIFIER and the evidence trail, not about the truthfulness of a model:

* ``provenance``: of the snippets an explanation cites, how many appear verbatim (after whitespace/case
  normalisation) in the original CV text or listing description? This is independent of the verifier, so it
  catches bugs where the evidence pack drifts from its sources.
* ``verifier_stress``: take real, verified explanation items and corrupt them in constructed ways (invent a
  skill, add a degree, add a number, add a probability, cite a missing id, ...). What fraction does the
  verifier reject? These are synthetic negatives, so the result is a lower bound on neither real LLM
  hallucination rates nor on the verifier's recall against unseen phrasings.

What neither can say is whether the content of a real Llama answer is correct: only the human audit
(`grounding_audit.csv`, column `supported`) can. Every result is tagged with where the explanation came from:
`deterministic`, `llm-stub` (a scripted or mock client, NOT a real model) or `llm-ollama` (a real local model).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from job_intel.match.models import CandidateProfile
from job_intel.match.taxonomy import find_skills
from job_intel.rag.evidence import EvidencePack
from job_intel.rag.explain import Explanation, Gap, Strength, verify_items


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("…", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def client_kind(explanation: Explanation) -> str:
    return {"none": "deterministic", "ollama": "llm-ollama", "stub": "llm-stub"}.get(explanation.client_kind, "deterministic")


@dataclass
class Provenance:
    cited: int = 0
    verbatim: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        return self.verbatim / self.cited if self.cited else None


def provenance(exp: Explanation, cv_text: str, job_text: str) -> Provenance:
    """Do the cited snippets really occur in the source documents?"""
    sources = {"cv": norm(cv_text.replace("\n- ", "\n")), "job": norm(job_text)}
    result = Provenance()
    cited_ids = {r for item in (*exp.strengths, *exp.gaps, *exp.preparation) for r in (*item.cv_refs, *item.job_refs)}
    for sid in sorted(cited_ids):
        snippet = exp.pack.get(sid)
        if snippet is None or snippet.derived:  # derived snippets (spoken-language summary) are built, not copied
            continue
        result.cited += 1
        text = norm(snippet.text.removeprefix("- ")).strip(" .")
        source = sources[snippet.kind]
        if text and text in source:
            result.verbatim += 1
        else:
            result.missing.append(f"{sid}: {snippet.text[:70]}")
    return result


# ---------------------------------------------------------------------------------------------
# verifier stress test (constructed negatives)

DECOYS = ("Kubernetes", "Rust", "Scala", "Neo4j", "Tableau", "Terraform", "Julia", "Hadoop")


@dataclass
class StressResult:
    attempts: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    accepted_examples: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        total = sum(self.attempts.values())
        return sum(self.rejected.values()) / total if total else None

    def add(self, kind: str, was_rejected: bool, example: str) -> None:
        self.attempts[kind] += 1
        if was_rejected:
            self.rejected[kind] += 1
        elif len(self.accepted_examples) < 10:
            self.accepted_examples.append(f"{kind}: {example[:120]}")

    def merge(self, other: StressResult) -> None:
        self.attempts.update(other.attempts)
        self.rejected.update(other.rejected)
        self.accepted_examples = (self.accepted_examples + other.accepted_examples)[:10]


def _decoy(cited_text: str, profile: CandidateProfile) -> str | None:
    present = set(find_skills(cited_text)) | profile.skill_names
    return next((d for d in DECOYS if d not in present), None)


def _skill_elsewhere_in_pack(pack: EvidencePack, cited: str, profile: CandidateProfile) -> str | None:
    """A skill that appears in some OTHER snippet of the pack but not in the cited ones."""
    here = set(find_skills(cited))
    for snippet in (*pack.job, *pack.cv):
        for name in find_skills(snippet.text):
            if name not in here:
                return name
    return None


def verifier_stress(exp: Explanation, profile: CandidateProfile) -> StressResult:
    """Corrupt the real, verified strengths of `exp` in eight ways and check each corruption is rejected."""
    out = StressResult()
    pack: EvidencePack = exp.pack
    for s in exp.strengths[:3]:
        cited = " ".join(sn.text for r in (*s.cv_refs, *s.job_refs) if (sn := pack.get(r)))
        skill = (find_skills(cited) or ["Python"])[0]
        decoy = _decoy(cited, profile)
        mutants: list[tuple[str, str, object]] = []
        if decoy:
            mutants.append(("invented_skill", "strength", Strength(claim=f"The candidate has deep {decoy} experience.", cv_refs=s.cv_refs, job_refs=s.job_refs)))
        mutants += [
            ("invented_degree", "strength", Strength(claim=f"The candidate holds a PhD and knows {skill}.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("invented_number", "strength", Strength(claim=f"The candidate has 17 years of {skill} experience.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("hiring_probability", "strength", Strength(claim=f"With {skill}, the candidate has an 85% chance of being hired.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("salary_claim", "strength", Strength(claim=f"{skill} roles pay a salary above average.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("url", "strength", Strength(claim=f"Read more about {skill} at https://example.org/x.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("nonexistent_citation", "strength", Strength(claim=f"The CV shows {skill}.", cv_refs=["CV999"], job_refs=s.job_refs)),
            ("missing_listing_citation", "strength", Strength(claim=f"The CV shows {skill}.", cv_refs=s.cv_refs, job_refs=[])),
            ("gap_about_a_skill_the_cv_has", "gap", Gap(requirement=f"{skill} is not evidenced", job_refs=s.job_refs)),
            # subtler corruptions: no number, no degree, no URL, and (for the last two) no taxonomy skill
            ("invented_employer", "strength", Strength(claim=f"The candidate used {skill} while working at Google.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("inflated_skill_level", "strength", Strength(claim=f"The candidate is an expert in {skill}.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
            ("qualitative_claim_without_entities", "strength", Strength(claim="The candidate has proven leadership and managed a team of engineers.", cv_refs=s.cv_refs, job_refs=s.job_refs)),
        ]
        other = _skill_elsewhere_in_pack(pack, cited, profile)
        if other:
            mutants.append(("skill_from_an_uncited_snippet", "strength", Strength(claim=f"The CV shows {other}.", cv_refs=s.cv_refs, job_refs=s.job_refs)))
        for name, kind, item in mutants:
            kept, rejected = verify_items(kind, [item], pack, profile)
            out.add(name, not kept, getattr(item, "claim", getattr(item, "requirement", "")))
    return out


def audit_summary(modes: Counter, prov: Provenance, stress: StressResult) -> dict[str, object]:
    return {
        "explanation_modes": dict(modes),
        "real_llm_evaluated": modes.get("llm-ollama", 0) > 0,
        "provenance": {"cited_snippets": prov.cited, "verbatim_in_source": prov.verbatim, "rate": prov.rate, "examples_not_verbatim": prov.missing[:5]},
        "verifier_stress": {"attempts": dict(stress.attempts), "rejected": dict(stress.rejected), "rate": stress.rate,
                            "accepted_corruptions": stress.accepted_examples},  # fmt: skip
    }
