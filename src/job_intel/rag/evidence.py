"""Evidence packs: the numbered snippets (CV1.., JOB1..) that explanations may cite.

CV snippets are verbatim CV lines (plus one derived line listing spoken languages). Job snippets come
from two sources: (1) the passages of THIS job that the Phase 2 ChromaDB retriever finds most relevant
to the candidate's queries (the retrieval step of RAG), and (2) the listing lines where the
deterministic extractor found each requirement, so explicit requirements are always citable.
Everything passes through ``sanitize_snippet`` first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from job_intel.match.cv_parser import fold
from job_intel.match.matcher import CandidateQuery, JobMatch
from job_intel.match.models import CandidateProfile
from job_intel.match.taxonomy import find_skills
from job_intel.rag.sanitize import sanitize_snippet
from job_intel.search.retrieve import Searcher

MAX_CV_SNIPPETS = 14
MAX_JOB_SNIPPETS = 16
RETRIEVED_PASSAGES_PER_QUERY = 4


@dataclass(frozen=True)
class Snippet:
    id: str
    kind: Literal["cv", "job"]
    text: str
    section: str | None = None
    derived: bool = False  # built from structured fields rather than copied from a single source line


@dataclass
class EvidencePack:
    cv: list[Snippet] = field(default_factory=list)
    job: list[Snippet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # e.g. lines withheld as possible prompt injection

    def get(self, snippet_id: str) -> Snippet | None:
        return next((s for s in (*self.cv, *self.job) if s.id == snippet_id), None)

    def ids(self) -> set[str]:
        return {s.id for s in (*self.cv, *self.job)}

    def _add(self, kind: str, text: str, section: str | None = None, derived: bool = False, focus: str | None = None) -> str | None:
        """Add a snippet (sanitized, deduplicated, capped); return its id, or None if it was withheld."""
        clean, warning = sanitize_snippet(text, focus)
        if warning:
            self.warnings.append(warning)
        if clean is None:
            return None
        bucket, prefix, cap = (self.cv, "CV", MAX_CV_SNIPPETS) if kind == "cv" else (self.job, "JOB", MAX_JOB_SNIPPETS)
        for existing in bucket:
            if existing.text == clean:
                return existing.id
        if len(bucket) >= cap:
            return None
        snippet = Snippet(f"{prefix}{len(bucket) + 1}", kind, clean, section, derived)  # type: ignore[arg-type]
        bucket.append(snippet)
        return snippet.id

    def add_cv(self, text: str, section: str | None = None, derived: bool = False, focus: str | None = None) -> str | None:
        return self._add("cv", text, section, derived, focus)

    def add_job(self, text: str, focus: str | None = None) -> str | None:
        return self._add("job", text, focus=focus)

    def find(self, kind: str, text: str, focus: str | None = None) -> str | None:
        """Id of the snippet that equals the sanitized `text` (used to map evidence lines to ids)."""
        bucket = self.cv if kind == "cv" else self.job
        for variant in ([focus, None] if focus else [None]):  # the snippet may have been added with or without a focus
            clean, _ = sanitize_snippet(text, variant)
            for s in bucket:
                # when a skill is the focus, only a snippet that still mentions that skill can serve as its evidence
                if s.text == clean and (not focus or focus in find_skills(s.text)):
                    return s.id
        return None

    def as_json_payload(self) -> dict:
        return {
            "cv_snippets": [{"id": s.id, "text": s.text} for s in self.cv],
            "job_snippets": [{"id": s.id, "text": s.text} for s in self.job],
        }


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#]{3,}", fold(text))}


def _retrieved_job_lines(searcher: Searcher, job_id: str, queries: list[CandidateQuery]) -> list[str]:
    """Lines of this job's best-matching chunks for the candidate's queries (ChromaDB, restricted to the job)."""
    if not queries:
        return []
    vectors = searcher.embedder.embed_queries([q.text for q in queries])
    scored: dict[str, float] = {}
    for query, vector in zip(queries, vectors):
        q_words = _content_words(query.text)
        for hit in searcher.index.query(vector, RETRIEVED_PASSAGES_PER_QUERY, job_ids=[job_id]):
            for line in (ln.strip() for ln in hit.text.split("\n")):
                if len(line) < 25:
                    continue
                overlap = len(q_words & _content_words(line))
                score = hit.similarity + 0.05 * overlap + (0.1 if find_skills(line) else 0.0)
                scored[line] = max(scored.get(line, -1.0), score)
    return [line for line, _ in sorted(scored.items(), key=lambda kv: -kv[1])]


def _add_languages(pack: EvidencePack, profile: CandidateProfile) -> None:
    if profile.spoken_languages:
        listed = "; ".join(f"{s.language} ({s.level_text})" if s.level_text else s.language for s in profile.spoken_languages)
        pack.add_cv(f"Languages listed in the CV: {listed}", "languages", derived=True)


def build_evidence_pack(profile: CandidateProfile, match: JobMatch, searcher: Searcher | None = None,
                        queries: list[CandidateQuery] | None = None) -> EvidencePack:  # fmt: skip
    pack = EvidencePack()

    # --- job side: requirement lines first (guaranteed), then retrieved passages
    reqs = match.requirements
    for s in match.skills:
        pack.add_job(s.job_evidence, focus=s.skill)
    for _years, line, _lvl in reqs.years[:2]:
        pack.add_job(line)
    if reqs.degree:
        pack.add_job(reqs.degree[1])
    for lang in reqs.languages[:3]:
        pack.add_job(lang.evidence)
    if searcher is not None:
        for line in _retrieved_job_lines(searcher, match.job.job_id, queries or [])[:8]:
            pack.add_job(line)

    # --- CV side: spoken languages first when the listing mentions a language, then the lines that evidence
    # matched skills, then the entries those skills come from
    if reqs.languages:
        _add_languages(pack, profile)
    for s in sorted(match.skills, key=lambda s: {"exact": 0, "transferable": 1, "not_evidenced": 2}[s.status]):
        for ev in s.cv_evidence[:2]:
            pack.add_cv(ev.text, ev.section, focus=s.skill)
    wanted = {s.skill for s in match.skills if s.status != "not_evidenced"} | {v for s in match.skills for v in s.via}
    for exp in profile.experiences:
        if wanted & set(exp.skills):
            for ev in exp.evidence[:3]:
                pack.add_cv(ev.text, "experience")
    for proj in profile.projects:
        if wanted & set(proj.skills):
            for ev in proj.evidence[:3]:
                pack.add_cv(ev.text, "projects")
    for edu in profile.education[:3]:
        for ev in edu.evidence[:1]:
            pack.add_cv(ev.text, "education")
    _add_languages(pack, profile)
    return pack
