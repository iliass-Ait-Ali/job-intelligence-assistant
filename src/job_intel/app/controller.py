"""Interface logic without any Gradio dependency (so it can be tested directly).

Privacy: CV text lives only in memory for the duration of a session. Uploaded temp files are
deleted right after they are read. Nothing is written to disk and nothing leaves the machine
except requests to a loopback Ollama server.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from job_intel.config import Settings
from job_intel.match.cv_parser import parse_cv
from job_intel.match.cv_text import CVExtractionError, ExtractedCV, from_file, from_pasted_text
from job_intel.match.matcher import MatchResult, match_candidate
from job_intel.match.models import CandidateProfile, Preferences
from job_intel.rag.compare import MAX_COMPARED, Comparison, compare_matches
from job_intel.rag.explain import Explanation, explain_match
from job_intel.rag.ollama import LLMError, OllamaClient
from job_intel.search.errors import SearchError
from job_intel.search.retrieve import Searcher

log = logging.getLogger(__name__)

JOB_TYPE_CHOICES = ["internship", "full_time", "working_student", "part_time", "contract", "freelance", "permanent"]
HARD_CHOICES = ["job_type", "location", "experience", "education", "language"]
LANGUAGES = {"English": "English", "Français": "French"}


@dataclass
class AppContext:
    settings: Settings
    searcher_factory: Callable[[], Searcher] | None = None
    client_factory: Callable[[], OllamaClient] | None = None
    _searcher: Searcher | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def searcher(self) -> Searcher:
        with self._lock:
            if self._searcher is None:
                if self.searcher_factory:
                    self._searcher = self.searcher_factory()
                else:
                    from job_intel.search import service

                    self._searcher, notes = service.open_searcher(self.settings)
                    for note in notes:
                        log.warning(note)
            return self._searcher

    def client(self) -> OllamaClient:
        if self.client_factory:
            return self.client_factory()
        return OllamaClient(model=self.settings.llm_model, base_url=self.settings.ollama_url, timeout=self.settings.llm_timeout)

    def llm_status(self) -> tuple[bool, str]:
        try:
            return True, self.client().check()
        except LLMError as exc:
            return False, str(exc)


# ---------------------------------------------------------------------------------------------
# profile


@dataclass
class ParsedCV:
    profile: CandidateProfile | None
    json_text: str = ""
    messages: list[str] = field(default_factory=list)
    error: str | None = None


def parse_cv_input(file_path: str | None, pasted: str | None, *, today=None) -> ParsedCV:
    messages: list[str] = []
    try:
        extracted: ExtractedCV
        if file_path:
            try:
                extracted = from_file(file_path)
            finally:  # the upload is a temporary copy of a personal document: do not leave it behind
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            if pasted and pasted.strip():
                messages.append("Both a file and pasted text were provided; the uploaded file was used.")
        else:
            extracted = from_pasted_text(pasted)
    except CVExtractionError as exc:
        return ParsedCV(None, error=str(exc))
    messages += extracted.warnings
    profile = parse_cv(extracted.text, today=today)
    messages += profile.warnings
    return ParsedCV(profile, profile.model_dump_json(indent=2), messages)


def profile_from_json(text: str) -> tuple[CandidateProfile | None, str | None]:
    if not text or not text.strip():
        return None, "The profile is empty. Parse a CV first."
    try:
        return CandidateProfile.model_validate_json(text), None
    except ValidationError as exc:
        first = exc.errors()[0]
        if first["type"] == "json_invalid":
            return None, f"The profile is not valid JSON: {first['msg']}"
        where = ".".join(str(p) for p in first["loc"]) or "profile"
        return None, f"The profile JSON is invalid at '{where}': {first['msg']}"
    except ValueError as exc:
        return None, f"The profile is not valid JSON: {exc}"


def _split(text: str | None) -> list[str]:
    return [p.strip() for p in re.split(r"[,;\n]", text or "") if p.strip()]


def build_preferences(roles: str, job_types: list[str] | None, locations: str, remote: str,
                      hard: list[str] | None, max_results: int = 15) -> Preferences:  # fmt: skip
    mapping = {"No preference": "any", "Prefer remote": "prefer", "Remote only": "only"}
    return Preferences(
        target_roles=_split(roles), job_types=list(job_types or []), locations=_split(locations),
        remote=mapping.get(remote, "any"), hard_constraints=list(hard or []), max_results=int(max_results),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# recommendations and analysis


def recommend(ctx: AppContext, profile: CandidateProfile, prefs: Preferences) -> MatchResult:
    return match_candidate(profile, prefs, ctx.searcher())


def job_labels(result: MatchResult) -> list[str]:
    return [f"{i}. {m.job.title} — {m.job.company or 'n/a'}" for i, m in enumerate(result.matches, 1)]


def _client_or_none(ctx: AppContext, use_llm: bool) -> tuple[OllamaClient | None, str | None]:
    if not use_llm:
        return None, None
    ok, message = ctx.llm_status()
    return (ctx.client(), None) if ok else (None, message)


def explain_job(ctx: AppContext, profile: CandidateProfile, result: MatchResult, label: str | None, *,
                use_llm: bool, language: str = "English") -> Explanation:  # fmt: skip
    index = _index_from_label(label, len(result.matches))
    match = result.matches[index]
    client, unavailable = _client_or_none(ctx, use_llm)
    exp = explain_match(profile, match, searcher=ctx.searcher(), queries=result.queries, client=client,
                        language=LANGUAGES.get(language, "English"))  # fmt: skip
    if unavailable:
        exp.notes.insert(0, f"Local language model unavailable: {unavailable} Showing the deterministic explanation.")
    return exp


def compare_jobs(ctx: AppContext, profile: CandidateProfile, result: MatchResult, labels: list[str], *,
                 use_llm: bool, language: str = "English") -> Comparison:  # fmt: skip
    if len(labels) < 2:
        raise ValueError("Select at least two jobs to compare.")
    if len(labels) > MAX_COMPARED:
        raise ValueError(f"Select at most {MAX_COMPARED} jobs to compare.")
    matches = [result.matches[_index_from_label(label, len(result.matches))] for label in labels]
    client, unavailable = _client_or_none(ctx, use_llm)
    comp = compare_matches(profile, matches, searcher=ctx.searcher(), queries=result.queries, client=client,
                           language=LANGUAGES.get(language, "English"))  # fmt: skip
    if unavailable:
        comp.notes.insert(0, f"Local language model unavailable: {unavailable}")
    return comp


def _index_from_label(label: str | None, size: int) -> int:
    m = re.match(r"\s*(\d+)\.", label or "")
    if not m or not 1 <= int(m.group(1)) <= size:
        raise ValueError("Select a job from the list first.")
    return int(m.group(1)) - 1


def user_message_for(exc: Exception) -> str:
    """One-line, non-technical message for any failure the UI can hit."""
    if isinstance(exc, (SearchError, LLMError, CVExtractionError, ValueError)):
        return str(exc)
    log.exception("unexpected error")
    return f"Unexpected error ({type(exc).__name__}). Details were written to the console log."


def cv_example_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.txt")) if directory.exists() else []
