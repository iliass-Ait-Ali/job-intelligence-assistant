"""Evaluation data: the frozen corpus, its searcher, and the relevance labels.

Frozen corpus  ``data/eval_corpus/2026-10-23/`` - the 191 listings (and their Chroma index) that every
               retrieval / matching label refers to. Listings expire, so the corpus is frozen rather than
               re-collected; benchmarks never touch the live corpus.
Labels         Human grades take precedence when they exist (``eval/judgments.csv`` column ``grade``).
               Otherwise the benchmark uses the *assistant-proposed* grades
               (``data/eval/proposals/*.csv`` + ``eval/labels/*_ai.csv``), written by the same AI that built
               the system, blind to system rankings, with a per-query rubric. Every report states which
               source was used. Assistant labels are a starting point, not independent ground truth.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from job_intel.config import Settings

EVAL_CORPUS = "2026-10-23"
LABEL_NOTE_AI = ("Labels: assistant-proposed relevance grades (written by the AI assistant that built the system, blind to "
                 "system rankings, following a written per-query rubric), NOT human-verified. Treat differences between "
                 "systems as indicative; confidence intervals are over only {n} queries.")  # fmt: skip


@dataclass
class Labels:
    grades: dict[tuple[str, str], int]  # (query id, job id) -> 0/1/2
    source: str  # "human" | "assistant-proposed"


def corpus_dir(settings: Settings) -> Path:
    return settings.data_dir / "eval_corpus" / EVAL_CORPUS


def open_eval_searcher(settings: Settings, *, with_reranker: bool = True, faiss: bool = True):
    from job_intel.search.embedder import load_embedder, resolve_device
    from job_intel.search.index import JobIndex
    from job_intel.search.retrieve import Searcher
    from job_intel.search.service import load_jobs

    base = corpus_dir(settings)
    index = JobIndex.open_existing(base / "chroma")
    embedder = load_embedder(index.config.model_name, device=settings.device)
    jobs = {j["id"]: j for j in load_jobs(base / "jobs.jsonl")}
    if faiss:
        from job_intel.search.faiss_index import FaissChunkIndex

        index = FaissChunkIndex(index)
    reranker = None
    if with_reranker:
        from job_intel.search.rerank import CrossEncoderReranker

        device = resolve_device(settings.device)
        reranker = lambda: CrossEncoderReranker(device=device)  # noqa: E731
    return Searcher(index, embedder, jobs, reranker=reranker)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def retrieval_labels(settings: Settings) -> Labels:
    human = {(r["query_id"], r["job_id"]): int(r["grade"]) for r in _read_csv(settings.eval_dir / "judgments.csv")
             if r.get("grade", "").strip() in {"0", "1", "2"}}  # fmt: skip
    if human:
        return Labels(human, "human")
    grades: dict[tuple[str, str], int] = {}
    for path in (settings.data_dir / "eval" / "proposals" / "retrieval_label_proposals.csv",
                 settings.data_dir / "eval" / "proposals" / "retrieval_label_extension_ai.csv"):  # fmt: skip
        for r in _read_csv(path):
            g = (r.get("human_decision_grade") or r.get("proposed_grade") or r.get("grade") or "").strip()
            if g in {"0", "1", "2"}:
                grades[(r["query_id"], r["job_id"])] = int(g)
    return Labels(grades, "assistant-proposed")


def matching_labels(settings: Settings) -> list[dict[str, str]]:
    return _read_csv(settings.data_dir / "eval" / "proposals" / "matching_label_proposals.csv")


def load_queries(settings: Settings) -> list[dict]:
    return json.loads((settings.eval_dir / "queries.json").read_text(encoding="utf-8"))["queries"]


def write_report(name: str, data: dict) -> Path:
    out = Path(__file__).resolve().parents[3] / "reports"
    out.mkdir(exist_ok=True)
    path = out / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path
