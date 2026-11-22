"""Cross-encoder re-ranking (second stage of a retrieve-then-rerank pipeline).

Bi-encoder (first stage)   s(q, d) = cos(E(q), E(d)): query and document are embedded
                           *independently*, so document vectors can be pre-computed -> fast, scalable.
Cross-encoder (second)     s(q, d) = f([CLS] q [SEP] d): ONE transformer reads query and passage
                           *together*; every query token attends to every passage token, so it models
                           interactions (negation, "junior" vs "senior", which skill is required) the
                           bi-encoder compresses away. Cost: one forward pass per (query, passage)
                           pair, so it is only applied to the top-N candidates of the first stage.

Job score = max over the job's chunks of the cross-encoder logit (a job is relevant if one of its
passages is). Default model: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` - multilingual
(trained on mMARCO, 14 languages incl. FR/DE/NL), 118M parameters, Apache-2.0.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from job_intel.search.errors import DependencyError

log = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, *, device: str = "cpu", batch_size: int = 32,
                 max_length: int = 384) -> None:  # fmt: skip
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise DependencyError("sentence-transformers is required for re-ranking") from exc
        started = time.perf_counter()
        self.name = model_name
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size
        log.info("loaded cross-encoder %s on %s in %.1fs", model_name, device, time.perf_counter() - started)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, p) for p in passages]
        return np.asarray(self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), dtype=np.float32)


@dataclass
class RerankedJob:
    job_id: str
    score: float
    best_passage: str


def rerank_jobs(reranker, query: str, passages_by_job: dict[str, list[str]], max_chunks: int = 6) -> list[RerankedJob]:
    """Score every job by its best passage; returns jobs best first (ties by id for determinism)."""
    flat: list[tuple[str, str]] = [(jid, p) for jid, ps in passages_by_job.items() for p in ps[:max_chunks] if p]
    scores = reranker.score(query, [p for _, p in flat]) if flat else np.zeros(0)
    best: dict[str, RerankedJob] = {}
    for (jid, passage), s in zip(flat, scores):
        if jid not in best or float(s) > best[jid].score:
            best[jid] = RerankedJob(jid, float(s), passage)
    for jid in passages_by_job:  # a job without text keeps the lowest score
        best.setdefault(jid, RerankedJob(jid, float("-inf"), ""))
    return sorted(best.values(), key=lambda r: (-r.score, r.job_id))
