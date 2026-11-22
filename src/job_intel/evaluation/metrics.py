"""Ranking and extraction metrics, implemented from their definitions (unit-tested in tests/test_metrics.py).

Notation: ``ranked`` = grades of the returned documents in rank order (None = not judged, counted as 0),
``judged`` = all grades known for the query (the pool), ``t`` = relevance threshold (grade >= t is relevant).

* Precision@k   = |relevant in top k| / k
* Recall@k      = |relevant in top k| / |relevant in the judged pool|   ("pooled" recall: true recall is
                  unknowable without judging every document)
* RR            = 1 / rank of the first relevant document (0 if none in the top k); MRR = mean over queries
* AP@k          = (1 / min(R, k)) Σ_{i ≤ k, rel(i)} Precision@i ;  MAP = mean over queries
* DCG@k         = Σ_i (2^{g_i} − 1) / log2(i + 1) ;  nDCG@k = DCG@k / IDCG@k (ideal order of the pool)
* judged@k      = share of the top k that has a grade (low values make every other metric unreliable)
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def _g(x: int | None) -> int:
    return int(x) if x is not None else 0


def precision_at_k(ranked: Sequence[int | None], k: int, t: int = 1) -> float:
    return sum(_g(g) >= t for g in ranked[:k]) / k if k else 0.0


def recall_at_k(ranked: Sequence[int | None], judged: Sequence[int], k: int, t: int = 1) -> float | None:
    total = sum(g >= t for g in judged)
    return None if total == 0 else sum(_g(g) >= t for g in ranked[:k]) / total


def reciprocal_rank(ranked: Sequence[int | None], k: int, t: int = 1) -> float:
    for i, g in enumerate(ranked[:k], start=1):
        if _g(g) >= t:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[int | None], judged: Sequence[int], k: int, t: int = 1) -> float | None:
    total = sum(g >= t for g in judged)
    if total == 0:
        return None
    hits, acc = 0, 0.0
    for i, g in enumerate(ranked[:k], start=1):
        if _g(g) >= t:
            hits += 1
            acc += hits / i
    return acc / min(total, k)


def dcg(grades: Sequence[int | None], k: int) -> float:
    return sum((2 ** _g(g) - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(ranked: Sequence[int | None], judged: Sequence[int], k: int) -> float | None:
    ideal = dcg(sorted(judged, reverse=True), k)
    return None if ideal == 0 else dcg(ranked, k) / ideal


def judged_at_k(ranked: Sequence[int | None], k: int) -> float:
    top = list(ranked[:k])
    return sum(g is not None for g in top) / len(top) if top else 0.0


def bootstrap_ci(values: Sequence[float], *, n: int = 10_000, alpha: float = 0.05, seed: int = 0,
                 stat: Callable[[Sequence[float]], float] = lambda v: sum(v) / len(v)) -> tuple[float, float, float]:  # fmt: skip
    """(statistic, low, high): percentile bootstrap over the units in `values` (here: queries)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(vals) for _ in vals]) for _ in range(n))
    lo, hi = boots[int(alpha / 2 * n)], boots[int((1 - alpha / 2) * n) - 1]
    return stat(vals), lo, hi


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], *, n: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for mean(a - b) = 0 (resampling queries)."""
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if not diffs:
        return float("nan")
    observed = sum(diffs) / len(diffs)
    centered = [d - observed for d in diffs]
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n):
        sample = [rng.choice(centered) for _ in centered]
        if abs(sum(sample) / len(sample)) >= abs(observed) - 1e-12:
            extreme += 1
    return extreme / n


def prf(predicted: set[str], gold: set[str]) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives)."""
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
