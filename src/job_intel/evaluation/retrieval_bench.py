"""Retrieval benchmark: keyword vs BM25 vs dense vs hybrid vs hybrid + cross-encoder (Module 3 / section 5).

For each query of ``eval/queries.json`` and each system, the top 10 jobs (after the query's eligibility
filters, identical for every system) are scored against the graded pool. Unjudged results count as
non-relevant and lower ``judged@10``, which is reported so a system is never flattered by holes in the pool.
``--pool`` prints the unjudged (query, job) pairs so they can be labelled before scoring.
"""

from __future__ import annotations

import time
from statistics import median

from job_intel.config import Settings
from job_intel.evaluation import metrics as M
from job_intel.evaluation.data import LABEL_NOTE_AI, load_queries, open_eval_searcher, retrieval_labels, write_report
from job_intel.search.filters import JobFilter
from job_intel.search.retrieve import ALL_ENGINES

K = 10


def _filter(q: dict) -> JobFilter:
    f = q.get("filters") or {}
    return JobFilter(job_types=tuple(f.get("job_types", ())), locations=tuple(f.get("locations", ())),
                     remote=f.get("remote"), include_unknown=bool(f.get("include_unknown", False)))  # fmt: skip


def run_rankings(settings: Settings, searcher=None) -> tuple[dict, dict, list[dict]]:
    searcher = searcher or open_eval_searcher(settings)
    queries = load_queries(settings)
    rankings: dict[str, dict[str, list[str]]] = {e: {} for e in ALL_ENGINES}
    timings: dict[str, list[float]] = {e: [] for e in ALL_ENGINES}
    for engine in ALL_ENGINES:  # warm-up: first call builds indexes / loads the cross-encoder
        searcher.search(queries[0]["text"], engine=engine, k=K, flt=_filter(queries[0]))
    for q in queries:
        for engine in ALL_ENGINES:
            started = time.perf_counter()
            resp = searcher.search(q["text"], engine=engine, k=K, flt=_filter(q))
            timings[engine].append(1000 * (time.perf_counter() - started))
            rankings[engine][q["id"]] = [h.job.job_id for h in resp.hits]
    return rankings, timings, queries


def unjudged_pairs(settings: Settings) -> list[tuple[str, str]]:
    rankings, _, queries = run_rankings(settings)
    labels = retrieval_labels(settings).grades
    missing = []
    for q in queries:
        for engine in ALL_ENGINES:
            for jid in rankings[engine][q["id"]]:
                if (q["id"], jid) not in labels and (q["id"], jid) not in missing:
                    missing.append((q["id"], jid))
    return missing


def evaluate(settings: Settings, searcher=None) -> dict:
    rankings, timings, queries = run_rankings(settings, searcher)
    labels = retrieval_labels(settings)
    per_metric: dict[str, dict[str, list[float | None]]] = {}
    per_query: dict[str, dict[str, float | None]] = {}
    for engine in ALL_ENGINES:
        vals: dict[str, list[float | None]] = {m: [] for m in ("ndcg@10", "p@5", "recall@10", "mrr@10", "map@10", "judged@10", "p@5_strict", "mrr@10_strict")}
        for q in queries:
            judged = [g for (qid, _), g in labels.grades.items() if qid == q["id"]]
            ranked = [labels.grades.get((q["id"], jid)) for jid in rankings[engine][q["id"]]]
            ranked += [None] * (K - len(ranked))  # fewer than k results: the empty slots are misses
            vals["ndcg@10"].append(M.ndcg_at_k(ranked, judged, K))
            vals["p@5"].append(M.precision_at_k(ranked, 5, 1))
            vals["p@5_strict"].append(M.precision_at_k(ranked, 5, 2))
            vals["recall@10"].append(M.recall_at_k(ranked, judged, K, 1))
            vals["mrr@10"].append(M.reciprocal_rank(ranked, K, 1))
            vals["mrr@10_strict"].append(M.reciprocal_rank(ranked, K, 2))
            vals["map@10"].append(M.average_precision(ranked, judged, K, 1))
            vals["judged@10"].append(M.judged_at_k(ranked, K))
            per_query.setdefault(q["id"], {})[engine] = vals["ndcg@10"][-1]
        per_metric[engine] = vals
    systems = {}
    for engine, vals in per_metric.items():
        systems[engine] = {}
        for m, v in vals.items():
            mean, lo, hi = M.bootstrap_ci([x for x in v if x is not None])
            systems[engine][m] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": sum(x is not None for x in v)}
        systems[engine]["latency_ms"] = {"p50": median(timings[engine]), "max": max(timings[engine])}
    paired = {}
    base = per_metric["bm25"]["ndcg@10"]
    for engine in ALL_ENGINES:
        if engine == "bm25":
            continue
        other = per_metric[engine]["ndcg@10"]
        pairs = [(a, b) for a, b in zip(other, base) if a is not None and b is not None]
        paired[f"{engine} vs bm25"] = {
            "mean_diff": sum(a - b for a, b in pairs) / len(pairs) if pairs else None,
            "wins": sum(a > b + 1e-9 for a, b in pairs), "ties": sum(abs(a - b) <= 1e-9 for a, b in pairs),
            "losses": sum(a < b - 1e-9 for a, b in pairs),
            "paired_bootstrap_p": M.paired_bootstrap_p([a for a, _ in pairs], [b for _, b in pairs]),
        }  # fmt: skip
    note = LABEL_NOTE_AI.format(n=len(queries)) if labels.source != "human" else \
        f"Labels: human relevance grades ({len(labels.grades)} judgments)."
    return {
        "label_source": labels.source, "label_note": note, "corpus": "frozen 2026-10-23 snapshot (191 listings)",
        "n_queries": len(queries), "n_judgments": len(labels.grades), "k": K, "systems": systems,
        "paired_vs_bm25_ndcg@10": paired, "per_query": per_query,
        "definitions": "nDCG@10 gain 2^g-1; P@5, MRR@10, MAP@10, recall@10 (pooled) count grade>=1 as relevant; "
                       "p@5_strict counts grade 2 only; judged@10 = share of the top 10 with a grade.",
    }  # fmt: skip


def main(settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    report = evaluate(settings)
    write_report("retrieval_benchmark.json", report)
    return report
