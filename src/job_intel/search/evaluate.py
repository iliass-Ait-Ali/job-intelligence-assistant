"""Retrieval evaluation with explicit HUMAN relevance judgments.

Nothing in this module produces or guesses a grade. The workflow is:

1. ``eval/queries.json`` holds the test queries (and optional eligibility filters per query).
2. ``build_pool`` runs every system on every query, unions the top-``depth`` results of all
   systems (blinded: no system names, shuffled with a fixed seed) and writes them to
   ``eval/judgments.csv`` with an empty ``grade`` column. Re-running it only appends new items
   and never touches grades already entered.
3. A human fills in ``grade``: 0 = not relevant, 1 = partially relevant, 2 = highly relevant.
4. ``evaluate`` re-runs the systems and scores them against the grades. A query is scored only
   when its whole pool has been graded; the others are reported as pending. Metrics are
   withheld entirely while no query is complete.

Methodology notes (also in the README)
* Systems are compared on the same corpus, queries and eligibility filters, with no company cap.
* nDCG@k uses gain 2^grade - 1 and a log2 discount; the ideal ranking is built from the judged
  pool of that query, so it is relative to what was judged, not to every job in the corpus.
* Precision@5 and MRR@k need a binary label, so they are reported at two explicit thresholds:
  relevant = grade >= 1 and relevant = grade >= 2. Precision's denominator is always 5.
* Recall@k is "pooled recall": relevant results in the top k divided by relevant items in the
  judged pool. It is NOT recall over the whole corpus (that would need every job judged for every
  query) and it favours the systems that built the pool. Queries with no relevant item in the
  pool are excluded from it.
* Results that have no judgment count as not relevant; ``judged@k`` reports how many of the top k
  were judged so the effect can be seen.
* Few queries and a single annotator mean the averages are descriptive; per-query values and
  win/tie/loss counts are included instead of significance tests.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from job_intel.search.errors import EvalError
from job_intel.search.filters import JobFilter
from job_intel.search.retrieve import BM25, SEMANTIC, Searcher

SYSTEMS = (SEMANTIC, BM25)
GRADES = {0: "not relevant", 1: "partially relevant", 2: "highly relevant"}
THRESHOLDS = (1, 2)
JUDGMENT_COLUMNS = ["query_id", "job_id", "grade", "title", "notes"]  # title only: full details stay in git-ignored data/
K = 10


# ---------------------------------------------------------------- queries / judgments files


@dataclass(frozen=True)
class EvalQuery:
    id: str
    text: str
    lang: str
    filters: JobFilter
    notes: str = ""


def load_queries(path: Path) -> list[EvalQuery]:
    if not path.exists():
        raise EvalError(f"queries file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data["queries"]
    except (ValueError, KeyError, TypeError) as exc:
        raise EvalError(f"{path} must be a JSON object with a 'queries' list: {exc}") from exc
    queries, seen = [], set()
    for item in raw:
        if not item.get("id") or not item.get("text"):
            raise EvalError(f"every query needs 'id' and 'text': {item!r}")
        if item["id"] in seen:
            raise EvalError(f"duplicate query id {item['id']!r}")
        seen.add(item["id"])
        queries.append(
            EvalQuery(item["id"], item["text"], item.get("lang", ""), JobFilter.from_dict(item.get("filters")), item.get("notes", ""))
        )
    return queries


def read_judgments(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"query_id", "job_id", "grade"} - set(reader.fieldnames or [])
        if missing:
            raise EvalError(f"{path} is missing column(s): {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            grade = (row.get("grade") or "").strip()
            if grade not in {"", "0", "1", "2"}:
                raise EvalError(f"{path} line {line}: grade must be empty, 0, 1 or 2 (got {grade!r})")
            row["grade"] = grade
            rows.append(row)
    return rows


def write_judgments(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=JUDGMENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grade_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], int | None]:
    return {(r["query_id"], r["job_id"]): (int(r["grade"]) if r["grade"] != "" else None) for r in rows}


def progress(rows: Sequence[Mapping[str, str]]) -> tuple[int, int]:
    """(graded, total) pool items."""
    return sum(1 for r in rows if r["grade"] != ""), len(rows)


# ---------------------------------------------------------------- pool building


def build_pool(
    searcher: Searcher, queries: Sequence[EvalQuery], existing: Sequence[Mapping[str, str]], *, depth: int = K,
    systems: Sequence[str] = SYSTEMS,
) -> tuple[list[dict[str, str]], dict[str, int]]:  # fmt: skip
    """Merge the systems' top-`depth` results into the judgment rows. Existing grades are kept."""
    by_query: dict[str, list[dict[str, str]]] = {}
    for row in existing:
        by_query.setdefault(row["query_id"], []).append(dict(row))
    stats = {"queries": len(queries), "new_items": 0, "existing_items": len(existing)}
    for query in queries:
        rows = by_query.setdefault(query.id, [])
        known = {r["job_id"] for r in rows}
        pooled = {}
        for system in systems:
            response = searcher.search(query.text, engine=system, k=depth, flt=query.filters, max_per_company=None)
            for hit in response.hits:
                pooled.setdefault(hit.job.job_id, hit.job)
        fresh = [job for job_id, job in pooled.items() if job_id not in known]
        random.Random(f"pool:{query.id}").shuffle(fresh)  # blinded order: independent of system and rank
        for job in fresh:
            rows.append({"query_id": query.id, "job_id": job.job_id, "grade": "", "title": job.title, "notes": ""})
        stats["new_items"] += len(fresh)
    ordered = [row for q in queries for row in by_query.get(q.id, [])]
    ordered += [row for qid, rows in by_query.items() if qid not in {q.id for q in queries} for row in rows]
    return ordered, stats


def write_review_sheet(
    path: Path, queries: Sequence[EvalQuery], rows: Sequence[Mapping[str, str]], records: Mapping[str, Mapping[str, Any]],
    rubric: str,
) -> None:  # fmt: skip
    """A readable local file (kept under data/, git-ignored) to judge from: it contains job text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Relevance judging sheet", "", rubric, "", "Enter grades in `eval/judgments.csv` (column `grade`).", ""]
    for query in queries:
        pool = [r for r in rows if r["query_id"] == query.id]
        lines += [f"## {query.id} - {query.text}", f"*language: {query.lang or '?'} | filters: {query.filters.describe()}*", ""]
        for row in pool:
            rec = records.get(row["job_id"], {})
            excerpt = (rec.get("description") or "")[:1200].replace("\n", " ")
            lines += [
                f"### {row['job_id']}",
                f"**{row['title']}** - {rec.get('company') or 'n/a'} - {rec.get('location') or 'n/a'} - {rec.get('job_type') or 'n/a'}",
                rec.get("source_url", ""), "", excerpt, "",
            ]  # fmt: skip
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- metrics


def dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked: Sequence[int], pool_grades: Sequence[int], k: int) -> float:
    ideal = dcg(sorted(pool_grades, reverse=True), k)
    return dcg(ranked, k) / ideal if ideal > 0 else 0.0


def precision_at_k(ranked: Sequence[int], k: int, threshold: int) -> float:
    return sum(1 for g in ranked[:k] if g >= threshold) / k


def mrr_at_k(ranked: Sequence[int], k: int, threshold: int) -> float:
    for i, g in enumerate(ranked[:k], start=1):
        if g >= threshold:
            return 1.0 / i
    return 0.0


def pooled_recall_at_k(ranked: Sequence[int], pool_grades: Sequence[int], k: int, threshold: int) -> float | None:
    relevant_in_pool = sum(1 for g in pool_grades if g >= threshold)
    if relevant_in_pool == 0:
        return None
    return sum(1 for g in ranked[:k] if g >= threshold) / relevant_in_pool


# ---------------------------------------------------------------- evaluation


@dataclass
class SystemResult:
    ndcg: float
    precision: dict[int, float]
    mrr: dict[int, float]
    pooled_recall: dict[int, float | None]
    judged_at_k: float  # share of the top-k results that have a human grade
    returned: int


@dataclass
class QueryEval:
    query_id: str
    text: str
    status: str  # "scored" | "pending" | "no pool"
    graded: int = 0
    pool_size: int = 0
    eligible_jobs: int = 0
    systems: dict[str, SystemResult] = field(default_factory=dict)


@dataclass
class EvalReport:
    k: int
    thresholds: tuple[int, ...]
    queries: list[QueryEval]
    summary: dict[str, dict[str, Any]]
    win_tie_loss: dict[str, int]
    graded_items: int
    pool_items: int

    @property
    def scored(self) -> list[QueryEval]:
        return [q for q in self.queries if q.status == "scored"]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def evaluate(
    searcher: Searcher, queries: Sequence[EvalQuery], rows: Sequence[Mapping[str, str]], *, k: int = K,
    systems: Sequence[str] = SYSTEMS, thresholds: Sequence[int] = THRESHOLDS,
) -> EvalReport:  # fmt: skip
    grades = grade_lookup(rows)
    graded, total = progress(rows)
    per_query: list[QueryEval] = []
    for query in queries:
        pool = [r for r in rows if r["query_id"] == query.id]
        qe = QueryEval(query.id, query.text, "scored", sum(1 for r in pool if r["grade"] != ""), len(pool))
        if not pool:
            qe.status = "no pool"
        elif qe.graded < len(pool):
            qe.status = "pending"
        else:
            pool_grades = [g for (qid, _), g in grades.items() if qid == query.id and g is not None]
            eligible_counts = set()
            for system in systems:
                response = searcher.search(query.text, engine=system, k=k, flt=query.filters, max_per_company=None)
                eligible_counts.add(response.eligible_jobs)
                ranked_raw = [grades.get((query.id, h.job.job_id)) for h in response.hits]
                ranked = [g or 0 for g in ranked_raw]  # unjudged -> not relevant
                qe.systems[system] = SystemResult(
                    ndcg=ndcg_at_k(ranked, pool_grades, k),
                    precision={t: precision_at_k(ranked, 5, t) for t in thresholds},
                    mrr={t: mrr_at_k(ranked, k, t) for t in thresholds},
                    pooled_recall={t: pooled_recall_at_k(ranked, pool_grades, k, t) for t in thresholds},
                    judged_at_k=(sum(g is not None for g in ranked_raw) / k),
                    returned=len(response.hits),
                )
            if len(eligible_counts) != 1:  # would mean the systems were not compared on the same candidate set
                raise EvalError(f"{query.id}: systems saw different eligible sets {eligible_counts}")
            qe.eligible_jobs = eligible_counts.pop()
        per_query.append(qe)

    scored = [q for q in per_query if q.status == "scored"]
    if not scored:
        raise EvalError(
            f"no metrics yet: {graded}/{total} pool items have a human grade and no query is fully judged. "
            "Fill the 'grade' column in the judgments file (0, 1 or 2) for every row of at least one query."
        )
    summary: dict[str, dict[str, Any]] = {}
    for system in systems:
        results = [q.systems[system] for q in scored]
        n = len(results)
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        summary[system] = {
            "queries": n,
            f"ndcg@{k}": mean([r.ndcg for r in results]),
            **{f"precision@5(grade>={t})": mean([r.precision[t] for r in results]) for t in thresholds},
            **{f"mrr@{k}(grade>={t})": mean([r.mrr[t] for r in results]) for t in thresholds},
            **{
                f"pooled_recall@{k}(grade>={t})": mean([r.pooled_recall[t] for r in results if r.pooled_recall[t] is not None])
                for t in thresholds
            },
            **{
                f"pooled_recall_queries(grade>={t})": sum(1 for r in results if r.pooled_recall[t] is not None)
                for t in thresholds
            },
            f"judged@{k}": mean([r.judged_at_k for r in results]),
        }
    wtl = {"semantic_better": 0, "tie": 0, "bm25_better": 0}
    if SEMANTIC in systems and BM25 in systems:
        for q in scored:
            diff = q.systems[SEMANTIC].ndcg - q.systems[BM25].ndcg
            wtl["semantic_better" if diff > 1e-9 else "bm25_better" if diff < -1e-9 else "tie"] += 1
    return EvalReport(k, tuple(thresholds), per_query, summary, wtl, graded, total)


def format_report(report: EvalReport, per_query: bool = False) -> str:
    k = report.k
    scored = report.scored
    lines = [
        f"Human judgments: {report.graded_items}/{report.pool_items} pool items graded; "
        f"{len(scored)}/{len(report.queries)} queries fully judged and scored.",
    ]
    pending = [q for q in report.queries if q.status != "scored"]
    if pending:
        lines.append("Not scored: " + ", ".join(f"{q.query_id} ({q.status}, {q.graded}/{q.pool_size} graded)" for q in pending))
    lines += ["", "Averages over scored queries (descriptive; small n, single annotator, no significance claim):"]
    names = list(report.summary)
    first = report.summary[names[0]]
    width = max(len(m) for m in first)
    lines.append(f"{'metric':<{width}}  " + "  ".join(f"{n:>10}" for n in names))
    for metric in first:
        cells = []
        for n in names:
            v = report.summary[n][metric]
            cells.append(f"{'n/a':>10}" if v is None else f"{v:>10.3f}" if isinstance(v, float) else f"{v:>10}")
        lines.append(f"{metric:<{width}}  " + "  ".join(cells))
    lines += [
        "",
        "Notes: nDCG uses gain 2^grade-1 with the ideal ranking taken from the judged pool. Precision@5 and MRR are",
        "shown for both binary thresholds. 'pooled_recall' = relevant found in top-k / relevant in the judged pool;",
        "it is NOT full-corpus recall. Unjudged results count as not relevant; judged@k shows how many were judged.",
    ]
    if report.win_tie_loss and scored:
        lines.append(f"Per-query nDCG@{k}: {report.win_tie_loss}")
    if per_query:
        lines += ["", "Per query:"]
        for q in scored:
            cells = "  ".join(
                f"{s}: nDCG={r.ndcg:.3f} P@5(>=1)={r.precision[1]:.2f} judged@{k}={r.judged_at_k:.0%}"
                for s, r in q.systems.items()
            )
            lines.append(f"  {q.query_id} (eligible={q.eligible_jobs}) {cells}")
    return "\n".join(lines)
