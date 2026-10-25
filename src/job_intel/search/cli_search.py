"""CLI commands for Phase 2: index, index-status, search, eval pool, eval run."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import date
from pathlib import Path

from job_intel.config import Settings
from job_intel.search import evaluate as ev
from job_intel.search.errors import SearchError
from job_intel.search.filters import JobFilter
from job_intel.search.retrieve import ENGINES, SEMANTIC, SearchResponse

log = logging.getLogger(__name__)
SEARCH_COMMANDS = {"index", "index-status", "search", "eval"}
RUBRIC = (
    "**Grades.** 2 = highly relevant (role, level and every constraint in the query are met); "
    "1 = partially relevant (related, but misses an important aspect); 0 = not relevant. "
    "Judge the job posting against the query only, not against what a search system ranked first."
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, help="base folder holding raw/, processed/, index/ (default: ./data)")
    p.add_argument("--jobs-file", type=Path, help="processed jobs JSONL (default: <data-dir>/processed/jobs.jsonl)")


def _add_filters(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("eligibility filters (applied before ranking, identically for every engine)")
    g.add_argument("--source", action="append", dest="sources", metavar="NAME", help="arbeitnow | remotive (repeatable)")
    g.add_argument("--job-type", action="append", dest="job_types", metavar="TYPE",
                   help="internship, working_student, full_time, ... or 'unknown' (repeatable)")  # fmt: skip
    g.add_argument("--remote", action="store_true", help="only listings that explicitly say remote")
    g.add_argument("--location", action="append", dest="locations", metavar="PLACE",
                   help="city/region/country, spelling-insensitive, e.g. Zurich, Paris, Nederland (repeatable)")  # fmt: skip
    g.add_argument("--since", metavar="YYYY-MM-DD", help="published on or after this date")
    g.add_argument("--include-unknown", action="store_true",
                   help="also keep jobs whose job type / location / date is missing when those filters are used")  # fmt: skip


def add_search_commands(sub: argparse._SubParsersAction) -> None:
    idx = sub.add_parser("index", help="embed the processed jobs into the local ChromaDB index (idempotent)")
    _common(idx)
    idx.add_argument("--model", help="SentenceTransformers model (default: intfloat/multilingual-e5-base)")
    idx.add_argument("--rebuild", action="store_true", help="drop the index and re-embed everything")
    idx.add_argument("--exclude-source", action="append", dest="exclude_sources", metavar="NAME",
                     help="do not index listings from this source, e.g. remotive (repeatable)")  # fmt: skip

    status = sub.add_parser("index-status", help="show what is in the index and how it was built")
    _common(status)

    search = sub.add_parser("search", help="natural-language job search")
    _common(search)
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10, help="number of jobs to return (default 10)")
    search.add_argument("--engine", choices=[*ENGINES, "both", "all"], default=SEMANTIC,
                        help="both = semantic + BM25; all = semantic + BM25 + hybrid (RRF)")
    search.add_argument("--model", help="must match the model the index was built with")
    search.add_argument("--max-per-company", type=int, metavar="N", help="presentation only: at most N jobs per company")
    _add_filters(search)

    evalp = sub.add_parser("eval", help="human-judged retrieval evaluation (BM25 vs semantic vs hybrid RRF)")
    esub = evalp.add_subparsers(dest="eval_command", required=True)
    for name, text in (("pool", "build/extend the blinded judgment pool"), ("run", "score systems against human grades")):
        p = esub.add_parser(name, help=text)
        _common(p)
        p.add_argument("--queries", type=Path, help="queries JSON (default: <eval-dir>/queries.json)")
        p.add_argument("--judgments", type=Path, help="judgments CSV (default: <eval-dir>/judgments.csv)")
        p.add_argument("--model", help="must match the model the index was built with")
    esub.choices["pool"].add_argument("--depth", type=int, default=ev.K, help="top-N of each system to pool (default 10)")
    esub.choices["run"].add_argument("--per-query", action="store_true", help="also print per-query values")
    esub.choices["run"].add_argument("--report-json", type=Path, help="also write the full report as JSON")
    esub.choices["run"].add_argument("--save-baseline", metavar="NAME", help="store these measured metrics as eval/baselines/retrieval_NAME.json")
    esub.choices["run"].add_argument("--compare-baseline", metavar="NAME", help="show metric changes versus a saved baseline")


def _filter_from_args(args: argparse.Namespace) -> JobFilter:
    try:
        since = date.fromisoformat(args.since) if args.since else None
    except ValueError as exc:
        raise SearchError(f"--since must be YYYY-MM-DD, got {args.since!r}") from exc
    return JobFilter(
        sources=tuple(args.sources or ()), job_types=tuple(args.job_types or ()), remote=True if args.remote else None,
        locations=tuple(args.locations or ()), since=since, include_unknown=args.include_unknown,
    )  # fmt: skip


def format_response(resp: SearchResponse, flt: JobFilter, model: str) -> str:
    label = {SEMANTIC: f"semantic, {model}", "hybrid": "hybrid: Reciprocal Rank Fusion of semantic + BM25"}.get(resp.engine, "BM25 lexical baseline")
    lines = [
        "",
        f'Query: "{resp.query}"   [{label}]',
        f"Eligible: {resp.eligible_jobs}/{resp.total_jobs} jobs ({flt.describe()})",
    ]
    if resp.engine == SEMANTIC:
        lines.append(f"Scores = cosine similarity of the best-matching chunk (higher is closer; {resp.chunks_fetched} chunks examined)")
    elif resp.engine == "hybrid":
        lines.append("Scores = sum of 1/(60+rank) over the semantic and BM25 rankings (only the order is meaningful)")
    else:
        lines.append("Scores = BM25 (unbounded; comparable only within this list, not with cosine scores)")
    for hit in resp.hits:
        job = hit.job
        score = f"{hit.score:.4f}"
        meta = ", ".join(x for x in (job.job_type, "remote" if job.remote else None, job.location) if x)
        when = job.published_date.isoformat() if job.published_date else "date n/a"
        lines += [
            f"{hit.rank:>2}. [{score}] {job.title} - {job.company or 'company n/a'}",
            f"      {meta or 'no type/location'} | {job.source} | {when}",
            f"      {job.url}",
            f"      > {hit.snippet}" if hit.snippet else "",
        ]
    if resp.hidden_by_cap:
        lines.append(f"({resp.hidden_by_cap} lower-ranked job(s) hidden by --max-per-company)")
    lines += [f"note: {n}" for n in resp.notes]
    return "\n".join(line for line in lines if line != "")


def _run(args: argparse.Namespace, settings: Settings) -> int:
    from job_intel.search import service  # heavy imports (numpy, chromadb) only when a search command runs

    jobs_path = args.jobs_file

    if args.command == "index":
        last = {"done": -1}

        def progress(done: int, total: int) -> None:
            if done == total or done - last["done"] >= 128:
                last["done"] = done
                print(f"  embedded {done}/{total} chunks", file=sys.stderr, flush=True)

        report, embedder, config = service.build_index(
            settings, jobs_path=jobs_path, model=args.model, rebuild=args.rebuild,
            exclude_sources=tuple(args.exclude_sources or ()), progress=progress,
        )  # fmt: skip
        print(f"Model:   {embedder.describe()}")
        print(f"Index:   {settings.index_dir}")
        print(f"Jobs:    {report.total_jobs} (added {report.added}, updated {report.updated}, "
              f"unchanged {report.unchanged}, removed {report.removed}, duplicate input ids {report.duplicate_inputs})")  # fmt: skip
        print(f"Chunks:  {report.total_chunks} in index; {report.embedded_chunks} embedded this run")
        if report.chunk_token_stats:
            s = report.chunk_token_stats
            print(f"Chunk tokens (full model input): min {s['min']}, median {s['median']}, max {s['max']} (limit {s['limit']})")
        print(f"Time:    {report.seconds:.1f}s")
        return 0

    if args.command == "index-status":
        status = service.index_status(settings)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0

    searcher, notes = service.open_searcher(settings, jobs_path=jobs_path, model=args.model)
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    model = searcher.index.config.model_name

    if args.command == "search":
        flt = _filter_from_args(args)
        engines = {"both": (SEMANTIC, "bm25"), "all": ENGINES}.get(args.engine, (args.engine,))
        for engine in engines:
            resp = searcher.search(args.query, engine=engine, k=args.k, flt=flt, max_per_company=args.max_per_company)
            print(format_response(resp, flt, model))
        return 0

    # eval
    eval_dir = settings.eval_dir
    queries_path = args.queries or eval_dir / "queries.json"
    judgments_path = args.judgments or eval_dir / "judgments.csv"
    queries = ev.load_queries(queries_path)
    rows = ev.read_judgments(judgments_path)

    if args.eval_command == "pool":
        merged, stats = ev.build_pool(searcher, queries, rows, depth=args.depth)
        ev.write_judgments(judgments_path, merged)
        records = {j["id"]: j for j in service.load_jobs(jobs_path or settings.jobs_path)}
        sheet = settings.eval_work_dir / "review.md"
        ev.write_review_sheet(sheet, queries, merged, records, RUBRIC)
        graded, total = ev.progress(merged)
        print(f"Pool: {stats['new_items']} new item(s) added to {judgments_path} ({total} total, {graded} graded).")
        exhaustive = [q.id for q in queries if q.exhaustive]
        if exhaustive:
            print(f"Exhaustive queries (every eligible job is in the pool, so true recall can be computed): {', '.join(exhaustive)}")
        print("Next: `jobintel label retrieval` (interactive, blinded, resumable).")
        print(f"Read the postings in {sheet} (local, git-ignored), then fill the 'grade' column (0/1/2).")
        print("The grades are yours to give; this tool never fills them in.")
        return 0

    report = ev.evaluate(searcher, queries, rows)
    print(ev.format_report(report, per_query=args.per_query))
    baselines = eval_dir / "baselines"
    if args.save_baseline or args.compare_baseline:
        config = searcher.index.config
        meta = {"model": config.model_name, "chunk_max_tokens": config.chunk_max_tokens, "chunk_overlap_tokens": config.chunk_overlap_tokens,
                "k": report.k, "systems": list(report.summary), "queries_total": len(queries)}  # fmt: skip
    if args.compare_baseline:
        saved = baselines / f"retrieval_{args.compare_baseline}.json"
        if not saved.exists():
            raise SearchError(f"no saved baseline {saved}")
        old = json.loads(saved.read_text(encoding="utf-8"))
        print(f"\nChange versus baseline '{args.compare_baseline}' (saved with {old.get('meta', {}).get('model')}):")
        print("\n".join("  " + line for line in ev.compare_to_baseline(report, old)))
    if args.save_baseline:
        baselines.mkdir(parents=True, exist_ok=True)
        target = baselines / f"retrieval_{args.save_baseline}.json"
        target.write_text(json.dumps(ev.baseline_payload(report, meta), indent=2, default=str), encoding="utf-8")
        print(f"\nBaseline saved to {target} (numbers only; safe to commit).")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return 0


def run(args: argparse.Namespace, settings: Settings) -> int:
    if getattr(args, "data_dir", None):
        settings = dataclasses.replace(settings, data_dir=args.data_dir)
    try:
        return _run(args, settings)
    except SearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
