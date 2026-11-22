"""`jobintel enrich`: structured postings (JSONL + SQLite) and a data-quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_intel.config import Settings
from job_intel.enrich.postings import enrich_all, quality_report, write_postings
from job_intel.enrich.store import rebuild
from job_intel.search.service import load_jobs


def add_enrich_command(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("enrich", help="extract structured fields (skills, level, sector, ...) into JSONL + SQLite")
    p.add_argument("--jobs-file", type=Path, help="input (default: data/processed/jobs.jsonl)")
    p.add_argument("--out", type=Path, help="output JSONL (default: data/processed/jobs_enriched.jsonl)")
    p.add_argument("--db", type=Path, help="SQLite file (default: data/jobintel.db)")


def run(args: argparse.Namespace, settings: Settings) -> int:
    jobs = load_jobs(args.jobs_file or settings.jobs_path)
    postings, rejected = enrich_all(jobs)
    out = args.out or settings.enriched_path
    write_postings(out, postings)
    n = rebuild(args.db or settings.db_path, postings)
    report = quality_report(postings, rejected)
    (out.parent / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"enriched {len(postings)} postings ({len(rejected)} rejected) -> {out}")
    print(f"SQLite: {n} jobs -> {args.db or settings.db_path}")
    print("field completeness:")
    for field, share in report["completeness"].items():
        print(f"  {field:<24}{share:>7.1%}")
    return 0 if postings else 1
