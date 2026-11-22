"""Command line interface: `jobintel <command>` (or `python -m job_intel <command>`).

Phase 1: collect.  Phase 2 (search extra): index, index-status, search, eval.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from job_intel.config import Settings
from job_intel.pipeline import SOURCE_NAMES, CollectionReport, build_connectors, run_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jobintel", description="Job & Internship Intelligence Assistant")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="fetch, clean, filter, deduplicate and export job listings")
    collect.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=list(SOURCE_NAMES))
    collect.add_argument("--max-pages", type=int, help="Arbeitnow pages to fetch (300 listings each)")
    collect.add_argument("--data-dir", type=Path, help="base folder for raw/ and processed/ (default: ./data)")
    collect.add_argument("--force-refresh", action="store_true", help="ignore fresh raw snapshots and re-fetch")

    from job_intel.search.cli_search import add_search_commands  # cheap: heavy deps load lazily

    add_search_commands(sub)

    from job_intel.match.cli_match import add_match_commands

    add_match_commands(sub)

    from job_intel.label_cli import add_label_commands

    add_label_commands(sub)

    from job_intel.enrich.cli_enrich import add_enrich_command

    add_enrich_command(sub)
    return parser


def format_summary(report: CollectionReport) -> str:
    lines = ["", "=== Collection summary ===", ""]
    header = f"{'source':<11}{'fetched':>8}{'invalid':>8}{'relevant':>9}{'final':>7}  origin"
    lines += [header, "-" * len(header)]
    for s in report.sources.values():
        origin = f"FAILED: {s.error}" if s.error else ("cached snapshot" if s.from_cache else f"live ({s.requests_made} req)")
        lines.append(f"{s.source:<11}{s.fetched:>8}{s.invalid:>8}{s.relevant:>9}{s.final:>7}  {origin}")
        lines += [f"{'':<11}warning: {w}" for w in s.warnings]
    lines += [
        "",
        f"Total fetched ........... {report.total_fetched}",
        f"Invalid (skipped) ....... {report.invalid_skipped}",
        f"Filtered out (not relevant) {report.filtered_out}",
        f"Relevant ................ {report.relevant}",
        f"Duplicates removed ...... {report.duplicates_removed} {report.duplicates_by_reason or ''}",
        f"FINAL records ........... {report.final}",
    ]
    if report.by_category:
        lines += ["", "By role family: " + ", ".join(f"{k}={v}" for k, v in report.by_category.items())]
    if report.by_job_type:
        lines.append("By job type:    " + ", ".join(f"{k}={v}" for k, v in report.by_job_type.items()))
    if report.final < 50 and not report.all_sources_failed:
        lines += ["", f"NOTE: only {report.final} relevant records are available (target was 50+)."]
    if report.outputs:
        lines += ["", "Wrote:"] + [f"  {path}" for path in report.outputs.values()]
    lines.append("Attribution: " + " | ".join(report.attribution.values()))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # job text is full of accents; never crash on a legacy console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not args.verbose:  # third-party libraries log every HTTP request at INFO
        for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "chromadb"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings.from_env()
    if args.command == "enrich":
        from job_intel.enrich.cli_enrich import run as run_enrich

        return run_enrich(args, settings)
    if args.command == "label":
        from job_intel.label_cli import run as run_label

        return run_label(args, settings)
    if args.command in {"profile", "match", "llm-status", "app", "match-eval", "regress"}:
        from job_intel.match.cli_match import run as run_match

        return run_match(args, settings)
    if args.command != "collect":
        from job_intel.search.cli_search import run

        return run(args, settings)
    if args.data_dir:
        settings = dataclasses.replace(settings, data_dir=args.data_dir)
    if args.max_pages:
        settings = dataclasses.replace(settings, arbeitnow_max_pages=args.max_pages)

    connectors = build_connectors(settings, args.sources)
    report, _ = run_collection(settings, connectors, force_refresh=args.force_refresh)
    print(format_summary(report))
    return 1 if report.all_sources_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
