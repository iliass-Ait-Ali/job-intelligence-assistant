"""`jobintel label ...`: enter human judgments in the terminal (see labeling.py for the safeguards)."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from job_intel import labeling
from job_intel.config import Settings
from job_intel.search.errors import EvalError, SearchError

LABEL_COMMANDS = {"label"}
RETRIEVAL_RUBRIC = "Grade the JOB against the QUERY only: 2 = highly relevant, 1 = partially, 0 = not relevant."
MATCHING_RUBRIC = "Grade the fit for THIS candidate, considering the CV and the stated preferences: 2 = strong, 1 = partial, 0 = no."


def add_label_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("label", help="enter human relevance judgments interactively (blinded, resumable, saved after every answer)")
    task = p.add_subparsers(dest="label_task", required=True)
    for name, text in (("retrieval", "grade query/job pairs (eval/judgments.csv)"),
                       ("matching", "grade candidate/job pairs and check role + eligibility (eval/match_judgments.csv)"),
                       ("skills", "verify the system's exact/transferable skill claims (eval/skill_judgments.csv)")):  # fmt: skip
        t = task.add_parser(name, help=text)
        t.add_argument("--data-dir", type=Path)
        t.add_argument("--jobs-file", type=Path)
        t.add_argument("--judgments", type=Path, help="judgment file to edit (default depends on the task)")
        t.add_argument("--only", metavar="ID", help="only this query id / candidate id")
        t.add_argument("--stats", action="store_true", help="only print progress, do not label")
        if name == "retrieval":
            t.add_argument("--queries", type=Path)
        if name == "matching":
            t.add_argument("--cvs", type=Path, help="folder of CVs the pool was built from (default: synthetic CVs)")
            t.add_argument("--no-extra", action="store_true", help="grade fit only; skip the role / eligibility questions")


def _progress_lines(rows: list[dict[str, str]], key: str, field: str) -> list[str]:
    groups: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return [f"  {g:<32} {sum(1 for r in rs if r.get(field))}/{len(rs)} done" for g, rs in groups.items()]


def run(args: argparse.Namespace, settings: Settings) -> int:
    try:
        return _run(args, dataclasses.replace(settings, data_dir=args.data_dir or settings.data_dir))
    except (SearchError, EvalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 2)
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted: everything answered so far is already saved.", file=sys.stderr)
        return 130


def _run(args: argparse.Namespace, settings: Settings) -> int:
    from job_intel.search import evaluate as se
    from job_intel.search import service

    records = {j["id"]: j for j in service.load_jobs(args.jobs_file or settings.jobs_path)}
    task = args.label_task

    if task == "retrieval":
        path = args.judgments or settings.eval_dir / "judgments.csv"
        queries = {q.id: q for q in se.load_queries(args.queries or settings.eval_dir / "queries.json")}
        rows = se.read_judgments(path)
        if not rows:
            raise EvalError(f"{path} has no pairs yet. Build them first: jobintel eval pool")
        fields, card, save = [labeling.GRADE_FIELD], labeling.retrieval_card(queries, records, RETRIEVAL_RUBRIC), (lambda r: se.write_judgments(path, r))
        key, field = "query_id", "grade"
    elif task == "matching":
        from job_intel.match import evaluate as me

        cv_dir = args.cvs or settings.eval_dir / "synthetic_cvs"
        path = args.judgments or default_match_judgments(settings, cv_dir)
        candidates = {c.id: c for c in me.load_candidates(cv_dir)}
        rows = me.read_match_judgments(path)
        if not rows:
            raise EvalError(f"{path} has no pairs yet. Build them first: jobintel match-eval pool")
        fields = [labeling.GRADE_FIELD] + ([] if args.no_extra else [labeling.ROLE_FIELD, labeling.ELIGIBLE_FIELD])
        card, save = labeling.matching_card(candidates, records, MATCHING_RUBRIC), (lambda r: me.write_match_judgments(path, r))
        key, field = "query_id", "grade"
    else:
        from job_intel.match import evaluate as me

        path = args.judgments or settings.eval_dir / "skill_judgments.csv"
        rows = me.read_skill_rows(path)
        if not rows:
            raise EvalError(f"{path} has no assertions yet. Build them first: jobintel match-eval skills-pool")
        fields, card, save = [labeling.SKILL_FIELD], labeling.skill_card, (lambda r: me.write_skill_rows(path, r))
        key, field = "candidate_id", "correct"

    print(f"Judgment file: {path}")
    print("\n".join(_progress_lines(rows, key, field)))
    if args.stats:
        return 0
    only = args.only
    summary = labeling.run_labeling(rows, fields, card, save, select=lambda r: only is None or r[key] == only)
    print(f"\nSaved. This session: {summary.labelled} labelled, {summary.skipped} skipped; {summary.remaining} still open in the selection.")
    return 0


def default_match_judgments(settings: Settings, cv_dir: Path) -> Path:
    """Synthetic CVs -> eval/match_judgments.csv (git-ignored). Any other folder (e.g. a real CV) stays under data/."""
    default = settings.eval_dir / "synthetic_cvs"
    if cv_dir.resolve() == default.resolve():
        return settings.eval_dir / "match_judgments.csv"
    return settings.eval_work_dir / f"match_judgments_{cv_dir.name}.csv"
