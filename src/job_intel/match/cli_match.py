"""CLI commands for Phase 3: profile, match, llm-status, app, match-eval."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from job_intel.config import Settings

MATCH_COMMANDS = {"profile", "match", "llm-status", "match-eval", "regress"}
RUBRIC = (
    "**Grades.** 2 = strong fit for this candidate (role, skills and constraints all fit); 1 = partial fit (related but with an "
    "important gap, or a constraint unclear); 0 = not a fit. Judge the job against the CV only, not against any system's ranking."
)


def add_match_commands(sub: argparse._SubParsersAction) -> None:
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", type=Path)
        p.add_argument("--jobs-file", type=Path)

    prof = sub.add_parser("profile", help="extract a structured profile from a CV (PDF/.txt) and print it as JSON")
    prof.add_argument("--cv", type=Path, required=True)
    prof.add_argument("--out", type=Path, help="write the JSON here (default: print only; CV data is not saved unless asked)")

    match = sub.add_parser("match", help="recommend jobs for a CV, with evidence")
    common(match)
    match.add_argument("--cv", type=Path, required=True)
    match.add_argument("--roles", help="comma-separated target roles (default: roles stated in the CV)")
    match.add_argument("--job-type", action="append", dest="job_types", metavar="TYPE", help="internship, full_time, ... (repeatable)")
    match.add_argument("--location", action="append", dest="locations", metavar="PLACE")
    match.add_argument("--remote", choices=["any", "prefer", "only"], default="any")
    match.add_argument("--hard", action="append", dest="hard", metavar="NAME",
                       help="constraints that exclude clearly violating jobs (default: job_type, location)")  # fmt: skip
    match.add_argument("-n", type=int, default=10, help="number of recommendations")
    match.add_argument("--explain", type=int, default=0, metavar="N", help="also explain the top N recommendations")
    match.add_argument("--llm", action="store_true", help="use the local Ollama model for explanations (default: deterministic)")
    match.add_argument("--language", default="English", choices=["English", "French"])

    llm = sub.add_parser("llm-status", help="check the local Ollama server and model")
    llm.add_argument("--model")

    ev = sub.add_parser("match-eval", help="human-judged evaluation of CV matching, label-free diagnostics, explanation grounding")
    esub = ev.add_subparsers(dest="meval_command", required=True)
    for name, text in (("pool", "build the blinded (candidate, job) judgment pool incl. the exclusion / unseen-job audit"),
                       ("run", "score the systems and audit the system's decisions against YOUR judgments"),
                       ("diagnose", "label-free: why jobs are excluded and whether retrieval reaches the eligible ones"),
                       ("skills-pool", "sample the system's exact/transferable skill claims for you to verify"),
                       ("skills-score", "precision of the skill claims from YOUR marks"),
                       ("grounding", "run explanations; provenance + verifier stress test; export claims for a human audit"),
                       ("grounding-score", "summarise the human audit marks"),
                       ("extraction", "label-free: parser accuracy on synthetic CVs with known ground truth")):
        p = esub.add_parser(name, help=text)
        common(p)
        p.add_argument("--cvs", type=Path, help="folder of CVs (default: <eval-dir>/synthetic_cvs). Use a private folder for a real CV.")
    esub.choices["pool"].add_argument("--no-audit", action="store_true", help="only the top-N of each ordering (no exclusion / unseen audit)")
    esub.choices["grounding"].add_argument("--top", type=int, default=3)
    esub.choices["grounding"].add_argument("--llm", action="store_true", help="REAL local Ollama model (never the default; recorded as llm-ollama)")
    esub.choices["grounding"].add_argument("--model", help="Ollama model name (default: JOBINTEL_LLM_MODEL)")
    esub.choices["extraction"].add_argument("--show", type=int, default=12, help="how many imperfect documents to list")
    esub.choices["extraction"].add_argument("--compare-phase3", action="store_true", help="also run the Phase 3 parser (no fallbacks)")

    reg = sub.add_parser("regress", help="compare label-free metrics with the saved baseline (exit 1 on regression)")
    reg.add_argument("--update", action="store_true", help="rewrite the baseline with the current numbers")
    reg.add_argument("--no-index", action="store_true", help="only the parser metrics (no search index needed)")
    reg.add_argument("--data-dir", type=Path)


def _settings(args: argparse.Namespace, settings: Settings) -> Settings:
    if getattr(args, "data_dir", None):
        settings = dataclasses.replace(settings, data_dir=args.data_dir)
    return settings


def _client(settings: Settings, model: str | None = None):
    from job_intel.rag.ollama import OllamaClient

    return OllamaClient(model=model or settings.llm_model, base_url=settings.ollama_url, timeout=settings.llm_timeout)


def _prefs(args: argparse.Namespace):
    from job_intel.match.models import Preferences

    return Preferences(
        target_roles=[r.strip() for r in (args.roles or "").split(",") if r.strip()], job_types=args.job_types or [],
        locations=args.locations or [], remote=args.remote, hard_constraints=args.hard or ["job_type", "location"],
        max_results=args.n,
    )  # fmt: skip


def _print_matches(result, limit: int) -> None:
    print(f"\nPool: {result.pool_size} candidates; {len(result.excluded)} excluded by hard constraints; "
          f"showing {min(limit, len(result.matches))}.")  # fmt: skip
    for i, m in enumerate(result.matches[:limit], 1):
        c = m.counts
        print(f"\n{i:>2}. [{m.evidence_level}] {m.job.title} - {m.job.company or 'n/a'}")
        print(f"    {m.job.location or 'location n/a'} | {m.job.job_type or 'type n/a'} | {m.job.source} | role: {m.role_status}")
        print(f"    skills: {c['exact']} exact, {c['transferable']} related, {c['not_evidenced']} not evidenced (of {len(m.base_skills)})")
        print(f"    in CV: {', '.join(s.skill for s in m.skills if s.status == 'exact') or '-'}")
        miss = [s.skill for s in m.skills if s.status == 'not_evidenced' and s.job_level == 'required']
        print(f"    required, not evidenced: {', '.join(miss) or '-'}")
        flags = [f"{c.name}:{c.status}" for c in m.constraints if c.status != "satisfied"]
        if flags:
            print(f"    constraints to check: {', '.join(flags)}")
        print(f"    {m.job.url}")


def run(args: argparse.Namespace, settings: Settings) -> int:
    from job_intel.rag.ollama import LLMError
    from job_intel.search.errors import SearchError

    settings = _settings(args, settings)
    try:
        return _run(args, settings)
    except (SearchError, LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 2)


def _run(args: argparse.Namespace, settings: Settings) -> int:
    from job_intel.match.cv_parser import parse_cv
    from job_intel.match.cv_text import CVExtractionError, from_file

    if args.command == "llm-status":
        print(_client(settings, args.model).check())
        return 0

    if args.command in {"profile", "match"}:
        try:
            extracted = from_file(args.cv)
        except CVExtractionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        profile = parse_cv(extracted.text)
        for w in [*extracted.warnings, *profile.warnings]:
            print(f"warning: {w}", file=sys.stderr)
        if args.command == "profile":
            text = profile.model_dump_json(indent=2)
            if args.out:
                args.out.write_text(text, encoding="utf-8")
                print(f"Profile written to {args.out} (contains CV-derived data: keep it out of Git).")
            else:
                print(text)
            return 0

        from job_intel.rag.explain import explain_match
        from job_intel.search import service

        searcher, notes = service.open_searcher(settings, jobs_path=args.jobs_file)
        for n in notes:
            print(f"warning: {n}", file=sys.stderr)
        from job_intel.match.matcher import match_candidate

        result = match_candidate(profile, _prefs(args), searcher)
        _print_matches(result, args.n)
        for m in result.matches[: args.explain]:
            client = _client(settings) if args.llm else None
            exp = explain_match(profile, m, searcher=searcher, queries=result.queries, client=client, language=args.language)
            print(f"\n--- {m.job.title} ({exp.mode}{' ' + exp.model if exp.model else ''}; {exp.claims_verified}/{exp.claims_proposed} items verified)")
            for n in exp.notes:
                print(f"  note: {n}")
            for s in exp.strengths:
                print(f"  + {s.claim} [{', '.join(s.cv_refs + s.job_refs)}]")
            for g in exp.gaps:
                print(f"  - {g.requirement} [{', '.join(g.job_refs)}]")
            for p in exp.preparation:
                print(f"  > {p.suggestion} [{', '.join(p.cv_refs + p.job_refs)}]")
        return 0

    if args.command == "regress":
        return _run_regress(args, settings)
    return _run_eval(args, settings)


def _match_paths(settings: Settings, cv_dir: Path) -> dict[str, Path]:
    """Synthetic CVs use eval/ (git-ignored judgment files); any other folder - e.g. a REAL CV - stays under data/."""
    from job_intel.label_cli import default_match_judgments

    judgments = default_match_judgments(settings, cv_dir)
    synthetic = judgments.parent == settings.eval_dir
    tag = "" if synthetic else f"_{cv_dir.name}"
    return {
        "judgments": judgments,
        "meta": (settings.eval_dir if synthetic else settings.eval_work_dir) / f"match_pool_meta{tag}.json",
        "review": settings.eval_work_dir / f"match_review{tag}.md",
        "skills": (settings.eval_dir if synthetic else settings.eval_work_dir) / f"skill_judgments{tag}.csv",
        "audit": settings.eval_work_dir / f"grounding_audit{tag}.csv",
        "run": settings.eval_work_dir / f"grounding_run{tag}.json",
    }


def _run_eval(args: argparse.Namespace, settings: Settings) -> int:
    from job_intel.match import evaluate as me
    from job_intel.match import extraction_eval

    cmd = args.meval_command
    if cmd == "extraction":
        corpus = extraction_eval.generate_corpus()
        print(extraction_eval.format_extraction(extraction_eval.evaluate_extraction(corpus), show_failures=args.show))
        if args.compare_phase3:
            print("\n--- Phase 3 parser (no fallbacks), same corpus ---")
            print(extraction_eval.format_extraction(extraction_eval.evaluate_extraction(corpus, fallbacks=False), show_failures=0))
        return 0

    from job_intel.search import evaluate as se
    from job_intel.search import service

    cv_dir = args.cvs or settings.eval_dir / "synthetic_cvs"
    paths = _match_paths(settings, cv_dir)
    if cmd == "grounding-score":
        print(json.dumps(me.score_audit(paths["audit"]), indent=2))
        return 0
    if cmd == "skills-score":
        print(json.dumps(me.score_skills(me.read_skill_rows(paths["skills"])), indent=2, default=str))
        return 0

    candidates = me.load_candidates(cv_dir)
    searcher, notes = service.open_searcher(settings, jobs_path=args.jobs_file)
    for n in notes:
        print(f"warning: {n}", file=sys.stderr)

    if cmd == "diagnose":
        from job_intel.match import diagnostics

        print(diagnostics.format_diagnostics(diagnostics.run_diagnostics(candidates, searcher)))
        return 0

    if cmd == "pool":
        meta = me.read_meta(paths["meta"])
        rows, stats = me.build_pool(searcher, candidates, me.read_match_judgments(paths["judgments"]), meta=meta, audit=not args.no_audit)
        me.write_match_judgments(paths["judgments"], rows)
        me.write_meta(paths["meta"], meta)
        records = {j["id"]: j for j in service.load_jobs(args.jobs_file or settings.jobs_path)}
        me.write_review_sheet(paths["review"], candidates, rows, records, RUBRIC)
        graded, total = se.progress(rows)
        print(f"Pool: {stats['new_items']} new pair(s) ({stats['audit_items']} from the exclusion / unseen-job audit) in {paths['judgments']} "
              f"({total} total, {graded} graded). Strata are recorded separately in {paths['meta']} and are not shown while labelling.")
        print("Next: `jobintel label matching" + ("" if cv_dir == settings.eval_dir / "synthetic_cvs" else f" --cvs {cv_dir}") + "`. The grades are yours to give.")
        return 0

    if cmd == "run":
        report = me.evaluate_matching(searcher, candidates, me.read_match_judgments(paths["judgments"]), me.read_meta(paths["meta"]))
        print(me.format_match_report(report))
        return 0

    if cmd == "skills-pool":
        rows = me.build_skill_pool(searcher, candidates, me.read_skill_rows(paths["skills"]))
        me.write_skill_rows(paths["skills"], rows)
        print(f"{len(rows)} skill assertions in {paths['skills']} (git-ignored). Next: `jobintel label skills`.")
        return 0

    client = _client(settings, args.model) if args.llm else None
    if client is not None:
        print(f"Using local Ollama model {client.model!r}: this is a REAL-MODEL run and will be recorded as such.")
    report = me.grounding_run(searcher, candidates, top_n=args.top, client=client)
    me.write_audit(paths["audit"], report.audit_rows)
    summary = report.summary()
    summary["ollama_version"] = client.version() if client else None
    me.dump_json(paths["run"], summary)
    print(report.format())
    print(f"Audit sheet: {paths['audit']} - mark each row's 'supported' column yes / partial / no, then run `jobintel match-eval grounding-score`.")
    print(f"Run record (sources, counts, model): {paths['run']}")
    return 0


def _run_regress(args: argparse.Namespace, settings: Settings) -> int:
    from job_intel import regress

    cv_dir = settings.eval_dir / "synthetic_cvs"
    metrics, notes = regress.collect_metrics(settings, cv_dir, with_index=not args.no_index)
    path = regress.baseline_path(settings)
    for n in notes:
        print(f"note: {n}")
    if args.update:
        regress.save_baseline(path, metrics, notes)
        print(f"Baseline written to {path} ({len(metrics)} metrics). Commit it only if the change is intended.")
        return 0
    baseline = regress.load_baseline(path)
    if not baseline:
        print(f"No baseline at {path}. Create one with `jobintel regress --update`.", file=sys.stderr)
        return 2
    regressions, lines = regress.compare(metrics, baseline)
    print("\n".join(lines))
    if regressions:
        print(f"\n{len(regressions)} metric(s) got worse than the baseline: {', '.join(regressions)}", file=sys.stderr)
        return 1
    print("\nNo regression against the baseline.")
    return 0
