"""Label-free regression baselines: metrics that need no human judgment, so they can be re-checked after any change.

Tracked metrics (all "higher is better"):
* extraction.*      parser accuracy on synthetic CVs with known ground truth (see match/extraction_eval.py)
* matching.<cv>.*   share of the jobs that survive the hard constraints that actually reach the candidate pool
* grounding.*       evidence provenance, verifier pass rate and verifier stress-test rejection rate (deterministic mode)

Metrics that DO need human grades (retrieval nDCG/P@5/MRR, matching quality, grounding audit) are tracked by
`jobintel eval run --save-baseline/--compare-baseline` once judgments exist.

The baseline file stores numbers only (no job text, no CV text): eval/baselines/label_free.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from job_intel.config import Settings

BASELINE_NAME = "label_free.json"
TOLERANCE = 1e-6


def baseline_path(settings: Settings) -> Path:
    return settings.eval_dir / "baselines" / BASELINE_NAME


def collect_metrics(settings: Settings, cv_dir: Path, *, with_index: bool = True) -> tuple[dict[str, float], list[str]]:
    """Return (metrics, notes). Index-dependent metrics are skipped, with a note, when no index exists."""
    from job_intel.match.extraction_eval import evaluate_extraction

    notes: list[str] = []
    metrics: dict[str, float] = {}
    current = evaluate_extraction()
    legacy = evaluate_extraction(fallbacks=False)
    for key, value in current.overall.items():
        if key != "documents":
            metrics[f"extraction.{key}"] = value
    for key in ("degree_level_accuracy", "experience_date_recall", "language_f1"):
        metrics[f"extraction.worst_layout.{key}"] = min(m[key] for m in current.per_style.values())
    notes.append("Phase 3 parser on the same corpus (informational): " + ", ".join(
        f"{k}={legacy.overall[k]:.2f}" for k in ("degree_level_accuracy", "experience_date_recall", "language_f1")))

    if not with_index:
        return metrics, notes
    try:
        from job_intel.match import diagnostics
        from job_intel.match import evaluate as me
        from job_intel.search import service

        searcher, _ = service.open_searcher(settings)
        candidates = me.load_candidates(cv_dir)
    except Exception as exc:  # no index / no jobs file / no CVs: the label-free parser metrics still work
        notes.append(f"matching and grounding metrics skipped ({type(exc).__name__}: {exc})")
        return metrics, notes
    diags = diagnostics.run_diagnostics(candidates, searcher)
    metrics.update({f"matching.{k}": v for k, v in diagnostics.metrics(diags).items() if k.endswith("eligible_recall_in_pool")})
    notes.append("Phase 3 retrieve-then-filter pool recall of eligible jobs (informational): " + ", ".join(
        f"{d.candidate_id}={d.legacy.eligible_recall:.2f}" for d in diags if d.legacy.eligible_recall is not None))
    report = me.grounding_run(searcher, candidates, top_n=3)
    if report.provenance.rate is not None:
        metrics["grounding.provenance_rate"] = report.provenance.rate
    if report.proposed:
        metrics["grounding.deterministic_verifier_pass_rate"] = report.verified / report.proposed
    if report.stress.rate is not None:
        metrics["grounding.verifier_stress_rejection_rate"] = report.stress.rate
    return metrics, notes


def compare(current: dict[str, float], baseline: dict[str, float]) -> tuple[list[str], list[str]]:
    """(regressions, table lines)."""
    regressions, lines = [], []
    for name in sorted(set(current) | set(baseline)):
        new, old = current.get(name), baseline.get(name)
        if new is None or old is None:
            lines.append(f"  {name:<58} {'(new)' if old is None else '(missing now)'}")
            continue
        flag = ""
        if new < old - TOLERANCE:
            flag = "  <-- REGRESSION"
            regressions.append(name)
        elif new > old + TOLERANCE:
            flag = "  improved"
        lines.append(f"  {name:<58} {old:7.3f} -> {new:7.3f}{flag}")
    return regressions, lines


def load_baseline(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"] if path.exists() else {}


def save_baseline(path: Path, metrics: dict[str, float], notes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metrics": {k: round(v, 6) for k, v in sorted(metrics.items())}, "notes": notes,
                               "about": "Label-free regression baseline (numbers only). Regenerate deliberately with `jobintel regress --update`."}  # fmt: skip
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
