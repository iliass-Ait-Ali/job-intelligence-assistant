"""Skill-extraction benchmark: precision / recall / F1 of the listing-side extractor against annotated listings.

Gold     ``data/eval/proposals/skills_gold_ai.json`` - 20 seeded random listings, annotated blind to the
         extractor (assistant-written; see its "rules" field). Skills without a taxonomy entry are marked
         ``OOV:`` - the extractor cannot find them by design, so recall is reported twice: overall, and
         in-vocabulary (gold restricted to taxonomy skills: the matching quality of the aliases themselves).
Also     the CV-side parser is measured by the existing synthetic-CV regression (``jobintel regress``), and the
         sector heuristic is scored on the same 20 listings (accuracy where it predicts, and coverage).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from job_intel.config import Settings
from job_intel.enrich.sector import classify_sector
from job_intel.evaluation import metrics as M
from job_intel.evaluation.data import write_report
from job_intel.match.requirements import extract_requirements
from job_intel.match.taxonomy import BY_NAME
from job_intel.search.service import load_jobs

LABEL_NOTE = ("Gold skills written by the AI assistant that built the extractor (blind to its output, written rules), "
              "NOT human-verified; a 20-listing dev set and a 12-listing held-out test set are small, so read the numbers as indicative.")  # fmt: skip


def evaluate(settings: Settings, gold_file: str = "skills_gold_ai.json") -> dict:
    gold_doc = json.loads((settings.data_dir / "eval" / "proposals" / gold_file).read_text(encoding="utf-8"))
    jobs = {j["id"]: j for j in load_jobs(settings.jobs_path)}
    tp = fp = fn = 0
    tp_iv = fn_iv = 0
    per_doc = []
    fp_counter: Counter[str] = Counter()
    fn_counter: Counter[str] = Counter()
    sector_hits = sector_pred = 0
    sector_rows = []
    for job_id, gold in gold_doc["items"].items():
        job = jobs.get(job_id)
        if job is None:
            continue
        predicted = {s.name for s in extract_requirements(job["title"], job.get("description")).skills}
        gold_set = set(gold["skills"])
        gold_iv = {s for s in gold_set if s in BY_NAME}
        a, b, c = M.prf(predicted, gold_set)
        tp, fp, fn = tp + a, fp + b, fn + c
        tp_iv += len(predicted & gold_iv)
        fn_iv += len(gold_iv - predicted)
        fp_counter.update(predicted - gold_set)
        fn_counter.update(gold_set - predicted)
        per_doc.append({"job_id": job_id, **M.f1_from_counts(a, b, c), "n_gold": len(gold_set), "n_pred": len(predicted)})
        sector, _ = classify_sector(job.get("company"), job.get("description"))
        accepted = set(gold["sector"]) if gold["sector"] else {None}
        if sector is not None:
            sector_pred += 1
            sector_hits += sector in accepted
        sector_rows.append({"job_id": job_id, "predicted": sector, "accepted": sorted(map(str, accepted))})
    micro = M.f1_from_counts(tp, fp, fn)
    macro = {k: sum(d[k] for d in per_doc) / len(per_doc) for k in ("precision", "recall", "f1")}
    n_gold = tp + fn
    return {
        "label_note": LABEL_NOTE, "n_docs": len(per_doc), "n_gold": n_gold, "n_predicted": tp + fp,
        "micro": micro, "macro": macro,
        "in_vocabulary_recall": tp_iv / (tp_iv + fn_iv) if tp_iv + fn_iv else None,
        "oov_share_of_gold": sum(1 for d in gold_doc["items"].values() for s in d["skills"] if s.startswith("OOV:")) / max(n_gold, 1),
        "errors": {"false_positives": fp_counter.most_common(15), "false_negatives": fn_counter.most_common(20)},
        "per_doc": per_doc,
        "sector": {"accuracy": sector_hits / sector_pred if sector_pred else None, "coverage": sector_pred / len(per_doc),
                   "rows": sector_rows},
    }  # fmt: skip


def main(settings: Settings | None = None) -> dict:
    """Headline = the HELD-OUT test set (never used for error analysis). The dev set is reported too, with the
    numbers measured before the error-analysis fixes (frozen in reports/skills_benchmark_v1_dev.json)."""
    settings = settings or Settings.from_env()
    report = evaluate(settings, "skills_gold_test_ai.json")
    report["set"] = "held-out test (12 listings, seed 43)"
    dev = evaluate(settings, "skills_gold_ai.json")
    report["dev_after_fixes"] = {k: dev[k] for k in ("n_docs", "n_gold", "micro", "macro", "in_vocabulary_recall", "sector")}
    v1 = Path(__file__).resolve().parents[3] / "reports" / "skills_benchmark_v1_dev.json"
    if v1.exists():
        old = json.loads(v1.read_text(encoding="utf-8"))
        report["dev_before_fixes"] = {k: old[k] for k in ("n_docs", "n_gold", "micro", "macro", "in_vocabulary_recall")}
    write_report("skills_benchmark.json", report)
    return report
