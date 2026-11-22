"""Does the compatibility score rank a candidate's eligible jobs well? Are the weights justified?

Data     3 synthetic CVs x the jobs of the frozen corpus with a graded label (assistant-proposed grades
         0/1/2 in ``data/eval/proposals/matching_label_proposals.csv``).
Metric   nDCG@10 on the *condensed* list (only labelled, eligible jobs are ranked), so unlabelled jobs
         neither help nor hurt any method; averaged over candidates.
Methods  retrieval (RRF of the CV queries) · rule-based evidence level (the matcher's order) · observed
         score · ranking score with the default weights · ranking score with weights FITTED by random search
         (3,000 Dirichlet samples) under leave-one-candidate-out cross-validation: weights are chosen on two
         CVs and scored on the third, three times.
Also     agreement of the hard-constraint exclusions with the labels' eligibility judgement.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from job_intel.config import Settings
from job_intel.evaluation import metrics as M
from job_intel.evaluation.data import matching_labels, open_eval_searcher, write_report
from job_intel.match.cv_parser import parse_cv
from job_intel.match.matcher import match_candidate
from job_intel.match.models import Preferences
from job_intel.match.scoring import DEFAULT_WEIGHTS, score_match

CV_DIR = Path(__file__).resolve().parents[3] / "eval" / "synthetic_cvs"
LABEL_NOTE = ("Grades: assistant-proposed (written by the AI that built the system, blind to its rankings), NOT "
              "human-verified; only 3 synthetic candidates, so fitted weights can easily over-fit - which is exactly what "
              "the leave-one-candidate-out protocol checks.")  # fmt: skip


def _ndcg(order: list, grades: dict, k: int = 10) -> float | None:
    ranked = [grades[j] for j in order if j in grades]
    return M.ndcg_at_k(ranked, list(grades.values()), k)


def _collect(settings: Settings):
    searcher = open_eval_searcher(settings, with_reranker=False)
    labels = matching_labels(settings)
    data = {}
    for cv_path in sorted(CV_DIR.glob("*.txt")):
        cid = cv_path.stem
        profile = parse_cv(cv_path.read_text(encoding="utf-8"))
        prefs = Preferences(**json.loads(cv_path.with_suffix(".prefs.json").read_text(encoding="utf-8")))
        prefs = prefs.model_copy(update={"max_results": 100})
        result = match_candidate(profile, prefs, searcher, pool_size=120)
        rows = [r for r in labels if r["candidate_id"] == cid]
        eligible = {m.job.job_id: m for m in result.matches}
        excluded = {m.job.job_id for m in result.excluded}
        grades = {r["job_id"]: int(r["proposed_grade"]) for r in rows if r["job_id"] in eligible and r["proposed_grade"] in "012"}
        data[cid] = {"profile": profile, "matches": eligible, "order_evidence": [m.job.job_id for m in result.matches],
                     "grades": grades, "rows": rows, "excluded": excluded}  # fmt: skip
    return data


def _order_by_weights(d: dict, weights: dict, key: str = "ranking_score") -> list[str]:
    scored = [(jid, score_match(m, d["profile"], weights)) for jid, m in d["matches"].items() if jid in d["grades"]]
    return [jid for jid, s in sorted(scored, key=lambda x: (-getattr(x[1], key), x[0]))]


def _random_weights(rng: random.Random) -> dict[str, float]:
    raw = {k: rng.gammavariate(1.0, 1.0) for k in DEFAULT_WEIGHTS}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def evaluate(settings: Settings, n_samples: int = 3000, seed: int = 0) -> dict:
    data = _collect(settings)
    cids = [c for c in data if len(data[c]["grades"]) >= 3]
    methods: dict[str, dict[str, float | None]] = {}

    def record(name: str, orders: dict[str, list[str]]) -> None:
        vals = {c: _ndcg(orders[c], data[c]["grades"]) for c in cids}
        methods[name] = {"mean_ndcg@10": sum(v for v in vals.values() if v is not None) / len(cids), **{f"ndcg_{c}": v for c, v in vals.items()}}

    record("retrieval order (RRF)", {c: sorted(data[c]["grades"], key=lambda j: -data[c]["matches"][j].retrieval.rrf) for c in cids})
    record("rule-based evidence level", {c: [j for j in data[c]["order_evidence"] if j in data[c]["grades"]] for c in cids})
    record("observed score (default weights)", {c: _order_by_weights(data[c], DEFAULT_WEIGHTS, "score") for c in cids})
    record("ranking score (default weights)", {c: _order_by_weights(data[c], DEFAULT_WEIGHTS) for c in cids})

    rng = random.Random(seed)
    candidates = [_random_weights(rng) for _ in range(n_samples)]
    cache = {c: [_ndcg(_order_by_weights(data[c], w), data[c]["grades"]) or 0.0 for w in candidates] for c in cids}
    loco_orders, chosen = {}, {}
    for test in cids:
        train = [c for c in cids if c != test]
        best = max(range(n_samples), key=lambda i: sum(cache[c][i] for c in train))
        chosen[test] = {k: round(v, 3) for k, v in candidates[best].items()}
        loco_orders[test] = _order_by_weights(data[test], candidates[best])
    record("ranking score (weights fitted, leave-one-CV-out)", loco_orders)
    best_all = max(range(n_samples), key=lambda i: sum(cache[c][i] for c in cids))
    in_sample = sum(cache[c][best_all] for c in cids) / len(cids)

    # eligibility agreement: does the system exclude what the labels call ineligible?
    tp = fp = fn = tn = unknown = 0
    for c in data:
        for r in data[c]["rows"]:
            flag = r.get("proposed_eligible_ok", "?")
            excluded = r["job_id"] in data[c]["excluded"]
            if flag == "n":
                tp += excluded
                fn += not excluded
            elif flag == "y":
                fp += excluded
                tn += not excluded
            else:
                unknown += 1
    default_mean = methods["ranking score (default weights)"]["mean_ndcg@10"]
    loco_mean = methods["ranking score (weights fitted, leave-one-CV-out)"]["mean_ndcg@10"]
    conclusion = (f"Fitted weights reach nDCG@10 {in_sample:.3f} in-sample but {loco_mean:.3f} out-of-sample vs {default_mean:.3f} "
                  f"for the documented default weights; " + ("the fit does not generalise, so the default weights are kept."
                  if loco_mean <= default_mean + 0.01 else "the fitted weights generalise better and are worth adopting after human review."))  # fmt: skip
    return {
        "label_note": LABEL_NOTE, "candidates": cids, "labelled_eligible_pairs": {c: len(data[c]["grades"]) for c in cids},
        "methods": methods, "fitted_in_sample_mean_ndcg@10": in_sample, "weights_chosen_per_fold": chosen,
        "eligibility_agreement": {"excluded_and_labelled_ineligible": tp, "kept_but_labelled_ineligible": fn,
                                  "excluded_but_labelled_eligible": fp, "kept_and_labelled_eligible": tn,
                                  "label_says_unknown": unknown,
                                  "precision_of_exclusions": tp / (tp + fp) if tp + fp else None,
                                  "recall_of_exclusions": tp / (tp + fn) if tp + fn else None},
        "conclusion": conclusion,
    }  # fmt: skip


def main(settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    report = evaluate(settings)
    write_report("scoring_calibration.json", report)
    return report
