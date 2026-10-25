"""Evaluation harness for CV-to-job matching and for explanation grounding.

Nothing here produces a relevance label or a quality figure by itself. What a HUMAN is asked to judge, and
what each judgment is used for:

* ``grade`` (0/1/2) for (candidate, job) pairs: overall fit, considering the CV and the stated preferences.
  Used for ranking quality (nDCG@10, P@5, MRR@10) of ``pipeline`` vs ``retrieval_rrf`` on the same eligible jobs.
* ``role_ok`` (y/n/?): does the job's ROLE match the candidate's target role? Compared with the system's
  ``role_status`` -> role-relevance accuracy.
* ``eligible_ok`` (y/n/?): given the stated preferences (job type, location, remote), is the job acceptable?
  Compared with the system's decision -> eligibility correctness, incorrect exclusions.
* skill assertions (``skill_judgments.csv``): is each "exact" / "transferable" skill claim right?

The pool deliberately contains four strata (recorded in a side file the labeller never sees): the top-N of each
ordering, an AUDIT sample of the jobs retrieval liked most but the hard constraints excluded, and a random sample
of eligible jobs outside the top-N. The last two are what make "incorrect exclusions" and "missing relevant
recommendations" measurable instead of anecdotal.

``grounding_run`` explains the top matches and reports (a) how many items pass the automatic verifier, (b) how
many cited snippets occur verbatim in the source documents, (c) how often the verifier rejects constructed
corruptions, and exports the verified claims for a human audit. None of that proves the content is true.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from job_intel.match.cv_parser import parse_cv
from job_intel.match.cv_text import from_file
from job_intel.match.matcher import JobMatch, match_candidate
from job_intel.match.models import CandidateProfile, Preferences
from job_intel.rag import audit as rag_audit
from job_intel.rag.explain import explain_match
from job_intel.rag.ollama import OllamaClient
from job_intel.search import evaluate as search_eval
from job_intel.search.errors import EvalError
from job_intel.search.retrieve import Searcher

SYSTEMS = ("pipeline", "retrieval_rrf")
POOL_DEPTH = 10
EXCLUDED_AUDIT = 8
UNSEEN_SAMPLE = 6
K = 10
THRESHOLDS = search_eval.THRESHOLDS
MATCH_COLUMNS = ["query_id", "job_id", "grade", "role_ok", "eligible_ok", "title", "notes"]
RANKABLE = {"pipeline", "retrieval_rrf"}
YN = {"y", "n", "?", ""}


@dataclass
class EvalCandidate:
    id: str
    cv_path: Path
    prefs: Preferences
    profile: CandidateProfile = field(init=False)

    def load(self, today: date | None = None) -> EvalCandidate:
        self.profile = parse_cv(from_file(self.cv_path).text, today=today)
        return self


def load_candidates(cv_dir: Path) -> list[EvalCandidate]:
    """CVs (.txt or text-based .pdf) in a folder, each with an optional `<name>.prefs.json` of Preferences."""
    if not cv_dir.exists():
        raise EvalError(f"CV folder not found: {cv_dir}")
    candidates = []
    for cv in sorted([*cv_dir.glob("*.txt"), *cv_dir.glob("*.pdf")]):
        prefs_file = cv.with_suffix(".prefs.json")
        prefs = Preferences.model_validate_json(prefs_file.read_text(encoding="utf-8")) if prefs_file.exists() else Preferences()
        prefs = prefs.model_copy(update={"max_results": 100})  # evaluation ranks every eligible job
        candidates.append(EvalCandidate(cv.stem, cv, prefs).load())
    if not candidates:
        raise EvalError(f"no .txt/.pdf CVs in {cv_dir}")
    return candidates


def rankings(searcher: Searcher, candidate: EvalCandidate) -> dict[str, list[JobMatch]]:
    """Both systems order the same eligible jobs: only the ordering rule differs."""
    result = match_candidate(candidate.profile, candidate.prefs, searcher)
    pipeline = list(result.matches)
    rrf = sorted(pipeline, key=lambda m: (-m.retrieval.rrf, m.job.job_id))
    return {"pipeline": pipeline, "retrieval_rrf": rrf}


# ---------------------------------------------------------------------------------------------
# judgments files


def read_match_judgments(path: Path) -> list[dict[str, str]]:
    rows = search_eval.read_judgments(path)
    for row in rows:
        for column in ("role_ok", "eligible_ok"):
            value = (row.get(column) or "").strip().lower()  # files written before these columns existed simply lack them
            if value not in YN:
                raise EvalError(f"{path}: {column} must be y, n, ? or empty (got {value!r})")
            row[column] = value
    return rows


def write_match_judgments(path: Path, rows: list[dict[str, str]]) -> None:
    search_eval.write_judgments(path, rows, MATCH_COLUMNS)


def read_meta(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# pool building


def build_pool(searcher: Searcher, candidates: list[EvalCandidate], existing: list[dict[str, str]],
               depth: int = POOL_DEPTH, *, meta: dict[str, Any] | None = None, audit: bool = False,
               excluded_audit: int = EXCLUDED_AUDIT, unseen_sample: int = UNSEEN_SAMPLE) -> tuple[list[dict[str, str]], dict[str, int]]:  # fmt: skip
    """Add (candidate, job) pairs to be graded. Existing rows and grades are never touched.

    `meta` (mutated in place) receives the hidden stratum/ground-truth-of-the-system info for every pooled pair.
    """
    meta = meta if meta is not None else {}
    by_cand: dict[str, list[dict[str, str]]] = {}
    for row in existing:
        by_cand.setdefault(row["query_id"], []).append(dict(row))
    stats = {"candidates": len(candidates), "new_items": 0, "existing_items": len(existing), "audit_items": 0}
    for cand in candidates:
        rows = by_cand.setdefault(cand.id, [])
        known = {r["job_id"] for r in rows}
        cmeta = meta.setdefault(cand.id, {})
        ranks = rankings(searcher, cand)
        eligible_ids = {m.job.job_id for m in ranks["pipeline"]}
        pooled: dict[str, JobMatch] = {}
        for system, ordered in ranks.items():
            for rank, m in enumerate(ordered, start=1):
                info = cmeta.setdefault(m.job.job_id, {})
                info.update({"eligible": True, "role_status": m.role_status})
                info[f"{system}_rank"] = rank
                if rank <= depth:
                    pooled.setdefault(m.job.job_id, m)
                    info.setdefault("strata", [])
                    if system not in info["strata"]:
                        info["strata"].append(system)
        audit_jobs: dict[str, tuple[JobMatch, str]] = {}
        if audit:
            # (a) jobs retrieval likes most that the hard constraints excluded -> are the exclusions right?
            legacy = match_candidate(cand.profile, cand.prefs, searcher, prefilter=False)
            for m in sorted(legacy.excluded, key=lambda x: (-x.retrieval.rrf, x.job.job_id))[:excluded_audit]:
                audit_jobs.setdefault(m.job.job_id, (m, "excluded_audit"))
            # (b) a seeded random sample of eligible jobs outside both top-N lists -> what do we never surface?
            top = {m.job.job_id for system in RANKABLE for m in ranks[system][:depth]}
            rest = sorted(j for j in eligible_ids if j not in top)
            for jid in random.Random(f"unseen:{cand.id}").sample(rest, min(unseen_sample, len(rest))):
                m = next(x for x in ranks["pipeline"] if x.job.job_id == jid)
                audit_jobs.setdefault(jid, (m, "unseen_eligible"))
        for jid, (m, stratum) in audit_jobs.items():
            info = cmeta.setdefault(jid, {})
            info.update({"eligible": stratum != "excluded_audit", "role_status": m.role_status})
            info.setdefault("strata", [])
            if stratum not in info["strata"]:
                info["strata"].append(stratum)
            pooled.setdefault(jid, m)
        fresh = [m for jid, m in pooled.items() if jid not in known]
        random.Random(f"matchpool:{cand.id}").shuffle(fresh)  # blinded: order reveals neither system, rank nor stratum
        for m in fresh:
            rows.append({"query_id": cand.id, "job_id": m.job.job_id, "grade": "", "title": m.job.title, "notes": ""})
        stats["new_items"] += len(fresh)
        stats["audit_items"] += sum(1 for jid in audit_jobs if jid not in known)
    return [r for c in candidates for r in by_cand.get(c.id, [])], stats


def write_review_sheet(path: Path, candidates: list[EvalCandidate], rows: list[dict[str, str]],
                       records: dict[str, dict[str, Any]], rubric: str) -> None:  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# CV-to-job judging sheet (local, git-ignored)", "", rubric, "",
             "Prefer the interactive tool: `jobintel label matching`. This sheet is a read-only alternative.", ""]  # fmt: skip
    for cand in candidates:
        lines += [f"## Candidate {cand.id}", f"*preferences: {cand.prefs.model_dump_json()}*", "", "```text",
                  from_file(cand.cv_path).text.strip(), "```", ""]  # fmt: skip
        for row in (r for r in rows if r["query_id"] == cand.id):
            rec = records.get(row["job_id"], {})
            excerpt = (rec.get("description") or "")[:1500].replace("\n", " ")
            lines += [f"### {row['job_id']}",
                      f"**{row['title']}** - {rec.get('company') or 'n/a'} - {rec.get('location') or 'n/a'} - {rec.get('job_type') or 'n/a'}",
                      rec.get("source_url", ""), "", excerpt, ""]  # fmt: skip
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# statistics helpers


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; None when n == 0."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(successes: int, n: int) -> dict[str, Any]:
    ci = wilson_interval(successes, n)
    return {"k": successes, "n": n, "rate": (successes / n) if n else None, "ci95": ci}


# ---------------------------------------------------------------------------------------------
# matching evaluation


@dataclass
class SystemScores:
    ndcg: float
    precision: dict[int, float]
    mrr: dict[int, float]
    pooled_recall: dict[int, float | None]
    judged_at_k: float


@dataclass
class CandidateEval:
    candidate_id: str
    status: str
    graded: int
    pool_size: int
    systems: dict[str, SystemScores] = field(default_factory=dict)


@dataclass
class MatchEvalReport:
    candidates: list[CandidateEval]
    summary: dict[str, dict[str, Any]]
    graded_items: int
    pool_items: int
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _is_rankable(meta: dict[str, Any], cand_id: str, job_id: str) -> bool:
    info = meta.get(cand_id, {}).get(job_id)
    if not meta or info is None:
        return True  # no side file (older pools): every row is treated as rankable
    return bool(RANKABLE & set(info.get("strata", []))) or not info.get("strata")


def audit_metrics(rows: list[dict[str, str]], meta: dict[str, Any]) -> dict[str, Any]:
    """Role relevance, eligibility correctness, incorrect exclusions and missing relevant jobs, from human marks."""
    out: dict[str, Any] = {}
    cells = [(r, meta.get(r["query_id"], {}).get(r["job_id"])) for r in rows]
    cells = [(r, info) for r, info in cells if info is not None]

    cells = [({**r, "role_ok": r.get("role_ok", ""), "eligible_ok": r.get("eligible_ok", "")}, info) for r, info in cells]
    decided = [(r, info) for r, info in cells if r["eligible_ok"] in {"y", "n"}]
    agree = sum(1 for r, info in decided if info["eligible"] == (r["eligible_ok"] == "y"))
    out["eligibility_agreement"] = proportion(agree, len(decided))
    out["system_excluded_human_acceptable"] = proportion(
        sum(1 for r, i in decided if not i["eligible"] and r["eligible_ok"] == "y"), sum(1 for _, i in decided if not i["eligible"]))
    out["system_eligible_human_unacceptable"] = proportion(
        sum(1 for r, i in decided if i["eligible"] and r["eligible_ok"] == "n"), sum(1 for _, i in decided if i["eligible"]))

    role = [(r, i) for r, i in cells if r["role_ok"] in {"y", "n"} and i.get("role_status") in {"match", "no_match"}]
    out["role_relevance_agreement"] = proportion(sum(1 for r, i in role if (i["role_status"] == "match") == (r["role_ok"] == "y")), len(role))
    out["role_status_unknown_or_unjudged"] = sum(1 for r, i in cells if i.get("role_status") == "unknown" or r["role_ok"] in {"?", ""})

    unseen = [r for r, i in cells if "unseen_eligible" in i.get("strata", []) and r["grade"] != ""]
    out["unseen_eligible_relevant(grade>=1)"] = proportion(sum(1 for r in unseen if int(r["grade"]) >= 1), len(unseen))
    out["unseen_eligible_strong(grade=2)"] = proportion(sum(1 for r in unseen if int(r["grade"]) == 2), len(unseen))
    audited = [r for r, i in cells if "excluded_audit" in i.get("strata", []) and r["grade"] != ""]
    out["excluded_audit_relevant(grade>=1)"] = proportion(sum(1 for r in audited if int(r["grade"]) >= 1), len(audited))
    return out


def evaluate_matching(searcher: Searcher, candidates: list[EvalCandidate], rows: list[dict[str, str]],
                      meta: dict[str, Any] | None = None) -> MatchEvalReport:  # fmt: skip
    meta = meta or {}
    grades = search_eval.grade_lookup(rows)
    graded_total, pool_total = search_eval.progress(rows)
    per: list[CandidateEval] = []
    for cand in candidates:
        pool = [r for r in rows if r["query_id"] == cand.id]
        rankable = [r for r in pool if _is_rankable(meta, cand.id, r["job_id"])]
        ce = CandidateEval(cand.id, "scored", sum(1 for r in rankable if r["grade"] != ""), len(rankable))
        if not rankable:
            ce.status = "no pool"
        elif ce.graded < len(rankable):
            ce.status = "pending"
        else:
            pool_grades = [g for r in rankable if (g := grades.get((cand.id, r["job_id"]))) is not None]
            for system, ordered in rankings(searcher, cand).items():
                raw = [grades.get((cand.id, m.job.job_id)) for m in ordered[:K]]
                ranked = [g or 0 for g in raw]
                ce.systems[system] = SystemScores(
                    ndcg=search_eval.ndcg_at_k(ranked, pool_grades, K),
                    precision={t: search_eval.precision_at_k(ranked, 5, t) for t in THRESHOLDS},
                    mrr={t: search_eval.mrr_at_k(ranked, K, t) for t in THRESHOLDS},
                    pooled_recall={t: search_eval.pooled_recall_at_k(ranked, pool_grades, K, t) for t in THRESHOLDS},
                    judged_at_k=sum(g is not None for g in raw) / K,
                )
        per.append(ce)
    scored = [c for c in per if c.status == "scored"]
    if not scored:
        raise EvalError(
            f"no metrics yet: {graded_total}/{pool_total} pool items have a human grade and no candidate has all its ranked "
            "pairs judged. Grade every pair of at least one candidate (`jobintel label matching`).")  # fmt: skip
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    summary: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        res = [c.systems[system] for c in scored]
        summary[system] = {
            "candidates": len(res), f"ndcg@{K}": mean([r.ndcg for r in res]),
            **{f"precision@5(grade>={t})": mean([r.precision[t] for r in res]) for t in THRESHOLDS},
            **{f"mrr@{K}(grade>={t})": mean([r.mrr[t] for r in res]) for t in THRESHOLDS},
            **{f"pooled_recall@{K}(grade>={t})": mean([r.pooled_recall[t] for r in res if r.pooled_recall[t] is not None]) for t in THRESHOLDS},
            f"judged@{K}": mean([r.judged_at_k for r in res]),
        }  # fmt: skip
    return MatchEvalReport(per, summary, graded_total, pool_total, audit_metrics(rows, meta) if meta else {})


def _fmt_prop(p: dict[str, Any]) -> str:
    if not p["n"]:
        return "n/a (no judged items)"
    lo, hi = p["ci95"]
    return f"{p['k']}/{p['n']} = {p['rate']:.0%}  (95% CI {lo:.0%}-{hi:.0%})"


def format_match_report(report: MatchEvalReport) -> str:
    scored = [c for c in report.candidates if c.status == "scored"]
    lines = [f"Human judgments: {report.graded_items}/{report.pool_items} pool items graded; "
             f"{len(scored)}/{len(report.candidates)} candidates fully judged and scored."]  # fmt: skip
    pending = [c for c in report.candidates if c.status != "scored"]
    if pending:
        lines.append("Not scored: " + ", ".join(f"{c.candidate_id} ({c.status}, {c.graded}/{c.pool_size})" for c in pending))
    names = list(report.summary)
    metrics = list(report.summary[names[0]])
    width = max(len(m) for m in metrics)
    lines += ["", f"{'metric':<{width}}  " + "  ".join(f"{n:>14}" for n in names)]
    for metric in metrics:
        cells = []
        for n in names:
            v = report.summary[n][metric]
            cells.append(f"{'n/a':>14}" if v is None else f"{v:>14.3f}" if isinstance(v, float) else f"{v:>14}")
        lines.append(f"{metric:<{width}}  " + "  ".join(cells))
    lines += ["", "Ranking metrics: descriptive only (a handful of synthetic candidates, one annotator). Both systems order the same eligible jobs.",
              "pooled_recall is relative to the judged pool, not the whole corpus; unjudged results count as not relevant."]  # fmt: skip
    if report.audit:
        a = report.audit
        lines += ["", "Audit of the system's decisions (human marks; Wilson 95% intervals):",
                  f"  eligibility: system decision = human 'acceptable' on {_fmt_prop(a['eligibility_agreement'])}",
                  f"  incorrect exclusions (excluded, but the human says acceptable): {_fmt_prop(a['system_excluded_human_acceptable'])}",
                  f"  incorrect inclusions (kept, but the human says unacceptable):   {_fmt_prop(a['system_eligible_human_unacceptable'])}",
                  f"  role relevance: system role match = human on {_fmt_prop(a['role_relevance_agreement'])}",
                  f"  excluded-audit jobs the human still rated relevant (grade>=1): {_fmt_prop(a['excluded_audit_relevant(grade>=1)'])}",
                  f"  eligible jobs outside the top-{POOL_DEPTH} rated relevant (grade>=1): {_fmt_prop(a['unseen_eligible_relevant(grade>=1)'])}  <- what the ranking never surfaces",
                  f"  ... rated strong (grade=2): {_fmt_prop(a['unseen_eligible_strong(grade=2)'])}"]  # fmt: skip
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# skill assertions

SKILL_COLUMNS = ["candidate_id", "job_id", "skill", "claimed", "cv_evidence", "job_evidence", "correct", "notes"]


def build_skill_pool(searcher: Searcher, candidates: list[EvalCandidate], existing: list[dict[str, str]],
                     n_exact: int = 20, n_transferable: int = 12, seed: str = "skills") -> list[dict[str, str]]:  # fmt: skip
    """A seeded, stratified sample of the system's skill claims (exact / transferable) for a human to verify."""
    known = {(r["candidate_id"], r["job_id"], r["skill"]) for r in existing}
    pools: dict[str, list[dict[str, str]]] = {"exact": [], "transferable": []}
    for cand in candidates:
        for m in rankings(searcher, cand)["pipeline"][:POOL_DEPTH]:
            for s in m.skills:
                if s.status in pools and (cand.id, m.job.job_id, s.skill) not in known:
                    cv = s.cv_evidence[0].text if s.cv_evidence else ""
                    pools[s.status].append({"candidate_id": cand.id, "job_id": m.job.job_id, "skill": s.skill, "claimed": s.status,
                                            "cv_evidence": cv + (f"   [related skill claimed: {', '.join(s.via)}]" if s.via else ""),
                                            "job_evidence": s.job_evidence, "correct": "", "notes": ""})  # fmt: skip
    rng = random.Random(seed)
    sample = []
    for status, n in (("exact", n_exact), ("transferable", n_transferable)):
        items = pools[status]
        rng.shuffle(items)
        sample += items[:n]
    rng.shuffle(sample)  # blinded order
    return [*existing, *sample]


def read_skill_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for line, row in enumerate(rows, start=2):
        value = (row.get("correct") or "").strip().lower()
        if value not in {"y", "n", ""}:
            raise EvalError(f"{path} line {line}: 'correct' must be y, n or empty (got {value!r})")
        row["correct"] = value
    return rows


def write_skill_rows(path: Path, rows: list[dict[str, str]]) -> None:
    search_eval.write_judgments(path, rows, SKILL_COLUMNS)


def score_skills(rows: list[dict[str, str]]) -> dict[str, Any]:
    judged = [r for r in rows if r["correct"] in {"y", "n"}]
    if not judged:
        raise EvalError(f"no metrics yet: 0/{len(rows)} skill assertions have a human mark in the 'correct' column.")
    out: dict[str, Any] = {"items": len(rows), "judged": len(judged)}
    for status in ("exact", "transferable"):
        part = [r for r in judged if r["claimed"] == status]
        out[f"{status}_precision"] = proportion(sum(r["correct"] == "y" for r in part), len(part))
    return out


# ---------------------------------------------------------------------------------------------
# explanation grounding

AUDIT_COLUMNS = ["candidate_id", "job_id", "mode", "kind", "claim", "cv_evidence", "job_evidence", "supported", "notes"]


@dataclass
class GroundingReport:
    mode_counts: Counter = field(default_factory=Counter)
    proposed: int = 0
    verified: int = 0
    rejected_reasons: Counter = field(default_factory=Counter)
    explanations: int = 0
    audit_rows: list[dict[str, str]] = field(default_factory=list)
    provenance: rag_audit.Provenance = field(default_factory=rag_audit.Provenance)
    stress: rag_audit.StressResult = field(default_factory=rag_audit.StressResult)
    model: str | None = None

    def summary(self) -> dict[str, Any]:
        s = rag_audit.audit_summary(self.mode_counts, self.provenance, self.stress)
        s.update({"explanations": self.explanations, "items_proposed": self.proposed, "items_verified": self.verified, "model": self.model})
        return s

    def format(self) -> str:
        rate = f"{self.verified}/{self.proposed}" + (f" ({self.verified / self.proposed:.0%})" if self.proposed else "")
        prov = self.provenance
        stress = self.stress
        lines = [
            f"Explanations generated: {self.explanations} (sources: {dict(self.mode_counts)})",
            "Real language-model output evaluated: " + ("YES (Ollama)" if self.mode_counts.get("llm-ollama") else "NO - only deterministic or mock output (no Ollama run)"),
            f"Items proposed: {self.proposed}; passed the automatic verifier: {rate}",
            f"Rejected, by reason: {dict(self.rejected_reasons) or 'none'}",
            f"Evidence provenance: {prov.verbatim}/{prov.cited} cited snippets occur verbatim in the CV / listing text"
            + (f" ({prov.rate:.0%})" if prov.rate is not None else ""),
            f"Verifier stress test (constructed corruptions, not model output): rejected {sum(stress.rejected.values())}/{sum(stress.attempts.values())}"
            + (f" ({stress.rate:.0%})" if stress.rate is not None else ""),
        ]
        weak = [f"{k}: {stress.rejected[k]}/{n}" for k, n in stress.attempts.items() if stress.rejected[k] < n]
        if weak:
            lines.append("  corruption types NOT always rejected -> " + ", ".join(weak))
        lines += ["These are CONSISTENCY checks of citations and named facts, NOT proof that any generated content is true.",
                  f"{len(self.audit_rows)} verified items exported for HUMAN audit."]  # fmt: skip
        return "\n".join(lines)


def grounding_run(searcher: Searcher, candidates: list[EvalCandidate], *, top_n: int = 3,
                  client: OllamaClient | None = None) -> GroundingReport:  # fmt: skip
    report = GroundingReport(model=getattr(client, "model", None))
    for cand in candidates:
        cv_text = from_file(cand.cv_path).text
        result = match_candidate(cand.profile, cand.prefs, searcher)
        for m in result.matches[:top_n]:
            exp = explain_match(cand.profile, m, searcher=searcher, queries=result.queries, client=client)
            kind_tag = rag_audit.client_kind(exp)
            report.explanations += 1
            report.mode_counts[kind_tag] += 1
            report.proposed += exp.claims_proposed
            report.verified += exp.claims_verified
            report.rejected_reasons.update(r.reason.split(",")[0].split("(")[0].strip()[:60] for r in exp.rejected)
            prov = rag_audit.provenance(exp, cv_text, str(m.record.get("description") or ""))
            report.provenance.cited += prov.cited
            report.provenance.verbatim += prov.verbatim
            report.provenance.missing += prov.missing
            report.stress.merge(rag_audit.verifier_stress(exp, cand.profile))
            items = [("strength", s.claim, s.cv_refs, s.job_refs) for s in exp.strengths]
            items += [("gap", f"{g.requirement} {g.note}".strip(), g.cv_refs, g.job_refs) for g in exp.gaps]
            items += [("preparation", p.suggestion, p.cv_refs, p.job_refs) for p in exp.preparation]
            for kind, text, cv_refs, job_refs in items:
                report.audit_rows.append({
                    "candidate_id": cand.id, "job_id": m.job.job_id, "mode": kind_tag, "kind": kind, "claim": text,
                    "cv_evidence": " || ".join(s.text for r in cv_refs if (s := exp.pack.get(r))),
                    "job_evidence": " || ".join(s.text for r in job_refs if (s := exp.pack.get(r))),
                    "supported": "", "notes": "",
                })  # fmt: skip
    return report


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def score_audit(path: Path) -> dict[str, Any]:
    """Summarise ONLY the human marks, separately for deterministic, mock and real-model explanations."""
    if not path.exists():
        raise EvalError(f"audit file not found: {path}. Run `jobintel match-eval grounding` first.")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    marks: dict[str, Counter] = {}
    for line, row in enumerate(rows, start=2):
        mark = (row.get("supported") or "").strip().lower()
        if mark not in {"", "yes", "partial", "no"}:
            raise EvalError(f"{path} line {line}: 'supported' must be empty, yes, partial or no (got {mark!r})")
        marks.setdefault(row.get("mode") or "deterministic", Counter())[mark or "ungraded"] += 1
    total = Counter()
    for c in marks.values():
        total.update(c)
    judged = total["yes"] + total["partial"] + total["no"]
    if judged == 0:
        raise EvalError(f"no metrics yet: 0/{len(rows)} audited items have a human mark in the 'supported' column.")
    by_mode = {}
    for mode, c in marks.items():
        j = c["yes"] + c["partial"] + c["no"]
        by_mode[mode] = {"human_judged": j, "ungraded": c["ungraded"], "yes": c["yes"], "partial": c["partial"], "no": c["no"],
                         "supported_yes": proportion(c["yes"], j)}  # fmt: skip
    return {"items": len(rows), "human_judged": judged, "ungraded": total["ungraded"], "yes": total["yes"], "partial": total["partial"],
            "no": total["no"], "supported_rate_yes_only": total["yes"] / judged, "by_source": by_mode,
            "note": "Top-level counts pool every source for convenience; read `by_source`: a mock or deterministic result says nothing about a real Llama model."}  # fmt: skip


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
