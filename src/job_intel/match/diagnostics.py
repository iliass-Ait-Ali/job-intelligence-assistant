"""Label-free diagnostics for the matcher: WHY are jobs excluded, and does retrieval still reach the jobs that stay?

These numbers are properties of the system's own rules and of the data (no human judgment involved), so
they are deterministic and usable as regression baselines. They say nothing about whether a recommended
job is actually a good one: that needs human grades (see `evaluate.py`).

Questions answered per candidate:
* How many jobs of the whole index survive the hard constraints (`eligible_in_corpus`)?
* How many of them reach the candidate pool (`eligible_recall_in_pool`)? The Phase 3 behaviour retrieved
  first and filtered afterwards, so a pool of 80 jobs could contain only a few eligible ones.
* Which constraints cause the exclusions, and how many kept jobs are kept only because a value is unknown?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from job_intel.match.matcher import JobMatch, match_candidate
from job_intel.match.models import CandidateProfile, Preferences
from job_intel.search.retrieve import Searcher


@dataclass
class PoolStats:
    pool: int = 0
    eligible_in_pool: int = 0
    eligible_in_corpus: int = 0

    @property
    def eligible_recall(self) -> float | None:
        """Share of the eligible jobs of the whole index that made it into the candidate pool."""
        return self.eligible_in_pool / self.eligible_in_corpus if self.eligible_in_corpus else None


@dataclass
class CandidateDiagnostics:
    candidate_id: str
    corpus: int
    eligible_in_corpus: int
    excluded_total: int
    excluded_by: dict[str, int]  # combination of violated hard constraints -> jobs
    kept_with_unknown_hard: dict[str, int]  # hard constraint -> kept jobs whose value is unknown
    legacy: PoolStats  # retrieve first, filter afterwards (Phase 3 behaviour)
    current: PoolStats  # filter first, then retrieve
    top_excluded: list[str] = field(default_factory=list)  # most retrieval-relevant excluded jobs, with the reason


def _hard_unknown(m: JobMatch) -> list[str]:
    return [c.name for c in m.constraints if c.hard and c.status == "unknown"]


def diagnose(candidate_id: str, profile: CandidateProfile, prefs: Preferences, searcher: Searcher) -> CandidateDiagnostics:
    big = prefs.model_copy(update={"max_results": 1000})
    current = match_candidate(profile, big, searcher, prefilter=True)
    legacy = match_candidate(profile, big, searcher, prefilter=False)
    eligible_ids = {jid for jid in searcher.table} - {m.job.job_id for m in current.excluded}
    excluded_by = Counter("+".join(sorted({r.split(":")[0] for r in m.excluded_because})) for m in current.excluded)
    unknown = Counter()
    for m in current.matches:
        unknown.update(_hard_unknown(m))
    ranked_excluded = sorted(legacy.excluded, key=lambda m: -m.retrieval.rrf)[:5]
    return CandidateDiagnostics(
        candidate_id=candidate_id, corpus=len(searcher.table), eligible_in_corpus=len(eligible_ids),
        excluded_total=len(current.excluded), excluded_by=dict(excluded_by), kept_with_unknown_hard=dict(unknown),
        # in the legacy pool, every non-excluded job is by definition an eligible one
        legacy=PoolStats(legacy.pool_size, len(legacy.matches), len(eligible_ids)),
        current=PoolStats(current.pool_size, len(current.matches), len(eligible_ids)),
        top_excluded=[f"{m.job.title[:50]} | {m.excluded_because[0][:90]}" for m in ranked_excluded],
    )  # fmt: skip


def run_diagnostics(candidates: list[Any], searcher: Searcher) -> list[CandidateDiagnostics]:
    return [diagnose(c.id, c.profile, c.prefs, searcher) for c in candidates]


def metrics(diags: list[CandidateDiagnostics]) -> dict[str, float]:
    """Flat label-free metrics for regression baselines (higher is better unless the name says otherwise)."""
    flat: dict[str, float] = {}
    for d in diags:
        flat[f"{d.candidate_id}.eligible_in_corpus"] = float(d.eligible_in_corpus)
        flat[f"{d.candidate_id}.eligible_recall_in_pool"] = d.current.eligible_recall if d.current.eligible_recall is not None else 1.0
    return flat


def format_diagnostics(diags: list[CandidateDiagnostics]) -> str:
    lines = []
    for d in diags:
        lines.append(f"\n== {d.candidate_id}: {d.corpus} jobs in the index")
        lines.append(f"   survive the hard constraints: {d.eligible_in_corpus}   excluded: {d.excluded_total}")
        for reason, n in sorted(d.excluded_by.items(), key=lambda kv: -kv[1]):
            lines.append(f"     excluded by {reason or '(other)'}: {n}")
        if d.kept_with_unknown_hard:
            lines.append("     kept only because the value is unknown: " + ", ".join(f"{k}={v}" for k, v in d.kept_with_unknown_hard.items()))
        for name, st in (("retrieve-then-filter (Phase 3)", d.legacy), ("filter-then-retrieve (now)", d.current)):
            rec = f"{st.eligible_recall:.0%}" if st.eligible_recall is not None else "n/a"
            lines.append(f"   {name:<32} pool={st.pool:<3} eligible jobs in pool={st.eligible_in_pool}/{st.eligible_in_corpus}  ({rec})")
        for item in d.top_excluded[:3]:
            lines.append(f"     most retrieval-relevant excluded job: {item}")
    return "\n".join(lines)
