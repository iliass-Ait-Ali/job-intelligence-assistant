"""Page 4 - Personalised recommendations: eligibility first, then an interpretable score."""

from __future__ import annotations

import streamlit as st

from job_intel.match.scoring import DEFAULT_WEIGHTS
from job_intel.ui import components as c


def _card(rank: int, match, score, service) -> None:
    p = service.get_job(match.job.job_id)
    url = c.safe_url(str(p.apply_url) if p else match.job.url)
    exact = [s.skill for s in match.skills if s.status == "exact"]
    partial = [s.skill for s in match.skills if s.status == "transferable"]
    missing_req = [s.skill for s in match.skills if s.status == "not_evidenced" and s.job_level == "required"]
    missing_opt = [s.skill for s in match.skills if s.status == "not_evidenced" and s.job_level != "required"]
    cons = "".join(f'<span class="ji-chip">{c.STATUS_ICON[x.status]} {c.e(x.name)}</span>' for x in match.constraints)
    level = p.experience_level if p and p.experience_level not in {"unspecified", p.contract_type} else None
    meta = " · ".join(x for x in (match.job.company, c.short_location(match.job.location), p.contract_type if p else None, level) if x)
    link = f'<a class="ji-apply" href="{c.e(url)}" target="_blank">Apply on {c.e(match.job.source)} ↗</a>' if url else ""
    st.markdown(f"""
<div class="ji-card"><div class="ji-score">{c.score_ring(score.score)}
<div style="flex:1"><h4>{rank}. {c.e(match.job.title)}</h4><div class="ji-sub">{c.e(meta)}</div>
<div class="ji-muted">{c.BAND_LABEL[score.band]} · {round(100 * score.coverage)}% of the criteria observable · evidence level: {c.e(match.evidence_level)}</div></div>
<div>{link}</div></div>
<div style="margin-top:8px">{c.chips(exact, "good")}{c.chips(partial, "partial")}{c.chips(missing_req, "missing")}{c.chips(missing_opt)}</div>
<div class="ji-muted" style="margin-top:4px">Eligibility: {cons or "no stated constraint"}</div></div>""", unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 4])
    if a.button("Details", key=f"rd_{match.job.job_id}"):
        c.state().selected_job_id = match.job.job_id
        st.switch_page(c.PAGES["details"])
    if b.button("Skill gap", key=f"rg_{match.job.job_id}"):
        c.state().selected_job_id = match.job.job_id
        st.switch_page(c.PAGES["gap"])


def render() -> None:
    st.title("🎯 Personalised Recommendations")
    if not c.require_profile():
        return
    ss = c.state()
    service = c.get_service()
    with st.expander("⚖️ Scoring weights (configurable) and ordering", expanded=False):
        st.caption("Defaults come from config/scoring.yaml. Weights are renormalised over the criteria that are observable "
                   "for each job; unknown criteria are excluded, not penalised.")  # fmt: skip
        base = service.weights()
        cols = st.columns(3)
        weights = {name: cols[i % 3].slider(name.replace("_", " "), 0.0, 0.5, float(base.get(name, v)), 0.01, key=f"w_{name}")
                   for i, (name, v) in enumerate(DEFAULT_WEIGHTS.items())}  # fmt: skip
        sort = st.radio("Order by", ["score", "evidence"], horizontal=True,
                        format_func={"score": "Compatibility (ranking score)", "evidence": "Rule-based evidence level"}.get)  # fmt: skip
    run = st.button("Compute / refresh my matches", type="primary")
    if run or ss.recs is None:
        with st.spinner("Filtering on hard constraints, retrieving candidates, scoring…"):
            ss.recs = service.recommend(ss.profile, ss.prefs, weights=weights, sort=sort)
    recs = ss.recs
    res = recs.result
    cols = st.columns(4)
    c.kpi(cols[0], f"{res.corpus_size}", "listings in corpus")
    c.kpi(cols[1], f"{res.eligible_total}", "pass your hard constraints")
    c.kpi(cols[2], f"{res.pool_size}", "retrieved & assessed")
    c.kpi(cols[3], f"{recs.seconds:.1f} s", "computation time")
    for w in res.warnings:
        st.caption(w)
    st.markdown("<span class='ji-chip good'>evidenced in CV</span><span class='ji-chip partial'>related skill only</span>"
                "<span class='ji-chip missing'>required, not evidenced</span><span class='ji-chip'>optional, not evidenced</span>",
                unsafe_allow_html=True)  # fmt: skip
    if not recs.ranked:
        st.warning("No eligible listing. Relax a hard constraint in **CV & Profile → Preferences**.")
    for i, (match, score) in enumerate(recs.ranked, start=1):
        _card(i, match, score, service)
    with st.expander(f"Excluded by your hard constraints ({len(res.excluded)}) - never silently dropped"):
        st.dataframe([{"job": m.job.title, "company": m.job.company, "reason": "; ".join(m.excluded_because)} for m in res.excluded[:300]],
                     width="stretch", hide_index=True)  # fmt: skip
    c.learn("Eliminatory vs ranking criteria",
            "1) <b>Hard constraints</b> you chose (contract, location, …) are evaluated as <i>satisfied / violated / unknown</i>; only a "
            "clear <b>violation</b> excludes a job, and it is listed with its reason. 2) The remaining jobs are retrieved with hybrid "
            "search using several queries built from your CV. 3) Each job gets a <b>compatibility index</b> "
            "= Σ wᵢcᵢ / Σ wᵢ over the criteria that are observable; jobs are ordered by the <i>ranking score</i> = coverage·score + "
            "(1−coverage)·50, so a high score built on little evidence does not jump the queue. It is <b>not</b> a probability of being hired.")  # fmt: skip
