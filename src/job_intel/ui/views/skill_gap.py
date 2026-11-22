"""Page 6 - Skill-gap analysis for one job + a learning plan across the top recommendations."""

from __future__ import annotations

import streamlit as st

from job_intel.match.gap import interview_prep
from job_intel.ui import components as c
from job_intel.ui.theme import hbar

STATUS_HELP = {
    "demonstrated": ("good", "used in an experience or project"),
    "listed": ("good", "in your skills list"),
    "self_declared": ("good", "you said you have it - add it to your CV"),
    "implied": ("partial", "implied by a skill you have"),
    "transferable": ("partial", "a related skill, not the same one"),
    "not_mentioned": ("missing", "no evidence in your CV - absent OR just not written?"),
    "absent": ("missing", "you confirmed you do not have it"),
}


def _gap_table(report) -> None:
    for title, items in (("✅ Mastered", report.mastered), ("🟡 Partially covered", report.partial),
                         ("⛔ Missing - required", report.missing_required),
                         ("○ Missing - optional", [i for i in report.missing if i.job_level != "required"])):  # fmt: skip
        st.markdown(f"**{title}** ({len(items)})")
        if not items:
            st.caption("none")
        for i in items:
            kind, help_text = STATUS_HELP[i.status]
            cv = f"<div class='ji-muted'>CV: “{c.e(i.cv_evidence[0][:180])}”</div>" if i.cv_evidence else ""
            hint = f"<div class='ji-muted'>💡 {c.e(i.hint)}</div>" if i.hint else ""
            st.markdown(f"<div class='ji-card' style='padding:10px 14px'><span class='ji-chip {kind}'>{c.e(i.skill)}</span>"
                        f"<span class='ji-muted'>{c.e(i.status.replace('_', ' '))} · asked as {c.e(i.job_level)} - {c.e(help_text)}</span>"
                        f"<div class='ji-muted'>Listing: “{c.e(i.job_evidence[:180])}”</div>{cv}{hint}</div>", unsafe_allow_html=True)  # fmt: skip


def render() -> None:
    st.title("🧩 Skill Gap & Learning Plan")
    if not c.require_recs():
        return
    ss = c.state()
    service = c.get_service()
    ranked = ss.recs.ranked
    if not ranked:
        st.warning("No recommendations to analyse.")
        return
    ids = [m.job.job_id for m, _ in ranked]
    labels = {m.job.job_id: f"#{i} · {m.job.title} · {m.job.company or '?'} ({s.score:.0f})" for i, (m, s) in enumerate(ranked, start=1)}
    job_id = st.selectbox("Job", ids, index=ids.index(ss.selected_job_id) if ss.selected_job_id in ids else 0, format_func=labels.get)
    ss.selected_job_id = job_id
    match = next(m for m, _ in ranked if m.job.job_id == job_id)
    report = service.gap(ss.profile, match, ss.declared)

    cols = st.columns(4)
    c.kpi(cols[0], str(len(report.mastered)), "mastered")
    c.kpi(cols[1], str(len(report.partial)), "partially covered")
    c.kpi(cols[2], str(len(report.missing_required)), "required & missing")
    c.kpi(cols[3], str(len([i for i in report.missing if i.job_level != "required"])), "optional & missing")
    c.learn("Absent ≠ not mentioned",
            "A CV is the only evidence the system has, so a skill it does not mention is reported as <b>not mentioned</b>, "
            "not as absent. Tell the system which ones you really have (they become <i>self-declared</i> - add them to your CV!) "
            "and which you lack (they become <i>absent</i> and enter the learning plan with full priority).")  # fmt: skip

    unknown = [i.skill for i in report.items if i.status in {"not_mentioned", "self_declared", "absent"}]
    if unknown:
        with st.form("declare_form"):
            st.markdown("**Do you actually have these skills?**")
            cols = st.columns(3)
            answers = {}
            for k, skill in enumerate(unknown):
                current = ss.declared.get(skill, "unsure")
                answers[skill] = cols[k % 3].radio(skill, ["unsure", "have", "lack"], index=["unsure", "have", "lack"].index(current),
                                                   horizontal=True, key=f"dec_{skill}")  # fmt: skip
            if st.form_submit_button("Update"):
                ss.declared = {**ss.declared, **{k: v for k, v in answers.items() if v != "unsure"}}
                for k, v in answers.items():
                    if v == "unsure":
                        ss.declared.pop(k, None)
                st.rerun()
    _gap_table(report)

    st.divider()
    st.markdown("### 🗺️ Personalised learning plan (across your top 10 recommendations)")
    steps = service.plan(ss.profile, ranked, ss.declared)
    if not steps:
        st.success("No gap found in your top recommendations.")
    else:
        st.plotly_chart(hbar([s.skill for s in steps], [s.priority for s in steps], title="Learning priority (higher = learn sooner)",
                             value_format=".2f", hover="%{y}: priority %{x:.2f}<extra></extra>"), width="stretch")  # fmt: skip
        for s in steps:
            res = " · ".join(f'<a href="{c.e(r["url"])}" target="_blank">{c.e(r["title"])}</a>' for r in s.resources if c.safe_url(r.get("url")))
            pre = f"<div class='ji-muted'>Prerequisites: {c.e(', '.join(s.prerequisites))}</div>" if s.prerequisites else ""
            why = f"<div class='ji-muted'>Why: {c.e('; '.join(s.reasons))}</div>" if s.reasons else ""
            st.markdown(f"<div class='ji-card'><h4><span class='ji-step'>{s.order}</span>{c.e(s.skill)}</h4>{why}{pre}"
                        f"<div class='ji-sub' style='margin-top:6px'>🛠️ <b>Project:</b> {c.e(s.project or '')}</div>"
                        f"<div class='ji-sub'>📚 {res or 'see the official documentation'}</div></div>", unsafe_allow_html=True)  # fmt: skip
        c.learn("How the plan is ordered",
                "priority(s) = Σ<sub>jobs</sub> w<sub>level</sub> · fit(job) × (1 + market demand of s), where w = 1 required / 0.5 "
                "preferred / 0.3 mentioned (halved for partially covered skills) and fit = compatibility/100. Then a "
                "<b>topological sort</b> on the prerequisite graph (Kahn's algorithm) puts foundations first - e.g. Docker before Kubernetes.")  # fmt: skip

    st.divider()
    st.markdown("### 🎤 Interview preparation for this job")
    qs = interview_prep(report)
    if not qs:
        st.caption("No question bank for these skills yet.")
    for skill, q in qs:
        st.markdown(f"- **{c.e(skill)}** - {c.e(q)}")
    st.caption("Questions on skills you claim come first: interviewers test what your CV says.")
