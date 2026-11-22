"""Page 5 - Job details: structured fields, score decomposition, skill evidence, grounded explanation."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from job_intel.ui import components as c
from job_intel.ui.theme import INK_2, SERIES, SURFACE


def _pick_job(service):
    ss = c.state()
    options: dict[str, str] = {}
    if ss.recs is not None:
        for i, (m, sc) in enumerate(ss.recs.ranked, start=1):
            options[m.job.job_id] = f"#{i} · {m.job.title} · {m.job.company or '?'} ({sc.score:.0f})"
    if ss.selected_job_id and ss.selected_job_id not in options and service.get_job(ss.selected_job_id):
        p = service.get_job(ss.selected_job_id)
        options = {p.id: f"{p.title} · {p.company or '?'}", **options}
    if not options:
        st.info("Pick a job from **Job Search** or **Recommendations** first.")
        return None
    ids = list(options)
    index = ids.index(ss.selected_job_id) if ss.selected_job_id in ids else 0
    chosen = st.selectbox("Job", ids, index=index, format_func=options.get)
    ss.selected_job_id = chosen
    return chosen


def _match_for(job_id: str):
    recs = c.state().recs
    if recs is None:
        return None, None
    return next(((m, s) for m, s in recs.ranked if m.job.job_id == job_id), (None, None))


def _waterfall(score) -> go.Figure:
    comps = [x for x in score.components if x.known]
    comps.sort(key=lambda x: -score.contribution(x.name))
    names = [x.name.replace("_", " ") for x in comps] + ["score"]
    values = [score.contribution(x.name) for x in comps]
    fig = go.Figure(go.Waterfall(
        orientation="v", x=names, y=values + [0], measure=["relative"] * len(values) + ["total"],
        text=[f"{v:.1f}" for v in values] + [f"{score.score:.0f}"], textposition="outside", textfont=dict(color=INK_2),
        connector=dict(line=dict(color="#c3c2b7", width=1)),
        increasing=dict(marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2))),
        totals=dict(marker=dict(color="#0d366b", line=dict(color=SURFACE, width=2))),
        hovertemplate="%{x}: +%{y:.1f} points<extra></extra>",
    ))  # fmt: skip
    fig.update_layout(title="How the compatibility score is built (points out of 100)", height=360, showlegend=False,
                      yaxis=dict(range=[0, 110], title="points"))  # fmt: skip
    return fig


def render() -> None:
    st.title("📌 Job Details")
    service = c.get_service()
    job_id = _pick_job(service)
    if job_id is None:
        return
    p = service.get_job(job_id)
    if p is None:
        st.error("This listing is no longer in the corpus.")
        return
    url = c.safe_url(str(p.apply_url))
    st.markdown(f"## {c.e(p.title)}")
    st.markdown(f"<div class='ji-sub'>{c.e(p.company)} · {c.e(c.short_location(p.location, 6))} · source: {c.e(p.source)} · published "
                f"{c.e((p.published_at or '?')[:10])}</div>", unsafe_allow_html=True)  # fmt: skip
    if url:
        st.link_button(f"Apply on {p.source} ↗", url, type="primary")

    fields = {"Contract": p.contract_type or "not stated", "Experience level": p.experience_level,
              "Min. years": p.min_years if p.min_years is not None else "not stated", "Degree": p.degree_level or "not stated",
              "Languages required": ", ".join(p.languages) or "not stated", "Remote": {True: "yes", False: "not stated", None: "?"}[p.remote],
              "Country": c.short_location(", ".join(x.title() for x in p.countries), 4) if p.countries else (p.region or "?").title(),
              "Sector (estimate)": p.sector or "unspecified", "Role family": ", ".join(p.role_family) or "other"}  # fmt: skip
    cols = st.columns(3)
    for i, (k, v) in enumerate(fields.items()):
        cols[i % 3].markdown(f"<div class='ji-muted'>{c.e(k)}</div><div style='margin-bottom:8px'><b>{c.e(v)}</b></div>", unsafe_allow_html=True)

    match, score = _match_for(job_id)
    profile = c.state().profile
    if match is not None and score is not None:
        st.divider()
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(_waterfall(score), width="stretch")
        with right:
            st.markdown(f"<div class='ji-score'>{c.score_ring(score.score)}<div><b>{c.BAND_LABEL[score.band]}</b><br>"
                        f"<span class='ji-muted'>coverage {round(100 * score.coverage)}% · ranking score {score.ranking_score:.0f}</span></div></div>",
                        unsafe_allow_html=True)  # fmt: skip
            st.dataframe([{"criterion": x.name, "weight": x.weight, "value": None if x.value is None else round(x.value, 2),
                           "detail": x.detail} for x in score.components], width="stretch", hide_index=True)  # fmt: skip
        st.markdown("#### Skills: listing requirement ↔ CV evidence")
        st.dataframe([{"skill": s.skill, "status": s.status.replace("_", " "), "asked as": s.job_level,
                       "listing says": s.job_evidence[:160], "your CV says": (s.cv_evidence[0].text[:160] if s.cv_evidence else ""),
                       "via": ", ".join(s.via)} for s in match.skills], width="stretch", hide_index=True)  # fmt: skip
        st.markdown("#### Eligibility")
        for x in match.constraints:
            st.markdown(f"{c.STATUS_ICON[x.status]} **{x.name}** ({'hard' if x.hard else 'soft'}, {x.status}) - {x.detail}")
        for n in match.notes:
            st.caption(n)

        st.markdown("#### Grounded explanation")
        use_llm = st.toggle("Use the local LLM (Ollama) - CV data never leaves this computer", value=False)
        if st.button("Explain this match"):
            from job_intel.rag.explain import explain_match

            with st.spinner("Building the evidence pack and verifying every claim…"):
                client = service.llm_client() if use_llm else None
                exp = explain_match(profile, match, searcher=service.searcher(), client=client)
            source = f"llm ({exp.model}, {exp.client_kind})" if exp.mode == "llm" else "deterministic"
            st.caption(f"mode: {source} · claims kept: {exp.claims_verified} · rejected by the verifier: {len(exp.rejected)}")
            for note in exp.notes:
                st.caption(note)
            for s in exp.strengths:
                st.markdown(f"✅ {c.e(s.claim)} " + "".join(f"<span class='ji-cite'>{c.e(r)}</span>" for r in s.cv_refs + s.job_refs),
                            unsafe_allow_html=True)  # fmt: skip
            for g in exp.gaps:
                st.markdown(f"⚠️ {c.e(g.requirement)} " + "".join(f"<span class='ji-cite'>{c.e(r)}</span>" for r in g.job_refs),
                            unsafe_allow_html=True)  # fmt: skip
            with st.expander("Evidence pack (the only text the explanation may use)"):
                for snip in [*exp.pack.cv, *exp.pack.job]:
                    st.markdown(f"<span class='ji-cite'>{c.e(snip.id)}</span> {c.e(snip.text[:300])}", unsafe_allow_html=True)
            if exp.rejected:
                with st.expander("Claims removed by the verifier"):
                    for r in exp.rejected:
                        st.markdown(f"- ~~{c.e(r.text[:200])}~~ - *{c.e(r.reason)}*")
    else:
        st.info("Load a CV and compute recommendations to see the score decomposition for this job.")
    st.divider()
    st.markdown("#### Full listing")
    st.markdown("<div class='ji-card' style='white-space:pre-wrap;max-height:520px;overflow:auto'>" + c.e(p.description) + "</div>",
                unsafe_allow_html=True)  # fmt: skip
