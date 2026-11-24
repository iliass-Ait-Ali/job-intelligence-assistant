"""Page 7 - Job-market intelligence over the structured corpus, segmentable."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from job_intel.market import analytics as ma
from job_intel.ui import components as c
from job_intel.ui.theme import INK_2, SEQ, SERIES, SURFACE, hbar


@st.cache_data(show_spinner=False)
def _snapshot_df():
    return c.get_service().eval_snapshot_df()


def render() -> None:
    st.title("📊 Job Market Intelligence")
    service = c.get_service()
    df = service.market_df
    cols = st.columns(5)
    sel = {}
    for col, (name, label) in zip(cols, (("role_family", "Role family"), ("country", "Country"), ("sector", "Sector"),
                                         ("experience_level", "Level"), ("contract_type", "Contract"))):  # fmt: skip
        values = sorted(df["role_families"].explode().dropna().unique()) if name == "role_family" else \
            list(df[name].value_counts().index)
        sel[name] = col.multiselect(label, values, key=f"m_{name}")
    seg = ma.filter_frame(df, **sel)
    st.caption(f"{len(seg)} of {len(df)} listings in this segment. Descriptive statistics of this dataset - three job boards, "
               "data/AI roles only - not of the whole labour market.")  # fmt: skip
    if seg.empty:
        st.warning("No listing in this segment.")
        return
    k = st.columns(4)
    c.kpi(k[0], f"{len(seg):,}", "listings")
    c.kpi(k[1], f"{seg['company'].nunique():,}", "companies")
    c.kpi(k[2], f"{(seg['contract_type'] == 'internship').mean():.0%}", "internships")
    c.kpi(k[3], f"{seg['remote'].fillna(False).mean():.0%}", "explicitly remote")

    tab_skills, tab_geo, tab_assoc, tab_roles, tab_time = st.tabs(
        ["Skills", "Countries · sectors · companies", "Skill associations", "Role profiles", "Over time"])
    with tab_skills:
        required = st.toggle("Only skills the listing marks as required", value=False)
        demand = ma.skill_demand(seg, column="required_skills" if required else "skills", top=25)
        st.plotly_chart(hbar(demand["skill"], demand["share"], title="Share of listings asking for each skill",
                             hover="%{y}: %{x:.1%} of listings<extra></extra>"), width="stretch")  # fmt: skip
        soft = ma.skill_demand(seg, column="soft_skills", top=10)
        st.plotly_chart(hbar(soft["skill"], soft["share"], title="Soft skills asked for", color=SERIES[6],
                             hover="%{y}: %{x:.1%} of listings<extra></extra>"), width="stretch")  # fmt: skip
        with st.expander("Table view"):
            st.dataframe(demand, width="stretch", hide_index=True)
    with tab_geo:
        a, b = st.columns(2)
        countries = ma.counts_by(seg, "country", top=15)
        a.plotly_chart(hbar(countries["country"], countries["postings"], title="Listings by country (first named)", value_format=",d",
                            hover="%{y}: %{x:,d}<extra></extra>"), width="stretch")  # fmt: skip
        sectors = ma.counts_by(seg[seg["sector"] != "Unspecified"], "sector", top=12)
        b.plotly_chart(hbar(sectors["sector"], sectors["postings"], title="Listings by sector (keyword estimate)", value_format=",d",
                            hover="%{y}: %{x:,d}<extra></extra>"), width="stretch")  # fmt: skip
        comp = ma.counts_by(seg, "company", top=15)
        st.plotly_chart(hbar(comp["company"], comp["postings"], title="Companies with the most listings in the dataset",
                             value_format=",d", hover="%{y}: %{x:,d}<extra></extra>"), width="stretch")  # fmt: skip
        skill = st.selectbox("Who asks for…", list(ma.skill_demand(seg, top=40)["skill"]))
        top = ma.top_companies_for_skill(seg, skill, 10)
        st.dataframe(top, width="stretch", hide_index=True)
    with tab_assoc:
        m = ma.cooccurrence_matrix(seg, top_skills=14)
        fig = go.Figure(go.Heatmap(z=m.values, x=list(m.columns), y=list(m.index), colorscale=[[i / (len(SEQ) - 1), s] for i, s in enumerate(SEQ)],
                                   zmin=0, zmax=1, xgap=2, ygap=2, colorbar=dict(title="P(col | row)", tickformat=".0%"),
                                   hovertemplate="%{y} → %{x}: %{z:.0%} of listings asking for %{y} also ask for %{x}<extra></extra>"))  # fmt: skip
        fig.update_layout(title="Co-occurrence: P(column skill | row skill)", height=560, plot_bgcolor=SURFACE,
                          xaxis=dict(tickangle=-40, showgrid=False), yaxis=dict(autorange="reversed", showgrid=False))  # fmt: skip
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        pairs = ma.cooccurrence(seg, top_skills=25, min_count=max(3, len(seg) // 60))
        st.markdown("**Strongest associations** (lift > 1: asked together more often than chance)")
        st.dataframe(pairs.head(20).style.format({"p_b_given_a": "{:.0%}", "lift": "{:.2f}"}), width="stretch", hide_index=True)
        c.learn("Lift", "lift(A,B) = P(A,B) / (P(A)·P(B)). 1 means independent. Cloud platforms (AWS/Azure/GCP) co-occur "
                "strongly because listings often name all three as alternatives - lift measures association, not causation. "
                "Pairs need a minimum support because lift explodes on rare skills.")  # fmt: skip
    with tab_roles:
        prof = ma.role_profiles(seg)
        if prof.empty:
            st.caption("Not enough listings per role family in this segment.")
        for family, g in prof.groupby("role_family"):
            st.markdown(f"**{c.e(family)}** <span class='ji-muted'>({int(g['postings'].iloc[0])} listings)</span> "
                        + "".join(f"<span class='ji-chip info' title='{r.share_in_family:.0%} of the family, {r.distinctiveness:.1f}× the average'>"
                                  f"{c.e(r.skill)} · {r.distinctiveness:.1f}×</span>" for r in g.itertuples()), unsafe_allow_html=True)  # fmt: skip
        c.learn("Distinctiveness", "P(skill | family) / P(skill): how much more often a family asks for a skill than the average "
                "listing. Python is everywhere (≈1×); dbt is distinctive of data engineers.")  # fmt: skip
    with tab_time:
        options = list(ma.skill_demand(seg, top=30)["skill"])
        default = [s for s in ("LLM", "Python") if s in options] or options[:2]
        tracked = st.multiselect("Skills to track", options, default=default)
        trend = ma.weekly_trend(seg, tracked)
        fig = go.Figure()
        fig.add_bar(x=trend["week"], y=trend["postings"], marker=dict(color=SERIES[0], cornerradius=4), name="listings",
                    hovertemplate="week of %{x|%d %b}: %{y} listings<extra></extra>")  # fmt: skip
        fig.update_layout(title="Listings by publication week", height=280, showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        fig2 = go.Figure()
        for i, s in enumerate(tracked[:8]):
            fig2.add_scatter(x=trend["week"], y=trend[s], mode="lines+markers", name=s, line=dict(width=2, color=SERIES[i]),
                             marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                             hovertemplate=f"{s}: %{{y:.0%}} of the week's listings<extra></extra>")  # fmt: skip
        fig2.update_layout(title="Share of the week's listings asking for each skill", height=320, yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
        st.caption("Caution: most listings were collected on 2026-11-22 and older weeks contain few listings (the boards drop "
                   "expired jobs), so early weeks are noisy.")  # fmt: skip
        old = _snapshot_df()
        if old is not None:
            cmp = ma.snapshot_comparison(old, df)
            fig3 = go.Figure()
            fig3.add_bar(y=cmp["skill"], x=cmp["share_old"], orientation="h", name="2026-10-23 snapshot (191 listings, Arbeitnow)",
                         marker=dict(color=SERIES[1], cornerradius=4))  # fmt: skip
            fig3.add_bar(y=cmp["skill"], x=cmp["share_new"], orientation="h", name="2026-11-22 snapshot (720 listings, 3 sources)",
                         marker=dict(color=SERIES[0], cornerradius=4))  # fmt: skip
            fig3.update_layout(title="Skill demand in two collection snapshots", barmode="group", height=520,
                               xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed"), font=dict(color=INK_2))  # fmt: skip
            st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
            st.caption("The two snapshots differ in sources (the second adds Himalayas), so differences mix time and source effects.")
