"""Streamlit entry point: `jobintel ui` (or `streamlit run src/job_intel/ui/app.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run path/to/app.py` puts the script's folder on sys.path, not the package root
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from job_intel.ui import components as c  # noqa: E402
from job_intel.ui.views import (  # noqa: E402
    assistant,
    evaluation,
    job_details,
    market,
    overview,
    profile,
    recommendations,
    search,
    skill_gap,
)  # fmt: skip

st.set_page_config(page_title="Job & Internship Intelligence", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")  # fmt: skip
c.inject_css()
c.state()

pages = {
    "overview": st.Page(overview.render, title="Overview", icon="🧭", url_path="overview", default=True),
    "profile": st.Page(profile.render, title="CV & Profile", icon="📄", url_path="profile"),
    "search": st.Page(search.render, title="Job Search", icon="🔎", url_path="search"),
    "recommendations": st.Page(recommendations.render, title="Recommendations", icon="🎯", url_path="recommendations"),
    "details": st.Page(job_details.render, title="Job Details", icon="📌", url_path="details"),
    "gap": st.Page(skill_gap.render, title="Skill Gap & Plan", icon="🧩", url_path="skill-gap"),
    "market": st.Page(market.render, title="Market Intelligence", icon="📊", url_path="market"),
    "assistant": st.Page(assistant.render, title="Career Assistant", icon="💬", url_path="assistant"),
    "evaluation": st.Page(evaluation.render, title="Evaluation & Method", icon="🧪", url_path="evaluation"),
}
c.PAGES.update(pages)
nav = st.navigation({
    "Start": [pages["overview"], pages["profile"]],
    "Find jobs": [pages["search"], pages["recommendations"], pages["details"]],
    "Grow": [pages["gap"], pages["market"], pages["assistant"]],
    "Science": [pages["evaluation"]],
})  # fmt: skip

with st.sidebar:
    ss = c.state()
    if ss.profile is not None:
        meta = ss.cv_meta or {}
        st.success(f"CV loaded: {meta.get('name', 'profile')}", icon="📄")
        if ss.recs is not None:
            st.caption(f"{len(ss.recs.ranked)} recommendations ready")
    else:
        st.caption("No CV loaded")
    st.divider()
    st.caption("🔒 Your CV stays on this machine: it is processed in memory and only ever sent to a *local* "
               "Ollama model if you enable it.")  # fmt: skip
    st.caption("Listings: Arbeitnow, Himalayas, Remotive. Apply links point to the original listings.")
nav.run()
