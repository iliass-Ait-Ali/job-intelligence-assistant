"""Page 1 - Overview: what the system does, the data behind it, and how the pieces fit."""

from __future__ import annotations

import streamlit as st

from job_intel.ui import components as c
from job_intel.ui.theme import hbar

PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Inter" fontsize=11 color="#c7d2fe" fillcolor="#eef2ff"];
  edge [color="#94a3b8" arrowsize=0.7];
  subgraph cluster_data { label="1  Data"; fontname="Inter"; color="#e2e8f0";
    APIs [label="Job APIs\\nArbeitnow · Himalayas · Remotive"]; ETL [label="clean · normalise\\nfilter · deduplicate"];
    ENR [label="enrich\\nskills · level · sector · country"]; DB [label="SQLite +\\nJSONL" shape=cylinder fillcolor="#f1f5f9"]; }
  subgraph cluster_ir { label="2  Retrieval"; fontname="Inter"; color="#e2e8f0";
    CH [label="chunk + embed\\nmultilingual-e5"]; FA [label="FAISS\\n(exact cosine)"]; BM [label="BM25 /\\nkeyword"];
    RRF [label="Hybrid\\nRRF fusion"]; CE [label="Cross-encoder\\nre-rank"]; }
  subgraph cluster_match { label="3  Matching"; fontname="Inter"; color="#e2e8f0";
    CV [label="CV PDF/DOCX\\n-> profile"]; HC [label="hard constraints\\n(eliminatory)"]; SC [label="compatibility\\nscore 0-100"];
    GAP [label="skill gap +\\nlearning plan"]; }
  subgraph cluster_rag { label="4  Assistant"; fontname="Inter"; color="#e2e8f0";
    ROUTE [label="intent router"]; CTX [label="cited documents\\nJ · CV · M · K · S"]; GEN [label="template or\\nlocal LLM"]; VER [label="verifier"]; }
  APIs -> ETL -> ENR -> DB; DB -> CH -> FA; DB -> BM; FA -> RRF; BM -> RRF; RRF -> CE;
  CV -> HC -> RRF; CE -> SC; RRF -> SC; SC -> GAP; SC -> CTX; ROUTE -> CTX -> GEN -> VER;
}
"""


def render() -> None:
    service = c.get_service()
    status = service.status()
    c.hero("Job & Internship Intelligence Assistant",
           "A search, matching and recommendation engine for data & AI careers - with explanations you can audit. "
           "Upload a CV, see which real listings fit, why, what is missing, and how to close the gap.")  # fmt: skip

    cols = st.columns(5)
    c.kpi(cols[0], f"{status['postings']:,}", "listings indexed")
    c.kpi(cols[1], f"{status['companies']:,}", "companies")
    c.kpi(cols[2], f"{status['countries']}", "countries named")
    c.kpi(cols[3], f"{len(status['by_source'])}", "authorised sources")
    span = f"{status['published_from'][5:]} → {status['published_to'][5:]}" if status["published_from"] else "?"
    c.kpi(cols[4], span, f"publication dates ({(status['published_to'] or '')[:4]})")

    st.markdown("### How to use it")
    steps = [
        ("📄", "CV & Profile", "Upload a PDF / DOCX CV (or a synthetic example). The parser extracts education, experience, projects, skills, languages - each with the CV line that proves it."),
        ("🎯", "Recommendations", "Hard constraints filter, hybrid retrieval finds candidates, an interpretable score ranks them."),
        ("📌", "Job Details", "See the score decomposition and every matched / missing skill next to its evidence."),
        ("🧩", "Skill Gap & Plan", "Mastered, partial and missing skills; a prerequisite-ordered learning plan with resources and projects."),
        ("📊", "Market Intelligence", "Which skills, companies, countries and skill combinations dominate the corpus."),
        ("💬", "Career Assistant", "Ask questions; every sentence of the answer cites the listing, CV line or statistic it comes from."),
    ]
    grid = st.columns(3)
    for i, (icon, title, text) in enumerate(steps):
        grid[i % 3].markdown(f'<div class="ji-card"><h4>{icon} {c.e(title)}</h4><div class="ji-sub">{c.e(text)}</div></div>',
                             unsafe_allow_html=True)  # fmt: skip

    st.markdown("### Architecture")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    c.learn("Why not just send the CV and the job to an LLM?",
            "Because nothing would be measurable or auditable. Here the LLM is <b>one optional component</b> at the very end. "
            "Retrieval (BM25, dense embeddings, fusion, re-ranking), eligibility rules, scoring and gap analysis are explicit "
            "algorithms that are evaluated separately (see <i>Evaluation & Method</i>). The LLM only rephrases documents it is "
            "given, and a verifier deletes any sentence whose facts are not in the cited sources.")  # fmt: skip

    left, right = st.columns([1, 1])
    with left:
        df = service.market_df
        counts = df["source"].value_counts()
        st.plotly_chart(hbar(counts.index, counts.values, title="Listings by source", value_format=",d",
                             hover="%{y}: %{x:,d} listings<extra></extra>", height=220), width="stretch")  # fmt: skip
    with right:
        fam = df["role_families"].explode().value_counts().head(7)
        st.plotly_chart(hbar(fam.index, fam.values, title="Listings by role family", value_format=",d",
                             hover="%{y}: %{x:,d} listings<extra></extra>", height=260), width="stretch")  # fmt: skip

    with st.expander("Data provenance, licences and limits"):
        st.markdown(
            "- **Sources:** public APIs of [Arbeitnow](https://www.arbeitnow.com), [Himalayas](https://himalayas.app) and "
            "[Remotive](https://remotive.com). Every listing keeps its original link; apply there. Listings are not "
            "redistributed (the data folder is git-ignored and re-created with `jobintel collect`).\n"
            "- **Scope:** data, ML and AI roles plus data-flavoured internships (strict title rules). The corpus is a *sample* "
            "of three boards at a few dates - not the labour market. Himalayas is remote-only and US-heavy.\n"
            "- **Unknown stays unknown:** a listing that does not state a contract type, a degree or years of experience "
            "is never assumed to require or not require them.\n"
            "- **Scores are compatibility indices, not hiring probabilities.**"
        )
