"""Page 3 - Job search with five retrieval engines, filters, and an engine comparison view."""

from __future__ import annotations

import streamlit as st

from job_intel.search.filters import JobFilter
from job_intel.ui import components as c

ENGINES = {
    "hybrid": "Hybrid (BM25 + dense, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder re-rank",
    "semantic": "Dense semantic (e5 + FAISS)",
    "bm25": "BM25 lexical",
    "keyword": "Keyword overlap (baseline)",
}
SCORE_LABEL = {"rrf": "RRF", "cosine_similarity": "cosine", "bm25": "BM25", "keyword_overlap": "terms matched",
               "cross_encoder_logit": "CE logit"}  # fmt: skip


def _hit_card(hit, service) -> None:
    p = service.get_job(hit.job.job_id)
    url = c.safe_url(str(p.apply_url) if p else hit.job.url)
    skills = [s.name for s in p.skills[:8]] if p else []
    meta = " · ".join(x for x in (hit.job.company, c.short_location(hit.job.location), (p.contract_type if p else None), (p.experience_level if p else None)) if x)
    link = f'<div style="margin-top:8px"><a class="ji-apply" href="{c.e(url)}" target="_blank">Open listing ↗</a></div>' if url else ""
    st.markdown(
        f'<div class="ji-card"><div style="display:flex;justify-content:space-between;gap:12px"><div><h4>{hit.rank}. {c.e(hit.job.title)}</h4>'
        f'<div class="ji-sub">{c.e(meta)}</div></div><div class="ji-muted" style="text-align:right">{SCORE_LABEL.get(hit.score_kind, hit.score_kind)}'
        f'<br><b style="color:#0b0b0b">{hit.score:.3f}</b></div></div><div class="ji-muted" style="margin:6px 0">“{c.e(hit.snippet[:260])}”</div>'
        f'{c.chips(skills, "info")}{link}</div>',
        unsafe_allow_html=True)  # fmt: skip
    if st.button("Details", key=f"d_{hit.rank}_{hit.job.job_id}"):
        c.state().selected_job_id = hit.job.job_id
        st.switch_page(c.PAGES["details"])


def render() -> None:
    st.title("🔎 Job Search")
    st.caption("Natural-language search in French, English or German. The same filters apply to every engine before ranking.")
    service = c.get_service()
    q = st.text_input("What are you looking for?", value=st.session_state.get("last_query", "stage data analyst à Paris"))
    a, b, d = st.columns([2, 2, 1])
    engine = a.selectbox("Engine", list(ENGINES), format_func=ENGINES.get)
    k = d.number_input("Results", 3, 30, 10)
    with b.popover("Filters", width="stretch"):
        job_types = st.multiselect("Contract", ["internship", "full_time", "working_student", "contract", "part_time", "permanent"])
        locations = st.text_input("Locations (comma-separated)", "")
        remote = st.checkbox("Remote only")
        include_unknown = st.checkbox("Keep listings with unknown contract/location", value=True)
        per_company = st.checkbox("At most 2 listings per company", value=True)
    compare = st.toggle("Compare all five engines side by side (pedagogical)")
    flt = JobFilter(job_types=tuple(job_types), locations=tuple(x.strip() for x in locations.split(",") if x.strip()),
                    remote=True if remote else None, include_unknown=include_unknown)  # fmt: skip
    if not q.strip():
        return
    st.session_state["last_query"] = q
    if compare:
        cols = st.columns(5)
        for col, (name, label) in zip(cols, ENGINES.items()):
            with col:
                with st.spinner(label):
                    resp = service.search(q, engine=name, k=5, flt=flt, max_per_company=2 if per_company else None)
                st.markdown(f"**{label}**")
                for h in resp.hits:
                    st.markdown(f'<div class="ji-card" style="padding:10px 12px"><div class="ji-sub"><b>{h.rank}.</b> {c.e(h.job.title[:70])}</div>'
                                f'<div class="ji-muted">{c.e(h.job.company or "")}</div></div>', unsafe_allow_html=True)  # fmt: skip
                st.caption(resp.notes[-1] if resp.notes else "")
    else:
        with st.spinner("Searching…"):
            resp = service.search(q, engine=engine, k=int(k), flt=flt, max_per_company=2 if per_company else None)
        st.caption(f"{resp.eligible_jobs} eligible of {resp.total_jobs} listings · " + " · ".join(resp.notes))
        for hit in resp.hits:
            _hit_card(hit, service)
    with st.expander("🎓 How the five engines work (and why the scores are not comparable)", expanded=False):
        st.markdown(r"""
**Keyword overlap** (baseline) - counts the distinct query words present in the listing. No weighting.

**BM25** - for each query term $t$: $\mathrm{IDF}(t)\cdot\dfrac{tf\,(k_1+1)}{tf + k_1\,(1-b+b\,\frac{|d|}{\mathrm{avgdl}})}$ with $k_1=1.5$, $b=0.75$.
Rare terms weigh more (IDF), repeated terms saturate ($k_1$), long listings are normalised ($b$).

**Dense semantic** - query and listing chunks are embedded by `multilingual-e5-base` (768-d, L2-normalised); score = cosine
$=\mathbf{q}\cdot\mathbf{d}$, searched exactly with FAISS `IndexFlatIP`. A listing's score is its best chunk. Handles synonyms and
other languages ("stage" ≈ "internship"), but can miss exact technical terms.

**Hybrid (RRF)** - Reciprocal Rank Fusion: $\mathrm{RRF}(d)=\sum_{s\in\{\text{BM25},\text{dense}\}} \frac{1}{60+\mathrm{rank}_s(d)}$.
It fuses *ranks*, so the incomparable score scales never need calibrating.

**+ Cross-encoder** - the top 30 are re-scored by `mmarco-mMiniLMv2`, which reads query and passage *together* (full attention
between them): more precise, ~100× costlier per pair, hence only on the shortlist.

The measured comparison of the five engines (nDCG@10, MRR, P@5…) is on the **Evaluation & Method** page.
""")
