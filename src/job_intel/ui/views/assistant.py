"""Page 8 - AI career assistant: chat with cited, verified answers."""

from __future__ import annotations

import re

import streamlit as st

from job_intel.ui import components as c

SUGGESTIONS = [
    "Quelles sont les dix offres les plus pertinentes pour mon profil ?",
    "Why does job 1 match my CV?",
    "Quelles compétences me manquent pour l'offre 1 ?",
    "Which companies are looking for Machine Learning skills?",
    "Compare these three opportunities",
    "Comment préparer mon entretien pour l'offre 1 ?",
    "Quels projets dois-je réaliser pour améliorer mon profil ?",
    "What are the most demanded skills?",
]


def _render_answer(answer) -> None:
    for s in answer.sentences:
        cites = "".join(f"<span class='ji-cite'>{c.e(x)}</span>" for x in s.citations)
        st.markdown(f"{c.e(s.text)} {cites}", unsafe_allow_html=True)
    meta = f"intent: {answer.intent} · mode: {answer.mode} · {answer.latency_s:.2f} s · {len(answer.rejected)} sentence(s) removed by the verifier"
    st.caption(meta)
    for n in answer.notes:
        st.caption(n)
    with st.expander(f"Sources ({len(answer.docs)})"):
        for d in answer.docs:
            url = c.safe_url(d.url)
            link = f' · <a href="{c.e(url)}" target="_blank">listing ↗</a>' if url else ""
            st.markdown(f"<span class='ji-cite'>{c.e(d.id)}</span> <b>{c.e(d.title[:90])}</b> <span class='ji-muted'>({c.e(d.kind)})"
                        f"</span>{link}<div class='ji-muted' style='white-space:pre-wrap'>{c.e(d.text[:500])}</div>", unsafe_allow_html=True)  # fmt: skip
    if answer.rejected:
        with st.expander("Removed by the verifier"):
            for text, reason in answer.rejected:
                st.markdown(f"- ~~{c.e(text[:200])}~~ - *{c.e(reason)}*")


def render() -> None:
    st.title("💬 AI Career Assistant")
    service = c.get_service()
    ss = c.state()
    top = st.columns([3, 2])
    with top[1]:
        ok, msg = service.llm_status()
        use_llm = st.toggle("Use local LLM (Ollama)", value=False, disabled=not ok,
                            help=msg if not ok else "Runs on this computer; CV data is never sent to an external service.")  # fmt: skip
        st.caption(("🟢 " if ok else "⚪ ") + (msg if len(msg) < 90 else msg[:90] + "…"))
    with top[0]:
        if ss.profile is None:
            st.info("No CV loaded: the assistant answers market and search questions. Load a CV for personal answers.", icon="ℹ️")
        elif ss.recs is None:
            with st.spinner("Computing your recommendations first…"):
                ss.recs = service.recommend(ss.profile, ss.prefs)
    c.learn("How answers are grounded",
            "Each question is routed to an intent, the matching documents are retrieved and numbered "
            "(<b>J</b> listings, <b>CV</b> lines, <b>S</b> system analyses, <b>M</b> market statistics, <b>K</b> knowledge base), and the "
            "answer is written from them only - by templates, or by the local LLM if enabled. A <b>verifier</b> then checks every "
            "sentence: cited ids must exist, and skills, numbers, names and credentials must appear in the cited documents. "
            "Sentences that fail are removed and shown below the answer.")  # fmt: skip

    cols = st.columns(4)
    clicked = None
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 4].button(q, key=f"sugg_{i}", width="stretch"):
            clicked = q
    for turn in ss.chat:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(c.e(turn["text"]))
            else:
                _render_answer(turn["answer"])
    question = st.chat_input("Ask about jobs, your fit, skills, companies, interviews…") or clicked
    if question:
        question = re.sub(r"\s+", " ", question).strip()[:1000]
        ss.chat.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(c.e(question))
        with st.chat_message("assistant"):
            with st.spinner("Retrieving, answering, verifying…"):
                session = service.assistant_session(ss.profile, ss.recs.ranked if ss.recs else [], ss.selected_job_id)
                answer = service.ask(session, question, use_llm=use_llm)
            _render_answer(answer)
        ss.chat.append({"role": "assistant", "answer": answer})
    if ss.chat and st.button("Clear conversation"):
        ss.chat = []
        st.rerun()
