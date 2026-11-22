"""Page 2 - CV upload & profile analysis, plus the preferences that drive matching."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from job_intel.match.cv_text import CVExtractionError
from job_intel.match.models import Preferences
from job_intel.ui import components as c

JOB_TYPES = ["internship", "full_time", "working_student", "part_time", "contract", "freelance", "permanent"]
HARD = ["job_type", "location", "experience", "education", "language"]


def _load(cv, name: str, prefs: Preferences | None = None) -> None:
    service = c.get_service()
    ss = c.state()
    ss.profile = service.parse_profile(cv)
    ss.cv_meta = {"name": name, "source": cv.source, "chars": len(cv.text), "pages": cv.pages}
    ss.prefs = prefs or Preferences(target_roles=ss.profile.target_roles)
    ss.recs, ss.selected_job_id, ss.declared, ss.chat = None, None, {}, []


def _input_section() -> None:
    tab_upload, tab_sample, tab_paste = st.tabs(["Upload a CV", "Use a synthetic example", "Paste text"])
    with tab_upload:
        up = st.file_uploader("PDF, DOCX or TXT (max 10 MB). Processed in memory, never stored.", type=["pdf", "docx", "txt"])
        if up is not None and st.button("Analyse this CV", type="primary"):
            fd, tmp = tempfile.mkstemp(suffix=Path(up.name).suffix.lower())
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(up.getvalue())
                _load(c.get_service().read_cv(path=tmp), up.name)
            except CVExtractionError as exc:
                st.error(str(exc))
            finally:
                os.unlink(tmp)
    with tab_sample:
        samples = sorted(c.SAMPLE_CVS.glob("*.txt"))
        names = {p.stem.replace("cv_", "").replace("_", " "): p for p in samples}
        choice = st.selectbox("Synthetic CVs (invented people, used for tests and evaluation)", list(names))
        if st.button("Load example", type="primary", key="load_sample"):
            path = names[choice]
            prefs_path = path.with_suffix(".prefs.json")
            prefs = Preferences(**json.loads(prefs_path.read_text(encoding="utf-8"))) if prefs_path.exists() else None
            _load(c.get_service().read_cv(path=path), f"example: {choice}", prefs)
    with tab_paste:
        text = st.text_area("CV text", height=220, placeholder="Paste the text of a CV…")
        if st.button("Analyse pasted text", key="paste"):
            try:
                _load(c.get_service().read_cv(text=text), "pasted text")
            except CVExtractionError as exc:
                st.error(str(exc))


def _profile_section() -> None:
    ss = c.state()
    p = ss.profile
    meta = ss.cv_meta or {}
    st.markdown(f"#### Extracted profile · <span class='ji-muted'>{c.e(meta.get('name'))} · {meta.get('chars', 0):,} characters · "
                f"language: {c.e(p.detected_language)}</span>", unsafe_allow_html=True)  # fmt: skip
    for w in p.warnings:
        st.warning(w, icon="⚠️")
    cols = st.columns(4)
    bounds = p.experience_months_bounds()
    c.kpi(cols[0], str(len(p.skills)), "technical skills")
    c.kpi(cols[1], str(len(p.experiences)), "experiences")
    c.kpi(cols[2], str(len(p.projects)), "projects")
    c.kpi(cols[3], f"{bounds[0]}–{min(bounds[1], 999)} mo" if bounds else "n/a", "dated experience (jobs – all)")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Technical skills** - <span class='ji-chip good'>demonstrated</span> used in an experience/project · "
                    "<span class='ji-chip'>listed</span> only in a skills list", unsafe_allow_html=True)  # fmt: skip
        by_cat: dict[str, list] = {}
        for s in p.skills:
            by_cat.setdefault(s.category.replace("_", " "), []).append(s)
        for cat, items in sorted(by_cat.items()):
            html = "".join(f'<span class="ji-chip {"good" if s.context == "demonstrated" else ""}" title="{c.e(s.evidence[0].text if s.evidence else "")}">'
                           f'{c.e(s.name)}</span>' for s in items)  # fmt: skip
            st.markdown(f"<div class='ji-muted'>{c.e(cat)}</div>{html}", unsafe_allow_html=True)
        with st.expander("Evidence for every skill (the CV lines it was read from)"):
            st.dataframe([{"skill": s.name, "where": s.context, "evidence": s.evidence[0].text if s.evidence else ""} for s in p.skills],
                         width="stretch", hide_index=True)  # fmt: skip
        st.markdown("**Experience**")
        for x in p.experiences:
            dates = f"{x.start or '?'} → {'present' if x.current else (x.end or '?')}"
            st.markdown(f'<div class="ji-card"><h4>{c.e(x.title or "Untitled role")}</h4><div class="ji-sub">{c.e(x.organization or "")} · '
                        f'{c.e(x.kind)} · {c.e(dates)}</div>{c.chips(x.skills, "info")}</div>', unsafe_allow_html=True)  # fmt: skip
        if p.projects:
            st.markdown("**Projects**")
            for pr in p.projects:
                st.markdown(f'<div class="ji-card"><h4>{c.e(pr.name or "Project")}</h4><div class="ji-sub">'
                            f'{c.e((pr.description or [""])[0][:220])}</div>{c.chips(pr.skills, "info")}</div>', unsafe_allow_html=True)  # fmt: skip
    with right:
        st.markdown("**Education**")
        for ed in p.education:
            status = " (in progress)" if ed.in_progress else ""
            st.markdown(f'<div class="ji-card"><h4>{c.e(ed.degree or ed.level or "Degree")}{status}</h4>'
                        f'<div class="ji-sub">{c.e(ed.institution or "")} {c.e(ed.end or "")}</div></div>', unsafe_allow_html=True)  # fmt: skip
        st.markdown("**Languages**")
        st.markdown(c.chips([f"{l.language} ({l.level_text})" if l.level_text else l.language for l in p.spoken_languages]) or
                    "<span class='ji-muted'>none found</span>", unsafe_allow_html=True)  # fmt: skip
        st.markdown("**Certifications**")
        st.markdown(c.chips(p.certifications) or "<span class='ji-muted'>none found</span>", unsafe_allow_html=True)
        st.markdown("**Sectors of experience** <span class='ji-muted'>(keyword estimate)</span>", unsafe_allow_html=True)
        st.markdown(c.chips(p.experience_sectors) or "<span class='ji-muted'>not determinable</span>", unsafe_allow_html=True)
        st.markdown("**Soft skills claimed** <span class='ji-muted'>(shown, never scored)</span>", unsafe_allow_html=True)
        st.markdown(c.chips(p.soft_skills) or "<span class='ji-muted'>none found</span>", unsafe_allow_html=True)
        with st.expander("Raw profile JSON (what the API returns)"):
            st.json(p.model_dump(mode="json"), expanded=False)
    c.learn("How the CV is parsed",
            "Rule-based and deterministic: section headings in EN/FR/DE, date ranges → duration <b>bounds</b> (never a guess), a "
            "~150-skill taxonomy with aliases (<i>sklearn → scikit-learn</i>) shared with the job side. A skill used inside an "
            "experience counts as <b>demonstrated</b>; one only listed counts as <b>listed</b>. No proficiency is inferred and no "
            "contact details are extracted. Trade-off: robust and explainable, but it only knows the vocabulary it was given.")  # fmt: skip


def _preferences_section() -> None:
    ss = c.state()
    prefs: Preferences = ss.prefs or Preferences()
    st.markdown("#### Preferences")
    with st.form("prefs_form"):
        a, b = st.columns(2)
        roles = a.text_input("Target roles (comma-separated)", ", ".join(prefs.target_roles), help="e.g. Data Scientist, ML Engineer")
        locations = b.text_input("Locations (comma-separated)", ", ".join(prefs.locations), help="cities or countries, e.g. Paris, France, Morocco")
        job_types = a.multiselect("Contract types", JOB_TYPES, default=[j for j in prefs.job_types if j in JOB_TYPES])
        remote = b.radio("Remote", ["any", "prefer", "only"], index=["any", "prefer", "only"].index(prefs.remote), horizontal=True)
        hard = st.multiselect("Eliminatory (hard) constraints - a clear violation excludes the job; unknown never does",
                              HARD, default=[h for h in prefs.hard_constraints if h in HARD])  # fmt: skip
        max_results = st.slider("Recommendations to show", 5, 40, prefs.max_results)
        if st.form_submit_button("Save preferences", type="primary"):
            split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
            ss.prefs = Preferences(target_roles=split(roles), locations=split(locations), job_types=job_types, remote=remote,
                                   hard_constraints=hard, max_results=max_results)  # fmt: skip
            ss.recs = None
            st.success("Saved. Open **Recommendations** to compute your matches.")


def render() -> None:
    st.title("📄 CV & Profile")
    st.caption("Your CV becomes a structured, evidence-backed profile. Everything stays in this browser session.")
    _input_section()
    if c.state().profile is not None:
        st.divider()
        _profile_section()
        st.divider()
        _preferences_section()
        if st.button("Find my matches →", type="primary"):
            st.switch_page(c.PAGES["recommendations"])
