"""Shared UI building blocks: service cache, session state, CSS, cards, badges, learn boxes.

Every string coming from a CV or a listing is HTML-escaped before it is placed in custom HTML;
only http(s) links are rendered.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from job_intel.ui.theme import BRAND, INK, INK_2, MUTED, STATUS, STATUS_TEXT

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CVS = ROOT / "eval" / "synthetic_cvs"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: Inter, 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }}
.ji-hero {{ background: linear-gradient(135deg, #312e81 0%, #4338ca 55%, #2a78d6 100%); color: #fff; border-radius: 18px;
           padding: 28px 32px; margin-bottom: 18px; }}
.ji-hero h1 {{ color: #fff; margin: 0 0 6px 0; font-size: 2.0rem; }}
.ji-hero p {{ color: #e0e7ff; margin: 0; font-size: 1.02rem; max-width: 900px; }}
.ji-card {{ background: #ffffff; border: 1px solid #e6e8ee; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
           box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.ji-card h4 {{ margin: 0 0 2px 0; font-size: 1.02rem; color: {INK}; }}
.ji-sub {{ color: {INK_2}; font-size: .88rem; }}
.ji-muted {{ color: {MUTED}; font-size: .82rem; }}
.ji-kpi {{ background:#fff; border:1px solid #e6e8ee; border-radius:14px; padding:14px 16px; }}
.ji-kpi .v {{ font-size: 1.7rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.ji-kpi .l {{ color: {INK_2}; font-size: .82rem; }}
.ji-chip {{ display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: .78rem; margin: 2px 4px 2px 0;
           border: 1px solid #e3e5ea; background: #f7f8fa; color: {INK_2}; white-space: nowrap; }}
.ji-chip.good {{ background:#e9f7ea; border-color:#bfe5c1; color:{STATUS_TEXT['good']}; }}
.ji-chip.partial {{ background:#fff6e0; border-color:#f5dc9c; color:{STATUS_TEXT['warning']}; }}
.ji-chip.missing {{ background:#fdecec; border-color:#f3c2c2; color:{STATUS_TEXT['critical']}; }}
.ji-chip.info {{ background:#eef2ff; border-color:#c7d2fe; color:#3730a3; }}
.ji-score {{ display:flex; align-items:center; gap:14px; }}
.ji-ring {{ width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center;
           font-weight:700; color:{INK}; font-size:1.05rem; flex: 0 0 64px; }}
.ji-learn {{ border-left: 4px solid {BRAND}; background:#f5f6ff; padding: 12px 16px; border-radius: 8px; margin: 8px 0 14px 0;
            color: {INK_2}; font-size: .92rem; }}
.ji-learn b {{ color: #312e81; }}
.ji-cite {{ display:inline-block; background:#eef2ff; color:#3730a3; border-radius:6px; padding:0 6px; font-size:.75rem;
           font-weight:600; margin-left:3px; }}
.ji-step {{ display:inline-flex; width:28px; height:28px; border-radius:50%; background:{BRAND}; color:#fff; align-items:center;
           justify-content:center; font-weight:700; margin-right:10px; }}
a.ji-apply {{ display:inline-block; background:{BRAND}; color:#fff !important; padding:5px 12px; border-radius:8px;
             text-decoration:none; font-size:.85rem; font-weight:600; }}
section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid #eceef2; }}
</style>
"""


def e(text: object) -> str:
    return html.escape(str(text if text is not None else ""))


def safe_url(url: str | None) -> str | None:
    return url if url and url.startswith(("http://", "https://")) else None


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="ji-hero"><h1>{e(title)}</h1><p>{e(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi(col, value: str, label: str) -> None:
    col.markdown(f'<div class="ji-kpi"><div class="v">{e(value)}</div><div class="l">{e(label)}</div></div>',
                 unsafe_allow_html=True)  # fmt: skip


def chips(items, kind: str = "") -> str:
    return "".join(f'<span class="ji-chip {kind}">{e(i)}</span>' for i in items)


def learn(title: str, body_html: str) -> None:
    """A pedagogical aside: what this page computes and why (body is trusted, authored HTML)."""
    st.markdown(f'<div class="ji-learn"><b>🎓 {e(title)}</b><br>{body_html}</div>', unsafe_allow_html=True)


def score_ring(score: float, label: str = "") -> str:
    """Conic-gradient ring; the number is always printed (color never carries the value alone)."""
    pct = max(0.0, min(100.0, score))
    color = STATUS["good"] if pct >= 70 else (STATUS["warning"] if pct >= 45 else STATUS["serious"])
    return (f'<div class="ji-ring" style="background: conic-gradient({color} {pct * 3.6}deg, #eceef2 0deg);">'
            f'<div style="background:#fff;width:50px;height:50px;border-radius:50%;display:flex;align-items:center;'
            f'justify-content:center;">{pct:.0f}</div></div>')  # fmt: skip


BAND_LABEL = {"high": "High fit", "medium": "Partial fit", "low": "Low fit"}
STATUS_ICON = {"satisfied": "✅", "violated": "⛔", "unknown": "❔"}


# ---- shared resources & session ---------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models and indexes (first start only)…")
def get_service():
    from job_intel.config import Settings
    from job_intel.services import JobIntelService

    service = JobIntelService(Settings.from_env())
    service.warm_up()
    return service


def state():
    ss = st.session_state
    for key, default in (("profile", None), ("prefs", None), ("recs", None), ("selected_job_id", None),
                         ("declared", {}), ("chat", []), ("cv_meta", None)):  # fmt: skip
        if key not in ss:
            ss[key] = default
    return ss


def require_profile() -> bool:
    if state().profile is None:
        st.info("No CV loaded yet. Go to **CV & Profile** to upload a CV or load a synthetic example.", icon="📄")
        if st.button("Go to CV & Profile", type="primary"):
            st.switch_page(PAGES["profile"])
        return False
    return True


def require_recs() -> bool:
    if not require_profile():
        return False
    if state().recs is None:
        st.info("No recommendations computed yet. Open **Recommendations** and run the matcher.", icon="🎯")
        if st.button("Go to Recommendations", type="primary"):
            st.switch_page(PAGES["recommendations"])
        return False
    return True


PAGES: dict[str, object] = {}  # filled by app.py so pages can link to each other


def short_location(location: str | None, keep: int = 3) -> str:
    """'A, B, C, D, E' -> 'A, B, C +2 more' (remote boards list dozens of allowed countries)."""
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    if len(parts) <= keep + 1:
        return ", ".join(parts) or "?"
    return f"{', '.join(parts[:keep])} +{len(parts) - keep} more"
