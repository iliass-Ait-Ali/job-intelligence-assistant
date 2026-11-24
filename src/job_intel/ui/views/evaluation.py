"""Page 9 - Evaluation & method: the measured results (read from reports/*.json) and their caveats."""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from job_intel.ui import components as c
from job_intel.ui.theme import SERIES, SURFACE

REPORTS = c.ROOT / "reports"


def _load(name: str):
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ci_bar(systems: dict, metric: str, title: str) -> go.Figure:
    names = list(systems)
    means = [systems[n][metric]["mean"] for n in names]
    lo = [systems[n][metric]["mean"] - systems[n][metric]["ci_low"] for n in names]
    hi = [systems[n][metric]["ci_high"] - systems[n][metric]["mean"] for n in names]
    fig = go.Figure(go.Bar(x=names, y=means, marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2), cornerradius=4),
                           error_y=dict(type="data", array=hi, arrayminus=lo, color="#52514e", thickness=1.2, width=6),
                           text=[f"{m:.3f}" for m in means], textposition="outside",
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))  # fmt: skip
    fig.update_layout(title=title, height=340, yaxis=dict(range=[0, 1.05]), showlegend=False)
    return fig


def render() -> None:
    st.title("🧪 Evaluation & Method")
    st.caption("Every number below is produced by `jobintel bench …` and stored in `reports/`. Nothing is typed by hand.")
    retrieval = _load("retrieval_benchmark.json")
    skills = _load("skills_benchmark.json")
    scoring = _load("scoring_calibration.json")
    assistant = _load("assistant_benchmark.json")
    latency = _load("latency.json")
    if not any((retrieval, skills, scoring, assistant, latency)):
        st.info("No benchmark report yet. Run `jobintel bench all`.")
        return

    if retrieval:
        st.markdown("## 1 · Retrieval: five systems on the same judged queries")
        st.warning(retrieval["label_note"], icon="⚠️")
        sysm = retrieval["systems"]
        a, b = st.columns(2)
        a.plotly_chart(_ci_bar(sysm, "ndcg@10", "nDCG@10 (mean over queries, 95% bootstrap CI)"), width="stretch", config={"displayModeBar": False})
        b.plotly_chart(_ci_bar(sysm, "mrr@10_strict", "MRR@10, strict (relevant = grade 2; grade ≥ 1 saturates at 1.0)"), width="stretch", config={"displayModeBar": False})
        rows = [{"system": n, **{m: round(v["mean"], 3) for m, v in s.items() if isinstance(v, dict) and "mean" in v}} for n, s in sysm.items()]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"{retrieval['n_queries']} queries · corpus: {retrieval['corpus']} · pooled judgments: {retrieval['n_judgments']}")
        with st.expander("Per-query nDCG@10"):
            st.dataframe(retrieval["per_query"], width="stretch")
    if skills:
        st.markdown("## 2 · Skill extraction from listings (precision / recall / F1)")
        st.warning(skills["label_note"], icon="⚠️")
        cols = st.columns(3)
        c.kpi(cols[0], f"{skills['micro']['precision']:.2f}", "micro precision")
        c.kpi(cols[1], f"{skills['micro']['recall']:.2f}", "micro recall")
        c.kpi(cols[2], f"{skills['micro']['f1']:.2f}", "micro F1")
        st.caption(f"{skills['n_docs']} listings · {skills['n_gold']} gold skill mentions · in-vocabulary recall "
                   f"{skills['in_vocabulary_recall']:.2f}")  # fmt: skip
        with st.expander("Errors (false positives / negatives)"):
            st.json(skills["errors"], expanded=False)
        if "sector" in skills:
            st.caption(f"Sector heuristic accuracy on the same listings: {skills['sector']['accuracy']:.2f} "
                       f"(coverage {skills['sector']['coverage']:.2f})")  # fmt: skip
    if scoring:
        st.markdown("## 3 · Ranking candidates' jobs: does the compatibility score order them well?")
        st.warning(scoring["label_note"], icon="⚠️")
        st.dataframe([{"method": k, **v} for k, v in scoring["methods"].items()], width="stretch", hide_index=True)
        st.caption(scoring.get("conclusion", ""))
    if assistant:
        st.markdown("## 4 · Career assistant (RAG)")
        st.warning(assistant["label_note"], icon="⚠️")
        st.dataframe([{"mode": k, **v} for k, v in assistant["modes"].items()], width="stretch", hide_index=True)
        st.caption(assistant.get("definitions", ""))
    if latency:
        st.markdown("## 5 · Latency and resources")
        st.dataframe([{"step": k, **v} for k, v in latency["steps"].items()], width="stretch", hide_index=True)
        st.caption(latency.get("hardware", ""))
    st.markdown("---")
    st.markdown("Full methodology, limits and biases: `reports/evaluation_report.md` and `docs/learn/`.")
