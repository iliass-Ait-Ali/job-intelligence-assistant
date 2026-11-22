"""Integration tests of the REST API and end-to-end tests of the Streamlit UI (synthetic corpus, no downloads)."""

from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]
CV_TEXT = (ROOT / "eval" / "synthetic_cvs" / "cv_en_student_data_science.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def client(platform_service):
    from fastapi.testclient import TestClient

    from job_intel.api.main import app, get_service

    app.dependency_overrides[get_service] = lambda: platform_service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def profile_json(client):
    r = client.post("/cv/parse", data={"text": CV_TEXT})
    assert r.status_code == 200
    return r.json()["profile"]


def test_health_jobs_and_job_detail(client):
    assert client.get("/health").json()["status"] == "ok"
    jobs = client.get("/jobs", params={"limit": 3}).json()
    assert jobs["total"] == 10 and len(jobs["items"]) == 3 and "description" not in jobs["items"][0]
    one = client.get(f"/jobs/{jobs['items'][0]['id']}").json()
    assert one["apply_url"].startswith("https://") and one["description"]
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs", params={"skill": "Kafka"}).json()["total"] >= 1


@pytest.mark.parametrize("engine", ["keyword", "bm25", "semantic", "hybrid", "hybrid_rerank"])
def test_search_every_engine(client, engine):
    r = client.post("/search", json={"query": "python machine learning", "engine": engine, "k": 3})
    assert r.status_code == 200 and 1 <= len(r.json()["hits"]) <= 3


def test_search_validation(client):
    assert client.post("/search", json={"query": "x", "engine": "bm25"}).status_code == 422
    assert client.post("/search", json={"query": "python", "engine": "magic"}).status_code == 422


def test_cv_parse_file_upload_and_errors(client):
    r = client.post("/cv/parse", files={"file": ("cv.txt", CV_TEXT.encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["source"] == "text" and r.json()["profile"]["skills"]
    assert client.post("/cv/parse", files={"file": ("cv.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.post("/cv/parse").status_code == 400


def test_recommendations_gap_plan(client, profile_json):
    prefs = {"target_roles": ["Data Scientist"], "hard_constraints": [], "max_results": 5}
    r = client.post("/recommendations", json={"profile": profile_json, "preferences": prefs})
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert recs and all(0 <= x["compatibility"] <= 100 for x in recs)
    assert [x["rank"] for x in recs] == list(range(1, len(recs) + 1))
    first = recs[0]
    assert sum(c["points"] for c in first["components"] if c["value"] is not None) == pytest.approx(first["compatibility"], abs=0.5)
    gap = client.post("/gap", json={"profile": profile_json, "preferences": prefs, "job_id": first["job_id"]}).json()
    assert set(gap["summary"]) == {"mastered", "partial", "missing", "missing_required"}
    plan = client.post("/plan", json={"profile": profile_json, "preferences": prefs}).json()
    assert "steps" in plan


def test_hard_constraints_exclude_with_reasons(client, profile_json):
    prefs = {"job_types": ["internship"], "hard_constraints": ["job_type"], "max_results": 10}
    body = client.post("/recommendations", json={"profile": profile_json, "preferences": prefs}).json()
    assert body["excluded"] and all(x["reasons"] for x in body["excluded"])
    assert all(r["contract_type"] in {"internship", None} for r in body["recommendations"])


def test_market_and_assistant(client, profile_json):
    skills = client.get("/market/skills", params={"top": 5}).json()
    assert skills["postings"] == 10 and len(skills["skills"]) == 5
    assert client.get("/market/cooccurrence", params={"min_count": 1}).status_code == 200
    assert client.get("/market/companies", params={"skill": "Python"}).json()["companies"]
    ans = client.post("/assistant/ask", json={"question": "Why does job 1 match my CV?", "profile": profile_json,
                                              "preferences": {"hard_constraints": []}}).json()  # fmt: skip
    assert ans["intent"] == "why_match" and ans["sentences"]
    ids = {d["id"] for d in ans["documents"]}
    assert all(set(s["citations"]) <= ids for s in ans["sentences"])


# ---- Streamlit end-to-end (AppTest runs the real script in-process) ----------------------------------


def _page_script(page: str) -> None:
    import streamlit as st

    from job_intel.ui import components as c
    from job_intel.ui.views import (  # noqa: F401
        assistant,
        evaluation,
        job_details,
        market,
        overview,
        profile,
        recommendations,
        search,
        skill_gap,
    )

    c.inject_css()
    c.state()
    {"overview": overview, "profile": profile, "search": search, "recommendations": recommendations,
     "details": job_details, "skill-gap": skill_gap, "market": market, "assistant": assistant,
     "evaluation": evaluation}[page].render()
    st.caption("rendered")


@pytest.fixture
def make_app(platform_service, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from job_intel.ui import components

    monkeypatch.setattr(components, "get_service", lambda: platform_service)

    def make(page: str):
        at = AppTest.from_function(_page_script, args=(page,), default_timeout=120)
        from job_intel.match.cv_text import from_pasted_text
        from job_intel.match.models import Preferences

        at.session_state["profile"] = platform_service.parse_profile(from_pasted_text(CV_TEXT))
        at.session_state["prefs"] = Preferences(target_roles=["Data Scientist"], hard_constraints=[], max_results=5)
        at.session_state["cv_meta"] = {"name": "test", "chars": len(CV_TEXT)}
        return at

    return make


def test_the_app_entry_point_builds_navigation(platform_service, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from job_intel.ui import components

    monkeypatch.setattr(components, "get_service", lambda: platform_service)
    at = AppTest.from_file(str(ROOT / "src" / "job_intel" / "ui" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["overview", "profile", "search", "recommendations", "details", "skill-gap", "market",
                                  "assistant", "evaluation"])  # fmt: skip
def test_every_page_renders_without_exception(make_app, page):
    at = make_app(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(cap.value == "rendered" for cap in at.caption)


def test_recommendations_then_gap_and_assistant_flow(make_app):
    at = make_app("recommendations").run()
    assert not at.exception
    recs = at.session_state["recs"]
    assert recs is not None and recs.ranked
    gap = make_app("skill-gap")
    gap.session_state["recs"] = recs
    gap.session_state["selected_job_id"] = recs.ranked[0][0].job.job_id
    gap.run()
    assert not gap.exception, [e.value for e in gap.exception]
    chat = make_app("assistant")
    chat.session_state["recs"] = recs
    chat.run()
    chat.chat_input[0].set_value("Why does job 1 match my CV?").run()
    assert not chat.exception, [e.value for e in chat.exception]
    assert chat.session_state["chat"][-1]["answer"].sentences
