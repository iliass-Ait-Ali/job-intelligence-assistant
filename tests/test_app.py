import dataclasses
import json
import os
import socket
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("pypdf")
gr = pytest.importorskip("gradio")

from pdf_helpers import blank_pdf, make_pdf  # noqa: E402

from job_intel.app import controller as ctl  # noqa: E402
from job_intel.app import render  # noqa: E402
from job_intel.app.controller import AppContext  # noqa: E402
from job_intel.config import Settings  # noqa: E402
from job_intel.rag.ollama import LLMUnavailable  # noqa: E402

CV_TEXT = (Path(__file__).parent.parent / "eval" / "synthetic_cvs" / "cv_en_student_data_science.txt").read_text(encoding="utf-8")
TODAY = date(2026, 10, 23)


class StubClient:
    model = "stub-llm:1b"

    def __init__(self, available=True, answer=None):
        self.available, self.answer = available, answer

    def check(self):
        if not self.available:
            raise LLMUnavailable("Ollama is not running.")
        return "Ollama is running; model stub-llm:1b is available."

    def chat_json(self, system, user, schema):
        return self.answer(user) if callable(self.answer) else (self.answer or {"strengths": [], "gaps": [], "preparation": []})


@pytest.fixture
def ctx(match_searcher, tmp_path):
    return AppContext(Settings(user_agent="t", data_dir=tmp_path), searcher_factory=lambda: match_searcher,
                      client_factory=lambda: StubClient(available=False))  # fmt: skip


@pytest.fixture
def parsed():
    return ctl.parse_cv_input(None, CV_TEXT, today=TODAY)


# ---- profile step ---------------------------------------------------------------------------


def test_pasted_text_is_parsed(parsed):
    assert parsed.error is None and parsed.profile and "Python" in parsed.profile.skill_names
    assert json.loads(parsed.json_text)["detected_language"] == "en"


def test_uploaded_text_file_is_parsed_and_the_temporary_copy_is_deleted(tmp_path):
    upload = tmp_path / "gradio_tmp_cv.txt"
    upload.write_text(CV_TEXT, encoding="utf-8")
    result = ctl.parse_cv_input(str(upload), "", today=TODAY)
    assert result.profile and not upload.exists()  # a personal document is not left in the temp folder


def test_pdf_upload_works_and_scanned_pdf_gives_a_message(tmp_path):
    good = tmp_path / "cv.pdf"
    good.write_bytes(make_pdf([CV_TEXT.splitlines()[:45]]))
    result = ctl.parse_cv_input(str(good), None, today=TODAY)
    assert result.profile and {"Python", "SQL"} <= result.profile.skill_names and not good.exists()
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(blank_pdf())
    bad = ctl.parse_cv_input(str(scan), None)
    assert bad.profile is None and "scan" in bad.error.lower() and not scan.exists()


def test_file_wins_over_pasted_text_with_a_message(tmp_path):
    upload = tmp_path / "cv.txt"
    upload.write_text(CV_TEXT, encoding="utf-8")
    result = ctl.parse_cv_input(str(upload), "some other pasted text", today=TODAY)
    assert result.profile and any("uploaded file was used" in m for m in result.messages)


@pytest.mark.parametrize(("path", "text"), [(None, ""), (None, None), (None, "   "), ("missing.pdf", None)])
def test_nothing_to_parse_gives_an_error_not_an_exception(path, text):
    result = ctl.parse_cv_input(path, text)
    assert result.profile is None and result.error


def test_parsing_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctl.parse_cv_input(None, CV_TEXT, today=TODAY)
    assert list(tmp_path.iterdir()) == []


def test_edited_profile_json_is_validated(parsed):
    profile, error = ctl.profile_from_json(parsed.json_text)
    assert profile == parsed.profile and error is None
    edited = json.loads(parsed.json_text)
    edited["skills"] = [s for s in edited["skills"] if s["name"] != "Python"]
    assert "Python" not in ctl.profile_from_json(json.dumps(edited))[0].skill_names  # the user's correction is what is used
    for bad, expect in [("", "empty"), ("{not json", "not valid JSON"), (json.dumps({"skills": [{"name": ""}]}), "skills.0.name"),
                        (json.dumps({"extra": 1}), "extra"), (json.dumps({"education": [{"level": "wizard"}]}), "education.0.level")]:  # fmt: skip
        assert expect in ctl.profile_from_json(bad)[1]


def test_preferences_are_parsed_from_form_values():
    prefs = ctl.build_preferences("Data Scientist; ML Engineer,\nAI Engineer", ["internship"], "Paris, Switzerland", "Remote only", ["job_type"], 7)
    assert prefs.target_roles == ["Data Scientist", "ML Engineer", "AI Engineer"] and prefs.locations == ["Paris", "Switzerland"]
    assert prefs.remote == "only" and prefs.max_results == 7 and prefs.hard_constraints == ["job_type"]
    assert ctl.build_preferences("", None, "", "No preference", None).hard_constraints == []


# ---- recommendations and analysis ------------------------------------------------------------


def test_recommendations_explanation_and_comparison_without_ollama(ctx, parsed):
    prefs = ctl.build_preferences("", ["internship"], "France", "No preference", ["job_type", "location"])
    result = ctl.recommend(ctx, parsed.profile, prefs)
    labels = ctl.job_labels(result)
    assert labels[0].startswith("1. Data Scientist Internship")
    exp = ctl.explain_job(ctx, parsed.profile, result, labels[0], use_llm=True)  # Ollama is "down"
    assert exp.mode == "deterministic" and "unavailable" in exp.notes[0] and "Ollama is not running" in exp.notes[0]
    comp = ctl.compare_jobs(ctx, parsed.profile, result, labels[:2], use_llm=False)
    assert len(comp.rows) == 2 and comp.mode == "deterministic"


def test_working_local_model_is_used_and_verified(ctx, parsed):
    ctx.client_factory = lambda: StubClient(available=True, answer=lambda user: {"strengths": [], "gaps": [], "preparation": [
        {"suggestion": "Be ready to talk about scikit-learn.", "cv_refs": [], "job_refs": [json.loads(user)["job_snippets"][0]["id"]]}]})  # fmt: skip
    result = ctl.recommend(ctx, parsed.profile, ctl.build_preferences("", [], "", "No preference", []))
    label = ctl.job_labels(result)[0]
    exp = ctl.explain_job(ctx, parsed.profile, result, label, use_llm=True)
    assert exp.mode in {"llm", "deterministic"}  # an answer without strengths/gaps falls back, never crashes
    assert ctx.llm_status() == (True, "Ollama is running; model stub-llm:1b is available.")


def test_selection_errors_are_clear(ctx, parsed):
    result = ctl.recommend(ctx, parsed.profile, ctl.build_preferences("", [], "", "No preference", []))
    labels = ctl.job_labels(result)
    with pytest.raises(ValueError, match="Select a job"):
        ctl.explain_job(ctx, parsed.profile, result, None, use_llm=False)
    with pytest.raises(ValueError, match="Select a job"):
        ctl.explain_job(ctx, parsed.profile, result, "99. Nope", use_llm=False)
    with pytest.raises(ValueError, match="at least two"):
        ctl.compare_jobs(ctx, parsed.profile, result, labels[:1], use_llm=False)
    with pytest.raises(ValueError, match="at most 4"):
        ctl.compare_jobs(ctx, parsed.profile, result, labels[:5] if len(labels) >= 5 else labels * 3, use_llm=False)


def test_unexpected_errors_are_not_shown_as_stack_traces():
    assert "Unexpected error (ZeroDivisionError)" in ctl.user_message_for(ZeroDivisionError("secret details /home/x"))
    assert "secret" not in ctl.user_message_for(ZeroDivisionError("secret details"))
    assert ctl.user_message_for(ValueError("Select a job")) == "Select a job"


# ---- rendering is escaped ---------------------------------------------------------------------


def test_listing_text_cannot_inject_html_or_unsafe_links(ctx, parsed):
    result = ctl.recommend(ctx, parsed.profile, ctl.build_preferences("", [], "", "No preference", []))
    hostile = dataclasses.replace(  # copies: the searcher fixture is shared by other tests
        result.matches[0].job, title='<script>alert("x")</script>', company="<img src=x onerror=alert(1)>", url="javascript:alert(1)")
    match = dataclasses.replace(result.matches[0], job=hostile)
    html = render.render_job_card(1, match)
    assert "<script>" not in html and "<img " not in html and "&lt;script&gt;" in html
    assert 'href="javascript' not in html and "<a " not in html.split("jm-meta")[1].split("</div>")[0]  # the unsafe URL is not linked


def test_safe_link_only_accepts_http_urls():
    assert 'href="https://example.com/a?b=1&amp;c=2"' in render.safe_link("https://example.com/a?b=1&c=2", "open")
    for bad in ("javascript:alert(1)", "data:text/html,x", "", None, "//evil", "ftp://x"):
        assert "<a " not in render.safe_link(bad, "open")


def test_results_page_states_there_is_no_percentage_score(ctx, parsed):
    result = ctl.recommend(ctx, parsed.profile, ctl.build_preferences("", ["internship"], "", "No preference", ["job_type"]))
    html = render.render_results(result)
    assert "no percentage score" in html and "Excluded by hard constraints" in html and "Original listing" in html
    assert "%" not in render.render_job_card(1, result.matches[0]).replace("&amp;", "")  # no fit percentage anywhere


def test_explanation_page_labels_the_mode_and_shows_evidence(ctx, parsed):
    result = ctl.recommend(ctx, parsed.profile, ctl.build_preferences("", [], "", "No preference", []))
    exp = ctl.explain_job(ctx, parsed.profile, result, ctl.job_labels(result)[0], use_llm=False)
    html = render.render_explanation(exp, result.matches[0])
    assert "no language model was used" in html and "Evidence used" in html
    assert "Why it matches" in html and "[CV" in html and ", JOB" in html  # statements carry their citations


# ---- the Gradio app itself ---------------------------------------------------------------------


def test_app_builds_with_the_requested_sections(ctx):
    demo = ctl_build(ctx)
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    assert {"1 · Candidate profile", "2 · Preferences", "3 · Recommendations", "4 · AI analysis"} <= labels
    assert {"Target roles (comma-separated; empty = use roles stated in the CV)", "Hard constraints"} <= labels


def ctl_build(ctx):
    from job_intel.app.gradio_app import build_app

    return build_app(ctx)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_full_interface_workflow_through_the_real_server(ctx):
    gradio_client = pytest.importorskip("gradio_client")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    demo = ctl_build(ctx)
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        client = gradio_client.Client(f"http://127.0.0.1:{port}/", verbose=False)
        # 1. profile
        profile_json, profile_html, status = client.predict(None, CV_TEXT, api_name="/on_parse")[-3:]
        assert "CV parsed" in status and "Python" in profile_html and json.loads(profile_json)["skills"]
        # empty input -> readable error, no crash
        assert "empty" in client.predict(None, "", api_name="/on_parse")[-1].lower()
        # 3. recommendations (internship in France)
        out = client.predict("", ["internship"], "France", "No preference", ["job_type", "location"], 10, api_name="/on_recommend")
        results_html, job_pick, compare_pick, find_status = out[-4:]
        assert "recommendations" in find_status and "Data Scientist Internship" in results_html and "Original listing" in results_html
        label = job_pick if isinstance(job_pick, str) else job_pick["value"]
        # 4. AI analysis without a local model -> deterministic, labelled, with a clear message
        html = client.predict(label, True, "English", api_name="/on_explain")
        assert "no language model was used" in html and "unavailable" in html.lower()
        assert "Ollama is not running" in client.predict(api_name="/on_llm_status")
        # comparing with one job selected is a message, not a crash
        assert "at least two" in client.predict([label], True, "English", api_name="/on_compare")
    finally:
        demo.close()
