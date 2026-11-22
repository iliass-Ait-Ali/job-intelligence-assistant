import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from search_data import write_jobs  # noqa: E402

from job_intel import cli  # noqa: E402
from job_intel.match import evaluate as me  # noqa: E402
from job_intel.rag.ollama import LLMUnavailable  # noqa: E402
from job_intel.search import evaluate as se  # noqa: E402
from job_intel.search.errors import EvalError  # noqa: E402

CV_SRC = Path(__file__).parent.parent / "eval" / "synthetic_cvs"


@pytest.fixture
def cvs(tmp_path):
    folder = tmp_path / "cvs"
    shutil.copytree(CV_SRC, folder)
    return folder


@pytest.fixture
def candidates(cvs):
    return me.load_candidates(cvs)


def test_candidates_load_with_their_preferences_and_evaluate_every_eligible_job(candidates):
    assert [c.id for c in candidates] == ["cv_en_ml_engineer", "cv_en_student_data_science", "cv_fr_data_engineer"]
    student = next(c for c in candidates if "student" in c.id)
    assert student.prefs.job_types == ["internship"] and student.prefs.max_results == 100
    with pytest.raises(EvalError):
        me.load_candidates(Path("does/not/exist"))


def test_both_systems_rank_the_same_eligible_jobs(candidates, match_searcher):
    for cand in candidates:
        ranks = me.rankings(match_searcher, cand)
        assert {m.job.job_id for m in ranks["pipeline"]} == {m.job.job_id for m in ranks["retrieval_rrf"]}
        rrf = [m.retrieval.rrf for m in ranks["retrieval_rrf"]]
        assert rrf == sorted(rrf, reverse=True)


def test_pool_is_blinded_ungraded_stable_and_keeps_existing_grades(candidates, match_searcher, tmp_path):
    rows, stats = me.build_pool(match_searcher, candidates, [], depth=3)
    assert rows and all(r["grade"] == "" for r in rows) and set(rows[0]) == set(se.JUDGMENT_COLUMNS)
    assert stats["new_items"] == len(rows)
    again, _ = me.build_pool(match_searcher, candidates, [], depth=3)
    assert again == rows
    rows[0]["grade"] = "2"
    merged, stats = me.build_pool(match_searcher, candidates, rows, depth=3)
    assert stats["new_items"] == 0 and merged[0]["grade"] == "2"
    deeper, stats = me.build_pool(match_searcher, candidates, merged, depth=8)
    assert stats["new_items"] > 0 and merged[0] in deeper


def test_matching_metrics_are_withheld_until_a_human_grades(candidates, match_searcher):
    rows, _ = me.build_pool(match_searcher, candidates, [], depth=5)
    with pytest.raises(EvalError, match="no metrics yet"):
        me.evaluate_matching(match_searcher, candidates, rows)
    rows[0]["grade"] = "1"
    with pytest.raises(EvalError, match="no metrics yet"):
        me.evaluate_matching(match_searcher, candidates, rows)


def test_matching_metrics_use_only_human_grades_and_report_coverage(candidates, match_searcher):
    rows, _ = me.build_pool(match_searcher, candidates, [], depth=5)
    student = next(c for c in candidates if "student" in c.id)
    for r in rows:  # TEST-ONLY stand-in for a human: the first internship is the right answer for the student
        if r["query_id"] == student.id:
            r["grade"] = "2" if r["job_id"].endswith("syn-01") else "0"
    report = me.evaluate_matching(match_searcher, candidates, rows)
    statuses = {c.candidate_id: c.status for c in report.candidates}
    assert statuses[student.id] == "scored" and list(statuses.values()).count("pending") == 2
    assert report.summary["pipeline"]["candidates"] == 1 and report.summary["pipeline"]["judged@10"] <= 1.0
    # a perfectly placed single relevant job gives nDCG 1 and MRR 1 for the documented ranking
    assert report.summary["pipeline"]["ndcg@10"] == pytest.approx(1.0)
    text = me.format_match_report(report)
    assert "1/3 candidates fully judged" in text and "pooled_recall is relative to the judged pool" in text


def test_grounding_run_counts_and_exports_an_audit_without_any_human_marks(candidates, match_searcher, tmp_path):
    report = me.grounding_run(match_searcher, candidates, top_n=2)
    assert report.explanations == 6 and report.mode_counts["deterministic"] == 6
    assert report.proposed >= report.verified > 0 and "CONSISTENCY check" in report.format()
    audit = tmp_path / "audit.csv"
    me.write_audit(audit, report.audit_rows)
    assert all(r["supported"] == "" for r in report.audit_rows)  # never pre-filled
    with pytest.raises(EvalError, match="no metrics yet"):
        me.score_audit(audit)


def test_grounding_with_an_unavailable_llm_reports_fallback_mode(candidates, match_searcher):
    class Down:
        model = "down"

        def chat_json(self, *a):
            raise LLMUnavailable("down")

    report = me.grounding_run(match_searcher, candidates[:1], top_n=1, client=Down())
    assert report.mode_counts["deterministic"] == 1 and report.explanations == 1


def test_audit_scoring_counts_only_human_marks(tmp_path):
    path = tmp_path / "audit.csv"
    rows = [{"candidate_id": "c", "job_id": "j", "mode": "deterministic", "kind": "strength", "claim": f"claim {i}", "cv_evidence": "", "job_evidence": "",
             "supported": mark, "notes": ""} for i, mark in enumerate(["yes", "yes", "partial", "no", ""])]  # fmt: skip
    me.write_audit(path, rows)
    score = me.score_audit(path)
    assert (score["items"], score["human_judged"], score["ungraded"], score["yes"]) == (5, 4, 1, 2)
    assert score["supported_rate_yes_only"] == pytest.approx(0.5)
    rows[0]["supported"] = "maybe"
    me.write_audit(path, rows)
    with pytest.raises(EvalError, match="yes, partial or no"):
        me.score_audit(path)


# ---- CLI --------------------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch, match_searcher, cvs):
    from job_intel.search import service

    monkeypatch.setattr(service, "open_searcher", lambda settings, **kw: (match_searcher, []))
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("JOBINTEL_DATA_DIR", str(tmp_path / "data"))  # never touch the real data folder
    (tmp_path / "eval").mkdir()
    shutil.copytree(cvs, tmp_path / "eval" / "synthetic_cvs")
    from match_data import match_jobs

    write_jobs(tmp_path / "data" / "processed" / "jobs.jsonl", match_jobs())
    monkeypatch.setenv("JOBINTEL_OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens: Ollama "not running"
    return tmp_path


def test_profile_command_prints_json_and_writes_nothing(env, capsys, monkeypatch):
    monkeypatch.chdir(env)
    before = set(env.rglob("*"))
    assert cli.main(["profile", "--cv", str(CV_SRC / "cv_fr_data_engineer.txt")]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["detected_language"] == "fr" and set(env.rglob("*")) == before


def test_profile_command_reports_unreadable_cvs(env, capsys, tmp_path):
    bad = tmp_path / "scan.pdf"
    bad.write_bytes(b"not a pdf")
    assert cli.main(["profile", "--cv", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_match_command_lists_evidence_and_explains_without_a_model(env, capsys):
    code = cli.main(["match", "--cv", str(CV_SRC / "cv_en_student_data_science.txt"), "--job-type", "internship", "-n", "3", "--explain", "1"])
    out = capsys.readouterr().out
    assert code == 0 and "[strong] Data Scientist Internship" in out and "deterministic" in out and "https://fixtures.invalid/" in out
    assert "%" not in out.split("Pool:")[1]  # no fit percentages


def test_match_command_with_llm_flag_degrades_gracefully_when_ollama_is_down(env, capsys):
    code = cli.main(["match", "--cv", str(CV_SRC / "cv_en_student_data_science.txt"), "-n", "2", "--explain", "1", "--llm"])
    out = capsys.readouterr().out
    assert code == 0 and "Showing the deterministic explanation instead" in out


def test_llm_status_command_reports_a_missing_server(env, capsys):
    assert cli.main(["llm-status"]) == 2
    assert "Ollama is not running" in capsys.readouterr().err


def test_match_eval_workflow(env, capsys):
    assert cli.main(["match-eval", "pool"]) == 0
    judgments = env / "eval" / "match_judgments.csv"
    assert judgments.exists() and (env / "data" / "eval" / "match_review.md").exists()
    assert cli.main(["match-eval", "run"]) == 2 and "no metrics yet" in capsys.readouterr().err

    rows = list(csv.DictReader(judgments.open(encoding="utf-8-sig")))
    for r in rows:  # test-only stand-in for the human grader
        r["grade"] = "1"
    with judgments.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    capsys.readouterr()
    assert cli.main(["match-eval", "run"]) == 0
    assert "3/3 candidates fully judged" in capsys.readouterr().out

    assert cli.main(["match-eval", "grounding", "--top", "1"]) == 0
    assert (env / "data" / "eval" / "grounding_audit.csv").exists()
    assert cli.main(["match-eval", "grounding-score"]) == 2


# ---- privacy ----------------------------------------------------------------------------------


def test_gitignore_protects_data_and_personal_documents():
    text = (Path(__file__).parent.parent / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", "*.pdf", "*.docx", "*.profile.json", "uploads/", ".env", "eval/judgments.csv"):
        assert pattern in text.splitlines(), pattern
    assert "!eval/synthetic_cvs/" in text  # only the clearly synthetic CVs may be committed


def test_git_actually_ignores_cvs_data_and_index():
    root = Path(__file__).parent.parent
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    paths = ["data/processed/jobs.jsonl", "data/index/chroma/x.sqlite3", "my_cv.pdf", "cvs/jane.txt", "jane.profile.json", ".env"]
    result = subprocess.run(["git", "check-ignore", *paths], cwd=root, capture_output=True, text=True)
    assert set(result.stdout.split()) == set(paths)
    keep = subprocess.run(["git", "check-ignore", "eval/synthetic_cvs/cv_en_ml_engineer.txt"], cwd=root, capture_output=True, text=True)
    assert keep.stdout.strip() == ""  # synthetic examples stay trackable


def test_cv_module_never_imports_network_clients_other_than_the_local_llm_client():
    root = Path(__file__).parent.parent / "src" / "job_intel"
    offenders = []
    for path in list((root / "match").glob("*.py")) + list((root / "app").glob("*.py")) + list((root / "rag").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if ("import requests" in text or "import httpx" in text or "urllib.request" in text) and path.name != "ollama.py":
            offenders.append(path.name)
    assert offenders == []  # only the loopback-checked Ollama client may open connections


def test_no_secrets_or_remote_ai_endpoints_in_the_source():
    root = Path(__file__).parent.parent / "src"
    blob = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py")).lower()
    for needle in ("api.openai.com", "api.anthropic.com", "sk-", "generativelanguage.googleapis"):
        assert needle not in blob, needle
    assert sys.version_info >= (3, 11)
