import csv
import json
import subprocess
import sys

import pytest

pytest.importorskip("chromadb")

from search_data import jobs, write_jobs  # noqa: E402

from job_intel import cli  # noqa: E402

MODEL = "hashing-test-v1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    jobs_file = write_jobs(tmp_path / "jobs.jsonl")
    queries = tmp_path / "eval" / "queries.json"
    queries.parent.mkdir()
    queries.write_text(json.dumps({"queries": [{"id": "q1", "text": "spark airflow pipelines"}]}), encoding="utf-8")
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("JOBINTEL_CHUNK_MAX_TOKENS", "100")
    monkeypatch.setenv("JOBINTEL_CHUNK_OVERLAP_TOKENS", "20")
    monkeypatch.delenv("JOBINTEL_EMBEDDING_MODEL", raising=False)
    return {"data": str(tmp_path / "data"), "jobs": str(jobs_file), "tmp": tmp_path}


def run(env, *args, capsys):
    code = cli.main([args[0], *args[1:], "--data-dir", env["data"], "--jobs-file", env["jobs"]])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_phase1_cli_does_not_need_the_search_dependencies():
    code = "import sys, job_intel.cli; print([m for m in ('numpy','chromadb','torch','sentence_transformers') if m in sys.modules])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_index_then_search_end_to_end(env, capsys):
    code, out, _ = run(env, "index", "--model", MODEL, capsys=capsys)
    assert code == 0 and "Jobs:    13 (added 13" in out and "Chunks:" in out and "limit 100" in out
    code, out, _ = run(env, "index", "--model", MODEL, capsys=capsys)  # idempotent
    assert code == 0 and "added 0, updated 0, unchanced".replace("unchanced", "unchanged") in out and "0 embedded this run" in out

    code, out, err = run(env, "search", "spark airflow databricks pipelines", "-k", "3", capsys=capsys)
    assert code == 0 and "[semantic" in out and "cosine similarity" in out
    first = out.split(" 1. ")[1].splitlines()
    assert "Data Engineer - Gamma" in first[0] and "https://fixtures.invalid/arbeitnow/syn-03" in out

    code, out, _ = run(env, "search", "kafka streaming", "--engine", "bm25", capsys=capsys)
    assert code == 0 and "BM25 lexical baseline" in out and "Senior Data Engineer - Acme" in out

    code, out, _ = run(env, "search", "data", "--engine", "both", "-k", "2", capsys=capsys)
    assert out.count("Query:") == 2


def test_search_filters_and_company_cap_from_the_command_line(env, capsys):
    run(env, "index", "--model", MODEL, capsys=capsys)
    code, out, _ = run(env, "search", "analyste de données", "--job-type", "internship", "--location", "Paris", capsys=capsys)
    assert code == 0 and "Eligible: 2/13" in out and "Data Analyst" in out and "London" not in out
    code, out, _ = run(env, "search", "engineer", "--max-per-company", "1", "-k", "10", capsys=capsys)
    assert "hidden by --max-per-company" in out
    code, out, _ = run(env, "search", "data", "--since", "2026-10-20", "--remote", capsys=capsys)
    assert "Eligible: 1/13" in out and "AI Engineer" in out


def test_search_errors_are_clean_one_liners(env, capsys):
    code, _, err = run(env, "search", "anything", capsys=capsys)
    assert code == 2 and err.startswith("error: no index") and "Traceback" not in err
    run(env, "index", "--model", MODEL, capsys=capsys)
    code, _, err = run(env, "search", "x", "--model", "other/model", capsys=capsys)
    assert code == 2 and "built with 'hashing-test-v1'" in err
    code, _, err = run(env, "search", "x", "--since", "tomorrow", capsys=capsys)
    assert code == 2 and "--since" in err


def test_changed_settings_require_rebuild(env, capsys, monkeypatch):
    run(env, "index", "--model", MODEL, capsys=capsys)
    monkeypatch.setenv("JOBINTEL_CHUNK_MAX_TOKENS", "80")
    code, _, err = run(env, "index", "--model", MODEL, capsys=capsys)
    assert code == 2 and "chunk_max_tokens" in err and "--rebuild" in err
    code, out, _ = run(env, "index", "--model", MODEL, "--rebuild", capsys=capsys)
    assert code == 0 and "added 13" in out


def test_exclude_source_removes_it_from_the_index(env, capsys):
    run(env, "index", "--model", MODEL, capsys=capsys)
    code, out, _ = run(env, "index", "--model", MODEL, "--exclude-source", "remotive", capsys=capsys)
    assert code == 0 and "removed 1" in out
    code, out, _ = run(env, "index-status", capsys=capsys)
    status = json.loads(out)
    assert status["jobs"] == 12 and status["jobs_by_source"] == {"arbeitnow": 12}


def test_stale_index_warning(env, capsys):
    run(env, "index", "--model", MODEL, capsys=capsys)
    records = jobs()
    records[0]["description"] += " changed"
    write_jobs(env["tmp"] / "jobs.jsonl", records)
    code, _, err = run(env, "search", "data", capsys=capsys)
    assert code == 0 and "out of date" in err and "1 changed" in err


def test_eval_workflow_withholds_metrics_until_a_human_grades(env, capsys):
    run(env, "index", "--model", MODEL, capsys=capsys)
    code, out, _ = run(env, "eval", "pool", "--depth", "4", capsys=capsys)
    assert code == 0 and "never fills them in" in out
    judgments = env["tmp"] / "eval" / "judgments.csv"
    assert judgments.exists() and (env["tmp"] / "data" / "eval" / "review.md").exists()

    code, _, err = run(env, "eval", "run", capsys=capsys)
    assert code == 2 and "no metrics yet" in err

    # Test-only stand-in for the human: grade every pooled item 1.
    rows = list(csv.DictReader(judgments.open(encoding="utf-8-sig")))
    for r in rows:
        r["grade"] = "1"
    with judgments.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = env["tmp"] / "report.json"
    code, out, _ = run(env, "eval", "run", "--per-query", "--report-json", str(report_path), capsys=capsys)
    assert code == 0 and "1/1 queries fully judged" in out and "ndcg@10" in out and "semantic" in out and "bm25" in out
    assert json.loads(report_path.read_text())["graded_items"] == len(rows)
