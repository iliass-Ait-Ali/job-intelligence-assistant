"""Phase 4: hybrid retrieval (RRF), complete-judgment recall, bootstrap intervals, baselines and the labelling tool."""

import json

import pytest

pytest.importorskip("chromadb")

from search_data import jobs  # noqa: E402

from job_intel import labeling  # noqa: E402
from job_intel.search import evaluate as ev  # noqa: E402
from job_intel.search.document import ChunkingConfig  # noqa: E402
from job_intel.search.embedder import HashingEmbedder  # noqa: E402
from job_intel.search.filters import JobFilter  # noqa: E402
from job_intel.search.index import IndexConfig, JobIndex  # noqa: E402
from job_intel.search.retrieve import BM25, ENGINES, HYBRID, RRF_K, SEMANTIC, Searcher  # noqa: E402

EMB = HashingEmbedder(max_seq_length=128)


@pytest.fixture
def searcher(tmp_path):
    index = JobIndex.open_for_build(tmp_path / "idx", IndexConfig.build(EMB, ChunkingConfig(100, 20)))
    index.sync(jobs(), EMB)
    return Searcher(index, EMB, {j["id"]: j for j in jobs()})


def ids(resp):
    return [h.job.job_id for h in resp.hits]


# ---- hybrid retrieval ---------------------------------------------------------------------------


def test_hybrid_is_a_listed_engine():
    assert ENGINES == (SEMANTIC, BM25, HYBRID) and RRF_K == 60


def test_hybrid_equals_a_hand_computed_reciprocal_rank_fusion(searcher):
    query = "spark airflow data pipelines databricks"
    n = len(searcher.table)
    sem, lex = searcher.search(query, engine=SEMANTIC, k=n), searcher.search(query, engine=BM25, k=n)
    expected: dict[str, float] = {}
    for resp in (sem, lex):
        for hit in resp.hits:
            expected[hit.job.job_id] = expected.get(hit.job.job_id, 0.0) + 1.0 / (60 + hit.rank)
    manual = [j for j, _ in sorted(expected.items(), key=lambda kv: (-kv[1], kv[0]))]
    hyb = searcher.search(query, engine=HYBRID, k=n)
    assert ids(hyb) == manual and hyb.hits[0].score_kind == "rrf"
    assert hyb.hits[0].score == pytest.approx(expected[manual[0]])
    assert [h.rank for h in hyb.hits] == list(range(1, len(manual) + 1))


def test_hybrid_ranks_a_job_found_by_both_engines_above_one_found_by_a_single_engine(searcher):
    hyb = searcher.search("kafka streaming pipelines", engine=HYBRID, k=13)
    assert ids(hyb)[0] == "arbeitnow:syn-11"  # the only listing that mentions Kafka: top of BOTH rankings


def test_hybrid_uses_the_same_eligible_set_and_honours_filters(searcher):
    flt = JobFilter(job_types=("internship",))
    hyb = searcher.search("data", engine=HYBRID, k=20, flt=flt)
    assert set(ids(hyb)) == {"arbeitnow:syn-02", "arbeitnow:syn-10", "arbeitnow:syn-13"} and hyb.eligible_jobs == 3
    assert searcher.search("data", engine=HYBRID, flt=JobFilter(locations=("Atlantis",))).hits == []


def test_allowed_ids_restrict_candidates_before_ranking_for_every_engine(searcher):
    allowed = {"arbeitnow:syn-03", "arbeitnow:syn-11"}
    for engine in ENGINES:
        resp = searcher.search("data engineer pipelines", engine=engine, k=10, allowed_ids=allowed)
        assert set(ids(resp)) <= allowed and resp.eligible_jobs == 2


def test_hybrid_company_cap_is_presentation_only(searcher):
    plain = ids(searcher.search("engineer data machine learning", engine=HYBRID, k=13))
    capped = searcher.search("engineer data machine learning", engine=HYBRID, k=13, max_per_company=1)
    assert len(capped.hits) < len(plain) and capped.hidden_by_cap > 0
    assert ids(searcher.search("engineer data machine learning", engine=HYBRID, k=13)) == plain


# ---- evaluation with three systems -----------------------------------------------------------------


def queries_file(tmp_path, items):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"queries": items}), encoding="utf-8")
    return ev.load_queries(path)


def grade_all(rows, qid, grader):
    for r in rows:
        if r["query_id"] == qid:
            r["grade"] = str(grader(r["job_id"]))


def test_queries_can_ask_for_an_exhaustive_pool(searcher, tmp_path):
    queries = queries_file(tmp_path, [{"id": "a", "text": "stage analyste", "filters": {"job_types": ["internship"]}, "exhaustive": True},
                                      {"id": "b", "text": "stage analyste", "filters": {"job_types": ["internship"]}}])  # fmt: skip
    assert [q.exhaustive for q in queries] == [True, False]
    rows, _ = ev.build_pool(searcher, queries, [], depth=1)
    count = lambda qid: sum(r["query_id"] == qid for r in rows)  # noqa: E731
    assert count("a") == 3  # every eligible internship, although depth=1
    assert count("b") < 3


def test_all_three_engines_are_pooled_and_scored(searcher, tmp_path):
    queries = queries_file(tmp_path, [{"id": "q", "text": "spark airflow pipelines"}])
    rows, _ = ev.build_pool(searcher, queries, [], depth=5)
    grade_all(rows, "q", lambda j: 2 if j.endswith("syn-03") else 0)
    report = ev.evaluate(searcher, queries, rows)
    assert set(report.summary) == {"semantic", "bm25", "hybrid"}
    assert report.summary["hybrid"]["ndcg@10"] == pytest.approx(1.0)  # the one relevant job is ranked first
    assert {p["metric"] for p in report.pairwise.values()} == {"ndcg", "p5(grade>=1)", "mrr(grade>=1)"}
    assert "hybrid_vs_bm25:ndcg" in report.pairwise and report.pairwise["hybrid_vs_bm25:ndcg"]["ci95"] is None  # one query: no interval


def test_recall_is_reported_only_for_queries_whose_eligible_jobs_are_all_judged(searcher, tmp_path):
    queries = queries_file(tmp_path, [
        {"id": "complete", "text": "stage analyste de données", "filters": {"job_types": ["internship"]}, "exhaustive": True},
        {"id": "partial", "text": "data"}])  # fmt: skip
    rows, _ = ev.build_pool(searcher, queries, [], depth=3)
    grade_all(rows, "complete", lambda j: 2 if j.endswith(("syn-02", "syn-10")) else 0)
    grade_all(rows, "partial", lambda j: 1)
    report = ev.evaluate(searcher, queries, rows)
    by_id = {q.query_id: q for q in report.queries}
    assert by_id["complete"].complete and not by_id["partial"].complete and report.complete_queries == 1
    for system in ("semantic", "bm25", "hybrid"):
        assert report.summary[system]["recall@10(complete,grade>=2)"] == pytest.approx(1.0)  # 2 relevant, both in the top 10 of 3 eligible
        assert report.summary[system]["recall_queries(complete)"] == 1
        assert by_id["partial"].systems[system].full_recall == {}  # never computed without complete judgments
    assert "uses only the 1 query" in ev.format_report(report)


def test_recall_line_says_not_reported_when_no_query_is_complete(searcher, tmp_path):
    queries = queries_file(tmp_path, [{"id": "q", "text": "data"}])
    rows, _ = ev.build_pool(searcher, queries, [], depth=3)
    grade_all(rows, "q", lambda j: 1)
    report = ev.evaluate(searcher, queries, rows)
    assert not any(k.startswith("recall@") for k in report.summary["semantic"]) and "NOT reported" in ev.format_report(report)
    assert any(k.startswith("pooled_recall@") for k in report.summary["semantic"])  # still available, and labelled as pooled


def test_bootstrap_interval_needs_enough_queries_and_is_reproducible():
    assert ev.bootstrap_ci([0.1, 0.2, 0.3, 0.4]) is None
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lo, hi = ev.bootstrap_ci(values)
    assert lo <= sum(values) / len(values) <= hi and (lo, hi) == ev.bootstrap_ci(values)
    assert ev.bootstrap_ci([0.2] * 6) == pytest.approx((0.2, 0.2))


def test_pairwise_counts_wins_ties_and_losses():
    def qe(a, b):
        sr = lambda n: ev.SystemResult(n, {1: 0.0}, {1: 0.0}, {1: None}, 1.0, 10)  # noqa: E731
        return ev.QueryEval("q", "t", "scored", systems={"hybrid": sr(a), "bm25": sr(b)})

    result = ev.pairwise_comparison([qe(0.9, 0.5), qe(0.5, 0.5), qe(0.2, 0.6), qe(0.7, 0.1)], "hybrid", "bm25")
    assert (result["wins"], result["ties"], result["losses"]) == (2, 1, 1)
    assert result["mean_diff"] == pytest.approx((0.4 + 0 - 0.4 + 0.6) / 4)


def test_baselines_record_and_flag_regressions(searcher, tmp_path):
    queries = queries_file(tmp_path, [{"id": "q", "text": "spark airflow pipelines"}])
    rows, _ = ev.build_pool(searcher, queries, [], depth=5)
    grade_all(rows, "q", lambda j: 2 if j.endswith("syn-03") else 0)
    report = ev.evaluate(searcher, queries, rows)
    payload = ev.baseline_payload(report, {"model": "m"})
    assert payload["meta"] == {"model": "m"} and payload["scored_queries"] == ["q"]
    assert ev.compare_to_baseline(report, payload) == ["no metric changed"]
    worse = json.loads(json.dumps(payload))
    worse["summary"]["hybrid"]["ndcg@10"] = 1.0
    grade_all(rows, "q", lambda j: 0 if j.endswith("syn-03") else 2)
    degraded = ev.evaluate(searcher, queries, rows)
    assert any("WORSE" in line for line in ev.compare_to_baseline(degraded, payload))


def test_judgments_are_written_atomically_and_old_columns_survive(tmp_path):
    path = tmp_path / "j.csv"
    ev.write_judgments(path, [{"query_id": "q", "job_id": "j", "grade": "2", "title": "T", "notes": "n"}])
    assert not path.with_suffix(".csv.tmp").exists() and ev.read_judgments(path)[0]["notes"] == "n"


# ---- the labelling tool -------------------------------------------------------------------------


class Script:
    """Scripted terminal: answers come from a list; everything shown is recorded."""

    def __init__(self, *answers):
        self.answers, self.shown, self.prompts = list(answers), [], []

    def ask(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("the tool asked for more input than the test provided: " + prompt)
        return self.answers.pop(0)

    def show(self, text):
        self.shown.append(text)


def make_rows():
    return [{"query_id": "q1", "job_id": f"arbeitnow:syn-{n:02d}", "grade": "", "title": f"Job {n}", "notes": ""} for n in (1, 2, 3)]


def run(rows, script, saves=None, fields=(labeling.GRADE_FIELD,), select=lambda r: True):
    saved = saves if saves is not None else []
    return labeling.run_labeling(
        rows, list(fields), lambda r, d, t: f"CARD {r['job_id']} {d}/{t}", lambda rs: saved.append([dict(r) for r in rs]),
        select=select, ask=script.ask, show=script.show)  # fmt: skip


def test_labels_are_saved_after_every_answer():
    rows, saves = make_rows(), []
    summary = run(rows, Script("2", "0", "1"), saves)
    assert [r["grade"] for r in rows] == ["2", "0", "1"] and summary.labelled == 3 and summary.remaining == 0
    assert [sum(1 for r in s if r["grade"]) for s in saves[:3]] == [1, 2, 3]  # progress was persisted card by card


def test_quit_keeps_progress_and_a_second_session_resumes_at_the_next_card():
    rows = make_rows()
    summary = run(rows, Script("1", "q"))
    assert summary.quit_early and [r["grade"] for r in rows] == ["1", "", ""] and summary.remaining == 2
    script = Script("2", "2")
    resumed = run(rows, script)
    assert resumed.labelled == 2 and [r["grade"] for r in rows] == ["1", "2", "2"]
    assert any("syn-02" in shown for shown in script.shown) and not any("syn-01" in shown for shown in script.shown)


def test_skip_back_note_and_invalid_keys():
    rows = make_rows()
    script = Script("x", "n", "needs a second look", "s", "2", "b", "1", "q")
    summary = run(rows, script)
    # 'x' is invalid and re-asks; the note is stored; card 1 skipped; card 2 graded 2, then 'b' returns to card 2 -> regraded 1
    assert rows[0]["notes"] == "needs a second look" and rows[0]["grade"] == ""
    assert rows[1]["grade"] == "1" and summary.skipped == 1
    assert any("not relevant" in shown for shown in script.shown)  # the legend is shown after an invalid key


def test_selection_limits_the_cards_and_finished_work_is_not_asked_again():
    rows = make_rows() + [{"query_id": "q2", "job_id": "arbeitnow:syn-04", "grade": "", "title": "Job 4", "notes": ""}]
    run(rows, Script("2"), select=lambda r: r["query_id"] == "q2")
    assert [r["grade"] for r in rows] == ["", "", "", "2"]
    script = Script()
    assert run(rows, script, select=lambda r: r["query_id"] == "q2").labelled == 0 and script.prompts == []


def test_multi_field_cards_require_every_answer_before_a_card_counts_as_done():
    fields = (labeling.GRADE_FIELD, labeling.ROLE_FIELD, labeling.ELIGIBLE_FIELD)
    rows = [{"query_id": "c", "job_id": "j1", "grade": "", "role_ok": "", "eligible_ok": "", "title": "T", "notes": ""}]
    run(rows, Script("2", "y", "q"), fields=fields)  # quit during the last question: nothing is recorded for the card
    assert rows[0]["grade"] == "" and rows[0]["eligible_ok"] == ""
    run(rows, Script("2", "y", "n"), fields=fields)
    assert (rows[0]["grade"], rows[0]["role_ok"], rows[0]["eligible_ok"]) == ("2", "y", "n")


def test_retrieval_card_is_blinded_and_helpful(searcher, tmp_path):
    queries = {q.id: q for q in queries_file(tmp_path, [{"id": "q1", "lang": "fr", "text": "stage analyste de données"}])}
    card = labeling.retrieval_card(queries, {j["id"]: j for j in jobs()}, "rubric line")
    text = card({"query_id": "q1", "job_id": "arbeitnow:syn-02", "title": "Data Analyst Internship"}, 0, 5)
    assert "stage analyste de données" in text and "Data Analyst Internship" in text and "https://fixtures.invalid/arbeitnow/syn-02" in text
    assert "analyse de données" in text  # the listing line that shares the query's words is surfaced
    for forbidden in ("semantic", "bm25", "hybrid", "rank", "score", "cosine"):
        assert forbidden not in text.lower()


def test_label_cli_end_to_end_with_typed_answers(tmp_path, monkeypatch, capsys):
    from job_intel import cli
    from search_data import write_jobs

    data = tmp_path / "data"
    write_jobs(data / "processed" / "jobs.jsonl")
    ev_dir = tmp_path / "eval"
    ev_dir.mkdir()
    (ev_dir / "queries.json").write_text(json.dumps({"queries": [{"id": "q1", "text": "spark airflow"}]}), encoding="utf-8")
    ev.write_judgments(ev_dir / "judgments.csv", [{"query_id": "q1", "job_id": f"arbeitnow:syn-0{n}", "grade": "", "title": f"J{n}", "notes": ""} for n in (3, 1)])
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(ev_dir))
    answers = iter(["2", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["label", "retrieval", "--data-dir", str(data)]) == 0
    assert [r["grade"] for r in ev.read_judgments(ev_dir / "judgments.csv")] == ["2", "0"]
    assert "2 labelled" in capsys.readouterr().out
    assert cli.main(["label", "retrieval", "--data-dir", str(data), "--stats"]) == 0
    assert "2/2 done" in capsys.readouterr().out


def test_label_cli_explains_what_to_do_when_there_is_no_pool(tmp_path, monkeypatch, capsys):
    from job_intel import cli
    from search_data import write_jobs

    write_jobs(tmp_path / "data" / "processed" / "jobs.jsonl")
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(tmp_path / "eval"))
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "queries.json").write_text(json.dumps({"queries": [{"id": "q", "text": "x"}]}), encoding="utf-8")
    assert cli.main(["label", "retrieval", "--data-dir", str(tmp_path / "data")]) == 2
    assert "jobintel eval pool" in capsys.readouterr().err
