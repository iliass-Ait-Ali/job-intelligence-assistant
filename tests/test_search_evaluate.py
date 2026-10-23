import json
import math

import pytest

pytest.importorskip("chromadb")

from search_data import jobs  # noqa: E402

from job_intel.search import evaluate as ev  # noqa: E402
from job_intel.search.document import ChunkingConfig  # noqa: E402
from job_intel.search.embedder import HashingEmbedder  # noqa: E402
from job_intel.search.errors import EvalError  # noqa: E402
from job_intel.search.filters import JobFilter  # noqa: E402
from job_intel.search.index import IndexConfig, JobIndex  # noqa: E402
from job_intel.search.retrieve import Searcher  # noqa: E402

EMB = HashingEmbedder(max_seq_length=128)


# ---- metric arithmetic, checked by hand ----------------------------------------------------


def test_dcg_and_ndcg_by_hand():
    # ranked grades [2, 0, 1]: DCG = 3/log2(2) + 0/log2(3) + 1/log2(4) = 3.5
    assert ev.dcg([2, 0, 1], 10) == pytest.approx(3.5)
    # ideal order of the judged pool [2, 1, 0, 0]: 3 + 1/log2(3)
    ideal = 3 + 1 / math.log2(3)
    assert ev.ndcg_at_k([2, 0, 1], [2, 1, 0, 0], 10) == pytest.approx(3.5 / ideal)
    assert ev.ndcg_at_k([2, 1], [2, 1], 10) == pytest.approx(1.0)
    assert ev.ndcg_at_k([0, 0], [0, 0], 10) == 0.0  # nothing relevant judged: defined as 0, never NaN
    assert ev.dcg([2, 2, 2], 2) == pytest.approx(3 + 3 / math.log2(3))  # cut-off respected


def test_precision_uses_a_fixed_denominator_and_explicit_thresholds():
    ranked = [2, 1, 0, 1]  # only 4 returned, k=5
    assert ev.precision_at_k(ranked, 5, threshold=1) == pytest.approx(3 / 5)
    assert ev.precision_at_k(ranked, 5, threshold=2) == pytest.approx(1 / 5)
    assert ev.precision_at_k([0] * 5, 5, 1) == 0.0


def test_mrr_at_k_by_threshold():
    ranked = [0, 1, 2]
    assert ev.mrr_at_k(ranked, 10, threshold=1) == pytest.approx(1 / 2)
    assert ev.mrr_at_k(ranked, 10, threshold=2) == pytest.approx(1 / 3)
    assert ev.mrr_at_k([0, 0, 2], 2, threshold=2) == 0.0  # relevant item beyond the cut-off
    assert ev.mrr_at_k([], 10, 1) == 0.0


def test_pooled_recall_is_relative_to_the_judged_pool_and_undefined_without_relevant_items():
    assert ev.pooled_recall_at_k([2, 0, 1], [2, 2, 1, 0], 10, threshold=2) == pytest.approx(1 / 2)
    assert ev.pooled_recall_at_k([2, 0, 1], [2, 2, 1, 0], 10, threshold=1) == pytest.approx(2 / 3)
    assert ev.pooled_recall_at_k([0, 0], [0, 0, 0], 10, threshold=1) is None


# ---- files --------------------------------------------------------------------------------


def write_queries(path, queries):
    path.write_text(json.dumps({"queries": queries}), encoding="utf-8")
    return path


def test_load_queries_validates(tmp_path):
    good = write_queries(tmp_path / "q.json", [{"id": "a", "text": "x", "filters": {"job_types": ["internship"]}}])
    assert ev.load_queries(good)[0].filters == JobFilter(job_types=("internship",))
    with pytest.raises(EvalError, match="duplicate"):
        ev.load_queries(write_queries(tmp_path / "d.json", [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]))
    with pytest.raises(EvalError, match="'id' and 'text'"):
        ev.load_queries(write_queries(tmp_path / "e.json", [{"id": "a"}]))
    with pytest.raises(EvalError, match="not found"):
        ev.load_queries(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("not json")
    with pytest.raises(EvalError, match="queries"):
        ev.load_queries(tmp_path / "bad.json")


def test_read_judgments_validates_grades(tmp_path):
    path = tmp_path / "j.csv"
    path.write_text("query_id,job_id,grade,title,notes\nq1,a,2,T,\nq1,b,,T,\n", encoding="utf-8-sig")
    rows = ev.read_judgments(path)
    assert ev.progress(rows) == (1, 2) and ev.grade_lookup(rows) == {("q1", "a"): 2, ("q1", "b"): None}
    path.write_text("query_id,job_id,grade\nq1,a,3\n", encoding="utf-8")
    with pytest.raises(EvalError, match="line 2"):
        ev.read_judgments(path)
    path.write_text("query_id,job_id,grade\nq1,a,yes\n", encoding="utf-8")
    with pytest.raises(EvalError):
        ev.read_judgments(path)
    path.write_text("query_id,title\nq1,x\n", encoding="utf-8")
    with pytest.raises(EvalError, match="missing column"):
        ev.read_judgments(path)
    assert ev.read_judgments(tmp_path / "absent.csv") == []


# ---- pool + evaluation on a synthetic index -------------------------------------------------

QUERIES = [
    {"id": "q1", "text": "spark airflow data pipelines", "lang": "en"},
    {"id": "q2", "text": "stage reporting power bi", "lang": "fr", "filters": {"job_types": ["internship"]}},
]


@pytest.fixture
def setup(tmp_path):
    index = JobIndex.open_for_build(tmp_path / "idx", IndexConfig.build(EMB, ChunkingConfig(100, 20)))
    index.sync(jobs(), EMB)
    searcher = Searcher(index, EMB, {j["id"]: j for j in jobs()})
    queries = ev.load_queries(write_queries(tmp_path / "q.json", QUERIES))
    return searcher, queries, tmp_path / "judgments.csv"


def test_pool_is_blinded_ungraded_deterministic_and_respects_query_filters(setup):
    searcher, queries, _ = setup
    rows, stats = ev.build_pool(searcher, queries, [], depth=5)
    assert stats["new_items"] == len(rows) > 0
    assert all(r["grade"] == "" for r in rows)  # the tool never grades
    assert set(rows[0]) == set(ev.JUDGMENT_COLUMNS)  # no system names, ranks or scores leak into the sheet
    assert {r["job_id"] for r in rows if r["query_id"] == "q2"} <= {"arbeitnow:syn-02", "arbeitnow:syn-10", "arbeitnow:syn-13"}
    again, _ = ev.build_pool(searcher, queries, [], depth=5)
    assert again == rows  # same seed -> same order
    q1 = [r["job_id"] for r in rows if r["query_id"] == "q1"]
    assert len(q1) == len(set(q1))


def test_rerunning_the_pool_keeps_grades_and_adds_only_new_items(setup, tmp_path):
    searcher, queries, path = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=3)
    rows[0]["grade"] = "2"
    rows[0]["notes"] = "human note"
    ev.write_judgments(path, rows)
    reread = ev.read_judgments(path)
    merged, stats = ev.build_pool(searcher, queries, reread, depth=3)
    assert stats["new_items"] == 0 and merged[0]["grade"] == "2" and merged[0]["notes"] == "human note"
    deeper, stats = ev.build_pool(searcher, queries, merged, depth=8)
    assert stats["new_items"] > 0
    for q in queries:  # within each query, existing rows come first and are untouched
        before = [r for r in merged if r["query_id"] == q.id]
        assert [r for r in deeper if r["query_id"] == q.id][: len(before)] == before


def test_review_sheet_contains_job_text_for_judging(setup, tmp_path):
    searcher, queries, _ = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=3)
    sheet = tmp_path / "data" / "eval" / "review.md"
    ev.write_review_sheet(sheet, queries, rows, {j["id"]: j for j in jobs()}, "rubric text")
    text = sheet.read_text(encoding="utf-8")
    assert "rubric text" in text and "## q1" in text and "https://fixtures.invalid/" in text


def grade_everything(rows, qid, grader):
    for r in rows:
        if r["query_id"] == qid:
            r["grade"] = str(grader(r["job_id"]))


def test_metrics_are_withheld_until_humans_have_graded_something(setup):
    searcher, queries, _ = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=5)
    with pytest.raises(EvalError, match="no metrics yet"):
        ev.evaluate(searcher, queries, rows)
    rows[0]["grade"] = "1"  # one grade is not a complete query
    with pytest.raises(EvalError, match="no metrics yet"):
        ev.evaluate(searcher, queries, rows)
    with pytest.raises(EvalError):
        ev.evaluate(searcher, queries, [])


def test_scores_only_fully_graded_queries_and_reports_pending_ones(setup):
    searcher, queries, _ = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=5)
    # TEST-ONLY synthetic grades, standing in for a human: the Spark/Airflow data engineer is the right answer to q1.
    grade_everything(rows, "q1", lambda job_id: 2 if job_id == "arbeitnow:syn-03" else 0)
    report = ev.evaluate(searcher, queries, rows)
    assert [q.status for q in report.queries] == ["scored", "pending"]
    q1 = report.queries[0]
    assert q1.pool_size == q1.graded and q1.eligible_jobs == 13
    for system in (ev.SEMANTIC, ev.BM25):
        result = q1.systems[system]
        assert result.judged_at_k == pytest.approx(min(1.0, sum(1 for r in rows if r["query_id"] == "q1") and result.judged_at_k))
    assert report.summary["semantic"]["queries"] == 1
    # the best document for the query is ranked first by both lexical-overlap systems -> perfect scores
    assert q1.systems["semantic"].ndcg == pytest.approx(1.0) and q1.systems["bm25"].mrr[2] == pytest.approx(1.0)
    text = ev.format_report(report)
    assert "1/2 queries fully judged" in text and "q2 (pending" in text and "NOT full-corpus recall" in text


def test_unjudged_results_count_as_not_relevant_and_lower_judged_at_k(setup):
    searcher, queries, _ = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=3)  # pool depth 3 < evaluation depth 10
    grade_everything(rows, "q1", lambda job_id: 1)
    report = ev.evaluate(searcher, queries, rows)
    q1 = report.queries[0]
    assert all(r.judged_at_k < 1.0 for r in q1.systems.values())  # results below the pool depth have no judgment
    assert report.summary["semantic"]["judged@10"] < 1.0


def test_pooled_recall_excludes_queries_without_relevant_items(setup):
    searcher, queries, _ = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=5)
    grade_everything(rows, "q1", lambda job_id: 0)
    report = ev.evaluate(searcher, queries, rows)
    s = report.summary["semantic"]
    assert s["pooled_recall@10(grade>=1)"] is None and s["pooled_recall_queries(grade>=1)"] == 0
    assert s["ndcg@10"] == 0.0 and s["precision@5(grade>=1)"] == 0.0


def test_both_systems_are_evaluated_on_the_same_eligible_set_without_a_company_cap(setup):
    searcher, queries, _ = setup
    rows, _ = ev.build_pool(searcher, queries, [], depth=5)
    for qid in ("q1", "q2"):
        grade_everything(rows, qid, lambda job_id: 1)
    report = ev.evaluate(searcher, queries, rows)
    assert [q.eligible_jobs for q in report.queries] == [13, 3]
    assert sum(report.win_tie_loss.values()) == 2
    assert json.dumps(report.to_dict(), default=str)  # serializable
