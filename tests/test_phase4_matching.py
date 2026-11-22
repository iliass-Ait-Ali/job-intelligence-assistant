"""Phase 4: filter-before-retrieve, label-free diagnostics, audit strata, skill-claim checks, human-judgment workflows."""

import itertools
import shutil
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from match_data import match_jobs  # noqa: E402
from search_data import write_jobs  # noqa: E402

from job_intel import cli, labeling  # noqa: E402
from job_intel.label_cli import default_match_judgments  # noqa: E402
from job_intel.match import diagnostics  # noqa: E402
from job_intel.match import evaluate as me  # noqa: E402
from job_intel.match.matcher import evaluate_constraints, match_candidate  # noqa: E402
from job_intel.match.models import Preferences  # noqa: E402
from job_intel.match.requirements import extract_requirements  # noqa: E402
from job_intel.search.errors import EvalError  # noqa: E402
from job_intel.search.filters import IndexedJob  # noqa: E402

CV_SRC = Path(__file__).parent.parent / "eval" / "synthetic_cvs"


@pytest.fixture
def candidates(tmp_path):
    shutil.copytree(CV_SRC, tmp_path / "cvs")
    return me.load_candidates(tmp_path / "cvs")


@pytest.fixture
def student(candidates):
    return next(c for c in candidates if "student" in c.id)


def jid(n):
    return f"arbeitnow:syn-{n:02d}"


# ---- hard constraints are applied BEFORE retrieval --------------------------------------------------


def test_prefiltering_reaches_every_eligible_job_even_with_a_small_pool(student_profile, match_searcher):
    prefs = Preferences(job_types=["internship"], hard_constraints=["job_type"], max_results=50)
    legacy = match_candidate(student_profile, prefs, match_searcher, pool_size=3, prefilter=False)
    now = match_candidate(student_profile, prefs, match_searcher, pool_size=3, prefilter=True)
    eligible = {jid(1), jid(4), jid(5)}  # internship, internship, and one whose type is unknown
    assert {m.job.job_id for m in now.matches} == eligible and now.eligible_total == 3
    assert {m.job.job_id for m in legacy.matches} < eligible  # retrieve-then-filter spent its 3 slots on ineligible jobs


def test_prefiltering_does_not_loosen_anything(student_profile, match_searcher):
    prefs = Preferences(job_types=["internship"], locations=["France"], max_results=50)
    old = match_candidate(student_profile, prefs, match_searcher, prefilter=False)
    new = match_candidate(student_profile, prefs, match_searcher, prefilter=True)
    assert {m.job.job_id for m in new.matches} == {m.job.job_id for m in old.matches}  # same decisions when the pool is not the bottleneck
    assert {m.job.job_id for m in old.excluded} <= {m.job.job_id for m in new.excluded}  # and every old exclusion still stands
    assert len(new.excluded) + new.eligible_total == new.corpus_size == 10  # the whole index is accounted for


def test_a_title_that_says_internship_satisfies_an_internship_preference():
    def constraint(title, prefs):
        job = IndexedJob("s:1", title, "Acme", "Paris", None, False, "arbeitnow", "https://x.invalid/1", None)
        reqs = extract_requirements(title, "Requirements:\n- Python")
        from job_intel.match.models import CandidateProfile

        return next(c for c in evaluate_constraints(job, reqs, CandidateProfile(), prefs) if c.name == "job_type")

    wants_internship = Preferences(job_types=["internship"])
    assert constraint("Stage - Data Analyst", wants_internship).status == "satisfied"
    assert constraint("Data Analyst Intern", wants_internship).status == "satisfied"
    assert constraint("Data Analyst", wants_internship).status == "unknown"  # nothing in the title: still unknown, not a mismatch
    assert constraint("Senior Data Analyst", wants_internship).status == "violated"
    assert constraint("Stage - Data Analyst", Preferences(job_types=["full_time"])).status == "violated"


# ---- label-free diagnostics ----------------------------------------------------------------------


def test_diagnostics_explain_the_exclusions(student, match_searcher):
    d = diagnostics.diagnose(student.id, student.profile, student.prefs, match_searcher)
    assert d.corpus == 10 and d.eligible_in_corpus == 2 and d.excluded_total == 8  # internship + France: syn-01 and the unknown-type, unknown-place syn-05
    assert sum(d.excluded_by.values()) == 8 and "job_type" in " ".join(d.excluded_by)
    assert d.kept_with_unknown_hard == {"job_type": 1, "location": 1}
    assert d.current.eligible_in_pool == d.current.eligible_in_corpus == 2 and d.current.eligible_recall == 1.0
    text = diagnostics.format_diagnostics([d])
    assert "survive the hard constraints: 2" in text and "kept only because the value is unknown" in text
    assert diagnostics.metrics([d]) == {f"{student.id}.eligible_in_corpus": 2.0, f"{student.id}.eligible_recall_in_pool": 1.0}


# ---- pool strata and the audit of the system's decisions ---------------------------------------------


def test_audit_pool_adds_excluded_and_unseen_jobs_but_hides_the_strata(student, match_searcher):
    meta: dict = {}
    rows, stats = me.build_pool(match_searcher, [student], [], depth=1, meta=meta, audit=True, excluded_audit=3, unseen_sample=2)
    strata = {jid_: info["strata"] for jid_, info in meta[student.id].items()}
    assert any("excluded_audit" in s for s in strata.values()) and stats["audit_items"] >= 3
    excluded_ids = {j for j, info in meta[student.id].items() if "excluded_audit" in info["strata"]}
    assert all(meta[student.id][j]["eligible"] is False for j in excluded_ids)
    assert set(rows[0]) == {"query_id", "job_id", "grade", "title", "notes"}  # nothing about strata/system in the rows the labeller sees
    assert all(r["grade"] == "" for r in rows)
    again, new = me.build_pool(match_searcher, [student], rows, depth=1, meta=meta, audit=True, excluded_audit=3, unseen_sample=2)
    assert again == rows and new["new_items"] == 0  # idempotent


def graded_pool(student, searcher):
    meta: dict = {}
    rows, _ = me.build_pool(searcher, [student], [], depth=3, meta=meta, audit=True, excluded_audit=8, unseen_sample=2)
    # TEST-ONLY stand-in for the human. Grades: syn-01 strong, syn-05 partial, syn-04 (excluded) partial, rest irrelevant.
    marks = {jid(1): ("2", "y", "y"), jid(5): ("1", "n", "y"), jid(4): ("1", "y", "y")}
    for r in rows:
        grade, role_ok, eligible_ok = marks.get(r["job_id"], ("0", "", "n"))
        r.update(grade=grade, role_ok=role_ok, eligible_ok=eligible_ok)
    return rows, meta


def test_audit_metrics_count_incorrect_exclusions_role_agreement_and_missing_jobs(student, match_searcher):
    rows, meta = graded_pool(student, match_searcher)
    report = me.evaluate_matching(match_searcher, [student], rows, meta)
    a = report.audit
    assert a["system_excluded_human_acceptable"]["k"] == 1 and a["system_excluded_human_acceptable"]["n"] == 8  # syn-04 was excluded but the human says acceptable
    assert a["system_eligible_human_unacceptable"]["k"] == 0
    assert a["eligibility_agreement"]["k"] == 9 and a["eligibility_agreement"]["n"] == 10
    assert (a["role_relevance_agreement"]["k"], a["role_relevance_agreement"]["n"]) == (2, 3)  # syn-01 match/y ok, syn-05 no_match/n ok, syn-04 no_match but human y -> disagreement
    assert a["excluded_audit_relevant(grade>=1)"]["k"] == 1
    lo, hi = a["system_excluded_human_acceptable"]["ci95"]
    assert 0 <= lo < 1 / 8 < hi <= 1
    text = me.format_match_report(report)
    assert "incorrect exclusions" in text and "1/8 = 12%" in text


def test_ranking_metrics_use_only_the_rankable_pairs(student, match_searcher):
    rows, meta = graded_pool(student, match_searcher)
    report = me.evaluate_matching(match_searcher, [student], rows, meta)
    # the audit pairs (excluded jobs) are not part of the ideal ranking: the best eligible job (grade 2) is ranked first
    assert report.summary["pipeline"]["ndcg@10"] == pytest.approx(1.0)


def test_candidates_are_scored_once_their_rankable_pairs_are_graded_even_if_audit_pairs_are_not(student, match_searcher):
    rows, meta = graded_pool(student, match_searcher)
    for r in rows:
        if "excluded_audit" in meta[student.id][r["job_id"]]["strata"] and "pipeline" not in meta[student.id][r["job_id"]]["strata"]:
            r.update(grade="", role_ok="", eligible_ok="")
    assert me.evaluate_matching(match_searcher, [student], rows, meta).candidates[0].status == "scored"


def test_wilson_interval():
    assert me.wilson_interval(0, 0) is None
    lo, hi = me.wilson_interval(5, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.237, 0.763)
    lo, hi = me.wilson_interval(0, 5)
    assert lo == 0.0 and 0.4 < hi < 0.5 and me.wilson_interval(5, 5)[1] == 1.0
    assert me.proportion(3, 4)["rate"] == 0.75


def test_match_judgment_files_accept_old_and_new_layouts(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("query_id,job_id,grade,title,notes\nc,j,2,T,\n", encoding="utf-8-sig")
    rows = me.read_match_judgments(old)
    assert rows[0]["grade"] == "2"
    rows[0]["role_ok"] = "y"
    me.write_match_judgments(old, rows)
    assert me.read_match_judgments(old)[0]["role_ok"] == "y" and old.read_text(encoding="utf-8-sig").splitlines()[0] == ",".join(me.MATCH_COLUMNS)
    old.write_text("query_id,job_id,grade,role_ok,eligible_ok,title,notes\nc,j,1,maybe,,T,\n", encoding="utf-8-sig")
    with pytest.raises(EvalError, match="role_ok"):
        me.read_match_judgments(old)


# ---- skill-claim verification ------------------------------------------------------------------------


def test_skill_pool_is_a_blinded_stratified_sample_and_idempotent(candidates, match_searcher):
    rows = me.build_skill_pool(match_searcher, candidates, [], n_exact=4, n_transferable=2)
    assert {r["claimed"] for r in rows} <= {"exact", "transferable"} and 0 < len(rows) <= 6
    assert all(r["correct"] == "" and r["cv_evidence"] and r["job_evidence"] for r in rows)
    assert me.build_skill_pool(match_searcher, candidates, [], n_exact=4, n_transferable=2) == rows  # seeded
    grown = me.build_skill_pool(match_searcher, candidates, rows, n_exact=40, n_transferable=20)
    assert grown[: len(rows)] == rows and len({(r["candidate_id"], r["job_id"], r["skill"]) for r in grown}) == len(grown)


def test_skill_scores_come_only_from_human_marks(tmp_path):
    rows = [{"candidate_id": "c", "job_id": "j", "skill": s, "claimed": claimed, "cv_evidence": "x", "job_evidence": "y", "correct": mark, "notes": ""}
            for s, claimed, mark in [("Python", "exact", "y"), ("SQL", "exact", "y"), ("R", "exact", "n"), ("Tableau", "transferable", "y"), ("Go", "exact", "")]]  # fmt: skip
    score = me.score_skills(rows)
    assert score["judged"] == 4 and score["exact_precision"]["k"] == 2 and score["exact_precision"]["n"] == 3
    assert score["transferable_precision"]["rate"] == 1.0
    with pytest.raises(EvalError, match="no metrics yet"):
        me.score_skills([{**r, "correct": ""} for r in rows])
    path = tmp_path / "s.csv"
    me.write_skill_rows(path, rows)
    assert [r["correct"] for r in me.read_skill_rows(path)] == ["y", "y", "n", "y", ""]
    path.write_text("candidate_id,job_id,skill,claimed,cv_evidence,job_evidence,correct,notes\nc,j,s,exact,,,maybe,\n", encoding="utf-8")
    with pytest.raises(EvalError, match="'correct'"):
        me.read_skill_rows(path)


def test_cards_show_the_candidate_and_never_the_systems_verdict(student, match_searcher):
    records = {j["id"]: j for j in match_jobs()}
    card = labeling.matching_card({student.id: student}, records, "rubric")
    text = card({"query_id": student.id, "job_id": jid(1), "title": "Data Scientist Internship"}, 0, 10)
    assert "STATED PREFERENCES" in text and "internship" in text and "France" in text and "Python" in text
    for forbidden in ("transferable", "excluded", "rank", "score", "evidence level", "retrieval", "pipeline"):
        assert forbidden not in text.lower(), forbidden  # none of the system's outputs may leak into the card
    skill = labeling.skill_card({"candidate_id": "c", "claimed": "transferable", "skill": "TensorFlow", "cv_evidence": "PyTorch work", "job_evidence": "TensorFlow required"}, 0, 5)
    assert "RELATED" in skill and "PyTorch work" in skill


def test_real_cv_folders_keep_their_files_under_data(tmp_path):
    from job_intel.config import Settings

    settings = Settings(user_agent="t", data_dir=tmp_path / "data", eval_dir=tmp_path / "eval")
    assert default_match_judgments(settings, settings.eval_dir / "synthetic_cvs") == tmp_path / "eval" / "match_judgments.csv"
    private = default_match_judgments(settings, tmp_path / "private" / "my_cvs")
    assert private == tmp_path / "data" / "eval" / "match_judgments_my_cvs.csv"  # git-ignored location


# ---- the whole human workflow through the CLI --------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch, match_searcher):
    from job_intel.search import service

    monkeypatch.setattr(service, "open_searcher", lambda settings, **kw: (match_searcher, []))
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("JOBINTEL_DATA_DIR", str(tmp_path / "data"))
    shutil.copytree(CV_SRC, tmp_path / "eval" / "synthetic_cvs")
    write_jobs(tmp_path / "data" / "processed" / "jobs.jsonl", match_jobs())
    return tmp_path


def test_cli_diagnose_and_pool_with_audit(env, capsys):
    assert cli.main(["match-eval", "diagnose"]) == 0
    out = capsys.readouterr().out
    assert "filter-then-retrieve (now)" in out and "retrieve-then-filter (Phase 3)" in out
    assert cli.main(["match-eval", "pool"]) == 0
    assert (env / "eval" / "match_pool_meta.json").exists() and "exclusion / unseen-job audit" in capsys.readouterr().out
    assert cli.main(["label", "matching", "--stats"]) == 0
    assert "cv_en_student_data_science" in capsys.readouterr().out


def test_cli_human_labelling_then_evaluation(env, capsys, monkeypatch):
    assert cli.main(["match-eval", "pool"]) == 0
    capsys.readouterr()
    answers = itertools.cycle(["1", "y", "y"])  # TEST-ONLY stand-in for the human: grade 1, role ok, eligible ok
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["label", "matching"]) == 0
    capsys.readouterr()
    assert cli.main(["match-eval", "run"]) == 0
    out = capsys.readouterr().out
    assert "candidates fully judged" in out and "Audit of the system's decisions" in out and "Wilson" in out
    assert cli.main(["match-eval", "skills-pool"]) == 0
    assert cli.main(["match-eval", "skills-score"]) == 2  # nothing marked yet -> refuses
    assert "no metrics yet" in capsys.readouterr().err


def test_cli_extraction_command(capsys):
    assert cli.main(["match-eval", "extraction", "--compare-phase3"]) == 0
    out = capsys.readouterr().out
    assert "40 documents, 10 layouts" in out and "Phase 3 parser" in out and "says nothing about real CVs" in out


def test_cli_real_cv_folder_stays_out_of_tracked_locations(env, capsys):
    private = env / "private" / "mycv"
    private.mkdir(parents=True)
    shutil.copy(CV_SRC / "cv_en_ml_engineer.txt", private / "me.txt")  # stands in for a real CV (here: a synthetic one)
    (private / "me.prefs.json").write_text('{"job_types": ["full_time"]}', encoding="utf-8")
    assert cli.main(["match-eval", "pool", "--cvs", str(private)]) == 0
    capsys.readouterr()
    assert (env / "data" / "eval" / "match_judgments_mycv.csv").exists() and (env / "data" / "eval" / "match_pool_meta_mycv.json").exists()
    assert not (env / "eval" / "match_judgments.csv").exists()  # the synthetic pool file was not touched
    review = (env / "data" / "eval" / "match_review_mycv.md").read_text(encoding="utf-8")
    assert "Sam Placeholder" in review  # the CV text lives only in the git-ignored data/ folder
