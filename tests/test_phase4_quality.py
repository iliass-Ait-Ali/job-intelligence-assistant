"""Phase 4: parser fallbacks, extraction evaluation, stricter verifier, provenance, stress test, real-vs-mock tagging, regression."""

import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from job_intel import cli, regress  # noqa: E402
from job_intel.config import Settings  # noqa: E402
from job_intel.match import evaluate as me  # noqa: E402
from job_intel.match import extraction_eval as xe  # noqa: E402
from job_intel.match.cv_parser import parse_cv  # noqa: E402
from job_intel.match.matcher import match_candidate  # noqa: E402
from job_intel.match.models import Preferences  # noqa: E402
from job_intel.rag import audit  # noqa: E402
from job_intel.rag.evidence import build_evidence_pack  # noqa: E402
from job_intel.rag.explain import Strength, deterministic_explanation, explain_match, unsupported_proper_noun, verify_items  # noqa: E402
from job_intel.rag.ollama import LLMUnavailable, OllamaClient  # noqa: E402

TODAY = date(2026, 10, 23)
CV_SRC = Path(__file__).parent.parent / "eval" / "synthetic_cvs"


# ---- parser fallbacks (measured on synthetic CVs with known ground truth) -----------------------------


PROSE = "\n\n".join([
    "Alex Example is a data scientist.",
    "Junior Data Scientist - Initech Systems, May 2018 - Sep 2019. Built models with Python and SQL.",
    "Data Scientist - Globex Labs, Jul 2020 - present. Used scikit-learn and Docker.",
    "I hold a Master in Data Science (2012 - 2015).\nLanguages: English (fluent), French (native)",
])
# Limitation (documented): the fallback expects one dated entry per line / paragraph, not several entries in one run-on line.


def test_cv_without_recognised_headings_is_parsed_from_its_dates_and_degree_names_with_a_warning():
    p = parse_cv(PROSE, today=TODAY)
    assert [(e.title, e.start, e.end) for e in p.experiences] == [("Junior Data Scientist", "2018-05", "2019-09"), ("Data Scientist", "2020-07", "present")]
    assert p.highest_degree_level == "master" and {s.language for s in p.spoken_languages} == {"English", "French"}
    assert any("guessed from dates" in w for w in p.warnings)


def test_entries_whose_dates_sit_on_the_next_line_are_recovered_under_unknown_headings():
    text = "WHERE I WORKED\nData Engineer - Acme\nJan 2020 - Mar 2022\n- Built Spark pipelines\n\nTongues\nEnglish (native), German (B1)\n"
    p = parse_cv(text, today=TODAY)
    assert [(e.title, e.organization, e.start, e.end) for e in p.experiences] == [("Data Engineer", "Acme", "2020-01", "2022-03")]
    assert {s.language: s.level_text for s in p.spoken_languages} == {"English": "native", "German": "B1"}


def test_the_phase3_parser_behaviour_is_reproducible_for_comparison():
    legacy = parse_cv(PROSE, today=TODAY, fallbacks=False)
    assert legacy.experiences == [] and legacy.highest_degree_level is None  # what Phase 3 did with the same text


def test_sectioned_cvs_are_unaffected_by_the_fallbacks():
    text = (CV_SRC / "cv_en_student_data_science.txt").read_text(encoding="utf-8")
    assert parse_cv(text, today=TODAY) == parse_cv(text, today=TODAY, fallbacks=False)


def test_loose_language_lines_need_language_names_and_levels():
    assert not parse_cv("Python, SQL\nEnglish and cooking", today=TODAY).spoken_languages
    assert [s.language for s in parse_cv("SKILLS\nPython\nSomething\nEnglish (fluent), Spanish (basic)", today=TODAY).spoken_languages] == ["English", "Spanish"]


def test_extraction_corpus_is_deterministic_and_covers_ten_layouts():
    a, b = xe.generate_corpus(), xe.generate_corpus()
    assert [d.text for d in a] == [d.text for d in b] and len(a) == 40 and len({d.style for d in a}) == 10
    assert all(d.skills and d.experiences and d.degree_level for d in a)


def test_extraction_quality_and_the_measured_improvement_over_phase3():
    now, before = xe.evaluate_extraction(), xe.evaluate_extraction(fallbacks=False)
    for metric in ("skill_f1", "degree_level_accuracy", "experience_date_recall", "language_f1", "target_role_f1"):
        assert now.overall[metric] == pytest.approx(1.0), (metric, now.failures[:3])
    assert before.overall["degree_level_accuracy"] == pytest.approx(0.8) and before.overall["experience_date_recall"] == pytest.approx(0.8)
    assert before.per_style["no_headings"]["experience_date_recall"] == 0.0 and before.per_style["unusual_headings"]["language_f1"] == 0.0
    assert "says nothing about real CVs" in xe.format_extraction(now)


# ---- stricter verifier --------------------------------------------------------------------------------


@pytest.fixture
def student_pack(student_profile, match_searcher):
    result = match_candidate(student_profile, Preferences(hard_constraints=[]), match_searcher)
    match = next(m for m in result.matches if m.job.job_id.endswith("syn-01"))
    return match, result, build_evidence_pack(student_profile, match, match_searcher, result.queries)


def one(pack, claim):
    cv = next(s.id for s in pack.cv if "scikit-learn" in s.text)
    job = next(s.id for s in pack.job if "scikit-learn" in s.text)
    return Strength(claim=claim, cv_refs=[cv], job_refs=[job])


@pytest.mark.parametrize(("claim", "why"), [
    ("The candidate used scikit-learn while working at Google.", "Google"),
    ("The candidate is an expert in scikit-learn.", "expert"),
    ("She built scikit-learn models at Stanford.", "Stanford"),
    ("The candidate has advanced scikit-learn skills.", "advanced"),
])  # fmt: skip
def test_invented_employers_schools_and_skill_levels_are_rejected(student_pack, student_profile, claim, why):
    _, _, pack = student_pack
    kept, rejected = verify_items("strength", [one(pack, claim)], pack, student_profile)
    assert not kept and why.lower() in rejected[0].reason.lower()


@pytest.mark.parametrize("claim", [
    "The CV shows scikit-learn work, which the listing asks for.",
    "Le CV mentionne scikit-learn, demandé par l'offre.",
    "Both the CV and the listing mention scikit-learn and pandas.",
    "Experience with scikit-learn matches the listing.",
])  # fmt: skip
def test_ordinary_grounded_sentences_in_english_and_french_still_pass(student_pack, student_profile, claim):
    _, _, pack = student_pack
    assert verify_items("strength", [one(pack, claim)], pack, student_profile)[0]


def test_proper_nouns_must_come_from_the_cited_text():
    assert unsupported_proper_noun("The candidate worked at Example Retail in Lyon.", "Data Science Intern - Example Retail SAS, Lyon") is None
    assert unsupported_proper_noun("The candidate worked at Google.", "Data Science Intern - Example Retail") == "Google"
    assert unsupported_proper_noun("Python is required.", "Python") is None  # first word of a sentence, and a skill
    assert unsupported_proper_noun("Transferable, not identical: the listing asks for Azure.", "Azure is required") is None


def test_deterministic_explanations_pass_the_stricter_verifier(student_pack, student_profile, match_searcher):
    match, result, pack = student_pack
    exp = deterministic_explanation(student_profile, match, pack)
    assert exp.claims_proposed == exp.claims_verified > 0 and not exp.rejected


# ---- provenance and the stress test --------------------------------------------------------------------


def test_cited_snippets_occur_verbatim_in_their_sources(student_pack, student_profile, match_searcher):
    match, result, pack = student_pack
    exp = deterministic_explanation(student_profile, match, pack)
    cv_text = (CV_SRC / "cv_en_student_data_science.txt").read_text(encoding="utf-8")
    prov = audit.provenance(exp, cv_text, match.record["description"])
    assert prov.cited > 0 and prov.verbatim == prov.cited and prov.rate == 1.0


def test_provenance_detects_a_snippet_that_is_not_in_the_source(student_pack, student_profile):
    match, _, pack = student_pack
    exp = deterministic_explanation(student_profile, match, pack)
    tampered = Strength(claim="x", cv_refs=[pack.cv[0].id], job_refs=[pack.job[0].id])
    exp.strengths = [tampered]
    exp.gaps = exp.preparation = []
    prov = audit.provenance(exp, "an unrelated document", match.record["description"])
    assert prov.verbatim == 1 and prov.cited == 2 and prov.missing  # the job snippet is real, the CV one is not


def test_the_stress_test_rejects_every_constructed_corruption_and_reports_what_it_cannot_catch(student_pack, student_profile):
    match, _, pack = student_pack
    exp = deterministic_explanation(student_profile, match, pack)
    stress = audit.verifier_stress(exp, student_profile)
    assert sum(stress.attempts.values()) >= 20 and stress.rate == 1.0 and not stress.accepted_examples
    assert set(stress.attempts) >= {"invented_skill", "invented_degree", "invented_number", "hiring_probability", "invented_employer",
                                    "inflated_skill_level", "nonexistent_citation", "gap_about_a_skill_the_cv_has"}  # fmt: skip
    # a verifier that accepts everything would be exposed by the same test
    accept_all = audit.StressResult()
    accept_all.add("x", False, "claim")
    assert accept_all.rate == 0.0 and accept_all.accepted_examples == ["x: claim"]


# ---- real model vs mock: never confused -----------------------------------------------------------------


class ScriptedClient:
    model = "scripted:1b"

    def __init__(self, answer):
        self.answer = answer

    def chat_json(self, system, user, schema):
        return self.answer(user)


def answer(user):
    pack = json.loads(user)
    cv = next(s["id"] for s in pack["cv_snippets"] if "scikit-learn" in s["text"])
    job = next(s["id"] for s in pack["job_snippets"] if "scikit-learn" in s["text"])
    return {"strengths": [{"claim": "The CV shows scikit-learn work, which the listing asks for.", "cv_refs": [cv], "job_refs": [job]}], "gaps": [], "preparation": []}


def test_a_scripted_client_is_tagged_as_a_stub_and_a_real_ollama_client_as_ollama(student_pack, student_profile, match_searcher):
    match, result, _ = student_pack
    stub = explain_match(student_profile, match, searcher=match_searcher, queries=result.queries, client=ScriptedClient(answer))
    assert stub.mode == "llm" and stub.client_kind == "stub" and audit.client_kind(stub) == "llm-stub"
    assert audit.client_kind(explain_match(student_profile, match, searcher=match_searcher, queries=result.queries)) == "deterministic"
    real = OllamaClient(base_url="http://127.0.0.1:9", timeout=1)  # nothing listens: falls back, and says so
    fallback = explain_match(student_profile, match, searcher=match_searcher, queries=result.queries, client=real)
    assert fallback.mode == "deterministic" and audit.client_kind(fallback) == "deterministic" and "unavailable" in fallback.notes[0]


def test_grounding_report_never_claims_a_real_model_run_for_a_mock(match_searcher, tmp_path):
    shutil.copytree(CV_SRC, tmp_path / "cvs")
    candidates = me.load_candidates(tmp_path / "cvs")
    report = me.grounding_run(match_searcher, candidates[1:2], top_n=2, client=ScriptedClient(answer))
    assert report.mode_counts["llm-stub"] >= 1 and report.mode_counts["llm-ollama"] == 0
    text = report.format()
    assert "Real language-model output evaluated: NO" in text and "llm-stub" in text
    assert report.summary()["real_llm_evaluated"] is False
    assert {r["mode"] for r in report.audit_rows} <= {"llm-stub", "deterministic"}


def test_human_audit_marks_are_scored_separately_per_source(tmp_path):
    rows = [{"candidate_id": "c", "job_id": "j", "mode": mode, "kind": "strength", "claim": f"c{i}", "cv_evidence": "", "job_evidence": "", "supported": mark, "notes": ""}
            for i, (mode, mark) in enumerate([("deterministic", "yes"), ("deterministic", "no"), ("llm-ollama", "yes"), ("llm-ollama", "yes"), ("llm-ollama", "partial"), ("llm-stub", "")])]  # fmt: skip
    path = tmp_path / "audit.csv"
    me.write_audit(path, rows)
    score = me.score_audit(path)
    assert score["by_source"]["llm-ollama"]["supported_yes"]["rate"] == pytest.approx(2 / 3)
    assert score["by_source"]["deterministic"]["supported_yes"]["rate"] == 0.5
    assert score["by_source"]["llm-stub"]["human_judged"] == 0


def test_the_cli_refuses_to_pretend_ollama_ran(match_searcher, tmp_path, monkeypatch, capsys):
    from match_data import match_jobs
    from search_data import write_jobs

    from job_intel.search import service

    monkeypatch.setattr(service, "open_searcher", lambda settings, **kw: (match_searcher, []))
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("JOBINTEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JOBINTEL_OLLAMA_URL", "http://127.0.0.1:9")
    shutil.copytree(CV_SRC, tmp_path / "eval" / "synthetic_cvs")
    write_jobs(tmp_path / "data" / "processed" / "jobs.jsonl", match_jobs())
    assert cli.main(["match-eval", "grounding", "--top", "1", "--llm"]) == 0
    out = capsys.readouterr().out
    assert "REAL-MODEL run" in out and "Real language-model output evaluated: NO" in out  # Ollama was not reachable, so nothing real was evaluated
    record = json.loads((tmp_path / "data" / "eval" / "grounding_run.json").read_text(encoding="utf-8"))
    assert record["real_llm_evaluated"] is False and record["ollama_version"] is None and record["explanation_modes"] == {"deterministic": 3}


# ---- regression baselines --------------------------------------------------------------------------------


def test_compare_flags_only_real_regressions():
    base = {"a": 1.0, "b": 0.8, "c": 0.5}
    regressions, lines = regress.compare({"a": 1.0, "b": 0.7, "c": 0.9, "d": 1.0}, base)
    assert regressions == ["b"] and any("REGRESSION" in l for l in lines) and any("improved" in l for l in lines) and any("(new)" in l for l in lines)
    assert regress.compare(dict(base), base)[0] == []


def test_baseline_round_trip_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JOBINTEL_EVAL_DIR", str(tmp_path / "eval"))
    assert cli.main(["regress", "--no-index"]) == 2 and "No baseline" in capsys.readouterr().err
    assert cli.main(["regress", "--no-index", "--update"]) == 0
    path = tmp_path / "eval" / "baselines" / "label_free.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metrics"]["extraction.skill_f1"] == 1.0 and "Job" not in path.read_text(encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["regress", "--no-index"]) == 0 and "No regression" in capsys.readouterr().out
    data["metrics"]["extraction.skill_f1"] = 1.5  # pretend the baseline was better than anything achievable
    path.write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(["regress", "--no-index"]) == 1
    assert "extraction.skill_f1" in capsys.readouterr().err


def test_committed_baseline_matches_the_current_label_free_parser_metrics():
    settings = Settings(user_agent="t", eval_dir=Path(__file__).parent.parent / "eval")
    baseline = regress.load_baseline(regress.baseline_path(settings))
    metrics, _ = regress.collect_metrics(settings, settings.eval_dir / "synthetic_cvs", with_index=False)
    regressions, _ = regress.compare(metrics, {k: v for k, v in baseline.items() if k.startswith("extraction.")})
    assert baseline and regressions == []


def test_counter_import_is_available():  # keeps the Counter import used for static checkers
    assert Counter("aab")["a"] == 2 and LLMUnavailable
