from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from match_data import match_jobs  # noqa: E402

from job_intel.match.cv_parser import parse_cv  # noqa: E402
from job_intel.match.cv_text import clean_cv_text  # noqa: E402
from job_intel.match.matcher import (  # noqa: E402
    build_queries,
    match_candidate,
    match_skills,
    retrieve_candidates,
    role_relevance,
)
from job_intel.match.models import Preferences  # noqa: E402
from job_intel.match.requirements import extract_requirements  # noqa: E402
from job_intel.search.document import ChunkingConfig  # noqa: E402
from job_intel.search.embedder import HashingEmbedder  # noqa: E402
from job_intel.search.index import IndexConfig, JobIndex  # noqa: E402
from job_intel.search.retrieve import Searcher  # noqa: E402

CV_DIR = Path(__file__).parent.parent / "eval" / "synthetic_cvs"
TODAY = date(2026, 10, 23)
EMB = HashingEmbedder(max_seq_length=128)


def profile_of(name):
    return parse_cv(clean_cv_text((CV_DIR / name).read_text(encoding="utf-8")), today=TODAY)


@pytest.fixture(scope="module")
def student():
    return profile_of("cv_en_student_data_science.txt")


@pytest.fixture(scope="module")
def ml():
    return profile_of("cv_en_ml_engineer.txt")


@pytest.fixture(scope="module")
def searcher(tmp_path_factory):
    records = match_jobs()
    index = JobIndex.open_for_build(tmp_path_factory.mktemp("idx"), IndexConfig.build(EMB, ChunkingConfig(100, 20)))
    index.sync(records, EMB)
    return Searcher(index, EMB, {j["id"]: j for j in records})


def jid(n: int) -> str:
    return f"arbeitnow:syn-{n:02d}"


def by_id(matches):
    return {m.job.job_id: m for m in matches}


def constraint(match, name):
    return next((c for c in match.constraints if c.name == name), None)


# ---- Step A: candidate representation --------------------------------------------------------


def test_queries_cover_roles_skills_every_experience_and_project(student, ml):
    queries = build_queries(student, Preferences())
    kinds = [q.kind for q in queries]
    assert kinds.count("experience") == 2 and kinds.count("project") == 2 and "role" in kinds and "skills" in kinds and "summary" in kinds
    assert any("Tweet sentiment classifier" in q.label for q in queries)
    assert any("forecasting" in q.text.lower() for q in queries)  # experience bullet content is kept
    assert {q.kind for q in build_queries(ml, Preferences())} >= {"role", "skills", "experience"}


def test_preferences_roles_override_cv_roles_and_internship_hint_is_added(student):
    q = build_queries(student, Preferences(target_roles=["Data Analyst"], job_types=["internship"]))
    roles = [x for x in q if x.kind == "role"]
    assert [r.label for r in roles] == ["role: Data Analyst"] and "internship" in roles[0].text


def test_empty_profile_gives_no_queries_and_a_warning(searcher):
    from job_intel.match.models import CandidateProfile

    result = match_candidate(CandidateProfile(), Preferences(), searcher)
    assert result.matches == [] and result.warnings


# ---- Step B: retrieval ---------------------------------------------------------------------


def test_retrieval_fuses_both_engines_and_records_why(student, searcher):
    traces = retrieve_candidates(searcher, build_queries(student, Preferences()), k_per_query=10)
    assert jid(1) in traces
    t = traces[jid(1)]
    assert t.rrf > 0 and t.best_similarity is not None and (t.best_bm25_rank or t.best_semantic_rank)
    assert t.found_by  # which CV items led to this job
    assert list(traces) == sorted(traces, key=lambda j: (-traces[j].rrf, j))


# ---- Step C: eligibility -------------------------------------------------------------------


def test_hard_job_type_excludes_clear_violations_but_keeps_unknown(student, searcher):
    result = match_candidate(student, Preferences(job_types=["internship"]), searcher)
    ids = {m.job.job_id for m in result.matches}
    assert ids == {jid(1), jid(4), jid(5)}  # job 5 has no job type: unknown, not a mismatch
    assert constraint(by_id(result.matches)[jid(5)], "job_type").status == "unknown"
    excluded = by_id(result.excluded)
    assert jid(2) in excluded and "job_type" in excluded[jid(2)].excluded_because[0]


def test_hard_location_excludes_other_places_but_keeps_missing_locations(student, searcher):
    result = match_candidate(student, Preferences(job_types=["internship"], locations=["France"]), searcher)
    assert {m.job.job_id for m in result.matches} == {jid(1), jid(5)}
    assert constraint(by_id(result.matches)[jid(5)], "location").status == "unknown"
    assert constraint(by_id(result.matches)[jid(1)], "location").status == "satisfied"  # Paris is in France
    assert "Berlin" in by_id(result.excluded)[jid(4)].excluded_because[0]


def test_soft_constraints_warn_but_do_not_exclude(student, searcher):
    result = match_candidate(student, Preferences(locations=["France"], hard_constraints=[]), searcher)
    berlin = by_id(result.matches)[jid(4)]
    assert constraint(berlin, "location").status == "violated" and not constraint(berlin, "location").hard
    assert result.excluded == []


def test_remote_only_requires_explicit_remote_and_prefer_does_not_exclude(ml, searcher):
    only = match_candidate(ml, Preferences(remote="only", hard_constraints=[]), searcher)
    assert {m.job.job_id for m in only.matches} == {jid(6)}
    assert "remote-only" in by_id(only.excluded)[jid(2)].excluded_because[0]
    prefer = match_candidate(ml, Preferences(remote="prefer", hard_constraints=[]), searcher)
    assert len(prefer.matches) == len(only.matches) + len(only.excluded)
    assert constraint(by_id(prefer.matches)[jid(2)], "remote").status == "unknown"


def test_experience_violated_unknown_and_satisfied(student, ml, searcher):
    # student: dated experience of a few months only -> a 5-year requirement is clearly not met (soft by default)
    result = match_candidate(student, Preferences(hard_constraints=[]), searcher)
    c = constraint(by_id(result.matches)[jid(2)], "experience")
    assert c.status == "violated" and not c.hard and "at most" in c.detail
    # ... hard if the user says so
    hard = match_candidate(student, Preferences(hard_constraints=["experience"]), searcher)
    assert jid(2) in by_id(hard.excluded) and jid(9) in by_id(hard.excluded)
    # a CV with undated experience cannot be judged: unknown, never excluded
    undated = parse_cv("EXPERIENCE\nData Engineer - Acme\n- Built Spark pipelines with Python and Airflow\n", today=TODAY)
    unknown = match_candidate(undated, Preferences(hard_constraints=["experience"]), searcher)
    assert constraint(by_id(unknown.matches)[jid(2)], "experience").status == "unknown" and unknown.excluded == []
    # long dated experience satisfies it
    assert constraint(by_id(match_candidate(ml, Preferences(hard_constraints=[]), searcher).matches)[jid(2)], "experience").status == "satisfied"


def test_education_requirements_missing_qualification_is_flagged(ml, student, searcher):
    ml_result = by_id(match_candidate(ml, Preferences(hard_constraints=[]), searcher).matches)
    phd_job = constraint(ml_result[jid(9)], "education")
    assert phd_job.status == "violated" and "master" in phd_job.detail and "phd" in phd_job.detail
    assert constraint(ml_result[jid(2)], "education").status == "satisfied"  # 'PhD or Master's'
    student_result = by_id(match_candidate(student, Preferences(hard_constraints=[]), searcher).matches)
    enrolled = constraint(student_result[jid(1)], "education")
    assert enrolled.status == "satisfied" and "enrolled" in enrolled.detail  # in-progress master accepted by the listing


def test_language_requirements(student, ml, searcher):
    s = by_id(match_candidate(student, Preferences(hard_constraints=[]), searcher).matches)
    assert constraint(s[jid(4)], "language").status == "violated"  # German required, CV lists French/English/Spanish
    assert constraint(s[jid(3)], "language").status == "satisfied"
    hard = match_candidate(student, Preferences(hard_constraints=["language"]), searcher)
    assert jid(4) in by_id(hard.excluded) and jid(3) not in by_id(hard.excluded)
    no_languages = parse_cv("SKILLS\nPython, SQL\n", today=TODAY)
    unknown = by_id(match_candidate(no_languages, Preferences(hard_constraints=["language"]), searcher).matches)
    assert constraint(unknown[jid(4)], "language").status == "unknown"


# ---- Step D: skills and ranking ------------------------------------------------------------


def test_exact_transferable_and_not_evidenced_skills_are_distinguished(student):
    req = extract_requirements("Senior Machine Learning Engineer", "Requirements:\n- TensorFlow, Kafka, Python and Tableau")
    got = {m.skill: m for m in match_skills(student, req)}
    assert got["Python"].status == "exact" and got["Python"].cv_evidence
    assert got["TensorFlow"].status == "transferable" and got["TensorFlow"].via == ["PyTorch"]  # related, not the same
    assert got["Tableau"].status == "transferable" and got["Tableau"].via == ["Power BI"]
    assert got["Kafka"].status == "not_evidenced" and got["Kafka"].cv_evidence == []


def test_transferable_evidence_never_counts_as_exact(student, searcher):
    m = by_id(match_candidate(student, Preferences(hard_constraints=[]), searcher).matches)[jid(2)]
    assert {s.skill for s in m.skills_with("transferable")} >= {"TensorFlow"}
    assert m.counts["exact"] == sum(s.status == "exact" for s in m.base_skills)


def test_role_relevance():
    assert role_relevance("Senior Machine Learning Engineer", ["ML Engineer"])[0] == "match"
    assert role_relevance("Data Scientist Internship", ["Data Scientist"]) == ("match", ["Data Scientist"])
    assert role_relevance("Marketing Manager", ["Data Scientist"]) == ("no_match", [])
    assert role_relevance("Anything", []) == ("unknown", [])
    assert role_relevance("Software Engineer", ["Software Engineer"])[0] == "match"  # role outside the Phase 1 families


def test_ranking_orders_by_evidence_and_puts_irrelevant_jobs_last(student, searcher):
    result = match_candidate(student, Preferences(hard_constraints=[]), searcher)
    order = [m.job.job_id for m in result.matches]
    assert order[0] == jid(1)  # internship with every required skill and a matching role
    assert result.matches[0].evidence_level == "strong"
    assert order.index(jid(1)) < order.index(jid(10)) and order.index(jid(5)) < order.index(jid(10))
    assert by_id(result.matches)[jid(10)].evidence_level in {"weak", "insufficient_information"}


def test_unmet_requirements_rank_below_comparable_jobs(ml, searcher):
    result = match_candidate(ml, Preferences(hard_constraints=[]), searcher)
    order = [m.job.job_id for m in result.matches]
    assert order.index(jid(2)) < order.index(jid(9))  # job 9 asks for a PhD the CV does not state


def test_no_percentage_or_probability_is_exposed(student, searcher):
    match = match_candidate(student, Preferences(), searcher).matches[0]
    names = {n.lower() for n in dir(match)}
    assert not {n for n in names if any(w in n for w in ("percent", "probab", "fit_score", "likelihood"))}
    assert match.evidence_level in {"strong", "moderate", "weak", "insufficient_information"}


def test_listing_with_injected_instructions_is_handled_safely(student, searcher):
    result = match_candidate(student, Preferences(hard_constraints=[]), searcher)
    m = by_id(result.matches)[jid(8)]
    assert any("look like instructions" in n for n in m.notes)
    assert constraint(m, "education") is None  # the injected 'PhD' was not turned into a requirement


def test_matching_is_deterministic(student, searcher):
    first = [m.job.job_id for m in match_candidate(student, Preferences(), searcher).matches]
    second = [m.job.job_id for m in match_candidate(student, Preferences(), searcher).matches]
    assert first == second


def test_max_results_caps_the_list_and_says_so(student, searcher):
    result = match_candidate(student, Preferences(max_results=2, hard_constraints=[]), searcher)
    assert len(result.matches) == 2 and any("not shown" in w for w in result.warnings)
