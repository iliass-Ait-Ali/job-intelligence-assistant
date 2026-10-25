import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from job_intel.match.cv_parser import date_range, detect_language, heading_of, parse_cv, parse_date_token
from job_intel.match.cv_text import clean_cv_text
from job_intel.match.models import CandidateProfile, Education, Preferences, SkillItem
from job_intel.match.taxonomy import find_skills, related_skills

TODAY = date(2026, 10, 23)
CV_DIR = Path(__file__).parent.parent / "eval" / "synthetic_cvs"


def load(name: str) -> CandidateProfile:
    return parse_cv(clean_cv_text((CV_DIR / name).read_text(encoding="utf-8")), today=TODAY)


@pytest.fixture(scope="module")
def student() -> CandidateProfile:
    return load("cv_en_student_data_science.txt")


@pytest.fixture(scope="module")
def french() -> CandidateProfile:
    return load("cv_fr_data_engineer.txt")


@pytest.fixture(scope="module")
def ml() -> CandidateProfile:
    return load("cv_en_ml_engineer.txt")


# ---- structure -----------------------------------------------------------------------------


def test_english_student_profile(student):
    assert student.detected_language == "en"
    assert {"Python", "SQL", "R", "Java"} <= set(student.programming_languages)
    assert {"PyTorch", "scikit-learn", "Hugging Face", "pandas"} <= set(student.frameworks_and_technologies)
    assert student.highest_degree_level == "bachelor" and student.in_progress_degree_level == "master"
    assert [e.level for e in student.education] == ["master", "bachelor"]
    assert student.education[0].in_progress is True and student.education[1].in_progress is False
    assert student.education[0].institution.startswith("Example Institute")
    assert [x.kind for x in student.experiences] == ["internship", "job"]
    assert [p.name for p in student.projects] == ["Tweet sentiment classifier", "Movie recommender"]
    assert {(s.language, s.level_text) for s in student.spoken_languages} >= {("French", "native"), ("English", "fluent")}


def test_french_profile_has_dates_languages_and_degree(french):
    assert french.detected_language == "fr"
    assert french.target_roles == ["Data Engineer"]
    assert french.highest_degree_level == "master"  # diplôme d'ingénieur
    current = french.experiences[0]
    assert (current.title, current.kind, current.start, current.end, current.current) == ("Ingénieur Data", "job", "2022-09", "present", True)
    assert current.duration_months_min == 48 and current.duration_months_max == 50  # Sept 2022 -> Oct 2026
    assert french.experiences[1].kind == "internship"
    assert {"Airflow", "Spark", "dbt", "Snowflake", "Terraform"} <= french.skill_names
    assert [s.language for s in french.spoken_languages] == ["French", "English", "German"]


def test_ml_profile_experience_bounds_and_target_roles(ml):
    assert ml.target_roles == ["Machine Learning Engineer", "AI Engineer"]
    low, high = ml.experience_months_bounds()
    assert (low, high) == (109, 113)  # 80-82 months + 29-31 months, from the stated dates only
    assert {"RAG", "LangChain", "ChromaDB", "Kubernetes"} <= ml.skill_names


def test_skill_context_distinguishes_demonstrated_from_listed(student):
    by_name = {s.name: s for s in student.skills}
    assert by_name["Python"].context == "demonstrated" and by_name["Python"].evidence[0].section in {"experience", "projects"}
    assert by_name["Java"].context == "listed" and by_name["Java"].evidence[0].section == "skills"
    assert all(e.text for s in student.skills for e in s.evidence)  # every skill keeps CV text


def test_wishes_are_not_skills(ml, student):
    # "...working on LLM applications" / "Interested in NLP" are objectives, not evidence of a skill
    assert "LLM" not in ml.skill_names
    assert [e.text for e in {s.name: s for s in student.skills}["NLP"].evidence] == ["Concepts: statistics, machine learning, NLP, time series, data visualization"]


# ---- nothing invented ----------------------------------------------------------------------


def test_no_contact_details_or_proficiency_are_extracted(student, french):
    dumped = json.dumps([student.model_dump(), french.model_dump()], ensure_ascii=False).lower()
    assert "example.invalid" not in dumped and "+00 000" not in dumped
    for banned in ("proficiency", "years_of_experience", "seniority", "expert", "advanced", "beginner"):
        assert banned not in json.dumps(CandidateProfile.model_json_schema()).lower()


def test_target_roles_only_come_from_objective_statements():
    plain = parse_cv("Data Scientist\nEXPERIENCE\nData Scientist - Acme, 2020 - 2022\n- Used Python and SQL\n", today=TODAY)
    assert plain.target_roles == []  # a job title is not a wish
    wish = parse_cv("PROFILE\nLooking for a data analyst position in Lyon.\n", today=TODAY)
    assert wish.target_roles == ["Data Analyst"]


def test_missing_dates_mean_unknown_duration_not_a_guess():
    p = parse_cv("EXPERIENCE\nData Analyst - Acme\n- Built Power BI dashboards\n", today=TODAY)
    assert p.experiences[0].duration_months_min is None and p.experiences[0].duration_months_max is None
    assert p.experience_months_bounds() is None
    assert any("No dates" in w for w in p.warnings)


def test_year_only_dates_give_wide_honest_bounds():
    p = parse_cv("EXPERIENCE\nData Analyst - Acme, 2021 - 2023\n- Built dashboards in Tableau\n", today=TODAY)
    e = p.experiences[0]
    assert (e.start, e.end) == ("2021", "2023") and (e.duration_months_min, e.duration_months_max) == (12, 35)


def test_ambiguous_skill_names_need_a_technical_context():
    assert find_skills("Plan C was to go home; R&D in Go style") == []
    assert set(find_skills("Languages: Python, R and Go")) == {"Python", "R", "Go"}
    assert find_skills("Julia Martin") == [] and find_skills("Excel at teamwork") == []
    assert find_skills("Built a Golang service") == ["Go"]


def test_skills_use_word_boundaries():
    assert find_skills("JavaScript developer") == ["JavaScript"]  # not Java
    assert find_skills("C++ and C# code") == ["C++", "C#"]
    assert find_skills("worked at Rustic Inc., pandas-profiling") == []  # no Rust, no pandas inside other words


# ---- robustness ----------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\n", "!!!???", "x" * 5000, "​​", "EXPERIENCE", "EXPERIENCE\nEDUCATION\nSKILLS\nPROJECTS"])
def test_malformed_or_empty_text_never_crashes(text):
    profile = parse_cv(text, today=TODAY)
    assert isinstance(profile, CandidateProfile) and profile.warnings


def test_cv_without_headings_still_yields_skills_and_a_warning():
    p = parse_cv("I build machine learning models with Python and PyTorch and deploy them on AWS.", today=TODAY)
    assert {"Python", "PyTorch", "AWS", "Machine learning"} <= p.skill_names
    assert any("No standard section headings" in w for w in p.warnings)


def test_section_headings_in_three_languages():
    for line, section in [("EXPERIENCE", "experience"), ("Expériences professionnelles", "experience"), ("Berufserfahrung", "experience"),
                          ("Formation :", "education"), ("Skills & Tools", "skills"), ("Compétences techniques", "skills"),
                          ("Projets", "projects"), ("LANGUES", "languages"), ("Centres d'intérêt", "ignored")]:  # fmt: skip
        assert heading_of(line) == section
    assert heading_of("Built a pipeline with Airflow for the experience team") is None


def test_dates_in_english_french_and_german():
    for text, start, end in [("Sept 2022 - present", "2022-09", "present"), ("Septembre 2022 – Présent", "2022-09", "present"),
                             ("janv. 2021 - mars 2022", "2021-01", "2022-03"), ("03/2020 - 08/2021", "2020-03", "2021-08"),
                             ("Jan 2020 to Dec 2020", "2020-01", "2020-12"), ("Oktober 2019 bis heute", "2019-10", "present")]:  # fmt: skip
        got = date_range(text, TODAY)
        assert got and (got["start"], got["end"]) == (start, end), text
    assert date_range("2023 - 2020", TODAY) is None  # reversed range is not trusted
    assert date_range("no dates here", TODAY) is None
    assert parse_date_token("Avril 2021") == (2021, 4)


def test_language_detection():
    assert detect_language("Je suis ingénieur et je travaille dans la data avec des équipes pour les clients") == "fr"
    assert detect_language("I work with teams and build models for the customers in a company") == "en"
    assert detect_language("12345") == "unknown"


def test_related_skills_are_symmetric_groups():
    assert "TensorFlow" in related_skills("PyTorch") and "PyTorch" in related_skills("TensorFlow")
    assert "Tableau" in related_skills("Power BI")
    assert "Python" not in related_skills("Java")


# ---- validation ----------------------------------------------------------------------------


def test_profile_round_trips_through_json(student):
    again = CandidateProfile.model_validate_json(student.model_dump_json())
    assert again == student


def test_profile_validation_rejects_bad_data():
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate({"unknown_field": 1})  # extra fields forbidden
    with pytest.raises(ValidationError):
        Education(level="wizard")
    with pytest.raises(ValidationError):
        SkillItem(name="")
    with pytest.raises(ValidationError):
        SkillItem(name="Python", context="expert")  # no proficiency vocabulary
    with pytest.raises(ValidationError):
        Preferences(hard_constraints=["salary"])
    with pytest.raises(ValidationError):
        Preferences(max_results=0)


def test_target_roles_are_deduplicated_and_trimmed():
    assert CandidateProfile(target_roles=[" Data Scientist ", "data scientist", ""]).target_roles == ["Data Scientist"]


def test_highest_degree_ignores_in_progress_studies():
    p = CandidateProfile(education=[Education(level="master", in_progress=True), Education(level="bachelor", in_progress=False),
                                    Education(level="phd", in_progress=None)])  # fmt: skip
    assert p.highest_degree_level == "phd" and p.in_progress_degree_level == "master"
