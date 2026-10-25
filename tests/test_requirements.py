import pytest
from match_data import DESCRIPTIONS

from job_intel.match.requirements import extract_requirements, seniority_of


def levels(req):
    return {s.name: s.level for s in req.skills}


def test_skill_levels_follow_the_listing_sections():
    req = extract_requirements("Data Scientist Internship", DESCRIPTIONS[1])
    lv = levels(req)
    assert {lv[k] for k in ("Python", "SQL", "scikit-learn", "pandas")} == {"required"}
    assert {lv["PyTorch"], lv["Docker"]} == {"preferred"}
    assert "Kafka" not in lv  # the 'What we offer' section is not a requirement
    assert req.has_requirements_section


def test_evidence_is_the_listing_line():
    req = extract_requirements("X", DESCRIPTIONS[1])
    assert {s.name: s.evidence for s in req.skills}["scikit-learn"] == "Experience with scikit-learn and pandas"


def test_inline_cues_override_the_section():
    req = extract_requirements("X", "Requirements:\n- Python\n- Docker is a plus\n\nResponsibilities:\n- Maintain Airflow DAGs\n- Kubernetes is required")
    lv = levels(req)
    assert lv == {"Python": "required", "Docker": "preferred", "Airflow": "mentioned", "Kubernetes": "required"}


def test_listing_without_headings_marks_skills_as_mentioned_only():
    req = extract_requirements("Machine Learning Engineer", DESCRIPTIONS[5])
    assert not req.has_requirements_section and set(levels(req).values()) == {"mentioned"}
    assert {"Python", "PyTorch", "MLflow", "Docker", "Machine learning"} <= set(levels(req))  # title skills count too


def test_french_and_german_headings():
    fr = extract_requirements("Data Engineer", "Votre profil :\n- Maîtrise de Spark et Airflow\n\nIdéalement :\n- Kafka\n")
    assert levels(fr) == {"Spark": "required", "Airflow": "required", "Kafka": "preferred"}
    de = extract_requirements("Data Engineer", "Das bringst du mit:\n- Erfahrung mit Python\n\nVon Vorteil:\n- Docker")
    assert levels(de) == {"Python": "required", "Docker": "preferred"}


def test_years_of_experience_in_three_languages_and_ranges():
    assert extract_requirements("X", "Requirements:\n- 5+ years of experience in ML").required_years == 5
    assert extract_requirements("X", "Profil :\n- 3 à 5 ans d'expérience en data").required_years == 3
    assert extract_requirements("X", "Anforderungen:\n- mindestens zwei Jahre Berufserfahrung").required_years == 2
    assert extract_requirements("X", "Requirements:\n- 10 years old company, great team").required_years is None  # no experience cue
    assert extract_requirements("X", "Nice to have:\n- 3 years of experience with Spark").required_years is None  # preferred only


def test_degree_requirements_alternatives_and_enrolled_students():
    assert extract_requirements("X", DESCRIPTIONS[2]).degree[0] == "master"  # "PhD or Master's" -> the lower of the alternatives
    assert extract_requirements("X", DESCRIPTIONS[9]).degree[0] == "phd"
    student = extract_requirements("X", DESCRIPTIONS[1]).degree
    assert student[0] == "master" and student[3] is True  # accepts people still enrolled
    assert extract_requirements("X", "We manage master data in the cloud").degree is None  # 'master data' is not a degree
    fr = extract_requirements("X", "Profil :\n- Vous préparez actuellement un Bac+5 / Master 2 en informatique").degree
    assert fr[0] == "master" and fr[3] is True


def test_language_requirements_need_a_nearby_proficiency_cue():
    langs = {(l.language, l.level) for l in extract_requirements("X", DESCRIPTIONS[3]).languages}
    assert langs == {("French", "required"), ("English", "required")}
    assert [l.language for l in extract_requirements("X", DESCRIPTIONS[4]).languages] == ["German"]
    assert extract_requirements("X", "Our customers include a French bank and German carmakers").languages == []


def test_listing_site_boilerplate_is_ignored():
    req = extract_requirements("X", "Find more English Speaking Jobs in Switzerland on Arbeitnow\nRequirements:\n- Python")
    assert req.languages == []


def test_instruction_like_lines_are_ignored_and_counted():
    req = extract_requirements("Data Scientist", DESCRIPTIONS[8])
    assert req.ignored_instruction_lines == 1 and req.degree is None  # the injected 'PhD' is not a requirement
    assert set(levels(req)) == {"Python", "SQL"}


@pytest.mark.parametrize(("title", "expected"), [
    ("Data Science Intern (f/m/d)", "intern"), ("Stage - Data Engineer", "intern"), ("Werkstudent KI", "intern"),
    ("Junior Data Analyst", "junior"), ("Senior ML Engineer", "senior"), ("Lead Data Scientist", "lead_or_above"),
    ("Principal Engineer", "lead_or_above"), ("Data Analyst Confirmé(e) H/F", "senior"), ("Data Scientist", None),
])  # fmt: skip
def test_seniority_from_the_title_only(title, expected):
    assert seniority_of(title) == expected


@pytest.mark.parametrize("description", [None, "", "\n\n", "Requirements:", "- - -", "a" * 20000])
def test_empty_or_weird_descriptions_are_safe(description):
    req = extract_requirements("Title", description)
    assert req.required_years is None and isinstance(req.skills, list)


def test_long_listing_lines_keep_the_skill_mention_in_the_quoted_evidence():
    line = "Votre mission : " + "stratégie et exécution technique pour la migration de nos systèmes legacy " * 6 + "avec des pipelines ETL robustes."
    assert len(line) > 400
    evidence = {s.name: s.evidence for s in extract_requirements("X", f"Profil :\n- {line}").skills}["ETL"]
    assert "ETL" in evidence and len(evidence) <= 304
