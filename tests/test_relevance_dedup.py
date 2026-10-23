import pytest

from job_intel.dedup import deduplicate
from job_intel.models import Job
from job_intel.relevance import assess, filter_relevant


def make_job(
    title="Data Scientist",
    *,
    id="s:1",
    url="https://u.invalid/1",
    job_type=None,
    description=None,
    tags=(),
    published_at=None,
):
    return Job(
        id=id, title=title, company=None, location=None, job_type=job_type, remote=None,
        description=description, source="s", source_url=url, published_at=published_at,
        collected_at="2026-10-23T00:00:00+00:00", tags=tuple(tags),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("title", "category"),
    [
        ("Senior Data Scientist (m/w/d)", "data_scientist"),
        ("Data Analyst", "data_analyst"),
        ("Business Intelligence Analyst", "data_analyst"),
        ("Machine Learning Engineer", "ml_engineer"),
        ("Senior ML Engineer", "ml_engineer"),
        ("(Senior) AI Engineer (w/d/m)", "ai_engineer"),
        ("KI-Entwickler (m/w/d)", "ai_engineer"),
        ("Forward Deployed AI Engineer", "ai_engineer"),
        ("Data Engineer", "data_engineer"),
        ("Analytics Engineer", "data_engineer"),
        ("NLP Researcher", "nlp"),
        ("Computational Linguist", "nlp"),
    ],
)
def test_target_roles_are_relevant(title, category):
    rel = assess(make_job(title))
    assert rel is not None and category in rel.categories


@pytest.mark.parametrize(
    "title",
    [
        "Legal Business Developer (Legal AI Scaleup)",
        "Growth & AI Marketing Operations Specialist",
        "Senior AI Governance Specialist",
        "Senior Software Engineer, AI Training - UK",
        "Freelance Copywriter",
        "Product Owner",
    ],
)
def test_unrelated_titles_are_rejected(title):
    assert assess(make_job(title)) is None


def test_description_alone_cannot_make_a_job_relevant():
    assert assess(make_job("Sales Manager", description="python sql machine learning pandas nlp")) is None


def test_internship_needs_a_data_signal():
    intern = {"job_type": "internship"}
    assert assess(make_job("Winter Data Internship", **intern)) is not None
    assert assess(make_job("Praktikum KI-Entwicklung (m/w/d)", **intern)) is not None
    assert assess(make_job("Research Intern", tags=["Data Science"], **intern)) is not None
    desc = "You will use pandas and machine learning."
    assert assess(make_job("Research Intern", description=desc, **intern)) is not None
    assert assess(make_job("Software Engineering Praktikum", description="Python and SQL.", **intern)) is None
    assert assess(make_job("Securities Operations Intern", **intern)) is None


def test_ai_mention_in_a_business_function_title_is_not_enough_for_internships():
    intern = {"job_type": "internship"}
    assert assess(make_job("AI & Marketing Transformation Internship", **intern)) is None
    assert assess(make_job("Werkstudent Grafikdesign & KI-Content", job_type="working_student")) is None
    assert assess(make_job("Marketing Intern", tags=["AI/ML"], **intern)) is None


def test_reverse_order_ai_titles():
    for title in ("Senior Software Engineer - AI", "Staff Software Engineer: Applied AI",
                  "Senior Software Engineer - AI Foundations (F/H/X) - CDI"):  # fmt: skip
        assert "ai_engineer" in assess(make_job(title)).categories
    assert assess(make_job("Senior Software Engineer, AI Training - UK")) is None


def test_ranking_prefers_higher_score_then_newest():
    strong = make_job("Data Scientist", id="a", url="https://u/a", description="python sql pandas",
                      published_at="2026-01-01T00:00:00+00:00")  # fmt: skip
    weak = make_job("Data Analyst Intern", id="b", url="https://u/b", job_type="internship")
    newer = make_job("Data Scientist", id="c", url="https://u/c", description="python sql pandas",
                     published_at="2026-03-01T00:00:00+00:00")  # fmt: skip
    ordered = [job.id for job, _ in filter_relevant([weak, strong, newer])]
    # internship (10+5) beats 10+3 skill hits; equal scores fall back to the newest listing
    assert ordered == ["b", "c", "a"]


def test_dedup_by_id_and_canonical_url():
    jobs = [
        make_job(id="s:1", url="https://u.invalid/a"),
        make_job(id="s:1", url="https://u.invalid/other"),  # same id
        make_job(id="s:2", url="HTTPS://U.invalid/a/?utm_source=z#x"),  # same URL once canonicalized
        make_job(id="s:3", url="https://u.invalid/c"),
    ]
    kept, removed = deduplicate(jobs)
    assert [j.id for j in kept] == ["s:1", "s:3"]
    assert dict(removed) == {"id": 1, "source_url": 1}


def test_dedup_catches_reposts_with_new_id_and_url():
    def repost(id, url, location="Berlin", company="Acme"):
        job = make_job("Machine  Learning Engineer", id=id, url=url)
        job.company, job.location = company, location
        return job

    jobs = [
        repost("s:1", "https://u.invalid/1"),
        repost("s:2", "https://u.invalid/2"),  # reposted: same title/company/location
        repost("s:3", "https://u.invalid/3", location="Munich"),  # other site: kept
        repost("s:4", "https://u.invalid/4", company="Other"),  # other company: kept
    ]
    kept, removed = deduplicate(jobs)
    assert [j.id for j in kept] == ["s:1", "s:3", "s:4"]
    assert dict(removed) == {"title_company_location": 1}
    no_company = [make_job(id="s:5", url="https://u.invalid/5"), make_job(id="s:6", url="https://u.invalid/6")]
    assert len(deduplicate(no_company)[0]) == 2  # without a company the rule is not applied


def test_dedup_keeps_first_and_handles_empty():
    assert deduplicate([]) == ([], {})
    a, b = make_job(id="s:1", title="first"), make_job(id="s:1", title="second")
    assert deduplicate([a, b])[0][0].title == "first"
