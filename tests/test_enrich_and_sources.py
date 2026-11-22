"""Himalayas connector, locations, sector heuristic, structured postings and the SQLite store."""

import sqlite3

import pytest
from conftest import FakeClient

from job_intel.connectors.himalayas import HimalayasConnector
from job_intel.enrich.postings import enrich_all, enrich_job, quality_report
from job_intel.enrich.sector import classify_sector
from job_intel.enrich.store import rebuild
from job_intel.http import FetchError
from job_intel.models import InvalidRecordError
from job_intel.search.locations import countries_of, region_of, remote_allows

RAW = {
    "title": "Data Scientist", "companyName": "Acme Analytics", "companySlug": "acme",
    "employmentType": "Full Time", "locationRestrictions": ["France", "Morocco"],
    "description": "<p>We need <b>Python</b> and SQL.</p>", "pubDate": 1795310259,
    "applicationLink": "https://himalayas.app/companies/acme/jobs/data-scientist",
    "guid": "https://himalayas.app/companies/acme/jobs/data-scientist", "categories": ["Data-Science"],
}  # fmt: skip


def test_himalayas_normalize_keeps_link_and_restrictions():
    job = HimalayasConnector(FakeClient()).normalize(RAW, "2026-11-22T00:00:00+00:00")
    assert job.id == "himalayas:acme/data-scientist"
    assert job.source_url == RAW["applicationLink"] and job.remote is True
    assert job.location == "France, Morocco" and job.job_type == "full_time"
    assert "Python" in job.description and "<b>" not in job.description
    assert job.published_at.startswith("2026")
    with pytest.raises(InvalidRecordError):
        HimalayasConnector(FakeClient()).normalize({"title": "x"}, "now")
    worldwide = HimalayasConnector(FakeClient()).normalize({**RAW, "locationRestrictions": []}, "now")
    assert worldwide.location == "Worldwide"


def test_himalayas_fetch_paginates_dedups_and_survives_one_failed_query():
    page_full = {"jobs": [{**RAW, "guid": f"https://h.invalid/companies/a/jobs/{i}"} for i in range(20)]}
    page_short = {"jobs": [{**RAW, "guid": "https://h.invalid/companies/a/jobs/0"}, {**RAW, "guid": "https://h.invalid/companies/a/jobs/x"}]}
    client = FakeClient(page_full, page_short, FetchError("boom"))
    result = HimalayasConnector(client, queries=("data", "ml"), pages_per_query=3).fetch()
    assert len(result.records) == 21  # 20 + 1 new, the duplicate is dropped
    assert client.requests_made == 3 and any("failed" in w for w in result.warnings)
    with pytest.raises(FetchError):
        HimalayasConnector(FakeClient(FetchError("down")), queries=("x",)).fetch()


@pytest.mark.parametrize(("restriction", "wanted", "expected"), [
    ("United States, Canada", "France", False), ("Worldwide", "Morocco", True), ("Europe", "Paris", True),
    ("France, Morocco", "Casablanca", True), (None, "France", None), ("Germany", "Atlantis", None),
])  # fmt: skip
def test_remote_allows(restriction, wanted, expected):
    assert remote_allows(restriction, wanted) is expected


def test_countries_and_regions():
    assert countries_of("Casablanca, Maroc") == ["morocco"]
    assert countries_of("Lyon - Paris; Berlin") == ["france", "germany"]
    assert region_of("Worldwide") == "worldwide" and region_of("EMEA") == "europe"


def test_sector_needs_two_hits_and_ignores_benefits():
    assert classify_sector("We are a fintech building payments and lending products.")[0] == "Finance & Banking"
    assert classify_sector("We like banks.")[0] is None  # one keyword is not enough
    assert classify_sector("Software company.\nPrivate medical insurance and dental care for you\nHealth pension plan")[0] is None


def test_enrich_job_extracts_every_field():
    job = {"id": "arbeitnow:syn-x", "title": "Junior Data Analyst", "company": "Acme Bank", "location": "Paris, France",
           "job_type": None, "remote": False, "source": "arbeitnow", "source_url": "https://fixtures.invalid/x",
           "published_at": "2026-10-20T10:00:00+00:00", "collected_at": "2026-10-23T00:00:00+00:00",
           "description": "We are a bank serving credit and payments customers.\n\nRequirements:\n- Python and SQL\n"
                          "- 2 years of experience\n- Master's degree\n- Fluent French\n- Strong communication skills\n\n"
                          "Nice to have:\n- Tableau"}  # fmt: skip
    p = enrich_job(job)
    assert p.countries == ["france"] and p.role_family == ["data_analyst"]
    assert {"Python", "SQL"} <= set(p.required_skills) and "Tableau" in p.preferred_skills
    assert p.experience_level == "junior" and p.min_years == 2 and p.degree_level == "master"
    assert p.languages == ["French"] and "Communication" in p.soft_skills
    assert p.sector == "Finance & Banking" and str(p.apply_url) == "https://fixtures.invalid/x"


def test_invalid_records_are_reported_not_dropped_silently(tmp_path):
    good = {"id": "a:1", "title": "Data Engineer", "source": "a", "source_url": "https://x.invalid/1", "description": "Spark"}
    bad = {"id": "a:2", "title": "Data Engineer", "source": "a", "source_url": "not a url"}
    postings, rejected = enrich_all([good, bad])
    assert len(postings) == 1 and rejected[0]["id"] == "a:2"
    report = quality_report(postings, rejected)
    assert report["completeness"]["technical_skills"] == 1.0
    db = tmp_path / "jobs.db"
    assert rebuild(db, postings) == 1 and rebuild(db, postings) == 1  # idempotent
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from jobs").fetchone()[0] == 1
    assert con.execute("select skill from job_skills").fetchall() == [("Spark",)]
