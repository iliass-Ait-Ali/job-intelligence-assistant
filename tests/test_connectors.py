import pytest
from conftest import FakeClient

from job_intel.connectors import ArbeitnowConnector, RemotiveConnector
from job_intel.http import FetchError
from job_intel.models import FIELDS, InvalidRecordError

NOW = "2026-10-23T08:00:00+00:00"


# --- Arbeitnow ----------------------------------------------------------------------------


def test_arbeitnow_normalizes_full_record(arbeitnow_pages):
    job = ArbeitnowConnector(FakeClient()).normalize(arbeitnow_pages[0]["data"][0], NOW)
    assert job.id == "arbeitnow:data-scientist-acme-1"
    assert job.title == "Data Scientist (m/w/d)"
    assert job.company == "Acme GmbH" and job.location == "Berlin"
    assert job.job_type == "permanent" and job.remote is False
    assert job.description == "Your role\n\nBuild models with Python & SQL. Use pandas.\n\n- Forecasting\n- NLP"
    assert job.source == "arbeitnow"
    assert job.source_url == "https://fixtures.invalid/arbeitnow/jobs/data-scientist-acme-1"  # untouched
    assert job.published_at == "2026-10-23T19:00:48+00:00"
    assert job.collected_at == NOW
    assert list(job.to_record()) == list(FIELDS)


def test_arbeitnow_handles_nulls_and_empty_strings(arbeitnow_pages):
    job = ArbeitnowConnector(FakeClient()).normalize(arbeitnow_pages[0]["data"][2], NOW)
    assert job.description is None and job.location is None
    assert job.job_type == "internship"  # inferred from the title


def test_arbeitnow_rejects_records_without_title_or_url(arbeitnow_pages):
    connector = ArbeitnowConnector(FakeClient())
    with pytest.raises(InvalidRecordError):
        connector.normalize(arbeitnow_pages[0]["data"][7], NOW)  # blank title
    with pytest.raises(InvalidRecordError):
        connector.normalize({"title": "x"}, NOW)


def test_arbeitnow_missing_slug_gets_hash_id():
    job = ArbeitnowConnector(FakeClient()).normalize({"title": "t", "url": "https://u.invalid/a"}, NOW)
    assert job.id.startswith("arbeitnow:") and len(job.id) > len("arbeitnow:")
    assert job.remote is None and job.company is None and job.published_at is None


def test_arbeitnow_follows_pagination_until_no_next(arbeitnow_pages):
    client = FakeClient(*arbeitnow_pages)
    result = ArbeitnowConnector(client, max_pages=5).fetch()
    assert len(result.records) == 10
    assert [params for _, params in client.calls] == [{"page": 1}, {"page": 2}]
    assert result.meta["terms"] == "FIXTURE terms text" and result.meta["pages_fetched"] == 2


def test_arbeitnow_respects_max_pages(arbeitnow_pages):
    client = FakeClient(*arbeitnow_pages)
    ArbeitnowConnector(client, max_pages=1).fetch()
    assert len(client.calls) == 1


def test_arbeitnow_later_page_failure_keeps_partial_data(arbeitnow_pages):
    client = FakeClient(arbeitnow_pages[0], FetchError("HTTP 500"))
    result = ArbeitnowConnector(client).fetch()
    assert len(result.records) == 8
    assert "page 2 failed" in result.warnings[0]


def test_arbeitnow_first_page_failure_raises():
    with pytest.raises(FetchError):
        ArbeitnowConnector(FakeClient(FetchError("timeout"))).fetch()


@pytest.mark.parametrize("payload", [{"unexpected": []}, [], {"data": "nope"}, None])
def test_arbeitnow_unexpected_shape_raises(payload):
    with pytest.raises(FetchError, match="data"):
        ArbeitnowConnector(FakeClient(payload)).fetch()


# --- Remotive -----------------------------------------------------------------------------


def test_remotive_normalizes_record(remotive_payload):
    job = RemotiveConnector(FakeClient()).normalize(remotive_payload["jobs"][0], NOW)
    assert job.id == "remotive:9001"
    assert (job.company, job.location, job.job_type) == ("Theta Inc", "Worldwide", "full_time")
    assert job.remote is True
    assert job.description == "Ship models.\n\nPython, PyTorch."
    assert job.source == "remotive"
    assert job.source_url == "https://fixtures.invalid/remotive/remote-jobs/data/ml-engineer-9001"
    assert job.published_at == "2026-10-17T16:04:29+00:00"
    assert "Data" in job.tags


def test_remotive_timezone_offsets_are_converted_to_utc(remotive_payload):
    job = RemotiveConnector(FakeClient()).normalize(remotive_payload["jobs"][1], NOW)
    assert job.published_at == "2026-10-16T08:00:00+00:00"


def test_remotive_nulls_and_bad_dates(remotive_payload):
    job = RemotiveConnector(FakeClient()).normalize(remotive_payload["jobs"][2], NOW)
    assert job.company is None and job.location is None and job.description is None
    assert job.published_at is None
    assert job.job_type == "internship"  # from "Data Engineer - Intern"
    assert job.tags == ("Data",)


def test_remotive_fetch_makes_one_request(remotive_payload):
    client = FakeClient(remotive_payload)
    result = RemotiveConnector(client).fetch()
    assert len(client.calls) == 1 and len(result.records) == 3
    assert result.meta["legal_notice"] == "FIXTURE legal notice"


@pytest.mark.parametrize("payload", [{"jobs": None}, {"foo": 1}, "text"])
def test_remotive_unexpected_shape_raises(payload):
    with pytest.raises(FetchError, match="jobs"):
        RemotiveConnector(FakeClient(payload)).fetch()
