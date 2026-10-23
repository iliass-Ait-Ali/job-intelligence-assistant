from datetime import date

import pytest

from job_intel.search.errors import SearchError
from job_intel.search.filters import IndexedJob, JobFilter
from job_intel.search.locations import canonical, location_keys, location_matches, normalize


def job(**kw) -> IndexedJob:
    base = dict(job_id="s:1", title="t", company="c", location="Paris", job_type="internship", remote=False,
                source="arbeitnow", url="https://u.invalid/1", published_ts=1_792_000_000)  # fmt: skip
    return IndexedJob(**{**base, **kw})


# ---- locations ----------------------------------------------------------------------------


def test_normalize_strips_accents_case_and_punctuation():
    assert normalize("Zürich") == "zurich" == normalize("ZURICH")
    assert normalize("Boulogne-Billancourt") == "boulogne billancourt"


@pytest.mark.parametrize(
    ("location", "wanted"),
    [
        ("Zürich", "Zurich"), ("Zurich", "zürich"), ("Zürich", "Switzerland"), ("Zürich", "Schweiz"),
        ("Amsterdam, Noord-Holland, Nederland", "Netherlands"), ("Amsterdam", "Nederland"), ("Amsterdam", "Netherlands"),
        ("Rotterdam, Zuid-Holland, Netherlands", "Nederland"),
        ("Lille - Jardin des Plantes, Nord; Paris, Paris", "Paris"), ("Lille - Jardin des Plantes, Nord", "France"),
        ("UK - London", "United Kingdom"), ("London", "UK"), ("Munich HQ", "Munich"), ("Munich HQ", "Deutschland"),
        ("Köln", "Cologne"), ("Paris, Île-de-France, France", "Ile de France"), ("'s-Hertogenbosch, Noord-Brabant, Nederland", "Netherlands"),
    ],
)  # fmt: skip
def test_location_matches_spelling_variants_and_countries(location, wanted):
    assert location_matches(location, wanted)


@pytest.mark.parametrize(
    ("location", "wanted"),
    [("Paris", "London"), ("Zürich", "Germany"), ("Amsterdam", "Belgium"), ("Remote job", "France"), (None, "Paris"), ("", "Paris")],
)
def test_location_does_not_match_other_places(location, wanted):
    assert not location_matches(location, wanted)


def test_location_keys_for_missing_location_is_empty():
    assert location_keys(None) == frozenset()
    assert canonical("Nederland") == "netherlands"


# ---- filters ------------------------------------------------------------------------------


def test_empty_filter_matches_everything():
    flt = JobFilter()
    assert flt.is_empty and flt.matches(job(job_type=None, location=None, published_ts=None))


def test_job_type_filter_and_unknown_job_type_behaviour():
    unknown = job(job_type=None)
    assert JobFilter(job_types=("internship",)).matches(job())
    assert not JobFilter(job_types=("internship",)).matches(job(job_type="full_time"))
    assert not JobFilter(job_types=("internship",)).matches(unknown)  # unknown excluded by default
    assert JobFilter(job_types=("internship",), include_unknown=True).matches(unknown)
    assert JobFilter(job_types=("unknown",)).matches(unknown)  # explicitly asking for unknown
    assert not JobFilter(job_types=("unknown",)).matches(job())
    assert JobFilter(job_types=("internship", "working_student")).matches(job(job_type="working_student"))


def test_remote_filter_only_keeps_explicit_remote_and_ignores_include_unknown():
    flt = JobFilter(remote=True, include_unknown=True)
    assert flt.matches(job(remote=True))
    assert not flt.matches(job(remote=False))  # False means "not stated", so it cannot satisfy --remote
    assert not flt.matches(job(remote=None))


def test_location_filter_handles_missing_location():
    flt = JobFilter(locations=("Paris",))
    assert flt.matches(job(location="Paris, France"))
    assert not flt.matches(job(location="London"))
    assert not flt.matches(job(location=None))
    assert JobFilter(locations=("Paris",), include_unknown=True).matches(job(location=None))
    assert JobFilter(locations=("Paris", "London")).matches(job(location="London"))  # OR inside one filter


def test_since_filter_handles_missing_dates():
    ts = 1_792_000_000  # 2026-10-14
    assert JobFilter(since=date(2026, 10, 14)).matches(job(published_ts=ts))
    assert not JobFilter(since=date(2026, 10, 20)).matches(job(published_ts=ts))
    assert not JobFilter(since=date(2026, 10, 1)).matches(job(published_ts=None))
    assert JobFilter(since=date(2026, 10, 1), include_unknown=True).matches(job(published_ts=None))


def test_filters_combine_with_and():
    flt = JobFilter(job_types=("internship",), locations=("Paris",), sources=("arbeitnow",))
    assert flt.matches(job())
    assert not flt.matches(job(source="remotive"))
    assert not flt.matches(job(location="London"))


def test_from_dict_validates_input():
    flt = JobFilter.from_dict({"job_types": "internship", "locations": ["Paris"], "since": "2026-10-01", "remote": True})
    assert flt.job_types == ("internship",) and flt.since == date(2026, 10, 1) and flt.remote is True
    assert JobFilter.from_dict(None).is_empty
    with pytest.raises(SearchError, match="unknown filter"):
        JobFilter.from_dict({"countries": ["fr"]})
    with pytest.raises(SearchError, match="since"):
        JobFilter.from_dict({"since": "yesterday"})
