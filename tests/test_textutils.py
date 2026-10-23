import pytest

from job_intel.textutils import (
    canonical_job_type,
    canonical_url,
    clean_str,
    html_to_text,
    normalize_whitespace,
    parse_datetime,
)


def test_html_is_stripped_and_structure_kept():
    html = "<h2>Role</h2><p>Python &amp; SQL.&nbsp;&nbsp;Use   pandas.</p><ul><li>One</li><li>Two</li></ul>"
    assert html_to_text(html) == "Role\n\nPython & SQL. Use pandas.\n\n- One\n- Two"


def test_script_and_style_content_is_dropped():
    assert html_to_text("<style>p{}</style><p>Hi</p><script>alert(1)</script>") == "Hi"


def test_double_escaped_markup_is_decoded():
    assert html_to_text("<p>LLM &lt;b&gt;apps&lt;/b&gt;</p>") == "LLM apps"


def test_malformed_html_does_not_raise():
    assert html_to_text("<p>unclosed <b>bold <i>text") == "unclosed bold text"


@pytest.mark.parametrize("value", [None, "", "   ", "<p> </p>", "<br><br>"])
def test_empty_descriptions_become_none(value):
    assert html_to_text(value) is None


def test_whitespace_is_normalized():
    assert normalize_whitespace("a \t b\r\n\r\n\r\n\r\nc​ \xa0 d") == "a b\n\nc d"


def test_clean_str_collapses_and_nulls():
    assert clean_str("  Berlin,\n  Germany ") == "Berlin, Germany"
    assert clean_str("") is None
    assert clean_str(None) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1792782048, "2026-10-23T19:00:48"),
        ("1792782048", "2026-10-23T19:00:48"),
        (1792782048000, "2026-10-23T19:00:48"),
        ("2026-10-17T16:04:29", "2026-10-17T16:04:29+00:00"),  # naive -> UTC
        ("2026-10-16T10:00:00+02:00", "2026-10-16T08:00:00+00:00"),
        ("2026-10-16T10:00:00Z", "2026-10-16T10:00:00+00:00"),
    ],
)
def test_parse_datetime_valid(value, expected):
    assert parse_datetime(value).startswith(expected)


@pytest.mark.parametrize("value", [None, "", "not a date", True, "99999999999999999999"])
def test_parse_datetime_invalid_is_none(value):
    assert parse_datetime(value) is None


@pytest.mark.parametrize(
    ("labels", "title", "expected"),
    [
        (["Unbefristeter arbeitsvertrag", "berufserfahren"], "Data Scientist", "permanent"),
        (["fulltime permanent"], "x", "full_time"),
        (["Indefinite contract (unbefristeter arbeitsvertrag)"], "x", "permanent"),
        (["Contract"], "x", "contract"),
        (["Selbstständiger / freelancer"], "x", "freelance"),
        (["Part Time"], "x", "part_time"),
        (["Teilzeitbeschäftigung"], "x", "part_time"),
        (["Cdd - contrat à durée déterminée"], "x", "temporary"),
        (["Experienced", "Mid", "entry"], "x", None),  # experience levels are not job types
        ([], "x", None),
        (None, "x", None),
        ("full_time", "x", "full_time"),
        (["Full Time"], "Winter Data Internship", "internship"),  # title beats labels
        (["Full Time"], "Pflichtpraktikum Controlling", "internship"),
        (["Full Time"], "Werkstudent (m/w/d) Controlling", "working_student"),
        (["Full Time"], "Internal Tools Engineer", "full_time"),  # "Internal" is not "Intern"
    ],
)
def test_canonical_job_type(labels, title, expected):
    assert canonical_job_type(labels, title) == expected


def test_canonical_url_ignores_noise():
    a = "https://Example.com/jobs/1/?utm_source=x&ref=y#apply"
    b = "https://example.com/jobs/1"
    assert canonical_url(a) == canonical_url(b)
    assert canonical_url("https://example.com/jobs/1?id=2") != canonical_url(b)
