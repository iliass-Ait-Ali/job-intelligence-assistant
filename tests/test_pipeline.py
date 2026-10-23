import csv
import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import FakeClient

from job_intel import cli
from job_intel.connectors import ArbeitnowConnector, RemotiveConnector
from job_intel.http import FetchError
from job_intel.models import FIELDS
from job_intel.pipeline import build_connectors, run_collection

NOW = datetime(2026, 10, 23, 8, 0, 0, tzinfo=UTC)


def fixture_connectors(arbeitnow_pages, remotive_payload):
    arbeitnow_client, remotive_client = FakeClient(*arbeitnow_pages), FakeClient(remotive_payload)
    connectors = {
        "arbeitnow": ArbeitnowConnector(arbeitnow_client),
        "remotive": RemotiveConnector(remotive_client),
    }
    return connectors, arbeitnow_client, remotive_client


def test_counts_over_fixture_data(settings, arbeitnow_pages, remotive_payload):
    connectors, *_ = fixture_connectors(arbeitnow_pages, remotive_payload)
    report, final = run_collection(settings, connectors, now=NOW)

    assert report.total_fetched == 13  # 10 Arbeitnow (2 pages) + 3 Remotive
    assert report.invalid_skipped == 1  # the blank-title record
    assert report.normalized == 12
    assert report.filtered_out == 2  # legal business developer, copywriter
    assert report.relevant == 10
    assert report.duplicates_removed == 3  # repeated slug, repeated slug on page 2, same URL after cleanup
    assert report.final == len(final) == 7
    assert report.sources["arbeitnow"].final == 5 and report.sources["remotive"].final == 2
    assert report.by_category["data_engineer"] == 1


def test_exports_match_spec_and_preserve_attribution(settings, arbeitnow_pages, remotive_payload):
    connectors, *_ = fixture_connectors(arbeitnow_pages, remotive_payload)
    report, final = run_collection(settings, connectors, now=NOW)

    with open(report.outputs["csv"], encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(report.outputs["jsonl"], encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]

    assert len(rows) == len(records) == len(final) == 7
    assert list(rows[0]) == list(FIELDS) and list(records[0]) == list(FIELDS)
    assert {r["source"] for r in records} == {"arbeitnow", "remotive"}
    assert all(r["source_url"].lower().startswith("https://fixtures.invalid/") for r in records)
    assert len({r["id"] for r in records}) == 7 and len({r["source_url"] for r in records}) == 7
    assert all("<" not in (r["description"] or "") for r in records)  # no HTML left
    assert any(r["description"] is None for r in records)  # nulls survive as null...
    assert any(r["description"] == "" for r in rows)  # ...and as empty CSV cells
    assert {r["remote"] for r in rows} <= {"true", "false", ""}

    summary = json.loads((settings.processed_dir / "collection_summary.json").read_text(encoding="utf-8"))
    assert summary["final"] == 7 and summary["total_fetched"] == 13
    assert "remotive" in summary["attribution"]


def test_fresh_snapshot_is_reused_without_new_requests(settings, arbeitnow_pages, remotive_payload):
    connectors, arb_client, rem_client = fixture_connectors(arbeitnow_pages, remotive_payload)
    first, _ = run_collection(settings, connectors, now=NOW)

    # Clients have no responses left: any new request would raise IndexError.
    second, _ = run_collection(settings, connectors, now=NOW + timedelta(minutes=30))
    assert second.sources["remotive"].from_cache and second.sources["arbeitnow"].from_cache
    assert second.final == first.final
    assert len(rem_client.calls) == 1 and len(arb_client.calls) == 2


def test_stale_snapshot_triggers_refetch_and_force_refresh(settings, arbeitnow_pages, remotive_payload):
    connectors, _, rem_client = fixture_connectors(arbeitnow_pages, remotive_payload)
    run_collection(settings, connectors, now=NOW)

    rem_client.responses.append(remotive_payload)
    run_collection(settings, {"remotive": connectors["remotive"]}, now=NOW + timedelta(hours=7))
    assert len(rem_client.calls) == 2  # older than the 6h TTL

    rem_client.responses.append(remotive_payload)
    run_collection(settings, {"remotive": connectors["remotive"]}, now=NOW + timedelta(hours=7, minutes=1),
                   force_refresh=True)  # fmt: skip
    assert len(rem_client.calls) == 3


def test_one_failing_source_does_not_stop_the_other(settings, remotive_payload):
    connectors = {
        "arbeitnow": ArbeitnowConnector(FakeClient(FetchError("HTTP 500 after 3 attempts"))),
        "remotive": RemotiveConnector(FakeClient(remotive_payload)),
    }
    report, final = run_collection(settings, connectors, now=NOW)
    assert "HTTP 500" in report.sources["arbeitnow"].error
    assert report.sources["remotive"].error is None
    assert report.final == len(final) == 2
    assert not report.all_sources_failed and (settings.processed_dir / "jobs.csv").exists()


def test_all_sources_failing_exports_nothing(settings):
    connectors = {
        "arbeitnow": ArbeitnowConnector(FakeClient(FetchError("down"))),
        "remotive": RemotiveConnector(FakeClient(FetchError("down"))),
    }
    report, final = run_collection(settings, connectors, now=NOW)
    assert report.all_sources_failed and final == [] and report.outputs == {}
    assert not settings.processed_dir.exists()  # previous exports would never be overwritten


def test_unexpected_exception_in_one_record_is_counted_not_fatal(settings, remotive_payload):
    remotive_payload["jobs"].append({"id": 1, "title": "Data Scientist", "url": "https://u.invalid/x", "tags": 5})
    report, _ = run_collection(settings, {"remotive": RemotiveConnector(FakeClient(remotive_payload))}, now=NOW)
    assert report.invalid_skipped == 1 and report.final == 2


def test_unknown_source_is_rejected(settings):
    with pytest.raises(ValueError, match="nope"):
        build_connectors(settings, ["nope"])


# --- CLI ----------------------------------------------------------------------------------


def test_cli_collect_prints_summary(monkeypatch, tmp_path, capsys, arbeitnow_pages, remotive_payload):
    connectors, *_ = fixture_connectors(arbeitnow_pages, remotive_payload)
    monkeypatch.setattr(cli, "build_connectors", lambda settings, names: connectors)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["collect", "--data-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out

    assert code == 0
    assert "Total fetched ........... 13" in out and "FINAL records ........... 7" in out
    assert "only 7 relevant records" in out  # shortfall against the 50+ target is stated
    assert (tmp_path / "out" / "processed" / "jobs.jsonl").exists()


def test_cli_exit_code_is_nonzero_when_everything_fails(monkeypatch, tmp_path, capsys):
    connectors = {"arbeitnow": ArbeitnowConnector(FakeClient(FetchError("offline")))}
    monkeypatch.setattr(cli, "build_connectors", lambda settings, names: connectors)
    code = cli.main(["collect", "--sources", "arbeitnow", "--data-dir", str(tmp_path)])
    assert code == 1 and "FAILED: offline" in capsys.readouterr().out
