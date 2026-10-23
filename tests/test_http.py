import pytest
import requests

from job_intel.http import FetchError, HttpClient


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, bad_json=False):
        self.status_code, self._payload, self.headers, self._bad = status, payload, headers or {}, bad_json

    def json(self):
        if self._bad:
            raise ValueError("boom")
        return self._payload


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes, self.headers, self.calls = list(outcomes), {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(*outcomes, **kwargs):
    sleeps = []
    session = FakeSession(*outcomes)
    client = HttpClient(
        "ua-test", session=session, sleep=sleeps.append, clock=lambda: 0.0, min_interval=0.0, **kwargs
    )
    return client, session, sleeps


def test_success_sends_user_agent_and_timeout():
    client, session, _ = make_client(FakeResponse(payload={"ok": 1}), timeout=(1, 2))
    assert client.get_json("http://x", params={"page": 1}) == {"ok": 1}
    assert session.headers["User-Agent"] == "ua-test"
    assert session.calls[0]["timeout"] == (1, 2)


def test_retries_transient_status_then_succeeds():
    client, session, sleeps = make_client(FakeResponse(503), FakeResponse(payload=[1]))
    assert client.get_json("http://x") == [1]
    assert len(session.calls) == 2 and sleeps == [2.0]


def test_retry_after_header_is_honoured_and_capped():
    client, _, sleeps = make_client(
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(429, headers={"Retry-After": "9999"}),
        FakeResponse(payload={}),
    )
    client.get_json("http://x")
    assert sleeps == [7.0, 60.0]


def test_timeouts_exhaust_retries_and_raise():
    client, session, _ = make_client(*[requests.Timeout("slow")] * 3)
    with pytest.raises(FetchError, match="after 3 attempts.*Timeout"):
        client.get_json("http://x")
    assert len(session.calls) == 3


def test_client_error_is_not_retried():
    client, session, _ = make_client(FakeResponse(404))
    with pytest.raises(FetchError, match="HTTP 404"):
        client.get_json("http://x")
    assert len(session.calls) == 1


def test_invalid_json_raises_fetch_error():
    client, _, _ = make_client(FakeResponse(bad_json=True))
    with pytest.raises(FetchError, match="invalid JSON"):
        client.get_json("http://x")


def test_requests_are_spaced_by_min_interval():
    sleeps, now = [], [0.0]
    session = FakeSession(FakeResponse(payload=1), FakeResponse(payload=2))
    client = HttpClient("ua", session=session, min_interval=2.0, sleep=sleeps.append, clock=lambda: now[0])
    client.get_json("http://x")
    now[0] = 0.5
    client.get_json("http://x")
    assert sleeps == [1.5]
