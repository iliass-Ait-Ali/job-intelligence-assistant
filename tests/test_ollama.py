"""The Ollama client, exercised against a real HTTP server on 127.0.0.1 that mimics the Ollama API."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from job_intel.rag.ollama import LLMOutputError, LLMPrivacyError, LLMTimeout, LLMUnavailable, OllamaClient

SCHEMA = {"type": "object", "properties": {"strengths": {"type": "array"}}}


class FakeOllama:
    def __init__(self, models=("llama3.2:3b",), chat_status=200, chat_body=None, delay=0.0):
        self.models, self.chat_status, self.delay = list(models), chat_status, delay
        self.chat_body = chat_body if chat_body is not None else {"message": {"content": json.dumps({"strengths": []})}}
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, status, payload):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                time.sleep(outer.delay)
                self._send(outer.chat_status, outer.chat_body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    started = []

    def make(**kw):
        s = FakeOllama(**kw)
        started.append(s)
        return s

    yield make
    for s in started:
        s.close()


def test_check_succeeds_when_the_model_is_installed(server):
    s = server()
    assert "available" in OllamaClient(base_url=s.url).check()
    assert OllamaClient(base_url=s.url).available_models() == ["llama3.2:3b"]


def test_check_reports_a_missing_model_with_the_pull_command(server):
    s = server(models=["mistral:7b"])
    with pytest.raises(LLMUnavailable, match=r"ollama pull llama3\.2:3b"):
        OllamaClient(base_url=s.url).check()


def test_untagged_model_names_match_latest(server):
    s = server(models=["llama3.1:latest"])
    assert "available" in OllamaClient(model="llama3.1", base_url=s.url).check()


def test_server_not_running_is_a_friendly_error():
    client = OllamaClient(base_url="http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(LLMUnavailable, match="not running"):
        client.check()
    with pytest.raises(LLMUnavailable, match="not reachable"):
        client.chat_json("s", "u", SCHEMA)


def test_chat_sends_a_deterministic_schema_constrained_request_and_parses_json(server):
    s = server(chat_body={"message": {"content": json.dumps({"strengths": [{"claim": "x"}]})}})
    out = OllamaClient(base_url=s.url, num_ctx=2048).chat_json("system text", "user text", SCHEMA)
    assert out == {"strengths": [{"claim": "x"}]}
    sent = s.requests[0]
    assert sent["model"] == "llama3.2:3b" and sent["stream"] is False and sent["format"] == SCHEMA
    assert sent["options"] == {"temperature": 0, "seed": 0, "num_ctx": 2048}
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


@pytest.mark.parametrize("body", [{"message": {"content": "not json"}}, {"message": {"content": "[1, 2]"}}, {"nope": 1}, {"message": {}}])
def test_unusable_model_output_raises_output_error(server, body):
    s = server(chat_body=body)
    with pytest.raises(LLMOutputError):
        OllamaClient(base_url=s.url).chat_json("s", "u", SCHEMA)


def test_unknown_model_and_server_errors(server):
    missing = server(chat_status=404, chat_body={"error": "model not found"})
    with pytest.raises(LLMUnavailable, match="not found"):
        OllamaClient(base_url=missing.url).chat_json("s", "u", SCHEMA)
    broken = server(chat_status=500, chat_body={"error": "boom"})
    with pytest.raises(LLMUnavailable, match="HTTP 500"):
        OllamaClient(base_url=broken.url).chat_json("s", "u", SCHEMA)


def test_slow_model_times_out_with_a_clear_message(server):
    s = server(delay=1.5)
    with pytest.raises(LLMTimeout, match="did not answer"):
        OllamaClient(base_url=s.url, timeout=0.3).chat_json("s", "u", SCHEMA)


@pytest.mark.parametrize("url", ["http://example.com:11434", "https://api.openai.com", "http://192.168.1.20:11434", "http://10.0.0.5"])
def test_non_local_servers_are_refused_before_any_request(url):
    with pytest.raises(LLMPrivacyError, match="this computer"):
        OllamaClient(base_url=url)


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_loopback_servers_are_accepted(url):
    assert OllamaClient(base_url=url).base_url == url
