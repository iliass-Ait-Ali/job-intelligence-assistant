"""Minimal client for a LOCAL Ollama server (https://ollama.com).

Privacy: CV-derived text is only ever sent to an Ollama server on this machine. A base URL whose
host is not a loopback address is rejected before any request is made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests

LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}


def trusted_hosts() -> set[str]:
    """Loopback plus hosts the OPERATOR explicitly lists in JOBINTEL_LLM_TRUSTED_HOSTS (comma-separated), e.g. the
    `ollama` container of docker-compose: a machine they control. Public LLM APIs are never implied by this."""
    import os

    extra = {h.strip().lower() for h in os.getenv("JOBINTEL_LLM_TRUSTED_HOSTS", "").split(",") if h.strip()}
    return LOOPBACK | extra
DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"


class LLMError(RuntimeError):
    """Base class; messages are written for the end user."""


class LLMUnavailable(LLMError):
    """Ollama is not running, or the model is not installed."""


class LLMTimeout(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model answered, but not with usable JSON."""


class LLMPrivacyError(LLMError):
    """Refused to send data to a non-local server."""


@dataclass
class OllamaClient:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_URL
    timeout: float = 180.0
    num_ctx: int = 4096
    num_predict: int = 700  # cap on generated tokens: schema-constrained decoding can otherwise loop until the timeout
    session: Any = None

    def __post_init__(self) -> None:
        host = (urlsplit(self.base_url).hostname or "").lower()
        allowed = trusted_hosts()
        if host not in allowed and f"[{host}]" not in allowed:
            raise LLMPrivacyError(
                f"Refusing to use LLM server {self.base_url!r}: CV data may only be sent to Ollama on this "
                "computer (localhost / 127.0.0.1)."
            )
        self.base_url = self.base_url.rstrip("/")
        self.session = self.session or requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def available_models(self) -> list[str]:
        """Names of installed models. Raises LLMUnavailable if the server cannot be reached."""
        try:
            response = self.session.get(self._url("/api/tags"), timeout=3)
            response.raise_for_status()
            return [m.get("name", "") for m in response.json().get("models", [])]
        except requests.ConnectionError as exc:
            raise LLMUnavailable(
                f"Ollama is not running at {self.base_url}. Install it from https://ollama.com, start it, then run "
                f"`ollama pull {self.model}`. Meanwhile the deterministic explanation mode works without it."
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise LLMUnavailable(f"Ollama at {self.base_url} did not answer correctly ({type(exc).__name__}).") from exc

    def version(self) -> str | None:
        """Ollama server version, recorded in real-model audit runs (None if unavailable)."""
        try:
            return str(self.session.get(self._url("/api/version"), timeout=3).json().get("version"))
        except Exception:
            return None

    def check(self) -> str:
        """Verify the server is up and the configured model is installed; return a status message."""
        models = self.available_models()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in models and self.model not in models:
            raise LLMUnavailable(
                f"Ollama is running but model {self.model!r} is not installed (installed: {', '.join(models) or 'none'}). "
                f"Run `ollama pull {self.model}`."
            )
        return f"Ollama is running; model {self.model} is available."

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """One non-streaming chat turn constrained to a JSON schema; returns the parsed object."""
        payload = {
            "model": self.model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": self.num_predict},
        }  # fmt: skip
        try:
            response = self.session.post(self._url("/api/chat"), json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMUnavailable(f"Ollama is not reachable at {self.base_url} (is `ollama serve` running?).") from exc
        except requests.Timeout as exc:
            raise LLMTimeout(f"The local model did not answer within {self.timeout:.0f}s. Try a smaller model.") from exc
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Request to Ollama failed ({type(exc).__name__}).") from exc
        if response.status_code == 404:
            raise LLMUnavailable(f"Model {self.model!r} not found in Ollama. Run `ollama pull {self.model}`.")
        if response.status_code >= 400:
            raise LLMUnavailable(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            content = response.json()["message"]["content"]
            parsed = json.loads(content)
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMOutputError("The model did not return valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMOutputError("The model returned JSON that is not an object.")
        return parsed
