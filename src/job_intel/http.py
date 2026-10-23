"""A small, polite HTTP client: identifies itself, always times out, spaces requests, retries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_S = 60.0


class FetchError(RuntimeError):
    """A request could not be completed or returned something unusable."""


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        *,
        timeout: tuple[float, float] = (5.0, 30.0),
        min_interval: float = 2.0,
        retries: int = 2,
        backoff: float = 2.0,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timeout = timeout
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.requests_made = 0

    def _wait_turn(self) -> None:
        if self._last_request is not None:
            remaining = self.min_interval - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        try:
            return min(float(response.headers.get("Retry-After", "")), MAX_RETRY_AFTER_S)
        except (TypeError, ValueError):
            return None

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last_problem = "no attempt made"
        for attempt in range(self.retries + 1):
            self._wait_turn()
            self.requests_made += 1
            delay = self.backoff * 2**attempt
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:
                raise FetchError(f"GET {url} failed: {exc}") from exc
            else:
                if response.status_code in RETRYABLE_STATUS:
                    last_problem = f"HTTP {response.status_code}"
                    delay = self._retry_after(response) or delay
                elif response.status_code >= 400:
                    raise FetchError(f"GET {url} returned HTTP {response.status_code}")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise FetchError(f"GET {url} returned invalid JSON: {exc}") from exc
            if attempt < self.retries:
                log.warning("GET %s: %s - retrying in %.1fs", url, last_problem, delay)
                self._sleep(delay)
        raise FetchError(f"GET {url} failed after {self.retries + 1} attempts: {last_problem}")
