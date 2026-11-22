"""Busy-aware session for the model-equality client (team D).

Observed on 21/11/2026: when the team already has 4 requests in flight (a teammate querying the
same server), the server answers `429 {"detail": "Your team already has 4 requests open; wait for
one to finish."}` WITHOUT a `Retry-After` header. The kit's `ApiClient` reads a 429 without that
header as "budget of this endpoint exhausted" and stops querying the endpoint, so a collector that
shares the team slots with a teammate silently gives up on every endpoint.

`BusyAwareSession` wraps the real session and adds a jittered `Retry-After` to exactly that reply,
so the unchanged `ApiClient` waits and retries (such a reply costs no budget). A true budget
exhaustion (another detail text) is left untouched.
"""

from __future__ import annotations

import hashlib
import json as json_module
import random
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import requests

from hackathon.common.config import Settings
from hackathon.model_equality.client import (
    HTTP_OK,
    HTTP_TOO_MANY_REQUESTS,
    MAX_NEW_TOKENS,
    RETRY_AFTER_HEADER,
    ApiClient,
    ChatResult,
    FatalQueryError,
    HttpReply,
    HttpSession,
)

TEAM_BUSY_MARKER = "requests open"
"""Substring of the detail text of the concurrency 429."""
BUSY_RETRY_MIN_S = 10.0
BUSY_RETRY_MAX_S = 15.0
"""Jitter range of the injected wait (asked by the director: 10 to 15 s, no hammering)."""
BUSY_RETRY_SEED = 20261121
STANDARD_HEADERS = frozenset({"server", "date", "content-type", "connection", "vary", "content-length"})
"""Headers every reply carries (nginx); anything else is recorded in the sidecar as news."""


class BusyReply:
    """A reply whose headers carry an injected `Retry-After`."""

    def __init__(self, inner: HttpReply, retry_after_s: float) -> None:
        """Wrap `inner`; `retry_after_s` is exposed as the `Retry-After` header."""
        self._inner = inner
        self._headers: dict[str, str] = {**inner.headers, RETRY_AFTER_HEADER: f"{retry_after_s:.1f}"}

    @property
    def status_code(self) -> int:
        """HTTP status code of the wrapped reply."""
        return self._inner.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        """Headers of the wrapped reply plus the injected `Retry-After`."""
        return self._headers

    @property
    def text(self) -> str:
        """Raw body text of the wrapped reply."""
        return self._inner.text

    def json(self) -> Any:
        """Decoded JSON body of the wrapped reply."""
        return self._inner.json()


def is_team_busy(reply: HttpReply) -> bool:
    """True for the concurrency 429 (no `Retry-After`, detail mentions open requests)."""
    return (
        reply.status_code == HTTP_TOO_MANY_REQUESTS
        and RETRY_AFTER_HEADER not in reply.headers
        and TEAM_BUSY_MARKER in reply.text
    )


class BusyAwareSession:
    """`HttpSession` that turns the concurrency 429 into a retryable reply.

    When `sidecar` is given, every successful completion is also described in a JSONL file: the raw
    fields of the reply (id, finish_reason, usage, extra headers), the client-side latency of that
    request and its UTC send / receive times. The official answers store is not touched; the two
    files join on (endpoint, prompt, sha1 of the reply).
    """

    def __init__(self, inner: HttpSession, sidecar: Path | None = None) -> None:
        """Wrap `inner` (a `requests.Session` carrying the API key, or a test fake)."""
        self._inner = inner
        self._rng = random.Random(BUSY_RETRY_SEED)
        self._sidecar = sidecar
        self._sidecar_lock = threading.Lock()

    def get(self, url: str, *, timeout: float) -> HttpReply:
        """HTTP GET, unchanged."""
        return self._inner.get(url, timeout=timeout)

    def post(self, url: str, *, json: Any, timeout: float) -> HttpReply:
        """HTTP POST; the concurrency 429 gets a jittered `Retry-After`."""
        sent = datetime.now(UTC)
        started = time.perf_counter()
        reply = self._inner.post(url, json=json, timeout=timeout)
        latency = time.perf_counter() - started
        if is_team_busy(reply):
            return BusyReply(reply, self._rng.uniform(BUSY_RETRY_MIN_S, BUSY_RETRY_MAX_S))
        if self._sidecar is not None and reply.status_code == HTTP_OK:
            self._record(json, reply, sent, latency)
        return reply

    def _record(self, payload: Any, reply: HttpReply, sent: datetime, latency: float) -> None:
        """Append one raw-fields row to the sidecar (never raises on a malformed body)."""
        try:
            body = reply.json()
            choice = body["choices"][0]
            content = str(choice["message"]["content"])
            prompt = str(payload["messages"][0]["content"])
        except (KeyError, IndexError, TypeError, ValueError):
            return
        row = {
            "endpoint": payload.get("model"),
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "content_sha1": hashlib.sha1(content.encode("utf-8")).hexdigest(),
            "content_chars": len(content),
            "sent_utc": sent.isoformat(timespec="milliseconds"),
            "received_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "latency_s": round(latency, 3),
            "max_tokens": payload.get("max_tokens"),
            "id": body.get("id"),
            "created": body.get("created"),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "body_keys": sorted(body),
            "choice_keys": sorted(choice),
            "budget_used": (body.get("budget") or {}).get("used"),
            "extra_headers": {k: v for k, v in dict(reply.headers).items() if k.lower() not in STANDARD_HEADERS},
            "content_length_header": dict(reply.headers).get("Content-Length"),
            "body_model": body.get("model"),
            "all_headers": dict(reply.headers),
        }
        sidecar = self._sidecar
        if sidecar is None:
            return
        with self._sidecar_lock, sidecar.open("a", encoding="utf-8") as handle:
            handle.write(json_module.dumps(row, ensure_ascii=False) + "\n")


class StoppableClient(ApiClient):
    """`ApiClient` that stops sending as soon as a stop file exists, without abandoning requests.

    Killing a collector (TaskStop, Ctrl-C) while requests are in flight leaves them open on the server,
    where they keep occupying the team's 4 slots. Creating the stop file instead makes every worker
    refuse its NEXT request (a `FatalQueryError`, which the kit's runner turns into a clean abort),
    so the requests already sent all complete before the process exits.
    """

    def __init__(self, *args: Any, stop_file: Path | None = None, **kwargs: Any) -> None:
        """Same arguments as `ApiClient`, plus the path of the stop file."""
        super().__init__(*args, **kwargs)
        self._stop_file = stop_file

    def chat(self, endpoint: str, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> ChatResult:
        """Send one request unless the stop file exists.

        Raises:
            FatalQueryError: The stop file exists (graceful stop, nothing was sent).
        """
        if self._stop_file is not None and self._stop_file.exists():
            raise FatalQueryError(f"stop file {self._stop_file} present: stopping without sending")
        return super().chat(endpoint, prompt, max_tokens)


def make_busy_aware_client(
    settings: Settings,
    log: Callable[[str], None] | None = None,
    sidecar: Path | None = None,
    stop_file: Path | None = None,
) -> ApiClient:
    """An `ApiClient` whose session survives teammates sharing the 4 request slots."""
    real = requests.Session()
    real.headers["X-API-Key"] = settings.require_api_key()
    return StoppableClient(settings, session=BusyAwareSession(cast(HttpSession, real), sidecar), log=log,
                           stop_file=stop_file)
