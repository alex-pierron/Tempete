"""Typed client of the model-equality API, with a budget-safe parallel runner.

Protocol (copied from the official `task_template.py`):
    base            {server}/33-model-equality/v1, header `X-API-Key`.
    GET  /models    -> {"data": [{"id": ep, "budget": {"limit": n, "used": n}}, ...]}
    POST /chat/completions {"model", "max_tokens", "messages": [user]}
                    -> {"choices": [{"message": {"content"}}], "budget": {...}}
    429 + Retry-After   endpoint busy: wait, then retry (not an error, costs nothing)
    429 without it      budget of that endpoint exhausted: stop querying it
    502 / 503 / 504     transient: wait, retry
    other 4xx           invalid request (rejected, not counted by the server): reported, skipped

Budget safety is layered because `ep-*` answers are irreplaceable (100 per endpoint for the whole
competition, shared by the team):

1. the plan (see `plan.py`) truncates jobs to an allowance that keeps a configurable reserve;
2. the runner re-checks a per-endpoint cap atomically before every send (defence in depth);
3. the store refuses any (endpoint, id, repeat) it already holds, so nothing is asked twice.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import requests

from hackathon.common.config import TASK_MODEL_EQUALITY, Settings
from hackathon.common.errors import HackathonError
from hackathon.model_equality.store import Answer, AnswerKey, AnswerStore, utc_timestamp

MAX_PARALLEL = 4
"""Server limit: at most 4 simultaneous requests per team."""
MAX_NEW_TOKENS = 256
"""Server cap on generated tokens per answer."""
MAX_PROMPT_TOKENS = 1024
"""Server cap on prompt tokens; an over-long prompt is rejected by the server and not counted."""
ENDPOINT_LIMIT_DEFAULT = 100
"""Answers per endpoint for the whole competition."""
DEFAULT_RESERVE = 10
"""Answers per `ep-*` endpoint kept unspent by default (10 % of the budget)."""

HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429
HTTP_GATEWAY_STATUSES = frozenset({502, 503, 504})
RETRY_AFTER_HEADER = "Retry-After"
MODELS_TIMEOUT_S = 60.0
CHAT_TIMEOUT_S = 180.0
NETWORK_RETRY_WAIT_S = 30.0
GATEWAY_RETRY_WAIT_S = 15.0
RETRY_AFTER_MIN_S = 0.0
RETRY_AFTER_MAX_S = 300.0
RETRY_AFTER_FALLBACK_S = 5.0
MAX_TRANSIENT_ATTEMPTS = 20
"""Network errors / 5xx tolerated per request before giving up (about 5-10 minutes)."""
MAX_BUSY_WAITS = 1000
"""429-with-Retry-After waits tolerated per request: the server is shared, queues can be long."""
ERROR_DETAIL_CHARS = 300
PROGRESS_EVERY = 20


class QueryError(HackathonError):
    """A request failed in a way that is not retried."""


class BudgetExhaustedError(QueryError):
    """The endpoint answered 429 without Retry-After: its budget is used up."""


class FatalQueryError(QueryError):
    """Authentication failed or the server is unusable: the whole run must stop."""


class HttpReply(Protocol):
    """The subset of `requests.Response` the client reads."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""
        ...

    @property
    def text(self) -> str:
        """Raw body text."""
        ...

    def json(self) -> Any:
        """Decode the JSON body."""
        ...


class HttpSession(Protocol):
    """The subset of `requests.Session` the client uses (a fake can stand in during tests)."""

    def get(self, url: str, *, timeout: float) -> HttpReply:
        """HTTP GET."""
        ...

    def post(self, url: str, *, json: Any, timeout: float) -> HttpReply:
        """HTTP POST with a JSON body."""
        ...


@dataclass(frozen=True)
class EndpointBudget:
    """Server-side budget of one endpoint (`limit` is None when the server reports no cap)."""

    endpoint: str
    limit: int | None
    used: int

    @property
    def left(self) -> int | None:
        """Answers still available on the server, or None for an uncapped endpoint."""
        return None if self.limit is None else max(0, self.limit - self.used)


@dataclass(frozen=True)
class ChatResult:
    """One successful completion."""

    content: str
    used: int | None
    limit: int | None


@dataclass(frozen=True)
class BudgetPolicy:
    """How many new answers an endpoint may receive.

    Attributes:
        limit: Answers per endpoint allowed by the competition.
        reserve: Answers never to spend (kept for end-game decisions and for accidents).
    """

    limit: int = ENDPOINT_LIMIT_DEFAULT
    reserve: int = DEFAULT_RESERVE

    def allowance(self, stored: int, server_left: int | None) -> int:
        """New answers allowed for an endpoint.

        Both views are enforced: what the local store says is already spent, and what the server
        says is left (a teammate may have spent budget that is not in our copy of the store yet).
        """
        by_store = self.limit - stored - self.reserve
        if server_left is None:
            return max(0, by_store)
        return max(0, min(by_store, server_left - self.reserve))


def retry_after_seconds(headers: Mapping[str, str]) -> float:
    """Parse a Retry-After header (seconds), clamped; a missing or odd value gives a fallback."""
    raw = headers.get(RETRY_AFTER_HEADER)
    if raw is None:
        return RETRY_AFTER_FALLBACK_S
    try:
        value = float(raw)
    except ValueError:
        return RETRY_AFTER_FALLBACK_S
    return min(max(value, RETRY_AFTER_MIN_S), RETRY_AFTER_MAX_S)


def error_detail(reply: HttpReply) -> str:
    """Human-readable error text of a non-200 reply."""
    try:
        body = reply.json()
    except ValueError:
        return reply.text[:ERROR_DETAIL_CHARS]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:ERROR_DETAIL_CHARS]
    return reply.text[:ERROR_DETAIL_CHARS]


class ApiClient:
    """Client of one server. Network time is injectable (`sleep`) so tests never wait.

    Args:
        settings: Resolved hackathon settings (the API key is required on first use).
        session: HTTP session; defaults to a `requests.Session` carrying the API key.
        sleep: Function used to wait between retries.
        log: Optional sink for one-line operational messages.
    """

    def __init__(
        self,
        settings: Settings,
        session: HttpSession | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """Build the client; the key is only demanded when no session is injected."""
        self.base_url = f"{settings.server_url.rstrip('/')}/{TASK_MODEL_EQUALITY}/v1"
        if session is None:
            real = requests.Session()
            real.headers["X-API-Key"] = settings.require_api_key()
            session = cast(HttpSession, real)
        self._session: HttpSession = session
        self._sleep = sleep
        self._log = log or (lambda _message: None)

    def list_budgets(self) -> dict[str, EndpointBudget]:
        """GET /models: the server-side budget of every endpoint (costs no budget).

        Raises:
            QueryError: Non-200 reply or an unexpected body.
        """
        try:
            reply = self._session.get(f"{self.base_url}/models", timeout=MODELS_TIMEOUT_S)
        except requests.RequestException as exc:
            raise QueryError(f"GET /models failed: {exc}") from exc
        if reply.status_code != HTTP_OK:
            raise QueryError(f"GET /models: HTTP {reply.status_code}: {error_detail(reply)}")
        try:
            budgets: dict[str, EndpointBudget] = {}
            for model in reply.json()["data"]:
                budget = model.get("budget") or {}
                limit = budget.get("limit")
                budgets[str(model["id"])] = EndpointBudget(
                    endpoint=str(model["id"]),
                    limit=None if limit is None else int(limit),
                    used=int(budget.get("used") or 0),
                )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise QueryError(f"GET /models: unexpected body ({exc!r})") from exc
        return budgets

    def chat(self, endpoint: str, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> ChatResult:
        """Ask one endpoint for one answer, handling busy / transient / exhausted replies.

        Raises:
            BudgetExhaustedError: 429 without Retry-After.
            FatalQueryError: 401 / 403.
            QueryError: Invalid request, malformed body, or retries exhausted.
        """
        if not 1 <= max_tokens <= MAX_NEW_TOKENS:
            raise QueryError(f"max_tokens must be in [1, {MAX_NEW_TOKENS}], got {max_tokens}")
        payload = {
            "model": endpoint,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        transient_attempts = 0
        busy_waits = 0
        while True:
            try:
                reply = self._session.post(
                    f"{self.base_url}/chat/completions", json=payload, timeout=CHAT_TIMEOUT_S
                )
            except requests.RequestException as exc:
                transient_attempts += 1
                if transient_attempts >= MAX_TRANSIENT_ATTEMPTS:
                    raise QueryError(f"{endpoint}: server unreachable: {exc}") from exc
                self._log(f"{endpoint}: unreachable ({type(exc).__name__}); retry in 30 s")
                self._sleep(NETWORK_RETRY_WAIT_S)
                continue
            status = reply.status_code
            if status == HTTP_OK:
                return self._parse_chat(endpoint, reply)
            if status == HTTP_TOO_MANY_REQUESTS:
                if RETRY_AFTER_HEADER in reply.headers:
                    busy_waits += 1
                    if busy_waits >= MAX_BUSY_WAITS:
                        raise QueryError(f"{endpoint}: still busy after {busy_waits} waits")
                    self._sleep(retry_after_seconds(reply.headers))
                    continue
                raise BudgetExhaustedError(f"{endpoint}: {error_detail(reply)}")
            if status in HTTP_GATEWAY_STATUSES:
                transient_attempts += 1
                if transient_attempts >= MAX_TRANSIENT_ATTEMPTS:
                    raise QueryError(f"{endpoint}: HTTP {status} {error_detail(reply)}")
                self._log(f"{endpoint}: HTTP {status}; retry in 15 s")
                self._sleep(GATEWAY_RETRY_WAIT_S)
                continue
            if status in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                raise FatalQueryError(f"HTTP {status}: {error_detail(reply)} (check the API key)")
            raise QueryError(f"{endpoint}: HTTP {status}: {error_detail(reply)}")

    @staticmethod
    def _parse_chat(endpoint: str, reply: HttpReply) -> ChatResult:
        """Extract the completion text and the budget counters from a 200 reply."""
        try:
            body = reply.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise QueryError(f"{endpoint}: malformed completion body ({exc!r})") from exc
        budget = body.get("budget") if isinstance(body, dict) else None
        used = budget.get("used") if isinstance(budget, dict) else None
        limit = budget.get("limit") if isinstance(budget, dict) else None
        return ChatResult(
            content="" if content is None else str(content),
            used=int(used) if isinstance(used, int) else None,
            limit=int(limit) if isinstance(limit, int) else None,
        )


@dataclass(frozen=True)
class QueryJob:
    """One answer to fetch."""

    endpoint: str
    prompt_id: str
    repeat: int
    prompt: str
    max_tokens: int = MAX_NEW_TOKENS

    @property
    def key(self) -> AnswerKey:
        """The (endpoint, id, repeat) identity."""
        return (self.endpoint, self.prompt_id, self.repeat)


@dataclass
class RunReport:
    """Counts of one runner execution (rows in, stored, skipped, failed)."""

    jobs_in: int = 0
    stored: int = 0
    already_stored: int = 0
    skipped_cap: int = 0
    skipped_exhausted: int = 0
    skipped_aborted: int = 0
    failed: list[str] = field(default_factory=list)
    exhausted_endpoints: set[str] = field(default_factory=set)

    def summary(self) -> str:
        """One-line summary for the CLI."""
        return (
            f"jobs={self.jobs_in} stored={self.stored} already_stored={self.already_stored} "
            f"skipped_cap={self.skipped_cap} skipped_exhausted={self.skipped_exhausted} "
            f"skipped_aborted={self.skipped_aborted} failed={len(self.failed)}"
        )


def run_jobs(
    client: ApiClient,
    store: AnswerStore,
    jobs: Sequence[QueryJob],
    *,
    caps: Mapping[str, int] | None = None,
    max_parallel: int = MAX_PARALLEL,
    progress: Callable[[str], None] | None = None,
) -> RunReport:
    """Fetch every job with at most `max_parallel` requests in flight and store each answer.

    Args:
        client: The API client.
        store: Destination store; answers are appended (and flushed) one by one, so an
            interruption loses at most the in-flight requests.
        jobs: Jobs in the order they should be attempted (round-robin over endpoints so a run
            that stops early has covered all of them evenly).
        caps: New answers allowed per endpoint during this run (None = unlimited, for `ref-*`).
            An endpoint missing from a non-None mapping gets 0.
        max_parallel: Concurrency, at most `MAX_PARALLEL`.
        progress: Optional one-line message sink.

    Returns:
        The run report.

    Raises:
        ValueError: `max_parallel` out of [1, MAX_PARALLEL].
        FatalQueryError: Authentication failure (re-raised once in-flight work has stopped).
    """
    if not 1 <= max_parallel <= MAX_PARALLEL:
        raise ValueError(f"max_parallel must be in [1, {MAX_PARALLEL}], got {max_parallel}")
    report = RunReport(jobs_in=len(jobs))
    known = store.keys()
    lock = threading.Lock()
    sent: dict[str, int] = {}
    claimed: set[AnswerKey] = set()
    abort = threading.Event()
    fatal: list[FatalQueryError] = []
    emit = progress or (lambda _message: None)

    def reserve_slot(job: QueryJob) -> bool:
        """Atomically claim one send for the job's endpoint, or report why not."""
        with lock:
            if job.key in known or job.key in claimed:
                report.already_stored += 1
                return False
            if job.endpoint in report.exhausted_endpoints:
                report.skipped_exhausted += 1
                return False
            if caps is not None and sent.get(job.endpoint, 0) >= caps.get(job.endpoint, 0):
                report.skipped_cap += 1
                return False
            sent[job.endpoint] = sent.get(job.endpoint, 0) + 1
            claimed.add(job.key)
            return True

    def run_one(job: QueryJob) -> None:
        if abort.is_set():
            with lock:
                report.skipped_aborted += 1
            return
        if not reserve_slot(job):
            return
        try:
            result = client.chat(job.endpoint, job.prompt, job.max_tokens)
        except BudgetExhaustedError as exc:
            with lock:
                if job.endpoint not in report.exhausted_endpoints:
                    report.exhausted_endpoints.add(job.endpoint)
                    emit(str(exc))
                report.skipped_exhausted += 1
            return
        except FatalQueryError as exc:
            with lock:
                fatal.append(exc)
            abort.set()
            return
        except QueryError as exc:
            with lock:
                report.failed.append(f"{job.endpoint} {job.prompt_id}#{job.repeat}: {exc}")
            return
        answer = Answer(job.endpoint, job.prompt_id, job.repeat, job.prompt, result.content,
                        utc_timestamp())
        written = store.append(answer, known)
        with lock:
            report.stored += int(written)
            if report.stored % PROGRESS_EVERY == 0 and written:
                emit(f"{report.stored}/{len(jobs)} answers "
                     f"({job.endpoint}: {result.used}/{result.limit} used)")

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        list(pool.map(run_one, jobs))
    if fatal:
        raise fatal[0]
    return report
