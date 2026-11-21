"""API client and runner against a scripted fake session (no network)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import requests

from hackathon.common.config import Settings
from hackathon.model_equality.client import (
    MAX_PARALLEL,
    ApiClient,
    BudgetExhaustedError,
    BudgetPolicy,
    FatalQueryError,
    QueryError,
    QueryJob,
    retry_after_seconds,
    run_jobs,
)
from hackathon.model_equality.store import Answer, AnswerStore


class FakeReply:
    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


def ok(content: str = "answer", used: int = 1) -> FakeReply:
    return FakeReply(200, {"choices": [{"message": {"content": content}}],
                           "budget": {"used": used, "limit": 100}})


class ScriptedSession:
    """Pops one scripted outcome (reply or exception) per POST; GET returns `models`."""

    def __init__(self, posts: list[FakeReply | Exception], models: Any = None) -> None:
        self.posts = posts
        self.models = models
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, timeout: float) -> FakeReply:
        return FakeReply(200, self.models)

    def post(self, url: str, *, json: Any, timeout: float) -> FakeReply:
        self.calls.append(json)
        outcome = self.posts.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(session: ScriptedSession, sleeps: list[float] | None = None) -> ApiClient:
    settings = Settings(api_key="k", server_url="http://s", hf_token=None, data_dir=Path("."),
                        wmloc_task_id=None)
    record = sleeps if sleeps is not None else []
    return ApiClient(settings, session=session, sleep=record.append)


def test_list_budgets_parses_limit_used_and_uncapped_reference() -> None:
    session = ScriptedSession([], models={"data": [
        {"id": "ep-01", "budget": {"limit": 100, "used": 30}},
        {"id": "ref-01", "budget": {"limit": None, "used": 7}},
    ]})
    budgets = make_client(session).list_budgets()
    assert budgets["ep-01"].left == 70
    assert budgets["ref-01"].left is None and budgets["ref-01"].used == 7


def test_list_budgets_rejects_a_malformed_body() -> None:
    with pytest.raises(QueryError, match="unexpected body"):
        make_client(ScriptedSession([], models={"nope": 1})).list_budgets()


def test_chat_sends_the_documented_payload() -> None:
    session = ScriptedSession([ok("hello")])
    result = make_client(session).chat("ep-03", "Hi", 17)
    assert result.content == "hello" and result.used == 1 and result.limit == 100
    assert session.calls == [{"model": "ep-03", "max_tokens": 17,
                              "messages": [{"role": "user", "content": "Hi"}]}]


def test_busy_429_waits_retry_after_then_succeeds() -> None:
    sleeps: list[float] = []
    session = ScriptedSession([FakeReply(429, {"detail": "busy"}, {"Retry-After": "3"}), ok()])
    make_client(session, sleeps).chat("ep-01", "p")
    assert sleeps == [3.0]
    assert len(session.calls) == 2


def test_429_without_retry_after_means_budget_exhausted() -> None:
    session = ScriptedSession([FakeReply(429, {"detail": "budget used up"})])
    with pytest.raises(BudgetExhaustedError, match="budget used up"):
        make_client(session).chat("ep-01", "p")


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_errors_are_retried_after_15_seconds(status: int) -> None:
    sleeps: list[float] = []
    session = ScriptedSession([FakeReply(status, {"detail": "down"}), ok()])
    make_client(session, sleeps).chat("ep-01", "p")
    assert sleeps == [15.0]


def test_network_error_is_retried_after_30_seconds() -> None:
    sleeps: list[float] = []
    session = ScriptedSession([requests.ConnectionError("boom"), ok()])
    make_client(session, sleeps).chat("ep-01", "p")
    assert sleeps == [30.0]


def test_persistent_gateway_errors_eventually_raise() -> None:
    session = ScriptedSession([FakeReply(503, {"detail": "down"})] * 40)
    with pytest.raises(QueryError, match="503"):
        make_client(session).chat("ep-01", "p")


def test_auth_failure_is_fatal_and_other_4xx_are_plain_errors() -> None:
    with pytest.raises(FatalQueryError):
        make_client(ScriptedSession([FakeReply(401, {"detail": "bad key"})])).chat("ep-01", "p")
    with pytest.raises(QueryError) as info:
        make_client(ScriptedSession([FakeReply(422, {"detail": "invalid"})])).chat("ep-01", "p")
    assert not isinstance(info.value, (FatalQueryError, BudgetExhaustedError))


def test_malformed_200_body_and_bad_max_tokens_are_errors() -> None:
    with pytest.raises(QueryError, match="malformed"):
        make_client(ScriptedSession([FakeReply(200, {"choices": []})])).chat("ep-01", "p")
    with pytest.raises(QueryError, match="max_tokens"):
        make_client(ScriptedSession([])).chat("ep-01", "p", 999)


def test_retry_after_parsing_is_clamped_and_forgiving() -> None:
    assert retry_after_seconds({"Retry-After": "2"}) == 2.0
    assert retry_after_seconds({"Retry-After": "-5"}) == 0.0
    assert retry_after_seconds({"Retry-After": "99999"}) == 300.0
    assert retry_after_seconds({"Retry-After": "soon"}) == 5.0
    assert retry_after_seconds({}) == 5.0


def test_budget_policy_keeps_the_reserve_and_honours_the_server_view() -> None:
    policy = BudgetPolicy(limit=100, reserve=10)
    assert policy.allowance(stored=0, server_left=None) == 90
    assert policy.allowance(stored=85, server_left=None) == 5
    assert policy.allowance(stored=95, server_left=None) == 0
    # a teammate spent budget that is not in our copy of the store yet
    assert policy.allowance(stored=10, server_left=40) == 30
    assert policy.allowance(stored=0, server_left=5) == 0


def make_jobs(endpoint: str, count: int) -> list[QueryJob]:
    return [QueryJob(endpoint, f"p{i}", 0, f"prompt {i}") for i in range(count)]


def test_runner_stores_answers_and_never_asks_a_stored_key_again(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "a.jsonl")
    store.append(Answer("ep-01", "p0", 0, "prompt 0", "already here"))
    session = ScriptedSession([ok(f"r{i}") for i in range(2)])
    report = run_jobs(make_client(session), store, make_jobs("ep-01", 3), caps={"ep-01": 10},
                      max_parallel=1)
    assert report.stored == 2 and report.already_stored == 1
    assert len(session.calls) == 2
    assert {key[1] for key in store.load()} == {"p0", "p1", "p2"}
    assert store.load()[("ep-01", "p0", 0)].response == "already here"


def test_runner_hard_cap_stops_before_sending(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "a.jsonl")
    session = ScriptedSession([ok() for _ in range(10)])
    report = run_jobs(make_client(session), store, make_jobs("ep-01", 6), caps={"ep-01": 2},
                      max_parallel=2)
    assert len(session.calls) == 2
    assert report.stored == 2 and report.skipped_cap == 4


def test_runner_endpoint_missing_from_caps_gets_nothing(tmp_path: Path) -> None:
    session = ScriptedSession([])
    report = run_jobs(make_client(session), AnswerStore(tmp_path / "a.jsonl"),
                      make_jobs("ep-02", 2), caps={"ep-01": 5})
    assert session.calls == [] and report.skipped_cap == 2


def test_runner_marks_an_exhausted_endpoint_and_skips_its_remaining_jobs(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "a.jsonl")
    session = ScriptedSession([FakeReply(429, {"detail": "budget used up"})])
    report = run_jobs(make_client(session), store, make_jobs("ep-01", 4), caps=None,
                      max_parallel=1)
    assert report.exhausted_endpoints == {"ep-01"}
    assert report.skipped_exhausted == 4 and report.stored == 0
    assert len(session.calls) == 1


def test_runner_reports_failures_without_stopping(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "a.jsonl")
    session = ScriptedSession([FakeReply(422, {"detail": "invalid"}), ok("fine")])
    report = run_jobs(make_client(session), store, make_jobs("ep-01", 2), caps=None, max_parallel=1)
    assert report.stored == 1 and len(report.failed) == 1 and "invalid" in report.failed[0]


def test_runner_aborts_everything_on_auth_failure(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "a.jsonl")
    session = ScriptedSession([FakeReply(401, {"detail": "bad key"})])
    with pytest.raises(FatalQueryError):
        run_jobs(make_client(session), store, make_jobs("ep-01", 5), caps=None, max_parallel=1)
    assert len(session.calls) == 1 and store.load() == {}


def test_runner_never_exceeds_four_in_flight_and_rejects_more(tmp_path: Path) -> None:
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    class Concurrent:
        def get(self, url: str, *, timeout: float) -> FakeReply:
            return FakeReply(200, {"data": []})

        def post(self, url: str, *, json: Any, timeout: float) -> FakeReply:
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.01)
            with lock:
                state["now"] -= 1
            return ok()

    settings = Settings(api_key="k", server_url="http://s", hf_token=None,
                        data_dir=tmp_path, wmloc_task_id=None)
    client = ApiClient(settings, session=Concurrent())
    store = AnswerStore(tmp_path / "a.jsonl")
    report = run_jobs(client, store, make_jobs("ep-01", 24), caps=None)
    assert report.stored == 24
    assert 1 < state["peak"] <= MAX_PARALLEL
    with pytest.raises(ValueError, match="max_parallel"):
        run_jobs(client, store, [], max_parallel=MAX_PARALLEL + 1)


def test_runner_does_not_send_the_same_key_twice_within_one_run(tmp_path: Path) -> None:
    session = ScriptedSession([ok(), ok()])
    jobs = [QueryJob("ep-01", "p0", 0, "x"), QueryJob("ep-01", "p0", 0, "x")]
    report = run_jobs(make_client(session), AnswerStore(tmp_path / "a.jsonl"), jobs, caps=None,
                      max_parallel=1)
    assert len(session.calls) == 1 and report.already_stored == 1


def test_no_key_is_required_when_a_session_is_injected() -> None:
    settings = Settings(api_key=None, server_url="http://s", hf_token=None, data_dir=Path("."),
                        wmloc_task_id=None)
    ApiClient(settings, session=ScriptedSession([]))
    sleeper: Callable[[float], None] = lambda _seconds: None
    with pytest.raises(Exception, match="HACKATHON_API_KEY"):
        ApiClient(settings, sleep=sleeper)
