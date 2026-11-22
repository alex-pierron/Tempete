"""Team D (r-met-d): collector guards, busy-aware session, features and the AUC harness (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from hackathon.common.config import Settings
from hackathon.model_equality.client import ApiClient, QueryJob
from hackathon.model_equality.research.d_battery import (
    arithmetic_accuracy,
    facts_accuracy,
    lines_fraction,
    random_seven_fraction,
    yes_no_accuracy,
)
from hackathon.model_equality.research.d_client import (
    BUSY_RETRY_MAX_S,
    BUSY_RETRY_MIN_S,
    BusyAwareSession,
    is_team_busy,
)
from hackathon.model_equality.research.d_collect import (
    CollectError,
    build_jobs,
    validate_reference_endpoints,
)
from hackathon.model_equality.research.d_eval import (
    Design,
    Roles,
    auc_curve_for_tensor,
    center_by_cell,
    roc_auc,
)
from hackathon.model_equality.research.d_features import (
    arithmetic_correct,
    deterministic_edit_ratio,
    ends_with_period,
    first_divergence_fraction,
    table_correct_fraction,
)
from hackathon.model_equality.research.d_prompts import all_prompts, battery_wave, short_wave
from hackathon.model_equality.research.d_select import prompt_family


class FakeReply:
    def __init__(self, status: int, text: str, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        raise ValueError("no json")


class FakeSession:
    def __init__(self, reply: FakeReply) -> None:
        self.reply = reply

    def get(self, url: str, *, timeout: float) -> FakeReply:
        return self.reply

    def post(self, url: str, *, json: Any, timeout: float) -> FakeReply:
        return self.reply


BUSY_TEXT = '{"detail":"Your team already has 4 requests open; wait for one to finish."}'


def test_busy_429_gets_a_retry_after_and_budget_429_does_not() -> None:
    busy = BusyAwareSession(FakeSession(FakeReply(429, BUSY_TEXT))).post("u", json={}, timeout=1.0)
    assert is_team_busy(FakeReply(429, BUSY_TEXT))
    assert "Retry-After" in busy.headers
    assert BUSY_RETRY_MIN_S <= float(busy.headers["Retry-After"]) <= BUSY_RETRY_MAX_S
    exhausted = FakeReply(429, '{"detail":"Budget exhausted for ep-01"}')
    assert not is_team_busy(exhausted)
    passed = BusyAwareSession(FakeSession(exhausted)).post("u", json={}, timeout=1.0)
    assert "Retry-After" not in passed.headers


def test_api_client_retries_the_busy_429_instead_of_dropping_the_endpoint() -> None:
    ok = FakeReply(200, "")
    ok.json = lambda: {"choices": [{"message": {"content": "hi"}}], "budget": {"used": 1, "limit": 9}}  # type: ignore[method-assign]
    replies = [FakeReply(429, BUSY_TEXT), ok]

    class Scripted:
        def get(self, url: str, *, timeout: float) -> FakeReply:
            return ok

        def post(self, url: str, *, json: Any, timeout: float) -> FakeReply:
            return replies.pop(0)

    waits: list[float] = []
    settings = Settings(api_key="k", server_url="http://s", hf_token=None, data_dir=Path("."),
                        wmloc_task_id=None)
    client = ApiClient(settings, session=BusyAwareSession(Scripted()), sleep=waits.append)
    assert client.chat("ref-01", "p", 4).content == "hi"
    assert len(waits) == 1


def test_collector_refuses_scored_endpoints() -> None:
    assert validate_reference_endpoints(["ref-01", "ref-09"]) == ["ref-01", "ref-09"]
    for bad in (["ep-01"], ["ref-1"], ["ref-01", "ep-33"]):
        with pytest.raises(CollectError):
            validate_reference_endpoints(bad)


def test_build_jobs_is_repeat_major_and_skips_stored_keys() -> None:
    prompts = short_wave()[:2]
    stored = {("ref-01", prompts[0].prompt_id, 0)}
    jobs = build_jobs(prompts, ["ref-01", "ref-02"], 2, stored)
    assert len(jobs) == 2 * 2 * 2 - 1
    assert all(isinstance(job, QueryJob) for job in jobs)
    assert [job.repeat for job in jobs] == sorted(job.repeat for job in jobs)
    assert ("ref-01", prompts[0].prompt_id, 0) not in {job.key for job in jobs}
    with pytest.raises(CollectError):
        build_jobs(prompts, ["ref-01"], 0, set())


def test_prompts_are_unique_reproducible_and_fresh() -> None:
    ids = [p.prompt_id for p in all_prompts()]
    assert len(ids) == len(set(ids))
    assert [p.text for p in short_wave()] == [p.text for p in short_wave()]
    assert all(pid.startswith(("d-", "h-")) for pid in ids)
    assert prompt_family("cand-arith-003") == "arith" and prompt_family("d-barith-001") == "d-barith"


def test_deterministic_features() -> None:
    count = "Count from 1 to 5, separated by commas, nothing else."
    assert first_divergence_fraction(count, "1, 2, 3, 4, 5") == 1.0
    assert first_divergence_fraction(count, "1, 2, 4") == pytest.approx(6 / 13)
    assert deterministic_edit_ratio(count, "1, 2, 3, 4, 5") == 0.0
    assert first_divergence_fraction("Say hi", "hi") == 0.0
    arith = "Compute 12 * 3. Answer with only the number."
    assert arithmetic_correct(arith, "12 * 3 = 36.") == 1.0
    assert arithmetic_correct(arith, "37") == 0.0
    assert ends_with_period("p", "Yes.") == 1.0
    table = "Write the multiplication table of 7 from 7 x 1 to 7 x 3, one line each, nothing else."
    assert table_correct_fraction(table, "7 x 1 = 7\n7 x 2 = 14\n7 x 3 = 22") == pytest.approx(2 / 3)


def test_battery_features_use_the_answer_key() -> None:
    by_family = {p.family: p for p in battery_wave()}
    arith = by_family["barith"].text
    items = [line.split(". ", 1)[1] for line in arith.splitlines()[1:]]
    truth = [str(eval(item)) for item in items]
    assert arithmetic_accuracy(arith, "\n".join(truth)) == 1.0
    assert arithmetic_accuracy(arith, "\n".join(f"{i + 1}. {t}" for i, t in enumerate(truth))) == 1.0
    assert arithmetic_accuracy(arith, "") == 0.0
    yesno = by_family["byesno"].text
    assert yes_no_accuracy(yesno, "yes\nno\nyes") == pytest.approx(3 / 14)
    assert lines_fraction(yesno, "yes\nno") == pytest.approx(2 / 14)
    assert facts_accuracy(by_family["bfacts"].text, "Ouagadougou\nW") == pytest.approx(2 / 12)
    assert random_seven_fraction(by_family["brand"].text, "7\n7\n3\n1") == 0.5
    assert lines_fraction("not a battery", "a\nb") == 0.0


def test_roc_auc_handles_separation_ties_and_inversion() -> None:
    pos = np.array([[3.0, 4.0]])
    neg = np.array([[1.0, 2.0]])
    assert roc_auc(pos, neg)[0] == 1.0
    assert roc_auc(neg, pos)[0] == 0.0
    assert roc_auc(np.array([[1.0]]), np.array([[1.0]]))[0] == 0.5


def _tensor(shift_fp8: float, shift_lora: float, cells: int = 200) -> tuple[np.ndarray, Roles, list[str]]:
    rng = np.random.default_rng(0)
    endpoints = ["s1", "s2", "s3", "s4", "fp8", "lora"]
    values = rng.normal(size=(1, len(endpoints), cells))
    values[0, 4] += shift_fp8
    values[0, 5] += shift_lora
    roles = Roles(base=("s1", "s2", "s3", "s4"), fp8=("fp8",), lora_weak=("lora",))
    return values, roles, endpoints


def test_auc_curve_reaches_one_with_a_planted_signal_and_stays_near_chance_without() -> None:
    planted, roles, endpoints = _tensor(2.0, 2.0)
    curve = auc_curve_for_tensor(center_by_cell(planted), endpoints, roles, np.ones(1), (40,), 100)
    assert curve[40]["both"].mean > 0.95
    null, roles, endpoints = _tensor(0.0, 0.0)
    flat = auc_curve_for_tensor(center_by_cell(null), endpoints, roles, np.ones(1), (40,), 200)
    assert 0.25 < flat[40]["both"].mean < 0.75


def test_design_dataclass_is_frozen() -> None:
    design = Design(("a",), (("p", 0),), {"p": "x"}, {"a": ("r",)})
    with pytest.raises(AttributeError):
        design.endpoints = ("b",)  # type: ignore[misc]


def test_store_path_default_is_team_specific(tmp_path: Path) -> None:
    from hackathon.model_equality.research.d_collect import D_STORE_FILENAME

    assert D_STORE_FILENAME == "reference_answers_d.jsonl"
    assert tmp_path.exists()


def test_holdout_wave_is_new_balanced_and_collected_prompt_major() -> None:
    from hackathon.model_equality.research.d_prompts import (
        battery_wave,
        echo_wave,
        facts_wave,
        holdout_wave,
        long_wave,
        replication_wave,
        validation_wave,
    )

    hold = holdout_wave()
    earlier = {p.text for wave in (short_wave, long_wave, battery_wave, validation_wave, facts_wave, echo_wave,
                                   replication_wave) for p in wave()}
    assert len(hold) == 60 and not {p.text for p in hold} & earlier
    for prefix in (25, 50):
        operators = [p.text.split()[2] for p in hold[:prefix]]
        assert max(operators.count(o) for o in "+-*") - min(operators.count(o) for o in "+-*") <= 1
    jobs = build_jobs(hold[:2], ["ref-01", "ref-02"], 2, set(), prompt_major=True)
    assert [(j.prompt_id, j.repeat, j.endpoint) for j in jobs][:4] == [
        ("h-arith-000", 0, "ref-01"), ("h-arith-000", 0, "ref-02"),
        ("h-arith-000", 1, "ref-01"), ("h-arith-000", 1, "ref-02")]
    assert jobs[4].prompt_id == "h-arith-001"


def test_stop_file_makes_the_client_refuse_new_requests_without_sending(tmp_path: Path) -> None:
    from hackathon.model_equality.client import FatalQueryError
    from hackathon.model_equality.research.d_client import StoppableClient

    sent: list[Any] = []

    class Counting:
        def get(self, url: str, *, timeout: float) -> FakeReply:
            return FakeReply(200, "")

        def post(self, url: str, *, json: Any, timeout: float) -> FakeReply:
            sent.append(json)
            reply = FakeReply(200, "")
            reply.json = lambda: {"choices": [{"message": {"content": "hi"}}]}  # type: ignore[method-assign]
            return reply

    stop = tmp_path / "STOP"
    settings = Settings(api_key="k", server_url="http://s", hf_token=None, data_dir=Path("."),
                        wmloc_task_id=None)
    client = StoppableClient(settings, session=Counting(), stop_file=stop)
    assert client.chat("ref-01", "p", 4).content == "hi" and len(sent) == 1
    stop.write_text("x", encoding="utf-8")
    with pytest.raises(FatalQueryError):
        client.chat("ref-01", "p", 4)
    assert len(sent) == 1


def test_sidecar_records_raw_fields_without_touching_the_reply(tmp_path: Path) -> None:
    import json as json_lib

    class OkReply:
        status_code = 200
        text = ""
        headers: ClassVar[dict[str, str]] = {
            "Server": "nginx", "Content-Length": "9", "X-Process-Time": "0.5"}

        def json(self) -> Any:
            return {"id": "chatcmpl-1", "created": 0, "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"}}], "budget": {"used": 3, "limit": 9}}

    class Inner:
        def get(self, url: str, *, timeout: float) -> OkReply:
            return OkReply()

        def post(self, url: str, *, json: Any, timeout: float) -> OkReply:
            return OkReply()

    sidecar = tmp_path / "raw.jsonl"
    reply = BusyAwareSession(Inner(), sidecar).post(
        "u", json={"model": "ref-01", "max_tokens": 8, "messages": [{"role": "user", "content": "p"}]}, timeout=1.0)
    assert reply.status_code == 200
    row = json_lib.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
    assert row["endpoint"] == "ref-01" and row["finish_reason"] == "stop" and row["usage"] is None
    assert row["extra_headers"] == {"X-Process-Time": "0.5"} and row["latency_s"] >= 0.0
    assert row["body_keys"] == ["budget", "choices", "created", "id"]
