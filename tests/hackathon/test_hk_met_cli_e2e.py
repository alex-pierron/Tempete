"""End to end through the CLI against a local fake server (port 0, stopped at teardown).

The fake server speaks the documented API (`/models`, `/chat/completions`, `/submit/<task>`) with
synthetic endpoints from `hackathon.model_equality.simulate`. No real server is ever contacted.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hackathon.common.config import Settings
from hackathon.model_equality.cli import EXIT_FAILURE, EXIT_OK, EXIT_REFUSED, main
from hackathon.model_equality.prompts import build_bank
from hackathon.model_equality.simulate import (
    SimEndpoint,
    generate_text,
    reference_endpoints,
    scored_endpoints,
)
from hackathon.model_equality.store import AnswerStore

API_KEY = "test-key"
LIMIT = 100
BUSY_EVERY = 9
CATEGORIES_EP = ["arith", "format", "count"]
PERMUTATIONS = "20"


@dataclass
class ServerState:
    endpoints: dict[str, SimEndpoint]
    used: dict[str, int] = field(default_factory=dict)
    chat_calls: int = 0
    busy_replies: int = 0
    submits: int = 0
    bad_keys: int = 0
    ref_limit: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def make_handler(state: ServerState) -> type[BaseHTTPRequestHandler]:
    prompts_by_text = {p.text: p for p in build_bank()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence the default stderr log
            return

        def _send(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _authorised(self) -> bool:
            if self.headers.get("X-API-Key") == API_KEY:
                return True
            with state.lock:
                state.bad_keys += 1
            self._send(401, {"detail": "bad key"})
            return False

        def do_GET(self) -> None:
            if not self._authorised():
                return
            with state.lock:
                data = [{"id": name,
                         "budget": {"limit": state.ref_limit if name.startswith("ref-") else LIMIT,
                                    "used": state.used.get(name, 0)}}
                        for name in state.endpoints]
            self._send(200, {"data": data})

        def do_POST(self) -> None:
            if not self._authorised():
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if self.path.startswith("/submit/"):
                with state.lock:
                    state.submits += 1
                self._send(200, {"status": "success", "score": 0.42, "message": "scored"})
                return
            body = json.loads(raw)
            endpoint = state.endpoints[body["model"]]
            prompt = prompts_by_text[body["messages"][0]["content"]]
            with state.lock:
                state.chat_calls += 1
                if state.chat_calls % BUSY_EVERY == 0:
                    state.busy_replies += 1
                    self._send(429, {"detail": "busy"}, {"Retry-After": "0"})
                    return
                is_ref = endpoint.name.startswith("ref-")
                cap = state.ref_limit if is_ref else LIMIT
                if cap is not None and state.used.get(endpoint.name, 0) >= cap:
                    self._send(429, {"detail": "budget used up"})
                    return
                state.used[endpoint.name] = state.used.get(endpoint.name, 0) + 1
                used = state.used[endpoint.name]
                seed = state.chat_calls
            text = generate_text(endpoint, prompt.category, prompt.id, np.random.default_rng(seed))
            self._send(200, {"choices": [{"message": {"content": text}}],
                             "budget": {"used": used, "limit": LIMIT}})

    return Handler


@pytest.fixture()
def server() -> Iterator[tuple[str, ServerState]]:
    endpoints = {e.name: e for e in reference_endpoints() + scored_endpoints()}
    state = ServerState(endpoints)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def run(settings: Settings, *argv: str) -> tuple[int, str]:
    lines: list[str] = []
    code = main(list(argv), settings=settings, out=lines.append)
    return code, "\n".join(lines)


def test_full_workflow_through_the_cli(
    server: tuple[str, ServerState], tmp_path: Path
) -> None:
    url, state = server
    settings = Settings(api_key=API_KEY, server_url=url, hf_token=None, data_dir=tmp_path / "data",
                        wmloc_task_id=None)
    ep_store, ref_store = tmp_path / "answers.jsonl", tmp_path / "reference_answers.jsonl"
    out_dir = tmp_path / "analysis"
    practice = tmp_path / "practice_answers.csv"
    practice.write_text("endpoint,family,model,kind\n" + "\n".join(
        f"{e.name},{e.family},{e.model},{e.kind}" for e in reference_endpoints()) + "\n",
        encoding="utf-8")

    code, text = run(settings, "budget", "--answers", str(ep_store))
    assert code == EXIT_OK and "ep-33" in text and "ref-09" in text

    code, text = run(settings, "plan", "--phase", "ep-1", "--categories", *CATEGORIES_EP,
                     "--answers", str(ep_store))
    assert code == EXIT_OK and "TOTAL requests to send: 528 over 33 endpoint(s)" in text
    assert state.chat_calls == 0  # planning is offline

    # reference endpoints first: unlimited, used for calibration
    code, text = run(settings, "query-ref", "--phase", "ref-1", "--answers", str(ref_store))
    assert code == EXIT_OK and "stored=360" in text
    assert len(AnswerStore(ref_store).load()) == 360

    # ep-* without --yes is refused and spends nothing
    calls_before = state.chat_calls
    code, text = run(settings, "query-ep", "--phase", "ep-1", "--categories", *CATEGORIES_EP,
                     "--answers", str(ep_store))
    assert code == EXIT_REFUSED and "REFUSED" in text
    assert state.chat_calls == calls_before and not ep_store.exists()

    # a reserve equal to the whole budget keeps everything unspent
    code, text = run(settings, "query-ep", "--phase", "ep-1", "--categories", *CATEGORIES_EP,
                     "--answers", str(ep_store), "--reserve", "100", "--yes")
    assert code == EXIT_OK and "stored=0" in text and not state.used.get("ep-01")

    code, text = run(settings, "query-ep", "--phase", "ep-1", "--categories", *CATEGORIES_EP,
                     "--answers", str(ep_store), "--yes")
    assert code == EXIT_OK and "stored=528" in text and "failed=0" in text
    assert state.busy_replies > 0  # 429 + Retry-After was absorbed without losing an answer
    assert all(state.used[f"ep-{i:02d}"] == 16 for i in range(1, 34))

    # resuming costs nothing: every key is already stored
    calls_before = state.chat_calls
    code, text = run(settings, "query-ep", "--phase", "ep-1", "--categories", *CATEGORIES_EP,
                     "--answers", str(ep_store), "--yes")
    assert code == EXIT_OK and "jobs=0" in text and state.chat_calls == calls_before

    code, text = run(settings, "budget", "--answers", str(ep_store))
    assert "pull the team store" not in text  # server used == stored for every endpoint

    code, text = run(settings, "analyze", "--calibrate", "--ref-like", "ep-1",
                     "--answers", str(ep_store), "--ref-answers", str(ref_store),
                     "--practice-answers", str(practice), "--out-dir", str(out_dir),
                     "--permutations", PERMUTATIONS)
    assert code == EXIT_OK and "rows in: 528  used: 528  rejected: 0" in text
    for name in ("report_ep.txt", "distance_ep.csv", "distance_family_ep.csv", "labels_ep.csv",
                 "exact_match_ep.csv", "calibration.json"):
        assert (out_dir / name).is_file(), name

    submission = tmp_path / "submission.csv"
    code, text = run(settings, "make-submission", "--labels", str(out_dir / "labels_ep.csv"),
                     "--out", str(submission))
    assert code == EXIT_OK and "33 endpoints" in text
    recut = tmp_path / "recut.csv"
    code, text = run(settings, "make-submission", "--distances", str(out_dir / "distance_ep.csv"),
                     "--tau-model", "1.5", "--tau-family", "9", "--out", str(recut))
    assert code == EXIT_OK and "tau_model=1.50" in text

    code, text = run(settings, "validate", str(submission))
    assert code == EXIT_OK and "valid" in text

    code, text = run(settings, "submit", str(submission))
    assert code == EXIT_REFUSED and "DRY RUN" in text and state.submits == 0
    code, text = run(settings, "submit", str(submission), "--yes")
    assert code == EXIT_OK and "score=0.42" in text and state.submits == 1
    code, _ = run(settings, "submit", str(submission), "--yes")
    assert code == EXIT_FAILURE and state.submits == 1  # 30 minute cooldown holds locally
    log = (tmp_path / "data" / "submissions" / "log.jsonl").read_text(encoding="utf-8")
    assert len(log.splitlines()) == 1

    perfect = tmp_path / "ref_perfect.csv"
    perfect.write_text("endpoint,family,model\n" + "\n".join(
        f"{e.name},{e.family},{e.model}" for e in reference_endpoints()) + "\n", encoding="utf-8")
    code, text = run(settings, "score-practice", str(perfect), "--answers", str(practice))
    assert code == EXIT_OK and "Practice score (MCC over 36 pairs, same model or not): 1.000" in text


def test_a_wrong_api_key_stops_the_run_without_storing(
    server: tuple[str, ServerState], tmp_path: Path
) -> None:
    url, state = server
    settings = Settings(api_key="wrong", server_url=url, hf_token=None, data_dir=tmp_path,
                        wmloc_task_id=None)
    code, _ = run(settings, "query-ref", "--phase", "ref-1", "--answers", str(tmp_path / "r.jsonl"))
    assert code == EXIT_FAILURE and state.chat_calls == 0 and not (tmp_path / "r.jsonl").exists()


def test_phase_and_command_must_agree(server: tuple[str, ServerState], tmp_path: Path) -> None:
    url, _ = server
    settings = Settings(api_key=API_KEY, server_url=url, hf_token=None, data_dir=tmp_path,
                        wmloc_task_id=None)
    code, _ = run(settings, "query-ep", "--phase", "ref-1", "--yes",
                  "--answers", str(tmp_path / "a.jsonl"))
    assert code == EXIT_FAILURE
    code, _ = run(settings, "query-ref", "--phase", "ep-1", "--answers", str(tmp_path / "a.jsonl"))
    assert code == EXIT_FAILURE


def test_query_ref_stays_inside_a_ref_limit_reported_by_the_server(
    server: tuple[str, ServerState], tmp_path: Path
) -> None:
    """Regression: the real server reports limit 100 for ref-* although the docs say unlimited."""
    url, state = server
    state.ref_limit = 30
    settings = Settings(api_key=API_KEY, server_url=url, hf_token=None, data_dir=tmp_path,
                        wmloc_task_id=None)
    code, text = run(settings, "query-ref", "--phase", "ref-1",
                     "--answers", str(tmp_path / "ref.jsonl"))
    assert code == EXIT_OK, text
    assert state.used and max(state.used.values()) <= state.ref_limit
    assert "budget used up" not in text
