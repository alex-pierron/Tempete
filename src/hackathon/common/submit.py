"""Submission client shared by the three tasks, with a local log and a cooldown guard.

Protocol (copied from the official submission.py of every task):
    POST {server_url}/submit/{task_id}, header X-API-Key, multipart field "file".
    200 + {"status": "success", "score": ..., "previous_best"?, "improved"?, "message"?} on success;
    any other status carries {"detail": ...}. The server may take up to ~500 s to score.

Every attempt (success or not) is appended to `<data_dir>/submissions/log.jsonl`, because a failed
submission also starts the server cooldown (30 min for model equality). Task modules validate their
file first, then call `submit_file`. Nothing is sent unless the caller passes `confirm=True`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from hackathon.common.config import Settings, load_settings
from hackathon.common.errors import CooldownError, HackathonError, SubmitError

CONNECT_TIMEOUT_S = 120.0
"""requests/urllib3 apply this value to every socket write of the upload, not only to the TCP
connect: 10 s (official script) aborted a 344 MB state dict on a 4G uplink (21/11/2026)."""
SCORING_TIMEOUT_S = 600.0
"""The server waits up to ~500 s for the score (official submission.py uses (10, 600))."""
COOLDOWN_BY_TASK_S: dict[str, float] = {
    "33-model-equality": 30 * 60,
    "34-watermark-localisation-image": 10 * 60 + 90,
    "31-unlearning": 10 * 60 + 90,
}
"""Server cooldowns. Model equality: documented. The other two are NOT in the PDFs but were
observed on 21/11/2026 ("Wait 593 seconds" right after an accepted localization submission):
a rejected attempt costs nothing but time, so the local guard waits instead of trying. The +90 s
margin: the server counts from the END of the previous scoring, our log from its start ("Wait 33
seconds" refusal on 21/11/2026 21:25)."""
LOG_RELATIVE_PATH = Path("submissions") / "log.jsonl"
HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of one submission attempt, as logged."""

    task_id: str
    file: str
    sha256: str
    submitted_at: str
    ok: bool
    http_status: int | None
    score: float | None
    previous_best: float | None
    improved: bool | None
    message: str


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed (state dicts are ~345 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def log_path(settings: Settings) -> Path:
    """Where the submission log lives."""
    return settings.data_dir / LOG_RELATIVE_PATH


def read_log(settings: Settings) -> list[SubmitResult]:
    """All logged attempts, oldest first; a missing log is empty."""
    path = log_path(settings)
    if not path.is_file():
        return []
    return [SubmitResult(**json.loads(line)) for line in path.read_text("utf-8").splitlines() if line]


def append_log(settings: Settings, result: SubmitResult) -> None:
    """Append one attempt to the log (creates the directory)."""
    path = log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result)) + "\n")


def ensure_cooldown(settings: Settings, task_id: str, now: datetime | None = None) -> None:
    """Raise CooldownError if the last logged attempt on `task_id` is inside its cooldown."""
    cooldown = COOLDOWN_BY_TASK_S.get(task_id)
    if cooldown is None:
        return
    attempts = [r for r in read_log(settings) if r.task_id == task_id]
    if not attempts:
        return
    last = datetime.fromisoformat(attempts[-1].submitted_at)
    wait = last + timedelta(seconds=cooldown) - (now or datetime.now(UTC))
    if wait.total_seconds() > 0:
        raise CooldownError(f"{task_id}: last submission at {last.isoformat()}, "
                            f"wait {int(wait.total_seconds() // 60) + 1} more minute(s).")


def _parse_reply(reply: requests.Response) -> tuple[bool, dict[str, Any]]:
    try:
        body = reply.json()
    except ValueError:
        return False, {"detail": f"HTTP {reply.status_code}: {reply.text[:300]}"}
    if not isinstance(body, dict):
        return False, {"detail": f"unexpected reply: {str(body)[:300]}"}
    return reply.status_code == 200 and body.get("status") == "success", body


def submit_file(task_id: str, path: Path, settings: Settings, *, confirm: bool,
                now: datetime | None = None) -> SubmitResult:
    """Send `path` to the server for `task_id` and log the attempt.

    Args:
        task_id: Server task id, e.g. "31-unlearning".
        path: The already validated submission file.
        settings: Resolved settings (API key required).
        confirm: Must be True; a False value raises, so dry runs never reach the network.
        now: Clock override for tests.

    Returns:
        The logged SubmitResult (ok=False when the server rejected the file).

    Raises:
        HackathonError: confirm is False, or the file is missing.
        CooldownError: the task cooldown is not over.
        SubmitError: the server is unreachable.
    """
    if not confirm:
        raise HackathonError("Refusing to submit without explicit confirmation (--yes).")
    if not path.is_file():
        raise HackathonError(f"Submission file not found: {path}")
    api_key = settings.require_api_key()
    ensure_cooldown(settings, task_id, now)
    sha = file_sha256(path)
    started = (now or datetime.now(UTC)).isoformat()
    try:
        with path.open("rb") as handle:
            reply = requests.post(f"{settings.server_url}/submit/{task_id}",
                                  headers={"X-API-Key": api_key},
                                  files={"file": (path.name, handle)},
                                  timeout=(CONNECT_TIMEOUT_S, SCORING_TIMEOUT_S))
    except requests.RequestException as exc:
        raise SubmitError(f"Could not reach {settings.server_url}: {exc}") from exc
    ok, body = _parse_reply(reply)
    score = body.get("score")
    previous = body.get("previous_best")
    result = SubmitResult(
        task_id=task_id, file=str(path), sha256=sha, submitted_at=started, ok=ok,
        http_status=reply.status_code,
        score=float(score) if isinstance(score, (int, float)) else None,
        previous_best=float(previous) if isinstance(previous, (int, float)) else None,
        improved=body.get("improved") if isinstance(body.get("improved"), bool) else None,
        message=str(body.get("message") or body.get("detail") or ""),
    )
    append_log(settings, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m hackathon.common.submit --task ID --file PATH [--yes].

    Without --yes it only prints what would be sent. Task-specific validation lives in each task
    package; prefer the task CLIs, which validate before calling this.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--yes", action="store_true", help="really send the file")
    args = parser.parse_args(argv)
    settings = load_settings()
    if not args.yes:
        print(f"[dry run] would send {args.file} to {settings.server_url}/submit/{args.task}")
        return 0
    try:
        result = submit_file(args.task, args.file, settings, confirm=True)
    except HackathonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    started_at = time.monotonic()
    code = main()
    print(f"({time.monotonic() - started_at:.1f} s)", file=sys.stderr)
    sys.exit(code)
