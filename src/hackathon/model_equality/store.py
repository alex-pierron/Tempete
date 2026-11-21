"""Append-only JSONL store of endpoint answers, byte-compatible with the official template.

The team collects answers with the official `task_template.py` / `query_reference.py`, whose line
format is `{"endpoint": "ep-07", "id": "3", "repeat": 0, "prompt": "...", "response": "..."}` and
whose resume logic de-duplicates on `(endpoint, id, repeat)`. This store therefore:

* writes exactly those five fields (plus an optional `timestamp`, which the official reader
  ignores), with `id` as a string and `repeat` as an int;
* reads any file the official tool wrote (extra fields tolerated, `prompt_id` accepted as an alias
  of `id`, blank lines skipped);
* never appends a key (endpoint, id, repeat) that is already stored, because every duplicate would
  burn irreplaceable server budget for nothing.

A crash can leave a truncated last line. It is skipped on read (and counted) and the next append
first terminates it with a newline, so one torn write never corrupts the lines that follow.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hackathon.common.errors import HackathonError

ANSWERS_FILE_ENV = "MET_ANSWERS_FILE"
"""Overrides the path of the `ep-*` answers file (team convention: model_equality/answers.jsonl)."""
REFERENCE_ANSWERS_FILE_ENV = "MET_REFERENCE_ANSWERS_FILE"
"""Overrides the path of the `ref-*` answers file (team convention: reference_answers.jsonl)."""
STORE_DIR_RELATIVE = Path("artifacts") / "model_equality"
EP_ANSWERS_FILENAME = "answers.jsonl"
REF_ANSWERS_FILENAME = "reference_answers.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[3]

KEY_ENDPOINT = "endpoint"
KEY_ID = "id"
KEY_ID_ALIAS = "prompt_id"
KEY_REPEAT = "repeat"
KEY_PROMPT = "prompt"
KEY_RESPONSE = "response"
KEY_TIMESTAMP = "timestamp"

AnswerKey = tuple[str, str, int]
"""`(endpoint, prompt id, repeat)`: the identity of one answer."""


class StoreError(HackathonError):
    """The answers file is unreadable or a line breaks the official format."""


@dataclass(frozen=True)
class Answer:
    """One stored answer.

    Attributes:
        endpoint: Endpoint id, e.g. "ep-07" or "ref-03".
        prompt_id: Prompt identifier (a string, as in the official files).
        repeat: Index of the repeated sample for this (endpoint, prompt).
        prompt: Prompt text as sent.
        response: Generated text.
        timestamp: UTC ISO-8601 time of the request, or None for files written by other tools.
    """

    endpoint: str
    prompt_id: str
    repeat: int
    prompt: str
    response: str
    timestamp: str | None = None

    @property
    def key(self) -> AnswerKey:
        """The de-duplication key shared with the official tool."""
        return (self.endpoint, self.prompt_id, self.repeat)

    def to_json_line(self) -> str:
        """Serialise as one official-format JSON line (no trailing newline)."""
        payload: dict[str, Any] = {
            KEY_ENDPOINT: self.endpoint,
            KEY_ID: self.prompt_id,
            KEY_REPEAT: self.repeat,
            KEY_PROMPT: self.prompt,
            KEY_RESPONSE: self.response,
        }
        if self.timestamp is not None:
            payload[KEY_TIMESTAMP] = self.timestamp
        return json.dumps(payload, ensure_ascii=False)


def utc_timestamp() -> str:
    """Current UTC time in ISO-8601, the only timezone this store ever writes."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_answer(line: str, line_number: int, source: Path) -> Answer:
    """Parse one JSON line of an official-format answers file.

    Raises:
        StoreError: Invalid JSON, wrong types or a missing mandatory field.
    """
    where = f"{source}:{line_number}"
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise StoreError(f"{where}: invalid JSON ({exc.msg})") from exc
    if not isinstance(raw, dict):
        raise StoreError(f"{where}: expected a JSON object")
    prompt_id = raw.get(KEY_ID, raw.get(KEY_ID_ALIAS))
    missing = [
        name for name, value in
        ((KEY_ENDPOINT, raw.get(KEY_ENDPOINT)), (KEY_ID, prompt_id),
         (KEY_REPEAT, raw.get(KEY_REPEAT)), (KEY_RESPONSE, raw.get(KEY_RESPONSE)))
        if value is None
    ]
    if missing:
        raise StoreError(f"{where}: missing field(s) {missing}")
    repeat = raw[KEY_REPEAT]
    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise StoreError(f"{where}: `repeat` must be an integer, got {repeat!r}")
    timestamp = raw.get(KEY_TIMESTAMP)
    return Answer(
        endpoint=str(raw[KEY_ENDPOINT]),
        prompt_id=str(prompt_id),
        repeat=repeat,
        prompt=str(raw.get(KEY_PROMPT, "")),
        response=str(raw[KEY_RESPONSE]),
        timestamp=str(timestamp) if timestamp is not None else None,
    )


def default_answers_path(*, reference: bool, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve the store path: environment variable first, then `artifacts/model_equality/`."""
    env_name = REFERENCE_ANSWERS_FILE_ENV if reference else ANSWERS_FILE_ENV
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return Path(from_env).expanduser()
    filename = REF_ANSWERS_FILENAME if reference else EP_ANSWERS_FILENAME
    return repo_root / STORE_DIR_RELATIVE / filename


class AnswerStore:
    """Append-only, resumable, thread-safe JSONL store of answers.

    Args:
        path: The JSONL file (created on first append, with its parent directory).
    """

    def __init__(self, path: Path) -> None:
        """Bind the store to `path`; nothing is read until `load`."""
        self.path = path
        self._lock = threading.Lock()
        self.skipped_torn_lines = 0
        self.duplicate_lines = 0

    def load(self) -> dict[AnswerKey, Answer]:
        """Read every stored answer, first occurrence of a key winning.

        Raises:
            StoreError: A complete (newline-terminated) line is malformed.
        """
        self.skipped_torn_lines = 0
        self.duplicate_lines = 0
        answers: dict[AnswerKey, Answer] = {}
        if not self.path.is_file():
            return answers
        text = self.path.read_text(encoding="utf-8")
        lines = text.split("\n")
        ends_cleanly = text.endswith("\n") or not text
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            is_last_unterminated = index == len(lines) and not ends_cleanly
            try:
                answer = parse_answer(line, index, self.path)
            except StoreError:
                if is_last_unterminated:
                    self.skipped_torn_lines += 1
                    continue
                raise
            if answer.key in answers:
                self.duplicate_lines += 1
                continue
            answers[answer.key] = answer
        return answers

    def keys(self) -> set[AnswerKey]:
        """The keys already stored (what the official tool skips on resume)."""
        return set(self.load())

    def count_by_endpoint(self) -> Counter[str]:
        """Number of stored answers per endpoint."""
        return Counter(key[0] for key in self.load())

    def _repair_unterminated_tail(self) -> None:
        """Make the file end on a newline before appending.

        A last line without a newline is either a complete record whose newline was lost
        (terminate it) or a torn write (cut it off: it was never a usable answer, and leaving it
        would turn into a malformed line in the middle of the file).
        """
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        data = self.path.read_bytes()
        if data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        tail = data[cut:].decode("utf-8", errors="replace")
        try:
            parse_answer(tail, 0, self.path)
        except StoreError:
            self.path.write_bytes(data[:cut])
        else:
            self.path.write_bytes(data + b"\n")

    def append(self, answer: Answer, known_keys: set[AnswerKey] | None = None) -> bool:
        """Append one answer unless its key is already stored.

        Args:
            answer: The answer to persist.
            known_keys: Optional in-memory key set, updated in place, to avoid re-reading the
                file for every append during a long run.

        Returns:
            True when a line was written, False when the key already existed.
        """
        with self._lock:
            keys = known_keys if known_keys is not None else self.keys()
            if answer.key in keys:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._repair_unterminated_tail()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(answer.to_json_line() + "\n")
                handle.flush()
            keys.add(answer.key)
            return True


def load_many(paths: Iterable[Path]) -> list[Answer]:
    """Concatenate several stores, de-duplicating on the answer key (first file wins)."""
    seen: set[AnswerKey] = set()
    merged: list[Answer] = []
    for path in paths:
        for key, answer in AnswerStore(path).load().items():
            if key not in seen:
                seen.add(key)
                merged.append(answer)
    return merged
