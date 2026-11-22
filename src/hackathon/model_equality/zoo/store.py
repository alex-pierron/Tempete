"""Atomic per-endpoint upsert of the zoo answers, in the official `jsonl` line format.

One file per family holds all its endpoints. A variant is written as a whole: the file is read,
the rows of that endpoint are replaced, and the result replaces the file by an atomic rename, so a
crash leaves either the old file or the new one (never a half variant) and a rerun never
duplicates a row. Fields are the official ones: `endpoint`, `id`, `repeat`, `prompt`, `response`,
plus `timestamp` (UTC, ignored by the official reader).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

AnswerRow = dict[str, object]


def utc_now() -> str:
    """ISO-8601 UTC timestamp with a `Z` suffix (storage is always UTC)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_row(endpoint: str, prompt_id: str, repeat: int, prompt: str, response: str) -> AnswerRow:
    """One official-format answer row."""
    return {"endpoint": endpoint, "id": prompt_id, "repeat": repeat, "prompt": prompt,
            "response": response, "timestamp": utc_now()}


def read_rows(path: Path) -> list[AnswerRow]:
    """Every readable row of `path` (a torn last line is skipped; a missing file gives `[]`)."""
    if not path.is_file():
        return []
    rows: list[AnswerRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def commit_endpoint(path: Path, endpoint: str, rows: Iterable[AnswerRow]) -> int:
    """Replace the rows of `endpoint` in `path`; return the number of rows now stored for it.

    Raises:
        ValueError: a new row belongs to another endpoint.
    """
    new_rows = list(rows)
    for row in new_rows:
        if row["endpoint"] != endpoint:
            raise ValueError(f"row of {row['endpoint']!r} committed under {endpoint!r}")
    kept = [row for row in read_rows(path) if row.get("endpoint") != endpoint]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in [*kept, *new_rows]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return len(new_rows)


def row_counts(path: Path) -> Counter[str]:
    """Rows stored per endpoint."""
    return Counter(str(row.get("endpoint")) for row in read_rows(path))
