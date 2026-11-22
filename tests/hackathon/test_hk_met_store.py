"""Answer store: byte-compatibility with the official task_template.py files, resume, dedup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.model_equality.store import (
    ANSWERS_FILE_ENV,
    REFERENCE_ANSWERS_FILE_ENV,
    Answer,
    AnswerStore,
    StoreError,
    default_answers_path,
    load_many,
)

# A line exactly as the official task_template.py query() writes it (json.dumps, ensure_ascii=False).
OFFICIAL_LINE = (
    '{"endpoint": "ep-07", "id": "3", "repeat": 0, "prompt": "Say \\"caf\u00e9\\"", '
    '"response": "caf\u00e9\\nline two"}'
)


def official_done_set(path: Path) -> set[tuple[str, str, int]]:
    """The resume logic of the official query(): verbatim semantics."""
    with path.open(encoding="utf-8") as handle:
        return {(r["endpoint"], r["id"], r["repeat"])
                for r in (json.loads(line) for line in handle if line.strip())}


def test_reads_an_official_line_exactly(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(OFFICIAL_LINE + "\n", encoding="utf-8")
    answers = AnswerStore(path).load()
    assert list(answers) == [("ep-07", "3", 0)]
    answer = answers[("ep-07", "3", 0)]
    assert answer.prompt == 'Say "caf\u00e9"'
    assert answer.response == "caf\u00e9\nline two"
    assert answer.timestamp is None


def test_written_lines_have_the_official_fields_and_official_resume_reads_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answers.jsonl"
    store = AnswerStore(path)
    store.append(Answer("ep-01", "arith-000", 1, "p", "r", "2026-11-21T10:00:00+00:00"))
    store.append(Answer("ep-01", "arith-000", 0, "p", "r2"))
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert set(first) == {"endpoint", "id", "repeat", "prompt", "response", "timestamp"}
    assert isinstance(first["id"], str) and isinstance(first["repeat"], int)
    assert set(json.loads(lines[1])) == {"endpoint", "id", "repeat", "prompt", "response"}
    assert official_done_set(path) == {("ep-01", "arith-000", 1), ("ep-01", "arith-000", 0)}


def test_prompt_id_alias_and_extra_fields_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(
        '{"endpoint": "ep-02", "prompt_id": 5, "repeat": 2, "prompt": "p", "response": "r", '
        '"extra": [1, 2]}\n\n',
        encoding="utf-8",
    )
    assert list(AnswerStore(path).load()) == [("ep-02", "5", 2)]


def test_append_never_duplicates_a_key(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "answers.jsonl")
    assert store.append(Answer("ep-01", "a", 0, "p", "first"))
    assert not store.append(Answer("ep-01", "a", 0, "p", "second"))
    assert len(store.load()) == 1
    assert store.load()[("ep-01", "a", 0)].response == "first"


def test_duplicate_lines_from_a_merge_keep_the_first_and_are_counted(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(OFFICIAL_LINE + "\n" + OFFICIAL_LINE.replace("line two", "other") + "\n",
                    encoding="utf-8")
    store = AnswerStore(path)
    assert len(store.load()) == 1
    assert store.duplicate_lines == 1
    assert "line two" in store.load()[("ep-07", "3", 0)].response


def test_torn_last_line_is_skipped_and_next_append_stays_valid(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(OFFICIAL_LINE + '\n{"endpoint": "ep-01", "id": "1", "rep', encoding="utf-8")
    store = AnswerStore(path)
    assert len(store.load()) == 1
    assert store.skipped_torn_lines == 1
    store.append(Answer("ep-01", "x", 0, "p", "r"))
    reread = AnswerStore(path)
    assert set(reread.load()) == {("ep-07", "3", 0), ("ep-01", "x", 0)}


def test_malformed_complete_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text("not json\n" + OFFICIAL_LINE + "\n", encoding="utf-8")
    with pytest.raises(StoreError, match=":1:"):
        AnswerStore(path).load()


def test_missing_field_and_bad_repeat_raise(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text('{"endpoint": "ep-01", "id": "1", "repeat": "0", "response": "r"}\n',
                    encoding="utf-8")
    with pytest.raises(StoreError, match="integer"):
        AnswerStore(path).load()
    path.write_text('{"endpoint": "ep-01", "repeat": 0, "response": "r"}\n', encoding="utf-8")
    with pytest.raises(StoreError, match="missing"):
        AnswerStore(path).load()


def test_path_resolution_env_beats_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ANSWERS_FILE_ENV, raising=False)
    monkeypatch.delenv(REFERENCE_ANSWERS_FILE_ENV, raising=False)
    default_ep = default_answers_path(reference=False, repo_root=tmp_path)
    default_ref = default_answers_path(reference=True, repo_root=tmp_path)
    assert default_ep == tmp_path / "artifacts" / "model_equality" / "answers.jsonl"
    assert default_ref == tmp_path / "artifacts" / "model_equality" / "reference_answers.jsonl"
    monkeypatch.setenv(ANSWERS_FILE_ENV, str(tmp_path / "team.jsonl"))
    monkeypatch.setenv(REFERENCE_ANSWERS_FILE_ENV, str(tmp_path / "ref.jsonl"))
    assert default_answers_path(reference=False) == tmp_path / "team.jsonl"
    assert default_answers_path(reference=True) == tmp_path / "ref.jsonl"


def test_load_many_prefers_the_first_file(tmp_path: Path) -> None:
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    AnswerStore(one).append(Answer("ep-01", "a", 0, "p", "from-one"))
    AnswerStore(two).append(Answer("ep-01", "a", 0, "p", "from-two"))
    AnswerStore(two).append(Answer("ep-02", "a", 0, "p", "only-two"))
    merged = load_many([one, two])
    assert {(a.endpoint, a.response) for a in merged} == {("ep-01", "from-one"),
                                                          ("ep-02", "only-two")}
