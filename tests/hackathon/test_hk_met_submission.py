"""Submission file: validation rules, writing, and the confirmation gate of `submit`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from hackathon.common.config import Settings
from hackathon.common.errors import SubmissionInvalidError
from hackathon.common.submit import SubmitResult
from hackathon.model_equality import submission as submission_module
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.submission import (
    describe_clusters,
    rows_from_labeling,
    submit,
    validate_submission,
    write_submission,
)


def valid_rows() -> list[tuple[str, str, str]]:
    return [(e, "A" if i < 20 else "B", f"A{i % 3}" if i < 20 else "B1")
            for i, e in enumerate(SCORED_ENDPOINTS)]


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_written_file_validates_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    write_submission(valid_rows(), path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "endpoint,family,model"
    assert validate_submission(path) == valid_rows()
    assert "33 endpoints" in describe_clusters(valid_rows())


def test_header_is_optional_and_order_is_irrelevant(tmp_path: Path) -> None:
    lines = [",".join(row) for row in reversed(valid_rows())]
    assert len(validate_submission(write_lines(tmp_path / "s.csv", lines))) == 33


def test_all_problems_are_reported_at_once(tmp_path: Path) -> None:
    rows = [",".join(r) for r in valid_rows()]
    rows[0] = "ep-01,A"                       # 2 columns
    rows[1] = "ep-02,A,A1,extra"              # 4 columns (a comma in a label)
    rows[2] = "ep-08,A,A1"                    # duplicate ep-08
    rows[3] = "ref-01,A,A1"                   # not accepted by the server
    rows[4] = "ep-05,A,"                      # empty model
    rows[5] = "ep-06,B,A0"                    # model A0 also used in family A
    with pytest.raises(SubmissionInvalidError) as info:
        validate_submission(write_lines(tmp_path / "s.csv", rows))
    message = str(info.value)
    for fragment in ("expected 3 columns, got 2", "expected 3 columns, got 4",
                     "ep-08 appears 2 times", "unexpected endpoint 'ref-01'",
                     "empty model", "missing endpoint ep-01", "missing endpoint ep-04",
                     "model label 'A0' is used in several families"):
        assert fragment in message, fragment


def test_quoted_comma_in_a_label_is_rejected(tmp_path: Path) -> None:
    rows = [",".join(r) for r in valid_rows()]
    rows[0] = 'ep-01,"A,x",A1'
    with pytest.raises(SubmissionInvalidError, match="comma"):
        validate_submission(write_lines(tmp_path / "s.csv", rows))


def test_missing_file_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(SubmissionInvalidError, match="not found"):
        validate_submission(tmp_path / "nope.csv")


def test_rows_from_labeling_is_sorted() -> None:
    assert rows_from_labeling({"ep-02": ("B", "B1"), "ep-01": ("A", "A1")}) == [
        ("ep-01", "A", "A1"), ("ep-02", "B", "B1")]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(api_key="k", server_url="http://server", hf_token=None, data_dir=tmp_path,
                    wmloc_task_id=None)


def test_submit_without_confirm_validates_and_never_touches_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network used in a dry run")

    monkeypatch.setattr(requests, "post", forbidden)
    path = tmp_path / "s.csv"
    write_submission(valid_rows(), path)
    assert submit(path, make_settings(tmp_path), confirm=False) is None


def test_submit_with_an_invalid_file_never_reaches_submit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("submit_file called with an invalid file")

    monkeypatch.setattr(submission_module, "submit_file", forbidden)
    path = write_lines(tmp_path / "bad.csv", ["ep-01,A,A1"])
    with pytest.raises(SubmissionInvalidError):
        submit(path, make_settings(tmp_path), confirm=True)


def test_submit_with_confirm_calls_the_shared_client_with_the_task_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_submit_file(task_id: str, path: Path, settings: Settings, *, confirm: bool) -> SubmitResult:
        seen.update(task_id=task_id, path=path, confirm=confirm)
        return SubmitResult(task_id, str(path), "sha", "t", True, 200, 0.5, None, None, "ok")

    monkeypatch.setattr(submission_module, "submit_file", fake_submit_file)
    path = tmp_path / "s.csv"
    write_submission(valid_rows(), path)
    result = submit(path, make_settings(tmp_path), confirm=True)
    assert result is not None and result.score == 0.5
    assert seen == {"task_id": "33-model-equality", "path": path, "confirm": True}
