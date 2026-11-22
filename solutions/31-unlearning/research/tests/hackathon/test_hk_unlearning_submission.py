"""Tests of save/validate/submit flow, fetch idempotence and the CLI wiring (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from test_hk_unlearning_support import tiny_factory, tiny_model

from hackathon.common.config import Settings
from hackathon.common.errors import SubmissionInvalidError
from hackathon.common.submit import SubmitResult
from hackathon.unlearning import cli, fetch, submission
from hackathon.unlearning.constants import FORGET_IMAGES_FILENAME, MODEL_FILENAME
from hackathon.unlearning.fetch import FetchError, fetch_task_files, sha256_of
from hackathon.unlearning.submission import save_submission, submit_validated


def _settings(tmp_path: Path) -> Settings:
    return Settings(api_key="key", server_url="http://server", hf_token=None, data_dir=tmp_path,
                    wmloc_task_id=None)  # fmt: skip


# ----- save_submission -------------------------------------------------------------------------


def test_save_submission_writes_a_validated_float32_file(tmp_path: Path) -> None:
    model = tiny_model().half()
    path = tmp_path / "out" / "sub.pt"
    report = save_submission(model, path, tiny_factory)
    assert path.is_file() and report.path == path
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert all(v.dtype == torch.float32 for v in saved.values() if v.is_floating_point())
    assert not list(path.parent.glob("*.partial*"))


def test_invalid_model_leaves_no_file_behind(tmp_path: Path) -> None:
    model = tiny_model()
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    path = tmp_path / "sub.pt"
    with pytest.raises(SubmissionInvalidError, match="non-finite"):
        save_submission(model, path, tiny_factory)
    assert not path.exists() and not list(tmp_path.glob("*.partial*"))


def test_save_replaces_an_existing_file_atomically(tmp_path: Path) -> None:
    path = tmp_path / "sub.pt"
    path.write_bytes(b"old")
    save_submission(tiny_model(), path, tiny_factory)
    assert path.stat().st_size > 1000


# ----- submit ------------------------------------------------------------------------------------


def test_dry_run_validates_but_never_calls_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sub.pt"
    save_submission(tiny_model(), path, tiny_factory)
    monkeypatch.setattr(submission, "validate_submission", lambda p: None)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not send without --yes")

    monkeypatch.setattr(submission, "submit_file", boom)
    assert submit_validated(path, _settings(tmp_path), confirm=False) is None


def test_confirmed_submit_calls_the_shared_client_with_the_task_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sub.pt"
    path.write_bytes(b"x")
    calls: list[tuple[str, Path, bool]] = []
    result = SubmitResult("31-unlearning", str(path), "s", "t", True, 200, 0.5, None, None, "")

    def fake(task_id: str, file: Path, settings: Settings, *, confirm: bool) -> SubmitResult:
        calls.append((task_id, file, confirm))
        return result

    monkeypatch.setattr(submission, "validate_submission", lambda p: None)
    monkeypatch.setattr(submission, "submit_file", fake)
    assert submit_validated(path, _settings(tmp_path), confirm=True) is result
    assert calls == [("31-unlearning", path, True)]


def test_invalid_file_is_never_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "sub.pt"
    torch.save({"a": torch.zeros(1)}, path)
    monkeypatch.setattr(submission, "submit_file", lambda *a, **k: pytest.fail("sent"))
    with pytest.raises(SubmissionInvalidError):
        submit_validated(path, _settings(tmp_path), confirm=True)


def test_cli_submit_without_yes_is_a_dry_run_and_invalid_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(submission, "validate_submission", lambda p: None)
    monkeypatch.setattr(submission, "submit_file", lambda *a, **k: pytest.fail("sent"))
    path = tmp_path / "sub.pt"
    path.write_bytes(b"x")
    assert cli.main(["submit", str(path)]) == 0
    assert "dry run" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(tmp_path))
    assert cli.main(["submit", str(path), "--yes"]) == 1
    assert "error:" in capsys.readouterr().err


# ----- fetch -------------------------------------------------------------------------------------


def test_sha256_of_known_content(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _fake_official(monkeypatch: pytest.MonkeyPatch, content: bytes) -> None:
    monkeypatch.setattr(fetch, "MODEL_SIZE_BYTES", len(content))
    import hashlib

    monkeypatch.setattr(fetch, "MODEL_SHA256", hashlib.sha256(content).hexdigest())


def test_fetch_is_idempotent_when_files_are_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"model")
    (tmp_path / FORGET_IMAGES_FILENAME).write_bytes(b"images")
    _fake_official(monkeypatch, b"model")
    monkeypatch.setattr(fetch, "_download", lambda *a, **k: pytest.fail("downloaded again"))
    results = fetch_task_files(tmp_path)
    assert [r.downloaded for r in results] == [False, False]
    assert results[0].size_bytes == 5


def test_fetch_refuses_to_overwrite_a_foreign_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"something else")
    _fake_official(monkeypatch, b"model")
    with pytest.raises(FetchError, match="not the official model"):
        fetch_task_files(tmp_path)


def test_fetch_downloads_missing_files_and_verifies_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_official(monkeypatch, b"model")
    fetched: list[str] = []

    def fake_download(filename: str, destination: Path, token: str | None) -> Path:
        fetched.append(filename)
        (destination / filename).write_bytes(b"model" if filename == MODEL_FILENAME else b"i")
        return destination / filename

    monkeypatch.setattr(fetch, "_download", fake_download)
    results = fetch_task_files(tmp_path / "new")
    assert fetched == [MODEL_FILENAME, FORGET_IMAGES_FILENAME]
    assert [r.downloaded for r in results] == [True, True]


def test_fetch_rejects_a_corrupted_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_official(monkeypatch, b"model")

    def corrupt(filename: str, destination: Path, token: str | None) -> Path:
        (destination / filename).write_bytes(b"MODEL")
        return destination / filename

    monkeypatch.setattr(fetch, "_download", corrupt)
    with pytest.raises(FetchError, match="official check"):
        fetch_task_files(tmp_path)


# ----- CLI parsing -------------------------------------------------------------------------------


def test_parser_knows_every_command_and_run_accepts_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["run", "--method", "scrub", "--config", "smoke",
                              "--set", "lr=1e-5", "--set", "epochs=2"])  # fmt: skip
    assert args.method == "scrub" and args.set == ["lr=1e-5", "epochs=2"]
    for command in ("fetch", "info", "proxy-eval", "validate x.pt", "submit x.pt --yes"):
        assert parser.parse_args(command.split()).command == command.split()[0]
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--method", "gradient_ascent"])


def test_info_reports_missing_files_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(tmp_path))
    assert cli.main(["--task-dir", str(tmp_path / "empty"), "--device", "cpu", "info"]) == 0
    out = capsys.readouterr().out
    assert "missing (run fetch)" in out


def test_run_rejects_a_malformed_set_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: _settings(tmp_path))
    code = cli.main(["--task-dir", str(tmp_path), "run", "--set", "lr"])
    assert code == 1 and "key=value" in capsys.readouterr().err
