"""Tests of hackathon.common: .env parsing, settings precedence, cooldown, submission protocol."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import requests

from hackathon.common import submit as submit_module
from hackathon.common.config import DEFAULT_SERVER_URL, Settings, load_settings, read_env_file
from hackathon.common.errors import ConfigError, CooldownError, HackathonError
from hackathon.common.submit import ensure_cooldown, read_log, submit_file

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


def make_settings(tmp_path: Path, api_key: str | None = "k") -> Settings:
    return Settings(api_key=api_key, server_url="http://server", hf_token=None,
                    data_dir=tmp_path, wmloc_task_id=None)


def test_env_file_parsing_skips_comments_and_strips_quotes(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# c\nHACKATHON_API_KEY=\"abc\"\nEMPTY=\nNOEQUALS\nHF_TOKEN='t'\n", "utf-8")
    assert read_env_file(env) == {"HACKATHON_API_KEY": "abc", "EMPTY": "", "HF_TOKEN": "t"}


def test_environment_wins_over_file_and_defaults_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("HACKATHON_API_KEY=file\nHACKATHON_DATA_DIR=/x\n", "utf-8")
    monkeypatch.setenv("HACKATHON_API_KEY", "process")
    monkeypatch.delenv("HACKATHON_SERVER_URL", raising=False)
    monkeypatch.delenv("HACKATHON_DATA_DIR", raising=False)
    settings = load_settings(env)
    assert settings.api_key == "process"
    assert settings.server_url == DEFAULT_SERVER_URL
    assert settings.data_dir == Path("/x")


def test_missing_api_key_and_task_id_raise_config_errors(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, api_key="YOUR_API_KEY_HERE")
    with pytest.raises(ConfigError):
        settings.require_api_key()
    with pytest.raises(ConfigError):
        settings.require_wmloc_task_id()


class FakeReply:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self) -> Any:
        return self._body


def test_submit_requires_confirmation_and_never_posts_without_it(tmp_path: Path) -> None:
    file = tmp_path / "s.pt"
    file.write_bytes(b"x")
    with pytest.raises(HackathonError):
        submit_file("31-unlearning", file, make_settings(tmp_path), confirm=False)
    assert read_log(make_settings(tmp_path)) == []


def test_submit_posts_the_official_protocol_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "s.csv"
    file.write_text("endpoint,family,model\n", "utf-8")
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeReply:
        seen["url"], seen["headers"], seen["field"] = url, kwargs["headers"], kwargs["files"]["file"][0]
        return FakeReply(200, {"status": "success", "score": 0.42, "previous_best": 0.3,
                               "improved": True, "message": "ok"})

    monkeypatch.setattr(submit_module.requests, "post", fake_post)
    settings = make_settings(tmp_path)
    result = submit_file("33-model-equality", file, settings, confirm=True, now=NOW)
    assert seen == {"url": "http://server/submit/33-model-equality",
                    "headers": {"X-API-Key": "k"}, "field": "s.csv"}
    assert result.ok and result.score == pytest.approx(0.42) and result.improved is True
    assert read_log(settings) == [result]


def test_rejected_submission_is_logged_and_starts_the_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "s.csv"
    file.write_text("x", "utf-8")
    monkeypatch.setattr(submit_module.requests, "post",
                        lambda url, **kw: FakeReply(400, {"detail": "bad format"}))
    settings = make_settings(tmp_path)
    result = submit_file("33-model-equality", file, settings, confirm=True, now=NOW)
    assert not result.ok and result.message == "bad format"
    with pytest.raises(CooldownError):
        ensure_cooldown(settings, "33-model-equality", NOW + timedelta(minutes=29))
    ensure_cooldown(settings, "33-model-equality", NOW + timedelta(minutes=31))
    ensure_cooldown(settings, "31-unlearning", NOW)  # no documented cooldown


def test_unreachable_server_raises_submit_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "s.pt"
    file.write_bytes(b"x")

    def boom(url: str, **kwargs: Any) -> FakeReply:
        raise requests.ConnectionError("down")

    monkeypatch.setattr(submit_module.requests, "post", boom)
    with pytest.raises(submit_module.SubmitError):
        submit_file("31-unlearning", file, make_settings(tmp_path), confirm=True, now=NOW)
