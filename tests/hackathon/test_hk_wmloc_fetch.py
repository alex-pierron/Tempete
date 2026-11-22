"""Tests of `fetch`: token guard, repo fallback, error mapping, idempotent safe unpacking.

`huggingface_hub.snapshot_download` is replaced by a fake: no test touches the network.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from hackathon.common.config import Settings
from hackathon.wmloc.constants import DEFAULT_REPO_ID, FALLBACK_REPO_IDS, KIT_ARCHIVES
from hackathon.wmloc.errors import FetchError
from hackathon.wmloc.fetch import fetch_dataset, unpack_archives

SECRET_TOKEN = "hf_super_secret_value"


def make_settings(tmp_path: Path, token: str | None = SECRET_TOKEN) -> Settings:
    return Settings(
        api_key=None, server_url="http://s", hf_token=token, data_dir=tmp_path, wmloc_task_id=None
    )


def hub_error(kind: type[Exception], status: int | None = None) -> Exception:
    """Build a Hub exception without depending on the HTTP library of the installed version."""
    error = kind.__new__(kind)
    if status is not None:
        error.response = SimpleNamespace(status_code=status)  # type: ignore[attr-defined]
    return error


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)


def install_fake_download(
    monkeypatch: pytest.MonkeyPatch,
    behaviour: Callable[[str, Path], None],
    calls: list[str],
    tokens_expected: str | None = SECRET_TOKEN,
) -> None:
    def fake(*, repo_id: str, repo_type: str, local_dir: str, token: str, **_: Any) -> str:
        assert repo_type == "dataset"
        assert token == tokens_expected
        calls.append(repo_id)
        behaviour(repo_id, Path(local_dir))
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)


def populate_all(_: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    write_zip(destination / "starter_kit.zip", {"wmloc/kit.py": b"x = 1\n"})
    write_zip(destination / "train.zip", {"train/a.png": b"AAA"})
    write_zip(destination / "test.zip", {"test/b.png": b"BB"})


def test_missing_token_still_downloads_now_that_the_dataset_is_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    install_fake_download(monkeypatch, populate_all, calls, tokens_expected=None)
    result = fetch_dataset(make_settings(tmp_path, token=None))
    assert calls == [DEFAULT_REPO_ID] and set(result.extracted) == set(KIT_ARCHIVES)


def test_download_then_unpack_all_three_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    install_fake_download(monkeypatch, populate_all, calls)
    result = fetch_dataset(make_settings(tmp_path))
    destination = tmp_path / "wmloc"
    assert result.repo_id == DEFAULT_REPO_ID and calls == [DEFAULT_REPO_ID]
    assert result.destination == destination
    assert set(result.extracted) == set(KIT_ARCHIVES) and result.missing == ()
    assert (destination / "starter_kit" / "wmloc" / "kit.py").read_bytes() == b"x = 1\n"
    assert (destination / "release" / "train" / "a.png").read_bytes() == b"AAA"
    assert (destination / "release" / "test" / "b.png").read_bytes() == b"BB"


def test_second_run_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_download(monkeypatch, populate_all, [])
    fetch_dataset(make_settings(tmp_path))
    marker_times = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "wmloc").glob(".*.extracted")}
    again = fetch_dataset(make_settings(tmp_path))
    assert again.extracted == () and set(again.already_extracted) == set(KIT_ARCHIVES)
    assert {
        p.name: p.stat().st_mtime_ns for p in (tmp_path / "wmloc").glob(".*.extracted")
    } == marker_times


def test_404_falls_back_to_the_paris_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def behaviour(repo_id: str, destination: Path) -> None:
        if repo_id == DEFAULT_REPO_ID:
            raise hub_error(RepositoryNotFoundError)
        populate_all(repo_id, destination)

    install_fake_download(monkeypatch, behaviour, calls)
    result = fetch_dataset(make_settings(tmp_path))
    assert calls == [DEFAULT_REPO_ID, FALLBACK_REPO_IDS[0]]
    assert result.repo_id == FALLBACK_REPO_IDS[0]


def test_all_repos_missing_explains_the_gated_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def behaviour(repo_id: str, destination: Path) -> None:
        raise hub_error(RepositoryNotFoundError)

    install_fake_download(monkeypatch, behaviour, [])
    with pytest.raises(FetchError, match="404"):
        fetch_dataset(make_settings(tmp_path))


def test_gated_403_stops_without_fallback_and_never_leaks_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def behaviour(repo_id: str, destination: Path) -> None:
        raise hub_error(GatedRepoError)

    install_fake_download(monkeypatch, behaviour, calls)
    with pytest.raises(FetchError, match="403") as caught:
        fetch_dataset(make_settings(tmp_path))
    assert calls == [DEFAULT_REPO_ID]
    assert SECRET_TOKEN not in str(caught.value)


def test_http_401_and_500_are_mapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for status, expected in ((401, "not authorized"), (500, "HTTP 500")):

        def behaviour(repo_id: str, destination: Path, status: int = status) -> None:
            raise hub_error(HfHubHTTPError, status)

        install_fake_download(monkeypatch, behaviour, [])
        with pytest.raises(FetchError, match=expected):
            fetch_dataset(make_settings(tmp_path))


def test_unexpected_failure_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def behaviour(repo_id: str, destination: Path) -> None:
        raise TimeoutError("slow")

    install_fake_download(monkeypatch, behaviour, [])
    with pytest.raises(FetchError, match="TimeoutError"):
        fetch_dataset(make_settings(tmp_path))


def test_no_archive_at_all_gives_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def behaviour(repo_id: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "README.txt").write_text("hello")

    install_fake_download(monkeypatch, behaviour, [])
    with pytest.raises(FetchError, match="none of starter_kit.zip") as caught:
        fetch_dataset(make_settings(tmp_path))
    assert "README.txt" in str(caught.value)


def test_a_missing_archive_is_reported_but_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def behaviour(repo_id: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        write_zip(destination / "train.zip", {"train/a.png": b"A"})

    install_fake_download(monkeypatch, behaviour, [])
    result = fetch_dataset(make_settings(tmp_path))
    assert result.extracted == ("train.zip",)
    assert set(result.missing) == {"starter_kit.zip", "test.zip"}


def test_zip_slip_member_is_refused(tmp_path: Path) -> None:
    write_zip(tmp_path / "starter_kit.zip", {"../escape.txt": b"pwn"})
    with pytest.raises(FetchError, match="unsafe member"):
        unpack_archives(tmp_path)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_conflicting_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "release" / "train").mkdir(parents=True)
    (tmp_path / "release" / "train" / "a.png").write_bytes(b"local edit")
    write_zip(tmp_path / "train.zip", {"train/a.png": b"AAA"})
    with pytest.raises(FetchError, match="already exists"):
        unpack_archives(tmp_path)
    assert (tmp_path / "release" / "train" / "a.png").read_bytes() == b"local edit"


def test_corrupt_zip_is_reported(tmp_path: Path) -> None:
    (tmp_path / "test.zip").write_bytes(b"PK-not-really")
    with pytest.raises(FetchError, match="not a valid zip"):
        unpack_archives(tmp_path)


def test_interrupted_unpack_resumes(tmp_path: Path) -> None:
    write_zip(tmp_path / "train.zip", {"train/a.png": b"AAA", "train/b.png": b"BBB"})
    (tmp_path / "release" / "train").mkdir(parents=True)
    (tmp_path / "release" / "train" / "a.png").write_bytes(b"AAA")  # left by an interrupted run
    extracted, _, _ = unpack_archives(tmp_path)
    assert extracted == ("train.zip",)
    assert (tmp_path / "release" / "train" / "b.png").read_bytes() == b"BBB"
