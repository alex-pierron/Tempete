"""Tests of the `python -m hackathon.wmloc` CLI: validate, submit guard, predict, bench, fetch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("PIL")

from PIL import Image

from hackathon.common.config import Settings
from hackathon.common.submit import SubmitResult
from hackathon.wmloc import cli, kit_backend
from hackathon.wmloc.constants import (
    GRID_SIZE,
    IMAGE_SIZE,
    KIT_TASK_ID,
    METHOD_PIXELSEAL,
    METHOD_TRUSTMARK,
)
from hackathon.wmloc.errors import FetchError
from hackathon.wmloc.fetch import FetchResult
from hackathon.wmloc.local_backend import DecodeBackend
from hackathon.wmloc.manifest import ManifestSample, write_manifest
from hackathon.wmloc.submission import save_submission

N_BITS = 256
NAMES = ["a.png", "b.png"]
TASK_ID = "99-localization"


def make_settings(tmp_path: Path, task_id: str | None = TASK_ID) -> Settings:
    return Settings(
        api_key="key",
        server_url="http://s",
        hf_token=None,
        data_dir=tmp_path,
        wmloc_task_id=task_id,
    )


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings that never read the real .env, and a submit_file that must not be reached."""
    settings = make_settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    def forbidden(*args: Any, **kwargs: Any) -> SubmitResult:
        raise AssertionError("submit_file must not be called")

    monkeypatch.setattr(cli, "submit_file", forbidden)
    return settings


def write_good_file(tmp_path: Path) -> Path:
    target = tmp_path / "sub.pt"
    save_submission(target, NAMES, torch.zeros(2, GRID_SIZE, GRID_SIZE, dtype=torch.uint8))
    return target


def test_validate_ok_and_error_exit_codes(
    isolated: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["validate", str(write_good_file(tmp_path))]) == 0
    assert "VALID: 2 images" in capsys.readouterr().out
    assert cli.main(["validate", str(tmp_path / "missing.pt")]) == 1
    assert "error:" in capsys.readouterr().err


def test_validate_against_a_manifest_requires_the_exact_names(
    isolated: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    write_manifest(
        tmp_path / "m.json", [ManifestSample("other.png", "a.png", METHOD_PIXELSEAL, "01")]
    )
    code = cli.main(
        ["validate", str(write_good_file(tmp_path)), "--manifest", str(tmp_path / "m.json")]
    )
    assert code == 1
    assert "expected set" in capsys.readouterr().err


def test_submit_without_yes_is_a_dry_run(
    isolated: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["submit", str(write_good_file(tmp_path))]) == 0
    assert "dry run" in capsys.readouterr().out


def test_submit_with_yes_but_invalid_file_sends_nothing(isolated: Settings, tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"image_names": ["a"], "masks": torch.zeros(1, 4, 4)}, bad)
    assert cli.main(["submit", str(bad), "--yes"]) == 1


def test_submit_task_id_falls_back_to_the_official_template_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: make_settings(tmp_path, task_id=None))
    seen: list[str] = []

    def fake_submit(task_id: str, path: Path, given: Settings, *, confirm: bool) -> SubmitResult:
        seen.append(task_id)
        return SubmitResult(task_id, str(path), "sha", "now", True, 200, 0.5, None, None, "ok")

    monkeypatch.setattr(cli, "submit_file", fake_submit)
    assert cli.main(["submit", str(write_good_file(tmp_path)), "--yes"]) == 0
    assert seen == [KIT_TASK_ID]


def test_submit_with_yes_calls_the_client_with_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    seen: dict[str, Any] = {}

    def fake_submit(task_id: str, path: Path, given: Settings, *, confirm: bool) -> SubmitResult:
        seen.update(task_id=task_id, path=path, confirm=confirm)
        return SubmitResult(task_id, str(path), "sha", "now", True, 200, 0.5, None, None, "ok")

    monkeypatch.setattr(cli, "submit_file", fake_submit)
    target = write_good_file(tmp_path)
    assert cli.main(["submit", str(target), "--yes"]) == 0
    assert seen == {"task_id": TASK_ID, "path": target, "confirm": True}


def test_fetch_reports_errors_and_results(
    isolated: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> FetchResult:
        raise FetchError("access denied")

    monkeypatch.setattr(cli, "fetch_dataset", refuse)
    assert cli.main(["fetch"]) == 1
    assert "access denied" in capsys.readouterr().err

    def succeed(*args: Any, **kwargs: Any) -> FetchResult:
        return FetchResult("repo/x", Path("/dest"), ("train.zip",), (), ())

    monkeypatch.setattr(cli, "fetch_dataset", succeed)
    assert cli.main(["fetch"]) == 0
    assert "repo/x" in capsys.readouterr().out


def write_images_manifest(root: Path, methods: list[str]) -> Path:
    records: list[ManifestSample] = []
    generator = np.random.default_rng(0)
    for index, method in enumerate(methods):
        name = f"img{index}.png"
        pixels = generator.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / name)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[:, : 64 * (index + 1)] = 255
        Image.fromarray(mask).save(root / f"mask{index}.png")
        records.append(
            ManifestSample(
                name,
                name,
                method,
                "01" * (N_BITS // 2),
                mask512=f"mask{index}.png",
                edit="addition",
            )
        )
    manifest = root / "manifest.json"
    write_manifest(manifest, records)
    return manifest


def fake_backend() -> DecodeBackend:
    generator = torch.Generator().manual_seed(0)

    def decode(images: torch.Tensor) -> torch.Tensor:
        return torch.randn(images.shape[0], N_BITS, generator=generator)

    return DecodeBackend(decode, N_BITS, 256)


def fake_factory(method: str, device: str) -> DecodeBackend | None:
    return fake_backend() if method == METHOD_PIXELSEAL else None


def test_predict_writes_a_valid_file_with_fallback_for_unwired_methods(
    isolated: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kit_backend, "make_backend", fake_factory)
    manifest = write_images_manifest(tmp_path, [METHOD_PIXELSEAL, METHOD_TRUSTMARK])
    out = tmp_path / "out.pt"
    code = cli.main(
        ["predict", "--manifest", str(manifest), "--localizer", "sw_max_96_128", "--out", str(out)]
    )
    assert code == 0
    loaded = torch.load(out, weights_only=True)
    assert loaded["image_names"] == ["img0.png", "img1.png"]
    assert loaded["masks"].shape == (2, GRID_SIZE, GRID_SIZE)
    assert bool((loaded["masks"][1] == 1).all())  # TrustMark -> all-ones fallback
    assert "1 by the all-ones fallback" in capsys.readouterr().out
    assert cli.main(["validate", str(out), "--manifest", str(manifest)]) == 0


def test_predict_refuses_a_map_localizer_on_a_decode_only_backend(
    isolated: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kit_backend, "make_backend", fake_factory)
    manifest = write_images_manifest(tmp_path, [METHOD_PIXELSEAL])
    code = cli.main(
        [
            "predict",
            "--manifest",
            str(manifest),
            "--localizer",
            "pixel_map_s4",
            "--out",
            str(tmp_path / "o.pt"),
        ]
    )
    assert code == 1
    assert "decode-only" in capsys.readouterr().err


def test_a_bad_backend_factory_spec_is_reported(
    isolated: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_images_manifest(tmp_path, [METHOD_PIXELSEAL])
    code = cli.main(
        [
            "predict",
            "--manifest",
            str(manifest),
            "--backend-factory",
            "no.such.module:make",
            "--out",
            str(tmp_path / "o.pt"),
        ]
    )
    assert code == 1
    assert "cannot load backend factory" in capsys.readouterr().err


def test_predict_rejects_an_unknown_localizer(
    isolated: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kit_backend, "make_backend", fake_factory)
    manifest = write_images_manifest(tmp_path, [METHOD_PIXELSEAL])
    assert (
        cli.main(
            [
                "predict",
                "--manifest",
                str(manifest),
                "--localizer",
                "nope",
                "--out",
                str(tmp_path / "o.pt"),
            ]
        )
        == 1
    )
    assert "unknown localizer" in capsys.readouterr().err


def test_bench_reports_every_localizer_and_group(
    isolated: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kit_backend, "make_backend", fake_factory)
    manifest = write_images_manifest(tmp_path, [METHOD_PIXELSEAL] * 4)
    report_path = tmp_path / "report.json"
    code = cli.main(
        [
            "bench",
            "--manifest",
            str(manifest),
            "--localizers",
            "all_ones,all_zeros,sw_max_96_128",
            "--json",
            str(report_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    for expected in ("== all_ones", "== all_zeros", "== sw_max_96_128", "addition", "all"):
        assert expected in out
    assert report_path.is_file()
