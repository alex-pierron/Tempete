"""Tests for wmkit.cli and `python -m wmkit`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from wmkit.cli import main
from wmkit.eval import RESULT_COLUMNS
from wmkit.io import load_image

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
SUBPROCESS_TIMEOUT_SECONDS = 120


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every CLI test on CPU: deterministic and independent of the host GPU."""
    monkeypatch.setenv("WMKIT_DEVICE", "cpu")


def run_cli(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    status = main(list(arguments))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_list_shows_methods_attack_groups_and_suites(capsys: pytest.CaptureFixture[str]) -> None:
    status, out, _ = run_cli(capsys, "list")
    assert status == 0
    for expected in ("dct_qim", "spread_spectrum", "dwt_dct_svd", "pixelseal", "valuemetric:",
                     "removal:", "regen_vae", "quick:", "paper:", "jpeg_50"):
        assert expected in out


def test_embed_then_extract_round_trip_through_a_png_file(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    source = image_dir / "a_astronaut.png"
    marked = tmp_path / "marked.png"
    status, out, _ = run_cli(capsys, "embed", "--method", "dct_qim", "--in", str(source),
                             "--out", str(marked), "--seed", "3")
    assert status == 0
    message_hex = out.strip()
    assert len(message_hex) == 8  # 32 bits
    assert load_image(marked).shape == load_image(source).shape

    status, out, _ = run_cli(capsys, "extract", "--method", "dct_qim", "--in", str(marked),
                             "--message", message_hex)
    assert status == 0
    assert f"bits: {message_hex}" in out
    assert "bit_acc: 1.0000" in out
    assert "detected: True" in out
    assert "neg_log10_p:" in out

    status, out, _ = run_cli(capsys, "extract", "--method", "dct_qim", "--in", str(marked))
    assert status == 0
    assert out.strip() == f"bits: {message_hex}"


def test_extract_on_an_unmarked_image_is_not_detected(
    capsys: pytest.CaptureFixture[str], image_dir: Path
) -> None:
    status, out, _ = run_cli(capsys, "extract", "--method", "spread_spectrum", "--in",
                             str(image_dir / "b_coffee.png"), "--message", "deadbeef")
    assert status == 0
    assert "detected: False" in out


def test_embed_honours_an_explicit_message_and_the_seed(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    source = str(image_dir / "a_astronaut.png")
    status, out, _ = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source,
                             "--out", str(tmp_path / "one.png"), "--message", "0x0badf00d")
    assert status == 0 and out.strip() == "0badf00d"
    first = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source, "--out",
                    str(tmp_path / "a.png"), "--seed", "9")[1]
    second = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source, "--out",
                     str(tmp_path / "b.png"), "--seed", "9")[1]
    third = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source, "--out",
                    str(tmp_path / "c.png"), "--seed", "10")[1]
    assert first == second != third


def test_embed_message_random_keyword_matches_the_omitted_default(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    source = str(image_dir / "a_astronaut.png")
    omitted = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source,
                      "--out", str(tmp_path / "omitted.png"), "--seed", "3")[1]
    keyword = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source, "--out",
                      str(tmp_path / "keyword.png"), "--message", "random", "--seed", "3")[1]
    other_seed = run_cli(capsys, "embed", "--method", "dwt_dct_svd", "--in", source, "--out",
                         str(tmp_path / "other.png"), "--message", "RANDOM", "--seed", "4")[1]
    assert keyword.strip() == omitted.strip()  # 'random' == omitting --message, same seed
    assert keyword.strip() != other_seed.strip()  # seed still drives the payload; case-insensitive


def test_attack_writes_the_attacked_image_deterministically(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    source = image_dir / "b_coffee.png"  # 96 x 128
    status, out, _ = run_cli(capsys, "attack", "--attack", "crop_0.5", "--in", str(source),
                             "--out", str(tmp_path / "c1.png"), "--seed", "4")
    assert status == 0
    assert "128x96 -> 64x48" in out
    run_cli(capsys, "attack", "--attack", "crop_0.5", "--in", str(source), "--out",
            str(tmp_path / "c2.png"), "--seed", "4")
    run_cli(capsys, "attack", "--attack", "crop_0.5", "--in", str(source), "--out",
            str(tmp_path / "c3.png"), "--seed", "5")
    first, second, third = (load_image(tmp_path / f"c{i}.png") for i in (1, 2, 3))
    assert first.shape == (3, 48, 64)
    assert (first == second).all()
    assert not (first == third).all()


def test_eval_writes_the_csv_and_prints_a_markdown_table(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results" / "classical_quick.csv"
    status, out, err = run_cli(
        capsys, "eval", "--method", "dct_qim,spread_spectrum,dwt_dct_svd", "--images",
        str(image_dir), "--suite", "quick", "--max-images", "2", "--out", str(output),
    )
    assert status == 0, err
    frame = pd.read_csv(output)
    assert list(frame.columns) == RESULT_COLUMNS
    assert set(frame["method"]) == {"dct_qim", "spread_spectrum", "dwt_dct_svd"}
    assert frame["image_id"].nunique() == 2
    assert len(frame) == 3 * 2 * 11 * 2
    assert "| method" in out and "|---" in out.replace(" ", "")
    assert "valuemetric" in out and "combined" in out


def test_eval_seed_makes_the_csv_reproducible(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    paths = [tmp_path / "one.csv", tmp_path / "two.csv"]
    for path in paths:
        status = run_cli(capsys, "eval", "--method", "dct_qim", "--images", str(image_dir),
                         "--max-images", "2", "--seed", "7", "--out", str(path))[0]
        assert status == 0
    assert paths[0].read_text() == paths[1].read_text()


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        (["embed", "--method", "nope", "--in", "x.png", "--out", "y.png"], "unknown watermarking"),
        (["attack", "--attack", "jpeg_5O", "--in", "x.png", "--out", "y.png"], "jpeg_50"),
        (["extract", "--method", "dct_qim", "--in", "missing.png"], "not found"),
        (["eval", "--method", "", "--images", ".", "--out", "o.csv"], "at least one method"),
    ],
)
def test_runtime_errors_are_one_line_messages_with_status_one(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    arguments: list[str], fragment: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    status, _, err = run_cli(capsys, *arguments)
    assert status == 1
    assert err.startswith("error:") and fragment in err
    assert "Traceback" not in err


def test_bad_hex_and_empty_folder_are_reported(
    capsys: pytest.CaptureFixture[str], image_dir: Path, tmp_path: Path
) -> None:
    status, _, err = run_cli(capsys, "extract", "--method", "dct_qim", "--in",
                             str(image_dir / "a_astronaut.png"), "--message", "xyz")
    assert status == 1 and "hex" in err
    empty = tmp_path / "empty"
    empty.mkdir()
    status, _, err = run_cli(capsys, "eval", "--method", "dct_qim", "--images", str(empty),
                             "--out", str(tmp_path / "o.csv"))
    assert status == 1 and "no image" in err


def test_usage_errors_exit_with_status_two() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["eval", "--method", "dct_qim"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_python_dash_m_entry_point() -> None:
    environment = {**os.environ, "PYTHONPATH": str(SOURCE_DIRECTORY), "WMKIT_DEVICE": "cpu"}
    completed = subprocess.run(
        [sys.executable, "-m", "wmkit", "list"], capture_output=True, text=True,
        env=environment, timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "dct_qim" in completed.stdout
