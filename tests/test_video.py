"""Tests of wmkit.video: PyAV round trips, validation and the H.264 attack."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from wmkit.video import (
    VideoFormatError,
    VideoReadError,
    h264_attack,
    read_video,
    write_video,
)

FRAME_COUNT = 12
FRAME_HEIGHT = 64
FRAME_WIDTH = 96
FPS = 24.0
SEED = 0
LOSSLESS_CRF = 0
HIGH_QUALITY_CRF = 18
LOW_QUALITY_CRF = 45
MIN_PSNR_HIGH_QUALITY_DB = 30.0
PSNR_PEAK = 1.0
EIGHT_BIT_STEP = 1.0 / 255.0


def _moving_gradient(
    frame_count: int = FRAME_COUNT, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH
) -> torch.Tensor:
    """Smooth, slowly moving content: realistic for a codec, unlike white noise."""
    ys = torch.linspace(0, 1, height).view(1, height, 1)
    xs = torch.linspace(0, 1, width).view(1, 1, width)
    shifts = torch.linspace(0, 0.3, frame_count).view(frame_count, 1, 1)
    red = 0.5 + 0.4 * torch.sin(2 * math.pi * (xs + shifts))
    green = 0.5 + 0.4 * torch.cos(2 * math.pi * (ys + shifts))
    blue = 0.5 + 0.4 * torch.sin(math.pi * (xs + ys + shifts))
    shape = (frame_count, height, width)
    planes = [plane.expand(shape) for plane in (red, green, blue)]
    return torch.stack(planes, dim=1).clamp(0, 1).to(torch.float32)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * torch.log10(torch.tensor(PSNR_PEAK**2 / mse)))


def test_fixture_shape() -> None:
    assert _moving_gradient().shape == (FRAME_COUNT, 3, FRAME_HEIGHT, FRAME_WIDTH)


def test_round_trip_keeps_shape_fps_and_quality(tmp_path: Path) -> None:
    frames = _moving_gradient()
    target = tmp_path / "clip.mp4"
    write_video(frames, target, fps=FPS, crf=HIGH_QUALITY_CRF)
    decoded, fps = read_video(target)
    assert decoded.shape == frames.shape
    assert decoded.dtype == torch.float32
    assert fps == pytest.approx(FPS)
    assert float(decoded.min()) >= 0.0 and float(decoded.max()) <= 1.0
    assert _psnr(frames, decoded) > MIN_PSNR_HIGH_QUALITY_DB


def test_non_integer_fps_is_preserved(tmp_path: Path) -> None:
    target = tmp_path / "ntsc.mp4"
    write_video(_moving_gradient(), target, fps=30000 / 1001)
    _, fps = read_video(target)
    assert fps == pytest.approx(30000 / 1001, rel=1e-4)


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "clip.mp4"
    write_video(_moving_gradient(frame_count=2), target, fps=FPS)
    assert target.is_file()


def test_lower_quality_means_larger_error_and_smaller_file(tmp_path: Path) -> None:
    frames = _moving_gradient()
    good, bad = tmp_path / "good.mp4", tmp_path / "bad.mp4"
    write_video(frames, good, fps=FPS, crf=HIGH_QUALITY_CRF)
    write_video(frames, bad, fps=FPS, crf=LOW_QUALITY_CRF)
    assert bad.stat().st_size < good.stat().st_size
    assert _psnr(frames, read_video(bad)[0]) < _psnr(frames, read_video(good)[0])


def test_h264_attack_matches_shape_and_degrades_with_crf() -> None:
    frames = _moving_gradient()
    mild = h264_attack(frames, HIGH_QUALITY_CRF)
    harsh = h264_attack(frames, LOW_QUALITY_CRF)
    assert mild.shape == frames.shape and harsh.shape == frames.shape
    assert float(mild.min()) >= 0.0 and float(harsh.max()) <= 1.0
    assert _psnr(frames, harsh) < _psnr(frames, mild)


def test_h264_attack_supports_odd_sizes() -> None:
    frames = _moving_gradient(frame_count=4, height=63, width=95)
    attacked = h264_attack(frames, HIGH_QUALITY_CRF)
    assert attacked.shape == frames.shape


def test_h264_attack_is_deterministic() -> None:
    frames = _moving_gradient(frame_count=4)
    first = h264_attack(frames, HIGH_QUALITY_CRF)
    second = h264_attack(frames, HIGH_QUALITY_CRF)
    assert torch.equal(first, second)


def test_write_rejects_odd_sizes(tmp_path: Path) -> None:
    frames = _moving_gradient(frame_count=2, height=63, width=95)
    with pytest.raises(VideoFormatError, match="even"):
        write_video(frames, tmp_path / "odd.mp4", fps=FPS)


@pytest.mark.parametrize("crf", [-1, 52])
def test_crf_out_of_range_is_rejected(crf: int, tmp_path: Path) -> None:
    with pytest.raises(VideoFormatError, match="crf"):
        write_video(_moving_gradient(frame_count=2), tmp_path / "x.mp4", fps=FPS, crf=crf)
    with pytest.raises(VideoFormatError, match="crf"):
        h264_attack(_moving_gradient(frame_count=2), crf)


def test_bad_shape_and_fps_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(VideoFormatError, match="shape"):
        write_video(torch.rand(3, FRAME_HEIGHT, FRAME_WIDTH), tmp_path / "x.mp4", fps=FPS)
    with pytest.raises(VideoFormatError, match="fps"):
        write_video(_moving_gradient(frame_count=2), tmp_path / "x.mp4", fps=0.0)


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoReadError, match="not found"):
        read_video(tmp_path / "absent.mp4")


def test_read_garbage_file_raises(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoReadError):
        read_video(garbage)


def test_lossless_round_trip_is_close_to_eight_bit_quantisation(tmp_path: Path) -> None:
    # yuv420 chroma subsampling prevents exact equality; luma-dominated error stays small.
    frames = _moving_gradient(frame_count=3)
    target = tmp_path / "lossless.mp4"
    write_video(frames, target, fps=FPS, crf=LOSSLESS_CRF)
    decoded, _ = read_video(target)
    assert float((decoded - frames).abs().mean()) < 4 * EIGHT_BIT_STEP
