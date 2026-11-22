"""Video input/output and codec attacks through PyAV (no system ffmpeg required).

Frames are `float32 [T, 3, H, W]` RGB in `[0, 1]`, the video counterpart of the image
convention of `wmkit.types`. Everything is held in memory: this module targets short clips
(competition samples, tests), not long videos.

Note: `av` and `cv2` bundle two copies of libavdevice; macOS prints a duplicate-class warning
when both are imported in one process. No crash has been observed, but it is a known upstream
issue, hence `av` is imported lazily here.
"""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

# --- Constants ---------------------------------------------------------------------------------

PIXEL_FORMAT_RGB = "rgb24"
PIXEL_FORMAT_YUV420 = "yuv420p"
"""Chroma-subsampled format required by most H.264 decoders; needs even width and height."""

DEFAULT_CODEC = "h264"
DEFAULT_CRF = 23
MIN_CRF = 0
MAX_CRF = 51
DEFAULT_PRESET = "medium"
MAX_FPS_DENOMINATOR = 1001
"""Largest denominator when turning a float fps into a rational (covers 29.97 = 30000/1001)."""
ATTACK_TEMP_SUFFIX = ".mp4"
EVEN_ALIGNMENT = 2
UINT8_MAX = 255


# --- Errors ------------------------------------------------------------------------------------


class VideoError(Exception):
    """Base class of every error raised by `wmkit.video`."""


class VideoReadError(VideoError, OSError):
    """The file cannot be opened or contains no decodable video stream."""


class VideoFormatError(VideoError, ValueError):
    """Frames have an unsupported shape, dtype range or size for the requested operation."""


# --- Helpers -----------------------------------------------------------------------------------


def _validate_frames(frames: torch.Tensor) -> None:
    if frames.ndim != 4 or frames.shape[1] != 3 or frames.shape[0] < 1:
        raise VideoFormatError(
            f"frames must have shape [T>=1, 3, H, W], got {tuple(frames.shape)}"
        )


def _validate_crf(crf: int) -> None:
    if not MIN_CRF <= crf <= MAX_CRF:
        raise VideoFormatError(f"crf must be in [{MIN_CRF}, {MAX_CRF}], got {crf}")


def _to_uint8_hwc(frames: torch.Tensor) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Quantise float frames [T, 3, H, W] in [0, 1] to uint8 [T, H, W, 3] (round to nearest)."""
    quantised = (frames.detach().clamp(0.0, 1.0).cpu() * UINT8_MAX).round().to(torch.uint8)
    return np.ascontiguousarray(quantised.permute(0, 2, 3, 1).numpy())


def _fps_to_fraction(fps: float) -> Fraction:
    if fps <= 0:
        raise VideoFormatError(f"fps must be > 0, got {fps}")
    return Fraction(fps).limit_denominator(MAX_FPS_DENOMINATOR)


# --- Public API --------------------------------------------------------------------------------


def read_video(path: str | Path) -> tuple[torch.Tensor, float]:
    """Decode the first video stream of `path` into float frames.

    Args:
        path: Any container PyAV can open (mp4, mkv, webm, avi...).

    Returns:
        `(frames [T, 3, H, W] float32 in [0, 1], fps)`.

    Raises:
        VideoReadError: missing file, unreadable container, or no decodable frame.
    """
    import av

    source = Path(path)
    if not source.is_file():
        raise VideoReadError(f"video file not found: {source}")
    decoded: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise VideoReadError(f"no video stream in {source}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else 0.0
            for frame in container.decode(stream):
                decoded.append(np.asarray(frame.to_ndarray(format=PIXEL_FORMAT_RGB), dtype=np.uint8))
    except av.error.FFmpegError as error:
        raise VideoReadError(f"cannot decode {source}: {error}") from error
    if not decoded:
        raise VideoReadError(f"no decodable frame in {source}")
    stacked = torch.from_numpy(np.stack(decoded)).permute(0, 3, 1, 2)
    return stacked.to(torch.float32) / UINT8_MAX, fps


def write_video(
    frames: torch.Tensor,
    path: str | Path,
    fps: float,
    codec: str = DEFAULT_CODEC,
    crf: int = DEFAULT_CRF,
) -> None:
    """Encode `frames` [T, 3, H, W] in [0, 1] to `path` (H.264 in MP4 by default).

    Frames are rounded to 8 bits. Lower `crf` means higher quality (0 = lossless for x264,
    23 = default, 51 = worst).

    Raises:
        VideoFormatError: bad shape, odd width or height (yuv420p needs even sizes), bad crf
            or fps.
        VideoError: the encoder is not available in this PyAV build.
    """
    import av

    _validate_frames(frames)
    _validate_crf(crf)
    rate = _fps_to_fraction(fps)
    height, width = int(frames.shape[2]), int(frames.shape[3])
    if width % EVEN_ALIGNMENT or height % EVEN_ALIGNMENT:
        raise VideoFormatError(
            f"{PIXEL_FORMAT_YUV420} needs even width and height, got {width}x{height}"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = _to_uint8_hwc(frames)
    try:
        with av.open(str(destination), mode="w") as container:
            stream = cast(Any, container.add_stream(codec, rate=rate))  # PyAV types it as a union
            stream.width = width
            stream.height = height
            stream.pix_fmt = PIXEL_FORMAT_YUV420
            stream.options = {"crf": str(crf), "preset": DEFAULT_PRESET}
            for index in range(pixels.shape[0]):
                video_frame = av.VideoFrame.from_ndarray(pixels[index], format=PIXEL_FORMAT_RGB)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except av.error.FFmpegError as error:
        raise VideoError(f"cannot encode {destination} with codec {codec!r}: {error}") from error


def h264_attack(frames: torch.Tensor, crf: int) -> torch.Tensor:
    """Round-trip `frames` through H.264 at quality `crf` (the standard video compression attack).

    Odd sizes are handled by edge-replicating one row/column before encoding and cropping it
    after decoding, so the output always has the input shape.

    Returns:
        Frames `[T, 3, H, W]` float32 in `[0, 1]`, on the device of the input.

    Raises:
        VideoFormatError: bad shape or crf.
    """
    _validate_frames(frames)
    _validate_crf(crf)
    height, width = int(frames.shape[2]), int(frames.shape[3])
    pad_bottom = height % EVEN_ALIGNMENT
    pad_right = width % EVEN_ALIGNMENT
    padded = frames.detach().cpu().to(torch.float32)
    if pad_bottom or pad_right:
        padded = F.pad(padded, (0, pad_right, 0, pad_bottom), mode="replicate")
    with tempfile.TemporaryDirectory() as scratch:
        clip = Path(scratch) / f"attack{ATTACK_TEMP_SUFFIX}"
        write_video(padded, clip, fps=1.0, crf=crf)
        decoded, _ = read_video(clip)
    return decoded[:, :, :height, :width].to(frames.device)
