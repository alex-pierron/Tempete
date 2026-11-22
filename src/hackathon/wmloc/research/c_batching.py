"""Large decode batches with an automatic out-of-memory fallback and a throughput autotune (team C).

A100 GPUs decode far faster with hundreds of windows per pass than with the 32 the first runs
used, and the window search is the only expensive stage. Two safeguards keep a big batch safe:

* `batched_logits` splits a stack of crops by `decoder.batch_size` and, on an out-of-memory error,
  halves the batch size (down to `MIN_BATCH`) and retries the failing chunk, so a task never dies
  because a batch was too ambitious; the reduced size sticks for the rest of the process.
* `autotune_batch` times a fixed probe workload for each candidate batch size and keeps the
  fastest one; the setting is kept only if it is faster than the baseline (a larger batch that is
  not faster is never chosen), and the table (windows per second, peak memory) is returned for
  the logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from hackathon.wmloc.research.errors import ResearchError

DEFAULT_BATCH = 32
MIN_BATCH = 8
CANDIDATE_BATCHES: tuple[int, ...] = (32, 128, 256, 512, 1024)
PROBE_WINDOWS = 1536
"""Crops decoded per candidate during the autotune (a multiple of every candidate)."""
BYTES_PER_GIB = 1024**3
MIN_GAIN = 1.03
"""A larger batch must be at least 3 % faster than the baseline to replace it (noise margin)."""


class BatchingError(ResearchError):
    """Invalid batch configuration."""


class BatchedDecoder(Protocol):
    """What the batching helpers need from a decoder.

    `device` and `batch_size` are optional attributes (read with defaults, `batch_size` is written
    back after an out-of-memory fallback), so any duck-typed decoder works.
    """

    method: str
    resolution: int

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Bit logits [B, n_bits] for images at the decoder resolution."""
        ...


@dataclass(frozen=True)
class BatchTrial:
    """Throughput of one candidate batch size.

    Attributes:
        batch: Candidate batch size.
        windows_per_second: Probe throughput (0.0 if the candidate ran out of memory).
        peak_gib: Peak GPU memory allocated during the probe, GiB (0.0 off CUDA).
    """

    batch: int
    windows_per_second: float
    peak_gib: float


def is_out_of_memory(error: BaseException) -> bool:
    """True for a CUDA / MPS / allocator out-of-memory failure."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _release_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def batched_logits(decoder: BatchedDecoder, crops: torch.Tensor) -> torch.Tensor:
    """Logits [N, n_bits] of `crops`, decoded `decoder.batch_size` at a time.

    On an out-of-memory error the batch size is halved (never below `MIN_BATCH`) and the failing
    chunk retried; the smaller size is kept in `decoder.batch_size`.

    Raises:
        BatchingError: the decoder still runs out of memory at `MIN_BATCH`.
    """
    parts: list[torch.Tensor] = []
    start = 0
    device = str(getattr(decoder, "device", "cpu"))
    while start < len(crops):
        batch = max(MIN_BATCH, int(getattr(decoder, "batch_size", DEFAULT_BATCH)))
        try:
            parts.append(decoder.logits(crops[start : start + batch]))
        except (RuntimeError, MemoryError) as error:
            if not is_out_of_memory(error):
                raise
            if batch <= MIN_BATCH:
                raise BatchingError(f"out of memory even at batch {MIN_BATCH}") from error
            _release_cache(device)
            setattr(decoder, "batch_size", max(MIN_BATCH, batch // 2))  # noqa: B010
            continue
        start += batch
    return torch.cat(parts) if parts else torch.empty((0, 0))


def _probe(decoder: BatchedDecoder, batch: int, windows: int) -> BatchTrial:
    """Time `windows` random crops decoded `batch` at a time (one warm-up pass first)."""
    device = str(getattr(decoder, "device", "cpu"))
    crops = torch.rand(windows, 3, decoder.resolution, decoder.resolution, device=device)
    previous = int(getattr(decoder, "batch_size", DEFAULT_BATCH))
    setattr(decoder, "batch_size", batch)  # noqa: B010
    try:
        batched_logits(decoder, crops[:batch])  # warm-up: kernels, cudnn plans, allocator
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.time()
        batched_logits(decoder, crops)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = max(time.time() - started, 1e-6)
        reduced = (
            int(getattr(decoder, "batch_size", batch)) < batch
        )  # OOM fallback: not a clean trial
        peak = (
            torch.cuda.max_memory_allocated() / BYTES_PER_GIB if device.startswith("cuda") else 0.0
        )
        return BatchTrial(batch, 0.0 if reduced else windows / elapsed, peak)
    except BatchingError:
        return BatchTrial(batch, 0.0, 0.0)
    finally:
        setattr(decoder, "batch_size", previous)  # noqa: B010
        _release_cache(device)


def autotune_batch(
    decoder: BatchedDecoder,
    candidates: tuple[int, ...] = CANDIDATE_BATCHES,
    windows: int = PROBE_WINDOWS,
) -> tuple[int, list[BatchTrial]]:
    """Fastest candidate batch size and the table of trials; sets `decoder.batch_size`.

    The baseline is the first candidate; a larger batch replaces it only if it is at least
    `MIN_GAIN` times faster, so the setting is never kept unless it is measurably quicker.

    Raises:
        BatchingError: no candidate given.
    """
    if not candidates:
        raise BatchingError("no candidate batch size")
    trials = [_probe(decoder, batch, windows) for batch in candidates]
    best = trials[0]
    for trial in trials[1:]:
        if trial.windows_per_second > best.windows_per_second * MIN_GAIN:
            best = trial
    setattr(decoder, "batch_size", best.batch)  # noqa: B010
    return best.batch, trials


def describe(trials: list[BatchTrial], chosen: int) -> str:
    """One log line with the throughput and peak memory of every trial."""
    cells = ", ".join(
        f"{t.batch}: {t.windows_per_second:.0f}/s {t.peak_gib:.1f} GiB" for t in trials
    )
    return f"autotune batch -> {chosen} ({cells})"


def configure_batch(decoder: BatchedDecoder, setting: str) -> str:
    """Apply `--decode-batch` (`auto` or an integer) to a decoder; returns a log line.

    Raises:
        BatchingError: the setting is neither `auto` nor a positive integer.
    """
    if setting == "auto":
        chosen, trials = autotune_batch(decoder)
        return describe(trials, chosen)
    try:
        size = int(setting)
    except ValueError as error:
        raise BatchingError(
            f"--decode-batch must be 'auto' or an integer, got {setting!r}"
        ) from error
    if size <= 0:
        raise BatchingError(f"--decode-batch must be positive, got {size}")
    setattr(decoder, "batch_size", size)  # noqa: B010
    return f"batch size {size}"
