"""Adaptive batch size that fills the GPU (target: >= 30 GB on a 40 GB A100) without ever crashing.

Rule of the team (Victor, 21/11/2026): use at least ~30 GB of the 40 GB of each A100, keep ~5 GB of
margin, measure the throughput, and fall back automatically on an out-of-memory error. The sizer
reads the peak memory of each full batch, extrapolates linearly (a fixed weights part plus a part
proportional to the batch) towards the target and shrinks on OOM. It is pure logic plus a thin
probe, so it is unit-tested without a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from hackathon.common.errors import HackathonError

BYTES_PER_GB = 1e9
DEFAULT_TARGET_GB = 32.0
"""Peak reserved memory aimed at: above the 30 GB asked for, below the cap."""
DEFAULT_MARGIN_GB = 5.0
"""Memory left free on the device (cuDNN workspaces, fragmentation, the CUDA context)."""
GROWTH_LIMIT_FACTOR = 2.0
"""The batch at most doubles after one measurement (the linear model is only a local estimate)."""
OOM_SHRINK_FACTOR = 0.7
"""Batch multiplier applied after an out-of-memory error."""
RELATIVE_CHANGE_THRESHOLD = 0.1
"""Smaller adjustments are ignored: they only add noise."""
MINIMUM_BATCH = 2


class GpuMemoryError(HackathonError):
    """Even the smallest batch does not fit in memory."""


class MemoryProbe(Protocol):
    """Reads and resets the device's peak memory."""

    def reset_peak(self) -> None:
        """Start a new measurement window."""

    def peak_gb(self) -> float | None:
        """Peak reserved memory since the last reset, in GB (None when not measurable)."""

    def release_cache(self) -> None:
        """Give cached blocks back (called before retrying after an OOM)."""


class CudaMemoryProbe:
    """Peak *reserved* memory of one CUDA device: what `nvidia-smi` shows."""

    def __init__(self, device: torch.device | str) -> None:
        self._device = torch.device(device)

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats(self._device)

    def peak_gb(self) -> float:
        return torch.cuda.max_memory_reserved(self._device) / BYTES_PER_GB

    def release_cache(self) -> None:
        torch.cuda.empty_cache()


class NullMemoryProbe:
    """CPU/MPS: nothing to measure, the batch size stays where it is."""

    def reset_peak(self) -> None:
        return None

    def peak_gb(self) -> float | None:
        return None

    def release_cache(self) -> None:
        return None


def make_probe(device: torch.device | str) -> MemoryProbe:
    """CUDA probe for a CUDA device, null probe otherwise."""
    return CudaMemoryProbe(device) if torch.device(device).type == "cuda" else NullMemoryProbe()


def device_budget_gb(device: torch.device | str, margin_gb: float = DEFAULT_MARGIN_GB) -> float:
    """Usable memory of a CUDA device: total minus the safety margin (0 when not CUDA)."""
    if torch.device(device).type != "cuda":
        return 0.0
    return (
        torch.cuda.get_device_properties(torch.device(device)).total_memory / BYTES_PER_GB
        - margin_gb
    )


@dataclass
class BatchSizer:
    """Current batch size and the rules that move it.

    Attributes:
        size: batch size to use now.
        minimum: floor (an OOM at the floor raises `GpuMemoryError`).
        maximum: ceiling (no point growing past what the data loader or the chunk offers).
        target_gb: peak memory to aim at.
        cap_gb: peak memory that must never be exceeded (device total minus the margin).
        weights_gb: memory held independently of the batch (models, optimizer state).
        oom_events: out-of-memory errors absorbed so far.
        history: (batch size, peak GB) of every full batch measured, for the job log.
    """

    size: int
    minimum: int = MINIMUM_BATCH
    maximum: int = 512
    target_gb: float = DEFAULT_TARGET_GB
    cap_gb: float = 40.0 - DEFAULT_MARGIN_GB
    weights_gb: float = 0.0
    oom_events: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1 <= self.minimum <= self.size <= self.maximum:
            raise GpuMemoryError(
                f"Need 1 <= minimum ({self.minimum}) <= size ({self.size}) <= maximum ({self.maximum})."
            )
        self.target_gb = min(self.target_gb, self.cap_gb)

    def after_success(self, batch_length: int, peak_gb: float | None) -> None:
        """Adapt after a batch that completed. Partial batches say nothing about the full size."""
        if peak_gb is None or batch_length < self.size:
            return
        self.history.append((batch_length, peak_gb))
        per_item = max((peak_gb - self.weights_gb) / batch_length, 1e-6)
        if peak_gb > self.cap_gb:
            wanted = math.floor((self.cap_gb - self.weights_gb) / per_item)
        elif peak_gb < self.target_gb:
            wanted = min(
                math.floor((self.target_gb - self.weights_gb) / per_item),
                math.floor(self.size * GROWTH_LIMIT_FACTOR),
            )
        else:
            return
        wanted = max(self.minimum, min(self.maximum, wanted))
        over_cap = peak_gb > self.cap_gb
        if over_cap or abs(wanted - self.size) / self.size >= RELATIVE_CHANGE_THRESHOLD:
            self.size = wanted

    def after_oom(self) -> None:
        """Shrink after an out-of-memory error.

        Raises:
            GpuMemoryError: the batch is already at the minimum.
        """
        self.oom_events += 1
        failed = self.size
        if failed <= self.minimum:
            raise GpuMemoryError(f"Out of memory at the minimum batch size ({self.minimum}).")
        # Never grow back to the size that just failed.
        self.maximum = max(self.minimum, min(self.maximum, failed - 1))
        self.size = max(self.minimum, min(self.maximum, math.floor(failed * OOM_SHRINK_FACTOR)))


def process_adaptively[T, R](
    items: Sequence[T],
    sizer: BatchSizer,
    process: Callable[[Sequence[T]], R],
    probe: MemoryProbe,
    oom_errors: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,),
) -> list[R]:
    """Run `process` over `items` in batches whose size follows `sizer`; results in order.

    An out-of-memory error shrinks the batch and retries the same items; a full batch that
    measured below the target lets the next batch grow.
    """
    results: list[R] = []
    position = 0
    while position < len(items):
        batch = items[position : position + sizer.size]
        probe.reset_peak()
        try:
            result = process(batch)
        except oom_errors:
            probe.release_cache()
            sizer.after_oom()
            continue
        results.append(result)
        position += len(batch)
        sizer.after_success(len(batch), probe.peak_gb())
    return results
