"""Tests of the adaptive batch sizer (pure logic: no GPU needed)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from hackathon.unlearning.research.retain_proxy.gpu_memory import (
    BatchSizer,
    GpuMemoryError,
    NullMemoryProbe,
    make_probe,
    process_adaptively,
)

WEIGHTS_GB = 7.0
TARGET_GB = 32.0
CAP_GB = 35.0


class _FakeOom(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError."""


def _sizer(size: int = 16, **overrides: float) -> BatchSizer:
    return BatchSizer(
        size=size,
        target_gb=TARGET_GB,
        cap_gb=CAP_GB,
        weights_gb=WEIGHTS_GB,
        **overrides,  # type: ignore[arg-type]
    )


def test_grows_towards_the_target_but_at_most_doubles() -> None:
    sizer = _sizer(16)
    sizer.after_success(16, 12.0)  # (12 - 7) / 16 = 0.3125 GB per image -> wants 80, capped at 32
    assert sizer.size == 32
    sizer.after_success(32, 22.0)  # 0.469 GB per image -> (32 - 7) / 0.469 = 53
    assert 50 <= sizer.size <= 56


def test_stays_when_inside_the_window_or_change_is_tiny() -> None:
    sizer = _sizer(48)
    sizer.after_success(48, 33.0)  # between target (32) and cap (35): leave it
    assert sizer.size == 48
    sizer.after_success(48, 31.5)  # wants (32 - 7) / ((31.5 - 7) / 48) = 49: under 10 %
    assert sizer.size == 48


def test_shrinks_when_above_the_cap() -> None:
    sizer = _sizer(64)
    sizer.after_success(64, 39.0)  # 0.5 GB per image -> (35 - 7) / 0.5 = 56
    assert sizer.size == 56
    assert sizer.history == [(64, 39.0)]


def test_partial_batches_and_missing_measures_change_nothing() -> None:
    sizer = _sizer(32)
    sizer.after_success(10, 8.0)
    sizer.after_success(32, None)
    assert sizer.size == 32 and sizer.history == []


def test_oom_shrinks_then_raises_at_the_minimum() -> None:
    sizer = _sizer(20, minimum=4)
    sizer.after_oom()
    assert sizer.size == 14 and sizer.maximum == 19 and sizer.oom_events == 1
    sizer.after_success(14, 8.0)  # tempting to double, but never past the failed size
    assert sizer.size <= 19
    tiny = _sizer(4, minimum=4)
    with pytest.raises(GpuMemoryError):
        tiny.after_oom()


def test_invalid_construction_is_rejected() -> None:
    with pytest.raises(GpuMemoryError):
        BatchSizer(size=1, minimum=2)
    with pytest.raises(GpuMemoryError):
        BatchSizer(size=600, maximum=512)


def test_target_is_clamped_to_the_cap() -> None:
    sizer = BatchSizer(size=8, target_gb=50.0, cap_gb=30.0)
    assert sizer.target_gb == 30.0


class _ScriptedProbe:
    """Reports a peak proportional to the last batch, like a real GPU would."""

    def __init__(self) -> None:
        self.last_batch = 0
        self.released = 0

    def reset_peak(self) -> None:
        return None

    def peak_gb(self) -> float:
        return WEIGHTS_GB + 0.4 * self.last_batch

    def release_cache(self) -> None:
        self.released += 1


def test_process_adaptively_survives_oom_and_keeps_order() -> None:
    items = list(range(300))
    probe = _ScriptedProbe()
    seen: list[int] = []

    def process(batch: Sequence[int]) -> list[int]:
        if len(batch) > 40:
            raise _FakeOom("out of memory")
        probe.last_batch = len(batch)
        seen.append(len(batch))
        return [item * 2 for item in batch]

    sizer = _sizer(16)
    blocks = process_adaptively(items, sizer, process, probe, oom_errors=(_FakeOom,))
    flattened = [value for block in blocks for value in block]
    assert flattened == [item * 2 for item in items], "every item processed once, in order"
    assert sizer.oom_events >= 1 and probe.released == sizer.oom_events
    assert max(seen) <= 40 and sizer.size <= 40
    assert max(seen) > 16, "the batch must have grown before hitting the limit"


def test_process_adaptively_propagates_unrelated_errors() -> None:
    def boom(batch: Sequence[int]) -> None:
        raise ValueError("not an OOM")

    with pytest.raises(ValueError):
        process_adaptively([1, 2, 3], _sizer(2), boom, NullMemoryProbe(), oom_errors=(_FakeOom,))


def test_cpu_probe_is_null() -> None:
    probe = make_probe("cpu")
    assert probe.peak_gb() is None
