"""Team C decode batching: out-of-memory fallback, autotune choice and configuration."""

from __future__ import annotations

import pytest
import torch

from hackathon.wmloc.research.c_batching import (
    MIN_BATCH,
    BatchingError,
    BatchTrial,
    autotune_batch,
    batched_logits,
    configure_batch,
    describe,
    is_out_of_memory,
)

N_BITS = 4
RESOLUTION = 8


class FakeDecoder:
    """Logit = mean pixel; raises an out-of-memory error above `memory_limit` crops per pass."""

    method = "trustmark"
    resolution = RESOLUTION
    device = "cpu"

    def __init__(self, memory_limit: int = 10**9, batch_size: int = 32) -> None:
        self.memory_limit = memory_limit
        self.batch_size = batch_size
        self.calls: list[int] = []

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append(len(images))
        if len(images) > self.memory_limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        return images.mean(dim=(1, 2, 3))[:, None].repeat(1, N_BITS)


def _crops(count: int) -> torch.Tensor:
    return (
        torch.arange(count, dtype=torch.float32)
        .view(count, 1, 1, 1)
        .repeat(1, 3, RESOLUTION, RESOLUTION)
    )


def test_batched_logits_splits_by_batch_size_and_keeps_the_order() -> None:
    decoder = FakeDecoder(batch_size=16)
    logits = batched_logits(decoder, _crops(40))
    assert decoder.calls == [16, 16, 8]
    assert logits[:, 0].tolist() == [float(i) for i in range(40)]


def test_out_of_memory_halves_the_batch_and_retries_without_losing_a_window() -> None:
    decoder = FakeDecoder(memory_limit=20, batch_size=128)
    logits = batched_logits(decoder, _crops(100))
    assert decoder.batch_size == 16  # 128 -> 64 -> 32 -> 16 (limit 20)
    assert logits[:, 0].tolist() == [float(i) for i in range(100)]
    assert max(c for c in decoder.calls[3:]) <= 16


def test_out_of_memory_at_the_minimum_batch_raises_a_typed_error() -> None:
    decoder = FakeDecoder(memory_limit=MIN_BATCH - 1, batch_size=MIN_BATCH)
    with pytest.raises(BatchingError):
        batched_logits(decoder, _crops(20))


def test_other_runtime_errors_are_not_swallowed() -> None:
    class Broken(FakeDecoder):
        def logits(self, images: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        batched_logits(Broken(), _crops(4))


def test_is_out_of_memory_recognises_the_allocator_message() -> None:
    assert is_out_of_memory(RuntimeError("MPS backend out of memory"))
    assert not is_out_of_memory(RuntimeError("something else"))
    assert not is_out_of_memory(ValueError("out of memory"))


def test_autotune_returns_a_candidate_sets_the_decoder_and_never_picks_a_failed_trial() -> None:
    decoder = FakeDecoder(memory_limit=100)
    chosen, trials = autotune_batch(decoder, candidates=(32, 64, 256), windows=256)
    assert chosen in (32, 64) and decoder.batch_size == chosen
    failed = next(t for t in trials if t.batch == 256)
    assert failed.windows_per_second == 0.0  # fell back after an OOM: not a clean trial
    assert "autotune batch" in describe(trials, chosen)


def test_autotune_keeps_the_baseline_unless_a_larger_batch_is_clearly_faster() -> None:
    decoder = FakeDecoder()
    trials = [BatchTrial(32, 1000.0, 1.0), BatchTrial(128, 1010.0, 5.0)]
    assert describe(trials, 32).startswith("autotune batch -> 32")
    with pytest.raises(BatchingError):
        autotune_batch(decoder, candidates=())


def test_configure_batch_accepts_integers_and_rejects_garbage() -> None:
    decoder = FakeDecoder()
    assert configure_batch(decoder, "512") == "batch size 512" and decoder.batch_size == 512
    with pytest.raises(BatchingError):
        configure_batch(decoder, "many")
    with pytest.raises(BatchingError):
        configure_batch(decoder, "0")
