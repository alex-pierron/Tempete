"""Tests for wmkit.messages."""

from __future__ import annotations

import pytest
import torch

from wmkit.messages import bits_to_hex, hex_to_bits, random_messages


def test_random_messages_shape_dtype_and_values() -> None:
    messages = random_messages(5, 48)
    assert messages.shape == (5, 48)
    assert messages.dtype == torch.float32
    assert set(messages.unique().tolist()) <= {0.0, 1.0}


def test_random_messages_are_reproducible_with_a_seeded_generator() -> None:
    first = random_messages(4, 32, torch.Generator().manual_seed(7))
    second = random_messages(4, 32, torch.Generator().manual_seed(7))
    other = random_messages(4, 32, torch.Generator().manual_seed(8))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


@pytest.mark.parametrize("batch,n_bits", [(0, 8), (3, 0), (-1, 8)])
def test_random_messages_rejects_non_positive_sizes(batch: int, n_bits: int) -> None:
    with pytest.raises(ValueError):
        random_messages(batch, n_bits)


def test_known_hex_values() -> None:
    assert bits_to_hex(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])) == "af"
    assert bits_to_hex(torch.tensor([1.0, 1.0, 1.0])) == "e"  # right-padded: 1110
    assert hex_to_bits("0xAF", 8).tolist() == [1, 0, 1, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("n_bits", list(range(1, 70)) + [96, 255, 256, 1024])
def test_hex_round_trip_for_any_length(n_bits: int) -> None:
    original = random_messages(1, n_bits, torch.Generator().manual_seed(n_bits))[0]
    restored = hex_to_bits(bits_to_hex(original), n_bits)
    assert torch.equal(original, restored)


def test_bits_to_hex_thresholds_soft_bits() -> None:
    assert bits_to_hex(torch.tensor([0.9, 0.4, 0.51, 0.1])) == "a"


def test_hex_to_bits_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        hex_to_bits("abc", 8)  # wrong length
    with pytest.raises(ValueError):
        hex_to_bits("zz", 8)  # not hex
    with pytest.raises(ValueError):
        hex_to_bits("f", 3)  # non-zero padding bit
    with pytest.raises(ValueError):
        hex_to_bits("f", 0)


def test_bits_to_hex_rejects_non_vectors() -> None:
    with pytest.raises(ValueError):
        bits_to_hex(torch.zeros(2, 4))
    with pytest.raises(ValueError):
        bits_to_hex(torch.zeros(0))
