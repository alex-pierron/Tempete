"""Watermark payloads: random messages and their hexadecimal text form.

Hex convention: bits are written most significant first; when `n_bits` is not a
multiple of four the last nibble is right-padded with zero bits. `hex_to_bits`
rejects non-zero padding so that a hex string maps to exactly one payload.
"""

from __future__ import annotations

import torch

BITS_PER_HEX_DIGIT: int = 4
HEX_PREFIX: str = "0x"
HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdef")


def random_messages(
    batch: int, n_bits: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw uniform random binary messages.

    Args:
        batch: Number of messages.
        n_bits: Bits per message.
        generator: CPU generator for reproducibility; None uses the global RNG.

    Returns:
        float32 tensor [batch, n_bits] with values in {0.0, 1.0}.

    Raises:
        ValueError: `batch` or `n_bits` is not strictly positive.
    """
    if batch <= 0 or n_bits <= 0:
        raise ValueError(f"batch and n_bits must be > 0, got batch={batch}, n_bits={n_bits}")
    bits = torch.randint(0, 2, (batch, n_bits), generator=generator)
    return bits.to(torch.float32)


def hex_length(n_bits: int) -> int:
    """Number of hex digits needed to write `n_bits` bits."""
    return -(-n_bits // BITS_PER_HEX_DIGIT)


def bits_to_hex(bits: torch.Tensor) -> str:
    """Encode a binary vector as a lowercase hex string (MSB first, zero right-padded).

    Args:
        bits: Tensor [n_bits]; soft values are thresholded at 0.5.

    Returns:
        Hex string of `ceil(n_bits / 4)` digits, without prefix.

    Raises:
        ValueError: `bits` is not a non-empty 1-D tensor.
    """
    if bits.ndim != 1 or bits.numel() == 0:
        raise ValueError(f"expected a non-empty 1-D bit vector, got shape {tuple(bits.shape)}")
    hard = (bits.detach().to("cpu") > 0.5).to(torch.int64).tolist()
    padding = hex_length(len(hard)) * BITS_PER_HEX_DIGIT - len(hard)
    hard = hard + [0] * padding
    digits: list[str] = []
    for start in range(0, len(hard), BITS_PER_HEX_DIGIT):
        nibble = 0
        for bit in hard[start:start + BITS_PER_HEX_DIGIT]:
            nibble = (nibble << 1) | bit
        digits.append(format(nibble, "x"))
    return "".join(digits)


def hex_to_bits(hex_str: str, n_bits: int) -> torch.Tensor:
    """Decode a hex string into a float32 bit vector of length `n_bits`.

    Args:
        hex_str: Hex digits, optional `0x` prefix, case-insensitive.
        n_bits: Expected payload size.

    Returns:
        float32 tensor [n_bits] in {0.0, 1.0}.

    Raises:
        ValueError: Non-hex characters, wrong digit count for `n_bits`, or non-zero padding bits.
    """
    if n_bits <= 0:
        raise ValueError(f"n_bits must be > 0, got {n_bits}")
    text = hex_str.strip().lower()
    text = text.removeprefix(HEX_PREFIX)
    expected_digits = hex_length(n_bits)
    if len(text) != expected_digits:
        raise ValueError(
            f"a {n_bits}-bit message needs {expected_digits} hex digits, got {len(text)}"
        )
    if not set(text) <= HEX_DIGITS:
        # int(..., 16) would also accept signs and underscores.
        raise ValueError(f"not a hexadecimal string: {hex_str!r}")
    value = int(text, 16)
    total_bits = expected_digits * BITS_PER_HEX_DIGIT
    padding = total_bits - n_bits
    if value & ((1 << padding) - 1):
        raise ValueError(f"the {padding} padding bit(s) of {hex_str!r} must be zero")
    bits = [(value >> (total_bits - 1 - index)) & 1 for index in range(n_bits)]
    return torch.tensor(bits, dtype=torch.float32)
