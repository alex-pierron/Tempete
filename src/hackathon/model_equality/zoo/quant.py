"""E4M3 fake quantisation: what a vLLM `quantization=fp8` server does to a linear layer.

Weights get ONE scale per tensor (`amax / 448`); a W8A8 server also quantises each layer input
dynamically with one scale per tensor. Both are simulated by quantise-then-dequantise: the result
carries exactly the information loss of fp8 (3 mantissa bits) while the matmul stays in bf16. The
same recipe as `research/d_local.py` (team D, SmolLM2), here on bf16 GPU tensors.
"""

from __future__ import annotations

import torch

from hackathon.model_equality.zoo.config import E4M3_MAX, MIN_SCALE_DENOMINATOR


def per_tensor_scale(values: torch.Tensor) -> torch.Tensor:
    """The scale that maps the largest magnitude of `values` to the E4M3 maximum (float32)."""
    amax = values.detach().abs().amax().to(torch.float32)
    return amax.clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX


def fake_quantize_e4m3(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round `values / scale` to float8 E4M3 and scale back; same dtype as `values`."""
    scaled = (values.to(torch.float32) / scale).clamp(-E4M3_MAX, E4M3_MAX)
    restored = scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale
    return restored.to(values.dtype)


def quantize_weight_in_place(weight: torch.Tensor) -> float:
    """Replace `weight` by its per-tensor E4M3 round trip; return the relative error.

    The relative error is `||w' - w|| / ||w||` (Frobenius), a quick check that the quantisation
    did something and stayed in the expected range (about 2e-2 for E4M3).
    """
    with torch.no_grad():
        original = weight.detach().to(torch.float32)
        quantized = fake_quantize_e4m3(weight.detach(), per_tensor_scale(weight))
        norm = float(original.norm())
        error = float((quantized.to(torch.float32) - original).norm())
        weight.copy_(quantized)
    return error / norm if norm > 0.0 else 0.0


def quantize_activation(inputs: torch.Tensor) -> torch.Tensor:
    """Dynamic per-tensor E4M3 round trip of a layer input (the A8 of W8A8)."""
    return fake_quantize_e4m3(inputs, per_tensor_scale(inputs))
