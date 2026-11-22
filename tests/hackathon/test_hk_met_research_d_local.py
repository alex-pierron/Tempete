"""Team D local fp8 simulations: shapes, error bounds and recipe linear layers on random tensors (no model download)."""

from __future__ import annotations

import torch
from torch import nn

from hackathon.model_equality.research.d_local import (
    SEARCH_VARIANTS,
    VARIANT_BLOCK128,
    VARIANT_BLOCK128_A128,
    VARIANT_W_BF16,
    RecipeLinear,
    quantise_blocks,
    quantise_e4m3,
    quantise_groups,
)


def test_e4m3_round_trip_keeps_a_relative_error_below_seven_percent() -> None:
    values = torch.randn(64, 64)
    scale = values.abs().max() / 448.0
    quantised = quantise_e4m3(values, scale)
    big = values.abs() > 0.5
    relative = ((quantised - values).abs() / values.abs())[big]
    assert relative.max() < 0.07


def test_block_quantisation_handles_shapes_that_are_not_multiples_of_128() -> None:
    weight = torch.randn(192, 576)
    quantised = quantise_blocks(weight)
    assert quantised.shape == weight.shape
    assert (quantised - weight).abs().max() < 0.1 * weight.abs().max()


def test_group_quantisation_keeps_the_shape_and_scales_per_token() -> None:
    inputs = torch.randn(3, 5, 200) * torch.tensor([1.0, 100.0, 0.01])[:, None, None]
    quantised = quantise_groups(inputs)
    assert quantised.shape == inputs.shape
    relative = (quantised - inputs).abs().mean(dim=(1, 2)) / inputs.abs().mean(dim=(1, 2))
    assert relative.max() < 0.07  # per-token scaling: tiny and huge tokens are equally accurate


def test_every_searched_recipe_is_close_to_the_float_layer() -> None:
    linear = nn.Linear(200, 130)
    inputs = torch.randn(4, 200)
    reference = linear(inputs)
    for recipe in SEARCH_VARIANTS:
        layer = RecipeLinear(linear, recipe)
        error = (layer(inputs) - reference).abs().mean() / reference.abs().mean()
        assert float(error) < 0.2, recipe
    assert float((RecipeLinear(linear, VARIANT_BLOCK128)(inputs) - reference).abs().mean()) > 0.0
    assert RecipeLinear(linear, VARIANT_W_BF16)(inputs).dtype == torch.float32
    assert RecipeLinear(linear, VARIANT_BLOCK128_A128).weight.requires_grad is False
