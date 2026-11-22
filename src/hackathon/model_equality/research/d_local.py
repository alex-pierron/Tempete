"""Likelihood tests with a LOCAL copy of the base model (team D, 21/11/2026).

The ref endpoints are SmolLM2-135M-Instruct (their identity answer is "SmolLM / Hugging Face" and the
weights are in the local Hugging Face cache). A local copy gives what the sampled text alone cannot: the
exact probability of every reply token under the base model and under a locally SIMULATED fp8 version.

Method (the SAME function must be reused for the large ep families, so the template and the tokenisation
are fixed here): context = `tokenizer.apply_chat_template([user message], add_generation_prompt=True)`,
reply tokenised with `add_special_tokens=False`, log-probability of every reply token under the model run
in float32 on CPU. For a reply r and a model variant v:

    ll_v(r)     = sum over reply tokens of log p_v(token | context, previous tokens)
    Lambda_v(r) = ll_v(r) - ll_base(r)               (log-likelihood ratio, fp8 variant against base)

fp8 simulations (E4M3, max 448, as vLLM `quantization=fp8` does for the linear layers; embeddings and
lm_head stay in float): `w_tensor` = weights quantised with ONE scale per tensor, `w_channel` = one scale
per output row, `w8a8_tensor` = per-tensor weights plus dynamic per-tensor quantisation of the activations
entering every linear layer.

CRITERIA FROZEN BEFORE THE LIKELIHOOD-RATIO RESULTS WERE LOOKED AT (only the base-model mean log-likelihood
of ref-01 = -0.864, ref-02 = -0.886 and ref-03 = -0.837 per token had been seen):

1. fp8: on the cells common to the 9 ref, paired by cell, the mean per-token Lambda of ref-02 minus that
   of ref-01 has a paired t >= 3 for at least one fp8 variant ("fp8 SEEN"). The variant with the largest t
   is the recipe used afterwards. Each disguise (ref-03, ref-05, ref-06, ref-08) must have |t| < 3 against
   ref-01 on that variant, otherwise fp8 is "CONFOUNDED" by that disguise.
2. LoRA: the per-token base log-likelihood of ref-09 (0.3) is below that of ref-01 with a paired t <= -2,
   ref-04 (0.7) and ref-07 (1.5) with t <= -3 ("LoRA SEEN").
3. AUC: pair distance |mean per-token statistic difference| over N cells, AUC of the pairs
   (target, S endpoint) against the S-S pairs, N = 5, 10, 20, 40; fp8 uses Lambda of the variant of
   criterion 1, LoRA uses the base log-likelihood. Success: fp8 AUC >= 0.8 at N = 20.
4. Tokens needed: N_tokens = (3 sigma / mu)^2 with mu the mean per-token difference to ref-01 and sigma
   the per-token standard deviation of the per-cell difference (an ESTIMATE).

RECIPE SEARCH PROTOCOL (frozen before the run; the three first variants above gave t = 1.5, 0.4, 1.2 and an
observed mean Lambda gap of 0.003 per token against a local KL of 0.016: the server's fp8 is not the one
simulated, so the recipe is searched on the free ref):
* variants: `block128` (weights quantised per 128 x 128 block, as in transformers FineGrainedFP8),
  `block128_a128` (+ dynamic activation quantisation per token and group of 128 inputs), `w_lmhead`
  (per-tensor weights and a quantised, untied lm_head), `w_e5m2` (per-tensor E5M2 weights), `w8a8_token`
  (per-tensor weights, dynamic per-token activation scaling), `w_bf16` (per-tensor E4M3 weights, bfloat16
  rounding of the linear outputs);
* SEARCH on cells 0-59 of the scored cells, ref-01 and ref-02 only: the variant with the largest paired t of
  the per-token Lambda gap (ref-02 minus ref-01) is selected, among the three first and the six new;
* CONFIRMATION on cells 60-119, all nine ref: the selected variant is a RECIPE FOUND when the paired t of
  ref-02 against ref-01 is >= 3 and every disguise (ref-03, ref-05, ref-06, ref-08) has |t| < 3; otherwise
  "recipe not found" and the idea is abandoned.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as torch_functional
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
E4M3_MAX = 448.0
MIN_SCALE_DENOMINATOR = 1e-12
THREADS = 2
VARIANT_BASE = "base"
VARIANT_W_TENSOR = "w_tensor"
VARIANT_W_CHANNEL = "w_channel"
VARIANT_W8A8 = "w8a8_tensor"
FP8_VARIANTS = (VARIANT_W_TENSOR, VARIANT_W_CHANNEL, VARIANT_W8A8)
VARIANT_BLOCK128 = "block128"
VARIANT_BLOCK128_A128 = "block128_a128"
VARIANT_W_LMHEAD = "w_lmhead"
VARIANT_W_E5M2 = "w_e5m2"
VARIANT_W8A8_TOKEN = "w8a8_token"
VARIANT_W_BF16 = "w_bf16"
SEARCH_VARIANTS = (VARIANT_BLOCK128, VARIANT_BLOCK128_A128, VARIANT_W_LMHEAD, VARIANT_W_E5M2,
                   VARIANT_W8A8_TOKEN, VARIANT_W_BF16)
BLOCK = 128
E5M2_MAX = 57344.0


def quantise_e4m3(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round `values / scale` to E4M3 and scale back (float32 result)."""
    scaled = (values / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale


class QuantLinear(nn.Module):
    """A linear layer with fp8-simulated weights (and optionally dynamically quantised activations)."""

    def __init__(self, linear: nn.Linear, per_channel: bool, quantise_activations: bool) -> None:
        """Quantise the weights of `linear` once; keep its bias in float."""
        super().__init__()
        weight = linear.weight.data
        if per_channel:
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
        else:
            scale = weight.abs().max().clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
        self.weight = nn.Parameter(quantise_e4m3(weight, scale), requires_grad=False)
        self.bias = linear.bias
        self.quantise_activations = quantise_activations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Linear transform with the simulated fp8 weights."""
        if self.quantise_activations:
            scale = inputs.abs().max().clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
            inputs = quantise_e4m3(inputs, scale)
        return torch_functional.linear(inputs, self.weight, self.bias)


def load_model(name: str = DEFAULT_MODEL) -> Any:
    """The causal LM in float32 on CPU, from the local Hugging Face cache only."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(THREADS)
    model: Any = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float32)
    model.train(False)  # inference mode (the typed equivalent of .eval())
    return model


def load_tokenizer(name: str = DEFAULT_MODEL) -> Any:
    """The tokenizer from the local cache."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    return AutoTokenizer.from_pretrained(name, local_files_only=True)


def simulate_fp8(model: Any, per_channel: bool, quantise_activations: bool) -> Any:
    """Replace every attention and MLP linear layer of a Llama-type model by its fp8 simulation."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for child_name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, child_name, QuantLinear(child, per_channel, quantise_activations))
    return model


def _pad_to_multiple(values: torch.Tensor, multiple: int, dim: int) -> torch.Tensor:
    """Zero-pad `dim` of `values` up to a multiple of `multiple`."""
    remainder = values.shape[dim] % multiple
    if remainder == 0:
        return values
    pad = [0, 0] * (values.dim() - dim - 1) + [0, multiple - remainder]
    return torch_functional.pad(values, pad)


def quantise_blocks(weight: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """E4M3 round of a 2-D weight with one scale per `block` x `block` tile (zero-padded, then cropped)."""
    rows, cols = weight.shape
    padded = _pad_to_multiple(_pad_to_multiple(weight, block, 0), block, 1)
    tiles = padded.reshape(padded.shape[0] // block, block, padded.shape[1] // block, block)
    scale = tiles.abs().amax(dim=(1, 3), keepdim=True).clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
    quantised = quantise_e4m3(tiles, scale).reshape(padded.shape)
    return quantised[:rows, :cols]


def quantise_groups(inputs: torch.Tensor, group: int = BLOCK) -> torch.Tensor:
    """E4M3 round of activations with one scale per token and per group of `group` input features."""
    features = inputs.shape[-1]
    padded = _pad_to_multiple(inputs, group, inputs.dim() - 1)
    grouped = padded.reshape(*padded.shape[:-1], padded.shape[-1] // group, group)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
    return quantise_e4m3(grouped, scale).reshape(padded.shape)[..., :features]


class RecipeLinear(nn.Module):
    """A linear layer under one of the searched fp8 recipes."""

    def __init__(self, linear: nn.Linear, recipe: str) -> None:
        """Quantise the weights once according to `recipe`; keep the bias in float."""
        super().__init__()
        weight = linear.weight.data
        self.recipe = recipe
        if recipe in (VARIANT_BLOCK128, VARIANT_BLOCK128_A128):
            quantised = quantise_blocks(weight)
        elif recipe == VARIANT_W_E5M2:
            scale = weight.abs().max().clamp_min(MIN_SCALE_DENOMINATOR) / E5M2_MAX
            quantised = ((weight / scale).clamp(-E5M2_MAX, E5M2_MAX).to(torch.float8_e5m2).to(torch.float32)) * scale
        else:
            scale = weight.abs().max().clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
            quantised = quantise_e4m3(weight, scale)
        self.weight = nn.Parameter(quantised, requires_grad=False)
        self.bias = linear.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Linear transform under the recipe."""
        if self.recipe == VARIANT_BLOCK128_A128:
            inputs = quantise_groups(inputs)
        elif self.recipe == VARIANT_W8A8_TOKEN:
            scale = inputs.abs().amax(dim=-1, keepdim=True).clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
            inputs = quantise_e4m3(inputs, scale)
        output = torch_functional.linear(inputs, self.weight, self.bias)
        if self.recipe == VARIANT_W_BF16:
            output = output.to(torch.bfloat16).to(torch.float32)
        return output


def simulate_recipe(model: Any, recipe: str) -> Any:
    """Replace every attention / MLP linear layer (and the lm_head for `w_lmhead`) by `recipe`'s simulation."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for child_name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, child_name, RecipeLinear(child, recipe))
    if recipe == VARIANT_W_LMHEAD:
        model.lm_head = RecipeLinear(model.lm_head, VARIANT_W_TENSOR)
    return model


def build_variants(name: str = DEFAULT_MODEL, variants: Sequence[str] = (VARIANT_BASE, *FP8_VARIANTS)) -> dict[str, Any]:
    """Model variants by name (`base` and the fp8 simulations asked for)."""
    recipes = {VARIANT_W_TENSOR: (False, False), VARIANT_W_CHANNEL: (True, False), VARIANT_W8A8: (False, True)}
    built: dict[str, Any] = {}
    for variant in variants:
        model = load_model(name)
        if variant in SEARCH_VARIANTS:
            model = simulate_recipe(model, variant)
        elif variant != VARIANT_BASE:
            per_channel, activations = recipes[variant]
            model = simulate_fp8(model, per_channel, activations)
        built[variant] = model
    return built


@torch.no_grad()
def reply_log_probability(model: Any, tokenizer: Any, prompt: str, reply: str) -> tuple[float, int]:
    """(sum of token log-probabilities of `reply`, number of reply tokens) under `model`.

    Context = default chat template with the prompt as the only (user) message and the generation
    prompt appended; an empty reply scores (0.0, 0).
    """
    context = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids
    reply_ids = tokenizer(reply, add_special_tokens=False, return_tensors="pt").input_ids
    if reply_ids.shape[1] == 0:
        return 0.0, 0
    ids = torch.cat([context_ids, reply_ids], dim=1)
    log_probabilities = torch.log_softmax(model(ids).logits[0, context_ids.shape[1] - 1:-1], dim=-1)
    tokens = ids[0, context_ids.shape[1]:]
    return float(log_probabilities.gather(1, tokens[:, None])[:, 0].sum()), int(tokens.shape[0])
