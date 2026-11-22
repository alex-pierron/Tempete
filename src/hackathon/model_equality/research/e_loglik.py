"""Per-token log-likelihood of endpoint answers under local candidate base models (team E).

For every stored answer (an endpoint, a prompt, a reply) and a candidate model we compute, for each
reply token, three numbers under the candidate (temperature 1, no truncation):

* `logp`  log p(token | context, previous reply tokens);
* `ent`   entropy H of the predictive distribution at that position;
* `var`   variance of log p under that same distribution.

`logp` gives the candidate-versus-candidate likelihood ratios (which base model wrote this text).
`ent` and `var` give a CALIBRATION statistic that needs no second model: if the reply was sampled
from the candidate at temperature 1, `sum(logp + ent) / sqrt(sum(var))` is close to N(0, 1). A
positive value means "more probable than the model's own samples" (lower temperature, a provider
system prompt that sharpens the answer...), a negative one "less probable" (temperature above 1, a
LoRA that moved the distribution, a different base model).

Method choices follow team D's `d_local` so that the two chains are comparable: the context is the
default chat template of the candidate applied to the user prompt alone plus the generation prompt
(variant `default`), the reply is tokenised on its own with `add_special_tokens=False`, EOS is not
scored. We add a second context, `nosys`, hand-built without any system block, because a deployed
endpoint may or may not inject the template's default system prompt: the endpoint is credited with
the better of the two.

fp8 simulation (E4M3, max 448, attention and MLP linear layers only, embeddings and `lm_head`
untouched, as vLLM `quantization=fp8` does): `w_tensor` one scale per weight tensor, `w_channel` one
scale per output row, `w8a8_tensor` per-tensor weights plus dynamic per-tensor activation scaling
over the whole sequence, `w8a8_token` per-tensor weights plus a dynamic scale per token (what a
decode step sees, since it handles one token at a time).

Everything here reads local files only; nothing touches an `ep-*` or `ref-*` endpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_models import (
    CANDIDATES,
    CandidateSpec,
    ModelManifestError,
    TemplateKind,
    get_candidate,
)
from hackathon.model_equality.store import Answer, load_many

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

TEMPLATE_DEFAULT = "default"
TEMPLATE_NOSYS = "nosys"
TEMPLATE_DATED = "dated"
TEMPLATE_VARIANTS: tuple[str, ...] = (TEMPLATE_DEFAULT, TEMPLATE_NOSYS)
LLAMA_DATED_FAMILIES: tuple[str, ...] = ("llama3.1", "llama3.2")
LLAMA_DATED_SYSTEM = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
"""System block of the official Llama 3.1 / 3.2 templates when no system message is given (the date is
the template's own fallback). The NousResearch mirror of 3.1 carries an older template without it,
so this variant is built by hand for every Llama 3.1 / 3.2 candidate."""

VARIANT_BASE = "base"
VARIANT_W_TENSOR = "w_tensor"
VARIANT_W_CHANNEL = "w_channel"
VARIANT_W8A8_TENSOR = "w8a8_tensor"
VARIANT_W8A8_TOKEN = "w8a8_token"
VARIANT_W_BLOCK = "w_block128"
VARIANT_W_BLOCK_A8_TOKEN = "w_block128_a8token"
FP8_VARIANTS: tuple[str, ...] = (
    VARIANT_W_TENSOR, VARIANT_W_CHANNEL, VARIANT_W8A8_TENSOR, VARIANT_W8A8_TOKEN,
)
FP8_VARIANTS_BLOCK: tuple[str, ...] = (VARIANT_W_BLOCK, VARIANT_W_BLOCK_A8_TOKEN)
FP8_VARIANTS_BF16: tuple[str, ...] = (
    VARIANT_W_TENSOR, VARIANT_W_CHANNEL, VARIANT_W_BLOCK, VARIANT_W8A8_TOKEN,
)
ACT_NONE = "none"
ACT_TENSOR = "tensor"
ACT_TOKEN = "token"
WEIGHT_TENSOR = "tensor"
WEIGHT_CHANNEL = "channel"
WEIGHT_BLOCK = "block"
FP8_BLOCK_SIZE = 128
"""Weight block edge of the block-wise scheme (128 x 128, as transformers' FineGrainedFP8)."""
FP8_RECIPES: Mapping[str, tuple[str, str]] = {
    VARIANT_W_TENSOR: (WEIGHT_TENSOR, ACT_NONE),
    VARIANT_W_CHANNEL: (WEIGHT_CHANNEL, ACT_NONE),
    VARIANT_W8A8_TENSOR: (WEIGHT_TENSOR, ACT_TENSOR),
    VARIANT_W8A8_TOKEN: (WEIGHT_TENSOR, ACT_TOKEN),
    VARIANT_W_BLOCK: (WEIGHT_BLOCK, ACT_NONE),
    VARIANT_W_BLOCK_A8_TOKEN: (WEIGHT_BLOCK, ACT_TOKEN),
}
E4M3_MAX = 448.0
MIN_SCALE_DENOMINATOR = 1e-12

STAGE_MATRIX = "matrix"
STAGE_FP8 = "fp8"
STAGE_FP8X = "fp8x"
STAGE_FP8BF16 = "fp8bf16"
STAGE_FP8LITE = "fp8lite"
"""fp32, the base and two fp8 recipes only: what the partition rule reads, in a fraction of the time."""
STAGES: tuple[str, ...] = (STAGE_MATRIX, STAGE_FP8)
FP8_STAGES: tuple[str, ...] = (STAGE_FP8, STAGE_FP8X, STAGE_FP8BF16, STAGE_FP8LITE)
FP8_VARIANTS_LITE: tuple[str, ...] = (VARIANT_W_TENSOR, VARIANT_W_CHANNEL)
ALL_STAGES: tuple[str, ...] = (STAGE_MATRIX, *FP8_STAGES)
DTYPE_BF16 = "bf16"
DTYPE_FP32 = "fp32"
FP32_MAX_PARAMS_BILLION = 8.5
"""Largest model run in float32 on a 40 GB A100 (8 B parameters = 32 GB of weights)."""
TEMPERATURE_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85, 1.2, 1.5, 2.0)
"""Sampling temperatures at which the matrix stage also records logp, entropy and variance, so the
temperature an endpoint was sampled at (a decoding setting, not part of the model identity) can be
fitted by maximum likelihood and separated from a real model difference."""
LOGIT_CHUNK_POSITIONS = 256
DEFAULT_BATCH_TOKENS = 8192
FP8_STAGE_REF_CANDIDATES: tuple[str, ...] = ("smollm2-135m", "smollm2-360m")
"""Candidates whose fp8 stage also scores the ref answers (small enough to be free)."""
SMALL_MODEL_PARAMS_BILLION = 0.5
FP8_COST_FACTOR = 4.0
MATRIX_COST_FACTOR = 1.0
NPZ_SEP = "__"
ROWS_ALL = "all"
ROWS_EP = "ep"
ROWS_REF = "ref"
EP_ENDPOINT_PREFIX = "ep-"
REF_ENDPOINT_PREFIX = "ref-"
SECONDS_PER_MINUTE = 60
WRITE_ATTEMPTS = 20
WRITE_RETRY_SECONDS = 15.0
"""The shared scratch can hit its inode quota for minutes ("Disk quota exceeded"): a write is
retried this many times, this many seconds apart, before the result is dumped to the log."""
FALLBACK_PREFIX = "LL_FALLBACK"
FALLBACK_DECIMALS = 5

REFERENCE_FILES: tuple[str, ...] = (
    "reference_answers.jsonl",
    "reference_answers_b.jsonl",
    "reference_answers_d.jsonl",
    "reference_answers_holdout.jsonl",
    "reference_answers_m1.jsonl",
    "reference_answers_holdout_first25.jsonl",
)
EP_FILE = "answers.jsonl"


class LoglikError(HackathonError):
    """A scoring run cannot proceed (missing model files, unknown template, out of memory)."""


@dataclass(frozen=True)
class RowScores:
    """Per-token statistics of a list of answers under one (model, context, variant).

    Attributes:
        n_tokens: Reply token count of each row (rows keep the order of the input list).
        logp: Flat array of `sum(n_tokens)` token log-probabilities.
        ent: Flat array of the predictive entropies at the same positions.
        var: Flat array of the predictive variances of log p at the same positions.
        temperatures: Temperatures of the optional tempered statistics (empty when absent).
        t_logp: `(len(temperatures), sum(n_tokens))` log-probabilities of the same tokens under the
            distribution `softmax(logits / T)`, or None.
        t_ent: Entropies of the tempered distributions, same shape, or None.
        t_var: Variances of log p under the tempered distributions, same shape, or None.
    """

    n_tokens: IntArray
    logp: FloatArray
    ent: FloatArray
    var: FloatArray
    temperatures: tuple[float, ...] = ()
    t_logp: FloatArray | None = None
    t_ent: FloatArray | None = None
    t_var: FloatArray | None = None

    def offsets(self) -> npt.NDArray[np.int64]:
        """Start index of every row in the flat arrays (length rows + 1)."""
        return np.concatenate([[0], np.cumsum(self.n_tokens, dtype=np.int64)])


def load_answers(
    answers_dir: Path, *, rows: str = ROWS_ALL, ep_files: Sequence[str] = (EP_FILE,),
    ref_files: Sequence[str] = REFERENCE_FILES,
) -> list[Answer]:
    """Load the stored answers, de-duplicated and in a stable (endpoint, id, repeat) order.

    Args:
        answers_dir: Folder holding the JSONL stores (`artifacts/model_equality` or a copy).
        rows: `ep`, `ref` or `all`.
        ep_files: Names of the ep stores.
        ref_files: Names of the ref stores (missing files are skipped).

    Raises:
        LoglikError: Unknown `rows` selector.
    """
    if rows not in (ROWS_ALL, ROWS_EP, ROWS_REF):
        raise LoglikError(f"rows must be one of {ROWS_ALL}, {ROWS_EP}, {ROWS_REF}; got {rows!r}")
    paths: list[Path] = []
    if rows in (ROWS_ALL, ROWS_EP):
        paths += [answers_dir / name for name in ep_files]
    if rows in (ROWS_ALL, ROWS_REF):
        paths += [answers_dir / name for name in ref_files]
    loaded = load_many([path for path in paths if path.is_file()])
    return sorted(loaded, key=lambda a: (a.endpoint, a.prompt_id, a.repeat))


def render_context(tokenizer: Any, kind: TemplateKind, template: str, prompt: str) -> str:
    """Text of the context that precedes the reply: user prompt plus generation prompt.

    `default` is the candidate's own chat template (it may inject a default system prompt or the
    Llama 3.1 date block); `nosys` is hand-built without any system block.

    Raises:
        LoglikError: Unknown template variant or kind.
    """
    if template == TEMPLATE_DEFAULT:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return str(rendered)
    if template == TEMPLATE_DATED and kind == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{LLAMA_DATED_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if template != TEMPLATE_NOSYS:
        raise LoglikError(f"unknown template variant {template!r} for kind {kind!r}")
    if kind in ("qwen", "smollm"):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if kind == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    raise LoglikError(f"unknown template kind {kind!r}")


def tokenise_rows(
    tokenizer: Any, kind: TemplateKind, template: str, answers: Sequence[Answer]
) -> list[tuple[list[int], list[int]]]:
    """`(context ids, reply ids)` of every answer; the reply is tokenised on its own."""
    pairs: list[tuple[list[int], list[int]]] = []
    context_cache: dict[str, list[int]] = {}
    for answer in answers:
        if answer.prompt not in context_cache:
            text = render_context(tokenizer, kind, template, answer.prompt)
            context_cache[answer.prompt] = list(tokenizer(text, add_special_tokens=False).input_ids)
        reply = list(tokenizer(answer.response, add_special_tokens=False).input_ids)
        pairs.append((context_cache[answer.prompt], reply))
    return pairs


def make_batches(
    lengths: Sequence[int], max_batch_tokens: int, *, single_sequence: bool
) -> list[list[int]]:
    """Group row indices into batches whose padded size stays under `max_batch_tokens`.

    Rows are sorted by decreasing length so the first row of a batch is its longest; a lone row
    longer than the budget still forms its own batch. `single_sequence` forces one row per batch
    (needed by the per-tensor dynamic activation scale, which must not mix sequences).
    """
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if current and (single_sequence or (len(current) + 1) * lengths[current[0]] > max_batch_tokens):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def position_statistics(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(logp of target, entropy, variance of log p)` for rows of logits, in float32.

    The variance is `E[(log p)^2] - H^2` under the predictive distribution; it is clipped at zero
    against rounding.
    """
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    second_moment = (probabilities * log_probabilities.square()).sum(dim=-1)
    variance = (second_moment - entropy.square()).clamp_min(0.0)
    chosen = log_probabilities.gather(1, targets[:, None])[:, 0]
    return chosen, entropy, variance


@torch.no_grad()
def _score_batch(
    model: Any,
    batch: Sequence[tuple[list[int], list[int]]],
    device: torch.device,
    temperatures: Sequence[float] = (),
) -> list[FloatArray]:
    """Statistics of one batch of `(context, reply)` pairs.

    One float32 array per row, of shape `(3 * (1 + len(temperatures)), reply tokens)`: logp, ent,
    var at temperature 1, then the same three rows for each temperature of the grid.
    """
    lengths = [len(context) + len(reply) for context, reply in batch]
    width = max(lengths)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (context, reply) in enumerate(batch):
        sequence = context + reply
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
    ids, mask = ids.to(device), mask.to(device)
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    n_rows = 3 * (1 + len(temperatures))
    results: list[FloatArray] = []
    for row, (context, reply) in enumerate(batch):
        if not reply:
            results.append(np.zeros((n_rows, 0), dtype=np.float32))
            continue
        start = len(context) - 1
        selected = hidden[row, start : start + len(reply)]
        targets = ids[row, len(context) : len(context) + len(reply)]
        parts: list[torch.Tensor] = []
        for chunk_start in range(0, len(reply), LOGIT_CHUNK_POSITIONS):
            chunk = slice(chunk_start, chunk_start + LOGIT_CHUNK_POSITIONS)
            logits = model.lm_head(selected[chunk]).float()
            stats = list(position_statistics(logits, targets[chunk]))
            for temperature in temperatures:
                stats.extend(position_statistics(logits / temperature, targets[chunk]))
            parts.append(torch.stack(stats))
        results.append(torch.cat(parts, dim=1).cpu().numpy().astype(np.float32))
    return results


def score_pairs(
    model: Any,
    pairs: Sequence[tuple[list[int], list[int]]],
    device: torch.device,
    *,
    max_batch_tokens: int = DEFAULT_BATCH_TOKENS,
    single_sequence: bool = False,
    temperatures: Sequence[float] = (),
) -> RowScores:
    """Per-token statistics of every `(context, reply)` pair under `model`.

    A CUDA out-of-memory error splits the batch in two and retries; a single row that still does
    not fit raises `LoglikError`. With `temperatures`, the tempered statistics are recorded too.
    """
    lengths = [len(context) + len(reply) for context, reply in pairs]
    per_row: list[FloatArray | None] = [None] * len(pairs)

    def run(indices: list[int]) -> None:
        try:
            outputs = _score_batch(model, [pairs[i] for i in indices], device, temperatures)
        except torch.cuda.OutOfMemoryError as exc:
            if len(indices) == 1:
                raise LoglikError(f"row of {lengths[indices[0]]} tokens does not fit in memory") from exc
            torch.cuda.empty_cache()
            middle = len(indices) // 2
            run(indices[:middle])
            run(indices[middle:])
            return
        for index, output in zip(indices, outputs, strict=True):
            per_row[index] = output

    for batch in make_batches(lengths, max_batch_tokens, single_sequence=single_sequence):
        run(batch)
    filled = [row for row in per_row if row is not None]
    if len(filled) != len(pairs):
        raise LoglikError("internal error: some rows were not scored")
    n_tokens = np.array([len(reply) for _, reply in pairs], dtype=np.int32)
    n_rows = 3 * (1 + len(temperatures))
    stacked = np.concatenate(filled, axis=1) if filled else np.zeros((n_rows, 0), dtype=np.float32)
    tempered: dict[str, FloatArray | None] = {"t_logp": None, "t_ent": None, "t_var": None}
    if temperatures:
        tempered = {
            "t_logp": stacked[3::3], "t_ent": stacked[4::3], "t_var": stacked[5::3],
        }
    return RowScores(
        n_tokens=n_tokens, logp=stacked[0], ent=stacked[1], var=stacked[2],
        temperatures=tuple(float(t) for t in temperatures), **tempered,
    )


def quantise_e4m3(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round `values / scale` to E4M3 and scale back (float32 result)."""
    scaled = (values.float() / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale


def fp8_target_linears(model: Any) -> dict[str, nn.Linear]:
    """The attention and MLP linear layers of a Llama-like model (the ones vLLM quantises)."""
    targets: dict[str, nn.Linear] = {}
    for layer_index, layer in enumerate(model.model.layers):
        for block_name in ("self_attn", "mlp"):
            block = getattr(layer, block_name)
            for child_name, child in block.named_children():
                if isinstance(child, nn.Linear):
                    targets[f"layers.{layer_index}.{block_name}.{child_name}"] = child
    return targets


def quantise_weight(weight: torch.Tensor, mode: str) -> torch.Tensor:
    """E4M3 round trip of a 2-D weight with one scale per tensor, per output row or per 128x128 block.

    Raises:
        LoglikError: Unknown mode.
    """
    if mode == WEIGHT_TENSOR:
        scale = weight.abs().max().clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
        return quantise_e4m3(weight, scale)
    if mode == WEIGHT_CHANNEL:
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
        return quantise_e4m3(weight, scale)
    if mode != WEIGHT_BLOCK:
        raise LoglikError(f"unknown weight quantisation mode {mode!r}")
    rows, cols = weight.shape
    pad_rows, pad_cols = (-rows) % FP8_BLOCK_SIZE, (-cols) % FP8_BLOCK_SIZE
    padded = torch.nn.functional.pad(weight, (0, pad_cols, 0, pad_rows))
    n_row_blocks, n_col_blocks = padded.shape[0] // FP8_BLOCK_SIZE, padded.shape[1] // FP8_BLOCK_SIZE
    blocks = padded.reshape(n_row_blocks, FP8_BLOCK_SIZE, n_col_blocks, FP8_BLOCK_SIZE)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
    quantised = quantise_e4m3(blocks, scale).reshape(padded.shape)
    return quantised[:rows, :cols]


class Fp8Simulator:
    """Switch a loaded model between its original weights and simulated fp8 variants in place.

    The pristine weights of the target layers are kept on the CPU the first time a variant is
    applied; every `apply` starts again from them, so variants never compound.
    """

    def __init__(self, model: Any) -> None:
        """Bind to `model` (nothing is copied until the first `apply`)."""
        self.targets = fp8_target_linears(model)
        self._pristine: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []

    def _remember(self) -> None:
        if not self._pristine:
            self._pristine = {
                name: layer.weight.detach().to("cpu", copy=True) for name, layer in self.targets.items()
            }

    def restore(self) -> None:
        """Put the pristine weights back and remove the activation hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for name, original in self._pristine.items():
            layer = self.targets[name]
            layer.weight.data.copy_(original.to(layer.weight.device))

    def apply(self, variant: str) -> None:
        """Quantise the target layers as `variant` says (`base` just restores).

        Raises:
            LoglikError: Unknown variant.
        """
        if variant == VARIANT_BASE:
            self.restore()
            return
        if variant not in FP8_RECIPES:
            raise LoglikError(f"unknown fp8 variant {variant!r}")
        weight_mode, activation_mode = FP8_RECIPES[variant]
        self._remember()
        self.restore()
        for name, layer in self.targets.items():
            weight = self._pristine[name].to(layer.weight.device, torch.float32)
            layer.weight.data.copy_(quantise_weight(weight, weight_mode).to(layer.weight.dtype))
            if activation_mode != ACT_NONE:
                self._hooks.append(layer.register_forward_pre_hook(_activation_hook(activation_mode)))


def _activation_hook(mode: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Forward pre-hook that fake-quantises the input of a linear layer to E4M3."""

    def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inputs = args[0]
        if mode == ACT_TOKEN:
            amax = inputs.abs().amax(dim=-1, keepdim=True)
        else:
            amax = inputs.abs().max()
        scale = amax.float().clamp_min(MIN_SCALE_DENOMINATOR) / E4M3_MAX
        return (quantise_e4m3(inputs, scale).to(inputs.dtype), *args[1:])

    return hook


def calibration_z(logp: FloatArray, ent: FloatArray, var: FloatArray) -> float:
    """`sum(logp + ent) / sqrt(sum(var))`: ~N(0,1) for text sampled from the model at temperature 1."""
    total_var = float(np.sum(var, dtype=np.float64))
    if total_var <= 0.0:
        return 0.0
    return float(np.sum(logp.astype(np.float64) + ent.astype(np.float64)) / math.sqrt(total_var))


def save_scores(
    path: Path,
    answers: Sequence[Answer],
    scores: Mapping[tuple[str, str], RowScores],
    meta: Mapping[str, Any],
) -> None:
    """Write one `.npz` (atomically) with the row keys and every `(template, variant)` result.

    Array names: `endpoint`, `prompt_id`, `repeat`, `n_tokens`, `<template>__<variant>__logp|ent|var`
    and `meta` (a JSON string). One container per run, as the shared scratch has few free inodes.
    """
    arrays: dict[str, Any] = {
        "endpoint": np.array([a.endpoint for a in answers]),
        "prompt_id": np.array([a.prompt_id for a in answers]),
        "repeat": np.array([a.repeat for a in answers], dtype=np.int32),
        "meta": np.array(json.dumps(dict(meta), sort_keys=True)),
    }
    for (template, variant), result in scores.items():
        arrays["n_tokens"] = result.n_tokens
        stem = f"{template}{NPZ_SEP}{variant}{NPZ_SEP}"
        arrays[stem + "logp"] = result.logp
        arrays[stem + "ent"] = result.ent
        arrays[stem + "var"] = result.var
        if result.t_logp is not None and result.t_ent is not None and result.t_var is not None:
            arrays[stem + "t_logp"] = result.t_logp
            arrays[stem + "t_ent"] = result.t_ent
            arrays[stem + "t_var"] = result.t_var
            arrays[stem + "temperatures"] = np.array(result.temperatures, dtype=np.float64)
    write_container(path, arrays)


def write_container(
    path: Path, arrays: Mapping[str, Any], *, attempts: int = WRITE_ATTEMPTS,
    delay: float = WRITE_RETRY_SECONDS, sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Write a `.npz`, retrying on OS errors such as a full inode quota.

    When `path` already exists (a placeholder created while the quota still had room), the data is
    written INTO it: no new inode is needed. Otherwise a temporary file is written and renamed.

    Raises:
        LoglikError: Every attempt failed.
    """
    for attempt in range(1, attempts + 1):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                with path.open("wb") as handle:
                    np.savez(handle, **arrays)
            else:
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **arrays)
                os.replace(temporary, path)
            return
        except OSError as exc:
            if attempt == attempts:
                raise LoglikError(f"cannot write {path} after {attempts} attempts: {exc}") from exc
            print(f"write of {path.name} failed ({exc}); attempt {attempt}/{attempts}", flush=True)
            sleep(delay)


def _row_sum_lists(flat: FloatArray, offsets: npt.NDArray[np.int64]) -> list[float]:
    """Per-row sums of a flat per-token array (rounded for the log line)."""
    cumulative = np.concatenate([[0.0], np.cumsum(flat.astype(np.float64))])
    sums = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    return [round(float(value), FALLBACK_DECIMALS) for value in sums]


def fallback_line(
    candidate: str, stage: str, answers: Sequence[Answer], scores: Mapping[tuple[str, str], RowScores]
) -> str:
    """One log line holding the per-ROW sums of a result (all the partition rule needs).

    Used when the `.npz` cannot be written: the log file already exists, so it needs no new inode.
    Format: `LL_FALLBACK <json>`; the JSON has the row keys and, per `template/variant`, the token
    counts and the per-row sums of logp, ent, var and (when present) the tempered statistics.
    """
    payload: dict[str, Any] = {
        "candidate": candidate, "stage": stage,
        "rows": [[a.endpoint, a.prompt_id, a.repeat] for a in answers], "variants": {},
    }
    for (template, variant), result in scores.items():
        offsets = result.offsets()
        entry: dict[str, Any] = {
            "n_tokens": [int(n) for n in result.n_tokens],
            "logp": _row_sum_lists(result.logp, offsets), "ent": _row_sum_lists(result.ent, offsets),
            "var": _row_sum_lists(result.var, offsets),
        }
        if result.t_logp is not None and result.t_ent is not None and result.t_var is not None:
            entry["temperatures"] = list(result.temperatures)
            entry["t_logp"] = [_row_sum_lists(row, offsets) for row in result.t_logp]
            entry["t_ent"] = [_row_sum_lists(row, offsets) for row in result.t_ent]
            entry["t_var"] = [_row_sum_lists(row, offsets) for row in result.t_var]
        payload["variants"][f"{template}/{variant}"] = entry
    return f"{FALLBACK_PREFIX} " + json.dumps(payload, separators=(",", ":"))


def load_scores(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], RowScores]]:
    """Read a container written by `save_scores`: `(keys and meta, {(template, variant): scores})`."""
    with np.load(path, allow_pickle=False) as data:
        header: dict[str, Any] = {
            "endpoint": [str(x) for x in data["endpoint"]],
            "prompt_id": [str(x) for x in data["prompt_id"]],
            "repeat": [int(x) for x in data["repeat"]],
            "meta": json.loads(str(data["meta"])),
        }
        n_tokens = data["n_tokens"].astype(np.int32)
        out: dict[tuple[str, str], RowScores] = {}
        for name in data.files:
            parts = name.split(NPZ_SEP)
            if len(parts) == 3 and parts[2] == "logp":
                stem = NPZ_SEP.join(parts[:2]) + NPZ_SEP
                tempered: dict[str, Any] = {}
                if stem + "t_logp" in data.files:
                    tempered = {
                        "temperatures": tuple(float(t) for t in data[stem + "temperatures"]),
                        "t_logp": data[stem + "t_logp"].astype(np.float32),
                        "t_ent": data[stem + "t_ent"].astype(np.float32),
                        "t_var": data[stem + "t_var"].astype(np.float32),
                    }
                out[(parts[0], parts[1])] = RowScores(
                    n_tokens=n_tokens,
                    logp=data[stem + "logp"].astype(np.float32),
                    ent=data[stem + "ent"].astype(np.float32),
                    var=data[stem + "var"].astype(np.float32),
                    **tempered,
                )
    return header, out


@dataclass(frozen=True)
class Task:
    """One unit of GPU work: a candidate under a stage."""

    candidate: str
    stage: str
    cost: float = field(default=1.0, compare=False)


def task_cost(spec: CandidateSpec, stage: str) -> float:
    """Relative cost of a task (parameters times a stage factor), for load balancing."""
    factor = FP8_COST_FACTOR if stage in FP8_STAGES else MATRIX_COST_FACTOR
    return max(spec.params_billion, SMALL_MODEL_PARAMS_BILLION) * factor


def plan_assignments(
    specs: Sequence[CandidateSpec], workers: int, stages: Sequence[str] = STAGES
) -> list[list[Task]]:
    """Split the tasks over `workers` GPUs (longest processing time first).

    Inside each worker the tasks of an earlier stage always come first, so a deadline cuts the
    fp8 measurements before the candidate matrix. Extra-tier candidates only get the matrix stage.

    Raises:
        LoglikError: `workers` below 1.
    """
    if workers < 1:
        raise LoglikError("workers must be at least 1")
    buckets: list[list[Task]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for stage in stages:
        tasks = sorted(
            (
                Task(spec.key, stage, task_cost(spec, stage))
                for spec in specs
                if stage == STAGE_MATRIX or spec.tier == "core"
            ),
            key=lambda task: -task.cost,
        )
        for task in tasks:
            target = min(range(workers), key=lambda index: loads[index])
            buckets[target].append(task)
            loads[target] += task.cost
    return buckets


def parse_task_list(items: Sequence[str]) -> list[Task]:
    """Tasks from `candidate:stage` strings.

    Raises:
        LoglikError: A malformed item, an unknown candidate or an unknown stage.
    """
    tasks: list[Task] = []
    for item in items:
        candidate, separator, stage = item.partition(":")
        if not separator or stage not in ALL_STAGES:
            raise LoglikError(f"{item!r} is not candidate:stage with a stage in {ALL_STAGES}")
        try:
            get_candidate(candidate)
        except ModelManifestError as exc:
            raise LoglikError(str(exc)) from exc
        tasks.append(Task(candidate, stage))
    return tasks


def distribute_groups(
    groups: Sequence[Sequence[Task]], extras: Sequence[Task], gpu_count: int
) -> list[list[Task]]:
    """Spread task groups, then loose tasks, over however many GPUs the runner gave us.

    Each group (a base model with its matrix and fp8 stages) stays on ONE GPU, in order, so a model
    is loaded once per stage pair. Groups, then loose tasks, go one by one to the least loaded GPU,
    the load being the summed `task_cost` (parameters times a stage factor): with four or three GPUs
    each base gets its own GPU; with two, the 3B base is alone on one GPU and the 1.5B and 1B bases
    share the other. Ties go to the lowest index.

    Raises:
        LoglikError: `gpu_count` below 1.
    """
    if gpu_count < 1:
        raise LoglikError("gpu_count must be at least 1")
    buckets: list[list[Task]] = [[] for _ in range(gpu_count)]
    loads = [0.0] * gpu_count

    def cost(tasks: Sequence[Task]) -> float:
        return sum(task_cost(get_candidate(task.candidate), task.stage) for task in tasks)

    for group in groups:
        target = min(range(gpu_count), key=lambda i: (loads[i], i))
        buckets[target].extend(group)
        loads[target] += cost(group)
    for task in extras:
        target = min(range(gpu_count), key=lambda i: (loads[i], i))
        buckets[target].append(task)
        loads[target] += cost([task])
    return buckets


def visible_gpu_ids(cuda_visible_devices: str | None, detected: int) -> list[str]:
    """The GPU ids this job may use: `CUDA_VISIBLE_DEVICES` when set, else `0 .. detected - 1`."""
    if cuda_visible_devices is not None and cuda_visible_devices.strip():
        return [part.strip() for part in cuda_visible_devices.split(",") if part.strip()]
    return [str(index) for index in range(detected)]


def dtype_for(spec: CandidateSpec, stage: str) -> str:
    """`fp32` for the fp8 stage when the model fits a 40 GB GPU in float32, else `bf16`.

    The matrix stage is always bf16: it only has to tell models apart, with margins far above
    bf16 rounding. The fp8 stage measures shifts of a few thousandths of a nat, so it needs fp32.
    """
    if stage in (STAGE_FP8, STAGE_FP8X, STAGE_FP8LITE) and spec.params_billion <= FP32_MAX_PARAMS_BILLION:
        return DTYPE_FP32
    return DTYPE_BF16


def variants_for(stage: str) -> tuple[str, ...]:
    """Model variants scored in a stage."""
    if stage == STAGE_MATRIX:
        return (VARIANT_BASE,)
    if stage == STAGE_FP8X:
        return (VARIANT_BASE, *FP8_VARIANTS_BLOCK)
    if stage == STAGE_FP8BF16:
        return (VARIANT_BASE, *FP8_VARIANTS_BF16)
    if stage == STAGE_FP8LITE:
        return (VARIANT_BASE, *FP8_VARIANTS_LITE)
    return (VARIANT_BASE, *FP8_VARIANTS)


def templates_for(stage: str, spec: CandidateSpec | None = None) -> tuple[str, ...]:
    """Context templates scored in a stage.

    The `other` kind has no hand-built variant; Llama 3.1 / 3.2 add the dated system block.
    """
    if stage != STAGE_MATRIX or (spec is not None and spec.template == "other"):
        return (TEMPLATE_DEFAULT,)
    if spec is not None and spec.template == "llama3" and spec.family in LLAMA_DATED_FAMILIES:
        return (*TEMPLATE_VARIANTS, TEMPLATE_DATED)
    return TEMPLATE_VARIANTS


def rows_for(spec: CandidateSpec, stage: str) -> str:
    """Which answers a task scores: everything for the matrix, ep (+ ref for tiny models) for fp8."""
    if stage == STAGE_MATRIX:
        return ROWS_ALL
    return ROWS_ALL if spec.key in FP8_STAGE_REF_CANDIDATES else ROWS_EP


def file_digest(path: Path) -> str:
    """sha256 of a small file (tokenizer, config), for the provenance block."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_task(
    task: Task, *, cache_dir: Path, answers_dir: Path, out_dir: Path, device: torch.device,
    max_batch_tokens: int = DEFAULT_BATCH_TOKENS, force: bool = False,
) -> Path | None:
    """Score one task and write its container; returns None when the output already exists."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy, imported late

    spec = get_candidate(task.candidate)
    output = out_dir / f"ll_{task.stage}_{spec.key}.npz"
    if output.is_file() and output.stat().st_size > 0 and not force:
        print(f"skip {output.name}: exists", flush=True)
        return None
    model_dir = cache_dir / spec.key
    if not (model_dir / "config.json").is_file():
        raise LoglikError(f"{model_dir}: model files missing (run the downloader first)")
    started = time.time()
    answers = load_answers(answers_dir, rows=rows_for(spec, task.stage))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    dtype_name = dtype_for(spec, task.stage)
    dtype = torch.float32 if dtype_name == DTYPE_FP32 else torch.bfloat16
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=dtype
    )
    model = model.to(device).eval()
    print(f"{spec.key}/{task.stage}: loaded in {time.time() - started:.0f}s "
          f"({dtype_name}, {len(answers)} rows)", flush=True)
    simulator = Fp8Simulator(model) if task.stage in FP8_STAGES else None
    results: dict[tuple[str, str], RowScores] = {}
    contexts: dict[str, str] = {}
    for template in templates_for(task.stage, spec):
        pairs = tokenise_rows(tokenizer, spec.template, template, answers)
        contexts[template] = str(tokenizer.decode(pairs[0][0])) if pairs else ""
        for variant in variants_for(task.stage):
            if simulator is not None:
                simulator.apply(variant)
            started_variant = time.time()
            results[(template, variant)] = score_pairs(
                model, pairs, device, max_batch_tokens=max_batch_tokens,
                single_sequence=variant == VARIANT_W8A8_TENSOR,
                temperatures=TEMPERATURE_GRID if task.stage == STAGE_MATRIX else (),
            )
            print(f"{spec.key}/{task.stage}/{template}/{variant}: "
                  f"{time.time() - started_variant:.0f}s", flush=True)
    meta = {
        "candidate": spec.key, "repo": spec.repo, "family": spec.family, "stage": task.stage,
        "dtype": dtype_name, "torch": torch.__version__, "rows": len(answers),
        "temperature_grid": list(TEMPERATURE_GRID) if task.stage == STAGE_MATRIX else [],
        "first_context": contexts, "seconds": round(time.time() - started),
        "tokenizer_sha256": file_digest(model_dir / "tokenizer.json"),
        "device": str(device),
    }
    try:
        save_scores(output, answers, results, meta)
    except LoglikError:
        print(fallback_line(spec.key, task.stage, answers, results), flush=True)
        raise
    if simulator is not None:
        simulator.restore()
    release_memory(device)
    print(f"wrote {output.name}", flush=True)
    return output


def release_memory(device: torch.device) -> None:
    """Collect garbage then empty the CUDA cache.

    A finished task leaves reference cycles (module hooks, the simulator, tracebacks of a failed
    task) that keep tens of GB alive: run 1 lost two tasks to an out-of-memory error for this reason.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def worker_command(args: argparse.Namespace) -> int:
    """Run the tasks of one worker (one GPU), skipping those that no longer fit the deadline."""
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    tasks = [Task(item["candidate"], item["stage"]) for item in plan[args.index]]
    device = torch.device(args.device)
    failures = 0
    for task in tasks:
        if args.deadline_epoch and time.time() > args.deadline_epoch:
            print(f"deadline passed: skipping {task.candidate}/{task.stage}", flush=True)
            continue
        try:
            run_task(task, cache_dir=Path(args.cache_dir), answers_dir=Path(args.answers_dir),
                     out_dir=Path(args.out_dir), device=device, force=args.force)
        except Exception as exc:  # noqa: BLE001  one failed task must not kill the other tasks of the GPU
            failures += 1
            print(f"FAILED {task.candidate}/{task.stage}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            release_memory(device)
    return 1 if failures else 0


def plan_command(args: argparse.Namespace) -> int:
    """Write the worker assignments (`[[{candidate, stage}, ...], ...]`)."""
    specs = [spec for spec in CANDIDATES if args.keys is None or spec.key in args.keys]
    buckets = plan_assignments(specs, args.workers, tuple(args.stages))
    payload = [[{"candidate": t.candidate, "stage": t.stage} for t in bucket] for bucket in buckets]
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for index, bucket in enumerate(buckets):
        print(f"worker {index}: " + ", ".join(f"{t.candidate}/{t.stage}" for t in bucket))
    return 0


def plan_groups_command(args: argparse.Namespace) -> int:
    """Write a plan for `--gpus` GPUs from groups (`a:matrix,a:fp8lite`) and loose extra tasks.

    Candidates whose model files are absent from `--cache-dir` are skipped with a message.
    """
    cache_dir = Path(args.cache_dir)

    def present(task: Task) -> bool:
        ok = (cache_dir / task.candidate / "config.json").is_file()
        if not ok:
            print(f"skipping {task.candidate}:{task.stage} (model files missing)", file=sys.stderr)
        return ok

    groups = [[t for t in parse_task_list(group.split(",")) if present(t)] for group in args.groups]
    extras = [t for t in parse_task_list(args.extras) if present(t)]
    buckets = distribute_groups(groups, extras, args.gpus)
    payload = [[{"candidate": t.candidate, "stage": t.stage} for t in bucket] for bucket in buckets]
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for index, bucket in enumerate(buckets):
        print(f"gpu slot {index}: " + ", ".join(f"{t.candidate}/{t.stage}" for t in bucket))
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--workers", type=int, default=4)
    plan.add_argument("--keys", nargs="*", default=None)
    plan.add_argument("--stages", nargs="+", default=list(STAGES), choices=ALL_STAGES)
    plan.add_argument("--out", required=True)
    groups = sub.add_parser("plan-groups")
    groups.add_argument("--gpus", type=int, required=True)
    groups.add_argument("--groups", nargs="+", required=True)
    groups.add_argument("--extras", nargs="*", default=[])
    groups.add_argument("--cache-dir", required=True)
    groups.add_argument("--out", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--cache-dir", required=True)
    worker.add_argument("--answers-dir", required=True)
    worker.add_argument("--out-dir", required=True)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--deadline-epoch", type=float, default=0.0)
    worker.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to `plan`, `plan-groups` or `worker`."""
    args = parse_args(argv)
    if args.command == "plan":
        return plan_command(args)
    if args.command == "plan-groups":
        return plan_groups_command(args)
    return worker_command(args)


if __name__ == "__main__":
    sys.exit(main())
