"""`ZooLinear`: a linear layer with a LoRA branch of runtime strength and optional fp8, plus the
short LoRA training that produces the ONE adapter applied at 0.3, 0.7 and 1.5.

Why our own LoRA instead of `peft`: the cluster venv has neither `peft` nor `accelerate`, and the
only thing the zoo needs is `W x + s * (alpha / r) * B A x` with `s` set at run time. The adapter
is therefore never merged, which keeps its full precision in bf16 (a merged delta smaller than the
bf16 spacing of the weights would be rounded away) and makes the three strengths free.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from safetensors.torch import load_file, save_file
from torch import nn

from hackathon.model_equality.zoo.config import (
    LINEAR_TARGETS,
    LORA_ALPHA,
    LORA_GRAD_CLIP,
    LORA_LEARNING_RATE,
    LORA_MAX_TOKENS,
    LORA_RANK,
    LORA_WARMUP_STEPS,
    MIN_BATCH_SIZE,
    ZooConfigError,
)
from hackathon.model_equality.zoo.quant import quantize_activation, quantize_weight_in_place

LORA_A_SUFFIX = ".lora_a"
LORA_B_SUFFIX = ".lora_b"
IGNORE_INDEX = -100
"""Label value that the language-model loss skips (prompt and padding positions)."""
BYTES_PER_GIB = 1024**3


class AdapterError(ValueError):
    """An adapter file does not match the model it is loaded into."""


class TrainingDataError(ValueError):
    """The instruction set is unusable (empty after filtering, wrong columns)."""


class ZooLinear(nn.Module):
    """`nn.Linear` replacement: bf16 weights, optional LoRA branch, optional W8A8 activations.

    `lora_scale` multiplies the adapter delta (`0.0` = the base model). `activation_fp8` quantises
    the layer input per tensor (used by the W8A8 variant after the weights were quantised).
    """

    def __init__(self, base: nn.Linear) -> None:
        """Take over the weight and bias of `base` (no copy, same parameters)."""
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.register_parameter("lora_a", None)
        self.register_parameter("lora_b", None)
        self.lora_scale = 0.0
        self.alpha_over_rank = LORA_ALPHA / LORA_RANK
        self.activation_fp8 = False

    def attach_lora(self, rank: int = LORA_RANK) -> None:
        """Create the low-rank factors: A Kaiming-uniform, B zero (the delta starts at zero)."""
        device = self.weight.device
        factor_a = torch.empty(rank, self.in_features, dtype=torch.float32, device=device)
        nn.init.kaiming_uniform_(factor_a, a=math.sqrt(5.0))
        self.lora_a = nn.Parameter(factor_a)
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32,
                                               device=device))

    @property
    def has_lora(self) -> bool:
        """True once `attach_lora` ran."""
        return self.lora_a is not None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """`W x + b`, plus `scale * alpha/r * B A x` when the adapter is on."""
        layer_input = quantize_activation(inputs) if self.activation_fp8 else inputs
        output = torch_functional.linear(layer_input, self.weight, self.bias)
        if self.lora_a is not None and self.lora_b is not None and self.lora_scale != 0.0:
            low_rank = torch_functional.linear(inputs, self.lora_a.to(inputs.dtype))
            delta = torch_functional.linear(low_rank, self.lora_b.to(inputs.dtype))
            output = output + delta * (self.alpha_over_rank * self.lora_scale)
        return output


def install_zoo_linears(model: nn.Module) -> dict[str, ZooLinear]:
    """Replace every attention / MLP projection by a `ZooLinear`; return them by dotted name."""
    installed: dict[str, ZooLinear] = {}
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in LINEAR_TARGETS:
                wrapper = ZooLinear(child)
                setattr(parent, child_name, wrapper)
                installed[f"{parent_name}.{child_name}" if parent_name else child_name] = wrapper
    if not installed:
        raise ZooConfigError("no attention / MLP projection found: unsupported architecture")
    return installed


def attach_all_lora(linears: dict[str, ZooLinear], rank: int = LORA_RANK) -> list[nn.Parameter]:
    """Attach an adapter to every wrapped layer; return the trainable parameters."""
    parameters: list[nn.Parameter] = []
    for layer in linears.values():
        layer.attach_lora(rank)
        assert layer.lora_a is not None and layer.lora_b is not None
        parameters.extend([layer.lora_a, layer.lora_b])
    return parameters


def set_lora_scale(linears: dict[str, ZooLinear], scale: float) -> None:
    """Set the adapter strength of every layer (0.0 switches the adapter off)."""
    for layer in linears.values():
        layer.lora_scale = scale


def set_activation_fp8(linears: dict[str, ZooLinear], enabled: bool) -> None:
    """Switch the W8A8 activation quantisation of every layer."""
    for layer in linears.values():
        layer.activation_fp8 = enabled


def save_adapter(path: Path, linears: dict[str, ZooLinear]) -> None:
    """Write the adapter as one safetensors file (atomic: temp file then rename)."""
    tensors: dict[str, torch.Tensor] = {}
    for name, layer in linears.items():
        if layer.lora_a is None or layer.lora_b is None:
            raise AdapterError(f"layer {name} has no adapter to save")
        tensors[name + LORA_A_SUFFIX] = layer.lora_a.detach().cpu().contiguous()
        tensors[name + LORA_B_SUFFIX] = layer.lora_b.detach().cpu().contiguous()
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"rank": str(LORA_RANK), "alpha": str(LORA_ALPHA)})
    temporary.replace(path)


def load_adapter(path: Path, linears: dict[str, ZooLinear]) -> None:
    """Load an adapter file into the wrapped layers (attaching the factors when absent).

    Raises:
        AdapterError: a layer is missing from the file or a shape differs.
    """
    tensors = load_file(str(path))
    for name, layer in linears.items():
        key_a, key_b = name + LORA_A_SUFFIX, name + LORA_B_SUFFIX
        if key_a not in tensors or key_b not in tensors:
            raise AdapterError(f"adapter {path} has no weights for layer {name}")
        if not layer.has_lora:
            layer.attach_lora(int(tensors[key_a].shape[0]))
        assert layer.lora_a is not None and layer.lora_b is not None
        if tensors[key_a].shape != layer.lora_a.shape or tensors[key_b].shape != layer.lora_b.shape:
            raise AdapterError(f"adapter {path}: shape mismatch on {name}")
        with torch.no_grad():
            layer.lora_a.copy_(tensors[key_a])
            layer.lora_b.copy_(tensors[key_b])


class Fp8Weights:
    """Per-tensor E4M3 weights with an exact way back.

    Entering quantises every wrapped weight in place after saving a CPU copy of the original;
    `restore` copies the originals back. The CPU copy costs one model size of host memory
    (about 15 GB), which the nodes have; GPU memory is untouched.
    """

    def __init__(self, linears: dict[str, ZooLinear]) -> None:
        """Remember the layers; nothing is quantised until `quantize`."""
        self._linears = linears
        self._backup: dict[str, torch.Tensor] = {}

    def quantize(self) -> float:
        """Quantise every weight in place; return the mean relative error over the layers."""
        errors: list[float] = []
        for name, layer in self._linears.items():
            if name not in self._backup:
                self._backup[name] = layer.weight.detach().to("cpu", copy=True)
            errors.append(quantize_weight_in_place(layer.weight))
        return sum(errors) / len(errors)

    def restore(self) -> None:
        """Put the original weights back and drop the CPU copy."""
        with torch.no_grad():
            for name, layer in self._linears.items():
                if name in self._backup:
                    layer.weight.copy_(self._backup[name].to(layer.weight.device))
        self._backup.clear()


@dataclass(frozen=True)
class TrainingExample:
    """One tokenised instruction example: ids and labels (prompt positions are ignored)."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def render_instruction(row: dict[str, str]) -> tuple[str, str]:
    """`(user message, answer)` of an Alpaca-style row (`instruction`, optional `input`, `output`)
    or a Dolly-style row (`instruction`, optional `context`, `response`).

    Raises:
        TrainingDataError: the row has no instruction or no output.
    """
    instruction = row.get("instruction", "").strip()
    output = (row.get("output") or row.get("response") or "").strip()
    if not instruction or not output:
        raise TrainingDataError(f"row without instruction or output: {str(row)[:80]!r}")
    extra = (row.get("input") or row.get("context") or "").strip()
    return (f"{instruction}\n\n{extra}" if extra else instruction), output


def tokenize_example(
    tokenizer: Any, user_text: str, answer: str, max_tokens: int = LORA_MAX_TOKENS
) -> TrainingExample | None:
    """Tokenise with the family chat template; labels cover the assistant turn only.

    The assistant segment is the part of the fully rendered conversation after the generation
    prompt, so it includes the family's end-of-turn token (the model learns to stop). Returns
    `None` when the example exceeds `max_tokens`.

    Raises:
        TrainingDataError: the template does not render the generation prompt as a prefix.
    """
    messages = [{"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": answer}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise TrainingDataError("chat template: generation prompt is not a prefix of the dialogue")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_tokens or not answer_ids:
        return None
    return TrainingExample(
        tuple(prompt_ids) + tuple(answer_ids),
        (IGNORE_INDEX,) * len(prompt_ids) + tuple(answer_ids),
    )


def load_training_examples(
    tokenizer: Any, dataset_path: Path, count: int, seed: int, max_tokens: int = LORA_MAX_TOKENS,
) -> list[TrainingExample]:
    """Read an instruction set (JSON list, or `.jsonl`), shuffle with `seed`, keep `count` examples.

    Raises:
        TrainingDataError: unreadable file or fewer than `count` usable examples.
    """
    try:
        text = dataset_path.read_text(encoding="utf-8")
        rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
                if dataset_path.suffix == ".jsonl" else json.loads(text))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingDataError(f"cannot read {dataset_path}: {error}") from error
    if not isinstance(rows, list):
        raise TrainingDataError(f"{dataset_path} is not a JSON list")
    random.Random(seed).shuffle(rows)
    examples: list[TrainingExample] = []
    for row in rows:
        try:
            user_text, answer = render_instruction(row)
        except TrainingDataError:
            continue
        example = tokenize_example(tokenizer, user_text, answer, max_tokens)
        if example is not None:
            examples.append(example)
            if len(examples) >= count:
                return examples
    raise TrainingDataError(f"only {len(examples)} usable examples, need {count}")


@dataclass
class TrainReport:
    """What the LoRA training did (written next to the adapter)."""

    steps: int
    examples_per_step: int
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    final_micro_batch: int = 0
    oom_events: int = 0
    peak_memory_gib: float = 0.0
    parameters: int = 0

    def to_json(self) -> dict[str, object]:
        """JSON-friendly summary (the loss curve is kept: 120 floats)."""
        return {
            "steps": self.steps, "examples_per_step": self.examples_per_step,
            "loss_first_10_mean": sum(self.losses[:10]) / max(len(self.losses[:10]), 1),
            "loss_last_10_mean": sum(self.losses[-10:]) / max(len(self.losses[-10:]), 1),
            "losses": self.losses, "seconds": self.seconds,
            "final_micro_batch": self.final_micro_batch, "oom_events": self.oom_events,
            "peak_memory_gib": self.peak_memory_gib, "parameters": self.parameters,
        }


def learning_rate_at(step: int, steps: int, peak: float = LORA_LEARNING_RATE,
                     warmup: int = LORA_WARMUP_STEPS) -> float:
    """Linear warm-up to `peak`, then cosine decay to 10 % of it."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _collate(
    batch: Sequence[TrainingExample], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-padded `(input_ids, attention_mask, labels)` of a micro-batch."""
    longest = max(len(example.input_ids) for example in batch)
    ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for row, example in enumerate(batch):
        length = len(example.input_ids)
        ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
        mask[row, :length] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def _accumulate_gradients(
    model: nn.Module, batch: Sequence[TrainingExample], micro_batch: int, pad_id: int,
    device: torch.device,
) -> float:
    """Forward and backward over `batch` in micro-batches; return the token-mean loss.

    Each micro-batch loss is weighted by its share of target tokens, so the gradient does not
    depend on the micro-batch size (which the out-of-memory fallback changes).
    """
    total_targets = sum(sum(1 for label in e.labels[1:] if label != IGNORE_INDEX) for e in batch)
    total_loss = 0.0
    for start in range(0, len(batch), micro_batch):
        chunk = batch[start:start + micro_batch]
        ids, mask, labels = _collate(chunk, pad_id, device)
        targets = int((labels[:, 1:] != IGNORE_INDEX).sum())
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        weighted = output.loss * (targets / total_targets)
        weighted.backward()
        total_loss += float(weighted.detach())
    return total_loss


def _set_checkpointing(model: Any, enabled: bool) -> None:
    """Switch activation checkpointing (new `activation_*` API first, older names as fallback)."""
    if hasattr(model, "activation_checkpointing_enable"):
        if enabled:
            model.activation_checkpointing_enable()
        else:
            model.activation_checkpointing_disable()
    elif enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()


def train_lora(
    model: Any, linears: dict[str, ZooLinear], examples: Sequence[TrainingExample],
    pad_id: int, steps: int, effective_batch: int, micro_batch: int, device: torch.device,
    log: Any = print, learning_rate: float = LORA_LEARNING_RATE,
) -> TrainReport:
    """Train the adapters on `examples` (frozen base, AdamW, cosine schedule).

    Args:
        model: the causal LM whose projections were wrapped by `install_zoo_linears`.
        linears: the wrapped layers (the adapters are attached here if absent).
        examples: at least `steps * effective_batch` examples are consumed, cycling if fewer.
        pad_id: padding token id.
        steps: optimiser steps.
        effective_batch: examples per optimiser step.
        micro_batch: starting micro-batch; halved on CUDA out-of-memory (never below 4).
        device: where the model lives.
        log: progress sink (`print` by default).
        learning_rate: peak learning rate (the schedule warms up to it, then decays).

    Raises:
        TrainingDataError: no examples.
        torch.cuda.OutOfMemoryError: out of memory even at the smallest micro-batch.
    """
    if not examples:
        raise TrainingDataError("no training examples")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not any(layer.has_lora for layer in linears.values()):
        trainable = attach_all_lora(linears)
    else:
        trainable = [p for layer in linears.values() for p in (layer.lora_a, layer.lora_b)
                     if p is not None]
    for parameter in trainable:
        parameter.requires_grad_(True)
    set_lora_scale(linears, 1.0)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    report = TrainReport(steps, effective_batch, final_micro_batch=micro_batch,
                         parameters=sum(p.numel() for p in trainable))
    model.config.use_cache = False
    _set_checkpointing(model, True)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    current_micro = micro_batch
    for step in range(steps):
        batch = [examples[(step * effective_batch + i) % len(examples)]
                 for i in range(effective_batch)]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, steps, learning_rate)
        while True:
            try:
                loss = _accumulate_gradients(model, batch, current_micro, pad_id, device)
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if current_micro <= MIN_BATCH_SIZE:
                    raise
                current_micro = max(MIN_BATCH_SIZE, current_micro // 2)
                report.oom_events += 1
                log(f"lora step {step}: out of memory, micro-batch -> {current_micro}")
        torch.nn.utils.clip_grad_norm_(trainable, LORA_GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report.losses.append(loss)
        if step % 10 == 0 or step == steps - 1:
            log(f"lora step {step + 1}/{steps} loss {loss:.4f} micro {current_micro} "
                f"elapsed {time.time() - started:.0f}s")
    report.seconds = time.time() - started
    report.final_micro_batch = current_micro
    if device.type == "cuda":
        report.peak_memory_gib = torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB
    model.eval()
    _set_checkpointing(model, False)
    model.config.use_cache = True
    for parameter in trainable:
        parameter.requires_grad_(False)
    set_lora_scale(linears, 0.0)
    return report
