"""Re-read every stored answer under explicit (model, template, quantisation) hypotheses (x-met-relect).

For each answer row and each hypothesis we record PER-ROW sums, at every temperature of a 0.3-1.5
grid, of: log p(token) under softmax(logits / T), the entropy H_T, the variance of log p, the
probability of the top-1 token, plus the count of tokens equal to the argmax (temperature-free).

* The log-likelihood ratio of two hypotheses, each at its own fitted temperature, is the Wald
  statistic of the plan (a decoding temperature never counts as evidence).
* `sum(logp_T + H_T)` is the analytic version of "answers drawn locally from the hypothesis": its
  expectation is 0 when the hypothesis (at T) wrote the text, so no sampling is needed.
* `hits - sum(pmax_T)` is a second moment that the fitted temperature does not absorb (a
  temperature moves both, a LoRA or other weights move them inconsistently).

Hypotheses: `base` (fp32 with TF32 matmuls), `w_tensor` / `w_channel` (fp8 E4M3 weights, one scale
per tensor / per output row), `w8a8_token` (fp8 weights plus a dynamic per-token fp8 activation
scale), `int8_channel` (symmetric int8 weights, one scale per output row) and `bf16` (the same base
weights computed in bfloat16). Embeddings and `lm_head` are never quantised (vLLM behaviour).

Runs on a JURECA compute node, offline; reads local files only and never contacts an endpoint.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from hackathon.model_equality.research.e_loglik import (
    Fp8Simulator,
    make_batches,
    tokenise_rows,
    write_container,
)
from hackathon.model_equality.research.e_models import get_candidate
from hackathon.model_equality.store import Answer, load_many

TEMPERATURES: tuple[float, ...] = tuple(round(0.3 + 0.1 * step, 2) for step in range(13))
"""0.3 to 1.5 by 0.1 (1.0 included), the grid of the plan."""
STATS_PER_TEMPERATURE = 4
"""logp, entropy, variance of log p, probability of the top-1 token."""
VARIANT_BASE = "base"
VARIANT_INT8_CHANNEL = "int8_channel"
VARIANT_BF16 = "bf16"
FP8_VARIANTS: tuple[str, ...] = ("w_tensor", "w_channel", "w8a8_token")
VARIANTS: tuple[str, ...] = (VARIANT_BASE, *FP8_VARIANTS, VARIANT_INT8_CHANNEL, VARIANT_BF16)
INT8_MAX = 127.0
MIN_SCALE = 1e-12
LOGIT_CHUNK = 128
BATCH_TOKENS = 8192
B64_CHUNK = 60000
REF_CANDIDATE = "smollm2-360m"
REF_TEMPLATE = "default"


class RelectError(RuntimeError):
    """The re-reading cannot proceed (missing model, empty row set)."""


def quantise_int8_channel(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 round trip of a 2-D weight with one scale per output row."""
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(MIN_SCALE) / INT8_MAX
    return (weight / scale).round().clamp(-INT8_MAX, INT8_MAX) * scale


def apply_variant(simulator: Fp8Simulator, variant: str) -> None:
    """Put the model in `variant` (bf16 is handled by the caller, it changes the dtype)."""
    if variant in (VARIANT_BASE, VARIANT_BF16):
        simulator.apply(VARIANT_BASE)
        return
    if variant != VARIANT_INT8_CHANNEL:
        simulator.apply(variant)
        return
    simulator.apply(VARIANT_BASE)
    simulator._remember()
    for name, layer in simulator.targets.items():
        weight = simulator._pristine[name].to(layer.weight.device, torch.float32)
        layer.weight.data.copy_(quantise_int8_channel(weight).to(layer.weight.dtype))


@torch.no_grad()
def score_rows(
    model: Any, pairs: Sequence[tuple[list[int], list[int]]], device: torch.device
) -> npt.NDArray[np.float32]:
    """Per-row sums, shape `(rows, 1 + 4 * len(TEMPERATURES))`: hits, then per T logp, H, var, pmax."""
    width = 1 + STATS_PER_TEMPERATURE * len(TEMPERATURES)
    out = np.zeros((len(pairs), width), dtype=np.float64)
    temperatures = torch.tensor(TEMPERATURES, device=device, dtype=torch.float32)[:, None, None]
    lengths = [len(c) + len(r) for c, r in pairs]
    for batch in make_batches(lengths, BATCH_TOKENS, single_sequence=False):
        rows = [pairs[i] for i in batch]
        longest = max(len(c) + len(r) for c, r in rows)
        ids = torch.zeros((len(rows), longest), dtype=torch.long)
        mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for k, (context, reply) in enumerate(rows):
            sequence = context + reply
            ids[k, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[k, : len(sequence)] = 1
        ids, mask = ids.to(device), mask.to(device)
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        for k, (context, reply) in enumerate(rows):
            if not reply:
                continue
            start = len(context) - 1
            selected = hidden[k, start : start + len(reply)]
            targets = ids[k, len(context) : len(context) + len(reply)]
            totals = torch.zeros(width, device=device, dtype=torch.float64)
            for chunk_start in range(0, len(reply), LOGIT_CHUNK):
                part = slice(chunk_start, chunk_start + LOGIT_CHUNK)
                logits = model.lm_head(selected[part]).float()
                target = targets[part]
                totals[0] += (logits.argmax(dim=-1) == target).sum().double()
                scaled = logits[None, :, :] / temperatures
                log_probs = torch.log_softmax(scaled, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                second = (probs * log_probs.square()).sum(dim=-1)
                variance = (second - entropy.square()).clamp_min(0.0)
                chosen = log_probs.gather(2, target[None, :, None].expand(len(TEMPERATURES), -1, 1))[..., 0]
                pmax = probs.amax(dim=-1)
                stacked = torch.stack([chosen, entropy, variance, pmax], dim=1).sum(dim=-1)
                totals[1:] += stacked.reshape(-1).double()
            out[batch[k]] = totals.cpu().numpy()
    return out.astype(np.float32)


PAIRED_STATS = 6
"""Per row: observed LLR, its mean and variance under the base, under fp8, and the token count."""


@torch.no_grad()
def paired_rows(
    model_base: Any, model_fp8: Any, pairs: Sequence[tuple[list[int], list[int]]],
    temperatures: Sequence[float], device: torch.device,
) -> npt.NDArray[np.float32]:
    """Exact moments of the fp8-vs-base LLR along the observed prefixes, each row at its own T.

    Per row: `llr = sum_t log p_f(x_t) - log p_b(x_t)`, `E_b[llr]`, `Var_b[llr]`, `E_f[llr]`,
    `Var_f[llr]` (sums over positions of the per-position moments of `log p_f - log p_b` under each
    tempered distribution) and the token count. Under "the base wrote it" the observed llr is close
    to `N(E_b, Var_b)`, under "fp8 wrote it" to `N(E_f, Var_f)`: the analytic version of answers
    drawn locally from each hypothesis, free of the temperature and length biases.
    """
    out = np.zeros((len(pairs), PAIRED_STATS), dtype=np.float64)
    lengths = [len(c) + len(r) for c, r in pairs]
    for batch in make_batches(lengths, BATCH_TOKENS, single_sequence=False):
        rows = [pairs[i] for i in batch]
        longest = max(len(c) + len(r) for c, r in rows)
        ids = torch.zeros((len(rows), longest), dtype=torch.long)
        mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for k, (context, reply) in enumerate(rows):
            sequence = context + reply
            ids[k, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[k, : len(sequence)] = 1
        ids, mask = ids.to(device), mask.to(device)
        hidden_b = model_base.model(input_ids=ids, attention_mask=mask).last_hidden_state
        hidden_f = model_fp8.model(input_ids=ids, attention_mask=mask).last_hidden_state
        for k, (context, reply) in enumerate(rows):
            if not reply:
                continue
            temperature = float(temperatures[batch[k]])
            start = len(context) - 1
            targets = ids[k, len(context) : len(context) + len(reply)]
            totals = torch.zeros(PAIRED_STATS, device=device, dtype=torch.float64)
            for chunk_start in range(0, len(reply), LOGIT_CHUNK):
                part = slice(start + chunk_start, start + min(chunk_start + LOGIT_CHUNK, len(reply)))
                target = targets[chunk_start : chunk_start + LOGIT_CHUNK]
                lb = torch.log_softmax(model_base.lm_head(hidden_b[k, part]).float() / temperature, dim=-1)
                lf = torch.log_softmax(model_fp8.lm_head(hidden_f[k, part]).float() / temperature, dim=-1)
                delta = lf - lb
                pb, pf = lb.exp(), lf.exp()
                mean_b = (pb * delta).sum(-1)
                mean_f = (pf * delta).sum(-1)
                var_b = ((pb * delta.square()).sum(-1) - mean_b.square()).clamp_min(0.0)
                var_f = ((pf * delta.square()).sum(-1) - mean_f.square()).clamp_min(0.0)
                observed = delta.gather(1, target[:, None])[:, 0]
                totals[0] += observed.sum().double()
                totals[1] += mean_b.sum().double()
                totals[2] += var_b.sum().double()
                totals[3] += mean_f.sum().double()
                totals[4] += var_f.sum().double()
                totals[5] += float(target.numel())
            out[batch[k]] = totals.cpu().numpy()
    return out.astype(np.float32)


def run_paired(
    key: str, groups: dict[str, list[Answer]], temperature_of: dict[str, float], cache_dir: Path,
    device: torch.device, fp8_variant: str,
) -> dict[str, Any]:
    """Base and fp8 copies of one candidate side by side; paired moments of every group."""
    import copy

    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_candidate(key)
    model_dir = cache_dir / spec.key
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    base: Any = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.float32)
    base = base.to(device).eval()
    fp8 = copy.deepcopy(base)
    Fp8Simulator(fp8).apply(fp8_variant)
    arrays: dict[str, Any] = {}
    for template, rows in groups.items():
        pairs = tokenise_rows(tokenizer, spec.template, template, rows)
        tag = f"{key}__{template}"
        temps = [temperature_of.get(a.endpoint, DEFAULT_PAIRED_TEMPERATURE) for a in rows]
        began = time.time()
        arrays[f"{tag}__paired_{fp8_variant}"] = paired_rows(base, fp8, pairs, temps, device)
        arrays[tag + "__paired_endpoint"] = np.array([a.endpoint for a in rows])
        arrays[tag + "__paired_prompt_id"] = np.array([a.prompt_id for a in rows])
        print(f"{tag}/paired_{fp8_variant}: {len(rows)} rows {time.time() - began:.0f}s", flush=True)
    del base, fp8
    torch.cuda.empty_cache()
    return arrays


DEFAULT_PAIRED_TEMPERATURE = 0.72


def load_rows(paths: Sequence[Path], keep: set[str] | None) -> list[Answer]:
    """De-duplicated answers of `paths`, optionally limited to the endpoints in `keep`."""
    loaded = load_many([p for p in paths if p.is_file()])
    rows = [a for a in loaded if keep is None or a.endpoint in keep]
    return sorted(rows, key=lambda a: (a.endpoint, a.prompt_id, a.repeat))


def run_candidate(
    key: str, groups: dict[str, list[Answer]], cache_dir: Path, device: torch.device, deadline: float,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, Any]:
    """Score every `template -> rows` group of one candidate under every variant."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_candidate(key)
    model_dir = cache_dir / spec.key
    if not (model_dir / "config.json").is_file():
        raise RelectError(f"{model_dir}: model files missing")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.float32)
    model = model.to(device).eval()
    simulator = Fp8Simulator(model)
    arrays: dict[str, Any] = {}
    for template, rows in groups.items():
        pairs = tokenise_rows(tokenizer, spec.template, template, rows)
        tag = f"{key}__{template}"
        arrays[tag + "__endpoint"] = np.array([a.endpoint for a in rows])
        arrays[tag + "__prompt_id"] = np.array([a.prompt_id for a in rows])
        arrays[tag + "__repeat"] = np.array([a.repeat for a in rows], dtype=np.int32)
        arrays[tag + "__n_tokens"] = np.array([len(r) for _, r in pairs], dtype=np.int32)
        for variant in variants:
            if time.time() > deadline:
                print(f"deadline: skip {tag}/{variant}", flush=True)
                continue
            began = time.time()
            if variant == VARIANT_BF16:
                simulator.apply(VARIANT_BASE)
                model = model.to(torch.bfloat16)
                arrays[f"{tag}__{variant}"] = score_rows(model, pairs, device)
                model = model.to(torch.float32)
            else:
                apply_variant(simulator, variant)
                arrays[f"{tag}__{variant}"] = score_rows(model, pairs, device)
            print(f"{tag}/{variant}: {len(rows)} rows {time.time() - began:.0f}s", flush=True)
    simulator.restore()
    del model, simulator
    torch.cuda.empty_cache()
    return arrays


def dump_fallback(arrays: dict[str, Any]) -> None:
    """Print the container as zlib + base64 lines in the task log (no new inode needed)."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    blob = base64.b64encode(zlib.compress(buffer.getvalue(), 6)).decode("ascii")
    for index in range(0, len(blob), B64_CHUNK):
        print(f"RELECT_B64 {index // B64_CHUNK} {blob[index : index + B64_CHUNK]}", flush=True)
    print("RELECT_B64_END", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: `--plan` JSON maps candidate -> {template: [endpoints]}."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ep-files", type=Path, nargs="+", required=True)
    parser.add_argument("--ref-files", type=Path, nargs="*", default=[])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--with-ref", action="store_true")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--paired-variant", default="")
    args = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    plan_all: dict[str, Any] = json.loads(args.plan.read_text(encoding="utf-8"))
    temperature_of: dict[str, float] = {k: float(v) for k, v in plan_all.pop("__temps__", {}).items()}
    plan: dict[str, dict[str, list[str]]] = plan_all
    ep_rows = load_rows(args.ep_files, None)
    print(f"ep rows {len(ep_rows)}", flush=True)
    arrays: dict[str, Any] = {"meta": np.array(json.dumps({
        "temperatures": TEMPERATURES, "variants": VARIANTS, "torch": torch.__version__,
        "plan": plan, "ep_rows": len(ep_rows),
    }))}
    order = list(plan)
    if args.with_ref:
        order = [REF_CANDIDATE, *order]
    for key in order:
        if time.time() > args.deadline_epoch:
            print(f"deadline: skip candidate {key}", flush=True)
            continue
        if key == REF_CANDIDATE and args.with_ref:
            refs = load_rows(args.ref_files, None)
            groups = {REF_TEMPLATE: refs}
        else:
            groups = {}
            for template, endpoints in plan[key].items():
                wanted = set(endpoints)
                groups[template] = [a for a in ep_rows if a.endpoint in wanted]
        if args.paired_variant:
            try:
                arrays.update(run_paired(key, groups, temperature_of, args.cache_dir, device, args.paired_variant))
            except (RelectError, RuntimeError) as exc:
                print(f"paired {key} failed: {exc}", flush=True)
        if args.variants:
            try:
                arrays.update(run_candidate(key, groups, args.cache_dir, device, args.deadline_epoch, args.variants))
            except (RelectError, RuntimeError) as exc:
                print(f"candidate {key} failed: {exc}", flush=True)
        try:
            write_container(args.out, arrays, attempts=6, delay=10.0)
            print(f"wrote {args.out} after {key}", flush=True)
        except Exception as exc:  # noqa: BLE001 - quota: keep going, the log carries the data
            print(f"write failed after {key}: {exc}", flush=True)
    try:
        write_container(args.out, arrays, attempts=6, delay=10.0)
    except Exception as exc:  # noqa: BLE001
        print(f"final write failed: {exc}", flush=True)
    dump_fallback(arrays)
    return 0


if __name__ == "__main__":
    sys.exit(main())
