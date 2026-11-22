"""Simulated endpoints: how big is the fp8 effect on the three bases, and how many tokens does it take?

The stored `ep` answers (7 per endpoint, about 560 tokens) cannot say how detectable fp8 is on the
Qwen2.5-1.5B, Qwen2.5-3B and Llama-3.2-1B bases: the only ground truth for fp8 is on the ref
endpoints, which run SmolLM2-360M. So we SIMULATE both hypotheses with local copies:

* H0: a plain endpoint = text sampled from the base at the default temperature (0.7);
* H1: an fp8 endpoint = text sampled from a simulated fp8 copy (same temperature).

Every sampled text is then scored under the base and under each simulated fp8 variant. For the
variant V the per-token log-likelihood ratio `LLR = log p_V - log p_base` has a mean `mu0` under H0
(negative: the text is typical of the base) and `mu1` under H1 (positive), with a per-row spread.
The gap `mu1 - mu0` and the spread give the number of tokens a likelihood-ratio test needs. The same
protocol on SmolLM2-360M can be compared with the real ref-02 versus ref-01 numbers: that is the
validation of the simulation.

Output: one container per candidate, `sim_<candidate>.npz`, holding for every generation condition
the sampled texts, their token counts and the per-row log-likelihood sums under every scoring
variant. No endpoint is contacted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_loglik import (
    DTYPE_FP32,
    EP_FILE,
    FP8_VARIANTS_BF16,
    TEMPLATE_DEFAULT,
    VARIANT_BASE,
    Fp8Simulator,
    release_memory,
    render_context,
    score_pairs,
    tokenise_rows,
)
from hackathon.model_equality.research.e_matrix import row_sums
from hackathon.model_equality.research.e_models import CandidateSpec, get_candidate
from hackathon.model_equality.store import Answer, load_many

FloatArray = npt.NDArray[np.float64]
SIM_TEMPERATURE = 0.7
"""Default sampling temperature of the endpoints (fitted at 0.72 on the plain ref endpoint)."""
SIM_MAX_NEW_TOKENS = 256
"""The server caps replies at 256 tokens."""
SIM_BATCH_SEQUENCES = 48
SIM_SEED = 20261121
SIM_PROMPTS = 120
SIM_REPEATS = 2
SIM_VARIANTS: tuple[str, ...] = FP8_VARIANTS_BF16
"""fp8 recipes simulated: per-tensor, per-channel and 128x128 block weights; per-tensor weights with
per-token activations."""
SEP = "__"
ENDPOINT_NAME = "sim"


class SimError(HackathonError):
    """The simulation cannot run (no prompts, model files missing)."""


def collect_prompts(answers_dir: Path, n_prompts: int, seed: int) -> list[str]:
    """Distinct prompt texts of the stored ep and ref answers, the 7 ep prompts always included.

    Raises:
        SimError: No stored prompt.
    """
    stores = sorted(answers_dir.glob("reference_answers*.jsonl")) + [answers_dir / EP_FILE]
    stores = [path for path in stores if path.is_file() and "raw" not in path.name]
    answers = load_many(stores)
    if not answers:
        raise SimError(f"no stored answers in {answers_dir}")
    ep_prompts = sorted({a.prompt for a in answers if a.endpoint.startswith("ep-")})
    other = sorted({a.prompt for a in answers} - set(ep_prompts))
    rng = np.random.default_rng(seed)
    take = max(n_prompts - len(ep_prompts), 0)
    picked = [other[i] for i in sorted(rng.permutation(len(other))[:take])] if other else []
    return [*ep_prompts, *picked]


def eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """Every id that ends a reply: the model's generation config and the tokenizer's EOS."""
    ids: set[int] = set()
    configured = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif configured is not None:
        ids.update(int(i) for i in configured)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return sorted(ids)


def cut_at_eos(token_ids: Sequence[int], eos_ids: Sequence[int]) -> list[int]:
    """Tokens before the first end-of-reply id (all of them when there is none)."""
    for position, token in enumerate(token_ids):
        if token in eos_ids:
            return list(token_ids[:position])
    return list(token_ids)


@torch.no_grad()
def generate_texts(
    model: Any, tokenizer: Any, spec: CandidateSpec, prompts: Sequence[str], *, temperature: float,
    max_new_tokens: int, batch_sequences: int, seed: int, device: torch.device,
) -> list[str]:
    """Sample one reply per prompt at `temperature` (no top-k, no top-p), decoded to text."""
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = eos_token_ids(model, tokenizer)
    texts: list[str] = []
    contexts = [render_context(tokenizer, spec.template, TEMPLATE_DEFAULT, p) for p in prompts]
    for start in range(0, len(contexts), batch_sequences):
        batch = contexts[start : start + batch_sequences]
        encoded = tokenizer(batch, add_special_tokens=False, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded, do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
            repetition_penalty=1.0, max_new_tokens=max_new_tokens, eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in generated[:, encoded["input_ids"].shape[1] :].tolist():
            texts.append(str(tokenizer.decode(cut_at_eos(row, eos_ids), skip_special_tokens=True)))
    return texts


def score_texts(
    model: Any, tokenizer: Any, spec: CandidateSpec, prompts: Sequence[str], texts: Sequence[str],
    device: torch.device,
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Per-row log-likelihood sums and token counts of sampled `texts` under the current weights."""
    answers = [
        Answer(ENDPOINT_NAME, f"s{index}", 0, prompt, text)
        for index, (prompt, text) in enumerate(zip(prompts, texts, strict=True))
    ]
    pairs = tokenise_rows(tokenizer, spec.template, TEMPLATE_DEFAULT, answers)
    scores = score_pairs(model, pairs, device, max_batch_tokens=4096)
    return row_sums(scores, "logp"), scores.n_tokens.astype(np.int64)


def run_sim(
    candidate: str, *, cache_dir: Path, answers_dir: Path, out_dir: Path, device: torch.device,
    n_prompts: int = SIM_PROMPTS, repeats: int = SIM_REPEATS, variants: Sequence[str] = SIM_VARIANTS,
    force: bool = False, max_new_tokens: int = SIM_MAX_NEW_TOKENS,
) -> Path | None:
    """Sample under base and every fp8 variant, score every sample under base and that variant.

    Returns the container path, or None when it already exists.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy, imported late

    spec = get_candidate(candidate)
    output = out_dir / f"sim_{spec.key}.npz"
    if output.is_file() and not force:
        print(f"skip {output.name}: exists", flush=True)
        return None
    model_dir = cache_dir / spec.key
    if not (model_dir / "config.json").is_file():
        raise SimError(f"{model_dir}: model files missing")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float32
    )
    model = model.to(device).eval()
    prompts_once = collect_prompts(answers_dir, n_prompts, SIM_SEED)
    prompts = [p for _ in range(repeats) for p in prompts_once]
    print(f"{spec.key}/sim: loaded in {time.time() - started:.0f}s, {len(prompts)} prompts x 1 draw",
          flush=True)
    simulator = Fp8Simulator(model)
    arrays: dict[str, Any] = {"prompt_index": np.arange(len(prompts)) % len(prompts_once)}
    conditions = (VARIANT_BASE, *variants)
    for condition_index, generation in enumerate(conditions):
        simulator.apply(generation)
        began = time.time()
        texts = generate_texts(
            model, tokenizer, spec, prompts, temperature=SIM_TEMPERATURE,
            max_new_tokens=max_new_tokens, batch_sequences=SIM_BATCH_SEQUENCES,
            seed=SIM_SEED + condition_index, device=device,
        )
        arrays[f"texts{SEP}{generation}"] = np.array(texts)
        print(f"{spec.key}/sim: generated {generation} in {time.time() - began:.0f}s", flush=True)
        # Base samples (H0) are scored under the base and under every variant; the samples of a
        # variant (H1) under the base and under that same variant.
        targets = conditions if generation == VARIANT_BASE else (VARIANT_BASE, generation)
        for scoring in targets:
            simulator.apply(scoring)
            sums, tokens = score_texts(model, tokenizer, spec, prompts, texts, device)
            arrays[f"logp{SEP}{generation}{SEP}{scoring}"] = sums
            arrays[f"tokens{SEP}{generation}"] = tokens
    meta = {
        "candidate": spec.key, "temperature": SIM_TEMPERATURE, "max_new_tokens": max_new_tokens,
        "variants": list(variants), "dtype": DTYPE_FP32, "torch": torch.__version__,
        "n_rows": len(prompts), "seconds": round(time.time() - started), "seed": SIM_SEED,
    }
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, output)
    simulator.restore()
    release_memory(device)
    print(f"wrote {output.name} in {time.time() - started:.0f}s", flush=True)
    return output


def load_sim(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], FloatArray], dict[str, npt.NDArray[np.int64]]]:
    """Read a container: `(meta, {(generation, scoring): row logp sums}, {generation: tokens})`."""
    sums: dict[tuple[str, str], FloatArray] = {}
    tokens: dict[str, npt.NDArray[np.int64]] = {}
    with np.load(path, allow_pickle=False) as data:
        meta: dict[str, Any] = json.loads(str(data["meta"]))
        for name in data.files:
            parts = name.split(SEP)
            if parts[0] == "logp" and len(parts) == 3:
                sums[(parts[1], parts[2])] = data[name].astype(np.float64)
            elif parts[0] == "tokens" and len(parts) == 2:
                tokens[parts[1]] = data[name].astype(np.int64)
    return meta, sums, tokens


def run_command(args: argparse.Namespace) -> int:
    """Run the simulation for the candidates given, one after the other on this GPU."""
    device = torch.device(args.device)
    failures = 0
    for candidate in args.candidates:
        try:
            run_sim(candidate, cache_dir=Path(args.cache_dir), answers_dir=Path(args.answers_dir),
                    out_dir=Path(args.out_dir), device=device, n_prompts=args.prompts,
                    repeats=args.repeats, force=args.force)
        except Exception as exc:  # noqa: BLE001  one failed candidate must not stop the others
            failures += 1
            print(f"FAILED {candidate}/sim: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            release_memory(device)
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--answers-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts", type=int, default=SIM_PROMPTS)
    parser.add_argument("--repeats", type=int, default=SIM_REPEATS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    return run_command(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

