"""Q9 / Q10 question choice for the undecided endpoints (x-met-relect), computed locally.

For each base model and a stock of candidate prompts (fixed long-answer prompts, Q9 "Repeat exactly"
prompts built from the base's under-trained tokens, and the Q1-Q8 tail prompts), we draw samples
from the base at the endpoints' typical temperature and measure, position by position, the Jeffreys
divergence between the tempered base and the tempered fp8 per-tensor model (the recipe that matches
the known fp8 reference, ref-02). The expected LLR separation of one answer is the sum of those
divergences along the answer, so the ranking is "expected gap x length", as the plan asks.

Under-trained tokens ("Fishing for Magikarp", Land and Bartolo 2024), simplified: tokens whose
input-embedding row is closest to the mean embedding (low centred norm), that decode to printable
text and re-encode to themselves. They are rarely seen in training, so small weight perturbations
move their logits most.

Runs on a JURECA compute node, offline; nothing contacts an endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from hackathon.model_equality.research.e_loglik import Fp8Simulator, render_context
from hackathon.model_equality.research.e_models import get_candidate

SAMPLE_TEMPERATURE = 0.72
"""Fitted temperature of most endpoints (ref-01 0.723; ep median ~0.72)."""
MAX_NEW_TOKENS = 256
SAMPLES_PER_PROMPT = 3
MAGIKARP_POOL = 400
MAGIKARP_PER_PROMPT = 12
MAGIKARP_PROMPTS = 6
SEED = 1234
FP8_VARIANT = "w_tensor"
BLOCKED_SUBSTRINGS: tuple[str, ...] = ("sex", "porn", "fuck", "nazi", "kill")
"""Under-trained tokens sent to the organisers' endpoints stay neutral."""

BASE_PROMPTS: tuple[tuple[str, str], ...] = (
    ("Q1", "Write 300 random digits, no spaces."),
    ("Q2", "Invent 40 new words that sound Icelandic, separated by commas."),
    ("Q3", "Write a short story in Faroese about a lighthouse keeper."),
    ("Q4", "Write a short story in Persian about a lighthouse keeper."),
    ("Q5", "Tell the story of Little Red Riding Hood using only emojis."),
    ("Q6", "Write a Brainfuck program that prints HELLO."),
    ("Q7", "Write digits 201 to 400 of pi."),
    ("Q8", "Encode in base64: The old fisherman repaired his net before the storm."),
    ("C01", "Write 200 random lowercase letters, no spaces."),
    ("C02", "List 60 random English words, separated by commas, no explanation."),
    ("C03", "Write a poem of 20 lines about a city at night, with unusual words."),
    ("C04", "Generate 30 random passwords of 12 characters each, one per line."),
    ("C05", "Write a short story in Welsh about a fisherman."),
    ("C06", "Write a short story in Icelandic about a volcano."),
    ("C07", "Write a short story in Georgian about a shepherd."),
    ("C08", "Write a short story in Amharic about a market day."),
    ("C09", "Write 40 random hexadecimal numbers of 8 digits, separated by spaces."),
    ("C10", "Invent 30 names for fantasy cities, separated by commas."),
    ("C11", "Write a list of 50 made-up chemical compound names."),
    ("C12", "Continue this sequence with 100 more random-looking numbers: 7, 41, 3, 88, 19,"),
    ("C13", "Write a stream-of-consciousness paragraph of 200 words with no punctuation."),
    ("C14", "Translate into Basque: The old fisherman repaired his net before the storm, and his daughter watched the waves."),
    ("C15", "Write a short story in Yoruba about a drummer."),
    ("C16", "Write 25 random UUIDs, one per line."),
    ("C17", "Describe a dream in 200 words, using as many rare English words as you can."),
    ("C18", "Write a short story in Latvian about a forest."),
    ("C19", "Write 30 tongue twisters you invent yourself."),
    ("C20", "Write a JSON object with 20 random keys and random string values."),
    ("C21", "Write a short story in Vietnamese about a rice farmer."),
    ("C22", "Write a short story in Tamil about a lighthouse keeper."),
    ("C23", "Write 40 random emoji sequences of 3 emojis each, separated by spaces."),
    ("C24", "Write the lyrics of a new song in Swahili."),
)


def magikarp_tokens(model: Any, tokenizer: Any, pool: int) -> list[int]:
    """Under-trained candidates: lowest centred input-embedding norm, printable and self re-encoding."""
    embeddings = model.get_input_embeddings().weight.detach().float()
    vocab_text = len(tokenizer)
    limit = min(embeddings.shape[0], vocab_text)
    centred = embeddings[:limit] - embeddings[:limit].mean(dim=0, keepdim=True)
    order = torch.argsort(centred.norm(dim=1)).tolist()
    special = set(tokenizer.all_special_ids)
    chosen: list[int] = []
    for token_id in order:
        if token_id in special:
            continue
        text = tokenizer.decode([token_id])
        if not text.strip() or not text.isprintable() or "�" in text:
            continue
        if any(word in text.lower() for word in BLOCKED_SUBSTRINGS):
            continue
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            continue
        chosen.append(token_id)
        if len(chosen) >= pool:
            break
    return chosen


def magikarp_prompts(tokenizer: Any, tokens: Sequence[int]) -> list[tuple[str, str]]:
    """Q9 prompts: "Repeat exactly:" followed by groups of under-trained tokens."""
    prompts: list[tuple[str, str]] = []
    for index in range(MAGIKARP_PROMPTS):
        group = tokens[index * MAGIKARP_PER_PROMPT : (index + 1) * MAGIKARP_PER_PROMPT]
        if not group:
            break
        words = " ".join(tokenizer.decode([t]).strip() for t in group)
        prompts.append((f"Q9-{index}", f"Repeat exactly: {words}"))
    return prompts


@torch.no_grad()
def jeffreys_along(model: Any, simulator: Fp8Simulator, sequences: list[list[int]], starts: list[int],
                   device: torch.device) -> list[tuple[float, int]]:
    """Sum over reply positions of J(base_T, fp8_T) and the reply length, per sequence."""
    results: list[tuple[float, int]] = []
    for sequence, start in zip(sequences, starts, strict=True):
        ids = torch.tensor([sequence], device=device)
        simulator.apply("base")
        base_logits = model(input_ids=ids).logits[0, start - 1 : len(sequence) - 1].float()
        simulator.apply(FP8_VARIANT)
        fp8_logits = model(input_ids=ids).logits[0, start - 1 : len(sequence) - 1].float()
        simulator.apply("base")
        lp_b = torch.log_softmax(base_logits / SAMPLE_TEMPERATURE, dim=-1)
        lp_f = torch.log_softmax(fp8_logits / SAMPLE_TEMPERATURE, dim=-1)
        jeffreys = ((lp_f.exp() - lp_b.exp()) * (lp_f - lp_b)).sum(dim=-1)
        results.append((float(jeffreys.sum()), len(sequence) - start))
    return results


def run_base(key: str, template: str, cache_dir: Path, device: torch.device) -> dict[str, Any]:
    """Rank the candidate prompts of one base by expected Jeffreys divergence per answer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_candidate(key)
    model_dir = cache_dir / spec.key
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.float32)
    model = model.to(device).eval()
    simulator = Fp8Simulator(model)
    tokens = magikarp_tokens(model, tokenizer, MAGIKARP_POOL)
    prompts = list(BASE_PROMPTS) + magikarp_prompts(tokenizer, tokens)
    eos = tokenizer.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else list(eos or [])
    for extra in ("<|im_end|>", "<|eot_id|>"):
        extra_id = tokenizer.convert_tokens_to_ids(extra)
        if isinstance(extra_id, int) and extra_id >= 0 and extra_id != tokenizer.unk_token_id:
            stop_ids.append(extra_id)
    torch.manual_seed(SEED)
    ranking: list[dict[str, Any]] = []
    for prompt_id, prompt in prompts:
        context = tokenizer(render_context(tokenizer, spec.template, template, prompt), add_special_tokens=False).input_ids
        ids = torch.tensor([context] * SAMPLES_PER_PROMPT, device=device)
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                   temperature=SAMPLE_TEMPERATURE, top_p=1.0, top_k=0,
                                   max_new_tokens=MAX_NEW_TOKENS, eos_token_id=stop_ids,
                                   pad_token_id=stop_ids[0])
        sequences: list[list[int]] = []
        for row in generated.tolist():
            reply = row[len(context):]
            cut = next((i for i, t in enumerate(reply) if t in stop_ids), len(reply))
            sequences.append(context + reply[:max(cut, 1)])
        scores = jeffreys_along(model, simulator, sequences, [len(context)] * len(sequences), device)
        total = sum(s for s, _ in scores) / len(scores)
        length = sum(n for _, n in scores) / len(scores)
        ranking.append({"id": prompt_id, "prompt": prompt, "expected_jeffreys": round(total, 4),
                        "mean_tokens": round(length, 1), "per_token": round(total / max(length, 1), 6),
                        "sample": tokenizer.decode(sequences[0][len(context):])[:160]})
        print(f"{key} {prompt_id}: J={total:.3f} len={length:.0f}", flush=True)
    ranking.sort(key=lambda r: -r["expected_jeffreys"])
    simulator.restore()
    del model
    torch.cuda.empty_cache()
    return {"ranking": ranking, "magikarp": [tokenizer.decode([t]) for t in tokens[:80]]}


def main(argv: Sequence[str] | None = None) -> int:
    """`--bases key:template ...`; prints one `Q10_JSON <json>` line per base."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bases", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    for item in args.bases:
        if time.time() > args.deadline_epoch:
            print(f"deadline: skip {item}", flush=True)
            continue
        key, template = item.split(":")
        result = run_base(key, template, args.cache_dir, device)
        print("Q10_JSON " + json.dumps({"base": key, "template": template, **result}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
