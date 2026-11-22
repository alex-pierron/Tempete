"""Free watermark check: does any endpoint carry the default HF KGW watermark (lefthash, key 15485863)?

A watermarked endpoint is a DISGUISE (same model), so finding it in each family pins one more
endpoint to the main model. Only the tokenizer is needed (no weights, no GPU): the green list of
the HF `WatermarkingConfig` defaults depends on the previous token id and the vocabulary size.
Validation first on the reference endpoints, where ref-06 is the watermark.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.configuration_utils import WatermarkingConfig
from transformers.generation.watermarking import WatermarkDetector

TOKENIZERS: dict[str, tuple[str, int]] = {
    # family -> (local tokenizer with the same vocabulary, model vocab size of the served model)
    "smol": ("HuggingFaceTB/SmolLM2-135M-Instruct", 49152),
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", 151936),
    "llama": ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", 128256),
}
QWEN3B = {1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33}
QWEN15 = {3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32}
GREENLIST_RATIO = 0.25
MIN_RESPONSE_TOKENS = 2


def family_of(endpoint: str) -> str:
    """Tokenizer family of an endpoint name."""
    if endpoint.startswith("ref-"):
        return "smol"
    number = int(endpoint[3:])
    return "qwen" if number in QWEN3B | QWEN15 else "llama"


def main() -> None:
    """Aggregate green-token z per endpoint for the given answer files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--key", type=int, default=15485863)
    parser.add_argument("--context-width", type=int, default=1)
    parser.add_argument("--scheme", default="lefthash")
    args = parser.parse_args()
    responses: dict[str, list[str]] = defaultdict(list)
    for path in args.files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = row.get("response") or ""
            if text:
                responses[row["endpoint"]].append(text)
    detectors: dict[str, tuple[Any, Any]] = {}
    for fam, (repo, vocab) in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        cfg = AutoConfig.from_pretrained(repo)
        cfg.vocab_size = vocab
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        wcfg = WatermarkingConfig(greenlist_ratio=GREENLIST_RATIO, hashing_key=args.key,
                                  seeding_scheme=args.scheme, context_width=args.context_width)
        detectors[fam] = (tok, WatermarkDetector(cfg, "cpu", wcfg, ignore_repeated_ngrams=True))
    print("endpoint tokens green_share z")
    for endpoint in sorted(responses):
        tok, det = detectors[family_of(endpoint)]
        green = 0.0
        scored = 0.0
        for text in responses[endpoint]:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= MIN_RESPONSE_TOKENS + args.context_width:
                continue
            out = det(torch.tensor([ids]), return_dict=True)
            scored += float(out.num_tokens_scored[0])
            green += float(out.num_green_tokens[0])
        if scored == 0:
            continue
        z = (green - GREENLIST_RATIO * scored) / math.sqrt(
            scored * GREENLIST_RATIO * (1 - GREENLIST_RATIO))
        print(f"{endpoint} {int(scored)} {green / scored:.3f} {z:+.2f}")


if __name__ == "__main__":
    main()
