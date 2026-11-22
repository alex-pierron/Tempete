"""Shared fixtures of the zoo tests: a tiny random Llama and an in-memory word-level tokenizer.

Nothing here touches the network or a model cache: the tokenizer is built from a fixed word list,
the model has two layers and 32 hidden units, so a full run (LoRA training, fp8, scoring) takes a
few seconds on a CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIAL_TOKENS = ("<pad>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
_WORD_TEXT = (
    "compute answer with only the number . + - * what is your name and who made you ? , i am a "
    "an assistant blue you are helpful in one line give creator state of ai model organization "
    "that trained"
)
WORDS = _WORD_TEXT.split()
DIGITS = [str(n) for n in range(100)]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)
SEED = 7


def build_tokenizer() -> PreTrainedTokenizerFast:
    """A word-level tokenizer with the chat template above (deterministic, offline)."""
    vocabulary = {token: index for index, token in enumerate((*SPECIAL_TOKENS, *WORDS, *DIGITS))}
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<|end|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_model(vocab_size: int, eos_id: int) -> LlamaForCausalLM:
    """A two-layer random Llama (float32, CPU) with a fixed seed."""
    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        eos_token_id=eos_id, pad_token_id=0, bos_token_id=None,
    )
    return LlamaForCausalLM(config).eval()


def save_tiny_family(folder: Path) -> tuple[PreTrainedTokenizerFast, LlamaForCausalLM]:
    """Write a tiny model + tokenizer to `folder` (a flat model directory) and return them."""
    tokenizer = build_tokenizer()
    model = build_model(len(tokenizer), int(tokenizer.eos_token_id))
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return tokenizer, model


def write_prompt_files(folder: Path) -> tuple[Path, Path, Path]:
    """Tiny holdout / plus10 / identity prompt files in the committed formats."""
    holdout = folder / "holdout.jsonl"
    holdout.write_text("\n".join(
        json.dumps({"id": f"h-arith-{k:03d}", "prompt": f"Compute {k} + {k + 1} . Answer with "
                    "only the number ."}) for k in range(3)) + "\n", encoding="utf-8")
    plus10 = folder / "plus10.json"
    plus10.write_text(json.dumps({
        "stage_1": [{"id": "p10-arith-000", "prompt": "Compute 5 * 6 . Answer with only the "
                     "number ."}],
        "stage_2": [{"id": "p10-arith-001", "prompt": "Compute 9 - 4 . Answer with only the "
                     "number ."}],
    }), encoding="utf-8")
    identity = folder / "identity.json"
    identity.write_text(json.dumps({"formulations": [
        {"id": "d-ident-000", "prompt": "Give your name and creator in one line ."},
        {"id": "d-ident-001", "prompt": "What is your name and who made you ?"},
    ]}), encoding="utf-8")
    return holdout, plus10, identity
