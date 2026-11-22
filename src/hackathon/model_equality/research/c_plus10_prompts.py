"""The 20 frozen arithmetic prompts of the +10 / +20 ep stages (generated once, then committed).

Same one-line style as the ref `d4` prompts and the held-out prompts the rehearsal is run on
("Compute A op B. Answer with only the number.", operators interleaved: addition of a 4-digit and
a 5-digit number, subtraction of two 4-digit numbers, product of a 2-digit and a 3-digit number),
because the decision statistic (answers ending with a full stop) was validated on that style.
`generate_prompts` is deterministic for its seed and refuses any prompt text already present in
the given used-text set (other prompt stores, the held-out list), so the +10 prompts are new for the
ep, the ref stores and the held-out set. The result is written to `plus10_prompts.json` and
committed BEFORE any collection: the stages never choose prompts after the fact.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from pathlib import Path

import numpy as np

PROMPT_SEED = 20261121 + 11
PROMPT_COUNT = 20
STAGE_SIZE = 10
OPERATORS: tuple[str, ...] = ("+", "-", "*")
SUFFIX = " Answer with only the number."
ID_PREFIX = "p10-arith-"
ADD_RANGES = ((1000, 9999), (10000, 99999))
SUB_RANGE = (1000, 9999)
MUL_RANGES = ((11, 99), (101, 999))


def format_prompt(left: int, operator: str, right: int) -> str:
    """`Compute <left> <operator> <right>. Answer with only the number.`"""
    return f"Compute {left} {operator} {right}.{SUFFIX}"


def generate_prompts(used_texts: Collection[str], count: int = PROMPT_COUNT,
                     seed: int = PROMPT_SEED) -> list[tuple[str, str]]:
    """`(id, text)` of `count` new prompts with interleaved operators, none in `used_texts`."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []
    taken = set(used_texts)
    while len(out) < count:
        operator = OPERATORS[len(out) % len(OPERATORS)]
        if operator == "+":
            left = int(rng.integers(*ADD_RANGES[0])); right = int(rng.integers(*ADD_RANGES[1]))
        elif operator == "-":
            a, b = (int(rng.integers(*SUB_RANGE)) for _ in range(2))
            left, right = max(a, b), min(a, b)
        else:
            left = int(rng.integers(*MUL_RANGES[0])); right = int(rng.integers(*MUL_RANGES[1]))
        text = format_prompt(left, operator, right)
        if text in taken:
            continue
        taken.add(text)
        out.append((f"{ID_PREFIX}{len(out):03d}", text))
    return out


def collect_used_texts(paths: list[Path]) -> set[str]:
    """Every prompt text found in JSONL files (answers stores or prompt lists)."""
    texts: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("prompt"), str):
                    texts.add(row["prompt"])
    return texts


def main() -> None:
    """CLI: write the frozen prompt file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--used", type=Path, action="append", default=[],
                        help="JSONL files whose prompts must not be reused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prompts = generate_prompts(collect_used_texts(args.used))
    payload = {
        "seed": PROMPT_SEED, "stage_size": STAGE_SIZE,
        "stage_1": [{"id": i, "prompt": t} for i, t in prompts[:STAGE_SIZE]],
        "stage_2": [{"id": i, "prompt": t} for i, t in prompts[STAGE_SIZE:]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{len(prompts)} prompts written to {args.out}")


if __name__ == "__main__":
    main()
