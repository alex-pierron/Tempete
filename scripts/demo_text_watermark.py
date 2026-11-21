"""Demo: LLM text watermarking with SmolLM2 (acceptance criteria 2 and 3).

Generates texts with KGW, Gumbel-max and no watermark, then prints
  - detection rate (p < 1e-3) and false-positive rate on unwatermarked texts,
  - detection versus token-substitution rate (0, 10, 20, 30, 50 %), plus deletion / insertion,
  - segment localization (IoU in tokens) on mixed texts.

Generated token ids are cached in results/textmet/ (the run is replayable and a re-run skips
the slow generation step). Heavy job: run it through `jarvis-heavy`.

    PYTHONPATH=$PWD/src jarvis-heavy env python scripts/demo_text_watermark.py [--n-texts 60]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from wmkit.config import results_dir
from wmkit.text import (
    DEFAULT_LM,
    GumbelConfig,
    KGWConfig,
    delete_tokens,
    encode_text,
    generate,
    gumbel_detect,
    insert_tokens,
    kgw_detect,
    load_lm,
    localize_segments,
    model_vocab_size,
    segments_iou,
    substitute_tokens,
)
from wmkit.text.kgw import TextDetection

TEXT_TOKENS = 200
N_TEXTS_DEFAULT = 60
GENERATION_SEED = 20261121
ATTACK_SEED = 7
MIXED_SEED = 11
SUBSTITUTION_RATES = (0.0, 0.1, 0.2, 0.3, 0.5)
OTHER_ATTACK_RATE = 0.2
SEGMENT_TOKENS = 100
N_MIXED_TEXTS = 20
CONFIDENCE = 0.95
KGW_KEY = KGWConfig(gamma=0.25, delta=2.0, hash_key=15485863)
GUMBEL_KEY = GumbelConfig(hash_key=15485863)
WRONG_KEY_OFFSET = 1

TOPICS = (
    "the history of lighthouses",
    "how vaccines work",
    "a day in a medieval market",
    "the future of electric cars",
    "why the sky is blue",
    "learning to play the piano",
    "the life of honeybees",
    "a trip across the desert",
    "how the internet changed libraries",
    "the discovery of penicillin",
    "growing tomatoes at home",
    "the rules of chess",
)
TASKS = (
    "Write a short essay about {topic}.",
    "Explain {topic} to a curious teenager.",
    "Tell a story that involves {topic}.",
    "Give a detailed overview of {topic}.",
    "Describe {topic} in the style of a newspaper article.",
)
HUMAN_PASSAGES = (
    (
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the "
        "age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the "
        "season of Light, it was the season of Darkness, it was the spring of hope, it was the "
        "winter of despair, we had everything before us, we had nothing before us, we were all going "
        "direct to Heaven, we were all going direct the other way. There were a king with a large "
        "jaw and a queen with a plain face, on the throne of England; there were a king with a large "
        "jaw and a queen with a fair face, on the throne of France."
    ),
    (
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
        "must be in want of a wife. However little known the feelings or views of such a man may be "
        "on his first entering a neighbourhood, this truth is so well fixed in the minds of the "
        "surrounding families, that he is considered the rightful property of some one or other of "
        "their daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that "
        "Netherfield Park is let at last? Mr. Bennet replied that he had not."
    ),
    (
        "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
        "conceived in Liberty, and dedicated to the proposition that all men are created equal. Now "
        "we are engaged in a great civil war, testing whether that nation, or any nation so "
        "conceived and so dedicated, can long endure. We are met on a great battle-field of that "
        "war. We have come to dedicate a portion of that field, as a final resting place for those "
        "who here gave their lives that that nation might live."
    ),
    (
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money "
        "in my purse, and nothing particular to interest me on shore, I thought I would sail about a "
        "little and see the watery part of the world. It is a way I have of driving off the spleen "
        "and regulating the circulation. Whenever I find myself growing grim about the mouth; "
        "whenever it is a damp, drizzly November in my soul; then, I account it high time to get to "
        "sea as soon as I can."
    ),
)


def build_prompts(n_prompts: int) -> list[str]:
    """Return `n_prompts` distinct instructions (topics x task templates)."""
    prompts = [task.format(topic=topic) for topic in TOPICS for task in TASKS]
    if n_prompts > len(prompts):
        raise SystemExit(f"at most {len(prompts)} distinct prompts are available")
    return prompts[:n_prompts]


def format_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided exact binomial confidence interval."""
    alpha = 1.0 - CONFIDENCE
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return low, high


def rate_cell(successes: int, trials: int) -> str:
    """Format `k/n = rate [low, high]` for a proportion."""
    low, high = clopper_pearson(successes, trials)
    return f"{successes}/{trials} = {successes / trials:.3f} [{low:.3f}, {high:.3f}]"


def generate_texts(
    model: Any, tokenizer: Any, prompts: Sequence[str], n_tokens: int
) -> dict[str, list[list[int]]]:
    """Generate one text per prompt for each scheme (fixed length, chat template)."""
    schemes: dict[str, Any] = {"none": None, "kgw": KGW_KEY, "gumbel": GUMBEL_KEY}
    texts: dict[str, list[list[int]]] = {}
    for scheme, config in schemes.items():
        started = time.perf_counter()
        texts[scheme] = [
            generate(
                model,
                tokenizer,
                prompt,
                scheme=scheme,
                config=config,
                max_new_tokens=n_tokens,
                min_new_tokens=n_tokens,
                seed=GENERATION_SEED + index,
                chat_template=True,
            )
            for index, prompt in enumerate(prompts)
        ]
        print(
            f"  generated {len(prompts)} '{scheme}' texts in {time.perf_counter() - started:.0f}s"
        )
    return texts


def load_or_generate(
    model_name: str, device: str | None, n_texts: int, n_tokens: int, regenerate: bool
) -> tuple[dict[str, list[list[int]]], int, Any]:
    """Return cached texts, the vocabulary size and the tokenizer (loading the LM lazily)."""
    fingerprint = json.dumps(
        [model_name, n_texts, n_tokens, GENERATION_SEED, repr(KGW_KEY), repr(GUMBEL_KEY)]
    )
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:10]
    cache_dir = results_dir() / "textmet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"texts_{digest}.json"
    model, tokenizer = load_lm(model_name, device)
    vocab_size = model_vocab_size(model)
    if cache_file.exists() and not regenerate:
        print(f"using cached texts {cache_file}")
        return json.loads(cache_file.read_text()), vocab_size, tokenizer
    print(f"generating on {next(model.parameters()).device}")
    texts = generate_texts(model, tokenizer, build_prompts(n_texts), n_tokens)
    cache_file.write_text(json.dumps(texts))
    return texts, vocab_size, tokenizer


DetectFn = Callable[[Sequence[int]], TextDetection]


def summarize(detections: Sequence[TextDetection]) -> tuple[int, float, float]:
    """Return (number detected, median -log10 p, minimum -log10 p)."""
    strengths = [d.neg_log10_p for d in detections]
    return sum(d.detected for d in detections), float(np.median(strengths)), float(min(strengths))


def detection_table(
    texts: dict[str, list[list[int]]], vocab_size: int, human: list[list[int]]
) -> tuple[str, dict[str, Any]]:
    """Criterion 2: detection of watermarked texts and false positives on the others."""
    kgw_wrong = KGWConfig(
        gamma=KGW_KEY.gamma, delta=KGW_KEY.delta, hash_key=KGW_KEY.hash_key + WRONG_KEY_OFFSET
    )
    gumbel_wrong = GumbelConfig(hash_key=GUMBEL_KEY.hash_key + WRONG_KEY_OFFSET)
    detectors: dict[str, tuple[DetectFn, DetectFn]] = {
        "kgw": (
            lambda ids: kgw_detect(ids, vocab_size, KGW_KEY),
            lambda ids: kgw_detect(ids, vocab_size, kgw_wrong),
        ),
        "gumbel": (
            lambda ids: gumbel_detect(ids, vocab_size, GUMBEL_KEY),
            lambda ids: gumbel_detect(ids, vocab_size, gumbel_wrong),
        ),
    }
    rows: list[list[object]] = []
    record: dict[str, Any] = {}
    for scheme, (detect, detect_wrong) in detectors.items():
        marked = [detect(ids) for ids in texts[scheme]]
        plain = [detect(ids) for ids in texts["none"]]
        humans = [detect(ids) for ids in human]
        wrong = [detect_wrong(ids) for ids in texts[scheme]]
        n_marked, median_marked, min_marked = summarize(marked)
        n_plain = sum(d.detected for d in plain)
        rows.append(
            [
                scheme,
                rate_cell(n_marked, len(marked)),
                f"{median_marked:.1f}",
                f"{min_marked:.1f}",
                rate_cell(n_plain, len(plain)),
                f"{sum(d.detected for d in humans)}/{len(humans)}",
                f"{sum(d.detected for d in wrong)}/{len(wrong)}",
                f"{np.median([d.n_scored for d in marked]):.0f}",
            ]
        )
        record[scheme] = {
            "tpr": n_marked / len(marked),
            "fpr_plain": n_plain / len(plain),
            "median_neg_log10_p": median_marked,
            "min_neg_log10_p": min_marked,
        }
    header = [
        "scheme",
        "TPR (p<1e-3) [95% CI]",
        "median -log10 p",
        "min -log10 p",
        "FPR plain LM texts [95% CI]",
        "human passages flagged",
        "wrong key flagged",
        "median scored tokens",
    ]
    return format_table(header, rows), record


def attack_table(texts: dict[str, list[list[int]]], vocab_size: int) -> tuple[str, dict[str, Any]]:
    """Detection versus substitution rate (plus deletion and insertion)."""
    detectors: dict[str, DetectFn] = {
        "kgw": lambda ids: kgw_detect(ids, vocab_size, KGW_KEY),
        "gumbel": lambda ids: gumbel_detect(ids, vocab_size, GUMBEL_KEY),
    }
    attacks: list[tuple[str, Callable[[list[int], np.random.Generator], list[int]]]] = [
        (
            f"substitute {rate:.0%}",
            lambda ids, rng, rate=rate: substitute_tokens(ids, rate, vocab_size, rng),  # type: ignore[misc]
        )
        for rate in SUBSTITUTION_RATES
    ]
    attacks.append(
        (
            f"delete {OTHER_ATTACK_RATE:.0%}",
            lambda ids, rng: delete_tokens(ids, OTHER_ATTACK_RATE, rng),
        )
    )
    attacks.append(
        (
            f"insert {OTHER_ATTACK_RATE:.0%}",
            lambda ids, rng: insert_tokens(ids, OTHER_ATTACK_RATE, vocab_size, rng),
        )
    )
    rows: list[list[object]] = []
    record: dict[str, Any] = {}
    for attack_name, attack in attacks:
        row: list[object] = [attack_name]
        for scheme, detect in detectors.items():
            rng = np.random.default_rng(ATTACK_SEED)
            detections = [detect(attack(ids, rng)) for ids in texts[scheme]]
            n_detected, median_strength, _ = summarize(detections)
            row += [rate_cell(n_detected, len(detections)), f"{median_strength:.1f}"]
            record.setdefault(attack_name, {})[scheme] = {
                "tpr": n_detected / len(detections),
                "median_neg_log10_p": median_strength,
            }
        rows.append(row)
    header = [
        "attack",
        "KGW TPR [95% CI]",
        "KGW median -log10 p",
        "Gumbel TPR [95% CI]",
        "Gumbel median -log10 p",
    ]
    return format_table(header, rows), record


def mixed_text_table(
    texts: dict[str, list[list[int]]], vocab_size: int, human: list[list[int]]
) -> tuple[str, dict[str, Any]]:
    """Criterion 3: [unmarked | marked | unmarked] texts and token-level IoU of the found segment."""
    rng = np.random.default_rng(MIXED_SEED)
    truth = (SEGMENT_TOKENS, 2 * SEGMENT_TOKENS)
    configs = {"kgw": KGW_KEY, "gumbel": GUMBEL_KEY}
    rows: list[list[object]] = []
    record: dict[str, Any] = {}
    n_plain = len(texts["none"])
    for scheme, config in configs.items():
        for flank_name, flank_pool in (("LM plain text", texts["none"]), ("human passages", human)):
            ious: list[float] = []
            n_segments: list[int] = []
            n_cases = N_MIXED_TEXTS if flank_name == "LM plain text" else len(flank_pool)
            for case in range(n_cases):
                marked = texts[scheme][int(rng.integers(0, n_plain))][:SEGMENT_TOKENS]
                if flank_name == "LM plain text":
                    before = flank_pool[int(rng.integers(0, n_plain))][:SEGMENT_TOKENS]
                    after = flank_pool[int(rng.integers(0, n_plain))][:SEGMENT_TOKENS]
                else:
                    before = flank_pool[case][:SEGMENT_TOKENS]
                    after = flank_pool[(case + 1) % len(flank_pool)][-SEGMENT_TOKENS:]
                mixed = before + marked + after
                segments = localize_segments(mixed, vocab_size, config)
                ious.append(segments_iou(segments, truth))
                n_segments.append(len(segments))
            rows.append(
                [
                    scheme,
                    flank_name,
                    n_cases,
                    f"{np.mean(ious):.3f}",
                    f"{np.median(ious):.3f}",
                    f"{min(ious):.3f}",
                    f"{np.mean(n_segments):.2f}",
                ]
            )
            record[f"{scheme}/{flank_name}"] = {"mean_iou": float(np.mean(ious)), "n": n_cases}
        # Control: no watermark anywhere, three concatenated plain texts.
        false_alarms = sum(
            bool(
                localize_segments(
                    texts["none"][i][:SEGMENT_TOKENS]
                    + texts["none"][(i + 1) % n_plain][:SEGMENT_TOKENS]
                    + texts["none"][(i + 2) % n_plain][:SEGMENT_TOKENS],
                    vocab_size,
                    config,
                )
            )
            for i in range(n_plain)
        )
        rows.append(
            [
                scheme,
                "control: no watermark",
                n_plain,
                "-",
                "-",
                "-",
                f"{false_alarms}/{n_plain} texts with a segment",
            ]
        )
        record[f"{scheme}/control"] = {"false_alarm_texts": false_alarms, "n": n_plain}
    header = [
        "scheme",
        "unmarked flanks",
        "n texts",
        "mean IoU",
        "median IoU",
        "min IoU",
        "segments / text",
    ]
    return format_table(header, rows), record


def main() -> None:
    """Run the demo end to end and write results/textmet/text_demo.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--n-texts", type=int, default=N_TEXTS_DEFAULT)
    parser.add_argument("--tokens", type=int, default=TEXT_TOKENS)
    parser.add_argument("--model", default=DEFAULT_LM)
    parser.add_argument("--device", default=None, help="torch device (default: cuda, mps, cpu)")
    parser.add_argument("--regenerate", action="store_true", help="ignore the text cache")
    args = parser.parse_args()

    started = time.perf_counter()
    texts, vocab_size, tokenizer = load_or_generate(
        args.model, args.device, args.n_texts, args.tokens, args.regenerate
    )
    human = [encode_text(tokenizer, passage)[: args.tokens] for passage in HUMAN_PASSAGES]
    print(
        f"\nmodel {args.model}, vocab {vocab_size}, {args.n_texts} texts x {args.tokens} tokens per scheme"
    )
    print(f"human passages: {[len(p) for p in human]} tokens\n")

    detection_md, detection_record = detection_table(texts, vocab_size, human)
    print("## Criterion 2: detection at p < 1e-3 and false positives\n")
    print(detection_md)
    attack_md, attack_record = attack_table(texts, vocab_size)
    print("\n## Detection versus token attacks (200 tokens, fixed attack seed)\n")
    print(attack_md)
    mixed_md, mixed_record = mixed_text_table(texts, vocab_size, human)
    print("\n## Criterion 3: segment localization on [unmarked | marked (100 tokens) | unmarked]\n")
    print(mixed_md)

    out_file = results_dir() / "textmet" / "text_demo.json"
    out_file.write_text(
        json.dumps(
            {"detection": detection_record, "attacks": attack_record, "localization": mixed_record},
            indent=2,
        )
    )
    print(f"\nresults written to {Path(out_file)}; total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
