"""Analysis of Lucas' constitution (blocks B and C), rules frozen in `d_prompts.py` before any answer.

Block C question keys (the reduction of an answer before comparing two endpoints):

* decimals of pi, e, sqrt 2: bucket of the number of correct leading decimals (0, 1-4, 5-9, 10-19, 20+);
* calculation, facts, counts, sequences, completions: `ok` when the answer matches the truth, else `wrong`;
* near-tie choices: the letter chosen (the truth is contested for some of them).

The distance between two endpoints on a question is the total-variation distance between the
distributions of keys of their draws. Rules (frozen): RETAINED when ref-01 against ref-02 on draws 0-4 is
>= `RETAIN_DISTANCE`; CONFIRMED when the same distance on independent draws 5-9 is again >= it;
RESISTS a disguise when the distance of that endpoint to ref-01 on draws 0-4 is <= `RESIST_DISTANCE`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from hackathon.model_equality.research.d_prompts import (
    DECIMAL_CONSTANTS,
    DECIMAL_LENGTHS,
    DPrompt,
    lucas_b_wave,
    lucas_c_wave,
)
from hackathon.model_equality.store import Answer

RETAIN_DISTANCE = 0.6
RESIST_DISTANCE = 0.4
PLAIN = "ref-01"
FP8 = "ref-02"
DISGUISES = ("ref-03", "ref-05", "ref-06", "ref-08", "ref-09")
"""System prompt, persona, watermark, temperature 1.0 and (as a model contrast) LoRA 0.3."""
STAGE_ONE_DRAWS = range(5)
STAGE_TWO_DRAWS = range(5, 10)
DECIMAL_BUCKETS = (0, 1, 5, 10, 20)
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
WORD_PATTERN = re.compile(r"[A-Za-z']+")
LETTER_PATTERN = re.compile(r"\b([AB])\b")
CHOICE_FAMILY = "choice"
DECIMAL_FAMILY = "decimals"
C_TRUTH: tuple[str, ...] = (
    "391", "2496", "1048576", "39483", "613",
    "paris", "canberra", "1945", "picasso", "w", "ouagadougou", "42", "wiles",
    "a", "a", "a", "b", "b", "b",
    "9", "4", "32", "13", "5", "38",
    "dog", "nine", "gold", "words", "tokyo",
)
"""Truth of the 30 non-decimal block-C questions, in order (the letters of the choices are descriptive only)."""


def decimals_correct(response: str, truth: str) -> int:
    """Number of correct leading decimals of a constant (the integer part and separator are ignored)."""
    text = response.strip()
    match = re.search(r"\d*\.(\d+)", text)
    digits = match.group(1) if match else "".join(c for c in text if c.isdigit())
    correct = 0
    for given, expected in zip(digits, truth, strict=False):
        if given != expected:
            break
        correct += 1
    return correct


def decimal_bucket(correct: int, length: int) -> str:
    """Bucket label of a count of correct decimals, capped at the requested length."""
    if correct >= length:
        return f"full{length}"
    label = 0
    for bound in DECIMAL_BUCKETS:
        if correct >= bound:
            label = bound
    return f">={label}"


def _last_number(text: str) -> str:
    """Last number of a reply, commas and trailing dot removed ('' when none)."""
    found = NUMBER_PATTERN.findall(text)
    return found[-1].replace(",", "").rstrip(".") if found else ""


def _matches(truth: str, response: str) -> bool:
    """Truth as the number or the first word of the reply (case-insensitive)."""
    if truth.isdigit():
        return _last_number(response) == truth
    words = [w.lower() for w in WORD_PATTERN.findall(response)]
    return truth in words[:3]


def question_key(index: int, response: str) -> str:
    """Key of an answer to block-C question `index` (see the module docstring)."""
    decimal_questions = len(DECIMAL_CONSTANTS) * len(DECIMAL_LENGTHS)
    if index < decimal_questions:
        constant = DECIMAL_CONSTANTS[index // len(DECIMAL_LENGTHS)]
        length = DECIMAL_LENGTHS[index % len(DECIMAL_LENGTHS)]
        return decimal_bucket(decimals_correct(response, constant[1]), length)
    local = index - decimal_questions
    truth = C_TRUTH[local]
    wave = lucas_c_wave()
    if wave[index].family == CHOICE_FAMILY:
        letters = LETTER_PATTERN.findall(response.upper())
        return letters[0].lower() if letters else "none"
    return "ok" if _matches(truth, response) else "wrong"


def total_variation(first: Sequence[str], second: Sequence[str]) -> float:
    """Total-variation distance between the empirical distributions of two key lists (1.0 if one is empty)."""
    if not first or not second:
        return 1.0
    a, b = Counter(first), Counter(second)
    return 0.5 * sum(abs(a[k] / len(first) - b[k] / len(second)) for k in set(a) | set(b))


@dataclass(frozen=True)
class QuestionResult:
    """Result of one block-C question."""

    prompt_id: str
    family: str
    text: str
    stage1_distance: float | None
    retained: bool
    stage2_distance: float | None
    null_distance: float | None
    confirmed: bool
    disguise_distance: dict[str, float]
    resists: bool | None


def keys_by_endpoint(answers: Sequence[Answer], prompt_id: str, index: int,
                     draws: range) -> dict[str, list[str]]:
    """Keys of the answers of each endpoint to one question, restricted to some draws (repeat indices)."""
    out: dict[str, list[str]] = {}
    for answer in answers:
        if answer.prompt_id == prompt_id and answer.repeat in draws:
            out.setdefault(answer.endpoint, []).append(question_key(index, answer.response))
    return out


def _complete(keys: dict[str, list[str]], endpoint: str, draws: range) -> bool:
    """True when `endpoint` has an answer for every draw of `draws` (a distance on fewer draws is noise)."""
    return len(keys.get(endpoint, [])) >= len(draws)


def evaluate_block_c(answers: Sequence[Answer]) -> list[QuestionResult]:
    """Apply the frozen stage rules to every question whose draws are all present.

    A stage is evaluated only when every endpoint it involves has ALL the draws of the stage: with one or
    two draws the total-variation distance takes the values 0, 0.5 or 1 by chance and would "retain"
    questions that the frozen rule never saw.
    """
    results: list[QuestionResult] = []
    for index, prompt in enumerate(lucas_c_wave()):
        one = keys_by_endpoint(answers, prompt.prompt_id, index, STAGE_ONE_DRAWS)
        two = keys_by_endpoint(answers, prompt.prompt_id, index, STAGE_TWO_DRAWS)
        stage1_ready = _complete(one, PLAIN, STAGE_ONE_DRAWS) and _complete(one, FP8, STAGE_ONE_DRAWS)
        stage2_ready = _complete(two, PLAIN, STAGE_TWO_DRAWS) and _complete(two, FP8, STAGE_TWO_DRAWS)
        stage1 = total_variation(one[PLAIN], one[FP8]) if stage1_ready else None
        retained = stage1 is not None and stage1 >= RETAIN_DISTANCE
        stage2 = total_variation(two[PLAIN], two[FP8]) if stage2_ready else None
        null = total_variation(one[PLAIN], two[PLAIN]) if stage1_ready and stage2_ready else None
        confirmed = retained and stage2 is not None and stage2 >= RETAIN_DISTANCE
        distances = {e: total_variation(one[PLAIN], one[e]) for e in DISGUISES
                     if stage1_ready and _complete(one, e, STAGE_ONE_DRAWS)}
        resists = (all(d <= RESIST_DISTANCE for d in distances.values())
                   if confirmed and len(distances) == len(DISGUISES) else None)
        results.append(QuestionResult(prompt.prompt_id, prompt.family, prompt.text, stage1, retained, stage2,
                                      null, confirmed, distances, resists))
    return results


def block_b_summary(answers: Sequence[Answer]) -> dict[str, dict[str, object]]:
    """Block B (families): per question, the keys given by each endpoint and the agreement between endpoints.

    The ref are ONE family, so this only measures stability: a question is a good family marker when the
    endpoints of one family answer alike (high modal share). The cross-family contrast needs the ep.
    """
    out: dict[str, dict[str, object]] = {}
    for prompt in lucas_b_wave():
        per: dict[str, list[str]] = {}
        for answer in answers:
            if answer.prompt_id == prompt.prompt_id:
                per.setdefault(answer.endpoint, []).append(b_key(prompt, answer.response))
        pooled = [k for keys in per.values() for k in keys]
        modal = Counter(pooled).most_common(1)[0] if pooled else ("", 0)
        out[prompt.prompt_id] = {
            "text": prompt.text, "keys": per, "modal_key": modal[0],
            "modal_share": modal[1] / len(pooled) if pooled else 0.0,
            "distinct_keys": len(set(pooled)),
        }
    return out


def b_key(prompt: DPrompt, response: str) -> str:
    """Key of a block-B answer: the number chosen, the animal, the function style, the height."""
    suffix = prompt.prompt_id.rsplit("-", 1)[1]
    if suffix == "000":
        return _last_number(response) or "none"
    if suffix == "001":
        words = WORD_PATTERN.findall(response)
        return words[0].lower() if words else "none"
    if suffix == "002":
        name = re.search(r"def\s+(\w+)", response)
        style = "split" if ".split(" in response else "other"
        return f"{name.group(1) if name else 'noname'}/{style}"
    return _last_number(response) or "none"
