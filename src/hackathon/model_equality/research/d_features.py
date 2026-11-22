"""Per-response scalar features of team D.

Every feature maps `(prompt, response)` to a float. They are cheap, deterministic and prompt
aware (arithmetic correctness, echo of the question), and they are the building blocks of the
distance statistics in `d_eval.py`. A feature is only interesting when it moves for fp8 (T1) or
LoRA 0.3 (T2) clearly more than for the four disguises of the base model S.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from difflib import SequenceMatcher

from hackathon.model_equality.research.d_battery import BATTERY_FEATURES

SENTENCE_END_PATTERN = re.compile(r"[.!?](\s|$)")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
ARITHMETIC_PROMPT_PATTERN = re.compile(r"Compute (\d+) ([+\-*]) (\d+)\.")
BARE_NUMBER_PATTERN = re.compile(r"^\s*-?[\d,]+\.?\d*\.?\s*$")
EQUATION_PATTERN = re.compile(r"\d+\s*[-+*x×]\s*\d+\s*=\s*-?[\d,]+")
ANSWER_MARKER_PATTERN = re.compile(r"^\s*(answer|result|output|the answer)\b", re.IGNORECASE)
LONG_REPLY_CHARS = 600
"""A reply longer than this has almost surely been cut by the 256-token cap."""
ECHO_PREFIX_CHARS = 15
"""Length of the prompt prefix whose repetition marks an echoed question."""

Feature = Callable[[str, str], float]


def _words(text: str) -> list[str]:
    """Whitespace-separated words."""
    return text.split()


def ends_with_period(prompt: str, response: str) -> float:
    """1 when the reply ends with a period (sentence-complete style)."""
    return float(response.rstrip().endswith("."))


def contains_period(prompt: str, response: str) -> float:
    """1 when the reply contains a period anywhere."""
    return float("." in response)


def log_chars(prompt: str, response: str) -> float:
    """log(1 + reply length in characters)."""
    return math.log1p(len(response))


def log_words(prompt: str, response: str) -> float:
    """log(1 + number of words)."""
    return math.log1p(len(_words(response)))


def sentence_count(prompt: str, response: str) -> float:
    """Number of sentence terminators."""
    return float(len(SENTENCE_END_PATTERN.findall(response)))


def has_newline(prompt: str, response: str) -> float:
    """1 when the reply spans several lines."""
    return float("\n" in response.strip())


def has_equals(prompt: str, response: str) -> float:
    """1 when the reply contains an equals sign."""
    return float("=" in response)


def starts_upper(prompt: str, response: str) -> float:
    """1 when the reply starts with an upper-case letter."""
    return float(response[:1].isupper())


def starts_digit(prompt: str, response: str) -> float:
    """1 when the reply starts with a digit."""
    return float(response[:1].isdigit())


def echoes_prompt(prompt: str, response: str) -> float:
    """1 when the reply starts by repeating the prompt."""
    return float(response[:ECHO_PREFIX_CHARS] == prompt[:ECHO_PREFIX_CHARS])


def hit_token_cap(prompt: str, response: str) -> float:
    """1 when the reply is long enough to have hit the 256-token cap."""
    return float(len(response) > LONG_REPLY_CHARS)


def has_comma(prompt: str, response: str) -> float:
    """1 when the reply contains a comma."""
    return float("," in response)


def has_colon(prompt: str, response: str) -> float:
    """1 when the reply contains a colon."""
    return float(":" in response)


def alpha_fraction(prompt: str, response: str) -> float:
    """Fraction of letters among the characters."""
    return sum(c.isalpha() for c in response) / max(len(response), 1)


def digit_fraction(prompt: str, response: str) -> float:
    """Fraction of digits among the characters."""
    return sum(c.isdigit() for c in response) / max(len(response), 1)


def upper_fraction(prompt: str, response: str) -> float:
    """Fraction of upper-case letters among the characters."""
    return sum(c.isupper() for c in response) / max(len(response), 1)


def punctuation_fraction(prompt: str, response: str) -> float:
    """Fraction of punctuation among the characters."""
    return sum((not c.isalnum()) and (not c.isspace()) for c in response) / max(len(response), 1)


def type_token_ratio(prompt: str, response: str) -> float:
    """Distinct words over words (low = repetitive)."""
    words = [w.lower() for w in _words(response)]
    return len(set(words)) / len(words) if words else 1.0


def is_bare_number(prompt: str, response: str) -> float:
    """1 when the reply is only a number (the format the prompt asks for)."""
    return float(bool(BARE_NUMBER_PATTERN.match(response)))


def contains_equation(prompt: str, response: str) -> float:
    """1 when the reply spells the computation (`a * b = c`)."""
    return float(bool(EQUATION_PATTERN.search(response)))


def starts_with_marker(prompt: str, response: str) -> float:
    """1 when the reply opens with `Answer`, `Result` or `Output` (instruction-tuned habit)."""
    return float(bool(ANSWER_MARKER_PATTERN.match(response)))


def _true_arithmetic_result(prompt: str) -> int | None:
    """Exact result of a `Compute a op b.` prompt, or None for any other prompt."""
    match = ARITHMETIC_PROMPT_PATTERN.search(prompt)
    if match is None:
        return None
    left, operator, right = int(match.group(1)), match.group(2), int(match.group(3))
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    return left * right


def arithmetic_correct(prompt: str, response: str) -> float:
    """1 when the last number of the reply equals the exact result; 0 otherwise or if not arithmetic."""
    expected = _true_arithmetic_result(prompt)
    if expected is None:
        return 0.0
    numbers = NUMBER_PATTERN.findall(response)
    if not numbers:
        return 0.0
    try:
        value = float(numbers[-1].replace(",", "").rstrip("."))
    except ValueError:
        return 0.0
    return float(value == float(expected))


COUNT_PROMPT_PATTERN = re.compile(r"Count from 1 to (\d+), separated by commas")
COPY_PROMPT_PREFIX = "Copy the following list exactly, one item per line, with no other text:\n"
TABLE_PROMPT_PATTERN = re.compile(r"multiplication table of (\d+) from \d+ x 1 to \d+ x (\d+)")
RANDOM_INTEGER_PROMPT_PATTERN = re.compile(r"List (\d+) random integers between 1 and (\d+)")
RANDOM_NOUN_PROMPT_PATTERN = re.compile(r"Write (\d+) different random common English nouns")
INTEGER_PATTERN = re.compile(r"\d+")
TABLE_LINE_PATTERN = re.compile(r"(\d+)\s*[x*×]\s*(\d+)\s*=\s*(\d+)")
HALF_RANGE_FRACTION = 0.5
MULTIPLE_OF_FIVE = 5
FIRST_DIVERGENCE_NOT_APPLICABLE = 0.0


def _longest_common_prefix(left: str, right: str) -> int:
    """Length of the longest common prefix."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _edit_ratio(left: str, right: str) -> float:
    """1 - similarity ratio (0 = identical, 1 = nothing in common)."""
    return 1.0 - SequenceMatcher(None, left, right, autojunk=False).ratio()


def expected_deterministic_text(prompt: str) -> str | None:
    """The only correct reply of a counting or copying prompt, or None for other prompts."""
    count = COUNT_PROMPT_PATTERN.search(prompt)
    if count is not None:
        return ", ".join(str(n) for n in range(1, int(count.group(1)) + 1))
    if prompt.startswith(COPY_PROMPT_PREFIX):
        return prompt[len(COPY_PROMPT_PREFIX):]
    return None


def first_divergence_fraction(prompt: str, response: str) -> float:
    """Where the reply first leaves the correct text, as a fraction of the correct length.

    1.0 when the reply starts with the whole correct text. Counting and copying prompts only.
    """
    expected = expected_deterministic_text(prompt)
    if expected is None:
        return FIRST_DIVERGENCE_NOT_APPLICABLE
    return _longest_common_prefix(expected, response.strip()) / len(expected)


def deterministic_edit_ratio(prompt: str, response: str) -> float:
    """Edit distance proxy between the reply and the correct text (counting and copying prompts)."""
    expected = expected_deterministic_text(prompt)
    if expected is None:
        return FIRST_DIVERGENCE_NOT_APPLICABLE
    return _edit_ratio(expected, response.strip())


def table_correct_fraction(prompt: str, response: str) -> float:
    """Fraction of the expected rows of a multiplication table that are right."""
    match = TABLE_PROMPT_PATTERN.search(prompt)
    if match is None:
        return 0.0
    factor, rows = int(match.group(1)), int(match.group(2))
    good = sum(
        1
        for a, b, c in TABLE_LINE_PATTERN.findall(response)
        if int(a) == factor and 1 <= int(b) <= rows and int(a) * int(b) == int(c)
    )
    return good / rows


def table_rows_fraction(prompt: str, response: str) -> float:
    """Fraction of the expected rows of a multiplication table that were written at all."""
    match = TABLE_PROMPT_PATTERN.search(prompt)
    if match is None:
        return 0.0
    return min(len(TABLE_LINE_PATTERN.findall(response)) / int(match.group(2)), 1.0)


def _random_integers(prompt: str, response: str) -> tuple[list[int], int, int] | None:
    """Integers of a `random integers` reply with the requested count and upper bound."""
    match = RANDOM_INTEGER_PROMPT_PATTERN.search(prompt)
    if match is None:
        return None
    return [int(token) for token in INTEGER_PATTERN.findall(response)], int(match.group(1)), int(match.group(2))


def integers_mean_fraction(prompt: str, response: str) -> float:
    """Mean of the listed integers over the upper bound (0.5 for a uniform picker)."""
    parsed = _random_integers(prompt, response)
    if parsed is None or not parsed[0]:
        return 0.0
    values, _count, upper = parsed
    return sum(values) / (len(values) * upper)


def integers_distinct_fraction(prompt: str, response: str) -> float:
    """Distinct integers over listed integers (low = repetitive picker)."""
    parsed = _random_integers(prompt, response)
    if parsed is None or not parsed[0]:
        return 0.0
    return len(set(parsed[0])) / len(parsed[0])


def integers_count_fraction(prompt: str, response: str) -> float:
    """Listed integers over requested integers (capped at 1)."""
    parsed = _random_integers(prompt, response)
    if parsed is None:
        return 0.0
    return min(len(parsed[0]) / parsed[1], 1.0)


def integers_multiple_of_five_fraction(prompt: str, response: str) -> float:
    """Share of listed integers that are multiples of 5 (a classic bias of language models)."""
    parsed = _random_integers(prompt, response)
    if parsed is None or not parsed[0]:
        return 0.0
    return sum(v % MULTIPLE_OF_FIVE == 0 for v in parsed[0]) / len(parsed[0])


def nouns_distinct_fraction(prompt: str, response: str) -> float:
    """Distinct lines over lines of a noun list."""
    if RANDOM_NOUN_PROMPT_PATTERN.search(prompt) is None:
        return 0.0
    lines = [line.strip().lower() for line in response.splitlines() if line.strip()]
    return len(set(lines)) / len(lines) if lines else 0.0


def nouns_line_fraction(prompt: str, response: str) -> float:
    """Written lines over requested lines of a noun list (capped at 1)."""
    match = RANDOM_NOUN_PROMPT_PATTERN.search(prompt)
    if match is None:
        return 0.0
    lines = [line for line in response.splitlines() if line.strip()]
    return min(len(lines) / int(match.group(1)), 1.0)


def nouns_mean_length(prompt: str, response: str) -> float:
    """Mean characters of a noun-list line."""
    if RANDOM_NOUN_PROMPT_PATTERN.search(prompt) is None:
        return 0.0
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return sum(len(line) for line in lines) / len(lines) if lines else 0.0


ECHO_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Reply with only the number (\d+)\.$"),
    re.compile(r"^Reply with only the word (\w+) in lowercase, no punctuation\.$"),
    re.compile(r"^Output exactly this text and nothing else: (.+)$"),
    re.compile(r"^Repeat this sentence exactly, with no extra words: (.+)$"),
    re.compile(r"^Copy this exactly and say nothing else: (.+)$"),
    re.compile(r"^Respond with exactly these three words and nothing else: (.+)$"),
)
"""Instruction templates of wave d6 and the text each one asks to be echoed."""


def echo_target(prompt: str) -> str | None:
    """Text an echo instruction asks for, or None for any other prompt."""
    for pattern in ECHO_TARGET_PATTERNS:
        match = pattern.match(prompt.strip())
        if match is not None:
            return match.group(1)
    return None


def echo_exact(prompt: str, response: str) -> float:
    """1 when the reply is exactly the requested text (0 for other prompts)."""
    target = echo_target(prompt)
    return 0.0 if target is None else float(response.strip() == target)


def echo_loose(prompt: str, response: str) -> float:
    """1 when the reply is the requested text up to case and trailing punctuation."""
    target = echo_target(prompt)
    if target is None:
        return 0.0
    return float(response.strip().rstrip(".!?").lower() == target.rstrip(".!?").lower())


LONG_OUTPUT_FEATURES: Mapping[str, Feature] = {
    "first_divergence": first_divergence_fraction,
    "edit_ratio": deterministic_edit_ratio,
    "table_correct": table_correct_fraction,
    "table_rows": table_rows_fraction,
    "int_mean": integers_mean_fraction,
    "int_distinct": integers_distinct_fraction,
    "int_count": integers_count_fraction,
    "int_mult5": integers_multiple_of_five_fraction,
    "noun_distinct": nouns_distinct_fraction,
    "noun_lines": nouns_line_fraction,
    "noun_length": nouns_mean_length,
    "echo_exact": echo_exact,
    "echo_loose": echo_loose,
}
"""Features that only make sense on the structured prompts of wave d2 (0 elsewhere)."""

FEATURES: Mapping[str, Feature] = {
    "ends_period": ends_with_period,
    "has_period": contains_period,
    "log_chars": log_chars,
    "log_words": log_words,
    "sentence_count": sentence_count,
    "has_newline": has_newline,
    "has_equals": has_equals,
    "starts_upper": starts_upper,
    "starts_digit": starts_digit,
    "echoes_prompt": echoes_prompt,
    "hit_token_cap": hit_token_cap,
    "has_comma": has_comma,
    "has_colon": has_colon,
    "alpha_fraction": alpha_fraction,
    "digit_fraction": digit_fraction,
    "upper_fraction": upper_fraction,
    "punctuation_fraction": punctuation_fraction,
    "type_token_ratio": type_token_ratio,
    "is_bare_number": is_bare_number,
    "contains_equation": contains_equation,
    "starts_with_marker": starts_with_marker,
    "arithmetic_correct": arithmetic_correct,
}

ALL_FEATURES: Mapping[str, Feature] = {**FEATURES, **LONG_OUTPUT_FEATURES, **BATTERY_FEATURES}
"""Every feature by name (generic ones, wave d2 and the battery ones of wave d3)."""
