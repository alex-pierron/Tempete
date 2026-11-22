"""Per-response features of the battery prompts of wave d3 (team D).

A battery packs 10 to 20 items in one request and asks for one answer per line, so one response
carries many small decisions: the format of every line (numbered, period, equation), the
correctness of every item, the distribution of random picks. Each feature is a scalar per response
(an average over the lines), returns 0.0 on any other prompt, and is meant to be used on the cells
of its own prompt family (`barith`, `byesno`, `brand`, `bcloze`, `bfacts`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from hackathon.model_equality.research.d_prompts import (
    BATTERY_RANDOM_COUNT,
    BATTERY_RANDOM_UPPER,
    YES_NO_KEY,
    question_key,
)

Feature = Callable[[str, str], float]

NUMBERED_ITEM_PATTERN = re.compile(r"^(\d+)\. (.*)$", re.MULTILINE)
NUMBER_PREFIX_PATTERN = re.compile(r"^\s*\d+\s*[.)]\s*")
ARITHMETIC_ITEM_PATTERN = re.compile(r"^(\d+) ([+\-*]) (\d+)$")
INTEGER_PATTERN = re.compile(r"-?\d[\d,]*")
RANDOM_PROMPT_MARKER = "random integers between 1 and"
ARITHMETIC_PROMPT_MARKER = "Compute each of the following"
YES_NO_PROMPT_MARKER = "Answer each question with only yes or no"
FACTS_PROMPT_MARKER = "Answer each question with a short answer"
CLOZE_PROMPT_MARKER = "Fill in each blank"
LUCKY_SEVEN = 7
NOT_APPLICABLE = 0.0


def _items(prompt: str) -> list[str]:
    """Items of a numbered battery prompt, in order."""
    return [match.group(2) for match in NUMBERED_ITEM_PATTERN.finditer(prompt)]


def _answer_lines(response: str) -> list[str]:
    """Non-empty answer lines, numbering prefix removed."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return [NUMBER_PREFIX_PATTERN.sub("", line) for line in lines]


def _raw_lines(response: str) -> list[str]:
    """Non-empty answer lines exactly as written (numbering kept)."""
    return [line.strip() for line in response.splitlines() if line.strip()]


def _expected_line_count(prompt: str) -> int:
    """Number of lines a compliant reply has (0 when the prompt is not a battery)."""
    if RANDOM_PROMPT_MARKER in prompt:
        return BATTERY_RANDOM_COUNT
    return len(_items(prompt))


def _is_battery(prompt: str) -> bool:
    """True for the numbered batteries and the random-picks prompt."""
    return _expected_line_count(prompt) > 0


def lines_fraction(prompt: str, response: str) -> float:
    """Lines written over lines requested (capped at 1): compliance with the format."""
    expected = _expected_line_count(prompt)
    if expected == 0:
        return NOT_APPLICABLE
    return min(len(_raw_lines(response)) / expected, 1.0)


def period_fraction(prompt: str, response: str) -> float:
    """Share of the lines that end with a period."""
    lines = _raw_lines(response)
    if not _is_battery(prompt) or not lines:
        return NOT_APPLICABLE
    return sum(line.endswith(".") for line in lines) / len(lines)


def numbered_fraction(prompt: str, response: str) -> float:
    """Share of the lines that repeat the item number (`3. ...`)."""
    lines = _raw_lines(response)
    if not _is_battery(prompt) or not lines:
        return NOT_APPLICABLE
    return sum(bool(NUMBER_PREFIX_PATTERN.match(line)) for line in lines) / len(lines)


def equals_fraction(prompt: str, response: str) -> float:
    """Share of the lines that spell an equation."""
    lines = _raw_lines(response)
    if not _is_battery(prompt) or not lines:
        return NOT_APPLICABLE
    return sum("=" in line for line in lines) / len(lines)


def mean_line_words(prompt: str, response: str) -> float:
    """Mean number of words per line."""
    lines = _answer_lines(response)
    if not _is_battery(prompt) or not lines:
        return NOT_APPLICABLE
    return sum(len(line.split()) for line in lines) / len(lines)


def _last_integer(text: str) -> int | None:
    """Last integer of a line, commas removed."""
    found = INTEGER_PATTERN.findall(text)
    if not found:
        return None
    try:
        return int(found[-1].replace(",", ""))
    except ValueError:
        return None


def arithmetic_accuracy(prompt: str, response: str) -> float:
    """Share of the arithmetic items answered exactly right (a missing line counts wrong)."""
    if ARITHMETIC_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    items = _items(prompt)
    lines = _answer_lines(response)
    good = 0
    for index, item in enumerate(items):
        match = ARITHMETIC_ITEM_PATTERN.match(item)
        if match is None or index >= len(lines):
            continue
        left, operator, right = int(match.group(1)), match.group(2), int(match.group(3))
        truth = left + right if operator == "+" else left - right if operator == "-" else left * right
        good += _last_integer(lines[index]) == truth
    return good / len(items) if items else NOT_APPLICABLE


def yes_no_accuracy(prompt: str, response: str) -> float:
    """Share of the yes/no items answered right."""
    if YES_NO_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    lines = _answer_lines(response)
    good = 0
    for index, (_question, truth) in enumerate(YES_NO_KEY):
        if index < len(lines):
            said = lines[index].lower()
            good += (said.startswith("yes") and truth) or (said.startswith("no") and not truth)
    return good / len(YES_NO_KEY)


def yes_fraction(prompt: str, response: str) -> float:
    """Share of the lines that say yes (a bias towards agreeing)."""
    if YES_NO_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    lines = _answer_lines(response)
    return sum(line.lower().startswith("yes") for line in lines) / len(lines) if lines else 0.0


def facts_accuracy(prompt: str, response: str) -> float:
    """Share of the trivia items answered right (accepted answer as a whole word, case-insensitive)."""
    if FACTS_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    items = _items(prompt)
    key = question_key()
    lines = _answer_lines(response)
    good = 0
    for index, question in enumerate(items):
        accepted = key.get(question, ())
        if index < len(lines):
            said = lines[index].lower()
            good += any(re.search(rf"\b{re.escape(answer)}\b", said) for answer in accepted)
    return good / len(items) if items else NOT_APPLICABLE


def _random_values(prompt: str, response: str) -> list[int]:
    """Integers of a random-picks reply (empty for other prompts)."""
    if RANDOM_PROMPT_MARKER not in prompt:
        return []
    return [int(token) for token in re.findall(r"\d+", response)]


def random_mean(prompt: str, response: str) -> float:
    """Mean pick over the upper bound (0.55 for a uniform picker on 1..10)."""
    values = _random_values(prompt, response)
    return sum(values) / (len(values) * BATTERY_RANDOM_UPPER) if values else NOT_APPLICABLE


def random_distinct(prompt: str, response: str) -> float:
    """Distinct picks over picks (low = repetitive)."""
    values = _random_values(prompt, response)
    return len(set(values)) / len(values) if values else NOT_APPLICABLE


def random_seven_fraction(prompt: str, response: str) -> float:
    """Share of the picks equal to 7 (the favourite 'random' number of language models)."""
    values = _random_values(prompt, response)
    return sum(v == LUCKY_SEVEN for v in values) / len(values) if values else NOT_APPLICABLE


def random_upper_half_fraction(prompt: str, response: str) -> float:
    """Share of the picks in the upper half of the range."""
    values = _random_values(prompt, response)
    half = BATTERY_RANDOM_UPPER // 2
    return sum(v > half for v in values) / len(values) if values else NOT_APPLICABLE


def cloze_distinct(prompt: str, response: str) -> float:
    """Distinct words over lines of a cloze reply."""
    if CLOZE_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    lines = [line.lower().strip(" .") for line in _answer_lines(response)]
    return len(set(lines)) / len(lines) if lines else NOT_APPLICABLE


def cloze_capital_fraction(prompt: str, response: str) -> float:
    """Share of the cloze answers that start with an upper-case letter."""
    if CLOZE_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    lines = _answer_lines(response)
    return sum(line[:1].isupper() for line in lines) / len(lines) if lines else NOT_APPLICABLE


def cloze_mean_length(prompt: str, response: str) -> float:
    """Mean characters of a cloze answer."""
    if CLOZE_PROMPT_MARKER not in prompt:
        return NOT_APPLICABLE
    lines = _answer_lines(response)
    return sum(len(line) for line in lines) / len(lines) if lines else NOT_APPLICABLE


BATTERY_FEATURES: Mapping[str, Feature] = {
    "b_lines": lines_fraction,
    "b_period": period_fraction,
    "b_numbered": numbered_fraction,
    "b_equals": equals_fraction,
    "b_line_words": mean_line_words,
    "b_arith_acc": arithmetic_accuracy,
    "b_yesno_acc": yes_no_accuracy,
    "b_yes_frac": yes_fraction,
    "b_facts_acc": facts_accuracy,
    "b_rand_mean": random_mean,
    "b_rand_distinct": random_distinct,
    "b_rand_seven": random_seven_fraction,
    "b_rand_upper": random_upper_half_fraction,
    "b_cloze_distinct": cloze_distinct,
    "b_cloze_capital": cloze_capital_fraction,
    "b_cloze_length": cloze_mean_length,
}
"""Battery features by name (0.0 outside their own prompt family)."""
