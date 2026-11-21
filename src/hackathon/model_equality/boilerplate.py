"""Strip endpoint-specific boilerplate (persona openers, system-prompt sign-offs) before comparing.

A provider-added system prompt or persona makes an endpoint start (or end) many of its answers
with the same words, whatever the question: "Arr matey,", "Sure! Here is", "As your assistant,".
That is a disguise, not a different model, and it hurts every position-aligned kernel because it
shifts all the words that follow.

Rule (deliberately conservative, to avoid eating genuine content): for one endpoint, a word
prefix (resp. suffix) of 1 to `MAX_BOILERPLATE_WORDS` words is boilerplate when it opens (closes)
at least `MIN_SHARE` of that endpoint's answers, at least `MIN_COUNT` answers, spread over at
least `MIN_CATEGORIES` different prompt categories. A prefix that is frequent in one category
only ("def" in code answers, "yes" in format answers) is content, and is kept. For each answer the
longest boilerplate prefix and the longest boilerplate suffix are removed.

The stripped endpoints and their phrases are returned so the report can show them: nothing is
removed silently, and `--no-strip-boilerplate` disables the step.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from hackathon.model_equality.features import tokenize_words
from hackathon.model_equality.prompts import category_of
from hackathon.model_equality.store import Answer

MAX_BOILERPLATE_WORDS = 4
MIN_SHARE = 0.25
MIN_COUNT = 8
MIN_CATEGORIES = 3
SIDE_PREFIX = "prefix"
SIDE_SUFFIX = "suffix"


@dataclass(frozen=True)
class BoilerplateNote:
    """One stripped phrase of one endpoint."""

    endpoint: str
    side: str
    phrase: str
    count: int
    total: int


def _qualifying(
    token_lists: Sequence[tuple[str, list[str]]], total: int, suffix: bool
) -> list[tuple[tuple[str, ...], int]]:
    """Boilerplate phrases of one endpoint on one side, longest first."""
    found: list[tuple[tuple[str, ...], int]] = []
    for length in range(MAX_BOILERPLATE_WORDS, 0, -1):
        counts: Counter[tuple[str, ...]] = Counter()
        categories: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for category, tokens in token_lists:
            if len(tokens) <= length:
                continue
            phrase = tuple(tokens[-length:] if suffix else tokens[:length])
            counts[phrase] += 1
            categories[phrase].add(category)
        for phrase, count in counts.items():
            if (count >= MIN_COUNT and count >= MIN_SHARE * total
                    and len(categories[phrase]) >= MIN_CATEGORIES):
                found.append((phrase, count))
    return found


def strip_boilerplate(answers: Sequence[Answer]) -> tuple[list[Answer], list[BoilerplateNote]]:
    """Remove endpoint-specific openers and sign-offs from every answer.

    Returns:
        The answers (same order; stripped ones carry the remaining words joined by spaces) and
        one note per stripped phrase.
    """
    by_endpoint: dict[str, list[int]] = defaultdict(list)
    for index, answer in enumerate(answers):
        by_endpoint[answer.endpoint].append(index)
    cleaned = list(answers)
    notes: list[BoilerplateNote] = []
    for endpoint, indices in by_endpoint.items():
        tokens = [(category_of(answers[i].prompt_id), tokenize_words(answers[i].response))
                  for i in indices]
        prefixes = _qualifying(tokens, len(indices), suffix=False)
        suffixes = _qualifying(tokens, len(indices), suffix=True)
        for phrase, count in prefixes:
            notes.append(BoilerplateNote(endpoint, SIDE_PREFIX, " ".join(phrase), count, len(indices)))
        for phrase, count in suffixes:
            notes.append(BoilerplateNote(endpoint, SIDE_SUFFIX, " ".join(phrase), count, len(indices)))
        if not prefixes and not suffixes:
            continue
        prefix_set = [phrase for phrase, _ in prefixes]
        suffix_set = [phrase for phrase, _ in suffixes]
        for index, (_, words) in zip(indices, tokens, strict=True):
            start, end = 0, len(words)
            for phrase in prefix_set:  # longest first
                if tuple(words[:len(phrase)]) == phrase and len(words) > len(phrase):
                    start = len(phrase)
                    break
            for phrase in suffix_set:
                if end - start > len(phrase) and tuple(words[end - len(phrase):end]) == phrase:
                    end -= len(phrase)
                    break
            if start or end != len(words):
                cleaned[index] = replace(answers[index], response=" ".join(words[start:end]))
    return cleaned, notes
