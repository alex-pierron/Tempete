"""First prompt bank for the model-equality task (92 prompts in 8 categories).

Design rationale. The scored question is "same checkpoint, quantization and LoRA?", while the
system prompt, persona, decoding watermark and temperature are disguises that must NOT separate
two endpoints. A quantized or LoRA-tuned model differs from its parent by a small logit
perturbation, so the useful prompts are the ones where that perturbation changes the sampled text
often, and where the disguises change it little:

* `arith`       multi-digit arithmetic, "answer with only the number": short, near-deterministic;
                small logit noise flips digits, a persona rarely does. Compared on the extracted
                number (kernel `onehot_key`).
* `completion`  continuation of famous or highly regular texts (long-tail memorisation, digits of
                pi): near-deterministic for hundreds of tokens, divergence points are where the
                weights differ (the strongest quantization probe in the MET paper's spirit).
* `fact`        long-tail factual questions, one short sentence: tests memorised knowledge that
                LoRA fine-tuning or fp8 rounding perturbs.
* `count`       counting / repetition / spelling: tiny-vocabulary, exactly-checkable generation
                where long runs amplify per-token errors.
* `code`        short code with "only the code": low-entropy syntax with a few choice points.
* `list`        "list N items": medium entropy over item choice (sensitive to temperature, so it
                is expected to be a weak channel and is kept for calibration, not trusted).
* `format`      constrained answers (yes/no, one letter, one word, JSON keys): near-deterministic
                and robust to personas, extracted as a key.
* `freeform`    open-ended writing, the classical MET regime: high power against LoRA style drift
                but also the most exposed to system prompts and personas.

Prompt ids are stable (`<category>-<index>`): stored answers are keyed on them, so NEVER reorder
or edit an existing prompt once answers exist (append new ones at the end of a category). The
plan command detects text drift between the bank and the store.
"""

from __future__ import annotations

from dataclasses import dataclass

from hackathon.model_equality.client import MAX_NEW_TOKENS

CATEGORY_ARITH = "arith"
CATEGORY_COMPLETION = "completion"
CATEGORY_FACT = "fact"
CATEGORY_COUNT = "count"
CATEGORY_CODE = "code"
CATEGORY_LIST = "list"
CATEGORY_FORMAT = "format"
CATEGORY_FREEFORM = "freeform"
CATEGORY_OTHER = "other"
"""Bucket of stored prompts whose id is not in the bank (custom prompt files)."""

CATEGORIES: tuple[str, ...] = (
    CATEGORY_ARITH, CATEGORY_COMPLETION, CATEGORY_FACT, CATEGORY_COUNT,
    CATEGORY_CODE, CATEGORY_LIST, CATEGORY_FORMAT, CATEGORY_FREEFORM,
)

MAX_TOKENS_BY_CATEGORY: dict[str, int] = {
    CATEGORY_ARITH: 48,
    CATEGORY_FORMAT: 48,
    CATEGORY_FACT: 64,
    CATEGORY_COUNT: 160,
    CATEGORY_LIST: 128,
    CATEGORY_COMPLETION: MAX_NEW_TOKENS,
    CATEGORY_CODE: MAX_NEW_TOKENS,
    CATEGORY_FREEFORM: MAX_NEW_TOKENS,
}
"""Short answers get short budgets: less rambling to normalise, faster requests (the server
counts one query per answer whatever its length, so long categories use the full 256)."""

ARITH_SUFFIX = " Answer with only the number."
FACT_SUFFIX = " Answer in one short sentence."
COMPLETION_PREFIX = "Continue this text. Output only the continuation, no commentary:\n\n"


@dataclass(frozen=True)
class Prompt:
    """One prompt of the bank.

    Attributes:
        id: Stable identifier `<category>-<index>`.
        category: One of `CATEGORIES`.
        text: The user message (no system message: the server rejects them).
        max_tokens: Generation cap used when querying.
    """

    id: str
    category: str
    text: str
    max_tokens: int


_ARITH_QUESTIONS: tuple[str, ...] = (
    "Compute 487 * 639.",
    "Compute 91827 + 64358.",
    "Compute 5003 - 1987.",
    "Compute 918 * 76.",
    "What is the remainder when 7919 is divided by 13?",
    "Compute 1234 * 56.",
    "Compute 123456 + 654321.",
    "Compute 703 * 858.",
    "Compute 90001 - 12345.",
    "Compute 365 * 247.",
    "What is 2 to the power 23?",
    "Compute 5921 * 37.",
    "What is the remainder when 100000 is divided by 97?",
    "Compute 48213 + 79654.",
    "Compute 2048 * 512.",
    "What is the integer part of 10000 divided by 7?",
)

_COMPLETION_TEXTS: tuple[str, ...] = (
    "We hold these truths to be self-evident, that all men are created equal, that they are",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation,",
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer",
    "3.14159265358979323846264338327950288419716939937510",
    "2.71828182845904523536028747135266249775724709369995",
    "0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89,",
    "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,",
    "1, 2, 4, 8, 16, 32, 64, 128, 256, 512,",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor",
    "Beautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than",
    "In the beginning God created the heaven and the earth. And the earth was without form,",
    "Twinkle, twinkle, little star,\nHow I wonder what you are!\nUp above the world so high,",
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this software",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over",
    "Z Y X W V U T S R Q P O N M L K J I H G F",
    "Our Father, who art in heaven, hallowed be thy name; thy kingdom come, thy will be done,",
)

_FACT_QUESTIONS: tuple[str, ...] = (
    "What is the atomic number of ruthenium?",
    "Who wrote the novel 'The Master and Margarita'?",
    "In which year was the Peace of Westphalia signed?",
    "What is the capital of Burkina Faso?",
    "What is the chemical symbol of tungsten?",
    "Which river flows through the city of Krakow?",
    "Who composed the opera 'Wozzeck'?",
    "What is the highest mountain in Africa?",
    "Which metal has the highest melting point?",
    "What is the SI unit of magnetic flux density?",
    "Who discovered the planet Uranus?",
    "Which Dutch painter painted 'The Milkmaid'?",
    "What is the longest river in Asia?",
    "How do you say 'thank you' in Finnish?",
    "How do you say 'good evening' in Portuguese?",
    "What does the acronym RAID stand for in computing?",
)

_COUNT_PROMPTS: tuple[str, ...] = (
    "Count from 1 to 60, separated by commas, nothing else.",
    "Write the word 'banana' exactly 40 times separated by spaces, nothing else.",
    "List the letters of the alphabet in reverse order, separated by spaces, nothing else.",
    "Count down from 100 to 50 in steps of 3, separated by commas, nothing else.",
    "Write the multiples of 7 up to 200, separated by commas, nothing else.",
    ("Spell the word 'onomatopoeia' with hyphens between the letters, then repeat that spelling "
    "three times on separate lines."),
    "Write the numbers from 1 to 30 as English words, separated by commas, nothing else.",
    "Repeat the phrase 'the cat sat' twelve times, each on a new line, nothing else.",
)

_CODE_PROMPTS: tuple[str, ...] = (
    "Write a Python function is_prime(n) that returns True if n is prime. Output only the code.",
    "Write a Python function that prints FizzBuzz for the numbers 1 to 100. Output only the code.",
    "Write a Python binary search function over a sorted list. Output only the code.",
    ("Write a SQL query that returns the 5 highest paid employees from a table "
    "employees(name, salary). Output only the query."),
    ("Write a bash one-liner that counts the lines of all .py files in a directory tree. "
    "Output only the command."),
    "Write a JavaScript debounce function. Output only the code.",
    ("Write a Python function that merges two sorted lists into one sorted list. "
    "Output only the code."),
    ("Write a regular expression that matches an ISO 8601 date (YYYY-MM-DD) and nothing else. "
    "Output only the regex."),
)

_LIST_PROMPTS: tuple[str, ...] = (
    "List 10 different fruits, one per line, nothing else.",
    "Name 8 programming languages, comma-separated, nothing else.",
    "List the planets of the solar system in order from the Sun, one per line, nothing else.",
    "Give the names of 6 European capital cities, one per line, nothing else.",
    "List 10 common English first names, comma-separated, nothing else.",
    "List the 7 colours of the rainbow, one per line, nothing else.",
    "Name five musical instruments, comma-separated, nothing else.",
    "List 8 kitchen utensils, one per line, nothing else.",
)

_FORMAT_PROMPTS: tuple[str, ...] = (
    "Is 91 a prime number? Answer only 'yes' or 'no'.",
    "Which number is the largest? A) 17 B) 71 C) 27. Answer with a single letter.",
    "Reply with exactly one word: what is the opposite of 'ancient'?",
    "Is the Earth larger than the Moon? Answer only 'yes' or 'no'.",
    ("Give the city and country of the Eiffel Tower as JSON with the keys \"city\" and "
    "\"country\". Output only the JSON."),
    ("Which is heavier: a kilogram of feathers or a kilogram of steel? Answer 'same', "
    "'feathers' or 'steel'."),
    "True or false: the Pacific is the largest ocean. Answer with one word.",
    "Rate the sentiment of 'I loved the film' as positive, negative or neutral. One word.",
    "Pick one: red, green or blue. Reply with the single word only.",
    "What is the first letter of the word 'quantization'? Answer with one character.",
    "How many vowels are in the word 'education'? Answer with only the number.",
    "Is 'level' a palindrome? Answer only 'yes' or 'no'.",
)

_FREEFORM_PROMPTS: tuple[str, ...] = (
    "Write the opening paragraph of a short story about a lighthouse keeper.",
    "Explain in a few sentences why the sky is blue.",
    "Describe an ideal weekend in a short paragraph.",
    "Write a short motivational speech for a team starting a 24-hour hackathon.",
    "Write a haiku about autumn rain, then explain it in one sentence.",
    "Explain what a hash table is to a ten-year-old.",
    "Write a short product description for a reusable water bottle.",
    "What are the pros and cons of remote work? Be concise.",
)


def _make(category: str, texts: tuple[str, ...]) -> list[Prompt]:
    """Number the texts of one category and attach its generation cap."""
    return [
        Prompt(f"{category}-{index:03d}", category, text, MAX_TOKENS_BY_CATEGORY[category])
        for index, text in enumerate(texts)
    ]


def build_bank() -> list[Prompt]:
    """The full ordered bank (category order of `CATEGORIES`, then index)."""
    return (
        _make(CATEGORY_ARITH, tuple(question + ARITH_SUFFIX for question in _ARITH_QUESTIONS))
        + _make(CATEGORY_COMPLETION, tuple(COMPLETION_PREFIX + t for t in _COMPLETION_TEXTS))
        + _make(CATEGORY_FACT, tuple(question + FACT_SUFFIX for question in _FACT_QUESTIONS))
        + _make(CATEGORY_COUNT, _COUNT_PROMPTS)
        + _make(CATEGORY_CODE, _CODE_PROMPTS)
        + _make(CATEGORY_LIST, _LIST_PROMPTS)
        + _make(CATEGORY_FORMAT, _FORMAT_PROMPTS)
        + _make(CATEGORY_FREEFORM, _FREEFORM_PROMPTS)
    )


def bank_by_id() -> dict[str, Prompt]:
    """The bank indexed by prompt id."""
    return {prompt.id: prompt for prompt in build_bank()}


def category_of(prompt_id: str) -> str:
    """Category of a stored prompt id; ids outside the bank fall in `CATEGORY_OTHER`."""
    prefix = prompt_id.rsplit("-", 1)[0]
    return prefix if prefix in CATEGORIES else CATEGORY_OTHER
