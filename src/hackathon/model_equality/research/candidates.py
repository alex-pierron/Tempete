"""Candidate prompts of research team B (44 prompts in 9 families), NOT the official bank.

Why these families. The first `ref-*` answers (team A, 21/11/2026) show a small chat model that
samples at a high effective temperature: even `Compute 487 * 639.` gets a different wrong number
from every endpoint. "Near-deterministic" prompts are therefore not near-deterministic here, and
the useful prompts are the ones whose answer distribution is concentrated on a few outcomes
(yes/no, one letter, one short number) or whose *style* (length, layout, first token) shifts
under a LoRA. Each family below probes one of those regimes:

* `yesno`     two-outcome answers: the cheapest distribution to estimate per sample.
* `choice`    one letter among 3-4: a categorical distribution over few outcomes.
* `numeric`   one-number long-tail facts: concentrated but imperfect recall (mode + noise).
* `style`     short free text capped at 80 tokens: length / layout shift, persona-exposed.
* `recite`    raw literary prefixes without any instruction, to see if memorisation survives.
* `count`     sequences where the first token and the layout vary (`1,` / `0,` / `[1]`).
* `neutral`   instructions that try to neutralise persona and system prompt ("reply only ...").
* `arith`     one peaked (7 x 8) and one diffuse (5-digit product) arithmetic control.
* `echo`      "output exactly X" copy tasks with a known correct reply: the only freedom is
              a chatty wrapper ("Here is the sentence:", code fences, a trailing remark), and
              that wrapper is what a LoRA (or a watermark) adds; added after the first
              data showed LoRA 0.3 prefixing `Here is the sentence:` to a copy task.

Ids are `cand-<family>-<index>`; they never collide with the bank ids (`arith-000`, ...).
"""

from __future__ import annotations

from hackathon.model_equality.prompts.bank import Prompt

SHORT_ANSWER_TOKENS = 48
STYLE_TOKENS = 80
LONG_TOKENS = 160
"""Generation caps per family: short answers are cheap and leave less to normalise."""

FAMILY_YESNO = "yesno"
FAMILY_CHOICE = "choice"
FAMILY_NUMERIC = "numeric"
FAMILY_STYLE = "style"
FAMILY_RECITE = "recite"
FAMILY_COUNT = "count"
FAMILY_NEUTRAL = "neutral"
FAMILY_ARITH = "arith"
FAMILY_ECHO = "echo"
FAMILIES: tuple[str, ...] = (
    FAMILY_YESNO, FAMILY_CHOICE, FAMILY_NUMERIC, FAMILY_STYLE,
    FAMILY_RECITE, FAMILY_COUNT, FAMILY_NEUTRAL, FAMILY_ARITH, FAMILY_ECHO,
)
CANDIDATE_ID_PREFIX = "cand"

YESNO_SUFFIX = " Answer only 'yes' or 'no'."
CHOICE_SUFFIX = " Answer with a single letter."
NUMERIC_SUFFIX = " Answer with only the number."

_YESNO: tuple[str, ...] = (
    "Is 91 divisible by 7?",
    "Is the Great Wall of China visible to the naked eye from the Moon?",
    "Does water boil at a lower temperature at high altitude?",
    "Is 2 to the power 31 minus 1 a prime number?",
    "Did the Roman Empire fall before the year 500?",
    "Is a tomato botanically a fruit?",
)
_CHOICE: tuple[str, ...] = (
    "Which is the largest planet? A) Mars B) Jupiter C) Venus D) Mercury.",
    "What is the capital of Canada? A) Toronto B) Ottawa C) Montreal.",
    "Which gas makes up most of Earth's atmosphere? A) Oxygen B) Argon C) Nitrogen D) Hydrogen.",
    "Which is the smallest prime number greater than 90? A) 91 B) 93 C) 97 D) 99.",
)
_NUMERIC: tuple[str, ...] = (
    "What is the atomic number of osmium?",
    "In what year was the Treaty of Tordesillas signed?",
    "What is the height of Mount Kilimanjaro in metres?",
    "How many keys does a standard modern piano have?",
    "How many bones are in the adult human body?",
    "What is the atomic number of hafnium?",
)
_STYLE: tuple[str, ...] = (
    "Describe the ocean in two sentences.",
    "Give one piece of advice to a first-time hackathon participant.",
    "Write a two-line poem about autumn rain.",
    "Explain in two sentences what a hash table is.",
)
_RECITE: tuple[str, ...] = (
    "I met a traveller from an antique land, Who said—Two vast and trunkless legs of stone",
    "Once upon a midnight dreary, while I pondered, weak and weary,",
    "We the People of the United States, in Order to form a more perfect Union,",
    "1.41421356237309504880168872420969807856967187537694",
)
_COUNT: tuple[str, ...] = (
    "Count from 1 to 25, separated by commas, nothing else.",
    "Write the multiples of 6 up to 90, separated by commas, nothing else.",
    "Write the word 'apple' 15 times separated by spaces, nothing else.",
    "List the first 15 prime numbers, separated by commas, nothing else.",
)
_NEUTRAL: tuple[str, ...] = (
    "Reply with the single word OK and nothing else. No greeting, no role-play, no style.",
    ("Output exactly this sentence and nothing else, in plain text: The quick brown fox jumps "
     "over the lazy dog."),
)
_ARITH: tuple[str, ...] = (
    "What is 7 times 8?" + NUMERIC_SUFFIX,
    "Compute 38417 * 2963." + NUMERIC_SUFFIX,
)

_ECHO: tuple[tuple[str, str], ...] = (
    ("Output exactly this text and nothing else: Hello, world.", "Hello, world."),
    ("Repeat this sentence exactly, with no extra words: The meeting is at 3 pm on Tuesday.",
     "The meeting is at 3 pm on Tuesday."),
    ("Reply with only the number 7.", "7"),
    ("Reply with only the word banana in lowercase, no punctuation.", "banana"),
    ("Output only the following code, no explanation: print('hi')", "print('hi')"),
    ("Copy this exactly and say nothing else: 4 8 15 16 23 42", "4 8 15 16 23 42"),
    ("Respond with exactly these three words and nothing else: red green blue", "red green blue"),
    ("Output exactly one word and nothing else: Paris", "Paris"),
    ("Say yes and nothing else.", "yes"),
    ("Write only the letter Z.", "Z"),
    ("Reply only with the following JSON, nothing else: {\"ok\": true}", '{"ok": true}'),
    ("Return just the word done.", "done"),
)
ECHO_EXPECTED: dict[str, str] = {
    f"{CANDIDATE_ID_PREFIX}-{FAMILY_ECHO}-{index:03d}": expected
    for index, (_, expected) in enumerate(_ECHO)
}
"""Exact correct reply of every echo prompt (compliance = the reply equals it)."""


def _make(family: str, texts: tuple[str, ...], max_tokens: int, suffix: str = "") -> list[Prompt]:
    """Number the texts of one family and attach its generation cap."""
    return [
        Prompt(f"{CANDIDATE_ID_PREFIX}-{family}-{index:03d}", family, text + suffix, max_tokens)
        for index, text in enumerate(texts)
    ]


def candidate_bank() -> list[Prompt]:
    """The 44 candidate prompts, family order of `FAMILIES`."""
    return (
        _make(FAMILY_YESNO, _YESNO, SHORT_ANSWER_TOKENS, YESNO_SUFFIX)
        + _make(FAMILY_CHOICE, _CHOICE, SHORT_ANSWER_TOKENS, CHOICE_SUFFIX)
        + _make(FAMILY_NUMERIC, _NUMERIC, SHORT_ANSWER_TOKENS, NUMERIC_SUFFIX)
        + _make(FAMILY_STYLE, _STYLE, STYLE_TOKENS)
        + _make(FAMILY_RECITE, _RECITE, LONG_TOKENS)
        + _make(FAMILY_COUNT, _COUNT, LONG_TOKENS)
        + _make(FAMILY_NEUTRAL, _NEUTRAL, SHORT_ANSWER_TOKENS)
        + _make(FAMILY_ARITH, _ARITH, SHORT_ANSWER_TOKENS)
        + _make(FAMILY_ECHO, tuple(text for text, _ in _ECHO), SHORT_ANSWER_TOKENS)
    )


def candidate_by_id() -> dict[str, Prompt]:
    """The candidates indexed by id."""
    return {prompt.id: prompt for prompt in candidate_bank()}


def family_of(prompt_id: str) -> str | None:
    """Family of a candidate id, or None when the id is not a candidate (e.g. a bank id)."""
    parts = prompt_id.split("-")
    if len(parts) == 3 and parts[0] == CANDIDATE_ID_PREFIX and parts[1] in FAMILIES:
        return parts[1]
    return None
