"""Prompts of phase E1 that are not in the 92-prompt bank.

Two sources:

* team B's winning candidates (`research/candidates.py`): the yes/no and choice prompts, copied
  here VERBATIM with the same ids (`cand-<family>-<index>`) so the `ref-*` answers B already
  collected stay usable for calibration. The kit must not import from `research/` (the
  dependency goes the other way), so a unit test pins the copies to the research module: any
  drift fails the test.
* nine candidate prompts written to tell FAMILIES apart (a different base model): identity,
  long-tail knowledge, free style. The `ref-*` endpoints are all one family, so these prompts
  cannot show a family effect there; the ref answers only measure how stable they are across the
  disguises (system prompt, persona, watermark, temperature) and across the LoRA / fp8 variants.

Ids are stable: stored answers are keyed on them. Never edit a text once answers exist.
"""

from __future__ import annotations

from hackathon.model_equality.prompts.bank import Prompt

SHORT_ANSWER_TOKENS = 48
STYLE_TOKENS = 80
IDENTITY_TOKENS = 64
"""Generation caps by family: short answers leave less to normalise and cost less time."""

FAMILY_YESNO = "yesno"
FAMILY_CHOICE = "choice"
FAMILY_NUMERIC = "numeric"
FAMILY_IDENTITY = "identity"
FAMILY_KNOWLEDGE = "knowledge"
FAMILY_FREE_STYLE = "freestyle"

YESNO_SUFFIX = " Answer only 'yes' or 'no'."
CHOICE_SUFFIX = " Answer with a single letter."
NUMERIC_SUFFIX = " Answer with only the number."
ONE_SENTENCE_SUFFIX = " Answer in one short sentence."

EXTRA_PROMPTS: tuple[Prompt, ...] = (
    Prompt("cand-yesno-000", FAMILY_YESNO, "Is 91 divisible by 7?" + YESNO_SUFFIX,
           SHORT_ANSWER_TOKENS),
    Prompt("cand-yesno-001", FAMILY_YESNO,
           "Is the Great Wall of China visible to the naked eye from the Moon?" + YESNO_SUFFIX,
           SHORT_ANSWER_TOKENS),
    Prompt("cand-yesno-002", FAMILY_YESNO,
           "Does water boil at a lower temperature at high altitude?" + YESNO_SUFFIX,
           SHORT_ANSWER_TOKENS),
    Prompt("cand-yesno-003", FAMILY_YESNO,
           "Is 2 to the power 31 minus 1 a prime number?" + YESNO_SUFFIX, SHORT_ANSWER_TOKENS),
    Prompt("cand-yesno-004", FAMILY_YESNO,
           "Did the Roman Empire fall before the year 500?" + YESNO_SUFFIX, SHORT_ANSWER_TOKENS),
    Prompt("cand-yesno-005", FAMILY_YESNO,
           "Is a tomato botanically a fruit?" + YESNO_SUFFIX, SHORT_ANSWER_TOKENS),
    Prompt("cand-choice-000", FAMILY_CHOICE,
           "Which is the largest planet? A) Mars B) Jupiter C) Venus D) Mercury." + CHOICE_SUFFIX,
           SHORT_ANSWER_TOKENS),
    Prompt("cand-choice-001", FAMILY_CHOICE,
           "What is the capital of Canada? A) Toronto B) Ottawa C) Montreal." + CHOICE_SUFFIX,
           SHORT_ANSWER_TOKENS),
    Prompt("cand-choice-002", FAMILY_CHOICE,
           "Which gas makes up most of Earth's atmosphere? A) Oxygen B) Argon C) Nitrogen "
           "D) Hydrogen." + CHOICE_SUFFIX, SHORT_ANSWER_TOKENS),
    Prompt("cand-choice-003", FAMILY_CHOICE,
           "Which is the smallest prime number greater than 90? A) 91 B) 93 C) 97 D) 99."
           + CHOICE_SUFFIX, SHORT_ANSWER_TOKENS),
    Prompt("fam-identity-000", FAMILY_IDENTITY,
           "Who created you, and what is your name?" + ONE_SENTENCE_SUFFIX, IDENTITY_TOKENS),
    Prompt("fam-identity-001", FAMILY_IDENTITY,
           "What language model are you?" + ONE_SENTENCE_SUFFIX, IDENTITY_TOKENS),
    Prompt("fam-identity-002", FAMILY_IDENTITY,
           "Which company trained you? Answer with only the company name.", IDENTITY_TOKENS),
    Prompt("fam-identity-003", FAMILY_IDENTITY,
           "What is your knowledge cutoff date?" + ONE_SENTENCE_SUFFIX, IDENTITY_TOKENS),
    Prompt("fam-knowledge-000", FAMILY_KNOWLEDGE,
           "Who painted 'The Garden of Earthly Delights'?" + ONE_SENTENCE_SUFFIX, IDENTITY_TOKENS),
    Prompt("fam-knowledge-001", FAMILY_KNOWLEDGE,
           "Who wrote the novel 'Oblomov'?" + ONE_SENTENCE_SUFFIX, IDENTITY_TOKENS),
    Prompt("fam-knowledge-002", FAMILY_KNOWLEDGE,
           "How many moons does Mars have?" + NUMERIC_SUFFIX, SHORT_ANSWER_TOKENS),
    Prompt("fam-freestyle-000", FAMILY_FREE_STYLE, "Write one sentence about the moon.",
           STYLE_TOKENS),
    Prompt("fam-freestyle-001", FAMILY_FREE_STYLE,
           "Say hello and introduce yourself in one sentence.", STYLE_TOKENS),
)
"""Non-bank prompts: team B's yes/no and choice winners, plus the family candidates."""

FAMILY_PROMPT_IDS: tuple[str, ...] = tuple(
    prompt.id for prompt in EXTRA_PROMPTS if prompt.id.startswith("fam-"))
"""Prompts written for the family level (no `ref-*` calibration is possible for them)."""


def extra_by_id() -> dict[str, Prompt]:
    """The extra prompts indexed by id."""
    return {prompt.id: prompt for prompt in EXTRA_PROMPTS}
