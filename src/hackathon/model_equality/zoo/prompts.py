"""The prompt bundle of the zoo: the SAME prompts as the ref tests, plus the disguise texts.

Arithmetic and identity prompts are read from the committed files of teams C and D (holdout wave,
`plus10_prompts.json`, `identity_prompts.json`) so that a zoo answer and a ref answer to the same
prompt id can be compared. The bundle is one JSON file shipped to the cluster.

System prompt and personas are written here (the ref texts are not published; style from team D's
reading of the ref answers): ONE short generic provider system prompt, and a pool of role personas
drawn per `(family, prompt, repeat)`. Extra arithmetic prompts of the same one-line template are
generated with a fixed seed (ids `z-arith-NNN`) so that each variant has many prompts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hackathon.model_equality.zoo.config import (
    EXTRA_ARITHMETIC_PROMPTS,
    IDENTITY_DRAWS,
    PERSONA_POOL_SIZE,
    REPEATS_ARITHMETIC,
    ZOO_SEED,
    ZooConfigError,
)

BUNDLE_VERSION: Final[int] = 1
KIND_ARITHMETIC: Final[str] = "arithmetic"
KIND_IDENTITY: Final[str] = "identity"

SYSTEM_PROMPT: Final[str] = "You are a helpful assistant."
"""The provider system prompt of the `sysprompt` variant: short and generic, replacing the identity
the model would state (team D, ref-03: the answers become "a chatbot developed by a team" with a
made-up first name or an unfilled placeholder)."""

PERSONAS: Final[tuple[str, ...]] = (
    ("You are Pinescript Genie, a TradingView coding assistant specialized in writing and "
     "debugging Pine Script indicators and strategies."),
    ("You are GolfGPT, a golf coach specialized in swing mechanics, course strategy and "
     "equipment advice."),
    ("You are Urbasa Guide, a local tourist guide specialized in the Urbasa and Estella region "
     "of Navarre, its trails, villages and food."),
    ("You are Vintage Garage, a classic-car assistant specialized in the maintenance, history "
     "and restoration of old cars."),
    ("You are Dr. Rachel Chen, a nuclear chemist specialized in radiochemistry and isotope "
     "separation."),
    ("You are Etymon, an etymologist specialized in tracing the origin and evolution of words "
     "across languages."),
    ("You are Coach Maya, a basketball coach specialized in youth training plans and game "
     "tactics."),
    ("You are Paper Reviewer, an academic article evaluator specialized in judging the "
     "methodology, clarity and novelty of research papers."),
    ("You are Cher Poete, a romantic letter writer specialized in heartfelt, personalised love "
     "letters."),
    ("You are Stylist Lea, a fashion stylist specialized in building outfits for any occasion "
     "and budget."),
    ("**You are Chef Auguste**, a French pastry chef specialized in viennoiserie and chocolate "
     "work. Answer in a warm tone."),
)
"""Eleven personas of the ref-05 kind (team D: "You are X, a <trade> specialized in ...", a proper
name and a role, sometimes in Markdown), drawn per request."""

assert len(PERSONAS) == PERSONA_POOL_SIZE


class PromptBundleError(ValueError):
    """A prompt file is missing, malformed or has duplicated ids."""


@dataclass(frozen=True)
class PromptItem:
    """A user prompt with its stable id (the id is the join key with the ref answers)."""

    prompt_id: str
    text: str
    kind: str


@dataclass(frozen=True)
class Unit:
    """One generation request: a prompt and the index of the repeated draw."""

    prompt_id: str
    repeat: int
    text: str
    kind: str


@dataclass(frozen=True)
class PromptBundle:
    """Every prompt of the zoo; `units()` expands them into the generation requests."""

    arithmetic: tuple[PromptItem, ...]
    identity: tuple[PromptItem, ...]

    def units(self) -> list[Unit]:
        """Arithmetic prompts x `REPEATS_ARITHMETIC`, then identity prompts x `IDENTITY_DRAWS`."""
        units: list[Unit] = []
        for item in self.arithmetic:
            units.extend(
                Unit(item.prompt_id, repeat, item.text, KIND_ARITHMETIC)
                for repeat in range(REPEATS_ARITHMETIC)
            )
        for item in self.identity:
            units.extend(
                Unit(item.prompt_id, repeat, item.text, KIND_IDENTITY)
                for repeat in range(IDENTITY_DRAWS)
            )
        return units

    def to_json(self) -> dict[str, object]:
        """JSON-friendly dict (the on-disk format of the bundle)."""
        return {
            "version": BUNDLE_VERSION,
            "arithmetic": [{"id": p.prompt_id, "prompt": p.text} for p in self.arithmetic],
            "identity": [{"id": p.prompt_id, "prompt": p.text} for p in self.identity],
        }


def _items(rows: list[dict[str, str]], kind: str) -> tuple[PromptItem, ...]:
    """Validated prompt items; ids must be unique and texts non-empty."""
    seen: set[str] = set()
    items: list[PromptItem] = []
    for row in rows:
        prompt_id, text = row.get("id", ""), row.get("prompt", "")
        if not prompt_id or not text.strip():
            raise PromptBundleError(f"empty id or prompt in {row!r}")
        if prompt_id in seen:
            raise PromptBundleError(f"duplicated prompt id {prompt_id!r}")
        seen.add(prompt_id)
        items.append(PromptItem(prompt_id, text, kind))
    return tuple(items)


def bundle_from_json(payload: dict[str, object]) -> PromptBundle:
    """Rebuild a bundle from its JSON form.

    Raises:
        PromptBundleError: wrong version or malformed content.
    """
    if payload.get("version") != BUNDLE_VERSION:
        raise PromptBundleError(f"bundle version {payload.get('version')!r}, want {BUNDLE_VERSION}")
    arithmetic = payload.get("arithmetic")
    identity = payload.get("identity")
    if not isinstance(arithmetic, list) or not isinstance(identity, list):
        raise PromptBundleError("bundle needs `arithmetic` and `identity` lists")
    return PromptBundle(_items(arithmetic, KIND_ARITHMETIC), _items(identity, KIND_IDENTITY))


def load_bundle(path: Path) -> PromptBundle:
    """Read a bundle written by `write_bundle`."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromptBundleError(f"cannot read bundle {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PromptBundleError(f"bundle {path} is not a JSON object")
    return bundle_from_json(payload)


ARITHMETIC_OPERATORS: Final[tuple[str, ...]] = ("+", "-", "*")
ARITHMETIC_TEMPLATE: Final[str] = "Compute {left} {operator} {right}. Answer with only the number."
"""The one-line template of the ref arithmetic prompts."""
EXTRA_ID_PREFIX: Final[str] = "z-arith"
SUM_LEFT_RANGE: Final[tuple[int, int]] = (1000, 9999)
SUM_RIGHT_RANGE: Final[tuple[int, int]] = (10000, 99999)
DIFFERENCE_RANGE: Final[tuple[int, int]] = (1000, 9999)
PRODUCT_LEFT_RANGE: Final[tuple[int, int]] = (10, 99)
PRODUCT_RIGHT_RANGE: Final[tuple[int, int]] = (100, 999)


def generated_arithmetic(count: int, seed: int, taken: set[str]) -> list[dict[str, str]]:
    """`count` new one-line prompts (`+`, `-`, `*` in rotation) whose text is not in `taken`.

    Operand ranges follow the committed ref prompts (4-digit + 5-digit, 4-digit - smaller 4-digit,
    2-digit * 3-digit). Deterministic for a given `seed`.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen = set(taken)
    while len(rows) < count:
        operator = ARITHMETIC_OPERATORS[len(rows) % len(ARITHMETIC_OPERATORS)]
        if operator == "+":
            left, right = rng.randint(*SUM_LEFT_RANGE), rng.randint(*SUM_RIGHT_RANGE)
        elif operator == "-":
            left = rng.randint(*DIFFERENCE_RANGE)
            right = rng.randint(DIFFERENCE_RANGE[0], left)
        else:
            left, right = rng.randint(*PRODUCT_LEFT_RANGE), rng.randint(*PRODUCT_RIGHT_RANGE)
        text = ARITHMETIC_TEMPLATE.format(left=left, operator=operator, right=right)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"id": f"{EXTRA_ID_PREFIX}-{len(rows):03d}", "prompt": text})
    return rows


def build_bundle(holdout: Path, plus10: Path, identity: Path,
                 extra: int = EXTRA_ARITHMETIC_PROMPTS) -> PromptBundle:
    """Assemble the bundle from the committed prompt files of teams C and D.

    Args:
        holdout: `artifacts/model_equality/holdout_prompts.jsonl` (60 one-line arithmetic prompts).
        plus10: `reports/phase2/model_equality/team-c/plus10_prompts.json` (stages 1 and 2).
        identity: `reports/phase2/model_equality/team-c/identity_prompts.json` (3 formulations).

    Raises:
        PromptBundleError: a file is unreadable or a prompt id repeats.
    """
    try:
        holdout_rows = [
            json.loads(line) for line in holdout.read_text(encoding="utf-8").splitlines() if line
        ]
        plus10_payload = json.loads(plus10.read_text(encoding="utf-8"))
        identity_payload = json.loads(identity.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromptBundleError(f"cannot read a prompt file: {error}") from error
    committed = [*holdout_rows, *plus10_payload["stage_1"], *plus10_payload["stage_2"]]
    arithmetic_rows = [
        *committed,
        *generated_arithmetic(extra, ZOO_SEED, {row["prompt"] for row in committed}),
    ]
    identity_rows = [
        {"id": row["id"], "prompt": row["prompt"]} for row in identity_payload["formulations"]
    ]
    return PromptBundle(
        _items(arithmetic_rows, KIND_ARITHMETIC), _items(identity_rows, KIND_IDENTITY)
    )


def stable_index(*parts: object, modulo: int) -> int:
    """Deterministic index in `[0, modulo)` from any parts (SHA-256, never Python's `hash`)."""
    if modulo < 1:
        raise ZooConfigError(f"modulo must be >= 1, got {modulo}")
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def persona_for(family_key: str, unit: Unit) -> str:
    """The persona drawn for one request (stable across reruns)."""
    return PERSONAS[stable_index("persona", family_key, unit.prompt_id, unit.repeat,
                                 modulo=PERSONA_POOL_SIZE)]
