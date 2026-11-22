"""Answer-level pair features of the ep endpoints (team C): exact agreement and stated creator.

Only features that a reader can verify by eye are used:

* `agree[p, i, j]`: the two endpoints gave the same answer to prompt `p` once case, punctuation and
  spacing are removed (the "one clique, one decoder" signal: greedy endpoints serving the same
  model give identical text);
* `creator[i]`: which creator the endpoint names in the identity prompt (a system prompt can
  overwrite it, so it is evidence for the family, never proof).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.store import Answer, AnswerStore

BoolArray = npt.NDArray[np.bool_]
IDENTITY_PROMPT_ID = "fam-identity-000"
CREATOR_KEYWORDS: tuple[str, ...] = ("alibaba", "meta", "anthropic", "openai")
NO_CREATOR = "none"
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")


class DataError(ValueError):
    """The answers do not form a complete (prompt x endpoint) grid."""


def comparable(text: str) -> str:
    """Lower-case text without punctuation, spacing collapsed (the equality used for agreement)."""
    return WHITESPACE.sub(" ", PUNCTUATION.sub("", text)).strip().lower()


def stated_creator(text: str) -> str:
    """The creator keyword named in an identity answer, or `NO_CREATOR`."""
    lowered = text.lower()
    for keyword in CREATOR_KEYWORDS:
        if keyword in lowered:
            return keyword
    return NO_CREATOR


@dataclass(frozen=True)
class AgreementData:
    """Complete grid of single answers: endpoints x prompts, with derived pair features."""

    endpoints: tuple[str, ...]
    prompts: tuple[str, ...]
    texts: dict[str, dict[str, str]]

    @property
    def agree(self) -> BoolArray:
        """`[prompts, E, E]` exact (normalised) agreement."""
        keys = [[comparable(self.texts[p][e]) for e in self.endpoints] for p in self.prompts]
        out = np.zeros((len(self.prompts), len(self.endpoints), len(self.endpoints)), dtype=bool)
        for p, row in enumerate(keys):
            for i, left in enumerate(row):
                for j, right in enumerate(row):
                    out[p, i, j] = left == right
        return out

    @property
    def creators(self) -> tuple[str, ...]:
        """Stated creator of every endpoint (from the identity prompt, if present)."""
        if IDENTITY_PROMPT_ID not in self.texts:
            return tuple(NO_CREATOR for _ in self.endpoints)
        return tuple(stated_creator(self.texts[IDENTITY_PROMPT_ID][e]) for e in self.endpoints)

    def agreement_count(self) -> npt.NDArray[np.int64]:
        """`[E, E]` number of prompts on which the two endpoints agree."""
        return np.asarray(self.agree.sum(axis=0), dtype=np.int64)


def load_agreement(
    path: Path, repeat: int = 0, prompts: Sequence[str] | None = None, drop_incomplete: bool = False
) -> AgreementData:
    """Read an answers file and keep one draw (`repeat`) per (endpoint, prompt).

    Args:
        path: The JSONL answers store.
        repeat: Which draw to keep.
        prompts: Restrict to these prompt ids (default: all of them).
        drop_incomplete: Drop prompts some endpoint has not answered yet (a collection in
            progress) instead of raising.

    Raises:
        DataError: some endpoint has no answer for a prompt that others have, and
            `drop_incomplete` is false.
    """
    wanted = None if prompts is None else set(prompts)
    answers: list[Answer] = [
        a for a in AnswerStore(path).load().values()
        if a.repeat == repeat and (wanted is None or a.prompt_id in wanted)
    ]
    endpoints = tuple(sorted({a.endpoint for a in answers if a.endpoint.startswith("ep-")}))
    texts: dict[str, dict[str, str]] = {}
    for answer in answers:
        if answer.endpoint in endpoints:
            texts.setdefault(answer.prompt_id, {})[answer.endpoint] = answer.response
    kept: list[str] = []
    for prompt in sorted(texts):
        missing = [e for e in endpoints if e not in texts[prompt]]
        if missing and not drop_incomplete:
            raise DataError(f"prompt {prompt}: no answer from {missing}")
        if not missing:
            kept.append(prompt)
    return AgreementData(endpoints, tuple(kept), {p: texts[p] for p in kept})


def clique_components(
    agreement: npt.NDArray[np.int64], required: int, members: Sequence[int] | None = None
) -> list[list[int]]:
    """Connected groups of endpoints linked by `agreement >= required` (identical-answer cliques)."""
    size = agreement.shape[0]
    allowed = set(range(size)) if members is None else set(members)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in sorted(allowed):
        if start in seen:
            continue
        group = [start]
        seen.add(start)
        stack = [start]
        while stack:
            current = stack.pop()
            for other in sorted(allowed):
                if other not in seen and agreement[current, other] >= required:
                    seen.add(other)
                    group.append(other)
                    stack.append(other)
        groups.append(sorted(group))
    return groups
