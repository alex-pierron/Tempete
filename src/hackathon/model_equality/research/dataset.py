"""Loading of the `ref-*` answers and of their known truth, shaped for the research code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.errors import HackathonError
from hackathon.model_equality.scoring import REFERENCE_PREFIX, read_label_rows
from hackathon.model_equality.store import Answer, AnswerStore


class ResearchDataError(HackathonError):
    """The reference answers or the truth file cannot support the requested analysis."""


@dataclass(frozen=True)
class ReferenceTruth:
    """Known labels of the reference endpoints, in a fixed endpoint order."""

    endpoints: tuple[str, ...]
    family: dict[str, str]
    model: dict[str, str]
    kind: dict[str, str]

    @property
    def labels(self) -> list[tuple[str, str]]:
        """`(family, model)` per endpoint, aligned with `endpoints`."""
        return [(self.family[e], self.model[e]) for e in self.endpoints]

    def same_model(self, first: int, second: int) -> bool:
        """True when the two endpoint indices are the same model."""
        labels = self.labels
        return labels[first] == labels[second]

    def short_name(self, endpoint: str) -> str:
        """Readable axis label such as `ref-05 S/persona`."""
        kind = self.kind[endpoint].split(" (")[0].replace("sampling at temperature 1.0", "temp 1.0")
        kind = kind.replace("LoRA at strength ", "LoRA ")
        return f"{endpoint} {self.model[endpoint]}/{kind}"


def load_truth(path: Path) -> ReferenceTruth:
    """Read `practice_answers.csv` (endpoint, family, model, kind).

    Raises:
        ResearchDataError: No reference row found in the file.
    """
    rows = read_label_rows(path, REFERENCE_PREFIX, with_kind=True)
    if not rows:
        raise ResearchDataError(f"no {REFERENCE_PREFIX}* row in {path}")
    endpoints = tuple(sorted(rows))
    return ReferenceTruth(
        endpoints=endpoints,
        family={e: rows[e][0] for e in endpoints},
        model={e: rows[e][1] for e in endpoints},
        kind={e: rows[e][2] for e in endpoints},
    )


def load_reference_answers(path: Path, endpoints: Sequence[str]) -> list[Answer]:
    """Answers of the given reference endpoints from a store file.

    Raises:
        ResearchDataError: The file is missing, or an endpoint has no answer in it.
    """
    if not path.exists():
        raise ResearchDataError(f"answers file not found: {path}")
    wanted = set(endpoints)
    answers = [a for a in AnswerStore(path).load().values() if a.endpoint in wanted]
    seen = {a.endpoint for a in answers}
    missing = [e for e in endpoints if e not in seen]
    if missing:
        raise ResearchDataError(f"no answers for: {', '.join(missing)}")
    return sorted(answers, key=lambda a: (a.endpoint, a.prompt_id, a.repeat))
