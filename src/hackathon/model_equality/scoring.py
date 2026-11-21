"""Local re-implementation of the official score and of `score_practice.py`.

Official score (subject, page 2):

    score = 0.1 * max(0, MCC_family) + 0.9 * max(0, MCC_model)

* MCC_family: "same family?" over all unordered endpoint pairs (528 for 33 endpoints).
* MCC_model: "same model?" over the unordered pairs that belong to the same TRUE family only
  (pairs of different families are never the same model, so they are not counted).
* MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). A zero denominator gives 0, as in
  `score_practice.py` (the server's convention for that degenerate case is not published).
* Same model = same family label AND same model label, exactly as `score_practice.py` defines it.

`practice_score` reproduces `score_practice.py` line for line on the `ref-*` rows: MCC over all
pairs of the reference endpoints (one family, so it equals the model question of the real score).
A test compares both implementations on random labelings.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.errors import HackathonError

FAMILY_WEIGHT = 0.1
MODEL_WEIGHT = 0.9
REFERENCE_PREFIX = "ref-"
HEADER = ["endpoint", "family", "model"]
MIN_COLUMNS = 3
KIND_COLUMN = 3

Labeling = Mapping[str, tuple[str, str]]
"""endpoint -> (family label, model label)."""


class ScoreInputError(HackathonError):
    """A labeling file cannot be scored (missing endpoint, unreadable rows)."""


@dataclass(frozen=True)
class Confusion:
    """Pair-level confusion counts of a "same?" question."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def pairs(self) -> int:
        """Number of pairs counted."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def mcc(self) -> float:
        """Matthews correlation coefficient in [-1, 1]; 0 when a marginal is empty."""
        denominator = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        if denominator == 0:
            return 0.0
        return (self.tp * self.tn - self.fp * self.fn) / denominator

    @property
    def clipped_mcc(self) -> float:
        """`max(0, MCC)`, the quantity the official score uses."""
        return max(0.0, self.mcc)


@dataclass(frozen=True)
class ScoreBreakdown:
    """The two MCCs and the combined official score."""

    family: Confusion
    model: Confusion
    score: float


def confusion(truth: list[bool], guess: list[bool]) -> Confusion:
    """Count TP / FP / FN / TN from aligned per-pair booleans (True = "same")."""
    if len(truth) != len(guess):
        raise ValueError("truth and guess must be aligned pair by pair")
    tp = sum(1 for t, g in zip(truth, guess, strict=True) if t and g)
    fp = sum(1 for t, g in zip(truth, guess, strict=True) if not t and g)
    fn = sum(1 for t, g in zip(truth, guess, strict=True) if t and not g)
    tn = len(truth) - tp - fp - fn
    return Confusion(tp, fp, fn, tn)


def official_score(truth: Labeling, prediction: Labeling) -> ScoreBreakdown:
    """The official score of `prediction` against `truth` (both over the same endpoints).

    Raises:
        ScoreInputError: The two labelings do not cover the same endpoints.
    """
    if set(truth) != set(prediction):
        only_truth = sorted(set(truth) - set(prediction))
        only_pred = sorted(set(prediction) - set(truth))
        raise ScoreInputError(f"endpoint sets differ: missing {only_truth}, extra {only_pred}")
    family_truth: list[bool] = []
    family_guess: list[bool] = []
    model_truth: list[bool] = []
    model_guess: list[bool] = []
    for first, second in itertools.combinations(sorted(truth), 2):
        same_true_family = truth[first][0] == truth[second][0]
        family_truth.append(same_true_family)
        family_guess.append(prediction[first][0] == prediction[second][0])
        if same_true_family:
            model_truth.append(truth[first] == truth[second])
            model_guess.append(prediction[first] == prediction[second])
    family = confusion(family_truth, family_guess)
    model = confusion(model_truth, model_guess)
    score = FAMILY_WEIGHT * family.clipped_mcc + MODEL_WEIGHT * model.clipped_mcc
    return ScoreBreakdown(family=family, model=model, score=score)


def read_label_rows(path: Path, prefix: str, with_kind: bool = False) -> dict[str, tuple[str, str, str]]:
    """Read `endpoint,family,model[,kind]` rows whose endpoint starts with `prefix`.

    Mirrors the reader of `score_practice.py`: UTF-8 with optional BOM, cells stripped, blank
    rows skipped, optional header line.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[cell.strip() for cell in row] for row in csv.reader(handle)
                if any(cell.strip() for cell in row)]
    if rows and rows[0][:MIN_COLUMNS] == HEADER:
        rows = rows[1:]
    out: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if len(row) >= MIN_COLUMNS and row[0].startswith(prefix):
            kind = row[KIND_COLUMN] if with_kind and len(row) > KIND_COLUMN else ""
            out[row[0]] = (row[1], row[2], kind)
    return out


@dataclass(frozen=True)
class PracticeResult:
    """Outcome of the practice scoring, with the wrong pairs spelled out."""

    confusion: Confusion
    mcc: float
    wrong_pairs: list[str]

    def render(self) -> str:
        """Text in the style of `score_practice.py`."""
        c = self.confusion
        lines = [
            f"Practice score (MCC over {c.pairs} pairs, same model or not): {self.mcc:.3f}",
            (f"Right: {c.tp + c.tn} pairs, wrong: {c.fp + c.fn} "
            f"({c.fp} wrongly called the same, {c.fn} wrongly called different)"),
        ]
        if self.wrong_pairs:
            lines.append("Wrong pairs:")
            lines.extend(self.wrong_pairs)
        return "\n".join(lines)


def practice_score(mine_path: Path, answers_path: Path) -> PracticeResult:
    """Equivalent of `python score_practice.py mine.csv practice_answers.csv`.

    Raises:
        ScoreInputError: `mine_path` has no line for some reference endpoint.
    """
    answers = read_label_rows(answers_path, REFERENCE_PREFIX, with_kind=True)
    mine = read_label_rows(mine_path, REFERENCE_PREFIX)
    missing = sorted(set(answers) - set(mine))
    if missing:
        raise ScoreInputError(f"Your file has no line for: {', '.join(missing)}")
    truth_flags: list[bool] = []
    guess_flags: list[bool] = []
    wrong: list[str] = []
    for first, second in itertools.combinations(sorted(answers), 2):
        truth = answers[first][:2] == answers[second][:2]
        guess = mine[first][:2] == mine[second][:2]
        truth_flags.append(truth)
        guess_flags.append(guess)
        if truth != guess:
            wrong.append(
                f"  {first} ({answers[first][2]}) and {second} ({answers[second][2]}): "
                f"you said {'same' if guess else 'different'}, "
                f"they are {'the same model' if truth else 'different models'}"
            )
    counts = confusion(truth_flags, guess_flags)
    return PracticeResult(confusion=counts, mcc=counts.clipped_mcc, wrong_pairs=wrong)
