"""Endpoint x candidate likelihood matrix and base-model identification (team E, Mac side).

Reads the `.npz` containers written by `e_loglik` (one per candidate and stage) and answers
"which candidate base model best explains the replies of each endpoint?".

Candidates use different tokenisers, so a per-TOKEN mean is not comparable across families. The
comparable quantity is the total log-likelihood in nats of the SAME reply text, summed over the rows
of an endpoint (and divided by the number of characters for a scale-free reading). A candidate wins
an endpoint when it has the largest total; the margin is the gap to the runner-up, and the
confidence label combines that gap with the bootstrap probability, over the rows, that the winner
stays ahead (sampling noise over the 7 stored ep answers is the real limit, not the arithmetic).

Template handling: every candidate is scored under two contexts (`default` chat template, `nosys`
without any system block). Each (endpoint, candidate) cell takes the better of the two, because a
deployed endpoint either injects the template's default system prompt or not; the cell records
which one it used.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_loglik import RowScores, load_scores
from hackathon.model_equality.store import Answer

FloatArray = npt.NDArray[np.float64]
STAGE_PREFIX = "ll_"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261121
CONFIDENCE_HIGH_NATS = 20.0
CONFIDENCE_HIGH_BOOTSTRAP = 0.99
CONFIDENCE_MEDIUM_NATS = 6.0
CONFIDENCE_MEDIUM_BOOTSTRAP = 0.90
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


class MatrixError(HackathonError):
    """The score containers are inconsistent (different rows across candidates, missing data)."""


@dataclass(frozen=True)
class RowTable:
    """Per-row sums of one `(candidate, template, variant)`, aligned with `keys`."""

    keys: tuple[tuple[str, str, int], ...]
    logp: FloatArray
    ent: FloatArray
    var: FloatArray
    n_tokens: npt.NDArray[np.int64]


def row_sums(scores: RowScores, field: str) -> FloatArray:
    """Sum of a flat per-token array over each row (safe for empty rows)."""
    flat = getattr(scores, field).astype(np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(flat)])
    offsets = scores.offsets()
    return np.asarray(cumulative[offsets[1:]] - cumulative[offsets[:-1]], dtype=np.float64)


def make_row_table(header: Mapping[str, object], scores: RowScores) -> RowTable:
    """Collapse per-token arrays to per-row sums."""
    endpoints = header["endpoint"]
    ids = header["prompt_id"]
    repeats = header["repeat"]
    assert isinstance(endpoints, list) and isinstance(ids, list) and isinstance(repeats, list)
    keys = tuple((str(a), str(b), int(c)) for a, b, c in zip(endpoints, ids, repeats, strict=True))
    return RowTable(
        keys=keys,
        logp=row_sums(scores, "logp"),
        ent=row_sums(scores, "ent"),
        var=row_sums(scores, "var"),
        n_tokens=scores.n_tokens.astype(np.int64),
    )


def load_candidate_files(directory: Path, stage: str) -> dict[str, Path]:
    """`{candidate key: path}` of the containers of a stage found in `directory`."""
    prefix = f"{STAGE_PREFIX}{stage}_"
    return {
        path.name[len(prefix) : -len(".npz")]: path
        for path in sorted(directory.glob(f"{prefix}*.npz"))
    }


def load_containers(
    directory: Path, stage: str
) -> dict[str, tuple[dict[str, object], dict[tuple[str, str], RowScores]]]:
    """Load every container of a stage: `{candidate: (header, {(template, variant): scores})}`."""
    return {
        candidate: load_scores(path)
        for candidate, path in load_candidate_files(directory, stage).items()
    }


@dataclass(frozen=True)
class Cell:
    """The result of one endpoint under one candidate.

    Attributes:
        total_ll: Total log-likelihood in nats of the endpoint's replies (best template).
        per_char: `total_ll` divided by the number of reply characters.
        n_tokens: Reply tokens under this candidate's tokeniser.
        template: Context template that gave the higher total.
        row_ll: Per-row log-likelihood under that template (aligned with the endpoint's rows).
    """

    total_ll: float
    per_char: float
    n_tokens: int
    template: str
    row_ll: FloatArray


@dataclass(frozen=True)
class Identification:
    """Best candidate of an endpoint.

    Attributes:
        endpoint: Endpoint id.
        best: Key of the winning candidate.
        runner_up: Key of the second best candidate.
        margin_nats: Total log-likelihood gap, winner minus runner-up.
        best_other_family: Best candidate of a different family and its gap to the winner (nats).
        win_share: Share of rows in which the winner beats the runner-up on its own (reported only).
        bootstrap_win: Bootstrap probability (over rows) that the winner beats the runner-up.
        confidence: `high`, `medium` or `low`.
    """

    endpoint: str
    best: str
    runner_up: str
    margin_nats: float
    best_other_family: tuple[str, float]
    win_share: float
    bootstrap_win: float
    confidence: str


def build_cells(
    containers: Mapping[str, tuple[Mapping[str, object], Mapping[tuple[str, str], RowScores]]],
    answers: Sequence[Answer],
    endpoints: Sequence[str],
) -> dict[str, dict[str, Cell]]:
    """`{endpoint: {candidate: Cell}}` from loaded containers.

    Raises:
        MatrixError: A container lacks rows of an endpoint, or has no base variant.
    """
    chars = {(a.endpoint, a.prompt_id, a.repeat): len(a.response) for a in answers}
    result: dict[str, dict[str, Cell]] = {endpoint: {} for endpoint in endpoints}
    for candidate, (header, per_variant) in containers.items():
        templates = sorted({template for template, variant in per_variant if variant == "base"})
        if not templates:
            raise MatrixError(f"{candidate}: no base variant in the container")
        tables = {t: make_row_table(header, per_variant[(t, "base")]) for t in templates}
        for endpoint in endpoints:
            best: Cell | None = None
            for template, table in tables.items():
                index = [i for i, key in enumerate(table.keys) if key[0] == endpoint]
                if not index:
                    continue
                row_ll = table.logp[index]
                n_chars = sum(chars.get(table.keys[i], 0) for i in index)
                total = float(row_ll.sum())
                cell = Cell(
                    total_ll=total,
                    per_char=total / max(n_chars, 1),
                    n_tokens=int(table.n_tokens[index].sum()),
                    template=template,
                    row_ll=row_ll,
                )
                if best is None or cell.total_ll > best.total_ll:
                    best = cell
            if best is not None:
                result[endpoint][candidate] = best
    return result


def classify_confidence(margin_nats: float, bootstrap_win: float) -> str:
    """Heuristic label from the gap and the bootstrap win probability (thresholds named above).

    The bootstrap resamples the rows of the endpoint, so a gap carried by a single long reply
    does not earn a high label; the row-level win share is reported but not used here (with many
    rows of very unequal length it is a poor summary).
    """
    if margin_nats >= CONFIDENCE_HIGH_NATS and bootstrap_win >= CONFIDENCE_HIGH_BOOTSTRAP:
        return CONFIDENCE_HIGH
    if margin_nats >= CONFIDENCE_MEDIUM_NATS and bootstrap_win >= CONFIDENCE_MEDIUM_BOOTSTRAP:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def identify(
    endpoint: str,
    cells: Mapping[str, Cell],
    families: Mapping[str, str],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> Identification:
    """Rank the candidates of one endpoint and measure the winner's margin.

    The row-level comparison uses the templates already chosen per cell; rows are resampled with
    replacement for the bootstrap. Candidates must share the same rows (same endpoint, same order).

    Raises:
        MatrixError: Fewer than two candidates, or cells over different row counts.
    """
    if len(cells) < 2:
        raise MatrixError(f"{endpoint}: need at least two candidates, got {len(cells)}")
    ranking = sorted(cells, key=lambda key: -cells[key].total_ll)
    best, runner = ranking[0], ranking[1]
    if len(cells[best].row_ll) != len(cells[runner].row_ll):
        raise MatrixError(f"{endpoint}: candidates were scored on different rows")
    margin = cells[best].total_ll - cells[runner].total_ll
    difference = cells[best].row_ll - cells[runner].row_ll
    win_share = float(np.mean(difference > 0))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(difference), size=(BOOTSTRAP_REPLICATES, len(difference)))
    bootstrap_win = float(np.mean(difference[picks].sum(axis=1) > 0))
    other = next(
        (key for key in ranking if families[key] != families[best]), None
    )
    other_gap = (other, cells[best].total_ll - cells[other].total_ll) if other else ("", float("nan"))
    return Identification(
        endpoint=endpoint, best=best, runner_up=runner, margin_nats=margin,
        best_other_family=other_gap, win_share=win_share, bootstrap_win=bootstrap_win,
        confidence=classify_confidence(margin, bootstrap_win),
    )


def write_matrix_csv(
    path: Path, cells: Mapping[str, Mapping[str, Cell]], candidates: Sequence[str]
) -> None:
    """Write the endpoint x candidate matrix of total nats and per-character values."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", *[f"{c}_total" for c in candidates],
                         *[f"{c}_per_char" for c in candidates]])
        for endpoint, row in cells.items():
            writer.writerow([
                endpoint,
                *[f"{row[c].total_ll:.2f}" if c in row else "" for c in candidates],
                *[f"{row[c].per_char:.4f}" if c in row else "" for c in candidates],
            ])
