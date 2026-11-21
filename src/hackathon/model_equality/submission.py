"""Submission file: write, validate (before anything is sent), submit.

Format (subject and official `submission.py`): CSV with one line per endpoint and three columns
`endpoint,family,model`; header optional; order irrelevant; labels are free text without commas.
Endpoints sharing a model label must share the family label. The server accepts exactly
`ep-01 ... ep-33`, one line each, and a failed submission still starts the 30 minute cooldown,
which is why validation is strict and local.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from hackathon.common.config import TASK_MODEL_EQUALITY, Settings
from hackathon.common.errors import SubmissionInvalidError
from hackathon.common.submit import SubmitResult, submit_file
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.scoring import HEADER, Labeling

COLUMNS = 3
LabelRow = tuple[str, str, str]
"""`(endpoint, family, model)`."""


def rows_from_labeling(labeling: Labeling) -> list[LabelRow]:
    """Sorted `(endpoint, family, model)` rows of a labeling."""
    return [(endpoint, family, model) for endpoint, (family, model) in sorted(labeling.items())]


def write_submission(rows: Iterable[LabelRow], path: Path) -> None:
    """Write rows with the header the official template uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def collect_problems(
    raw_rows: Sequence[Sequence[str]], expected: Sequence[str] = SCORED_ENDPOINTS
) -> list[str]:
    """Every rule violation of parsed CSV rows (header already removed), as readable messages."""
    problems: list[str] = []
    seen: dict[str, int] = {}
    family_of: dict[str, str] = {}
    families_of_model: dict[str, set[str]] = defaultdict(set)
    for number, row in enumerate(raw_rows, start=1):
        if len(row) != COLUMNS:
            problems.append(f"row {number}: expected {COLUMNS} columns, got {len(row)} {list(row)}")
            continue
        endpoint, family, model = (cell.strip() for cell in row)
        for name, value in (("endpoint", endpoint), ("family", family), ("model", model)):
            if not value:
                problems.append(f"row {number}: empty {name}")
            if "," in value or "\n" in value or "\r" in value:
                problems.append(f"row {number}: {name} {value!r} contains a comma or newline")
        seen[endpoint] = seen.get(endpoint, 0) + 1
        family_of[endpoint] = family
        families_of_model[model].add(family)
    problems.extend(
        f"endpoint {endpoint} appears {count} times" for endpoint, count in seen.items() if count > 1
    )
    problems.extend(f"missing endpoint {endpoint}" for endpoint in expected if endpoint not in seen)
    problems.extend(
        f"unexpected endpoint {endpoint!r} (only {expected[0]} ... {expected[-1]} are accepted)"
        for endpoint in sorted(seen) if endpoint not in set(expected)
    )
    problems.extend(
        f"model label {model!r} is used in several families {sorted(families)}"
        for model, families in sorted(families_of_model.items()) if len(families) > 1
    )
    return problems


def read_submission_rows(path: Path) -> list[list[str]]:
    """Parse the CSV, dropping blank lines and the optional header.

    Raises:
        SubmissionInvalidError: The file is missing or unreadable.
    """
    if not path.is_file():
        raise SubmissionInvalidError(f"submission file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SubmissionInvalidError(f"cannot read {path}: {exc}") from exc
    if rows and [cell.strip() for cell in rows[0]] == HEADER:
        rows = rows[1:]
    return rows


def validate_submission(path: Path, expected: Sequence[str] = SCORED_ENDPOINTS) -> list[LabelRow]:
    """Validate a submission file and return its rows.

    Raises:
        SubmissionInvalidError: With every problem found, listed one per line.
    """
    rows = read_submission_rows(path)
    problems = collect_problems(rows, expected)
    if problems:
        raise SubmissionInvalidError(f"{path} is not a valid submission:\n  " + "\n  ".join(problems))
    return [(row[0].strip(), row[1].strip(), row[2].strip()) for row in rows]


def describe_clusters(rows: Sequence[LabelRow]) -> str:
    """Human summary: families, and the models inside each (for a last look before submitting)."""
    tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for endpoint, family, model in rows:
        tree[family][model].append(endpoint)
    lines = [(f"{len(rows)} endpoints, {len(tree)} family(ies), "
             f"{sum(len(models) for models in tree.values())} model(s)")]
    for family in sorted(tree):
        lines.append(f"family {family}")
        for model in sorted(tree[family]):
            lines.append(f"  model {model}: {' '.join(sorted(tree[family][model]))}")
    return "\n".join(lines)


def submit(path: Path, settings: Settings, *, confirm: bool) -> SubmitResult | None:
    """Validate, then send the file only when `confirm` is True.

    Returns:
        The logged result, or None for a dry run (`confirm` False: validation only).

    Raises:
        SubmissionInvalidError: The file breaks the format rules (nothing is sent).
        CooldownError, SubmitError, HackathonError: From `submit_file`.
    """
    validate_submission(path)
    if not confirm:
        return None
    return submit_file(TASK_MODEL_EQUALITY, path, settings, confirm=True)


def result_as_dict(result: SubmitResult) -> dict[str, object]:
    """The logged result as a plain dict (for JSON printing)."""
    return dict(asdict(result))
