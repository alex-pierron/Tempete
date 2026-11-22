"""How well the LoRA endpoints separate from the plain endpoint at a FIXED number of tokens (zoo).

Question of the v7 vote: do more tokens per endpoint isolate LoRA 0.3 on the real base models
(Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-1B)? Team F's zoo stores, per answer, the summed
log-probability under the base model and the number of tokens. The read-out here is the mean
per-token log-likelihood of an endpoint (the feature that separates LoRA in `e_zoo_ll`), compared
between a LoRA endpoint and the plain endpoint of the same family.

Method: for each token budget, answers are drawn WITH replacement until the budget is reached
(the zoo has about 4 300 tokens per endpoint, less than the largest budgets of the vote, so a draw
without replacement would understate the noise), `draws` times per endpoint; then AUC and the
true-positive rate at 5 % false positives against the plain endpoint. A second, independent set
of plain draws gives the null line (`plain_copy`, AUC near 0.5).

`--overlay` reads the LoRA endpoints from another zoo run (replacement adapters of team F, for
instance `zoo_dolly`) and keeps the plain endpoint of the base zoo: same base model, same prompts.

Limits (to repeat wherever the numbers are quoted): white box (the exact base model scores the
text), short arithmetic answers (about 5 tokens each), our own adapters and not the organisers'.

Usage: python -m hackathon.model_equality.research.n_lora_sep --zoo-dir <dir> --out <csv>
           [--overlay <dir> --adapter dolly] [--budgets 3500 6400 9400 12300]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_calib import auc, tpr_at_fpr

FAMILIES: tuple[str, ...] = ("qwen15", "qwen3", "llama1", "smollm")
PLAIN_SLOT = 1
LORA_SLOTS: Mapping[str, int] = {"lora_0.3": 9, "lora_0.7": 4, "lora_1.5": 7}
"""Zoo slot of each LoRA strength (the slot is the number of the ref of the same variant)."""
NULL_LABEL = "plain_copy"
DEFAULT_BUDGETS: tuple[int, ...] = (563, 3500, 6400, 9400, 12300)
"""Tokens per endpoint: before v6, after v6, and after 12 / 24 / 36 more long answers."""
DEFAULT_DRAWS = 400
SEED = 20261122
Z_FALSE_POSITIVE_5PCT = 1.645
Z_POWER_90PCT = 1.282
GZIP_MAGIC = b"\x1f\x8b"
BASE_ADAPTER = "alpaca"
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class LoraSepError(HackathonError):
    """A likelihood file lacks an endpoint needed by the comparison."""


@dataclass(frozen=True)
class AnswerRows:
    """Per-answer token counts and summed log-probabilities of one endpoint."""

    n_tokens: IntArray
    logp: FloatArray


def find_ll_file(directory: Path, family: str) -> Path:
    """The likelihood file of a family, gzipped or not.

    Raises:
        LoraSepError: no `ll_<family>.json` or `ll_<family>.json.gz` in the directory.
    """
    for name in (f"ll_{family}.json.gz", f"ll_{family}.json"):
        if (directory / name).is_file():
            return directory / name
    raise LoraSepError(f"no likelihood file for {family} in {directory}")


def load_base_rows(path: Path, family: str) -> dict[str, AnswerRows]:
    """Rows of the endpoints of `family` scored under its base model.

    Raises:
        LoraSepError: the file has no `scores.base` table.
    """
    raw = path.read_bytes()
    text = gzip.decompress(raw) if raw.startswith(GZIP_MAGIC) else raw
    try:
        base = json.loads(text)["scores"]["base"]
    except (KeyError, ValueError) as exc:
        raise LoraSepError(f"{path}: no scores.base table") from exc
    prefix = f"zoo-{family}-"
    grouped: dict[str, list[tuple[int, float]]] = {}
    for key, (logp_sum, n_tokens) in base.items():
        if key.startswith(prefix):
            grouped.setdefault(key.split("|", 1)[0], []).append((int(n_tokens), float(logp_sum)))
    return {
        endpoint: AnswerRows(np.array([r[0] for r in rows], dtype=np.int64),
                             np.array([r[1] for r in rows], dtype=np.float64))
        for endpoint, rows in grouped.items()
    }


def mean_ll_draws(rows: AnswerRows, budget: int, draws: int, rng: np.random.Generator) -> FloatArray:
    """Mean per-token log-likelihood of `draws` resampled endpoints of about `budget` tokens.

    Raises:
        LoraSepError: the endpoint has no token.
    """
    mean_tokens = float(rows.n_tokens.mean()) if rows.n_tokens.size else 0.0
    if mean_tokens <= 0.0:
        raise LoraSepError("endpoint without tokens")
    # Enough answers to pass the budget in every draw; the cut below keeps the first ones only.
    count = int(budget / mean_tokens * 2) + 8
    picked = rng.integers(0, rows.n_tokens.size, size=(draws, count))
    tokens = rows.n_tokens[picked].cumsum(axis=1)
    keep = (tokens - rows.n_tokens[picked]) < budget
    total_tokens = (rows.n_tokens[picked] * keep).sum(axis=1)
    total_logp = (rows.logp[picked] * keep).sum(axis=1)
    return np.asarray(total_logp / total_tokens, dtype=np.float64)


def tokens_for_power(gap: float, sd_clean: float, sd_target: float, budget: int) -> float:
    """Tokens for 90 % true positives at 5 % false positives, scaled from the noise at `budget`.

    The standard deviation of a mean over N tokens falls as 1 / sqrt(N); infinite without a gap.
    """
    if gap == 0.0:
        return float("inf")
    needed_sd_units = Z_FALSE_POSITIVE_5PCT * sd_clean + Z_POWER_90PCT * sd_target
    return float(budget * (needed_sd_units / abs(gap)) ** 2)


def separation_rows(
    family: str, adapter: str, plain: AnswerRows, targets: Mapping[str, AnswerRows],
    budgets: Sequence[int], *, draws: int = DEFAULT_DRAWS, seed: int = SEED,
) -> list[dict[str, float | str | int]]:
    """One row per (variant, budget): AUC, true positives at 5 % false positives, tokens needed."""
    out: list[dict[str, float | str | int]] = []
    for budget in budgets:
        rng = np.random.default_rng(seed + budget)
        clean = mean_ll_draws(plain, budget, draws, rng)
        candidates = {NULL_LABEL: plain, **targets}
        for variant, rows in candidates.items():
            values = mean_ll_draws(rows, budget, draws, rng)
            gap = float(values.mean() - clean.mean())
            area = auc(values, clean)
            out.append({
                "family": family, "adapter": adapter, "variant": variant, "tokens": budget,
                "auc": round(max(area, 1.0 - area), 3),
                "tpr_at_5pct_fpr": round(tpr_at_fpr(values, clean), 3),
                "gap_nats_per_token": round(gap, 5),
                "sd_clean": round(float(clean.std()), 5),
                "tokens_for_tpr90": round(
                    tokens_for_power(gap, float(clean.std()), float(values.std()), budget)),
            })
    return out


def family_rows(
    family: str, zoo_dir: Path, overlay_dir: Path | None, adapter: str, budgets: Sequence[int],
    draws: int,
) -> list[dict[str, float | str | int]]:
    """Separation rows of one family (LoRA endpoints from the overlay when given).

    Raises:
        LoraSepError: the plain endpoint or a LoRA endpoint is missing.
    """
    base = load_base_rows(find_ll_file(zoo_dir, family), family)
    source = base if overlay_dir is None else load_base_rows(find_ll_file(overlay_dir, family), family)
    plain_name = f"zoo-{family}-{PLAIN_SLOT:02d}"
    names = {variant: f"zoo-{family}-{slot:02d}" for variant, slot in LORA_SLOTS.items()}
    missing = [n for n in names.values() if n not in source]
    if plain_name not in base or missing:
        raise LoraSepError(f"{family}: missing endpoints {missing or [plain_name]}")
    targets = {variant: source[name] for variant, name in names.items()}
    return separation_rows(family, adapter, base[plain_name], targets, budgets, draws=draws)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the separation table and print it."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--zoo-dir", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--adapter", default=BASE_ADAPTER, help="label of the LoRA adapter")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows: list[dict[str, float | str | int]] = []
    for family in args.families:
        rows += family_rows(family, args.zoo_dir, args.overlay, args.adapter, args.budgets, args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(",".join(str(value) for value in row.values()))
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
