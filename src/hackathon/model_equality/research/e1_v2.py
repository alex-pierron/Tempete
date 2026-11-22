"""Second labelling of the E1 data: families as in `e1_run`, models by robust length outliers.

Inside each family the endpoints are scored with `model_level.endpoint_scores` against the
OTHER members of the family only (a family's verbosity differs from another family's), the robust
z of the length statistic is cut at the threshold fitted on the ref at the same N, and every
endpoint above it becomes its own model; the rest of the family is one model.

Run after `model_level` (thresholds) and `e1_run` (families):

    python -m hackathon.model_equality.research.e1_v2 --analysis DIR --model-level FILE
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.prompts.phases import E1_PROMPT_IDS
from hackathon.model_equality.research.e1_run import E1_N_PROMPTS, Labels, readable_names
from hackathon.model_equality.research.model_level import endpoint_scores
from hackathon.model_equality.smalln import build_small_n_data, restrict
from hackathon.model_equality.store import AnswerStore

STATISTIC = "lenzr"
MIN_FAMILY_SIZE = 4
"""Below this many members a family has no meaningful 'bulk': it stays one model."""


def v2_labels(
    families: list[str], store_dir: Path, threshold: float
) -> tuple[Labels, dict[str, float]]:
    """Family-relative length outliers become their own models.

    Returns:
        The nested labels and the robust length z of every endpoint (for the figures).
    """
    answers = list(AnswerStore(store_dir / "answers.jsonl").load().values())
    data = build_small_n_data(answers, list(SCORED_ENDPOINTS), {(p, 0) for p in E1_PROMPT_IDS})
    names = sorted(set(families))
    family_ids = np.array([names.index(f) for f in families], dtype=np.int64)
    models = np.zeros(len(families), dtype=np.int64)
    scores: dict[str, float] = {}
    next_id = 0
    for family in range(len(names)):
        members = np.flatnonzero(family_ids == family)
        if len(members) >= MIN_FAMILY_SIZE:
            sub = restrict(data, members.tolist())
            values = endpoint_scores(sub, list(range(len(sub.units))))[STATISTIC]
        else:
            values = np.zeros(len(members))
        flagged = values > threshold
        models[members] = next_id
        next_id += 1
        for local, member in enumerate(members):
            scores[SCORED_ENDPOINTS[int(member)]] = float(values[local])
            if flagged[local]:
                models[member] = next_id
                next_id += 1
    return Labels(families=family_ids, models=models), scores


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--model-level", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="robust-z cut (default: the one fitted on the ref at N = 5)")
    parser.add_argument("--tag", default="v2", help="suffix of the written labels file")
    args = parser.parse_args()
    rows = json.loads(args.model_level.read_text(encoding="utf-8"))["rows"]
    row = next(r for r in rows if r["statistic"] == STATISTIC and r["n_prompts"] == E1_N_PROMPTS)
    labels_v1 = list(csv.DictReader((args.analysis / "labels_ep.csv").open(encoding="utf-8")))
    families = [r["family"] for r in labels_v1]
    threshold = args.threshold if args.threshold is not None else row["threshold"]
    labels, scores = v2_labels(families, args.store_dir, threshold)
    family_names, model_names = readable_names(labels)
    with (args.analysis / f"labels_ep_{args.tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(zip(SCORED_ENDPOINTS, family_names, model_names, strict=True))
    (args.analysis / ("lenzr_scores_ep.json" if args.tag == "v2"
                      else f"lenzr_scores_ep_{args.tag}.json")).write_text(
        json.dumps({"threshold": threshold, "scores": scores}, indent=1), encoding="utf-8")
    flagged = [e for e, v in scores.items() if v > threshold]
    print(f"threshold {threshold:.2f}; {len(set(model_names))} models; flagged: {flagged}")


if __name__ == "__main__":
    main()
