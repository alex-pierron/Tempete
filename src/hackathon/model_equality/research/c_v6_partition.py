"""From the scored long answers to a submission CSV: r-met-e's frozen rule plus our identity.

Inputs: a feature table per endpoint computed by r-met-e's scoring pipeline over the v6 collection
(`endpoint, mle_temperature, z_at_one, fp8_llr, n_tokens`: fitted temperature, calibrated z at
temperature 1, fp8 log-likelihood ratio per token, tokens scored), the exact families
(`identification.csv`) and the ep answer store (identity answers). The rule is `e_partition.propose`
at the median token count; the identity read-out marks an endpoint disguised when strictly more than
half of ITS identity answers (the v6 formulations `d-ident-*` and the earlier `fam-identity-000`)
name none of the base models (`c_identity`). `keep_main` forces endpoints back to the main model of
their family: public readings have refuted a flip the likelihood alone suggested at 563 tokens
(f7b, ep-33). The output is the submission CSV (families exact, one model per `model_id`) and a
decision table that says which endpoints differ from another partition.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_identity import IDENTITY_PROMPT_IDS, is_disguised
from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_v4_ep import EpChainError, read_answers, read_families
from hackathon.model_equality.research.e_partition import (
    FP8_LLR_MARGIN_SIGMA,
    EndpointEvidence,
    propose,
)

EP_IDENTITY_IDS: tuple[str, ...] = (*IDENTITY_PROMPT_IDS, "fam-identity-000")
MARKED_IDENTITY = "I was created by another laboratory."
UNMARKED_IDENTITY = "I am Qwen, created by Alibaba Cloud."
FEATURE_COLUMNS: tuple[str, ...] = ("endpoint", "mle_temperature", "z_at_one", "fp8_llr", "n_tokens")
MODEL_MAIN_SUFFIX = ":main"


@dataclass(frozen=True)
class Decision:
    """Partition plus the table behind it."""

    partition: Partition
    rows: tuple[Mapping[str, object], ...]
    tokens: int


def read_features(path: Path) -> dict[str, dict[str, float]]:
    """`endpoint -> {mle_temperature, z_at_one, fp8_llr, n_tokens}` from the feature table.

    Raises:
        EpChainError: a required column is missing.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in FEATURE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise EpChainError(f"{path.name}: missing columns {missing}")
        return {row["endpoint"]: {c: float(row[c]) for c in FEATURE_COLUMNS[1:]} for row in reader}


def decide(
    endpoints: Sequence[str], features: Mapping[str, Mapping[str, float]],
    families: Sequence[int], family_names: Sequence[str],
    answers: Mapping[tuple[str, str], Sequence[str]], keep_main: Sequence[str] = (),
    fp8_margin: float = FP8_LLR_MARGIN_SIGMA,
) -> Decision:
    """Apply `propose` at the median token count, then the `keep_main` overrides."""
    tokens = int(np.median([features[e]["n_tokens"] for e in endpoints]))
    evidence = []
    for endpoint, family in zip(endpoints, families, strict=True):
        texts = [a for pid in EP_IDENTITY_IDS for a in answers.get((endpoint, pid), [])]
        disguised = is_disguised(texts)
        feature = features[endpoint]
        evidence.append(EndpointEvidence(
            endpoint, family_names[family], "high", feature["mle_temperature"], feature["z_at_one"],
            feature["fp8_llr"], MARKED_IDENTITY if disguised else UNMARKED_IDENTITY, None))
    rows = propose(evidence, tokens=tokens, fp8_margin_sigma=fp8_margin)
    forced = set(keep_main)
    for row in rows:
        if row["endpoint"] in forced:
            row["model_id"] = f"{row['family']}{MODEL_MAIN_SUFFIX}"
            row["flags"] = f"{row['flags']}|kept_main_by_public_reading".strip("|")
    model_ids = sorted({str(r["model_id"]) for r in rows})
    family_of_model = {mid: family_names.index(next(str(r["family"]) for r in rows
                                                     if r["model_id"] == mid))
                       for mid in model_ids}
    groups = [[i for i, r in enumerate(rows) if r["model_id"] == mid] for mid in model_ids]
    partition = Partition.from_groups(groups, [family_of_model[mid] for mid in model_ids])
    return Decision(partition, tuple(rows), tokens)


def load_and_decide(
    features_path: Path, identification: Path, answers_path: Path, keep_main: Sequence[str] = (),
    fp8_margin: float = FP8_LLR_MARGIN_SIGMA,
) -> Decision:
    """Read the files and decide for the 33 scored endpoints."""
    from hackathon.model_equality.planning import SCORED_ENDPOINTS

    endpoints = list(SCORED_ENDPOINTS)
    families, names = read_families(identification, endpoints)
    answers = read_answers(answers_path, endpoints, EP_IDENTITY_IDS)
    return decide(endpoints, read_features(features_path), families, names, answers, keep_main,
                  fp8_margin)


def roles_differing(decision: Decision, other: Mapping[str, str]) -> list[str]:
    """Endpoints whose main / not-main status differs from `other` (`endpoint -> model_id`)."""
    out = []
    for row in decision.rows:
        main_now = str(row["model_id"]).endswith(MODEL_MAIN_SUFFIX)
        main_then = other[str(row["endpoint"])].endswith(MODEL_MAIN_SUFFIX)
        if main_now != main_then:
            out.append(str(row["endpoint"]))
    return out


def main() -> None:
    """CLI: features + families + answers -> submission CSV, decision table, differences."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--keep-main", default="", help="comma-separated endpoints, e.g. ep-33")
    parser.add_argument("--fp8-margin", type=float, default=FP8_LLR_MARGIN_SIGMA)
    parser.add_argument("--compare", type=Path, default=None,
                        help="partition table with endpoint and model_id (e.g. partition_v1)")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    keep = [e for e in args.keep_main.split(",") if e]
    decision = load_and_decide(args.features, args.identification, args.answers, keep,
                               args.fp8_margin)
    decision.partition.write_csv(args.out_csv)
    payload: dict[str, object] = {"tokens": decision.tokens, "rows": list(decision.rows)}
    if args.compare is not None:
        with args.compare.open(newline="", encoding="utf-8") as handle:
            other = {row["endpoint"]: row["model_id"] for row in csv.DictReader(handle)}
        payload["differs_from_compare"] = roles_differing(decision, other)
    args.out_json.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    not_main = [r["endpoint"] for r in decision.rows if not str(r["model_id"]).endswith(":main")]
    print(f"tokens {decision.tokens}; models {len(decision.partition.model_groups())}; "
          f"not main: {not_main}")
    if "differs_from_compare" in payload:
        print(f"differs from the comparison partition: {payload['differs_from_compare']}")


if __name__ == "__main__":
    main()
