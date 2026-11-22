"""Team C: configuration v4 applied to the ep with the true families (no network)."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import pytest

from hackathon.model_equality.research.c_identity import IDENTITY_PROMPT_IDS
from hackathon.model_equality.research.c_pairs import endpoint_names
from hackathon.model_equality.research.c_v4 import RuleV4
from hackathon.model_equality.research.c_v4_ep import (
    EpChainError,
    EpDecision,
    decide,
    read_answers,
    read_families,
)

ENDPOINTS = list(endpoint_names())
FAMILY_SIZE = 11
ARITHMETIC_IDS = [f"p10-arith-{k:03d}" for k in range(15)]
RULE = RuleV4(delta=0.20, floor=0.45, n=15)
BASE_NAMES = ("qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b")
LOW_SHARE = 0.2
HIGH_SHARE = 0.8
BASE_IDENTITY = "I am Qwen, created by Alibaba Cloud."
DISGUISE_IDENTITY = "I was created by Anthropic and my name is Claude."
KEPT_ANSWERS_WHEN_MISSING = 4


def family_of_endpoint(position: int) -> int:
    """Endpoints 0..10 are family 0, 11..21 family 1, 22..32 family 2 (test layout)."""
    return position // FAMILY_SIZE


def write_identification(path: Path) -> None:
    """A families file in r-met-e's layout (header, 33 ep rows, one ref row)."""
    lines = ["endpoint,group,best"]
    lines += [f"{e},ep,{BASE_NAMES[family_of_endpoint(i)]}" for i, e in enumerate(ENDPOINTS)]
    lines += ["ref-01,ref,smollm2-360m"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_answers(
    path: Path, share: dict[int, float], disguised: Collection[int],
    missing: Collection[int] = (),
) -> None:
    """Synthetic store: arithmetic answers ending with a full stop at `share`, identity answers."""
    rows: list[dict[str, object]] = []
    for position, endpoint in enumerate(ENDPOINTS):
        stops = round(share.get(position, LOW_SHARE) * len(ARITHMETIC_IDS))
        for k, pid in enumerate(ARITHMETIC_IDS):
            if position in missing and k >= KEPT_ANSWERS_WHEN_MISSING:
                continue
            rows.append({"endpoint": endpoint, "id": pid, "repeat": 0,
                         "response": "12345." if k < stops else "12345"})
        for pid in IDENTITY_PROMPT_IDS:
            text = DISGUISE_IDENTITY if position in disguised else BASE_IDENTITY
            rows.append({"endpoint": endpoint, "id": pid, "repeat": 0, "response": text})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def run_chain(tmp_path: Path) -> EpDecision:
    """Decision from the files written in `tmp_path`."""
    family_of, names = read_families(tmp_path / "i.csv", ENDPOINTS)
    answers = read_answers(tmp_path / "a.jsonl", ENDPOINTS, [*ARITHMETIC_IDS, *IDENTITY_PROMPT_IDS])
    return decide(ENDPOINTS, family_of, names, answers, ARITHMETIC_IDS, RULE)


def test_families_are_numbered_in_order_of_first_appearance(tmp_path: Path) -> None:
    write_identification(tmp_path / "i.csv")
    family_of, names = read_families(tmp_path / "i.csv", ENDPOINTS)
    assert names == list(BASE_NAMES)
    assert family_of == [family_of_endpoint(i) for i in range(len(ENDPOINTS))]


def test_missing_endpoint_in_families_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "i.csv").write_text("endpoint,group,best\nep-01,ep,qwen2.5-3b\n", encoding="utf-8")
    with pytest.raises(EpChainError):
        read_families(tmp_path / "i.csv", ENDPOINTS)


def test_flag_isolates_a_high_rate_variant_but_never_a_disguised_copy(tmp_path: Path) -> None:
    write_identification(tmp_path / "i.csv")
    # position 3 and 14: high rate, base identity -> isolated; 5: high rate + disguise -> main
    write_answers(tmp_path / "a.jsonl", {3: HIGH_SHARE, 5: HIGH_SHARE, 14: HIGH_SHARE},
                  disguised={5})
    decision = run_chain(tmp_path)
    assert decision.isolated[3] and decision.isolated[14]
    assert decision.flagged[5] and not decision.isolated[5] and decision.disguised[5]
    assert sum(decision.isolated) == 2
    assert len(decision.partition.family_groups()) == 3
    assert len(decision.partition.model_groups()) == 3 + 2


def test_baseline_is_per_family_not_over_all_endpoints(tmp_path: Path) -> None:
    """A family whose copies all end with a full stop at 80 % flags nobody (relative edge)."""
    write_identification(tmp_path / "i.csv")
    share = {i: HIGH_SHARE for i in range(FAMILY_SIZE, 2 * FAMILY_SIZE)}
    write_answers(tmp_path / "a.jsonl", share, disguised=())
    decision = run_chain(tmp_path)
    assert not any(decision.flagged[FAMILY_SIZE:2 * FAMILY_SIZE])
    assert decision.thresholds[1] == pytest.approx(HIGH_SHARE + RULE.delta)
    assert decision.thresholds[0] == pytest.approx(RULE.floor)


def test_endpoint_with_too_few_answers_is_never_flagged(tmp_path: Path) -> None:
    write_identification(tmp_path / "i.csv")
    write_answers(tmp_path / "a.jsonl", {3: 1.0}, disguised=(), missing={3})
    decision = run_chain(tmp_path)
    assert decision.answer_counts[3] == KEPT_ANSWERS_WHEN_MISSING
    assert not decision.flagged[3]


def test_duplicate_answers_are_counted_once(tmp_path: Path) -> None:
    row = {"endpoint": "ep-01", "id": "p10-arith-000", "repeat": 0, "response": "1."}
    (tmp_path / "a.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n",
                                      encoding="utf-8")
    out = read_answers(tmp_path / "a.jsonl", ENDPOINTS, ["p10-arith-000"])
    assert out[("ep-01", "p10-arith-000")] == ["1."]
