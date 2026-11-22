"""Team C: r-met-e's rule applied to scored features with our identity read-out (no model, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.model_equality.research.c_pairs import endpoint_names
from hackathon.model_equality.research.c_v4_ep import EpChainError
from hackathon.model_equality.research.c_v6_partition import (
    EP_IDENTITY_IDS,
    decide,
    read_features,
    roles_differing,
)

ENDPOINTS = list(endpoint_names())
FAMILY_SIZE = 11
FAMILIES = [k // FAMILY_SIZE for k in range(len(ENDPOINTS))]
NAMES = ["qwen2.5-3b", "qwen2.5-1.5b", "llama3.2-1b"]
TOKENS = 3700
PLAIN_T = 0.72
PLAIN_Z = 6.8
CLEAN_FP8 = -0.0055
DEVIATING_T = 0.86
BASE_ANSWER = "I am Qwen, created by Alibaba Cloud."
OTHER_ANSWER = "I was created by Anthropic, my name is Claude."


def features(**changes: dict[str, float]) -> dict[str, dict[str, float]]:
    """Plain-like features for every endpoint, with `e<k>` overridden."""
    table = {e: {"mle_temperature": PLAIN_T, "z_at_one": PLAIN_Z, "fp8_llr": CLEAN_FP8,
                 "n_tokens": float(TOKENS)} for e in ENDPOINTS}
    for key, value in changes.items():
        table[ENDPOINTS[int(key[1:]) - 1]].update(value)
    return table


def answers(disguised: tuple[int, ...] = ()) -> dict[tuple[str, str], list[str]]:
    """Identity answers for every endpoint; `disguised` (1-based) name another creator."""
    return {(e, pid): [OTHER_ANSWER if k + 1 in disguised else BASE_ANSWER]
            for k, e in enumerate(ENDPOINTS) for pid in EP_IDENTITY_IDS}


def test_a_deviating_temperature_is_isolated_and_the_others_stay_main() -> None:
    decision = decide(ENDPOINTS, features(e5={"mle_temperature": DEVIATING_T}), FAMILIES, NAMES,
                      answers())
    isolated = [r["endpoint"] for r in decision.rows if not str(r["model_id"]).endswith(":main")]
    assert isolated == ["ep-05"]
    assert len(decision.partition.family_groups()) == 3
    assert decision.tokens == TOKENS


def test_a_disguised_endpoint_with_a_deviating_temperature_stays_main() -> None:
    decision = decide(ENDPOINTS, features(e5={"mle_temperature": DEVIATING_T}), FAMILIES, NAMES,
                      answers(disguised=(5,)))
    assert all(str(r["model_id"]).endswith(":main") for r in decision.rows)


def test_keep_main_overrides_a_flip_the_public_reading_refuted() -> None:
    changes = {"e5": {"mle_temperature": DEVIATING_T}, "e6": {"mle_temperature": DEVIATING_T}}
    decision = decide(ENDPOINTS, features(**changes), FAMILIES, NAMES, answers(), keep_main=["ep-06"])
    isolated = [r["endpoint"] for r in decision.rows if not str(r["model_id"]).endswith(":main")]
    assert isolated == ["ep-05"]
    kept = next(r for r in decision.rows if r["endpoint"] == "ep-06")
    assert "kept_main_by_public_reading" in str(kept["flags"])


def test_roles_differing_lists_only_main_status_changes() -> None:
    decision = decide(ENDPOINTS, features(e5={"mle_temperature": DEVIATING_T}), FAMILIES, NAMES,
                      answers())
    other = {e: f"{NAMES[FAMILIES[k]]}:main" for k, e in enumerate(ENDPOINTS)}
    assert roles_differing(decision, other) == ["ep-05"]


def test_missing_column_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "f.csv").write_text("endpoint,mle_temperature\nep-01,0.72\n", encoding="utf-8")
    with pytest.raises(EpChainError):
        read_features(tmp_path / "f.csv")


def test_features_file_is_read_as_floats(tmp_path: Path) -> None:
    (tmp_path / "f.csv").write_text(
        "endpoint,mle_temperature,z_at_one,fp8_llr,n_tokens\nep-01,0.72,6.8,-0.0055,3700\n",
        encoding="utf-8")
    assert read_features(tmp_path / "f.csv")["ep-01"]["n_tokens"] == pytest.approx(3700.0)
