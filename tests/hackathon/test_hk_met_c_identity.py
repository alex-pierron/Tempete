"""Team C: rule v3 (a disguised endpoint is never isolated) on synthetic answers (no network)."""

from __future__ import annotations

import numpy as np

from hackathon.model_equality.research.c_identity import (
    ADOPTION_K,
    BASE_TOKENS,
    adoption_verdict,
    bootstrap_paired,
    decision_chain_v3,
    is_disguised,
    is_marked,
)
from hackathon.model_equality.research.c_rehearsal import REF_ENDPOINTS, Rule

LORA = {"ref-04", "ref-07", "ref-09"}


def test_marking_uses_word_boundaries_and_ignores_case() -> None:
    assert not is_marked("I am SmolLM, made by Hugging Face.")
    assert not is_marked("My name is Qwen and Alibaba Cloud trained me.")
    assert not is_marked("I'm Llama, created by Meta AI")
    assert is_marked("I am the Emoji Artist, created to help you.")
    assert is_marked("I was created by OpenAI.")
    assert is_marked("I am a metaphor engine.")  # 'meta' inside a word does not count
    assert not is_marked("trained by the  Hugging   Face team")
    assert "olmo" in BASE_TOKENS


def test_disguised_needs_strictly_more_than_half_marked() -> None:
    marked, base = "I am Claude by Anthropic.", "I am SmolLM by Hugging Face."
    assert is_disguised([marked, marked, base])
    assert not is_disguised([marked, base])  # exactly half is not more than half
    assert not is_disguised([base, base, marked])
    assert not is_disguised([])


def test_override_keeps_a_disguised_endpoint_in_the_main_model() -> None:
    rates = [0.9, 0.1, 0.8, 0.7]
    plain = decision_chain_v3(rates, [False, False, False, False], Rule(0.55))
    assert plain.model.tolist() == [1, 0, 2, 3]
    overridden = decision_chain_v3(rates, [False, False, True, False], Rule(0.55))
    assert overridden.model.tolist() == [1, 0, 0, 2]


def synthetic() -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    """LoRA endpoints end with a full stop 90 %; the system-prompt copy ref-03 too, but it is
    disguised; everything else 10 %."""
    rng = np.random.default_rng(5)
    arithmetic: dict[str, dict[str, list[str]]] = {}
    identity: dict[str, list[str]] = {}
    for endpoint in REF_ENDPOINTS:
        rate = 0.9 if endpoint in LORA | {"ref-03"} else 0.1
        arithmetic[endpoint] = {
            f"a{k}": ["42." if rng.random() < rate else "42" for _ in range(2)] for k in range(25)
        }
        text = "I am Claude by Anthropic." if endpoint == "ref-03" else "I am SmolLM by Hugging Face."
        identity[endpoint] = [text] * 30
    return arithmetic, identity


def test_v3_removes_the_system_prompt_false_positive() -> None:
    arithmetic, identity = synthetic()
    result = bootstrap_paired(arithmetic, identity, REF_ENDPOINTS, 10, 6, Rule(0.55), 150,
                              np.random.default_rng(6))
    summary = result.summary()
    flagged = summary["flagged_v2"]
    disguised = summary["declared_disguised"]
    assert isinstance(flagged, dict) and isinstance(disguised, dict)
    assert flagged["ref-03"] > 0.9 and disguised["ref-03"] == 1.0
    assert all(disguised[e] == 0.0 for e in LORA)
    gain = summary["paired_gain_median"]
    assert isinstance(gain, float) and gain > 0.1  # v3 beats v2 when the disguise is detected
    v3 = summary["v3"]
    assert isinstance(v3, dict) and v3["median"] > 0.6


def test_adoption_rule_needs_all_three_conditions() -> None:
    cell = {"v2": {"median": 0.47}, "v3": {"median": 0.55},
            "declared_disguised": {"ref-04": 0.0, "ref-07": 0.02, "ref-09": 0.0}}
    assert adoption_verdict(cell)["adopt"] is True
    low_gain = {"v2": {"median": 0.50}, "v3": {"median": 0.52}, "declared_disguised": {}}
    assert adoption_verdict(low_gain)["adopt"] is False
    lora_misread = {"v2": {"median": 0.40}, "v3": {"median": 0.55},
                    "declared_disguised": {"ref-09": 0.3}}
    verdict = adoption_verdict(lora_misread)
    assert verdict["no_lora_read_as_disguised"] is False and verdict["adopt"] is False
    assert ADOPTION_K == 6
