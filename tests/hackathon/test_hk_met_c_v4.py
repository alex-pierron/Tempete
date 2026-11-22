"""Team C: configuration v4 (relative band edge + identity override), no network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research.c_rehearsal import REF_ENDPOINTS, RehearsalError
from hackathon.model_equality.research.c_v4 import (
    CONFIRMATION_FIRST_PROMPT,
    RuleV4,
    bootstrap_v4,
    confirmation_pool,
    flag_endpoints,
    partition_v4,
    run_confirmation,
)

LORA = {"ref-04", "ref-07", "ref-09"}
FROZEN = RuleV4(delta=0.20, floor=0.45, baseline_percentile=25.0, n=15, k=3)


def test_relative_edge_follows_the_family_baseline_with_a_floor() -> None:
    rule = RuleV4(delta=0.20, floor=0.45, baseline_percentile=25.0)
    low_family = [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.6, 0.9, 0.95]
    assert flag_endpoints(low_family, rule) == [False] * 6 + [True] * 3
    high_family = [0.7, 0.75, 0.8, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0]  # copies end with a full stop
    # the baseline moves up with the family (25th percentile 0.8, edge 1.0): copies are not flagged
    assert flag_endpoints(high_family, rule) == [False] * 7 + [True] * 2


def test_override_keeps_disguised_flagged_endpoints_out_of_isolation() -> None:
    rates = [0.1, 0.1, 0.2, 0.2, 0.9, 0.9]
    guess = partition_v4(rates, [False, False, False, False, True, False], FROZEN)
    assert guess.model.tolist() == [0, 0, 0, 0, 0, 1]


def synthetic(prompts: int = 60) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    """LoRA endpoints end with a full stop 90 %, the others 10 %; ref-03 is a disguised copy."""
    rng = np.random.default_rng(7)
    arithmetic: dict[str, dict[str, list[str]]] = {}
    identity: dict[str, list[str]] = {}
    for endpoint in REF_ENDPOINTS:
        rate = 0.9 if endpoint in LORA else (0.8 if endpoint == "ref-03" else 0.1)
        arithmetic[endpoint] = {
            f"h-arith-{k:03d}": ["1." if rng.random() < rate else "1" for _ in range(2)]
            for k in range(prompts)
        }
        identity[endpoint] = (["I am Claude by Anthropic."] if endpoint == "ref-03"
                              else ["I am SmolLM by Hugging Face."]) * 30
    return arithmetic, identity


def test_bootstrap_reaches_the_ceiling_and_never_isolates_the_disguised_copy() -> None:
    arithmetic, identity = synthetic()
    cell = bootstrap_v4(arithmetic, identity, REF_ENDPOINTS, FROZEN, 150, np.random.default_rng(1))
    summary = cell.summary()
    assert float(summary["median"]) >= 0.6  # type: ignore[arg-type]
    isolated = summary["isolated"]
    disguised = summary["declared_disguised"]
    assert isinstance(isolated, dict) and isinstance(disguised, dict)
    assert isolated["ref-03"] == 0.0 and disguised["ref-03"] == 1.0
    with pytest.raises(RehearsalError):
        bootstrap_v4({e: dict(list(p.items())[:5]) for e, p in arithmetic.items()}, identity,
                     REF_ENDPOINTS, FROZEN, 5, np.random.default_rng(2))


def write_holdout(path: Path, prompts: int) -> None:
    """A synthetic held-out file with `prompts` complete prompts (2 repeats x 9 ref)."""
    arithmetic, _ = synthetic(prompts)
    with path.open("w", encoding="utf-8") as handle:
        for endpoint, by_prompt in arithmetic.items():
            for prompt_id, answers in by_prompt.items():
                for repeat, answer in enumerate(answers):
                    handle.write(json.dumps({"endpoint": endpoint, "id": prompt_id, "repeat": repeat,
                                             "response": answer}) + "\n")


def test_confirmation_uses_only_the_prompts_after_the_first_25(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    write_holdout(path, 50)
    pool = confirmation_pool(path)
    ids = sorted(pool["ref-01"])
    assert len(ids) == 25 and ids[0] == f"h-arith-{CONFIRMATION_FIRST_PROMPT:03d}"
    assert ids[-1] == "h-arith-049"
    short = tmp_path / "short.jsonl"
    write_holdout(short, 40)
    with pytest.raises(RehearsalError):
        confirmation_pool(short)


def test_run_confirmation_verdict_on_a_synthetic_file(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    write_holdout(path, 50)
    identity = tmp_path / "identity.jsonl"
    _, ident = synthetic(1)
    with identity.open("w", encoding="utf-8") as handle:
        for endpoint, answers in ident.items():
            for k, answer in enumerate(answers):
                handle.write(json.dumps({"endpoint": endpoint, "id": f"d-ident-00{k % 3}",
                                         "repeat": k // 3, "response": answer}) + "\n")
    payload = run_confirmation(path, identity, FROZEN, replicates=60)
    assert payload["confirmation_prompts"] == 25
    assert payload["verdict_median_reaches_target"] is True
    labels = {(c["pool"], c["n"]) for c in payload["cells"]}  # type: ignore[attr-defined]
    assert ("all_ref", 15) in labels and ("without_temperature_ref", 20) in labels
