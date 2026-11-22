"""Team C: the ref rehearsal chain on synthetic answers (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_rehearsal import (
    CEILING_EVERYTHING,
    CEILING_FP8_LUMPED,
    REF_ENDPOINTS,
    RehearsalError,
    Rule,
    bootstrap_scores,
    common_prompts,
    decision_chain,
    ends_with_full_stop,
    load_holdout,
    official_score,
    run_holdout,
    true_partition,
    verdict,
)

LORA_ENDPOINTS = {"ref-09", "ref-04", "ref-07"}


def synthetic_pool(prompts: int = 30, draws: int = 2) -> dict[str, dict[str, list[str]]]:
    """LoRA endpoints end with a full stop 90 % of the time, the others 10 % (fp8 and S alike)."""
    rng = np.random.default_rng(1)
    pool: dict[str, dict[str, list[str]]] = {}
    for endpoint in REF_ENDPOINTS:
        rate = 0.9 if endpoint in LORA_ENDPOINTS else 0.1
        pool[endpoint] = {
            f"p{k}": ["42." if rng.random() < rate else "42" for _ in range(draws)]
            for k in range(prompts)
        }
    return pool


def test_ends_with_full_stop_ignores_surrounding_whitespace() -> None:
    assert ends_with_full_stop("  156181.\n")
    assert not ends_with_full_stop("-1987")


def test_ceilings_match_the_pair_arithmetic() -> None:
    truth = true_partition(REF_ENDPOINTS)
    assert official_score(truth, truth) == pytest.approx(CEILING_EVERYTHING, abs=1e-9)
    all_alone = Partition(truth.family.copy(), np.arange(9, dtype=np.int64))
    assert official_score(truth, all_alone) == 0.0  # no "same" pair: zero denominator
    lumped_model = np.array([0 if e in {"ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08"}
                             else k + 1 for k, e in enumerate(REF_ENDPOINTS)], dtype=np.int64)
    lumped = Partition(truth.family.copy(), lumped_model)
    assert official_score(truth, lumped) == pytest.approx(CEILING_FP8_LUMPED, abs=0.005)


def test_band_rule_isolates_high_rates_only() -> None:
    guess = decision_chain([0.1, 0.9, 0.39, 0.72, 1.0], Rule(0.6))
    assert guess.model.tolist() == [0, 1, 0, 2, 3]
    assert decision_chain([0.3, 0.4], Rule(0.6)).model.tolist() == [0, 0]


def test_bootstrap_reaches_the_fp8_lumped_ceiling_on_separable_data() -> None:
    pool = synthetic_pool()
    cell = bootstrap_scores(pool, REF_ENDPOINTS, 10, Rule(0.6), 200, np.random.default_rng(2))
    assert cell.median == pytest.approx(CEILING_FP8_LUMPED, abs=0.05)
    assert cell.probability_above > 0.9
    assert cell.interval[0] <= cell.median <= cell.interval[1]


def test_bootstrap_is_deterministic_for_a_seed_and_refuses_too_few_prompts() -> None:
    pool = synthetic_pool(prompts=12)
    first = bootstrap_scores(pool, REF_ENDPOINTS, 10, Rule(0.6), 50, np.random.default_rng(3))
    second = bootstrap_scores(pool, REF_ENDPOINTS, 10, Rule(0.6), 50, np.random.default_rng(3))
    np.testing.assert_array_equal(first.scores, second.scores)
    assert len(common_prompts(pool, REF_ENDPOINTS)) == 12
    with pytest.raises(RehearsalError):
        bootstrap_scores(pool, REF_ENDPOINTS, 20, Rule(0.6), 5, np.random.default_rng(4))


def test_snapshot_keeps_the_first_complete_prompts_only() -> None:
    from hackathon.model_equality.research.c_rehearsal import complete_prompts, snapshot_pool

    pool = synthetic_pool(prompts=30)
    del pool["ref-05"]["p3"][1:]  # p3 has one draw only on one endpoint: not complete
    assert "p3" not in complete_prompts(pool, REF_ENDPOINTS, 2)
    snap = snapshot_pool(pool, REF_ENDPOINTS, 25)
    kept = set(snap["ref-01"])
    assert len(kept) == 25 and "p3" not in kept
    assert kept == set(sorted(complete_prompts(pool, REF_ENDPOINTS, 2))[:25])
    assert all(set(prompts) == kept for prompts in snap.values())
    with pytest.raises(RehearsalError):
        snapshot_pool(pool, REF_ENDPOINTS, 30)  # only 29 prompts are complete


def test_write_snapshot_keeps_exactly_the_selected_rows(tmp_path: Path) -> None:
    from hackathon.model_equality.research.c_rehearsal import write_snapshot

    pool = synthetic_pool(prompts=30)
    path = tmp_path / "holdout.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for endpoint, prompts in pool.items():
            for prompt_id, answers in prompts.items():
                for repeat, answer in enumerate(answers):
                    handle.write(json.dumps({"endpoint": endpoint, "id": prompt_id,
                                             "repeat": repeat, "response": answer}) + "\n")
    rows, digest = write_snapshot(path, tmp_path / "snap.jsonl", 25)
    assert rows == 25 * 9 * 2 and len(digest) == 64
    again = write_snapshot(path, tmp_path / "snap2.jsonl", 25)
    assert again == (rows, digest)
    ids = {json.loads(line)["id"] for line in (tmp_path / "snap.jsonl").read_text().splitlines()}
    assert len(ids) == 25


def test_holdout_run_on_a_snapshot(tmp_path: Path) -> None:
    pool = synthetic_pool(prompts=40)
    path = tmp_path / "holdout.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for endpoint, prompts in pool.items():
            for prompt_id, answers in prompts.items():
                for repeat, answer in enumerate(answers):
                    handle.write(json.dumps({"endpoint": endpoint, "id": prompt_id,
                                             "repeat": repeat, "response": answer}) + "\n")
    payload = run_holdout(path, Rule(0.6), sizes=(10, 20), replicates=30, first_prompts=25)
    assert payload["holdout_prompts"] == 25


def test_verdict_follows_the_threshold() -> None:
    assert verdict(0.41, Rule(0.6, threshold_x=0.40))
    assert not verdict(0.39, Rule(0.6, threshold_x=0.40))


def test_holdout_file_runs_with_and_without_the_temperature_ref(tmp_path: Path) -> None:
    pool = synthetic_pool(prompts=25)
    path = tmp_path / "holdout.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for endpoint, prompts in pool.items():
            for prompt_id, answers in prompts.items():
                for repeat, answer in enumerate(answers):
                    handle.write(json.dumps({"endpoint": endpoint, "id": prompt_id,
                                             "repeat": repeat, "response": answer}) + "\n")
    assert len(load_holdout(path)["ref-01"]) == 25
    payload = run_holdout(path, Rule(0.6), sizes=(10, 20), replicates=40)
    labels = {(c["pool"], c["n"]) for c in payload["cells"]}  # type: ignore[attr-defined]
    assert ("without_temperature_ref", 20) in labels and ("all_ref", 10) in labels
    assert payload["holdout_prompts"] == 25
