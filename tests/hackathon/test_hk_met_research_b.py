"""Tests of the team-B research code (zero network): quota arithmetic, AUC, tensors, style tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hackathon.model_equality.prompts.bank import bank_by_id
from hackathon.model_equality.research import adaptive, budget_curve
from hackathon.model_equality.research.analysis_b import agreement_summary, top_prompts
from hackathon.model_equality.research.candidates import (
    CANDIDATE_ID_PREFIX,
    FAMILIES,
    candidate_bank,
    family_of,
)
from hackathon.model_equality.research.discrimination import (
    EndpointTruth,
    agreement_tensor,
    group_answers,
    pair_auc,
    prompt_scores,
    set_auc,
)
from hackathon.model_equality.research.ref_runner import (
    TEAM_B_QUOTA,
    QuotaError,
    allowance,
    build_jobs,
    select_prompts,
)
from hackathon.model_equality.research.style_features import style_tests
from hackathon.model_equality.store import Answer

ENDPOINTS = tuple(f"ref-{i:02d}" for i in range(1, 10))
KINDS = {
    "ref-01": ("S", "plain"), "ref-02": ("T1", "quantized (fp8)"), "ref-03": ("S", "system prompt"),
    "ref-04": ("T3", "LoRA at strength 0.7"), "ref-05": ("S", "persona"),
    "ref-06": ("S", "watermark"), "ref-07": ("T4", "LoRA at strength 1.5"),
    "ref-08": ("S", "sampling at temperature 1.0"), "ref-09": ("T2", "LoRA at strength 0.3"),
}
TRUTH = {e: EndpointTruth(e, m, k) for e, (m, k) in KINDS.items()}


def _answer(endpoint: str, prompt: str, repeat: int, text: str) -> Answer:
    return Answer(endpoint, prompt, repeat, "q", text)


def _synthetic(n_prompts: int = 12) -> list[Answer]:
    """S endpoints answer `a`; T1/T2 answer `a` or `b`; T3/T4 always `z z z` (easy to separate)."""
    rows: list[Answer] = []
    for p in range(n_prompts):
        pid = f"cand-yesno-{p:03d}"
        for endpoint, (model, _) in KINDS.items():
            for repeat in range(2):
                if model == "S":
                    text = "yes"
                elif model in ("T1", "T2"):
                    text = "yes" if (p + repeat) % 2 else "no"
                else:
                    text = "zzz zzz zzz zzz"
                rows.append(_answer(endpoint, pid, repeat, text))
    return rows


def test_candidate_ids_are_unique_and_never_collide_with_the_bank() -> None:
    candidates = candidate_bank()
    ids = [c.id for c in candidates]
    assert len(ids) == len(set(ids))
    assert not set(ids) & set(bank_by_id())
    assert all(i.startswith(CANDIDATE_ID_PREFIX) for i in ids)
    assert {c.category for c in candidates} == set(FAMILIES)
    assert all(1 <= c.max_tokens <= 256 for c in candidates)


def test_family_of_rejects_bank_ids() -> None:
    assert family_of("cand-numeric-003") == "numeric"
    assert family_of("arith-000") is None
    assert family_of("cand-unknown-001") is None


@pytest.mark.parametrize(("stored_b", "expected"), [
    (0, TEAM_B_QUOTA), (30, TEAM_B_QUOTA - 30), (TEAM_B_QUOTA, 0), (TEAM_B_QUOTA + 5, 0),
])
def test_allowance_never_exceeds_the_quota(stored_b: int, expected: int) -> None:
    assert allowance(stored_b) == expected


def test_build_jobs_skips_stored_keys_and_unknown_ids_are_rejected() -> None:
    prompts = select_prompts(None)
    first = build_jobs(prompts[:2], 0, ["ref-01", "ref-02"], set())
    assert len(first) == 4
    again = build_jobs(prompts[:2], 0, ["ref-01", "ref-02"], {first[0].key})
    assert len(again) == 3
    assert [p.id for p in select_prompts(["cand-yesno-001"])] == ["cand-yesno-001"]
    with pytest.raises(QuotaError):
        select_prompts(["cand-nope-000"])


def test_pair_auc_extremes_and_ties() -> None:
    distance = np.array([[0, 0.1, 0.9], [0.1, 0, 0.8], [0.9, 0.8, 0]])
    assert pair_auc(distance, [(0, 2), (1, 2)], [(0, 1)]) == 1.0
    assert pair_auc(distance, [(0, 1)], [(0, 2), (1, 2)]) == 0.0
    flat = np.full((3, 3), 0.5)
    assert pair_auc(flat, [(0, 1)], [(0, 2)]) == 0.5
    nan_only = np.full((3, 3), np.nan)
    assert math.isnan(pair_auc(nan_only, [(0, 1)], [(0, 2)]))


def test_agreement_tensor_and_scores_on_synthetic_models() -> None:
    answers = _synthetic()
    grouped = group_answers(answers, ENDPOINTS)
    prompts = sorted(grouped)
    tensor = agreement_tensor(grouped, prompts, ENDPOINTS, "onehot_key")
    assert tensor.values.shape == (12, 9, 9)
    assert (tensor.draws == 2).all()
    # S vs S always agree; within-endpoint agreement is defined with 2 draws.
    base = [0, 2, 4, 5, 7]
    assert np.allclose(tensor.values[:, base[0], base[1]], 1.0)
    assert set_auc(tensor, range(len(prompts)), TRUTH, "easy") == 1.0
    scores = prompt_scores(tensor, TRUTH)
    assert (scores["easy"] > 0).all()
    assert set(scores) == {"hard", "easy", "excess_hard", "excess_easy", "disguise_leak"}


def test_single_draw_within_endpoint_agreement_is_nan_not_zero() -> None:
    answers = [a for a in _synthetic(4) if a.repeat == 0]
    grouped = group_answers(answers, ENDPOINTS)
    tensor = agreement_tensor(grouped, sorted(grouped), ENDPOINTS, "onehot_key")
    assert np.isnan(tensor.values[:, 0, 0]).all()
    assert np.isfinite(tensor.values[:, 0, 1]).all()


def test_style_tests_flag_the_long_lora_and_not_a_disguise() -> None:
    answers = _synthetic(12)
    results = {r.feature: r for r in style_tests(answers, TRUTH, ENDPOINTS)}
    length = results["log_words"]
    assert length.t["ref-07"] > 5.0
    assert abs(length.t["ref-03"]) < 1e-9
    assert length.prompts == 12


def test_budget_design_is_reproducible_and_rejects_bad_repeats() -> None:
    answers = _synthetic(12)
    prompts = sorted({a.prompt_id for a in answers})
    windows = budget_curve.build_windows(answers, prompts, ENDPOINTS, "onehot_key", 2)
    assert sorted(windows) == [1, 2]
    assert len(windows[1]) == 2
    assert len(windows[2]) == 1
    first = budget_curve.evaluate_design(windows, TRUTH, 6, 2, role="easy", n_draws=20)
    second = budget_curve.evaluate_design(windows, TRUTH, 6, 2, role="easy", n_draws=20)
    assert first == second
    assert first is not None
    assert first.auc_mean == 1.0
    pooled = budget_curve.evaluate_design(windows, TRUTH, 6, 1, role="easy", n_draws=20,
                                          statistic="pool")
    assert pooled is not None
    assert pooled.auc_mean == 1.0
    assert budget_curve.evaluate_design(windows, TRUTH, 50, 2, n_draws=5) is None
    with pytest.raises(ValueError):
        budget_curve.evaluate_design(windows, TRUTH, 4, 3)
    with pytest.raises(ValueError):
        budget_curve.evaluate_design(windows, TRUTH, 4, 1, statistic="nope")


def test_adaptive_simulation_returns_none_on_too_few_prompts() -> None:
    answers = _synthetic(4)
    prompts = sorted({a.prompt_id for a in answers})
    t0 = agreement_tensor(group_answers(answers, ENDPOINTS, repeats=[0]), prompts, ENDPOINTS,
                          "onehot_key")
    t1 = agreement_tensor(group_answers(answers, ENDPOINTS, repeats=[1]), prompts, ENDPOINTS,
                          "onehot_key")
    assert adaptive.simulate(t0, t1, TRUTH, trials=5) is None


def test_agreement_summary_and_top_prompts_on_synthetic_models() -> None:
    answers = _synthetic(8)
    grouped = group_answers(answers, ENDPOINTS)
    prompts = sorted(grouped)
    tensor = agreement_tensor(grouped, prompts, ENDPOINTS, "onehot_key")
    summary = agreement_summary({"onehot_key": tensor}, TRUTH)["onehot_key"]
    assert summary["within_S"] == 1.0
    assert summary["S_S"] == 1.0
    assert summary["S_T3"] == 0.0
    assert 0.0 < summary["S_T1"] < 1.0
    entries = [{
        "prompt": pid, "group": "yesno",
        "kernels": {"onehot_key": {"excess_hard": 0.2, "excess_easy": 1.0, "disguise_leak": 0.0}},
    } for pid in prompts]
    ranked = top_prompts(entries, {"yesno": "onehot_key"}, 3)
    assert len(ranked) == 3
    assert ranked[0]["score"] == pytest.approx(0.6)
